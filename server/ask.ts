/**
 * Question answering ladder (TRD §5.3), kept out of server.ts so it can be
 * tested with a fake LLM.
 *
 *   1. deterministic engines  greeting, inventory, exact fields        instant
 *   2. "weak" engine results  (which document? / not recorded)         re-checked
 *        against stored CONTENT: BM25 picks the document by what it says,
 *        not only by its file name, and asks only when truly ambiguous
 *   3. local LLM on the few retrieved chunks, verified against them    bounded
 *   4. closest verbatim passage / deterministic field summary / honest "no"
 *
 * An LLM answer is shown only if every ID/date/amount in it is found in the
 * retrieved text. Otherwise the user gets verbatim text or an honest limitation.
 */
import {
  answerQueryLocally,
  answerFromRawText,
  looksSuspect,
  resolveQueryDocument,
  whichDocumentAnswer,
  type GroundedResult
} from './query';
import { retrieveChunks, isUsableChunk, type RetrievalHit } from './retrieval';
import { answerStructured } from './structured-answer';
import {
  buildAnswerPrompt,
  buildSummaryContext,
  buildSummaryPrompt,
  checkGrounding,
  checkGroundingText,
  detectAnswerLang,
  fieldsSummaryResult,
  llmAnswerResult,
  noMatchResult,
  notClearResult,
  passageResult,
  summaryResult,
  type AnswerLang
} from './answer';

export interface AskDeps {
  /** Whether LLM synthesis is allowed (OLLAMA_SYNTHESIS != 0). */
  synthesis: boolean;
  ollamaReady: () => Promise<boolean>;
  generate: (prompt: string, opts: { signal?: AbortSignal; numPredict?: number }) => Promise<string>;
  log?: (message: string) => void;
}

/** Engine answers that only mean "I could not tell which file / find it by name". */
const WEAK_INTENTS = new Set(['which_document', 'not_found_in_document']);

/** Structured intents that read plain header fields shared by many documents. */
const HEADER_INTENTS = new Set([
  'structured_enrollment',
  'structured_roll',
  'structured_father',
  'structured_institute',
  'structured_course',
  'structured_gender',
  'structured_name'
]);

/** A second document counts as a real rival when it scores this close to the best. */
const AMBIGUITY_RATIO = 0.75;

const SUMMARY_RE =
  /\b(summari[sz]e|summary|explain|overview|describe|details?|about|contains?|jankari|samjhao|batao)\b|what\s+(?:is|does)\s+(?:this|it|the)\b/i;

export async function askQuestion(
  query: string,
  allDocuments: any[],
  scopeId: number | undefined,
  deps: AskDeps,
  signal?: AbortSignal
): Promise<GroundedResult> {
  const log = deps.log ?? (() => {});

  // Scope is a document id; an unknown/deleted id degrades to the full library.
  const scopedDocs = scopeId ? allDocuments.filter(d => d.id === scopeId) : allDocuments;
  const relevantDocs = scopedDocs.length > 0 ? scopedDocs : allDocuments;

  if (relevantDocs.length === 0) {
    return {
      route: 'out_of_scope',
      answer: 'No documents are currently indexed to answer this question. Please upload a document first.',
      evidence: [],
      is_insufficient_evidence: true
    };
  }

  const lang = detectAnswerLang(query);

  // 0. Structured documents (result sheets, statements): sectioned facts and
  //    subject tables are answered exactly, with every section's value shown.
  for (const doc of relevantDocs) {
    const structured = answerStructured(query, doc, lang);
    if (!structured) continue;
    // Header facts (roll number, name...) can also live in other documents:
    // with several documents let the normal engine handle ambiguity.
    if (relevantDocs.length > 1 && HEADER_INTENTS.has(structured.intent || '')) continue;
    return structured;
  }

  // 1. Deterministic engines first.
  const localAnswer = answerQueryLocally(query, allDocuments, scopeId);

  // 2. Weak engine answers are re-checked against what the documents actually say.
  if (WEAK_INTENTS.has(localAnswer.intent || '')) {
    // "Date of birth" exists in two documents: show each document's value
    // instead of asking, or (worse) guessing which documents are relevant.
    if (localAnswer.intent === 'which_document') {
      const across = answerFieldAcrossDocuments(query, relevantDocs, lang);
      if (across) return across;
    }
    return resolveByContent(query, relevantDocs, localAnswer, lang, deps, signal, log);
  }

  const rawText = relevantDocs[0]?.raw_text || '';
  const rawWorthReading = rawText.length > 300 && !looksSuspect(rawText);
  const isDefinitive =
    !localAnswer.is_insufficient_evidence ||
    localAnswer.intent === 'roll_number_not_found' ||
    (localAnswer.intent === 'no_verified_fields' && !rawWorthReading);
  if (isDefinitive) return localAnswer;

  // Readable docs without fields: instant verbatim excerpt.
  if (rawWorthReading) {
    const excerpt = answerFromRawText(query, relevantDocs);
    if (excerpt) return excerpt;
  }

  return resolveByContent(query, relevantDocs, localAnswer, lang, deps, signal, log);
}

/* ------------------------------------------------------------------ */

async function resolveByContent(
  query: string,
  docs: any[],
  localAnswer: GroundedResult,
  lang: AnswerLang,
  deps: AskDeps,
  signal: AbortSignal | undefined,
  log: (m: string) => void
): Promise<GroundedResult> {
  const hits = retrieveChunks(query, docs, { k: 8 });

  if (hits.length === 0) {
    // Nothing in the stored text mentions the question's terms.
    const summary = await trySummary(query, docs, lang, deps, signal, log);
    if (summary) return summary;
    // Never ask "which document?" for a question no document can answer.
    return localAnswer.intent === 'which_document' ? noMatchResult(docs.length, lang) : localAnswer;
  }

  // Which document do the best chunks come from?
  const bestByDoc = new Map<number, RetrievalHit[]>();
  for (const h of hits) {
    const list = bestByDoc.get(h.doc.id) || [];
    list.push(h);
    bestByDoc.set(h.doc.id, list);
  }
  const ranked = [...bestByDoc.entries()].sort((a, b) => b[1][0].score - a[1][0].score);
  const topScore = ranked[0][1][0].score;
  const rivals = ranked.filter(([, h]) => h[0].score >= topScore * AMBIGUITY_RATIO);

  if (rivals.length > 1) {
    // Genuine ambiguity: ask, but only between the files that really match.
    return whichDocumentAnswer(lang === 'hinglish' ? 'hinglish' : 'en', rivals.map(([, h]) => h[0].doc), query);
  }

  return answerFromHits(query, ranked[0][1].slice(0, 3), lang, deps, signal, log);
}

async function answerFromHits(
  query: string,
  hits: RetrievalHit[],
  lang: AnswerLang,
  deps: AskDeps,
  signal: AbortSignal | undefined,
  log: (m: string) => void
): Promise<GroundedResult> {
  const fileName = hits[0].chunk.file_name || hits[0].doc.file_name;

  if (deps.synthesis && !signal?.aborted && (await deps.ollamaReady())) {
    try {
      const raw = await deps.generate(buildAnswerPrompt(query, hits, lang), { signal });
      const verdict = checkGrounding(raw, hits);
      if (verdict.ok) return llmAnswerResult(raw, hits, lang);
      log(`[ask] LLM answer rejected: ${verdict.reason}${verdict.detail ? ` (${verdict.detail})` : ''}`);
      // The model said the context lacks the answer: say so. It invented
      // something: show the document's own words instead.
      return verdict.reason === 'model_not_found' ? notClearResult(fileName, lang) : passageResult(hits, lang);
    } catch (err: any) {
      if (signal?.aborted) return notClearResult(fileName, lang);
      log(`[ask] LLM unavailable, using closest passage: ${String(err?.message || err).slice(0, 100)}`);
    }
  }
  return passageResult(hits, lang);
}

/** "Explain / summarise / what does this contain" for a document we can identify. */
async function trySummary(
  query: string,
  docs: any[],
  lang: AnswerLang,
  deps: AskDeps,
  signal: AbortSignal | undefined,
  log: (m: string) => void
): Promise<GroundedResult | null> {
  if (!SUMMARY_RE.test(query)) return null;
  const doc = docs.length === 1 ? docs[0] : resolveQueryDocument(query, docs).doc;
  if (!doc) return null;

  const chunks = (doc.chunks || []).filter((c: any) => isUsableChunk(c.text));
  const deterministic = fieldsSummaryResult(doc, lang);

  if (chunks.length > 0 && deps.synthesis && !signal?.aborted && (await deps.ollamaReady())) {
    const context = buildSummaryContext(doc, chunks);
    try {
      const raw = await deps.generate(buildSummaryPrompt(doc, context, lang), { signal, numPredict: 120 });
      const verdict = checkGroundingText(raw, context);
      if (verdict.ok) return summaryResult(doc, raw, context, lang);
      log(`[ask] summary rejected: ${verdict.reason}${verdict.detail ? ` (${verdict.detail})` : ''}`);
    } catch (err: any) {
      if (signal?.aborted) return null;
      log(`[ask] summary LLM unavailable: ${String(err?.message || err).slice(0, 100)}`);
    }
  }

  if (deterministic) return deterministic;
  if (chunks.length > 0) {
    return passageResult([{ doc, chunk: chunks[0], score: 1, matched: [] }], lang);
  }
  return null;
}

/* ------------------------------------------------------------------ */
/* One field asked, several documents have it                          */
/* ------------------------------------------------------------------ */

const FIELD_INTENT = /^(exact_lookup_|roll_number_lookup|name_lookup)/;

/**
 * Ask each document alone. If at least one has a verified field for this
 * question, answer with every document's value (one line if they agree).
 */
export function answerFieldAcrossDocuments(query: string, docs: any[], lang: AnswerLang): GroundedResult | null {
  if (docs.length < 2) return null;
  const per: Array<{ doc: any; ev: any }> = [];
  for (const doc of docs) {
    const r = answerQueryLocally(query, [doc], undefined);
    const ev = r.evidence?.[0];
    if (!r.is_insufficient_evidence && FIELD_INTENT.test(r.intent || '') && ev?.value) per.push({ doc, ev });
  }
  if (per.length === 0) return null;

  const label = per[0].ev.field_name || 'Value';
  const norm = (v: unknown) => String(v).toLowerCase().replace(/[^\p{L}\p{N}]+/gu, '');
  const agree = new Set(per.map(p => norm(p.ev.value))).size === 1;
  const lines = per.map(p => `- **${p.doc.file_name}**: ${p.ev.value}`).join('\n');
  const head = agree
    ? `${label}: **${per[0].ev.value}**`
    : lang === 'hinglish'
      ? `**${label}** alag-alag documents me alag hai:`
      : `**${label}** differs between documents:`;
  const answer = agree && per.length === 1 ? head : `${head}\n${lines}`;

  return {
    route: 'exact_field',
    detected_language: lang,
    intent: 'field_across_documents',
    answer,
    evidence: per.map(p => ({
      document_id: p.doc.id,
      file_name: p.doc.file_name,
      field_name: p.ev.field_name,
      value: p.ev.value,
      snippet: p.ev.snippet,
      page_number: p.ev.page_number ?? 1,
      confidence: p.ev.confidence ?? 0.9
    })),
    is_insufficient_evidence: false
  };
}
