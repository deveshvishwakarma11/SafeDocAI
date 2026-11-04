/**
 * Small-model-friendly answering (TRD §8.3 grounding contract).
 *
 * The LLM is asked for a SHORT PLAIN-TEXT answer from a few retrieved chunks.
 * It is never asked for JSON or evidence: provenance is attached by us from the
 * chunks we retrieved, and the answer is verified against those chunks before
 * the user sees it ("LLM proposes, deterministic checks decide").
 */
import type { RetrievalHit } from './retrieval';
import type { GroundedResult } from './query';

export type AnswerLang = 'en' | 'hinglish';

const MAX_CHUNK_CHARS = 450;

export function detectAnswerLang(query: string): AnswerLang {
  return /[\u0900-\u097F]/.test(query) ||
    /\b(kya|hai|hain|batao|kaun|kitna|kitne|kitni|mera|meri|mere|kiske|naam|kaise|konsa|ye|yeh|ko|se|me|mein|mai|pr|aur|bhai|bata|dikhao|ka|ki|ke|hua|hue|tha|thi|mila|mile|kab|kyun|kyu|wala|wali|liye|uska|iska|apna|natija|jankari)\b/i.test(query)
    ? 'hinglish'
    : 'en';
}

function clip(text: string, max: number): string {
  const t = text.replace(/\s+/g, ' ').trim();
  return t.length > max ? `${t.slice(0, max)}…` : t;
}

export function buildAnswerPrompt(query: string, hits: RetrievalHit[], lang: AnswerLang): string {
  const context = hits
    .map((h, i) => `[${i + 1}] ${h.chunk.file_name || h.doc.file_name} (page ${h.chunk.page_number ?? 1}): ${clip(h.chunk.text, MAX_CHUNK_CHARS)}`)
    .join('\n');
  const language =
    lang === 'hinglish' ? 'Hinglish (Hindi written in English letters)' : 'English';

  return `You are SafeDocAI. Answer the QUESTION using ONLY the CONTEXT below.
Rules: reply in ${language}; at most 2 short sentences; copy names, numbers and IDs exactly as written in the CONTEXT; never use outside knowledge; if the CONTEXT does not contain the answer, reply exactly NOT_FOUND.

CONTEXT:
${context}

QUESTION: ${query}
ANSWER:`;
}

/** Strip separators/case so "06/06/2006" and "06 06 2006" compare equal. */
function squash(s: string): string {
  return (s || '').toLowerCase().replace(/[^\p{L}\p{M}\p{N}]+/gu, '');
}

/**
 * IDs, dates and amounts in the answer: tokens with at least two digits
 * (roll numbers, account numbers, dates, money, years).
 */
export function criticalTokens(answer: string): string[] {
  const out: string[] = [];
  const re = /[\p{L}\p{N}][\p{L}\p{N}\-/.,]*[\p{L}\p{N}]/gu;
  for (const m of answer.matchAll(re)) {
    const tok = m[0];
    if ((tok.match(/\d/g) || []).length < 2) continue;
    const sq = squash(tok);
    if (sq.length >= 3 && !out.includes(sq)) out.push(sq);
  }
  return out;
}

const HEDGES = /\b(typically|generally|usually|commonly|normally|in general)\b/i;

export type GroundingVerdict =
  | { ok: true }
  | { ok: false; reason: 'empty' | 'model_not_found' | 'ungrounded_value' | 'generic_knowledge'; detail?: string };

export function checkGrounding(answer: string, hits: RetrievalHit[]): GroundingVerdict {
  return checkGroundingText(
    answer,
    hits.map(h => `${h.chunk.text} ${h.chunk.file_name || ''}`).join(' ')
  );
}

/** Same check against any context text (used for document summaries too). */
export function checkGroundingText(answer: string, contextRaw: string): GroundingVerdict {
  const text = (answer || '').trim();
  if (!text) return { ok: false, reason: 'empty' };
  if (/^\W*NOT_FOUND\b/i.test(text)) return { ok: false, reason: 'model_not_found' };

  const context = squash(contextRaw);

  for (const tok of criticalTokens(text)) {
    if (!context.includes(tok)) return { ok: false, reason: 'ungrounded_value', detail: tok };
  }

  const hedge = text.match(HEDGES);
  if (hedge && !new RegExp(`\\b${hedge[1]}\\b`, 'i').test(contextRaw)) {
    return { ok: false, reason: 'generic_knowledge', detail: hedge[1] };
  }
  return { ok: true };
}

export function evidenceFromHits(hits: RetrievalHit[]) {
  const top = hits[0]?.score || 1;
  return hits.map(h => ({
    document_id: h.doc.id,
    file_name: h.chunk.file_name || h.doc.file_name,
    snippet: clip(h.chunk.text, 160),
    page_number: h.chunk.page_number,
    confidence: Number(Math.min(0.95, 0.5 + 0.45 * (h.score / top)).toFixed(2))
  }));
}

export function llmAnswerResult(answer: string, hits: RetrievalHit[], lang: AnswerLang): GroundedResult {
  return {
    route: 'semantic_rag',
    detected_language: lang,
    intent: 'llm_grounded_answer',
    answer: answer.trim(),
    evidence: evidenceFromHits(hits),
    is_insufficient_evidence: false
  };
}

/** Verbatim closest passage — used when the model is unavailable or too slow. */
export function passageResult(hits: RetrievalHit[], lang: AnswerLang): GroundedResult {
  const best = hits[0];
  const name = best.chunk.file_name || best.doc.file_name;
  const page = best.chunk.page_number ?? 1;
  const intro =
    lang === 'hinglish'
      ? `Sabse close passage **${name}** (page ${page}) se mila. Ye document ka original text hai, generated answer nahi:`
      : `Closest passage found in **${name}** (page ${page}). This is the document's own text, not a generated answer:`;
  return {
    route: 'semantic_rag',
    detected_language: lang,
    intent: 'retrieved_passage',
    answer: `${intro}\n\n> ${clip(best.chunk.text, 420)}`,
    evidence: evidenceFromHits(hits.slice(0, 1)),
    is_insufficient_evidence: false
  };
}

/** Context for "explain / summarise this document": key fields + opening chunks. */
export function buildSummaryContext(doc: any, chunks: Array<{ text: string; page_number?: number }>): string {
  const fields = (doc.fields || [])
    .slice(0, 8)
    .map((f: any) => `${f.display_label}: ${f.field_value}`)
    .join('; ');
  const body = chunks
    .slice(0, 2)
    .map(c => clip(c.text, MAX_CHUNK_CHARS))
    .join('\n');
  return `${fields ? `Key fields: ${fields}\n` : ''}${body}`.trim();
}

export function buildSummaryPrompt(doc: any, context: string, lang: AnswerLang): string {
  const language =
    lang === 'hinglish' ? 'Hinglish (Hindi written in English letters)' : 'English';
  return `You are SafeDocAI. Summarise the document below in at most 3 short bullet points using ONLY the CONTEXT.
Rules: reply in ${language}; copy names, numbers and IDs exactly as written; never use outside knowledge; if the CONTEXT is unreadable or has no useful content, reply exactly NOT_FOUND.

DOCUMENT: ${doc.file_name} (${doc.category || 'Document'})
CONTEXT:
${context}

QUESTION: Summarise this document.
ANSWER:`;
}

export function summaryResult(doc: any, answer: string, context: string, lang: AnswerLang): GroundedResult {
  return {
    route: 'semantic_rag',
    detected_language: lang,
    intent: 'llm_document_summary',
    answer: answer.trim(),
    evidence: [
      {
        document_id: doc.id,
        file_name: doc.file_name,
        snippet: clip(context, 160),
        confidence: 0.8
      }
    ],
    is_insufficient_evidence: false
  };
}

/** Deterministic summary from verified fields (no model). */
export function fieldsSummaryResult(doc: any, lang: AnswerLang): GroundedResult | null {
  const fields = (doc.fields || []).slice(0, 8);
  if (fields.length === 0) return null;
  const list = fields.map((f: any) => `- **${f.display_label}**: ${f.field_value}`).join('\n');
  return {
    route: 'exact_field',
    detected_language: lang,
    intent: 'fields_summary',
    answer:
      lang === 'hinglish'
        ? `**${doc.file_name}** ke verified details:\n\n${list}`
        : `Verified details from **${doc.file_name}**:\n\n${list}`,
    evidence: fields.map((f: any) => ({
      document_id: doc.id,
      file_name: doc.file_name,
      field_name: f.display_label,
      value: f.field_value,
      snippet: f.evidence_text,
      page_number: f.page_number,
      confidence: f.confidence
    })),
    is_insufficient_evidence: false
  };
}

export function noMatchResult(docCount: number, lang: AnswerLang): GroundedResult {
  return {
    route: 'out_of_scope',
    detected_language: lang,
    intent: 'no_match_in_library',
    answer:
      lang === 'hinglish'
        ? `Aapke ${docCount} stored documents me iske baare me kuch nahi mila. Aisa sawal poochiye jo aapke documents se juda ho.`
        : `I couldn't find anything related to this in your ${docCount} stored document${docCount === 1 ? '' : 's'}. Try a question about the contents of your files.`,
    evidence: [],
    is_insufficient_evidence: true
  };
}

export function notClearResult(fileName: string, lang: AnswerLang): GroundedResult {
  return {
    route: 'semantic_rag',
    detected_language: lang,
    intent: 'not_clear_in_document',
    answer:
      lang === 'hinglish'
        ? `**${fileName}** me is sawal ka saaf jawab verify nahi ho paya, isliye main andaza nahi laga raha.`
        : `I couldn't verify a clear answer to this in **${fileName}**, so I won't guess.`,
    evidence: [],
    is_insufficient_evidence: true
  };
}
