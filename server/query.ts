/**
 * Grounded query answering (server side).
 *  - buildGroundingPrompt: grounding prompt for the local Ollama model (TRD §7.2)
 *  - validateEvidence: LLM output is a candidate, not ground truth (PRD FR-06) —
 *    every evidence item must be verifiable against the stored raw_text.
 *  - answerQueryLocally: deterministic offline fallback grounded in real fields.
 */
import { looksLikePersonName } from './text-utils';
import { cleanExtractedValue } from './extract';

export interface EvidenceItem {
  document_id?: number;
  file_name?: string;
  field_name?: string;
  value?: string;
  snippet?: string;
  page_number?: number;
  confidence?: number;
}

export interface GroundedResult {
  route?: string;
  detected_language?: string;
  intent?: string;
  answer?: string;
  evidence?: EvidenceItem[];
  clarification_options?: Array<{ label: string; value: string }>;
  is_insufficient_evidence?: boolean;
}

export function buildGroundingPrompt(query: string, documents: any[]): string {
  const docsContext = documents.map((doc: any) => {
    const fieldsSummary = (doc.fields || [])
      .slice(0, 25)
      .map(
        (f: any) =>
          `  - ${f.display_label} (${f.field_name}): "${f.field_value}" [Evidence: "${f.evidence_text}"]`
      )
      .join('\n');

    // Bounded excerpt: a 500 KB doc must NOT become a 100k-token prompt —
    // huge prompts made Ollama prefill for minutes and peg the CPU.
    const raw = doc.raw_text || '';
    const excerpt =
      raw.length > 1200
        ? `${raw.slice(0, 1200)}\n[... excerpt truncated; ${raw.length} chars total ...]`
        : raw;

    return `DOCUMENT: ${doc.file_name} (ID: ${doc.id}, Category: ${doc.category})
EXTRACTED METADATA FIELDS:
${fieldsSummary || '  (None)'}
DOCUMENT TEXT EXCERPT (truncated for brevity):
${excerpt}
---`;
  }).join('\n\n');

  return `You are SafeDocAI, a privacy-first personal document assistant.
User Query: "${query}"

DOCUMENTS STORED ON DEVICE:
${docsContext}

STRICT GROUNDING RULES:
1. If the user is greeting you (e.g. "hi", "hii", "hello", "hey", "namaste", "kaise ho"), respond warmly in their language, introduce yourself as SafeDocAI personal document assistant, and invite them to ask about their indexed documents! Set route to "greeting" and is_insufficient_evidence to false.
2. If the user asks in Hindi or Hinglish (e.g. "kya hai ye 10th", "mera roll number kya hai", "marks kitne hai", "kiske naam pe hai", "details batao"):
   - Respond in friendly conversational Hindi / Hinglish.
   - If they ask about an uploaded document:
     Explain what the document contains in detail based ONLY on the stored text and fields: candidate name, roll number, school/board name, subjects, marks, year of passing, etc.
   - Never fabricate data that does not exist in the documents.
3. If the user asks in English, respond in clear English.
4. If the question asks for an exact field (e.g. roll number, date of birth, balance, due date, amount, IFSC, marks):
   - State the exact value boldly first.
   - Cite which document and quote evidence.
5. If the requested information is genuinely NOT in the documents:
   - Honestly state that it could not be found in the stored documents. Never fabricate facts.
6. RESPONSE SIZE LIMITS (the local model is small and runs on CPU):
   - answer field: maximum 30 words, no markdown headings.
   - Include AT MOST ONE evidence entry with a verbatim snippet of at most 120 characters.
   - Output single-line MINIFIED JSON only — no code fences, no extra keys.

Return ONLY a valid JSON object matching this schema:
{
  "route": "exact_field" | "semantic_rag" | "clarification" | "greeting" | "out_of_scope",
  "detected_language": "en" | "hi" | "hinglish",
  "intent": "short intent description",
  "answer": "Concise, grounded answer markdown",
  "is_insufficient_evidence": boolean,
  "evidence": [
    {
      "document_id": number,
      "file_name": "filename.pdf",
      "field_name": "Roll Number",
      "value": "2023CS001",
      "snippet": "verbatim evidence quote",
      "page_number": 1,
      "confidence": 0.99
    }
  ],
  "clarification_options": [
    { "label": "option text", "value": "full query prompt" }
  ]
}`;
}

/** Loose JSON extraction: handles ```json fences and stray prose around the object. */
export function parseJsonLoose(text: string): any {
  if (!text) return {};
  const stripped = text.replace(/```json/g, '').replace(/```/g, '').trim();
  try {
    return JSON.parse(stripped);
  } catch {
    const start = stripped.indexOf('{');
    const end = stripped.lastIndexOf('}');
    if (start >= 0 && end > start) {
      try {
        return JSON.parse(stripped.slice(start, end + 1));
      } catch {
        return {};
      }
    }
    return {};
  }
}

/**
 * Evidence validation: every claimed snippet/value must exist verbatim
 * (case-insensitive) in the cited document's stored text or extracted fields.
 * Invalid items are dropped; if everything was invalid the answer is flagged.
 */
/**
 * OCR quality heuristic: real prose carries stopwords; mangled OCR often does not.
 * Used only to add an honest caveat — never to fabricate or hide content.
 */
const STOPWORDS = new Set([
  'the', 'and', 'of', 'to', 'in', 'is', 'are', 'for', 'on', 'at', 'by', 'or',
  'name', 'number', 'date', 'total', 'amount', 'date of', 'from', 'with', 'this',
  'that', 'shall', 'must', 'been', 'was', 'were', 'be', 'any', 'not', 'per',
  'document', 'copy', 'issued', 'government', 'india', 'mobile', 'email',
  'grade', 'percent', 'student', 'invoice', 'payment', 'receipt', 'bank',
  'balance', 'signature', 'verified', 'report', 'statement', 'school', 'board'
]);

export function looksSuspect(text: string): boolean {
  const sample = (text || '').slice(0, 3000).toLowerCase();
  const words = sample.split(/[^a-z0-9]+/).filter(w => w.length > 1);
  if (words.length < 8) return true;
  const stop = words.filter(w => STOPWORDS.has(w)).length;
  return stop / words.length < 0.1;
}

/**
 * Instant verbatim-excerpt answer for readable documents that carry no
 * standard extracted fields (contracts, letters, scanned prose).
 *
 * WHY NOT THE LLM: on CPU-only laptops (measured 2.4 tok/s on an A8-7410) a
 * schema-valid grounded answer needs 30–60s of ALL-core generation — exactly
 * the laptop stutter this feature exists to remove. Instead we quote the
 * best-matching passage VERBATIM from stored raw_text: instant, offline,
 * and impossible to fabricate because the snippet is copied, not generated.
 *
 * Returns null when nothing honestly quotable exists (garbled/short text,
 * no query-term overlap, or no content terms at all).
 */
const QUERY_STOPWORDS = new Set([
  // English
  'the', 'and', 'for', 'are', 'was', 'were', 'with', 'that', 'this', 'these', 'those',
  'what', 'which', 'whose', 'where', 'when', 'why', 'how', 'who', 'does', 'did',
  'have', 'has', 'had', 'you', 'your', 'mine', 'from', 'about', 'into', 'please',
  'tell', 'show', 'give', 'explain', 'describe', 'summarize', 'summary', 'mention',
  'document', 'documents', 'file', 'files', 'pdf', 'detail', 'details',
  'information', 'content', 'contents', 'says', 'saying', 'there', 'here',
  // Hinglish / Hindi transliterated
  'kya', 'kiye', 'hai', 'hain', 'ho', 'hoon', 'mera', 'meri', 'mere', 'tum', 'aap',
  'ye', 'yeh', 'woh', 'iska', 'iski', 'iske', 'kiska', 'kiske', 'kiski', 'kya',
  'batao', 'bata', 'dikhao', 'batao', 'kaise', 'konsa', 'konsi', 'kitna', 'kitni',
  'kitne', 'aur', 'lekin', 'se', 'ko', 'me', 'mai', 'mein', 'ka', 'ke', 'ki', 'par',
  'pr', 'bhi', 'to', 'pta', 'do', 'karo', 'kar'
]);

/** Content terms only, length ≥ 3 (short Hinglish particles like "ka" are stopped). */
function queryContentTerms(query: string): string[] {
  const raw = query.toLowerCase().split(/[^a-z0-9]+/).filter(Boolean);
  return [...new Set(raw.filter(w => w.length >= 3 && !QUERY_STOPWORDS.has(w)))];
}

/** Questions whose answer must contain digits (numbers, ids, dates, amounts). */
const NUMBERISH_QUERY =
  /\b(?:number|no|nos|id|code|roll|pnr|amount|total|balance|date|dob|year|session|mobile|phone|contact|account|ifsc|receipt|invoice|marks|percentage|cgpa|pin|validity|due)\b/i;

/** Light stem key: "termination" & "terminate" → "termi" (5-char shared root). */
function stemKey(word: string): string {
  return word.length >= 6 ? word.slice(0, 5) : word;
}

/** Split raw text into quotable windows (~2 sentences, hard-capped for OCR blobs). */
function splitWindows(text: string): string[] {
  const rough = text
    .split(/(?<=[.!?।])\s+|\n+/)
    .map(s => s.trim())
    .filter(Boolean);
  const windows: string[] = [];
  let cur = '';
  for (const s of rough) {
    if (cur && cur.length + s.length + 1 > 260) {
      windows.push(cur);
      cur = '';
    }
    cur = cur ? `${cur} ${s}` : s;
    while (cur.length > 260) {
      windows.push(cur.slice(0, 260));
      cur = cur.slice(260);
    }
  }
  if (cur.trim()) windows.push(cur.trim());
  return windows.filter(w => w.length >= 15);
}

/* ------------------------------------------------------------------ */
/* Query normalization, document scope and line-level grounding        */
/* ------------------------------------------------------------------ */

/** Common user typos (and OCR-style misspellings) → the field vocabulary. */
const QUERY_TYPO_FIXES: Array<[RegExp, string]> = [
  [/\bcollage\b/g, 'college'],
  [/\bcollge\b/g, 'college'],
  [/\badress\b/g, 'address'],
  [/\baddrss\b/g, 'address'],
  [/\bmoblie\b/g, 'mobile'],
  [/\bmoble\b/g, 'mobile'],
  [/\bgrup\b/g, 'group'],
  [/\benrolement\b/g, 'enrolment'],
  [/\brollno\b/g, 'roll no'],
  [/\bschooling\b/g, 'school']
];

/** Words that identify a field group the user explicitly excluded. */
const FIELD_EXCLUSION_WORDS: Array<[RegExp, string[]]> = [
  [/\bcandidate\b/, ['candidate_name']],
  [/\bstudent\s*name\b/, ['candidate_name']],
  [/\bfather\b/, ['father_name']],
  [/\bmother\b/, ['mother_name']],
  [/\broll\b/, ['roll_number']],
  [/\bblood\b/, ['blood_group']],
  [/\bdob\b|date\s*of\s*birth/, ['date_of_birth']],
  [/\baddress\b/, ['residential_address']],
  [/\bcollege\b|\bschool\b/, ['school_name']]
];

export function normalizeQueryText(query: string): string {
  let q = ` ${query.toLowerCase().trim()} `;
  for (const [re, replacement] of QUERY_TYPO_FIXES) q = q.replace(re, replacement);
  return q.replace(/\s+/g, ' ').trim();
}

/**
 * "college name not candidate" means: answer the college field, never the
 * candidate-name field. Every phrase after "not" marks a group to skip.
 */
export function excludedFieldGroups(query: string): Set<string> {
  const q = normalizeQueryText(query);
  const out = new Set<string>();
  for (const part of q.split(/\bnot\b/).slice(1)) {
    const head = part.trim().split(/[,.;?]/)[0].slice(0, 30);
    for (const [re, fields] of FIELD_EXCLUSION_WORDS) {
      if (re.test(head)) fields.forEach(f => out.add(f));
    }
  }
  return out;
}

/** Query wording → document category, used to pick the right file. */
const DOC_TYPE_KEYWORDS: Array<[RegExp, string]> = [
  [/\bid\s*card\b|\bidentity\b|\baadhaar\b|\bpan\s*card\b|\bstudent\s*id\b|\bid\b/, 'Identity Document'],
  [/\bmarksheet\b|\bmarks\b|\bdegree\b|\bresult\b|\btranscript\b/, 'Marksheet / Degree'],
  [/\bbill\b|\belectricity\b|\bwater\b|\bgas\b/, 'Utility Bill'],
  [/\bpassbook\b|\bbank\b|\bstatement\b|\baccount\b/, 'Bank Statement / Passbook'],
  [/\bpolicy\b|\binsurance\b|\bpremium\b/, 'Insurance Policy'],
  [/\breceipt\b|\binvoice\b|\bpayment\b/, 'Receipt / Invoice'],
  [/\bticket\b|\btrain\b|\birctc\b|\bpnr\b|\bboarding\b/, 'Travel Ticket'],
  [/\bapplication\b|\bgate\b/, 'Application Form']
];

/**
 * Pick the document a content question is about. With one document this is
 * unambiguous; with several it matches the file name and the category the
 * question names, and reports ambiguity instead of silently using documents[0]
 * (that silently answered ID-card questions from a train ticket).
 */
export function resolveQueryDocument(
  query: string,
  documents: any[]
): { doc: any | null; ambiguous: boolean } {
  if (documents.length === 0) return { doc: null, ambiguous: false };
  if (documents.length === 1) return { doc: documents[0], ambiguous: false };

  const q = normalizeQueryText(query);
  let best: any = null;
  let bestScore = 0;

  for (const doc of documents) {
    let score = 0;
    const base = String(doc.file_name || '').toLowerCase().replace(/\.[^.]+$/, '');
    for (const part of base.split(/[^a-z0-9]+/)) {
      if (part.length >= 3 && q.includes(part)) score += 3;
    }
    for (const [re, category] of DOC_TYPE_KEYWORDS) {
      if (re.test(q) && doc.category === category) score += 4;
    }
    if (score > bestScore) {
      bestScore = score;
      best = doc;
    }
  }

  if (bestScore === 0) return { doc: null, ambiguous: true };
  return { doc: best, ambiguous: false };
}

/** Best matching transcribed line for a query (verbatim, no generation). */
export function bestMatchingLine(
  query: string,
  text: string
): { line: string; score: number } | null {
  const terms = queryContentTerms(query).map(stemKey);
  if (terms.length === 0) return null;  // A labelled-but-valueless line ("PNR Train No./Name Class") is not an answer
  // to a numbers question; require a digit in the line for those.
  const wantsValue = NUMBERISH_QUERY.test(query);

  let best: { line: string; score: number } | null = null;
  for (const rawLine of String(text || '').split(/\r?\n/)) {
    const line = rawLine.replace(/\s+/g, ' ').trim();
    if (line.length < 6 || !/[a-zA-Z]{2}/.test(line)) continue;
    if (wantsValue && !/[0-9]/.test(line)) continue;
    const keys = new Set(
      line.toLowerCase().split(/[^a-z0-9]+/).filter(Boolean).map(stemKey)
    );
    const score = terms.filter(t => keys.has(t)).length;

    if (score > 0 && (!best || score > best.score)) best = { line, score };
  }
  return best;
}

/** Best matching stored vector chunk for a query. */
function bestMatchingChunk(
  query: string,
  doc: any
): { text: string; score: number } | null {
  const terms = queryContentTerms(query).map(stemKey);
  if (terms.length === 0) return null;
  let best: { text: string; score: number } | null = null;
  for (const chunk of Array.isArray(doc.chunks) ? doc.chunks : []) {
    const keys = new Set(
      String(chunk.text || '').toLowerCase().split(/[^a-z0-9]+/).filter(Boolean).map(stemKey)
    );
    const score = terms.filter(t => keys.has(t)).length;
    if (score > 0 && (!best || score > best.score)) best = { text: chunk.text, score };
  }
  return best;
}

/** Quote the best matching passage from a document, with an honest caveat. */
function excerptAnswer(
  lang: 'en' | 'hinglish',
  doc: any,
  passage: string,
  intent: string
): GroundedResult {
  const excerpt = passage.length > 420 ? `${passage.slice(0, 420)}…` : passage;
  return {
    route: 'semantic_rag',
    detected_language: lang,
    intent,
    answer:
      lang === 'en'
        ? `From **\`${doc.file_name}\`**:\n\n> ${excerpt}\n\n*Quoted verbatim from this document's stored text.*`
        : `**\`${doc.file_name}\`** me yeh mila:\n\n> ${excerpt}\n\n*Yeh passage isi document ke stored text se jaisa hai waisa quote kiya gaya hai.*`,
    is_insufficient_evidence: false,
    evidence: [
      {
        document_id: doc.id,
        file_name: doc.file_name,
        snippet: excerpt.slice(0, 300),
        page_number: 1,
        confidence: 0.85
      }
    ]
  };
}

/** Ask which document — used whenever several files could answer.
 *  `documents[0]` guessing is what answered ID-card questions from a ticket. */
export function whichDocumentAnswer(
  lang: 'en' | 'hinglish',
  documents: any[],
  query: string
): GroundedResult {
  const listing = documents
    .map((d: any) => `- **${d.file_name}** (${d.category})`)
    .join('\n');
  return {
    route: 'clarification',
    detected_language: lang,
    intent: 'which_document',
    answer:
      lang === 'en'
        ? `You have ${documents.length} documents — which one should I answer from?\n\n${listing}`
        : `Aapke paas ${documents.length} documents hain — kis document se jawab chahiye?\n\n${listing}`,
    is_insufficient_evidence: false,
    evidence: [],
    clarification_options: documents.map((d: any) => ({
      label: d.file_name,
      value: `${query} (in ${d.file_name})`,
      document_id: d.id
    }))
  };
}

/** Field nouns that appear in almost every question: not distinctive on their own. */
const GENERIC_FIELD_TERMS = new Set([
  'name', 'naam', 'number', 'numbe', 'numb', 'num', 'no', 'id', 'code', 'card', 'value', 'val'
]);

/**
 * Honest not-found: say what is missing, then show what this document DOES
 * contain (fields + closest transcribed line). Never a dead end, never a guess.
 */
function notFoundAnswer(
  lang: 'en' | 'hinglish',
  doc: any,
  query: string,
  label?: string,
  /** When set, the "closest line" must mention this subject too. */
  subject?: string
): GroundedResult {
  const fields = (doc.fields || []).map((f: any) => f.display_label);
  const hit = bestMatchingLine(query, doc.raw_text || '');
  // When the question names a concrete thing ("college name") and only generic
  // words ("name") matched, the quoted line answers a different question —
  // "college name" used to quote the student's Name line. Quote only when a
  // distinctive term actually appears in the line.
  const distinctiveTerms = queryContentTerms(query)
    .filter(t => !GENERIC_FIELD_TERMS.has(t) && !GENERIC_FIELD_TERMS.has(stemKey(t)))
    .map(stemKey);
  const hitKeys = hit
    ? new Set(
        hit.line.toLowerCase().split(/[^a-z0-9]+/).filter(Boolean).map(stemKey)
      )
    : null;
  const hitIsRelevant =
    !hit ||
    distinctiveTerms.length === 0 ||
    distinctiveTerms.some(t => hitKeys?.has(t));
  const closest =
    hit &&
    hitIsRelevant &&
    (!subject || hit.line.toLowerCase().includes(subject)) ? hit : null;
  const hasFields = fields.length > 0;

  const answerParts: string[] = [];
  answerParts.push(
    lang === 'en'
      ? `${label ? `**${label}** is` : 'That information is'} not recorded in **\`${doc.file_name}\`**.`
      : `${label ? `**${label}**` : 'Yeh jankari'} **\`${doc.file_name}\`** me nahi hai.`
  );
  if (hasFields) {
    answerParts.push(
      lang === 'en'
        ? `This document does contain: ${fields.join(', ')}.`
        : `Is document me yeh fields hain: ${fields.join(', ')}.`
    );
  }
  if (closest) {
    answerParts.push(
      lang === 'en'
        ? `Closest transcribed line:\n\n> ${closest.line}`
        : `Sabse kareeb ki transcribed line:\n\n> ${closest.line}`
    );
  }
  if (looksSuspect(doc.raw_text || '')) {
    answerParts.push(
      lang === 'en'
        ? '_The scan quality looks low — try **Re-run OCR** in the document viewer for better text._'
        : '_Scan quality kam lag rahi hai — document viewer me **Re-run OCR** try karein._'
    );
  }

  return {
    route: 'exact_field',
    detected_language: lang,
    intent: 'not_found_in_document',
    answer: answerParts.join('\n\n'),
    is_insufficient_evidence: true,
    evidence: closest
      ? [
          {
            document_id: doc.id,
            file_name: doc.file_name,
            snippet: closest.line.slice(0, 250),
            page_number: 1,
            confidence: 0.4
          }
        ]
      : []
  };
}

export function answerFromRawText(query: string, documents: any[]): GroundedResult | null {
  const terms = queryContentTerms(query);
  if (terms.length === 0) return null;
  const termKeys = terms.map(stemKey);

  const isHindiOrHinglish =
    /[ऀ-ॿ]/.test(query) ||
    /\b(kya|hai|batao|kaun|kitna|kitne|mera|meri|kiske|naam|kaise|konsa|ye|yeh)\b/i.test(query);
  const lang: 'en' | 'hi' | 'hinglish' = isHindiOrHinglish ? 'hinglish' : 'en';

  let best: { doc: any; window: string; score: number } | null = null;

  for (const doc of documents) {
    const raw = (doc.raw_text || '').trim();
    if (raw.length < 300 || looksSuspect(raw)) continue; // garble is never quoted

    for (const w of splitWindows(raw)) {
      const winKeys = new Set(
        w.toLowerCase().split(/[^a-z0-9]+/).filter(Boolean).map(stemKey)
      );
      const matched = termKeys.filter(t => winKeys.has(t)).length;
      if (matched === 0) continue;
      if (!best || matched > best.score) best = { doc, window: w, score: matched };
    }
  }

  if (!best) return null;

  const excerpt = best.window.length > 400 ? `${best.window.slice(0, 400)}…` : best.window;
  const fileName = best.doc.file_name;

  return {
    route: 'semantic_rag',
    detected_language: lang,
    intent: 'raw_excerpt_answer',
    answer: lang === 'en'
      ? `Closest matching passage in \`${fileName}\`:\n\n> ${excerpt}\n\n*Quoted verbatim from the stored document text — no standard fields were extracted for this file, so the source is shown directly.*`
      : `\`${fileName}\` me is sawaal se judi sabse relevant line:\n\n> ${excerpt}\n\n*Yeh text seedhe document se jaisa hai waisa hi quote kiya gaya hai — is file se koi standard field extract nahi hua, isliye source sidha dikhaya gaya hai.*`,
    is_insufficient_evidence: false,
    evidence: [
      {
        document_id: best.doc.id,
        file_name: fileName,
        snippet: excerpt.substring(0, 300),
        page_number: 1,
        confidence: Math.min(0.9, 0.5 + best.score * 0.1)
      }
    ],
    clarification_options: []
  };
}

/** Instant inventory answer — "what documents do you have" never needs the LLM. */
function inventoryAnswer(query: string, documents: any[], lang: 'en' | 'hi' | 'hinglish'): GroundedResult | null {
  const q = query.toLowerCase();
  // Tight on purpose: only genuine "inventory" phrasings. Bare "what ... document"
  // must NOT hijack content questions like "explain what this document contains".
  const asksForDocs =
    /\b(you have|do you have|have any|mere paas|list your|what documents|which documents|what files|which files|how many documents|how many files)\b/i.test(q) ||
    /\b(documents|files|docs)\b[^?]{0,25}\b(do you have|hai|hain)\b/i.test(q);

  if (!asksForDocs) return null;

  const lines = documents.map(
    (d: any) =>
      `• **${d.file_name}** (${d.category}) — ${Array.isArray(d.fields) ? d.fields.length : 0} verified fields`
  );
  const hi = lang !== 'en';
  return {
    route: 'document_retrieval',
    detected_language: lang,
    intent: 'document_inventory',
    answer: hi
      ? `Aapke paas abhi **${documents.length} documents** indexed hain:\n\n${lines.join('\n')}\n\nKisi bhi document ke baare me sawaal pooch sakte hain.`
      : `You currently have **${documents.length} document${documents.length === 1 ? '' : 's'}** indexed:\n\n${lines.join('\n')}\n\nAsk me anything about any of them.`,
    is_insufficient_evidence: false,
    evidence: documents.slice(0, 3).map((d: any) => ({
      document_id: d.id,
      file_name: d.file_name,
      snippet: (d.raw_text || '').substring(0, 150),
      page_number: 1,
      confidence: 1.0
    })),
    clarification_options: []
  };
}

export function validateEvidence(result: GroundedResult, documents: any[]): GroundedResult {
  if (!Array.isArray(result.evidence) || result.evidence.length === 0) {
    return { ...result, evidence: result.evidence || [] };
  }

  const docById = new Map<number, any>();
  for (const d of documents) docById.set(d.id, d);

  const kept: EvidenceItem[] = [];
  for (const ev of result.evidence) {
    const doc = ev.document_id !== undefined ? docById.get(ev.document_id) : undefined;
    if (!doc) continue;

    const haystack = `${doc.raw_text || ''}`.toLowerCase();
    const fieldHaystack = (doc.fields || [])
      .map((f: any) => `${f.field_value} ${f.evidence_text}`)
      .join(' ')
      .toLowerCase();

    const checks: string[] = [];
    if (ev.snippet) checks.push(ev.snippet.toLowerCase());
    if (ev.value) checks.push(ev.value.toLowerCase());

    if (checks.length === 0) continue;
    const verifiable = checks.every(
      c => haystack.includes(c) || fieldHaystack.includes(c)
    );
    if (verifiable) kept.push(ev);
  }

  const allDropped = result.evidence.length > 0 && kept.length === 0;
  return {
    ...result,
    evidence: kept,
    is_insufficient_evidence: allDropped ? true : result.is_insufficient_evidence
  };
}

/** One exact-field answer: value first, then document + verbatim evidence. */
function fieldAnswer(lang: 'en' | 'hinglish', doc: any, field: any): GroundedResult {
  const label = field.display_label;
  const value = field.field_value;
  return {
    route: 'exact_field',
    detected_language: lang,
    intent: `exact_lookup_${field.field_name}`,
    answer:
      lang === 'en'
        ? `According to your **\`${doc.file_name}\`**, the **${label}** is:\n\n### **${value}**\n\nVerified from Page ${field.page_number || 1} · Evidence: "${field.evidence_text}"`
        : `Aapke **\`${doc.file_name}\`** ke anusar, **${label}** hai:\n\n### **${value}**\n\nPage ${field.page_number || 1} · Evidence: "${field.evidence_text}"`,
    is_insufficient_evidence: false,
    evidence: [
      {
        document_id: doc.id,
        file_name: doc.file_name,
        field_name: label,
        value,
        snippet: field.evidence_text,
        page_number: field.page_number || 1,
        confidence: field.confidence
      }
    ]
  };
}

/**
 * Fallback for documents whose stored fields predate the current extractor:
 * quote the real line from raw_text instead of dumping unrelated fields.
 */
function rawLineAnswer(
  lang: 'en' | 'hinglish',
  doc: any,
  label: string,
  value: string,
  evidence: string
): GroundedResult {
  return {
    route: 'exact_field',
    detected_language: lang,
    intent: `raw_line_lookup_${label.toLowerCase().replace(/[^a-z]+/g, '_')}`,
    answer:
      lang === 'en'
        ? `Your **\`${doc.file_name}\`** states **${label}: ${value}**.\n\n> ${evidence}\n\n*Quoted verbatim from the stored transcription — this file was scanned before the current field extractor, so the line is shown directly.*`
        : `Aapke **\`${doc.file_name}\`** me **${label}: ${value}** likha hai.\n\n> ${evidence}\n\n*Yeh line stored transcription se jaisi hai waisi hi quote ki gayi hai.*`,
    is_insufficient_evidence: false,
    evidence: [
      {
        document_id: doc.id,
        file_name: doc.file_name,
        field_name: label,
        value,
        snippet: evidence,
        page_number: 1,
        confidence: 0.9
      }
    ]
  };
}

/**
 * Plain-language field intents, answered directly from extracted fields.
 * Without this, every unlisted question (blood group, DOB, address…) fell
 * through to the "here are all the details" dump — four different questions,
 * one identical answer.
 */
type FieldIntent = {
  field: string;
  label: string;
  test: RegExp;
  raw?: (doc: any) => { value: string; evidence: string } | null;
};

/** A field asked for in plain words, with no bespoke transcription fallback. */
function plainIntent(field: string, label: string, test: RegExp): FieldIntent {
  return { field, label, test };
}

const DATE_PATTERN = /\d{1,2}[\/\-.]\d{1,2}[\/\-.]\d{2,4}/;

/**
 * Words that turn "... name" into a labelled-field question rather than a
 * question about the document holder.
 */
const NAME_QUALIFIER =
  /\b(?:college|school|institution|institute|university|principal|father|mother|guardian|parent|bank|branch|company|merchant|seller|vendor|holder|board|party|hotel|shop|store|hospital|doctor|teacher|manager|owner)\b/i;

const FIELD_INTENTS: FieldIntent[] = [
  {
    field: 'blood_group',
    label: 'Blood Group',
    test: /blood\s*(?:group|type)|rakt\s*(?:group|samuh)/i,
    raw: d => {
      const m = (d.raw_text || '').match(/blood\s*(?:group|type)\s*[:.=]?\s*((?:AB|A|B|O)\s*[+\-])/i);
      return m ? { value: m[1].replace(/\s+/g, ''), evidence: m[0].trim() } : null;
    }
  },
  {
    field: 'date_of_birth',
    label: 'Date of Birth',
    test: /date\s*of\s*birth|\bdob\b|birth\s*date|janam\s*(?:tithi|din)/i,
    raw: d => {
      const line: string | undefined = String(d.raw_text || '')
        .split(/\r?\n/)
        .find(
          l =>
            /(?:^|[^a-z])(?:dob|d\.o\.b|date\s*of\s*birth|008|oo8)\b/i.test(l) &&
            DATE_PATTERN.test(l)
        );
      const value = line?.match(DATE_PATTERN)?.[0];
      return line && value ? { value, evidence: line.trim() } : null;
    }
  },
  {
    field: 'contact_number',
    label: 'Contact Number',
    test: /(?:contact|mobile|phone)\s*(?:no\.?|number)?|मोबाइल/i,
    raw: d => {
      const m = (d.raw_text || '').match(/(?:contact|mobile|phone)\s*(?:no\.?|number)?\s*[:.=]?\s*(?:\+?91[\s-]?)?([6-9]\d{9})\b/i);
      return m ? { value: m[1], evidence: m[0].trim() } : null;
    }
  },
  {
    field: 'course_branch',
    label: 'Course / Branch',
    test: /\bcourse\b|\bbranch\b|program(?:me)?/i,
    raw: d => {
      const m = (d.raw_text || '').match(/course\s*(?:\/\s*branch)?\s*[:.=]?\s*([A-Za-z][A-Za-z.\s]{1,24}(?:[\/|&]\s*[A-Za-z][A-Za-z.\s]{1,16})?)/i);
      if (!m) return null;
      const value = m[1].replace(/\s+/g, ' ').replace(/\s+(?:is|si|ad|he|re)$/i, '').trim();
      return value.length >= 2 ? { value, evidence: m[0].trim() } : null;
    }
  },
  {
    field: 'student_id',
    label: 'Student ID',
    // "enrolment id" is its OWN field — claiming it here answered "Enrollment ID"
    // questions with a missing-Student-ID message.
    test: /student\s*id|student\s*code/i,
    raw: d => {
      const m = (d.raw_text || '').match(/student\s*id\s*[:.=]?\s*([A-Z0-9]+(?:[\/|-][A-Z0-9]+){1,6})/i);
      return m ? { value: m[1], evidence: m[0].trim() } : null;
    }
  },
  {
    field: 'school_name',
    label: 'School / Institution',
    test: /\bcollege\b|\bschool\b|\binstitution\b|\buniversity\b|\binstitute\b/i,
    raw: d => {
      const line: string | undefined = String(d.raw_text || '')
        .split(/\r?\n/)
        .map(l => l.trim())
        .find(
          l =>
            l.length >= 6 &&
            l.length <= 95 &&
            /\b(college|university|institute|school|academy)\b/i.test(l) &&
            !/\b(address|signature|student|course|roll)\b/i.test(l)
        );
      return line ? { value: line, evidence: line } : null;
    }
  },
  {
    field: 'residential_address',
    label: 'Residential Address',
    test: /\baddress\b|\bpata\b/i,
    raw: d => {
      const m = (d.raw_text || '').match(/(?:home\s*address|address|पता)\s*[:.=]?\s*([^\r\n]{5,120})/i);
      return m ? { value: m[1].trim(), evidence: m[0].trim() } : null;
    }
  },
  {
    field: 'session',
    label: 'Session / Validity',
    // Bare "valid" ("is this a valid id card") is not a session question —
    // only validity/expiry phrasing is.
    test: /\bsession\b|\bvalidity\b|\bvalid\s*(?:till|upto|up\s*to|thru|through)\b|kab\s*tak\s*valid/i,
    raw: d => {
      const m = (d.raw_text || '').match(/session\s*[:.=]?\s*(\d{4}\s*[-–\/]\s*\d{4})/i);
      return m ? { value: m[1].replace(/\s+/g, ''), evidence: m[0].trim() } : null;
    }
  },

  // Every other extracted field gets a plain-language question too, so "kuch
  // bhi poocho" does not fall off the end of the lookups. These carry no `raw`
  // regex on purpose: when the field is missing the ladder below quotes the
  // real transcribed line before saying anything is absent.
  plainIntent('enrollment_id', 'Enrollment ID', /enrol{1,2}ment\s*(?:id|no\.?|number)|enrol{1,2}ment\s*code/i),
  plainIntent('father_name', "Father's Name", /father(?:'s)?\s*(?:name|naam)|पिता|pita(?:ji)?\s*ka\s*naam/i),
  plainIntent('mother_name', "Mother's Name", /mother(?:'s)?\s*(?:name|naam)|माता|mata(?:ji)?\s*ka\s*naam/i),
  plainIntent('board_name', 'Board / University', /\bboard\b|\buniversity\b|\bcbse\b|\bicse\b|\bbseb\b/i),
  plainIntent('year_of_passing', 'Passing Year', /passing\s*year|year\s*of\s*passing|\bpass(?:ed)?\b|kis\s*saal/i),
  plainIntent('result_status', 'Result / Division', /\bresult\b|\bdivision\b|pass\s*(?:hua|hui|hai)|fail\s*(?:hua|hui|hai)/i),
  plainIntent('total_marks', 'Total Marks', /total\s*marks|marks\s*obtained|\bmarks\b|\bscore\b/i),
  plainIntent('percentage', 'Overall Percentage', /percentage|\bpercent\b|pratishat|kitne\s*percent/i),
  plainIntent('cgpa', 'CGPA / Grade Point', /\bcgpa\b|\bsgpa\b|\bgpa\b|grade\s*point/i),
  plainIntent('aadhaar_number', 'Aadhaar Card Number', /aadha?ar|आधार/),
  plainIntent('pan_number', 'PAN Card Number', /\bpan\s*(?:card|number|no\.?)?\b|permanent\s*account\s*number/i),
  plainIntent('account_number', 'Bank Account Number', /account\s*(?:no\.?|number)|a\/c\s*(?:no\.?|number)/i),
  plainIntent('ifsc_code', 'IFSC Code', /\bifsc\b/i),
  plainIntent('consumer_number', 'Consumer Number (CA)', /consumer\s*(?:no\.?|number|id)|\bca\s*no\.?\b|meter\s*(?:no\.?|number)/i),
  plainIntent('total_amount', 'Amount Due', /total\s*amount|amount\s*due|bill\s*amount|amount\s*payable|kitna\s*(?:amount|paisa|rupaye|paise|bill)/i),
  plainIntent('due_date', 'Payment Due Date', /due\s*date|payment\s*due|last\s*date|due\s*kab/i),
  plainIntent('receipt_number', 'Receipt / Invoice No.', /receipt\s*(?:no\.?|number)|invoice\s*(?:no\.?|number)|bill\s*(?:no\.?|number)|tax\s*invoice/i),
  plainIntent('receipt_date', 'Receipt Date', /receipt\s*date|invoice\s*date|bill\s*date|kis\s*din\s*(?:ka|ki)/i),
  plainIntent('tax_amount', 'Tax / GST Amount', /\b(?:gst|cgst|sgst|igst|vat|tax)\b/i),
  plainIntent('payment_method', 'Payment Method', /payment\s*(?:mode|method)|paid\s*(?:by|via|through)|kis\s*se\s*pay/i)
];

const KIND_BY_CATEGORY: Record<string, string> = {
  'Identity Document': 'identity card (ID card)',
  'Marksheet / Degree': 'marksheet / degree document',
  'Utility Bill': 'utility bill',
  'Bank Statement / Passbook': 'bank statement / passbook',
  'Insurance Policy': 'insurance policy',
  'Receipt / Invoice': 'receipt / invoice',
  'Travel Ticket': 'travel ticket',
  'Application Form': 'application form',
  'General Document': 'document'
};

/** "Which kind of document/id is this?" — a real answer, not a field dump. */
function documentTypeAnswer(
  query: string,
  documents: any[],
  lang: 'en' | 'hinglish'
): GroundedResult | null {
  const q = query.toLowerCase();
  const asksType =
    /\b(?:what|which)\s+(?:kind|type|sort)\s+of\s+(?:id|card|document|file|paper|doc)\b/.test(q) ||
    /\b(?:what|which)\s+(?:kind|type)\s+(?:is\s+)?(?:it|this)\b/.test(q) ||
    /\bwhat\s+is\s+this\s+(?:id|card|document|file|paper|doc)\b/.test(q) ||
    /\bidentify\s+(?:this|the)\s+(?:document|file|id)\b/.test(q) ||
    /\b(?:kaun\s*sa|konsa)\s+(?:document|dastavej|id|card)\b/.test(q) ||
    /^\s*(?:ye|yeh)\s+(?:kya\s+hai|kaun\s*sa)\b/.test(q) ||
    // "is this a marksheet?" / "kya ye marksheet hai" — the document type must
    // be the SUBJECT right after the verb. "what is the receipt number" is a
    // field question, not a "what kind of document is this" question, and the
    // lookahead below keeps it out.
    /\b(?:is|kya)\s+(?:this|it|ye|yeh|the\s+document|the\s+file)?\s*(?:a|an|ek)?\s*(?:marksheet|id\s*card|aadhaar|pan\s*card|passport|bill|passbook|policy|invoice|receipt)\b(?!\s*(?:no\b|nos\b|number|date|amount|total|id\b))/.test(
      q
    ) ||
    // "kya ye bill hai" — short window, and never when the type is being asked
    // about as a labelled field ("aadhaar number kya hai" wants the number).
    /\b(?:marksheet|id\s*card|aadhaar|bill|passbook|policy|invoice)\b(?!\s*(?:no\b|number|date))[^?]{0,6}\b(?:hai|hain|kya|is\s*it)\b/.test(
      q
    );
  if (!asksType) return null;

  // "which kind of id is this" must look at the ID card, not documents[0]
  // (the newest upload was a train ticket, so the answer described the ticket).
  const resolved = resolveQueryDocument(query, documents);
  if (resolved.ambiguous || !resolved.doc) {
    return whichDocumentAnswer(lang, documents, query);
  }

  const doc = resolved.doc;
  const kind = KIND_BY_CATEGORY[doc.category] || 'document';
  const fieldOf = (name: string) => (doc.fields || []).find((f: any) => f.field_name === name);

  const rows: Array<[string, any]> = [
    ['Student / Holder', fieldOf('candidate_name')],
    ['Institution', fieldOf('school_name')],
    ['Course / Branch', fieldOf('course_branch')],
    ['Session / Validity', fieldOf('session')],
    ['Roll Number', fieldOf('roll_number')],
    ['Account Number', fieldOf('account_number')],
    ['Amount Due', fieldOf('total_amount')]
  ];
  const bullets = rows
    .filter(([, f]) => f)
    .map(([label, f]) => `- **${label}:** ${f.field_value}`)
    .join('\n');

  const isIdentity = doc.category === 'Identity Document';

  // "is it a marksheet?" style yes/no questions get a direct verdict first.
  const namedType = q.match(/\b(marksheet|id\s*card|aadhaar|pan\s*card|passport|bill|passbook|policy|invoice|receipt)\b/)?.[1];
  const kindMatchesNamed = namedType
    ? new RegExp(namedType.replace(/\s+/g, '\\s+'), 'i').test(kind)
    : true;
  const article = namedType && /^[aeiou]/i.test(namedType) ? 'an' : 'a';
  const verdict =
    namedType && !kindMatchesNamed
      ? lang === 'en'
        ? `No — this document is not ${article} ${namedType}. `
        : `Nahi — yeh ${namedType} nahi hai. `
      : namedType
        ? lang === 'en'
          ? `Yes — this document is ${article} ${namedType}. `
          : `Haan — yeh ${namedType} hai. `
        : '';

  const note = isIdentity
    ? lang === 'en'
      ? '\n\nIt is an identity card, not a marksheet — it carries no marks or grades.'
      : '\n\nYeh ek identity card hai, marksheet nahi — isme marks ya grades nahi hote.'
    : '';

  const kindArticle = /^[aeiou]/i.test(kind) ? 'an' : 'a';
  const answer =
    lang === 'en'
      ? `${verdict}**\`${doc.file_name}\`** is ${kindArticle} **${kind}**.${note}${
          bullets ? `\n\n${bullets}` : ''
        }\n\nAsk for any single field (blood group, date of birth, course…) and I will quote its exact line.`
      : `${verdict}**\`${doc.file_name}\`** ek **${kind}** hai.${note}${
          bullets ? `\n\n${bullets}` : ''
        }\n\nKisi bhi ek field (blood group, date of birth, course…) ke liye poochhein, main uski exact line quote karunga.`;

  const evidenceField = fieldOf('school_name') || fieldOf('candidate_name');
  return {
    route: 'document_retrieval',
    detected_language: lang,
    intent: 'document_type_profile',
    answer,
    is_insufficient_evidence: false,
    evidence: evidenceField
      ? [
          {
            document_id: doc.id,
            file_name: doc.file_name,
            field_name: evidenceField.display_label,
            value: evidenceField.field_value,
            snippet: evidenceField.evidence_text,
            page_number: 1,
            confidence: evidenceField.confidence
          }
        ]
      : [
          {
            document_id: doc.id,
            file_name: doc.file_name,
            snippet: (doc.raw_text || '').slice(0, 160),
            page_number: 1,
            confidence: 0.9
          }
        ]
  };
}

/**
 * Deterministic local query response grounded strictly in REAL extracted fields
 * and transcribed text. Never returns fabricated strings.
 */
export function answerQueryLocally(
  query: string,
  allDocuments: any[],
  scopeDocId?: number
): GroundedResult {
  // One document, one chat: when the viewer has a document open, every content
  // answer comes from THAT file. Inventory/greeting still describe the library;
  // an unknown scope id falls back to the whole library instead of answering
  // "no documents".
  const scoped = scopeDocId
    ? (allDocuments || []).filter((d: any) => d.id === scopeDocId)
    : [];
  const documents = scoped.length > 0 ? scoped : allDocuments || [];

  // Typo-tolerant, neg aware query ("collage name not candidate").
  const normQuery = normalizeQueryText(query);
  const excluded = excludedFieldGroups(query);
  // NOTE: "number" used to be a Hinglish marker, so plain English questions
  // ("what is my contact number") were answered in Hinglish. Language markers
  // must be words that are actually Hindi/Hinglish.
  const isHindiOrHinglish =
    /[ऀ-ॿ]/.test(query) ||
    /\b(kya|hai|batao|kaun|kitna|kitne|mera|meri|kiske|naam|kaise|konsa|ye|yeh|ko|se|me|mai|pr|aur|bhai|bata|dikhao)\b/i.test(query);

  // Greeting check
  if (/^(hi|hii|hello|hey|namaste|pranam|kaise ho)\b/i.test(normQuery)) {
    const docNames = allDocuments.map(d => d.file_name).join(', ');
    return {
      route: 'greeting',
      detected_language: isHindiOrHinglish ? 'hinglish' : 'en',
      intent: 'greeting',
      answer: isHindiOrHinglish
        ? `Namaste! Main **SafeDocAI** hoon, aapka personal document assistant. Aapke paas abhi **${allDocuments.length} documents** indexed hain (${docNames || 'koyi document nahi'}). Aap mujhse kisi bhi document ke roll number, marks, student name ya details ke baare me pooch sakte hain!`
        : `Hello! I am **SafeDocAI**, your privacy-first personal document intelligence assistant. You have **${allDocuments.length} documents** indexed (${docNames || 'none'}). Ask me any question regarding your documents!`,
      is_insufficient_evidence: false,
      evidence: [],
      clarification_options: []
    };
  }

  if (allDocuments.length === 0) {
    return {
      route: 'out_of_scope',
      detected_language: isHindiOrHinglish ? 'hinglish' : 'en',
      intent: 'no_documents',
      answer: isHindiOrHinglish
        ? 'Aapke paas abhi koi document upload nahi hai. Kripya pehle document upload karein.'
        : 'No documents are currently indexed. Please upload a document first.',
      is_insufficient_evidence: true,
      evidence: []
    };
  }

  // 0. Inventory questions are answered instantly, no model needed.
  //    Inventory always describes the whole library, even when one document
  //    is open in the viewer.
  const inventory = inventoryAnswer(query, allDocuments, isHindiOrHinglish ? 'hinglish' : 'en');
  if (inventory) return inventory;

  const lang: 'en' | 'hinglish' = isHindiOrHinglish ? 'hinglish' : 'en';

  // 0a. "Which kind of document/id is this?" — answer the question itself.
  const typeProfile = documentTypeAnswer(query, documents, lang);
  if (typeProfile) return typeProfile;

  // 0b. Plain field questions (blood group, DOB, contact, course, address…).
  //     Extracted field first; stored-transcription line as an honest fallback;
  //     if neither exists, say so for THAT field — never fall through to a
  //     looser "name" branch (that is how "college name" returned a person).
  /** Label of the field this question asked for (null when it asked for none). */
  let askedFieldLabel: string | null = null;

  for (const intent of FIELD_INTENTS) {
    if (!intent.test.test(normQuery)) continue;
    if (excluded.has(intent.field)) continue;

    const candidates = documents.filter((d: any) =>
      (d.fields || []).some((f: any) => f.field_name === intent.field)
    );
    if (candidates.length === 1) {
      const field = (candidates[0].fields || []).find(
        (f: any) => f.field_name === intent.field
      );
      if (field) return fieldAnswer(lang, candidates[0], field);
    }
    if (candidates.length > 1) {
      const resolved = resolveQueryDocument(query, candidates);
      if (resolved.doc) {
        const field = (resolved.doc.fields || []).find(
          (f: any) => f.field_name === intent.field
        );
        if (field) return fieldAnswer(lang, resolved.doc, field);
      }
      return whichDocumentAnswer(lang, candidates, query);
    }

    if (!intent.raw) {
      // No bespoke transcription regex for this field: let the general ladder
      // below quote the real line or chunk before declaring anything missing —
      // and do not let a single-field question fall into the summary/name
      // branches ("what is my percentage" used to return a whole field dump).
      askedFieldLabel = intent.label;
      break;
    }

    const rawHits = documents
      .map((d: any) => ({ doc: d, hit: intent.raw?.(d) }))
      .filter((x: { hit: { value: string; evidence: string } | null | undefined }) => Boolean(x.hit)) as Array<{
        doc: any;
        hit: { value: string; evidence: string };
      }>;
    if (rawHits.length === 1) {
      const { doc, hit } = rawHits[0];
      return rawLineAnswer(lang, doc, intent.label, hit.value, hit.evidence);
    }

    const resolved = resolveQueryDocument(query, documents);
    if (resolved.doc) return notFoundAnswer(lang, resolved.doc, query, intent.label);
    return whichDocumentAnswer(lang, documents, query);
  }

  // 1. Roll Number query
  if (
    !askedFieldLabel &&
    (normQuery.includes('roll') || normQuery.includes('enrolment') || normQuery.includes('enrollment') || normQuery.includes('hall ticket'))
  ) {
    for (const doc of documents) {
      const field = (doc.fields || []).find((f: any) => f.field_name === 'roll_number');
      let rollVal = field?.field_value;

      if (!rollVal && doc.raw_text) {
        const rollMatch = doc.raw_text.match(/(?:roll\s*(?:no\.?|number|code)|enrol(?:l)?ment)\s*[:=-]?\s*([A-Z0-9_\/-]{4,20})/i);
        if (rollMatch) {
          rollVal = cleanExtractedValue(rollMatch[1]);
        }
      }

      if (rollVal) {
        return {
          route: 'exact_field',
          detected_language: isHindiOrHinglish ? 'hinglish' : 'en',
          intent: 'roll_number_lookup',
          answer: isHindiOrHinglish
            ? `Aapka **Roll Number** hai: **\`${rollVal}\`**\n\n📌 **Source Document:** \`${doc.file_name}\`\n🔍 **Evidence:** "${field?.evidence_text || `Roll No: ${rollVal}`}"`
            : `Your **Roll Number** is **\`${rollVal}\`**.\n\n📌 **Source Document:** \`${doc.file_name}\`\n🔍 **Evidence:** "${field?.evidence_text || `Roll No: ${rollVal}`}"`,
          is_insufficient_evidence: false,
          evidence: [
            {
              document_id: doc.id,
              file_name: doc.file_name,
              field_name: 'Roll Number',
              value: rollVal,
              snippet: field?.evidence_text || `Roll No: ${rollVal}`,
              page_number: 1,
              confidence: 0.99
            }
          ]
        };
      }
    }

    return {
      route: 'exact_field',
      detected_language: isHindiOrHinglish ? 'hinglish' : 'en',
      intent: 'roll_number_not_found',
      answer: isHindiOrHinglish
        ? `Aapke uploaded documents me se kisi me bhi **Roll Number** ka spasht record nahi mila. Kripya check karein ki uploaded image/PDF me roll number clearly visible hai ya nahi.`
        : `Could not locate a **Roll Number** in your uploaded documents. Please check if the roll number is clearly legible in the uploaded file.`,
      is_insufficient_evidence: true,
      evidence: []
    };
  }

  // 2. Candidate / Student Name query — skipped when the user excluded it, and
  //    skipped for "<qualifier> name" questions ("principal name", "college
  //    name", "father name"), which ask about that labelled field — answering
  //    them with the person's name was plainly wrong.
  if (
    !askedFieldLabel &&
    !excluded.has('candidate_name') &&
    !NAME_QUALIFIER.test(normQuery) &&
    (normQuery.includes('naam') || normQuery.includes('name') || normQuery.includes('kiske'))
  ) {
    // Collect the name from EVERY document (not just the first one that has any
    // text shaped like a name), then answer once or list the differences.
    const found: Array<{ doc: any; field: any; nameVal: string }> = [];
    for (const doc of documents) {
      const field = (doc.fields || []).find((f: any) => f.field_name === 'candidate_name' || f.field_name === 'consumer_name');
      let nameVal: string | undefined = field?.field_value;

      if (!nameVal && doc.raw_text) {
        const nameMatch = doc.raw_text.match(/(?:candidate(?:\s*['’]s)?\s*name|student(?:\s*['’]s)?\s*name|\bname)\s*[:=-]\s*([^\n\r]{2,50})/i);
        if (nameMatch) {
          const candidate = cleanExtractedValue(nameMatch[1]);
          if (looksLikePersonName(candidate)) nameVal = candidate;
        }
      }
      if (nameVal && looksLikePersonName(nameVal)) found.push({ doc, field, nameVal });
    }

    if (found.length > 0) {
      const lang = isHindiOrHinglish ? 'hinglish' : 'en';
      const norm = (v: string) => v.toLowerCase().replace(/[^a-z0-9]+/g, '');
      const distinct = new Set(found.map(f => norm(f.nameVal)));
      const files = found.map(f => `\`${f.doc.file_name}\``).join(', ');
      const answer =
        distinct.size === 1
          ? isHindiOrHinglish
            ? `Aapka naam: **${found[0].nameVal}**\n\n📌 **Document${found.length > 1 ? 's' : ''}:** ${files}`
            : `Name: **${found[0].nameVal}**\n\n📌 **Document${found.length > 1 ? 's' : ''}:** ${files}`
          : `${isHindiOrHinglish ? 'Alag-alag documents me alag naam hai:' : 'The documents show different names:'}\n${found
              .map(f => `- **\`${f.doc.file_name}\`**: ${f.nameVal}`)
              .join('\n')}`;
      return {
        route: 'exact_field',
        detected_language: lang,
        intent: 'name_lookup',
        answer,
        is_insufficient_evidence: false,
        evidence: found.map(f => ({
          document_id: f.doc.id,
          file_name: f.doc.file_name,
          field_name: f.field?.display_label || 'Name',
          value: f.nameVal,
          snippet: f.field?.evidence_text || f.nameVal,
          page_number: 1,
          confidence: f.field?.confidence || 0.9
        }))
      };
    }
  }

  // 3. Explicit summary / details request only.
  //    "number", "kya hai" and bare "batao" used to be funneled here, so any
  //    question the router did not recognise got the same all-fields dump.
  // "What is written in this document" / "isme kya likha hai" asks for the
  // whole file, not a missing field — it used to answer "not recorded".
  const asksForWholeContent =
    /\bwhat(?:'s|\s+is)?\s+(?:written|mentioned|there)\s+in\b/.test(normQuery) ||
    /\bwhat\s+does\s+(?:this|the|my)\s+\w+\s+(?:say|contain|have)\b/.test(normQuery) ||
    /\bwhat(?:'s|\s+is)\s+(?:in|inside)\s+(?:this|the|my)\s+(?:document|file|pdf|card|doc)\b/.test(
      normQuery
    ) ||
    /\b(?:kya|kyaa)\s+likha\s+(?:hai|hain)?\b/.test(normQuery) ||
    /\b(?:kis|kya)\s*baare\s+(?:me|mein)\b/.test(normQuery) ||
    /\b(?:what(?:'s|\s+is)?\s+(?:this|the|it)(?:\s+document)?\s+about)\b/.test(normQuery) ||
    /\b(?:is|iss|us|ye|yeh)\s+(?:document|file|pdf|card|doc)\s+(?:me|mein)\s+(?:kya|kya\s+kya)\b/.test(
      normQuery
    ) ||
    /\b(?:document|file|pdf|card|doc)\s+(?:me|mein)\s+(?:kya|kya\s+kya)\s+(?:hai|hain|likha)?\b/.test(
      normQuery
    );

  const wantsSummary =
    !askedFieldLabel &&
    (asksForWholeContent ||
      /\b(marksheet|marks|percentage|result|details?|summary|summari[sz]e|overview)\b/.test(normQuery) ||
    /\b(?:10th|12th)\b/.test(normQuery) ||
    /\b(all\s+(?:the\s+)?(?:information|details)|poori\s*jankari|sab\s*kuch|kya\s*kya\s+hai|sab\s+batao|sab\s+details)\b/.test(
      normQuery
    ) ||
    /\btell\s+me\s+about\s+(?:this|the|my)\s+(?:document|file|pdf|id)\b/.test(normQuery));

  if (wantsSummary) {
    const targetDoc =
      resolveQueryDocument(query, documents).doc ||
      documents.find(d =>
        d.category === 'Marksheet / Degree' ||
        d.file_name.toLowerCase().includes('marksheet') ||
        d.file_name.toLowerCase().includes('10th') ||
        d.file_name.toLowerCase().includes('college') ||
        d.file_name.toLowerCase().includes('degree')
      ) ||
      documents[0];

    if (targetDoc) {
      const fields = targetDoc.fields || [];
      const lines: string[] = [];

      for (const f of fields) {
        lines.push(`- **${f.display_label}:** **${f.field_value}**`);
      }

      if (lines.length > 0) {
        return {
          route: 'exact_field',
          detected_language: isHindiOrHinglish ? 'hinglish' : 'en',
          intent: 'document_details',
          answer: isHindiOrHinglish
            ? `### 📄 **Document Details (${targetDoc.file_name}):**\n\n` +
              lines.join('\n') +
              `\n\n*Yeh saare records aapke document ke actual OCR text se extract kiye gaye hain.*`
            : `### 📄 **Document Summary (${targetDoc.file_name}):**\n\n` +
              lines.join('\n') +
              `\n\n*All details extracted directly from actual OCR transcription.*`,
          is_insufficient_evidence: false,
          evidence: fields.slice(0, 3).map((f: any) => ({
            document_id: targetDoc.id,
            file_name: targetDoc.file_name,
            field_name: f.display_label,
            value: f.field_value,
            snippet: f.evidence_text,
            page_number: 1,
            confidence: f.confidence
          }))
        };
      }

      const textSnippet = targetDoc.raw_text?.substring(0, 500) || '';
      return {
        route: 'semantic_rag',
        detected_language: isHindiOrHinglish ? 'hinglish' : 'en',
        intent: 'raw_ocr_summary',
        answer: isHindiOrHinglish
          ? `### 📄 **Document Content (${targetDoc.file_name}):**\n\nDocument ka OCR text nimnlikhit hai:\n\n\`\`\`\n${textSnippet}\n\`\`\`\n\nAap isme se kisi bhi specific field ke baare me pooch sakte hain.`
          : `### 📄 **Document Content (${targetDoc.file_name}):**\n\nTranscribed text excerpt:\n\n\`\`\`\n${textSnippet}\n\`\`\`\n\nFeel free to ask about any specific item in this document.`,
        is_insufficient_evidence: false,
        evidence: [
          {
            document_id: targetDoc.id,
            file_name: targetDoc.file_name,
            snippet: textSnippet.substring(0, 150),
            page_number: 1,
            confidence: 0.95
          }
        ]
      };
    }
  }

  // 4. Nothing matched a known field: answer from THIS document's text
  //    (verbatim line → chunk), otherwise say honestly what is missing.
  const primaryDoc = resolveQueryDocument(query, documents).doc;
  if (!primaryDoc) return whichDocumentAnswer(lang, documents, query);

  // A "<subject> name/number" question must be answered by text that actually
  // mentions the subject: without this, "principal name" quoted the nearest
  // "Name : …" line even though the principal is not in the file.
  const subject = subjectQualifier(normQuery);
  const mentionsSubject = (text: string) =>
    !subject || text.toLowerCase().includes(subject);

  const lineHit = bestMatchingLine(query, primaryDoc.raw_text || '');
  if (lineHit && mentionsSubject(lineHit.line)) {
    return excerptAnswer(lang, primaryDoc, lineHit.line, 'transcribed_line_answer');
  }

  const chunkHit = bestMatchingChunk(query, primaryDoc);
  if (chunkHit && mentionsSubject(chunkHit.text)) {
    return excerptAnswer(lang, primaryDoc, chunkHit.text, 'chunk_excerpt_answer');
  }

  return notFoundAnswer(lang, primaryDoc, query, askedFieldLabel || undefined, subject || undefined);
}

/**
 * The labelled subject a question is about ("principal name", "train number").
 * Used to refuse quoting unrelated text that merely shares a generic word.
 */
const SUBJECT_NOUNS =
  /\b(aadhaar|aadhar|account|bank|bill|board|branch|branch\s*code|college|company|doctor|enrol{1,2}ment|father|guardian|hospital|hotel|ifsc|institute|institution|invoice|manager|marksheet|mother|owner|pan|parent|passbook|passport|pnr|policy|principal|receipt|school|seller|shop|store|student|ticket|train|university|vendor|merchant)\b/i;

function subjectQualifier(query: string): string | null {
  const m = query.match(SUBJECT_NOUNS);
  return m ? stemKey(m[1].toLowerCase()) : null;
}
