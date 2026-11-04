/**
 * Local chunk retrieval (TRD §8): given a question, pick the few stored chunks
 * that can actually answer it, so the LLM prompt stays tiny and relevant.
 *
 * Lexical BM25 on purpose: instant, fully offline, zero extra model download,
 * deterministic and unit-testable. Handles English, Hinglish (roman) and
 * Devanagari tokens. An embedding rerank can be layered on top later without
 * changing this module's contract.
 */
import { chunkText } from './chunker';

export interface ChunkLike {
  chunk_id?: string;
  document_id?: number;
  file_name?: string;
  page_number?: number;
  chunk_index?: number;
  text: string;
}

export interface RetrievalHit {
  doc: any;
  chunk: ChunkLike;
  score: number;
  matched: string[];
}

export interface RetrieveOptions {
  /** Max chunks returned (default 3). */
  k?: number;
  /** Share of the question's content terms a chunk must contain (default 0.34). */
  minCoverage?: number;
}

const STOPWORDS = new Set(
  (
    // English
    'a an the is are was were be been am do does did of in on at to for from by with about as and or but if then ' +
    'what which who whom whose when where why how this that these those it its my me mine your you i we our they their ' +
    'can could should would will shall may might must any all some tell show give find get please there here ' +
    // Hinglish / Hindi function words
    'kya hai hain ho tha thi the ka ki ke ko se me mein mai main pe par pr aur ya bhi hi to toh ye yeh wo woh ' +
    'mera meri mere apna apni tera teri kaun kaunsa konsa kitna kitne kitni kab kahan kaise kyun kyu ' +
    'batao bata dikhao dikha de do dena hu hoon bhai yaar ' +
    'का की के को से में मैं पर और या भी तो ये यह वो है हैं था थी क्या मेरा मेरी मेरे कौन कितना कब कहाँ कैसे बताओ'
  ).split(/\s+/)
);

const TOKEN_RE = /[\p{L}\p{M}\p{N}]+/gu;

/** Cheap English plural fold so "marks"/"mark", "dates"/"date" meet. */
function stem(word: string): string {
  if (/^[a-z]+$/.test(word) && word.length > 3 && word.endsWith('s') && !word.endsWith('ss')) {
    return word.slice(0, -1);
  }
  return word;
}

/** All tokens, lowercased and stemmed (stopwords kept). */
export function tokenize(text: string): string[] {
  return ((text || '').toLowerCase().match(TOKEN_RE) || []).map(stem);
}

/** Question terms that carry meaning (stopwords and 1-char tokens removed). */
export function contentTerms(query: string): string[] {
  const raw = (query || '').toLowerCase().match(TOKEN_RE) || [];
  const out: string[] = [];
  for (const w of raw) {
    if (STOPWORDS.has(w) || w.length < 2) continue;
    const s = stem(w);
    if (!out.includes(s)) out.push(s);
  }
  return out;
}

/** Garbled OCR must never reach the prompt: need mostly real letters/digits. */
export function isUsableChunk(text: string): boolean {
  const compact = (text || '').replace(/\s+/g, '');
  if (compact.length < 20) return false;
  const good = (compact.match(/[\p{L}\p{M}\p{N}]/gu) || []).length;
  return good / compact.length >= 0.7;
}

function chunksOf(doc: any): ChunkLike[] {
  if (Array.isArray(doc.chunks) && doc.chunks.length > 0) return doc.chunks;
  return chunkText(doc.id, doc.file_name || '', doc.raw_text || '');
}

const K1 = 1.2;
const B = 0.75;

export function retrieveChunks(query: string, docs: any[], options: RetrieveOptions = {}): RetrievalHit[] {
  const k = options.k ?? 3;
  const minCoverage = options.minCoverage ?? 0.34;
  const qTerms = contentTerms(query);
  if (qTerms.length === 0) return [];

  type Cand = { doc: any; chunk: ChunkLike; tokens: string[]; tf: Map<string, number> };
  const cands: Cand[] = [];
  for (const doc of docs) {
    if (doc.status && doc.status !== 'success') continue;
    for (const chunk of chunksOf(doc)) {
      if (!isUsableChunk(chunk.text)) continue;
      const tokens = tokenize(chunk.text);
      const tf = new Map<string, number>();
      for (const t of tokens) tf.set(t, (tf.get(t) || 0) + 1);
      cands.push({ doc, chunk, tokens, tf });
    }
  }
  if (cands.length === 0) return [];

  const N = cands.length;
  const avgdl = cands.reduce((s, c) => s + c.tokens.length, 0) / N || 1;
  const df = new Map<string, number>();
  for (const t of qTerms) df.set(t, cands.filter(c => c.tf.has(t)).length);

  const hits: RetrievalHit[] = [];
  for (const c of cands) {
    let score = 0;
    const matched: string[] = [];
    for (const t of qTerms) {
      const f = c.tf.get(t) || 0;
      if (f === 0) continue;
      matched.push(t);
      const n = df.get(t) || 0;
      const idf = Math.log(1 + (N - n + 0.5) / (n + 0.5));
      score += (idf * f * (K1 + 1)) / (f + K1 * (1 - B + (B * c.tokens.length) / avgdl));
    }
    if (matched.length === 0) continue;
    if (matched.length / qTerms.length < minCoverage) continue;

    // Small nudge when the question names the file/category ("my bank statement").
    const nameTokens = new Set(tokenize(`${c.doc.file_name || ''} ${c.doc.category || ''}`));
    for (const t of qTerms) if (nameTokens.has(t)) score += 0.3;

    hits.push({ doc: c.doc, chunk: c.chunk, score, matched });
  }

  hits.sort((a, b) => b.score - a.score);
  return hits.slice(0, k);
}
