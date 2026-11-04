/**
 * OCR quality pipeline (PRD FR-05, TRD §2 "preprocessing").
 *
 * Findings from real ID-card photos that drive this module:
 *   1. Page segmentation must be AUTO (set on the worker in extract.ts). With
 *      tesseract.js' default, columns merge and lines such as the white-on-colour
 *      college banner and the line under "Distt." were silently dropped.
 *   2. Phone photos are often 2000+ px wide; we grayscale, stretch contrast and
 *      cap the width, which is faster and reads more reliably.
 *   3. Raw OCR output is full of low-confidence junk (icon glyphs, vertical
 *      text, random Devanagari). Word confidences let us drop that junk and keep
 *      one clean line per text line, in reading order.
 *
 * `sharp` is an optional dependency: without it we OCR the original image and
 * still clean the transcript.
 */

export interface OcrWord {
  text: string;
  confidence: number;
  bbox: { x0: number; y0: number; x1: number; y1: number };
}

export interface OcrLine {
  words: OcrWord[];
}

/**
 * First pass never uses more than this width. Measured on a real 2333 px ID-card
 * photo: 1800 px read every field; 1400 px dropped the address and a digit.
 * (tesseract.js results are not monotonic in scale, so we do not guess a
 * "perfect" size: we use a sane cap and a rescue pass for poor results.)
 */
export const FIRST_PASS_MAX_WIDTH = 1800;
/** A pass with fewer trustworthy words than this triggers a rescue attempt. */
export const POOR_PASS_WORDS = 12;

export interface PrepareOptions {
  /** Target width in px (default: original width, capped at FIRST_PASS_MAX_WIDTH). */
  width?: number;
  /** Grayscale + contrast stretch (default true). */
  gray?: boolean;
}

/**
 * Rescue attempts, in order. Grayscale is the default because it was perfect on
 * real ID-card photos, but it erases coloured banners (a 728x90 ticket banner
 * went from 7 lines to 0), so the first rescue keeps the colours.
 */
export const RESCUE_VARIANTS: Array<{ scale: number; gray: boolean }> = [
  { scale: 1, gray: false },
  { scale: 0.65, gray: true }
];

const DEVANAGARI = /[\u0900-\u097F]/;
const HAS_DIGIT = /\d/;
const HAS_ALNUM = /[\p{L}\p{N}]/u;
const KEEP_SHORT = new Set(['no', 'id', 'dr', 'mr', 'ms', 'of', 'to', 'in', 'on', 'at', 'by', 'or', 'is', 'as', 'rs', 'pn']);
const KEEP_SYMBOLS = new Set([':', '&', '-', '/', '|', ',', '.', '+', '(', ')', '₹', '%', '#']);

/* ------------------------------------------------------------------ */
/* Structure helpers                                                   */
/* ------------------------------------------------------------------ */

/** Flatten tesseract.js `blocks` into lines of words (reading order). */
export function linesFromBlocks(blocks: any[] | null | undefined): OcrLine[] {
  const lines: OcrLine[] = [];
  for (const block of blocks || []) {
    for (const para of block?.paragraphs || []) {
      for (const line of para?.lines || []) {
        const words: OcrWord[] = (line?.words || [])
          .filter((w: any) => w && typeof w.text === 'string' && w.bbox)
          .map((w: any) => ({ text: w.text.trim(), confidence: Number(w.confidence) || 0, bbox: w.bbox }))
          .filter((w: OcrWord) => w.text.length > 0);
        if (words.length > 0) lines.push({ words });
      }
    }
  }
  return lines;
}

function median(values: number[]): number {
  const s = [...values].sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

/** Median height of confident, real words (ignores logos and stray glyphs). */
export function medianWordHeight(lines: OcrLine[]): number | null {
  const strong: number[] = [];
  const any: number[] = [];
  for (const line of lines) {
    for (const w of line.words) {
      const h = w.bbox.y1 - w.bbox.y0;
      if (h <= 0) continue;
      if (w.confidence >= 50 && w.text.length >= 3) any.push(h);
      if (w.confidence >= 60 && w.text.length >= 4 && /^\p{L}+$/u.test(w.text)) strong.push(h);
    }
  }
  const pool = strong.length >= 5 ? strong : any;
  return pool.length >= 5 ? median(pool) : null;
}

/** How much trustworthy text a pass produced (used to pick the better pass). */
export function passScore(lines: OcrLine[]): number {
  let n = 0;
  for (const line of lines) {
    for (const w of line.words) if (w.confidence >= 60 && w.text.length >= 3) n++;
  }
  return n;
}

/* ------------------------------------------------------------------ */
/* Transcript cleanup                                                  */
/* ------------------------------------------------------------------ */

function keepWord(w: OcrWord): boolean {
  const t = w.text;
  const c = w.confidence;
  if (!HAS_ALNUM.test(t)) return KEEP_SYMBOLS.has(t) && c >= 30;
  // Random Devanagari is the most common OCR hallucination on English cards.
  if (DEVANAGARI.test(t)) return c >= 60;
  // IDs, dates and amounts: keep unless the engine is nearly sure it is junk.
  if (HAS_DIGIT.test(t)) return c >= 35;
  if (c < 45) return false;
  if (t.length <= 2 && c < 75 && !KEEP_SHORT.has(t.toLowerCase())) return false;
  return true;
}

/** One OCR line -> clean text, or null if nothing trustworthy is left. */
export function cleanLine(line: OcrLine): string | null {
  const kept = line.words.filter(keepWord);
  // A line that starts with a lone symbol (logo / border artefact) loses it.
  while (kept.length > 0 && !HAS_ALNUM.test(kept[0].text) && kept[0].text !== ':') kept.shift();
  const text = kept.map(w => w.text).join(' ').replace(/\s+/g, ' ').trim();
  const alnum = (text.match(/[\p{L}\p{N}]/gu) || []).length;
  return alnum < 2 ? null : text; // vertical "I D E N T I T Y" letters, icons, specks
}

/**
 * Turn OCR lines into a clean transcript: one text line per output line,
 * junk words removed, no leading symbol debris.
 */
export function cleanLines(lines: OcrLine[]): string {
  const out: string[] = [];
  for (const line of lines) {
    const text = cleanLine(line);
    if (text) out.push(text);
  }
  return out.join('\n');
}

/* ------------------------------------------------------------------ */
/* Image preparation (sharp is optional)                               */
/* ------------------------------------------------------------------ */

let sharpModule: any | null | undefined;

async function loadSharp(): Promise<any | null> {
  if (sharpModule !== undefined) return sharpModule;
  try {
    sharpModule = (await import('sharp')).default;
  } catch {
    console.warn('[ocr] sharp not available; OCR will use the original image (install optional dependency "sharp" for better accuracy)');
    sharpModule = null;
  }
  return sharpModule;
}

export interface PreparedImage {
  buffer: Buffer;
  width: number;
  height: number;
}

/**
 * Optional grayscale + contrast stretch, then resize. Returns null if sharp is
 * missing or the image cannot be decoded (caller then OCRs the original).
 */
export async function prepareImage(input: Buffer, options: PrepareOptions = {}): Promise<PreparedImage | null> {
  const sharp = await loadSharp();
  if (!sharp) return null;
  try {
    const meta = await sharp(input, { failOn: 'none' }).metadata();
    const swapped = (meta.orientation || 1) >= 5;
    const originalWidth = (swapped ? meta.height : meta.width) || 0;
    if (!originalWidth) return null;
    const target = Math.round(
      Math.min(3200, Math.max(500, options.width ?? Math.min(originalWidth, FIRST_PASS_MAX_WIDTH)))
    );
    let pipeline = sharp(input, { failOn: 'none' }).rotate(); // honour EXIF orientation
    if (options.gray !== false) pipeline = pipeline.grayscale().normalise({ lower: 1, upper: 99 });
    const { data, info } = await pipeline
      .resize({ width: target, kernel: 'lanczos3' })
      .png()
      .toBuffer({ resolveWithObject: true });
    return { buffer: data, width: info.width, height: info.height };
  } catch (err) {
    console.warn('[ocr] image preparation failed, using original:', String((err as Error)?.message || err).slice(0, 120));
    return null;
  }
}

/* ------------------------------------------------------------------ */
/* Orchestration                                                       */
/* ------------------------------------------------------------------ */

interface Pass {
  lines: OcrLine[];
  rawText: string;
}

async function recognizePass(worker: any, image: Buffer): Promise<Pass> {
  const result = await worker.recognize(image, {}, { text: true, blocks: true });
  return {
    lines: linesFromBlocks(result?.data?.blocks),
    rawText: (result?.data?.text || '').trim()
  };
}

/** Max rescue passes (OCR_RESCUE=0 disables them for the fastest uploads). */
function rescueBudget(): number {
  const n = Number(process.env.OCR_RESCUE);
  return Number.isFinite(n) && n >= 0 ? Math.min(RESCUE_VARIANTS.length, Math.floor(n)) : 1;
}

/**
 * OCR one page image. `worker` is a tesseract.js worker (reused by the caller).
 * Returns a clean, line-per-line transcript. One pass normally; one extra pass
 * (colour instead of grayscale) only if the first produced little trustworthy text.
 */
export async function runAdaptiveOcr(worker: any, buffer: Buffer): Promise<string> {
  const first = await prepareImage(buffer);
  let best = await recognizePass(worker, first ? first.buffer : buffer);

  if (first && passScore(best.lines) < POOR_PASS_WORDS) {
    for (const variant of RESCUE_VARIANTS.slice(0, rescueBudget())) {
      const prepared = await prepareImage(buffer, { width: first.width * variant.scale, gray: variant.gray });
      if (!prepared) continue;
      const retry = await recognizePass(worker, prepared.buffer);
      if (passScore(retry.lines) > passScore(best.lines)) best = retry;
    }
  }

  const cleaned = cleanLines(best.lines);
  // Never return nothing: if cleanup removed everything, fall back to raw text.
  return cleaned.length > 0 ? cleaned : best.rawText;
}
