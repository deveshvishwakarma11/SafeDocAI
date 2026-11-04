/**
 * SafeDocAI extraction pipeline (fully offline capable):
 *  - PDF text via pdf-parse (with an honest raw-stream fallback)
 *  - Scanned PDFs / images via Tesseract.js; if data/tessdata holds local
 *    traineddata files the worker loads them from disk (offline, no CDN)
 *  - Deterministic regex entity extractor that only returns values that
 *    actually exist in the transcribed text — never fabricated data.
 */
import { JUNK_VALUE, ORG_WORDS, looksLikePersonName } from './text-utils';
import { runAdaptiveOcr } from './ocr-pipeline';
import { hasStructure, headerFacts, normKey, parseStructure } from './structure';
import fs from 'node:fs';
import path from 'node:path';
// @ts-ignore
import { PDFParse } from 'pdf-parse';
// @ts-ignore
import Tesseract from 'tesseract.js';

export interface RawEntityAnalysis {
  category: string;
  page_count: number;
  fields: Array<{
    field_name: string;
    display_label: string;
    field_value: string;
    evidence_text: string;
    page_number: number;
    confidence: number;
    value_type: string;
  }>;
}

/* ------------------------------------------------------------------ */
/* OCR worker (singleton)                                              */
/* ------------------------------------------------------------------ */

export type OcrStatus = 'active_eng_hin' | 'active_eng_only' | 'unavailable';

let workerPromise: Promise<any> | null = null;
let ocrState: OcrStatus = 'unavailable';

export function ocrStatus(): OcrStatus {
  return ocrState;
}

/** Languages we can serve OFFLINE from a local tessdata directory. */
export function pickOcrLanguages(tessdataDir: string): {
  langs: string;
  options: { langPath: string; cachePath: string; gzip: boolean } | undefined;
} {
  const available = ['eng', 'hin'].filter(l =>
    fs.existsSync(path.join(tessdataDir, `${l}.traineddata`))
  );
  if (available.length > 0) {
    return {
      langs: available.join('+'),
      // `gzip: false` — shipped traineddata files are plain, not .gz;
      // tesseract.js would otherwise request `<lang>.traineddata.gz` and crash.
      options: {
        langPath: tessdataDir,
        cachePath: path.join(tessdataDir, '.cache'),
        gzip: false
      }
    };
  }
  // No local files: fall back to tesseract.js defaults (CDN or its own cache) —
  // reported honestly through ocrStatus().
  return { langs: 'eng', options: undefined };
}

/**
 * Choose which embedded JPEGs to OCR for a scanned PDF:
 * stream order (≈ page order), skip tiny decorative images, cap page count so
 * one huge scan cannot pin the CPU for minutes during upload.
 */
export const MAX_OCR_PAGES = 10;
export const MIN_OCR_IMAGE_BYTES = 12_000;

export function selectOcrImages(images: Buffer[]): Buffer[] {
  return images.filter(img => img.length >= MIN_OCR_IMAGE_BYTES).slice(0, MAX_OCR_PAGES);
}

/** Join per-page OCR transcripts with form-feed page breaks (page order kept). */
export function joinOcrPageTexts(pageTexts: string[]): string {
  return pageTexts
    .map(t => (t || '').trim())
    .filter(t => t.length > 0)
    .join('\f');
}

export async function getTesseractWorker(): Promise<any> {
  if (!workerPromise) {
    workerPromise = (async () => {
      try {
        const tessdataDir = path.resolve(process.cwd(), 'data', 'tessdata');
        const { langs, options } = pickOcrLanguages(tessdataDir);
        // errorHandler is REQUIRED: without it tesseract.js `throw`s inside its
        // message handler for a bad/corrupt image, which lands as an uncaught
        // exception and takes the whole server down. The job promise still
        // rejects, so callers' try/catch keeps containing per-page OCR failures.
        const worker = await Tesseract.createWorker(langs, 1, {
          ...options,
          errorHandler: (err: unknown) => {
            console.warn('[ocr] tesseract worker error (contained):', String(err)?.slice(0, 300));
          }
        });
        // Automatic page segmentation (PSM 3). tesseract.js' default treats the page
        // as one block, which merges columns and drops lines on cards and forms.
        await worker.setParameters({
          tessedit_pageseg_mode: Tesseract.PSM.AUTO,
          preserve_interword_spaces: '1'
        });
        ocrState = langs.includes('hin') ? 'active_eng_hin' : 'active_eng_only';
        return worker;
      } catch (err) {
        console.error(
          '[ocr] Tesseract worker init failed (offline and no cached traineddata?):',
          (err as Error).message
        );
        ocrState = 'unavailable';
        workerPromise = null;
        throw err;
      }
    })();
  }
  return workerPromise;
}

export async function recognizeImage(buffer: Buffer): Promise<string> {
  const worker = await getTesseractWorker();
  return (await runAdaptiveOcr(worker, buffer)).trim();
}

/* ------------------------------------------------------------------ */
/* PDF text extraction                                                 */
/* ------------------------------------------------------------------ */

export async function extractTextFromPdfBuffer(
  buffer: Buffer
): Promise<{ text: string; pages: number }> {
  try {
    const uint8 = new Uint8Array(buffer);
    const parser: any = new PDFParse(uint8);
    if (typeof parser.load === 'function') {
      await parser.load();
    }
    const result = await parser.getText();
    const rawText = result?.text?.trim() || '';
    const cleanText = rawText.replace(/\n{3,}/g, '\n\n').trim();
    // Keep ANY real text layer, however short. A 4-char layer still beats an
    // "OCR could not read this" note — it is genuine extracted text, and the
    // scanned-PDF path (length < 30) will still OCR the pages on top of it.
    if (cleanText.length > 0) {
      return { text: cleanText, pages: result?.total || 1 };
    }
  } catch (err) {
    console.warn('PDFParse failed, checking fallback:', (err as Error).message);
  }

  // Fallback: extract text tokens from uncompressed / partial PDF text streams
  try {
    const str = buffer.toString('latin1');
    const textTokens: string[] = [];
    const tjRegex = /\(([^)\\]{2,})\)\s*Tj/g;
    let match: RegExpExecArray | null;
    while ((match = tjRegex.exec(str)) !== null) {
      const token = match[1].replace(/\\([()\\])/g, '$1').trim();
      if (token && !/^[\x00-\x1F]+$/.test(token)) {
        textTokens.push(token);
      }
    }
    if (textTokens.length > 0) {
      return { text: textTokens.join(' '), pages: 1 };
    }
  } catch {
    // Ignore regex stream fallback error
  }

  return { text: '', pages: 1 };
}

/** Extract embedded JPEG images from a scanned PDF buffer (largest first). */
export function extractImagesFromPdfBuffer(buffer: Buffer): Buffer[] {
  const images: Buffer[] = [];
  let i = 0;
  while (i < buffer.length - 3) {
    if (buffer[i] === 0xff && buffer[i + 1] === 0xd8 && buffer[i + 2] === 0xff) {
      const start = i;
      let j = start + 3;
      while (j < buffer.length - 1) {
        if (buffer[j] === 0xff && buffer[j + 1] === 0xd9) {
          const imgBuf = buffer.subarray(start, j + 2);
          if (imgBuf.length > 3000) {
            images.push(imgBuf);
          }
          i = j + 1;
          break;
        }
        j++;
      }
      if (j >= buffer.length - 1) break;
    }
    i++;
  }
  return images;
}

export function cleanExtractedValue(str: string | undefined): string {
  if (!str) return '';
  const firstLine = str.split(/[\r\n]/)[0];
  return firstLine.replace(/[:=|\-_]+$/, '').replace(/^[:=|\-_]+/, '').trim();
}

/**
 * Trim OCR tail noise from a labelled value.
 *
 * OCR commonly glues the next label / a stray fragment onto the same line
 * (`DEVESH VISHWAKARMA gery “जा`, `Mr. Ashok Vishwakarma bs 2`). A stored value
 * must be the real value only — junk tails made the viewer and every answer
 * look broken even though the words themselves were correct.
 */
/**
 * Common OCR garbage fragments that follow a real value on the same line
 * (e.g. "B.Tech | CSE is Si", "Mr. Ashok Vishwakarma bs 2").
 */
export const OCR_NOISE_TAIL = /\s+(?:is|si|ad|he|re|bs|ee|ie)\b.*$/i;

export function trimOcrTail(str: string | undefined): string {
  if (!str) return '';
  let v = str.split(/[\r\n]/)[0].trim();
  // Cut where the next field label was glued on: "...  Blood Group:" / "... Signature"
  v = v.replace(
    /\s{2,}(?=[A-Z][A-Za-z .]{2,}?\s*[:=-])|\s+(?:signature|blood\s*group|dob\b|d\.o\.b|contact|address|session|course)\b.*$/i,
    ''
  );
  // Cut known OCR filler fragments before the trailing-token sweep below.
  v = v.replace(OCR_NOISE_TAIL, '');
  // Drop a trailing fragment that is not part of the value: lowercase stub or lone digits.
  const words = v.split(/\s+/).filter(Boolean);
  while (words.length > 1) {
    const last = words[words.length - 1];
    if (/^[a-z]{1,4}$/.test(last) || /^\d{1,2}$/.test(last) || /^[^\x20-\x7E]+$/.test(last)) {
      words.pop();
    } else break;
  }
  // Keep a trailing period: it is often part of a real abbreviation ("Distt.").
  return words.join(' ').replace(/[\s:|=,\-_;/]+$/, '').trim();
}

/**
 * Conservative OCR text cleanup, applied once before storing/clause matching.
 *
 * Only removes characters that carry no meaning (NULs, zero-width marks, soft
 * hyphens, stray control bytes) and normalises line endings/blank runs. Words,
 * digits and page breaks are preserved verbatim, because every answer quotes
 * this text directly — over-cleaning would break the verbatim guarantee.
 */
export function normalizeOcrText(text: string): string {
  if (!text) return '';
  return text
    .replace(/\r\n?/g, '\n')
    .replace(/\u00a0/g, ' ')
    .replace(/[\u200b-\u200d\ufeff]/g, '')
    .replace(/\u00ad/g, '')
    .replace(/[\u0000-\u0008\u000b\u000e-\u001f]/g, '')
    .replace(/[ \t]+$/gm, '')
    .replace(/\n{4,}/g, '\n\n\n')
    .trim();
}

/**
 * Person-name cleanup: keeps the real name and drops OCR noise.
 * Devanagari glyphs from the Hindi model pass are never part of a latin name.
 */
export function cleanPersonName(str: string | undefined): string {
  if (!str) return '';
  let v = trimOcrTail(str);
  // Anything from the first non-ASCII glyph onwards is OCR noise, not the name.
  v = v.replace(/[^\x20-\x7E][\s\S]*$/, '').trim();
  const words = v.split(/\s+/).filter(Boolean);
  while (words.length > 1) {
    const last = words[words.length - 1];
    if (/^[a-z]{1,4}$/.test(last) || /^\d{1,2}$/.test(last)) {
      words.pop();
    } else break;
  }
  return words.join(' ').replace(/[\s:|=,\-_.;/]+$/, '').trim();
}

/* ------------------------------------------------------------------ */
/* Deterministic entity extractor                                      */
/* ------------------------------------------------------------------ */

/* ------------------------------------------------------------------ */
/* Address (may continue on the next line)                             */
/* ------------------------------------------------------------------ */

/** A first line ending like this clearly continues on the next line. */
const ADDRESS_CONTINUES =
  /(?:\b(?:distt?|district|po|p\.o|post|tehsil|teh|taluka|ps|p\.s|thana|near|opp|via)\b\.?|,)\s*$/i;
const NOT_PLACE_LINE =
  /\b(signature|authority|student|roll|name|blood|contact|police|session|course|dob|issued|valid)\b|[:=]/i;

/**
 * Continuation text from the line after the address label, e.g.
 * "p> | Gorakhpur" -> "Gorakhpur". Returns null when the line is not a
 * plausible place name (a field label, signature caption, OCR debris...).
 */
export function addressContinuation(line: string | undefined): string | null {
  if (!line || NOT_PLACE_LINE.test(line)) return null;
  // A vertical rule/border often shows up as "|": the place name is what follows it.
  const segment = line.includes('|') ? line.split('|').filter(p => p.trim()).pop() || '' : line;
  const tokens = segment.split(/\s+/).filter(t => /[\p{L}\p{N}]/u.test(t));
  // Drop leading debris such as "oe", "p>", "ad".
  while (tokens.length > 0 && /^[a-z]{1,2}\W*$/.test(tokens[0])) tokens.shift();
  if (tokens.length === 0 || tokens.length > 4) return null;
  if (!tokens.every(t => /^[\p{L}\p{M}\p{N}.,\-/]+$/u.test(t))) return null;
  if (!/^[\p{Lu}\p{N}\u0900-\u097F]/u.test(tokens[0])) return null;
  return tokens.join(' ').replace(/[\s,.\-]+$/, '');
}

export function extractAddress(rawText: string): { value: string; evidence: string } | null {
  const lines = rawText.split(/\r?\n/);
  const labelRe =
    /(?:home\s*address|residential\s*address|permanent\s*address|address|पता)\s*[:=\-]?\s*([^\r\n]{5,120})/i;
  for (let i = 0; i < lines.length; i++) {
    const m = lines[i].match(labelRe);
    if (!m) continue;
    let value = trimOcrTail(m[1]);
    if (value.length < 5 || !/[A-Za-z]{3}/.test(value)) continue;
    // A bare "Address" label next to an organisation (e.g. a ticket's railway office)
    // is not the person's address.
    if (!/(?:home|residential|permanent|communication|correspondence|present|current)/i.test(m[0]) && ORG_WORDS.test(value)) continue;
    let evidence = m[0].trim();
    // "...Kauriram Distt." + next line "Gorakhpur" is one address.
    // (trimOcrTail strips a trailing comma, so test the raw text as well)
    if (ADDRESS_CONTINUES.test(value) || ADDRESS_CONTINUES.test(m[1].trim())) {
      const extra = addressContinuation(lines[i + 1]);
      if (extra) {
        value = `${value.replace(/,\s*$/, '')} ${extra}`.replace(/\s+/g, ' ').trim();
        evidence = `${evidence}\n${lines[i + 1].trim()}`;
      }
    }
    return { value, evidence };
  }
  return null;
}

/* ------------------------------------------------------------------ */
/* Structured documents: header facts replace guessy regex fields       */
/* ------------------------------------------------------------------ */

/**
 * For result sheets (repeating sessions/semesters) the regex extractor used to
 * store wrong values (the college line as "Candidate Name", the label "Code" as
 * the course, the first SGPA as "CGPA"). Those fields are replaced by the
 * header facts read from the document's own "Label : Value" cells. Values that
 * repeat per section (SGPA, result status) are NOT flattened here: they are
 * answered with their section by the structured answer engine.
 */
const STRUCTURED_REPLACED = new Set([
  'candidate_name', 'father_name', 'roll_number', 'enrollment_id', 'school_name', 'course_branch', 'cgpa', 'result_status'
]);

const stripCode = (v: string) => v.replace(/^\(\s*\d+\s*\)\s*/, '').trim();

export function applyStructuredHeader<T extends { field_name: string }>(fields: T[], rawText: string): T[] {
  const structure = parseStructure(rawText);
  if (!hasStructure(structure)) return fields;
  const header = headerFacts(structure);
  const get = (key: string) => header.find(f => normKey(f.key) === normKey(key) && f.value);

  const kept = fields.filter(f => !STRUCTURED_REPLACED.has(f.field_name));
  const add = (name: string, label: string, value: string | undefined, evidence: string | undefined, type = 'text') => {
    if (!value || !evidence) return;
    kept.push({
      field_name: name,
      display_label: label,
      field_value: value,
      evidence_text: evidence,
      page_number: 1,
      confidence: 0.98,
      value_type: type
    } as unknown as T);
  };

  const name = get('Name');
  const father = get("Father's Name");
  const roll = get('RollNo');
  const enrollment = get('EnrollmentNo');
  const institute = get('Institute Code & Name');
  const course = get('Course Code & Name');
  const branch = get('Branch Code & Name');
  const gender = get('Gender');

  add('candidate_name', 'Candidate Name', name?.value, name?.line);
  add('father_name', "Father's Name", father?.value, father?.line);
  add('roll_number', 'Roll Number', roll?.value, roll?.line, 'id');
  add('enrollment_id', 'Enrollment Number', enrollment?.value, enrollment?.line, 'id');
  add('school_name', 'School / Institution', institute && stripCode(institute.value), institute?.line);
  add(
    'course_branch',
    'Course / Branch',
    course || branch ? [course && stripCode(course.value), branch && stripCode(branch.value)].filter(Boolean).join(' · ') : undefined,
    (course || branch)?.line
  );
  add('gender', 'Gender', gender?.value, gender?.line);
  return kept;
}

/**
 * Real Entity Extractor:
 * ONLY extracts values that ACTUALLY exist in the OCR transcribed text.
 * NEVER creates fake dummy numbers, dummy parents, dummy schools, or fabricated marks.
 */
export function extractRealEntities(
  rawText: string,
  fileName: string,
  fileType: string
): RawEntityAnalysis {
  const lowerName = fileName.toLowerCase();
  const lowerText = rawText.toLowerCase();
  const combined = `${lowerName} ${lowerText}`;

  // 1. Detect category accurately.
  // Marksheet detection must NOT win on a bare "Roll No": student ID cards carry
  // roll numbers too, and that single token used to label an ID card as a
  // marksheet — which then ran marksheet-only field patterns on it.
  const marksheetSignals =
    /marksheet|grade\s*card|transcript|semester|10th|12th|high\s*school|intermediate|secondary\s*school|board\s*of|cbse|icse|bseb|up\s*board|\bdegree\b|university/i;
  const identitySignals =
    /student\s*id|\bid\s*card\b|blood\s*group|aadhaar|uidai|pan\s*card|passport|election\s*commission|voter|date\s*of\s*birth/i;

  // A reservation slip / boarding pass lists acceptable ID proofs ("Aadhaar /
  // PAN / Passport") and a traveller's details, so travel-signal detection must
  // run BEFORE identity detection or a train ticket is filed as an ID card.
  const travelSignals =
    /irctc|electronic reservation slip|boarding\s*pass|\bpnr\b|train\s*no\.|coach\s*no\.|departure|arrival|flight/i;
  // Exam/application forms mention "10th", "12th" and graduation details in
  // their qualification sections, which otherwise files them as marksheets.
  const applicationSignals =
    /application\s*form|graduate\s*aptitude\s*test|\bgate\s*20\d\d\b|\bform\s*no\.|declaration\s*by\s*(?:the\s*)?(?:candidate|applicant)/i;

  let category = 'General Document';
  if (travelSignals.test(combined)) {
    category = 'Travel Ticket';
  } else if (applicationSignals.test(combined)) {
    category = 'Application Form';
  } else if (identitySignals.test(combined) && !marksheetSignals.test(combined)) {
    category = 'Identity Document';
  } else if (marksheetSignals.test(combined)) {
    category = 'Marksheet / Degree';
  } else if (
    // A receipt/invoice is a payment record first. Strong signals (receipt no,
    // cash memo, GSTIN, "amount paid") win over a coincidental "bank" word so a
    // shop receipt is not filed as a bank statement.
    /tax\s*invoice|invoice\s*(?:no|number|date)|receipt\s*(?:no|number|date)|cash\s*memo|payment\s*receipt|amount\s*paid|gstin/i.test(
      combined
    ) &&
    !/electricity|kwh|tariff|consumer\s*(?:no|number)|sum\s*insured/i.test(combined)
  ) {
    category = 'Receipt / Invoice';
  } else if (
    combined.includes('electricity') ||
    combined.includes('bill') ||
    combined.includes('power') ||
    combined.includes('kwh') ||
    combined.includes('tariff') ||
    combined.includes('water board')
  ) {
    category = 'Utility Bill';
  } else if (
    combined.includes('aadhaar') ||
    combined.includes('uidai') ||
    combined.includes('pan card') ||
    combined.includes('income tax department') ||
    combined.includes('passport') ||
    combined.includes('election commission') ||
    combined.includes('voter')
  ) {
    category = 'Identity Document';
  } else if (
    combined.includes('passbook') ||
    combined.includes('bank') ||
    combined.includes('ifsc') ||
    combined.includes('statement') ||
    combined.includes('account number')
  ) {
    category = 'Bank Statement / Passbook';
  } else if (
    combined.includes('policy') ||
    combined.includes('insurance') ||
    combined.includes('sum insured') ||
    combined.includes('premium')
  ) {
    category = 'Insurance Policy';
  } else if (combined.includes('invoice') || combined.includes('receipt') || combined.includes('tax invoice')) {
    category = 'Receipt / Invoice';
  }

  const fields: RawEntityAnalysis['fields'] = [];
  const addField = (
    field_name: string,
    display_label: string,
    field_value: string,
    evidence_text: string,
    confidence = 0.98,
    value_type = 'text'
  ) => {
    if (!field_value || field_value.trim().length === 0) return;
    if (JUNK_VALUE.test(field_value.trim())) return; // "NA", "null", "--" are not answers
    fields.push({
      field_name,
      display_label,
      field_value: field_value.trim(),
      evidence_text: evidence_text.trim(),
      page_number: 1,
      confidence,
      value_type,
    });
  };

  // --- 1. Candidate / Student Name ---
  const nameMatch = rawText.match(
    /(?:candidate(?:\s*['’]s)?\s*name|student(?:\s*['’]s)?\s*name|name\s*of\s*(?:the\s*)?(?:candidate|student|applicant|holder))\s*[:=-]?\s*([^\n\r]{2,50})/i
  ) || rawText.match(
    /(?:this\s+is\s+to\s+certify\s+that|certified\s+that)\s+([A-Za-z\s.]{3,40})/i
  ) || rawText.match(
    /\bname\s*[:=-]\s*([^\n\r]{2,50})/i
  );

  if (nameMatch) {
    const val = cleanPersonName(nameMatch[1]);
    if (
      val.length > 2 &&
      !/^(of|the|is|to|examination|board|secondary|education|central|government)$/i.test(val)
    ) {
      // OCR noise was stripped from this value → report lower confidence honestly.
      const confidence = val === cleanExtractedValue(nameMatch[1]) ? 0.99 : 0.94;
      addField('candidate_name', 'Candidate Name', val, nameMatch[0].trim(), confidence, 'name');
    }
  }

  // Forms that print the VALUE first and the label under it in brackets:
  //   DEVESH VISHWAKARMA
  //   (Full Name of the Applicant)
  if (!fields.some(f => f.field_name === 'candidate_name')) {
    const lines = rawText.split(/\r?\n/).map(l => l.trim()).filter(Boolean);
    const bracketLabel =
      /^\(\s*(?:full\s+name\s+of\s+the\s+(?:applicant|candidate|student)|(?:applicant|candidate|student)['’]?s?\s+(?:full\s+)?name|name\s+of\s+the\s+(?:applicant|candidate|student))\s*\)$/i;
    for (let i = 1; i < lines.length; i++) {
      if (bracketLabel.test(lines[i]) && looksLikePersonName(lines[i - 1])) {
        addField('candidate_name', 'Candidate Name', lines[i - 1], `${lines[i - 1]}\n${lines[i]}`, 0.95, 'name');
        break;
      }
    }
  }

  // --- 2. Father's Name ---
  const fatherMatch = rawText.match(
    /(?:father(?:\s*['’]s)?\s*name|father\s*name|s\/o|son\s*of)\s*[:=-]?\s*([^\n\r]{2,50})/i
  );
  if (fatherMatch) {
    const val = cleanPersonName(fatherMatch[1]);
    if (val.length > 2) {
      const confidence = val === cleanExtractedValue(fatherMatch[1]) ? 0.98 : 0.93;
      addField('father_name', "Father's Name", val, fatherMatch[0].trim(), confidence, 'name');
    }
  }

  // --- 3. Mother's Name ---
  const motherMatch = rawText.match(
    /(?:mother(?:\s*['’]s)?\s*name|mother\s*name|d\/o|daughter\s*of)\s*[:=-]?\s*([^\n\r]{2,50})/i
  );
  if (motherMatch) {
    const val = cleanPersonName(motherMatch[1]);
    if (val.length > 2) {
      const confidence = val === cleanExtractedValue(motherMatch[1]) ? 0.98 : 0.93;
      addField('mother_name', "Mother's Name", val, motherMatch[0].trim(), confidence, 'name');
    }
  }

  // --- 4. Roll Number / Hall Ticket / Enrollment ---
  // Word boundaries matter: without them "reg" matched inside "regretted" and
  // stored "retted" as a Roll Number — fabricated data in a grounded app.
  const rollMatch = rawText.match(
    /\b(?:roll\s*(?:no\.?|number|code)|enrol(?:l)?ment\s*(?:no\.?|number)?|reg(?:istration)?\s*(?:no\.?|number)|hall\s*ticket\s*(?:no\.?|number)?|seat\s*(?:no\.?|number)?)\s*[:=\-]?\s*([A-Z0-9_\/-]{4,25})/i
  );
  if (rollMatch) {
    const val = cleanExtractedValue(rollMatch[1]);
    if (val.length >= 4) {
      addField('roll_number', 'Roll Number', val, rollMatch[0].trim(), 0.99, 'identifier');
    }
  }

  // --- 5. School / College / Institution ---
  const schoolMatch = rawText.match(
    /(?:school(?:\s*name)?|college(?:\s*name)?|institution|institute|center|centre)\s*[:=-]\s*([^\n\r]{3,65})/i
  );
  if (schoolMatch) {
    const val = cleanExtractedValue(schoolMatch[1]);
    if (val.length >= 3) {
      addField('school_name', 'School / Institution', val, schoolMatch[0].trim(), 0.97, 'text');
    }
  } else {
    // Unlabelled letterhead: the institution name is usually the first line that
    // names a college/university/school (student ID cards, certificates).
    const letterhead = rawText
      .split(/\r?\n/)
      .map(l => l.trim())
      .find(
        l =>
          l.length >= 6 &&
          l.length <= 95 &&
          // "collage" is a real, common OCR misspelling of "college".
          /\b(college|collage|university|institute|school|academy)\b/i.test(l) &&
          !/\b(address|signature|student|course|roll)\b/i.test(l)
      );
    if (letterhead) {
      addField('school_name', 'School / Institution', letterhead.replace(/[|]+/g, ' ').trim(), letterhead, 0.96, 'text');
    }
  }

  // --- 6. Board / University ---
  if (combined.includes('central board of secondary education') || /\bcbse\b/i.test(rawText)) {
    addField('board_name', 'Examination Board', 'Central Board of Secondary Education (CBSE)', 'CBSE', 0.99, 'text');
  } else if (combined.includes('council for the indian school certificate') || /\bicse\b/i.test(rawText) || /\bcisce\b/i.test(rawText)) {
    addField('board_name', 'Examination Board', 'Council for the Indian School Certificate Examinations (ICSE)', 'ICSE', 0.99, 'text');
  } else if (combined.includes('up board') || combined.includes('madhyamik shiksha parishad')) {
    addField('board_name', 'Examination Board', 'Board of High School and Intermediate Education Uttar Pradesh', 'UP Board', 0.99, 'text');
  } else if (combined.includes('bihar school examination board') || /\bbseb\b/i.test(rawText)) {
    addField('board_name', 'Examination Board', 'Bihar School Examination Board (BSEB)', 'BSEB', 0.99, 'text');
  } else {
    const boardMatch = rawText.match(/(?:board|university|council)\s*[:=-]\s*([^\n\r]{4,60})/i);
    if (boardMatch) {
      const val = cleanExtractedValue(boardMatch[1]);
      addField('board_name', 'Board / University', val, boardMatch[0].trim(), 0.96, 'text');
    }
  }

  // --- 7. Date of Birth (tolerates OCR mangling of the "DOB" label) ---
  const DATE_RE =
    /([0-9]{1,2}[\/\-\.][0-9]{1,2}[\/\-\.][0-9]{2,4}|[0-9]{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]+\s+[0-9]{4})/;
  const dobLabelMatch = rawText.match(
    /(?:date\s*of\s*birth|d\.?\s*o\.?\s*b\.?|जन्म\s*(?:तिथि|दिनांक)?)\s*\)?\s*[:=\-]?\s*([0-9]{1,2}[\/\-\.][0-9]{1,2}[\/\-\.][0-9]{2,4}|[0-9]{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]+\s+[0-9]{4})/i
  );

  if (dobLabelMatch) {
    addField('date_of_birth', 'Date of Birth', dobLabelMatch[1].trim(), dobLabelMatch[0].trim(), 0.99, 'date');
  } else if (
    // Layouts that print the label in brackets with the value on the NEXT line
    // ("(D.O.B)\n06/06/2006") — search a short window after the label only.
    (() => {
      const near = rawText.match(
        /(?:date\s*of\s*birth|d\.?\s*o\.?\s*b\.?)\s*\)?\s*[:=\-]?\s*([\s\S]{0,160})/i
      );
      const dateInWindow = near?.[1]?.match(DATE_RE)?.[0];
      if (!near || !dateInWindow) return false;
      const evidence = `${near[0].split(/\r?\n/).filter(Boolean).slice(0, 2).join(' ')}`.trim();
      addField('date_of_birth', 'Date of Birth', dateInWindow, evidence, 0.9, 'date');
      return true;
    })()
  ) {
    // added above
  } else {
    // OCR often reads "DOB" as "008"/"OO8". Find the label TOKEN on a line and
    // take the first date on that same line — never an arbitrary date elsewhere.
    const dobLine = rawText
      .split(/\r?\n/)
      .find(
        line =>
          /(?:^|[^a-z])(?:008|oo8|dob|d\.o\.b|date\s*of\s*birth)\b/i.test(line) && DATE_RE.test(line)
      );
    const dobValue = dobLine?.match(DATE_RE)?.[1];
    if (dobLine && dobValue) {
      addField('date_of_birth', 'Date of Birth', dobValue.trim(), dobLine.trim(), 0.92, 'date');
    }
  }

  // --- 7b. Blood group (ID cards / medical records) ---
  const bloodMatch = rawText.match(
    /(?:blood\s*(?:group|grp|type)|रक्त\s*समूह)\s*[:=\-]?\s*((?:AB|A|B|O)\s*(?:\+|\-|\bpos(?:itive)?\b|\bneg(?:ative)?\b))/i
  );
  if (bloodMatch) {
    const val = bloodMatch[1]
      .replace(/\s*(?:pos(?:itive)?|positive)\b/i, '+')
      .replace(/\s*(?:neg(?:ative)?|negative)\b/i, '-')
      .replace(/\s+/g, '')
      .replace(/[+\-]$/, m => m)
      .toUpperCase();
    addField('blood_group', 'Blood Group', val, bloodMatch[0].trim(), 0.96, 'text');
  }

  // --- 7c. Contact / mobile number ---
  const contactMatch = rawText.match(
    /(?:contact\s*(?:no\.?|number)?|mobile(?:\s*(?:no\.?|number))?|phone(?:\s*(?:no\.?|number))?|मोबाइल)\s*\)?\s*[:=\-]?\s*(?:\+?91[\s-]?)?([6-9][0-9]{9})\b/i
  );
  if (contactMatch) {
    addField('contact_number', 'Contact Number', contactMatch[1], contactMatch[0].trim(), 0.97, 'identifier');
  }

  // --- 7d. Student ID / enrollment ID (e.g. KIPM/761/CSE/067) ---
  const studentIdMatch = rawText.match(
    /(?:student\s*id|enrol{1,2}ment\s*id|student\s*code)\s*\)?\s*[:=\-]?\s*([A-Z0-9]+(?:[\/|-][A-Z0-9]+){1,6})/i
  );
  if (!studentIdMatch) {
    // Bracketed label form: "(Enrollment ID)  M103B71"
    const idFromBracket = rawText.match(/enrol{1,2}ment\s*id\s*\)\s*([A-Z0-9]{4,20})/i);
    if (idFromBracket) {
      addField('enrollment_id', 'Enrollment ID', idFromBracket[1], idFromBracket[0].trim(), 0.94, 'identifier');
    }
  }
  if (studentIdMatch) {
    addField('student_id', 'Student ID', studentIdMatch[1].trim(), studentIdMatch[0].trim(), 0.98, 'identifier');
  }

  // --- 7e. Course / branch ---
  const courseMatch = rawText.match(
    /(?:course\s*(?:\/\s*branch)?|branch|program(?:me)?)\s*[:=\-]?\s*([A-Za-z][A-Za-z. \t]{1,24}(?:[\/|&][ \t]*[A-Za-z][A-Za-z. \t]{1,16})?)/i
  );
  if (courseMatch) {
    const val = trimOcrTail(courseMatch[1].replace(/\s+/g, ' '));
    if (val.length >= 2) {
      addField('course_branch', 'Course / Branch', val, courseMatch[0].trim(), 0.95, 'text');
    }
  }

  // --- 7f. Residential address (first clean line only — never OCR gibberish tails) ---
  const address = extractAddress(rawText);
  if (address) {
    addField('residential_address', 'Residential Address', address.value, address.evidence, 0.93, 'text');
  }

  // --- 8. Year of Passing / Session ---
  const sessionMatch = rawText.match(
    /(?:session|valid(?:ity)?(?:\s*(?:up\s*to|till))?)\s*[:=\-]?\s*([0-9]{4}\s*[-–\/]\s*[0-9]{4})/i
  );
  if (sessionMatch) {
    addField(
      'session',
      'Session / Validity',
      sessionMatch[1].replace(/\s+/g, ''),
      sessionMatch[0].trim(),
      0.98,
      'date'
    );
  }

  // A 4-digit year followed by a dash is a range, not a passing year.
  const yearMatch = sessionMatch
    ? null
    : rawText.match(
        /(?:passing\s*year|year\s*of\s*(?:passing|exam|examination)|dated?|exam\s*date)\s*[:=\-]?\s*([0-9]{4})(?![0-9\-–\/]|[A-Za-z]+\s*[-–])/i
      ) || rawText.match(/\b(?:examination|annual)\s+([0-9]{4})\b/i);

  if (yearMatch) {
    const val = cleanExtractedValue(yearMatch[1]);
    addField('year_of_passing', 'Passing Year', val, yearMatch[0].trim(), 0.98, 'date');
  }

  // --- 9. Total Marks / Score ---
  const marksMatch = rawText.match(
    /(?:total\s*marks|grand\s*total|marks\s*obtained|total)\s*[:=-]?\s*([0-9]{2,4}\s*(?:\/|\s+out\s+of\s+)\s*[0-9]{2,4}|[0-9]{2,4})/i
  );
  if (marksMatch) {
    const val = cleanExtractedValue(marksMatch[1]);
    addField('total_marks', 'Total Marks', val, marksMatch[0].trim(), 0.98, 'amount');
  }

  // --- 10. Percentage / CGPA ---
  // Percentage needs a real context: a label, or a line that is about marks / results.
  // A bare "57%" anywhere in the text ("IR recovers only 57% of cost") is not a
  // percentage field and must never be stored as one.
  const pctMatch =
    rawText.match(/(?:percentage|percent|pratishat)\s*[:=\-]?\s*([0-9]{2}(?:\.[0-9]{1,2})?%?)/i) ||
    rawText
      .split(/\r?\n/)
      .filter(l => /\b(marks|result|score|aggregate|cgpa|division)\b/i.test(l))
      .map(l => l.match(/\b([0-9]{2}(?:\.[0-9]{1,2})?)\s*%/))
      .find(Boolean);
  if (pctMatch) {
    const val = cleanExtractedValue(pctMatch[1]);
    addField('percentage', 'Overall Percentage', val.includes('%') ? val : `${val}%`, pctMatch[0].trim(), 0.99, 'identifier');
  }

  const cgpaMatch = rawText.match(/(?:cumulative\s*cgpa|cgpa|sgpa|gpa)\s*[:=-]?\s*([0-9]\.[0-9]{1,2})/i);
  if (cgpaMatch) {
    addField('cgpa', 'CGPA / Grade Point', cgpaMatch[1].trim(), cgpaMatch[0].trim(), 0.98, 'identifier');
  }

  // --- 11. Result / Division ---
  const resultMatch = rawText.match(
    /(?:result|division|final\s*result|status)\s*[:=-]?\s*(PASS(?:ED)?(?:\s+WITH\s+[A-Za-z]+)?|FIRST(?:\s*DIV(?:ISION)?)?|SECOND(?:\s*DIV(?:ISION)?)?|THIRD(?:\s*DIV(?:ISION)?)?|DISTINCTION|QUALIFIED|COMPARTMENT|FAIL(?:ED)?)/i
  );
  if (resultMatch) {
    addField('result_status', 'Result / Division', cleanExtractedValue(resultMatch[1]), resultMatch[0].trim(), 0.99, 'text');
  }

  // --- 12. Aadhaar / PAN / Identity numbers ---
  const aadhaarMatch = rawText.match(/\b([2-9]{1}[0-9]{3}\s[0-9]{4}\s[0-9]{4})\b/);
  if (aadhaarMatch) {
    addField('aadhaar_number', 'Aadhaar Card Number', aadhaarMatch[1], aadhaarMatch[0], 0.99, 'identifier');
  }

  const panMatch = rawText.match(/\b([A-Z]{5}[0-9]{4}[A-Z])\b/);
  if (panMatch) {
    addField('pan_number', 'PAN Card Number', panMatch[1], panMatch[0], 0.99, 'identifier');
  }

  // --- 13. Utility Bill fields ---
  const caMatch = rawText.match(/(?:consumer\s*(?:no\.?|number|id)|ca\s*no\.?|k\s*no\.?|account\s*id)\s*[:=-]?\s*([0-9A-Z]{6,16})/i);
  if (caMatch) {
    addField('consumer_number', 'Consumer Number (CA)', caMatch[1].trim(), caMatch[0].trim(), 0.99, 'identifier');
  }

  const amountMatch = rawText.match(
    // "Sub Total" is not the payable amount — a bare `total` must not match it.
    /(?:total\s*amount(?:\s*due)?|grand\s*total|bill\s*amount|amount\s*paid|amount\s*payable|net\s*payable|(?<!sub\s)total)\s*[:=-]?\s*(₹\s*[0-9,]+(?:\.[0-9]{2})?|[0-9,]+(?:\.[0-9]{2})?\s*(?:inr|rs|rupees)?)/i
  );
  if (amountMatch) {
    addField('total_amount', 'Amount Due', amountMatch[1].trim(), amountMatch[0].trim(), 0.98, 'amount');
  }

  const dueMatch = rawText.match(/(?:due\s*date|payment\s*due\s*date)\s*[:=-]?\s*([0-9]{1,2}[\/\-\.][0-9]{1,2}[\/\-\.][0-9]{2,4}|[0-9]{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]+\s+[0-9]{4})/i);
  if (dueMatch) {
    addField('due_date', 'Payment Due Date', dueMatch[1].trim(), dueMatch[0].trim(), 0.98, 'date');
  }

  // --- 13a. Receipt / invoice fields ---
  // A receipt is the document people most often ask "kitna, kab, kis ko" about,
  // but it carries none of the ID/marksheet fields. These are read verbatim
  // from labelled lines only — no invented merchant names or totals.
  const receiptNoMatch = rawText.match(
    /(?:receipt|invoice|bill|token|cash\s*memo)\s*(?:no\.?|number|#)\s*[:=-]?\s*([A-Z0-9][A-Z0-9\/\-]{2,20})/i
  );
  if (receiptNoMatch) {
    addField('receipt_number', 'Receipt / Invoice No.', receiptNoMatch[1].trim(), receiptNoMatch[0].trim(), 0.97, 'identifier');
  }

  const receiptDateMatch = rawText.match(
    /(?:receipt\s*date|invoice\s*date|bill\s*date|dated|date)\s*[:=-]?\s*([0-9]{1,2}[\/\-.][0-9]{1,2}[\/\-.][0-9]{2,4}|[0-9]{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]+\s+[0-9]{4})/i
  );
  if (receiptDateMatch) {
    addField('receipt_date', 'Receipt Date', receiptDateMatch[1].trim(), receiptDateMatch[0].trim(), 0.95, 'date');
  }

  const taxMatch = rawText.match(
    // "CGST 9%: 108.00" — the RATE sits before the separator, the amount after it.
    /(?:gst|cgst|sgst|igst|vat|tax)\s*(?:(?:@\s*)?[0-9]{1,2}(?:\.[0-9]{1,2})?\s*%?\s*(?=[:=-]))?[:=-]?\s*(?:₹\s*)?([0-9][0-9,]*(?:\.[0-9]{1,2})?)(?![0-9])/i
  );
  if (taxMatch) {
    const val = cleanExtractedValue(taxMatch[1]);
    // A bare "GSTIN 09ABCDE1234F1Z5" must not become a tax amount.
    if (/[0-9]/.test(val) && !/^[0-9]{2}[A-Z]{5}/i.test(val)) {
      addField('tax_amount', 'Tax / GST Amount', val, taxMatch[0].trim(), 0.93, 'amount');
    }
  }

  const paidByMatch = rawText.match(
    /(?:paid\s*(?:by|via|through)|payment\s*(?:mode|method)|mode\s*of\s*payment)\s*[:=-]?\s*(upi|gpay|google\s*pay|phonepe|paytm|net\s*banking|debit\s*card|credit\s*card|card|cash|cheque|neft|imps)\b/i
  );
  if (paidByMatch) {
    addField(
      'payment_method',
      'Payment Method',
      paidByMatch[1].toUpperCase(),
      paidByMatch[0].trim(),
      0.95,
      'text'
    );
  }

  // --- 14. Bank Account & IFSC ---
  const acctMatch = rawText.match(/(?:account\s*(?:no\.?|number)|a\/c\s*no\.?)\s*[:=-]?\s*([0-9]{9,18})/i);
  if (acctMatch) {
    addField('account_number', 'Bank Account Number', acctMatch[1].trim(), acctMatch[0].trim(), 0.99, 'identifier');
  }

  const ifscMatch = rawText.match(/\b([A-Z]{4}0[A-Z0-9]{6})\b/);
  if (ifscMatch) {
    addField('ifsc_code', 'IFSC Code', ifscMatch[1].trim(), ifscMatch[0].trim(), 0.99, 'code');
  }

  return { category, page_count: 1, fields: applyStructuredHeader(fields, rawText) };
}
