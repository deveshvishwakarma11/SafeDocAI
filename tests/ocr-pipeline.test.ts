import { test } from 'node:test';
import assert from 'node:assert/strict';
import sharp from 'sharp';
import {
  cleanLine,
  cleanLines,
  linesFromBlocks,
  passScore,
  runAdaptiveOcr,
  POOR_PASS_WORDS,
  type OcrLine,
  type OcrWord
} from '../server/ocr-pipeline';
import { addressContinuation, extractAddress, extractRealEntities } from '../server/extract';

/** Build an OCR line from [text, confidence] pairs. */
function line(...pairs: Array<[string, number]>): OcrLine {
  let x = 0;
  const words: OcrWord[] = pairs.map(([text, confidence]) => {
    const w = { text, confidence, bbox: { x0: x, y0: 0, x1: x + 40, y1: 28 } };
    x += 50;
    return w;
  });
  return { words };
}

/* ---------------- transcript cleanup ---------------- */

test('low-confidence junk words are dropped, real words are kept', () => {
  const l = line(['Name', 96], [':', 80], ['DEVESH', 95], ['VISHWAKARMA', 94], ['gery', 20], ['“जा', 30]);
  assert.equal(cleanLine(l), 'Name : DEVESH VISHWAKARMA');
});

test('random Devanagari is dropped unless the engine is confident', () => {
  assert.equal(cleanLine(line(['Signature', 90], ['की', 40], ['Authority', 88])), 'Signature Authority');
  assert.equal(cleanLine(line(['नाम', 85], [':', 80], ['देवेश', 82])), 'नाम : देवेश');
});

test('digit tokens survive moderate confidence (IDs are critical)', () => {
  assert.equal(cleanLine(line(['Contact', 90], ['No.', 90], [':9163791592', 42])), 'Contact No. :9163791592');
});

test('vertical "I D E N T I T Y" letters and icon specks produce no line', () => {
  assert.equal(cleanLine(line(['1', 17])), null);
  assert.equal(cleanLine(line(['N', 40], ['T', 35])), null);
  assert.equal(cleanLine(line(['&', 0], ['७', 34])), null);
});

test('leading border/logo symbol is stripped from a real line', () => {
  assert.equal(cleanLine(line(['\\', 89], ['KIPM', 95], ['COLLEGE', 94])), 'KIPM COLLEGE');
});

test('cleanLines gives one line per text line', () => {
  const out = cleanLines([
    line(['Roll', 95], ['No.', 95], [':', 90], ['2407510100067', 93]),
    line(['x', 5]),
    line(['Police', 90], ['Station', 90])
  ]);
  assert.equal(out, 'Roll No. : 2407510100067\nPolice Station');
});

test('linesFromBlocks flattens tesseract.js blocks', () => {
  const blocks = [
    { paragraphs: [{ lines: [{ words: [{ text: 'Hello', confidence: 90, bbox: { x0: 0, y0: 0, x1: 10, y1: 10 } }] }] }] }
  ];
  const lines = linesFromBlocks(blocks);
  assert.equal(lines.length, 1);
  assert.equal(lines[0].words[0].text, 'Hello');
  assert.deepEqual(linesFromBlocks(undefined), []);
});

/* ---------------- adaptive OCR / rescue pass ---------------- */

async function testImage(): Promise<Buffer> {
  return sharp({ create: { width: 900, height: 300, channels: 3, background: '#ffffff' } }).png().toBuffer();
}

function fakeWorker(results: OcrLine[][]) {
  let calls = 0;
  return {
    get calls() {
      return calls;
    },
    async recognize() {
      const lines = results[Math.min(calls, results.length - 1)];
      calls++;
      return { data: { text: '', blocks: [{ paragraphs: [{ lines }] }] } };
    }
  };
}

const goodLines: OcrLine[] = Array.from({ length: 6 }, (_, i) =>
  line([`Label${i}`, 95], ['value', 94], ['number', 93])
);

test('a good first pass is used as-is (single OCR call)', async () => {
  const worker = fakeWorker([goodLines]);
  const text = await runAdaptiveOcr(worker, await testImage());
  assert.equal(worker.calls, 1);
  assert.ok(text.includes('Label0 value number'));
});

test('a poor first pass triggers one rescue pass and the better result wins', async () => {
  const poor = [line(['Enjoy', 96], ['Ticket', 96])];
  assert.ok(passScore(poor) < POOR_PASS_WORDS);
  const worker = fakeWorker([poor, goodLines]);
  const text = await runAdaptiveOcr(worker, await testImage());
  assert.equal(worker.calls, 2);
  assert.ok(text.includes('Label5'));
});

test('rescue never makes the result worse', async () => {
  const mediocre = [line(['Enjoy', 96], ['Ticket', 96], ['Booking', 95])];
  const worse = [line(['x', 10])];
  const worker = fakeWorker([mediocre, worse]);
  const text = await runAdaptiveOcr(worker, await testImage());
  assert.equal(text, 'Enjoy Ticket Booking');
});

test('an undecodable image falls back to OCR on the original bytes', async () => {
  const worker = fakeWorker([goodLines]);
  const text = await runAdaptiveOcr(worker, Buffer.from('not an image'));
  assert.equal(worker.calls, 1);
  assert.ok(text.includes('Label0'));
});

/* ---------------- multi-line address ---------------- */

test('address continues on the next line after "Distt." (real ID card case)', () => {
  const text =
    'Police Station : Kauriram\nHome Address : Vill. Dhaska Po. Kauriram Distt.\nGorakhpur\nStudent Signature of Issuing Authority';
  const a = extractAddress(text);
  assert.equal(a?.value, 'Vill. Dhaska Po. Kauriram Distt. Gorakhpur');
  assert.ok(a?.evidence.includes('Gorakhpur'));
});

test('continuation ignores OCR debris before the place name', () => {
  assert.equal(addressContinuation('p> | Gorakhpur'), 'Gorakhpur');
  assert.equal(addressContinuation('oe | Gorakhpur'), 'Gorakhpur');
  assert.equal(addressContinuation('Lucknow 226001'), 'Lucknow 226001');
});

test('continuation never swallows a field label or caption', () => {
  assert.equal(addressContinuation('Signature of Student Signature of Issuing Authority'), null);
  assert.equal(addressContinuation('Blood Group : A+'), null);
  assert.equal(addressContinuation(undefined), null);
  assert.equal(addressContinuation('this line has far too many words to be a place name'), null);
});

test('a complete one-line address is left alone', () => {
  const a = extractAddress('Address : 12 MG Road Lucknow 226001\nSignature of Student');
  assert.equal(a?.value, '12 MG Road Lucknow 226001');
});

test('comma at the end of the first line also continues the address', () => {
  const a = extractAddress('Address: 12 MG Road,\nLucknow\nPhone: 9876543210');
  assert.equal(a?.value, '12 MG Road Lucknow');
});

/* ---------------- end to end on a clean transcript ---------------- */

const CARD = [
  'KIPM COLLEGE OF ENGINEERING & TECHNOLOGY',
  'PLOT No. BL 1 & 2, SECTOR-9, GIDA, GORAKHPUR-273 209',
  'Student ID: KIPM/761/CSE/067 Session: 2024-2028',
  'Name : DEVESH VISHWAKARMA',
  "Father's Name : Mr. Ashok Vishwakarma",
  'Course/Branch : B.Tech | CSE',
  'Roll No. : 2407510100067',
  'Blood Group : A+ DOB : 06-06-2006',
  'Contact No. : 9163791592',
  'Police Station : Kauriram',
  'Home Address : Vill. Dhaska Po. Kauriram Distt.',
  'Gorakhpur'
].join('\n');

test('clean ID-card transcript yields the full address and a clean course', () => {
  const fields = extractRealEntities(CARD, 'ID Card.pdf', 'pdf').fields;
  const get = (n: string) => fields.find(f => f.field_name === n)?.field_value;
  assert.equal(get('residential_address'), 'Vill. Dhaska Po. Kauriram Distt. Gorakhpur');
  assert.equal(get('course_branch'), 'B.Tech | CSE'); // must not glue the next line's "Roll No."
  assert.equal(get('contact_number'), '9163791592');
  assert.equal(get('date_of_birth'), '06-06-2006');
});
