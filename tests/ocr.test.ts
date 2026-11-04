/**
 * Phase 1 OCR wiring tests — offline tessdata selection, multi-page scanned
 * PDF image selection, and page transcript joining. Pure logic, no worker
 * spawn (the worker itself loads real traineddata at runtime).
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {
  pickOcrLanguages,
  selectOcrImages,
  joinOcrPageTexts,
  MAX_OCR_PAGES,
  MIN_OCR_IMAGE_BYTES
} from '../server/extract';

function tempTessdata(files: string[]): string {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'safedoc-tess-'));
  for (const f of files) fs.writeFileSync(path.join(dir, f), 'fake-traineddata');
  return dir;
}

test('pickOcrLanguages: eng+hin both present → offline eng+hin from local dir', () => {
  const dir = tempTessdata(['eng.traineddata', 'hin.traineddata']);
  try {
    const r = pickOcrLanguages(dir);
    assert.equal(r.langs, 'eng+hin');
    assert.equal(r.options?.langPath, dir);
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test('pickOcrLanguages: only eng → eng, still served from local dir (no CDN)', () => {
  const dir = tempTessdata(['eng.traineddata']);
  try {
    const r = pickOcrLanguages(dir);
    assert.equal(r.langs, 'eng');
    assert.equal(r.options?.langPath, dir);
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test('pickOcrLanguages: empty dir → honest fallback to tesseract.js defaults', () => {
  const dir = tempTessdata([]);
  try {
    const r = pickOcrLanguages(dir);
    assert.equal(r.langs, 'eng');
    assert.equal(r.options, undefined);
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test('this repo ships offline eng + hin traineddata in data/tessdata', () => {
  const dir = path.resolve('data', 'tessdata');
  assert.ok(fs.existsSync(path.join(dir, 'eng.traineddata')), 'eng.traineddata missing');
  assert.ok(fs.existsSync(path.join(dir, 'hin.traineddata')), 'hin.traineddata missing');
  const r = pickOcrLanguages(dir);
  assert.equal(r.langs, 'eng+hin');
  assert.equal(r.options?.langPath, dir);
});

test('selectOcrImages: keeps stream order, drops tiny images, caps page count', () => {
  const tiny = Buffer.alloc(MIN_OCR_IMAGE_BYTES - 1, 1);
  const pages = Array.from({ length: MAX_OCR_PAGES + 4 }, (_, i) =>
    Buffer.concat([Buffer.from(`PAGE_${i}_`), Buffer.alloc(20_000, i)])
  );
  const input = [tiny, ...pages];
  const out = selectOcrImages(input);

  assert.equal(out.length, MAX_OCR_PAGES, 'must cap at MAX_OCR_PAGES');
  // Order preserved (page order), tiny decorative image skipped
  assert.equal(out[0].subarray(0, 7).toString(), 'PAGE_0_');
  assert.equal(out[1].subarray(0, 7).toString(), 'PAGE_1_');
  assert.ok(out.every(b => b.length >= MIN_OCR_IMAGE_BYTES));
});

test('joinOcrPageTexts: page order kept, form-feed separators, empties dropped', () => {
  assert.equal(
    joinOcrPageTexts(['Page one text', '', '  Page two  ', 'Page three']),
    'Page one text\fPage two\fPage three'
  );
  assert.equal(joinOcrPageTexts([]), '');
  assert.equal(joinOcrPageTexts(['', '   ']), '');
});
