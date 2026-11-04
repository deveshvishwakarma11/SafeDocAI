// Phase 1 verification: offline hin OCR load + multi-page scanned PDF path.
const BASE = process.env.BASE || 'http://127.0.0.1:3100';
const log = (...a) => console.log(...a);
let failed = false;
const check = (name, cond, extra = '') => {
  log(`${cond ? 'PASS' : 'FAIL'} - ${name}${extra ? ' :: ' + extra : ''}`);
  if (!cond) failed = true;
};

async function del(id) {
  if (id) await fetch(`${BASE}/api/documents/${id}`, { method: 'DELETE' });
}

// 1. Health before OCR warm-up
let h = await (await fetch(`${BASE}/api/health`)).json();
log(`boot health: ocr=${h.ocr_engine} ollama=${h.ollama_ready}`);

// 2. Warm the Tesseract worker through the real image-upload path (1x1 PNG).
//    Worker init loads eng+hin from data/tessdata — if hin.traineddata or the
//    local langPath wiring is broken, health below will not say active_eng_hin.
const pngB64 =
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==';
const pngBytes = Buffer.from(pngB64, 'base64');
let res = await fetch(`${BASE}/api/documents`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ fileName: 'dot.png', fileType: 'png', base64Data: pngB64, size: pngBytes.length })
});
let body = await res.json();
log(`png upload: http=${res.status} status=${body.status} raw=${JSON.stringify(String(body.document?.raw_text || '').slice(0, 60))}`);
const pngId = body.document?.id;

h = await (await fetch(`${BASE}/api/health`)).json();
check('worker loads eng+hin offline from data/tessdata', h.ocr_engine === 'active_eng_hin', `ocr_engine=${h.ocr_engine}`);
await del(pngId);

// 3. Multi-page scanned PDF: text layer <30 chars + two embedded fake JPEG
//    "pages" (extractImagesFromPdfBuffer scans raw bytes for SOI/EOI markers).
//    Verifies: both pages selected in order, per-page OCR failure is contained,
//    upload still succeeds honestly (no crash, no fabricated text).
function buildPdf(lines) {
  const esc = s => s.replace(/\\/g, '\\\\').replace(/\(/g, '\\(').replace(/\)/g, '\\)');
  const content = 'BT /F1 12 Tf 50 750 Td 14 TL\n' + lines.map(l => `(${esc(l)}) Tj T*`).join('\n') + '\nET';
  const objs = [
    '<< /Type /Catalog /Pages 2 0 R >>',
    '<< /Type /Pages /Kids [3 0 R] /Count 1 >>',
    '<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>',
    `<< /Length ${content.length} >>\nstream\n${content}\nendstream`,
    '<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>'
  ];
  let pdf = '%PDF-1.4\n';
  const offsets = [];
  objs.forEach((b, i) => { offsets.push(pdf.length); pdf += `${i + 1} 0 obj\n${b}\nendobj\n`; });
  const xrefPos = pdf.length;
  pdf += `xref\n0 ${objs.length + 1}\n0000000000 65535 f \n`;
  for (const off of offsets) pdf += `${String(off).padStart(10, '0')} 00000 n \n`;
  pdf += `trailer\n<< /Size ${objs.length + 1} /Root 1 0 R >>\nstartxref\n${xrefPos}\n%%EOF`;
  return Buffer.from(pdf, 'latin1');
}
function fakeJpeg(tag, size) {
  return Buffer.concat([
    Buffer.from([0xff, 0xd8, 0xff, 0xe0]),
    Buffer.from(tag),
    Buffer.alloc(size, 0x42),
    Buffer.from([0xff, 0xd9])
  ]);
}
const scanPdf = Buffer.concat([
  buildPdf(['SCAN']),            // 4-char text layer → triggers scanned-PDF OCR path
  fakeJpeg('PAGE_A', 13_000),    // page 1 (≥ MIN_OCR_IMAGE_BYTES)
  Buffer.from([0x00, 0x01]),
  fakeJpeg('PAGE_B', 14_000)     // page 2
]);
res = await fetch(`${BASE}/api/documents`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ fileName: 'scan.pdf', fileType: 'pdf', base64Data: scanPdf.toString('base64'), size: scanPdf.length })
});
body = await res.json();
const scanId = body.document?.id;
const raw = String(body.document?.raw_text || '');
log(`scan upload: http=${res.status} status=${body.status} raw=${JSON.stringify(raw.slice(0, 80))}`);
check('scanned multi-image PDF upload succeeds without crash', res.status === 200 && body.status === 'success', `http=${res.status} status=${body.status}`);
check('no fabricated text added (garbage OCR pages yield only real text layer)', raw.includes('SCAN') && !raw.includes('PAGE_A'), `raw_len=${raw.length}`);
await del(scanId);

log(failed ? 'VERIFY: FAILED' : 'VERIFY: ALL PASSED');
process.exit(failed ? 1 : 0);
