/**
 * Phase 1 end-to-end smoke test against the running local server.
 * Run: node e2e-smoke.mjs   (expects SafeDocAI on http://127.0.0.1:3100)
 */
import fs from 'node:fs';
import path from 'node:path';

const BASE = process.env.E2E_BASE || 'http://127.0.0.1:3100';
const results = [];
function check(name, cond, extra = '') {
  results.push(`${cond ? 'PASS' : 'FAIL'} - ${name}${extra ? ' :: ' + extra : ''}`);
}

// --- Minimal PDF with real text (correct xref offsets) ---
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
  objs.forEach((body, i) => {
    offsets.push(pdf.length);
    pdf += `${i + 1} 0 obj\n${body}\nendobj\n`;
  });
  const xrefPos = pdf.length;
  pdf += `xref\n0 ${objs.length + 1}\n0000000000 65535 f \n`;
  for (const off of offsets) pdf += `${String(off).padStart(10, '0')} 00000 n \n`;
  pdf += `trailer\n<< /Size ${objs.length + 1} /Root 1 0 R >>\nstartxref\n${xrefPos}\n%%EOF`;
  return Buffer.from(pdf, 'latin1');
}

const pdf = buildPdf([
  'APEX INSTITUTE OF TECHNOLOGY',
  'Candidate Name: Rahul Sharma',
  "Father's Name: Ramesh Sharma",
  "Mother's Name: Sunita Sharma",
  'Roll No: 2023CS001',
  'Total Marks: 456 / 500',
  'Percentage: 91.2%',
  'Result: FIRST DIVISION'
]);
const b64 = pdf.toString('base64');

// 1. Health
const health = await (await fetch(`${BASE}/api/health`)).json();
// Pre-existing inventory (e.g. the user's own documents) must survive this run.
const baseline = (await (await fetch(`${BASE}/api/documents`)).json()).length;
check('health ok + offline mode', health.status === 'ok' && health.status === 'ok', JSON.stringify({ ollama: health.ollama_ready, ocr: health.ocr_engine, schema: health.db_schema_version }));

// 2. Upload real PDF
let res = await fetch(`${BASE}/api/documents`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ fileName: 'College Marksheet.pdf', fileType: 'pdf', base64Data: b64, size: pdf.length })
});
let body = await res.json();
const docId = body.document?.id;
check('upload success (honest pipeline)', res.ok && body.status === 'success' && docId > 0, `http=${res.status} id=${docId}`);
const fieldNames = (body.document?.fields || []).map(f => f.field_name);
check('raw_text extracted via pdf-parse', (body.document?.raw_text || '').includes('Rahul Sharma'), (body.document?.raw_text || '').slice(0, 60).replace(/\n/g, ' '));
check('roll_number + parent fields extracted', fieldNames.includes('roll_number') && fieldNames.includes('father_name'), fieldNames.join(','));
check('data_url present for immediate preview', Boolean(body.document?.data_url?.startsWith('data:application/pdf')));
check('chunks built server-side (500/50)', (body.document?.chunks || []).length > 0, `${(body.document?.chunks || []).length} chunks`);
check('category detected', body.document?.category === 'Marksheet / Degree', body.document?.category);

// 3. Duplicate -> 409
res = await fetch(`${BASE}/api/documents`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ fileName: 'Copy.pdf', fileType: 'pdf', base64Data: b64, size: pdf.length })
});
body = await res.json();
check('duplicate rejected with 409 (FR-04)', res.status === 409 && body.status === 'duplicate', `http=${res.status}`);

// 4. Unsupported -> 422
res = await fetch(`${BASE}/api/documents`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ fileName: 'x.docx', fileType: 'docx', base64Data: b64, size: 10 })
});
check('unsupported format rejected 422', res.status === 422, `http=${res.status}`);

// 5. List + managed file URL
const list = await (await fetch(`${BASE}/api/documents`)).json();
const newDoc = list.find(d => d.id === docId);
check('list returns baseline+1 doc with file_url', list.length === baseline + 1 && Boolean(newDoc?.file_url), `${list.length} docs (baseline ${baseline})`);
const fileRes = await fetch(`${BASE}${newDoc.file_url}`);
const fileBytes = Buffer.from(await fileRes.arrayBuffer());
check('file endpoint streams stored original bytes', fileRes.ok && fileBytes.equals(pdf), `${fileBytes.length} bytes`);

// 6. Managed store layout on disk
check('original persisted under data/documents', fs.existsSync(path.resolve(newDoc.file_path)), newDoc.file_path);
check('understood artifact written', fs.readdirSync(path.resolve('data', 'understood')).length > 0);

// 7. Grounded query via Ollama (offline)
const t0 = Date.now();
res = await fetch(`${BASE}/api/ask-query`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ query: 'mera roll number kya hai?' })
});
body = await res.json();
const grounded = (body.answer || '').includes('2023CS001') || JSON.stringify(body.evidence || []).includes('2023CS001');
check('ask-query answers grounded in real value', res.ok && grounded, `${Date.now() - t0}ms route=${body.route} ans=${(body.answer || '').slice(0, 80).replace(/\n/g, ' ')}`);

// 8. Reprocess from stored original
res = await fetch(`${BASE}/api/documents/${docId}/reprocess`, { method: 'POST' });
body = await res.json();
check('reprocess from managed original succeeds', res.ok && body.status === 'success' && (body.document?.fields || []).length > 0, `fields=${(body.document?.fields || []).length}`);

// 9. Diagnostics are real
const diag = await (await fetch(`${BASE}/api/diagnostics`)).json();
check('diagnostics counts real + offline', diag.sqlite_records_count === baseline + 1 && diag.vector_chunks_count > 0 && diag.storage_bytes_used > 0 && diag.network_egress_status === 'zero_egress_local', JSON.stringify(diag));

// 10. Delete + artifact purge
res = await fetch(`${BASE}/api/documents/${docId}`, { method: 'DELETE' });
body = await res.json();
check('delete success', res.ok && body.success === true);
const list2 = await (await fetch(`${BASE}/api/documents`)).json();
check('own doc deleted, baseline inventory kept', list2.length === baseline && !list2.some(d => d.id === docId), `${list2.length} docs (baseline ${baseline})`);
check('managed artifacts purged', !fs.existsSync(path.resolve(newDoc.file_path)));

console.log(results.join('\n'));
if (results.some(r => r.startsWith('FAIL'))) {
  process.exit(1);
}
console.log('ALL E2E CHECKS PASSED');
