import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {
  openDatabase,
  getSchemaVersion,
  reconcileInterrupted,
  createProcessingDocument,
  finalizeDocument,
  setDocumentFields,
  setDocumentChunks,
  listDocuments,
  getDocument,
  findByContentHash,
  supersedeOlderVersions,
  deleteDocument,
  clearDocuments,
  getDiagnosticCounts,
  runIntegrityCheck,
  SCHEMA_VERSION
} from '../server/db';
import type { ExtractedField } from '../src/types/index';

function tempDb() {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'safedoc-db-'));
  const db = openDatabase(path.join(dir, 'test.sqlite'));
  return { db, dir };
}

function makeInput(name: string, hash: string) {
  const stem = `${name.replace(/[^a-zA-Z0-9_-]/g, '_')}_${hash.substring(0, 10)}`;
  return {
    file_name: name,
    file_type: 'pdf',
    file_path: `data/documents/${stem}.pdf`,
    content_hash: hash,
    artifact_stem: stem,
    file_size_bytes: 1234
  };
}

test('schema migrates to version 1 and passes integrity check', () => {
  const { db } = tempDb();
  assert.equal(getSchemaVersion(db), SCHEMA_VERSION);
  assert.equal(runIntegrityCheck(db).ok, true);
});

test('lifecycle: processing → success with fields/chunks round-trip', () => {
  const { db } = tempDb();
  const id = createProcessingDocument(db, makeInput('marksheet.pdf', 'a'.repeat(64)));

  // §08: documents are never born `success`
  const before = listDocuments(db);
  assert.equal(before.length, 1);
  assert.equal(before[0].status, 'processing');

  finalizeDocument(db, id, {
    raw_text: 'Roll No: 12345',
    category: 'Marksheet / Degree',
    page_count: 2,
    status: 'success'
  });

  const field: ExtractedField = {
    id: `f_${id}_1`,
    document_id: id,
    field_name: 'roll_number',
    display_label: 'Roll Number',
    field_value: '12345',
    evidence_text: 'Roll No: 12345',
    page_number: 1,
    confidence: 0.99,
    value_type: 'identifier',
    created_at: new Date().toISOString()
  };
  setDocumentFields(db, id, [field]);
  setDocumentChunks(db, id, 'marksheet.pdf', [
    { chunk_id: `doc_${id}_chunk_0`, page_number: 1, chunk_index: 0, text: 'Roll No: 12345 ......', token_count: 6 }
  ]);

  const doc = getDocument(db, id)!;
  assert.equal(doc.status, 'success');
  assert.equal(doc.page_count, 2);
  assert.equal(doc.category, 'Marksheet / Degree');
  assert.equal(doc.fields.length, 1);
  assert.equal(doc.fields[0].field_value, '12345');
  assert.equal(doc.chunks.length, 1);
  assert.equal(doc.file_url, `/api/documents/${id}/file`);

  assert.deepEqual(getDiagnosticCounts(db), { documents: 1, fields: 1, chunks: 1 });
});

test('content-hash dedupe: lookup works and unique index rejects re-insert', () => {
  const { db } = tempDb();
  const id = createProcessingDocument(db, makeInput('a.pdf', 'b'.repeat(64)));

  const found = findByContentHash(db, 'b'.repeat(64));
  assert.ok(found);
  assert.equal(found!.id, id);
  assert.equal(findByContentHash(db, 'c'.repeat(64)), undefined);

  assert.throws(() => createProcessingDocument(db, makeInput('copy.pdf', 'b'.repeat(64))));
});

test('version replacement: older same-name rows become superseded (hidden, retained)', () => {
  const { db } = tempDb();
  const v1 = createProcessingDocument(db, makeInput('bill.pdf', '1'.repeat(64)));
  finalizeDocument(db, v1, { raw_text: 'v1 text', category: 'Utility Bill', page_count: 1, status: 'success' });
  const v2 = createProcessingDocument(db, makeInput('bill.pdf', '2'.repeat(64)));
  finalizeDocument(db, v2, { raw_text: 'v2 text', category: 'Utility Bill', page_count: 1, status: 'success' });

  const superseded = supersedeOlderVersions(db, 'bill.pdf', v2);
  assert.equal(superseded, 1);

  const visible = listDocuments(db);
  assert.equal(visible.length, 1);
  assert.equal(visible[0].id, v2);
  assert.equal(visible[0].raw_text, 'v2 text');

  const all = listDocuments(db, true);
  assert.equal(all.length, 2);
  const old = all.find(d => d.id === v1)!;
  assert.equal(old.status, 'superseded');
  assert.ok(old.error_message?.includes('newer version'));
});

test('delete cascades fields/chunks and reports artifact paths for purge', () => {
  const { db } = tempDb();
  const id = createProcessingDocument(db, makeInput('x.pdf', 'd'.repeat(64)));
  finalizeDocument(db, id, { raw_text: 'text', category: 'General Document', page_count: 1, status: 'success' });
  setDocumentFields(db, id, [
    {
      id: `f_${id}_1`,
      document_id: id,
      field_name: 'a',
      display_label: 'A',
      field_value: 'v',
      evidence_text: 'v',
      page_number: 1,
      confidence: 1,
      value_type: 'text',
      created_at: new Date().toISOString()
    }
  ]);

  const removed = deleteDocument(db, id)!;
  assert.equal(removed.id, id);
  assert.ok(removed.file_path.startsWith('data/documents/'));
  assert.ok(removed.artifact_stem.length > 0);

  assert.equal(getDocument(db, id), undefined);
  assert.equal(getDiagnosticCounts(db).fields, 0);
  assert.equal(runIntegrityCheck(db).ok, true);
  assert.equal(deleteDocument(db, 999), undefined);
});

test('reconcileInterrupted marks stale processing rows failed, idempotently', () => {
  const { db } = tempDb();
  createProcessingDocument(db, makeInput('stuck.pdf', 'e'.repeat(64)));
  createProcessingDocument(db, makeInput('also-stuck.pdf', 'f'.repeat(64)));

  const fixed = reconcileInterrupted(db);
  assert.equal(fixed, 2);
  for (const d of listDocuments(db)) {
    assert.equal(d.status, 'failed');
    assert.ok(d.error_message?.includes('Interrupted'));
  }
  assert.equal(reconcileInterrupted(db), 0);
});

test('clearDocuments empties the registry and reports rows for artifact purge', () => {
  const { db } = tempDb();
  createProcessingDocument(db, makeInput('one.pdf', '1'.repeat(63) + 'a'));
  createProcessingDocument(db, makeInput('two.pdf', '2'.repeat(63) + 'b'));

  const rows = clearDocuments(db);
  assert.equal(rows.length, 2);
  assert.equal(listDocuments(db).length, 0);
  assert.deepEqual(getDiagnosticCounts(db), { documents: 0, fields: 0, chunks: 0 });
});
