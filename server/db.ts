/**
 * SafeDocAI SQLite Storage Engine
 * --------------------------------
 * Implements the Backend Schema document §03/§04 (proposed target model):
 *  - documents / metadata_fields / document_chunks tables
 *  - content_hash + artifact_stem UNIQUE constraints (§09)
 *  - status lifecycle: processing -> success | failed, plus `superseded`
 *    for version replacement (hidden from retrieval, row retained)
 *  - additive migrations tracked with PRAGMA user_version (§11)
 *  - foreign_key_check after migration (§11 integrity rule)
 */
import { DatabaseSync } from 'node:sqlite';
import fs from 'node:fs';
import path from 'node:path';
import type {
  StoredDocument,
  ExtractedField,
  DocumentChunk,
  DocumentType,
  DocumentCategory
} from '../src/types/index';

export const SCHEMA_VERSION = 1;
export const DEFAULT_PROFILE_ID = 'default_local_profile';

export interface DocumentRow {
  id: number;
  profile_id: string;
  file_name: string;
  file_type: string;
  category: string;
  file_path: string;
  content_hash: string;
  artifact_stem: string;
  status: string;
  error_message: string | null;
  upload_timestamp: string;
  processed_at: string | null;
  file_size_bytes: number;
  page_count: number;
  raw_text: string;
}

export function openDatabase(dbPath: string): DatabaseSync {
  fs.mkdirSync(path.dirname(dbPath), { recursive: true });
  const db = new DatabaseSync(dbPath);
  db.exec('PRAGMA foreign_keys = ON;');
  migrate(db);
  return db;
}

function migrate(db: DatabaseSync): void {
  const row = db.prepare('PRAGMA user_version').get() as { user_version: number };
  const current = row?.user_version ?? 0;

  if (current < 1) {
    db.exec(`
      BEGIN;
      CREATE TABLE IF NOT EXISTS documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        profile_id TEXT NOT NULL DEFAULT '${DEFAULT_PROFILE_ID}',
        file_name TEXT NOT NULL,
        file_type TEXT NOT NULL,
        category TEXT NOT NULL DEFAULT 'General Document',
        file_path TEXT NOT NULL,
        content_hash TEXT NOT NULL,
        artifact_stem TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'processing',
        error_message TEXT,
        upload_timestamp TEXT NOT NULL,
        processed_at TEXT,
        file_size_bytes INTEGER NOT NULL DEFAULT 0,
        page_count INTEGER NOT NULL DEFAULT 1,
        raw_text TEXT NOT NULL DEFAULT ''
      );
      CREATE UNIQUE INDEX IF NOT EXISTS uq_documents_content_hash ON documents(content_hash) WHERE status != 'failed';
      CREATE UNIQUE INDEX IF NOT EXISTS uq_documents_artifact_stem ON documents(artifact_stem) WHERE status != 'failed';
      CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);
      CREATE INDEX IF NOT EXISTS idx_documents_file_name ON documents(file_name);

      CREATE TABLE IF NOT EXISTS metadata_fields (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
        field_name TEXT NOT NULL,
        display_label TEXT NOT NULL,
        field_value TEXT NOT NULL,
        evidence_text TEXT NOT NULL DEFAULT '',
        page_number INTEGER,
        confidence REAL NOT NULL DEFAULT 0,
        value_type TEXT NOT NULL DEFAULT 'text',
        source_start INTEGER,
        source_end INTEGER,
        created_at TEXT NOT NULL
      );
      CREATE INDEX IF NOT EXISTS idx_metadata_fields_document ON metadata_fields(document_id);
      CREATE INDEX IF NOT EXISTS idx_metadata_fields_name ON metadata_fields(field_name);

      CREATE TABLE IF NOT EXISTS document_chunks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
        chunk_id TEXT NOT NULL,
        file_name TEXT NOT NULL,
        page_number INTEGER NOT NULL DEFAULT 1,
        chunk_index INTEGER NOT NULL,
        text TEXT NOT NULL,
        token_count INTEGER NOT NULL DEFAULT 0
      );
      CREATE INDEX IF NOT EXISTS idx_document_chunks_document ON document_chunks(document_id);

      PRAGMA user_version = ${SCHEMA_VERSION};
      COMMIT;
    `);
  }

  // §11: post-migration integrity gate
  const violations = db.prepare('PRAGMA foreign_key_check').all();
  if (violations.length > 0) {
    throw new Error(`SQLite foreign_key_check failed: ${violations.length} violation(s)`);
  }
}

export function getSchemaVersion(db: DatabaseSync): number {
  const row = db.prepare('PRAGMA user_version').get() as { user_version: number };
  return row?.user_version ?? 0;
}

/** §08 status contract: rows stuck in `processing` after a crash/restart are honestly marked failed. */
export function reconcileInterrupted(db: DatabaseSync): number {
  const result = db
    .prepare(
      `UPDATE documents
         SET status = 'failed',
             error_message = 'Interrupted during processing (server restart). Re-upload or re-run OCR.'
       WHERE status = 'processing'`
    )
    .run();
  return Number(result.changes || 0);
}

export interface CreateDocumentInput {
  profile_id?: string;
  file_name: string;
  file_type: string;
  file_path: string;
  content_hash: string;
  artifact_stem: string;
  file_size_bytes: number;
  upload_timestamp?: string;
}

/** Insert a row in `processing` state (§08: no document is ever born `success`). */
export function createProcessingDocument(db: DatabaseSync, input: CreateDocumentInput): number {
  const result = db
    .prepare(
      `INSERT INTO documents
        (profile_id, file_name, file_type, category, file_path, content_hash, artifact_stem,
         status, upload_timestamp, file_size_bytes, page_count, raw_text)
       VALUES (?, ?, ?, 'General Document', ?, ?, ?, 'processing', ?, ?, 1, '')`
    )
    .run(
      input.profile_id ?? DEFAULT_PROFILE_ID,
      input.file_name,
      input.file_type,
      input.file_path,
      input.content_hash,
      input.artifact_stem,
      input.upload_timestamp ?? new Date().toISOString(),
      input.file_size_bytes
    );
  return Number(result.lastInsertRowid);
}

export interface FinalizeDocumentInput {
  raw_text: string;
  category: string;
  page_count: number;
  status: 'success' | 'failed';
  error_message?: string | null;
  processed_at?: string;
}

export function finalizeDocument(db: DatabaseSync, id: number, input: FinalizeDocumentInput): void {
  db.prepare(
    `UPDATE documents
       SET raw_text = ?, category = ?, page_count = ?, status = ?, error_message = ?, processed_at = ?
     WHERE id = ?`
  ).run(
    input.raw_text,
    input.category,
    Math.max(1, input.page_count),
    input.status,
    input.error_message ?? null,
    input.processed_at ?? new Date().toISOString(),
    id
  );
}

export function setDocumentFields(db: DatabaseSync, docId: number, fields: ExtractedField[]): void {
  db.prepare('DELETE FROM metadata_fields WHERE document_id = ?').run(docId);
  const insert = db.prepare(
    `INSERT INTO metadata_fields
      (document_id, field_name, display_label, field_value, evidence_text, page_number,
       confidence, value_type, source_start, source_end, created_at)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
  );
  for (const f of fields) {
    insert.run(
      docId,
      f.field_name,
      f.display_label,
      f.field_value,
      f.evidence_text ?? '',
      f.page_number ?? null,
      typeof f.confidence === 'number' ? f.confidence : 0,
      f.value_type ?? 'text',
      f.source_start ?? null,
      f.source_end ?? null,
      f.created_at || new Date().toISOString()
    );
  }
}

export function setDocumentChunks(db: DatabaseSync, docId: number, fileName: string, chunks: Omit<DocumentChunk, 'document_id' | 'file_name'>[]): void {
  db.prepare('DELETE FROM document_chunks WHERE document_id = ?').run(docId);
  const insert = db.prepare(
    `INSERT INTO document_chunks (document_id, chunk_id, file_name, page_number, chunk_index, text, token_count)
     VALUES (?, ?, ?, ?, ?, ?, ?)`
  );
  for (const c of chunks) {
    insert.run(
      docId,
      c.chunk_id,
      fileName,
      c.page_number,
      c.chunk_index,
      c.text,
      c.token_count ?? Math.ceil(c.text.length / 4)
    );
  }
}

export function replaceDocumentText(db: DatabaseSync, docId: number, rawText: string, category: string): void {
  db.prepare('UPDATE documents SET raw_text = ?, category = ?, processed_at = ? WHERE id = ?').run(
    rawText,
    category,
    new Date().toISOString(),
    docId
  );
}

function loadFields(db: DatabaseSync, docId: number): ExtractedField[] {
  const rows = db
    .prepare('SELECT * FROM metadata_fields WHERE document_id = ? ORDER BY id')
    .all(docId) as any[];
  return rows.map(r => ({
    id: `f_${docId}_${r.id}`,
    document_id: docId,
    field_name: r.field_name,
    display_label: r.display_label,
    field_value: r.field_value,
    evidence_text: r.evidence_text,
    page_number: r.page_number ?? undefined,
    confidence: r.confidence,
    value_type: r.value_type,
    source_start: r.source_start ?? undefined,
    source_end: r.source_end ?? undefined,
    created_at: r.created_at
  }));
}

function loadChunks(db: DatabaseSync, docId: number): DocumentChunk[] {
  const rows = db
    .prepare('SELECT * FROM document_chunks WHERE document_id = ? ORDER BY chunk_index')
    .all(docId) as any[];
  return rows.map(r => ({
    chunk_id: r.chunk_id,
    document_id: docId,
    file_name: r.file_name,
    page_number: r.page_number,
    chunk_index: r.chunk_index,
    text: r.text,
    token_count: r.token_count
  }));
}

export function rowToDocument(db: DatabaseSync, row: DocumentRow, withContent = true): StoredDocument {
  return {
    id: row.id,
    profile_id: row.profile_id,
    file_name: row.file_name,
    file_type: row.file_type as DocumentType,
    category: row.category as DocumentCategory,
    file_path: row.file_path,
    content_hash: row.content_hash,
    artifact_stem: row.artifact_stem,
    status: row.status as StoredDocument['status'],
    error_message: row.error_message ?? undefined,
    upload_timestamp: row.upload_timestamp,
    processed_at: row.processed_at ?? undefined,
    file_size_bytes: row.file_size_bytes,
    page_count: row.page_count,
    raw_text: withContent ? row.raw_text : '',
    file_url: `/api/documents/${row.id}/file`,
    fields: withContent ? loadFields(db, row.id) : [],
    chunks: withContent ? loadChunks(db, row.id) : []
  };
}

export function listDocuments(db: DatabaseSync, includeSuperseded = false): StoredDocument[] {
  const sql = includeSuperseded
    ? 'SELECT * FROM documents ORDER BY datetime(upload_timestamp) DESC'
    : "SELECT * FROM documents WHERE status != 'superseded' ORDER BY datetime(upload_timestamp) DESC";
  const rows = db.prepare(sql).all() as unknown as DocumentRow[];
  return rows.map(r => rowToDocument(db, r));
}

export function listDocumentRows(db: DatabaseSync): DocumentRow[] {
  return db.prepare('SELECT * FROM documents ORDER BY id').all() as unknown as DocumentRow[];
}

export function getDocument(db: DatabaseSync, id: number): StoredDocument | undefined {
  const row = db.prepare('SELECT * FROM documents WHERE id = ?').get(id) as DocumentRow | undefined;
  return row ? rowToDocument(db, row) : undefined;
}

export function findByContentHash(db: DatabaseSync, hash: string): StoredDocument | undefined {
  // Failed rows never count as duplicates — the user must be able to retry
  const row = db.prepare("SELECT * FROM documents WHERE content_hash = ? AND status != 'failed'").get(hash) as
    | DocumentRow
    | undefined;
  return row ? rowToDocument(db, row) : undefined;
}

/** Housekeeping: drop failed attempts of the same bytes before a retry insert. */
export function purgeFailedByHash(db: DatabaseSync, hash: string): number {
  const result = db.prepare("DELETE FROM documents WHERE content_hash = ? AND status = 'failed'").run(hash);
  return Number(result.changes || 0);
}

/**
 * Version replacement (AppFlow M-4 / my review S-2): when a NEW version of the
 * same file name is indexed successfully, older success rows with the same name
 * are marked `superseded` — hidden from retrieval, never deleted.
 */
export function supersedeOlderVersions(db: DatabaseSync, fileName: string, keepId: number): number {
  const result = db
    .prepare(
      `UPDATE documents
          SET status = 'superseded',
              error_message = 'Replaced by a newer version of the same file name.'
        WHERE id != ? AND lower(file_name) = lower(?) AND status = 'success'`
    )
    .run(keepId, fileName);
  return Number(result.changes || 0);
}

export interface DeleteResultRow {
  id: number;
  file_path: string;
  artifact_stem: string;
}

export function deleteDocument(db: DatabaseSync, id: number): DeleteResultRow | undefined {
  const row = db.prepare('SELECT id, file_path, artifact_stem FROM documents WHERE id = ?').get(id) as
    | DeleteResultRow
    | undefined;
  if (!row) return undefined;
  db.prepare('DELETE FROM documents WHERE id = ?').run(id);
  return row;
}

export function clearDocuments(db: DatabaseSync): DeleteResultRow[] {
  const rows = db.prepare('SELECT id, file_path, artifact_stem FROM documents').all() as unknown as DeleteResultRow[];
  db.prepare('DELETE FROM documents').run();
  return rows;
}

export interface DiagnosticCounts {
  documents: number;
  fields: number;
  chunks: number;
}

export function getDiagnosticCounts(db: DatabaseSync): DiagnosticCounts {
  const docs = db.prepare("SELECT COUNT(*) AS n FROM documents WHERE status != 'superseded'").get() as { n: number };
  const fields = db.prepare('SELECT COUNT(*) AS n FROM metadata_fields').get() as { n: number };
  const chunks = db.prepare('SELECT COUNT(*) AS n FROM document_chunks').get() as { n: number };
  return { documents: docs.n, fields: fields.n, chunks: chunks.n };
}

export function runIntegrityCheck(db: DatabaseSync): { ok: boolean; problems: string[] } {
  const problems: string[] = [];
  const fk = db.prepare('PRAGMA foreign_key_check').all();
  if (fk.length > 0) problems.push(`foreign_key_check: ${fk.length} violation(s)`);
  const quick = db.prepare('PRAGMA quick_check').all() as { quick_check: string }[];
  if (quick.length !== 1 || quick[0].quick_check !== 'ok') {
    problems.push(`quick_check: ${JSON.stringify(quick)}`);
  }
  return { ok: problems.length === 0, problems };
}
