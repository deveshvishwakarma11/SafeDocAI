import express, { Request, Response } from 'express';
import { createServer as createViteServer } from 'vite';
import path from 'node:path';
import fs from 'node:fs';
import crypto from 'node:crypto';
import dotenv from 'dotenv';

import {
  openDatabase,
  reconcileInterrupted,
  createProcessingDocument,
  finalizeDocument,
  setDocumentFields,
  setDocumentChunks,
  replaceDocumentText,
  listDocuments,
  listDocumentRows,
  getDocument,
  findByContentHash,
  purgeFailedByHash,
  supersedeOlderVersions,
  deleteDocument,
  clearDocuments,
  getDiagnosticCounts,
  runIntegrityCheck,
  getSchemaVersion
} from './server/db';
import {
  MAX_FILE_BYTES,
  ensureStoreDirs,
  saveOriginal,
  readOriginal,
  originalExists,
  originalAbsolutePath,
  deleteManagedArtifacts,
  writeUnderstoodArtifact,
  bytesUsed,
  mimeForType,
  toDataUrl
} from './server/fileStore';
import { chunkText } from './server/chunker';
import {
  extractTextFromPdfBuffer,
  extractImagesFromPdfBuffer,
  extractRealEntities,
  normalizeOcrText,
  recognizeImage,
  getTesseractWorker,
  selectOcrImages,
  joinOcrPageTexts,
  ocrStatus
} from './server/extract';
import { askQuestion } from './server/ask';
import { ollamaStatus, ollamaGenerate, synthesisEnabled, OLLAMA_MODEL } from './server/ollama';
import type { ExtractedField } from './src/types/index';

dotenv.config();

const app = express();
const PORT = Number(process.env.PORT) || 3000;
// Loopback only (TRD §9): documents must never be reachable from the LAN.
const HOST = process.env.HOST || '127.0.0.1';

// Enable large JSON payloads for document uploads (base64 PDFs and high-res images)
app.use(express.json({ limit: '50mb' }));

/* ------------------------------------------------------------------ */
/* Managed store + SQLite database (Backend Schema §03/§04)             */
/* ------------------------------------------------------------------ */

const DATA_ROOT = path.resolve(process.cwd(), 'data');
ensureStoreDirs(DATA_ROOT);
const database = openDatabase(path.join(DATA_ROOT, 'safedoc.sqlite'));

// §08 status contract: rows stuck in `processing` after a crash are marked failed.
const reconciled = reconcileInterrupted(database);
if (reconciled > 0) {
  console.warn(`[startup-reconcile] marked ${reconciled} interrupted 'processing' document(s) as failed`);
}

// Pre-warm the Tesseract worker in the background (tolerates offline failure)
getTesseractWorker()
  .then(() => console.log(`  ocr     : ready (${ocrStatus()})`))
  .catch(e => console.warn('Tesseract worker pre-warm warning:', e.message));

/* ------------------------------------------------------------------ */
/* Health                                                               */
/* ------------------------------------------------------------------ */

app.get('/api/health', async (req: Request, res: Response) => {
  res.json({
    status: 'ok',
    ollama_ready: await ollamaStatus(),
    ollama_model: OLLAMA_MODEL,
    ocr_engine: ocrStatus(),
    db_schema_version: getSchemaVersion(database),
    timestamp: new Date().toISOString()
  });
});

/* ------------------------------------------------------------------ */
/* Document ingestion pipeline                                          */
/* ------------------------------------------------------------------ */

function generateArtifactStem(fileName: string, hash: string): string {
  // Same algorithm as src/utils/crypto.ts so stems stay stable end-to-end
  const cleanName = fileName.replace(/[^a-zA-Z0-9_-]/g, '_').toLowerCase();
  return `${cleanName}_${hash.substring(0, 10)}`;
}

interface IngestResult {
  raw_text: string;
  category: string;
  page_count: number;
  fields: ExtractedField[];
}

function toExtractedFields(docId: number, rawFields: any[]): ExtractedField[] {
  return (rawFields || [])
    .filter(f => f && f.field_name && f.field_value)
    .map((f, idx) => ({
      id: `f_${docId}_${idx + 1}`,
      document_id: docId,
      field_name: String(f.field_name),
      display_label: f.display_label || String(f.field_name).replace(/_/g, ' '),
      field_value: String(f.field_value),
      evidence_text: f.evidence_text || String(f.field_value),
      page_number: f.page_number || 1,
      confidence: typeof f.confidence === 'number' ? f.confidence : 0.95,
      value_type: f.value_type || 'text',
      created_at: new Date().toISOString()
    }));
}

function honestNoTextNote(fileName: string, fileType: string): string {
  return `DOCUMENT: ${fileName}\nType: ${fileType.toUpperCase()}\nUploaded: ${new Date().toLocaleDateString()}\n\nNote: OCR scanned this file but could not identify clear text characters. Please verify document clarity and resolution.`;
}

/**
 * Text extraction + understanding for one file buffer.
 * Fully local: pdf-parse / Tesseract, then deterministic rule-based extraction.
 * No document content ever leaves this machine.
 */
async function ingestDocument(buffer: Buffer, fileName: string, fileType: string): Promise<IngestResult> {
  let extractedText = '';
  let pageCount = 1;

  const isPdf = fileType === 'pdf';
  const isImage = ['png', 'jpg', 'jpeg'].includes(fileType);

  if (isPdf) {
    try {
      const pdfExtract = await extractTextFromPdfBuffer(buffer);
      extractedText = pdfExtract.text;
      pageCount = pdfExtract.pages;

      // Scanned PDF (< 30 chars of text layer): OCR EVERY embedded page image
      // in stream order (page order), bounded by MAX_OCR_PAGES so one huge
      // scan can't pin the CPU for minutes. Pages join with form-feed (\f).
      if (extractedText.trim().length < 30) {
        const images = selectOcrImages(extractImagesFromPdfBuffer(buffer));
        if (images.length > 0) {
          const pageTexts: string[] = [];
          for (const img of images) {
            try {
              pageTexts.push(await recognizeImage(img));
            } catch (tessErr) {
              console.warn('Tesseract OCR on scanned PDF page failed:', (tessErr as Error).message);
            }
          }
          const ocrText = joinOcrPageTexts(pageTexts);
          if (ocrText.length > extractedText.length) {
            extractedText = ocrText;
            pageCount = Math.max(pageCount, images.length);
          }
        }
      }
    } catch (e) {
      console.warn('PDF processing error:', (e as Error).message);
    }
  } else if (isImage) {
    try {
      const ocrText = await recognizeImage(buffer);
      if (ocrText.length > 0) {
        extractedText = ocrText;
      }
    } catch (tessErr) {
      console.warn('Tesseract OCR on image failed:', (tessErr as Error).message);
    }
  }

  // Deterministic local extraction from actual OCR text
  const cleanText = normalizeOcrText(extractedText);
  const analysis = extractRealEntities(cleanText, fileName, fileType);
  const finalRawText = cleanText.trim().length > 0
    ? cleanText
    : honestNoTextNote(fileName, fileType);

  return {
    raw_text: finalRawText,
    category: analysis.category,
    page_count: Math.max(pageCount, analysis.page_count),
    fields: toExtractedFields(0, analysis.fields)
  };
}

/* ------------------------------------------------------------------ */
/* Document routes (Backend Schema §08 status contract)                 */
/* ------------------------------------------------------------------ */

app.post('/api/documents', async (req: Request, res: Response) => {
  let docId: number | null = null;
  try {
    const { fileName, fileType, base64Data } = req.body || {};

    if (!fileName || typeof fileName !== 'string') {
      res.status(400).json({ status: 'failed', error: 'File name is required' });
      return;
    }

    const extension = fileName.split('.').pop()?.toLowerCase() || '';
    let docType = 'other';
    if (extension === 'pdf') docType = 'pdf';
    else if (['jpg', 'jpeg'].includes(extension)) docType = 'jpg';
    else if (extension === 'png') docType = 'png';
    else {
      res.status(422).json({
        status: 'failed',
        error: `Unsupported format: .${extension}. SafeDocAI accepts PDF, JPG, JPEG, and PNG.`
      });
      return;
    }

    if (!base64Data || typeof base64Data !== 'string') {
      res.status(400).json({ status: 'failed', error: 'File content (base64Data) is required' });
      return;
    }

    const buffer = Buffer.from(base64Data, 'base64');
    if (buffer.length === 0) {
      res.status(400).json({ status: 'failed', error: 'File content is empty or not valid base64' });
      return;
    }
    if (buffer.length > MAX_FILE_BYTES) {
      res.status(413).json({
        status: 'failed',
        error: `File exceeds the ${Math.round(MAX_FILE_BYTES / (1024 * 1024))} MB limit (received ${(buffer.length / (1024 * 1024)).toFixed(1)} MB).`
      });
      return;
    }

    // Content-hash dedupe (FR-04) — server-authoritative
    const hash = crypto.createHash('sha256').update(buffer).digest('hex');
    const existing = findByContentHash(database, hash);
    if (existing) {
      res.status(409).json({
        status: 'duplicate',
        document: existing,
        error: `Exact duplicate detected: This file matches "${existing.file_name}" already indexed on ${new Date(existing.upload_timestamp).toLocaleDateString()}.`
      });
      return;
    }

    const artifactStem = generateArtifactStem(fileName, hash);
    // Drop any failed attempt of the same bytes so a retry can succeed
    purgeFailedByHash(database, hash);
    docId = createProcessingDocument(database, {
      file_name: fileName,
      file_type: docType,
      file_path: `data/documents/${artifactStem}.${docType}`,
      content_hash: hash,
      artifact_stem: artifactStem,
      file_size_bytes: buffer.length
    });

    // Persist original bytes to the managed file store BEFORE extraction (FR-03)
    let storedPath: string;
    try {
      storedPath = saveOriginal(DATA_ROOT, buffer, artifactStem, docType);
    } catch (storeErr: any) {
      finalizeDocument(database, docId, {
        raw_text: '',
        category: 'General Document',
        page_count: 1,
        status: 'failed',
        error_message: `Could not persist original to managed store: ${storeErr.message}`
      });
      res.status(500).json({ status: 'failed', error: storeErr.message });
      return;
    }

    const ingest = await ingestDocument(buffer, fileName, docType);

    finalizeDocument(database, docId, {
      raw_text: ingest.raw_text,
      category: ingest.category,
      page_count: ingest.page_count,
      status: 'success'
    });
    setDocumentFields(database, docId, ingest.fields);
    const chunks = chunkText(docId, fileName, ingest.raw_text);
    setDocumentChunks(database, docId, fileName, chunks);

    // Version replacement: older same-name rows become `superseded`, retained
    supersedeOlderVersions(database, fileName, docId);

    writeUnderstoodArtifact(DATA_ROOT, artifactStem, {
      document_id: docId,
      file_name: fileName,
      file_type: docType,
      content_hash: hash,
      artifact_stem: artifactStem,
      category: ingest.category,
      file_path: storedPath,
      created_at: new Date().toISOString(),
      extracted_metadata: ingest.fields,
      chunks_count: chunks.length
    });

    const doc = getDocument(database, docId)!;
    res.json({ status: 'success', document: { ...doc, data_url: toDataUrl(buffer, docType) } });
  } catch (error: any) {
    console.error('Error processing document:', error);
    if (docId !== null) {
      try {
        finalizeDocument(database, docId, {
          raw_text: '',
          category: 'General Document',
          page_count: 1,
          status: 'failed',
          error_message: error.message || 'Unknown ingestion error'
        });
      } catch {
        // keep original error surfaced below
      }
    }
    res.status(500).json({
      status: 'failed',
      error: error.message || 'Document ingestion failed unexpectedly.'
    });
  }
});

app.get('/api/documents', (req: Request, res: Response) => {
  const includeSuperseded = req.query.include_superseded === '1';
  res.json(listDocuments(database, includeSuperseded));
});

app.get('/api/documents/:id', (req: Request, res: Response) => {
  const doc = getDocument(database, Number(req.params.id));
  if (!doc) {
    res.status(404).json({ error: 'Document not found' });
    return;
  }
  res.json(doc);
});

/** Original bytes straight from the managed store — previews survive reloads. */
app.get('/api/documents/:id/file', (req: Request, res: Response) => {
  const doc = getDocument(database, Number(req.params.id));
  if (!doc) {
    res.status(404).json({ error: 'Document not found' });
    return;
  }
  if (!originalExists(DATA_ROOT, doc.file_path)) {
    res.status(404).json({ error: 'Original file is not present in the managed store' });
    return;
  }
  res.setHeader('Content-Type', mimeForType(doc.file_type));
  res.setHeader('Cache-Control', 'private, max-age=3600');
  fs.createReadStream(originalAbsolutePath(DATA_ROOT, doc.file_path)).pipe(res);
});

/** Re-run OCR + field extraction from the STORED original (no client bytes needed). */
app.post('/api/documents/:id/reprocess', async (req: Request, res: Response) => {
  try {
    const id = Number(req.params.id);
    const doc = getDocument(database, id);
    if (!doc) {
      res.status(404).json({ error: 'Document not found' });
      return;
    }
    if (!originalExists(DATA_ROOT, doc.file_path)) {
      res.status(409).json({
        error: 'Original bytes are missing from the managed store. Please re-upload the file.'
      });
      return;
    }

    const buffer = readOriginal(DATA_ROOT, doc.file_path);
    const ingest = await ingestDocument(buffer, doc.file_name, doc.file_type);

    finalizeDocument(database, id, {
      raw_text: ingest.raw_text,
      category: ingest.category,
      page_count: ingest.page_count,
      status: 'success'
    });
    setDocumentFields(database, id, ingest.fields);
    const chunks = chunkText(id, doc.file_name, ingest.raw_text);
    setDocumentChunks(database, id, doc.file_name, chunks);

    writeUnderstoodArtifact(DATA_ROOT, doc.artifact_stem, {
      document_id: id,
      file_name: doc.file_name,
      file_type: doc.file_type,
      content_hash: doc.content_hash,
      artifact_stem: doc.artifact_stem,
      category: ingest.category,
      file_path: doc.file_path,
      created_at: new Date().toISOString(),
      extracted_metadata: ingest.fields,
      chunks_count: chunks.length
    });

    res.json({ status: 'success', document: getDocument(database, id) });
  } catch (error: any) {
    console.error('Reprocess error:', error);
    res.status(500).json({ status: 'failed', error: error.message || 'Reprocess failed' });
  }
});

app.delete('/api/documents/:id', (req: Request, res: Response) => {
  const removed = deleteDocument(database, Number(req.params.id));
  if (!removed) {
    res.status(404).json({ success: false, error: 'Document not found' });
    return;
  }
  deleteManagedArtifacts(DATA_ROOT, removed.file_path, removed.artifact_stem);
  res.json({ success: true, removedId: removed.id });
});

app.delete('/api/documents', (req: Request, res: Response) => {
  const removed = clearDocuments(database);
  for (const row of removed) {
    deleteManagedArtifacts(DATA_ROOT, row.file_path, row.artifact_stem);
  }
  res.json({ success: true, removed: removed.length });
});

/** Settings → Rebuild Vector Index: re-chunk every success document from raw_text. */
app.post('/api/index/rebuild', (req: Request, res: Response) => {
  const rows = listDocumentRows(database);
  let rebuilt = 0;
  for (const row of rows) {
    if (row.status !== 'success') continue;
    const chunks = chunkText(row.id, row.file_name, row.raw_text);
    setDocumentChunks(database, row.id, row.file_name, chunks);
    rebuilt++;
  }
  res.json({ success: true, rebuilt });
});

app.get('/api/diagnostics', async (req: Request, res: Response) => {
  const counts = getDiagnosticCounts(database);
  const integrity = runIntegrityCheck(database);
  const ollamaReady = await ollamaStatus();
  res.json({
    sqlite_status: integrity.ok ? 'healthy' : 'degraded',
    sqlite_records_count: counts.documents,
    sqlite_metadata_count: counts.fields,
    schema_version: getSchemaVersion(database),
    vector_engine_status: 'healthy',
    vector_chunks_count: counts.chunks,
    local_llm_status: ollamaReady ? 'connected_ollama' : 'unavailable',
    ollama_synthesis_enabled: synthesisEnabled(),
    ocr_engine_status: ocrStatus(),
    network_egress_status: 'zero_egress_local',
    storage_bytes_used: bytesUsed(DATA_ROOT)
  });
});

/* ------------------------------------------------------------------ */
/* Grounded Q&A: engines → retrieval → local Ollama (verified) → honest */
/* ------------------------------------------------------------------ */

app.post('/api/ask-query', async (req: Request, res: Response) => {
  // The client sends only the question (+ optional open-document id). Documents
  // are read here from SQLite: no library upload per question, and the server
  // is the single source of truth for what is searchable.
  const query = typeof req.body?.query === 'string' ? req.body.query.trim() : '';
  if (!query) {
    res.status(400).json({ error: 'Query is required' });
    return;
  }
  const scopeId = typeof req.body?.activeDocFilter === 'number' ? req.body.activeDocFilter : undefined;

  // Stop button / closed tab => cancel generation so the CPU is freed at once.
  const abort = new AbortController();
  res.on('close', () => {
    if (!res.writableEnded) abort.abort();
  });

  try {
    const documents = listDocuments(database).filter(d => d.status === 'success');
    const result = await askQuestion(
      query,
      documents,
      scopeId,
      {
        synthesis: synthesisEnabled(),
        ollamaReady: ollamaStatus,
        generate: (prompt, opts) => ollamaGenerate(prompt, { signal: opts.signal, numPredict: opts.numPredict }),
        log: m => console.warn(m)
      },
      abort.signal
    );
    if (!res.writableEnded && !abort.signal.aborted) res.json(result);
  } catch (error: any) {
    console.error('Error answering query:', error);
    if (!res.writableEnded && !abort.signal.aborted) {
      res.status(500).json({ error: 'Could not answer this question. Please try again.' });
    }
  }
});

/* ------------------------------------------------------------------ */
/* Vite dev middleware / static prod build                              */
/* ------------------------------------------------------------------ */

async function startServer() {
  if (process.env.NODE_ENV !== 'production') {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa'
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.resolve(process.cwd(), 'dist');
    app.use(express.static(distPath));
    app.get('*', (req: Request, res: Response) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  const ollamaReady = await ollamaStatus();
  app.listen(PORT, HOST, () => {
    console.log(`SafeDocAI server running on http://${HOST}:${PORT}`);
    console.log(`  storage : ${DATA_ROOT}`);
    console.log(`  llm     : ${ollamaReady ? `ollama (${OLLAMA_MODEL})` : 'none detected'}`);
    console.log('  ocr     : warming up in background...');
  });
}

startServer();
