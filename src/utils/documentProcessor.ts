import { StoredDocument } from '../types';
import { db } from './storage';

/**
 * Client-side ingestion coordinator (Phase 1 backend migration).
 * The client now only validates + ships raw bytes to the local server; the
 * server owns hashing, dedupe, OCR/extraction, chunking and persistence
 * (Backend Schema §08 status contract — no fabricated success anywhere).
 */
export class DocumentProcessor {
  public static async processDocument(
    file: File | { name: string; type: string; size: number; text: string },
    options: {
      onProgress?: (stage: string, progress: number) => void;
      isCancelled?: () => boolean;
    } = {}
  ): Promise<{
    status: 'success' | 'duplicate' | 'failed' | 'cancelled';
    document?: StoredDocument;
    error?: string;
  }> {
    const { onProgress, isCancelled } = options;

    if (isCancelled && isCancelled()) {
      return { status: 'cancelled', error: 'Upload cancelled by user' };
    }

    onProgress?.('Validating format and readability...', 10);
    const fileName = file.name;
    const extension = fileName.split('.').pop()?.toLowerCase() || '';

    let docType = 'other';
    if (extension === 'pdf') docType = 'pdf';
    else if (['jpg', 'jpeg'].includes(extension)) docType = 'jpg';
    else if (extension === 'png') docType = 'png';
    else {
      return { status: 'failed', error: `Unsupported format: .${extension}. SafeDocAI accepts PDF, JPG, JPEG, and PNG.` };
    }

    if (isCancelled && isCancelled()) return { status: 'cancelled' };

    onProgress?.('Reading file bytes locally...', 25);
    let base64Data = '';
    let size = file.size;

    if ('text' in file && typeof file.text === 'string') {
      // Synthetic text payload (tests/demos): ship the text as UTF-8 bytes
      base64Data = btoa(unescape(encodeURIComponent(file.text)));
      size = Math.round(base64Data.length * 0.75);
    } else if (file instanceof File) {
      const buffer = await file.arrayBuffer();
      const bytes = new Uint8Array(buffer);
      let binary = '';
      for (let i = 0; i < bytes.byteLength; i++) {
        binary += String.fromCharCode(bytes[i]);
      }
      base64Data = btoa(binary);
    } else {
      return { status: 'failed', error: 'Unsupported file object' };
    }

    if (isCancelled && isCancelled()) return { status: 'cancelled' };

    onProgress?.('Uploading to local SafeDocAI engine (loopback only)...', 45);

    try {
      const res = await db.uploadDocument({
        fileName,
        fileType: docType,
        base64Data,
        size
      });

      if (isCancelled && isCancelled()) return { status: 'cancelled' };

      if (res.status === 'duplicate') {
        return { status: 'duplicate', document: res.document, error: res.error };
      }

      if (res.document) {
        onProgress?.('Persisting verified fields & vector chunks...', 95);
        onProgress?.('Completed', 100);
        return { status: 'success', document: res.document };
      }

      return { status: 'failed', error: res.error || 'The local engine returned no document.' };
    } catch (e: any) {
      return { status: 'failed', error: e.message || 'Ingestion failed during processing.' };
    }
  }

  /**
   * Honest placeholder for documents whose text could not be recognized.
   * NEVER fabricates fields or sample content.
   */
  public static generateSimulatedOcr(fileName: string, type: string): string {
    const clean = fileName.replace(/\.[^.]+$/, '').replace(/[_\-\.]+/g, ' ');
    return `DOCUMENT: ${fileName.toUpperCase()}
Type: ${type.toUpperCase()}
File: ${clean}

No transcribable text could be recognized in this file by the local OCR engine.
The original file is preserved in the managed store — you can re-run OCR from the document header at any time.`;
  }
}
