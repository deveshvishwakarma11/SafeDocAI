/**
 * API-backed document repository (Phase 1 backend migration).
 * Replaces the old localStorage database: the Node server now owns SQLite +
 * the managed file store (Backend Schema §03/§04). The UI keeps the same `db`
 * handle, but every method is async and hits the local loopback API.
 */
import { StoredDocument, RuntimeDiagnostics } from '../types';

interface ApiErrorBody {
  status?: string;
  error?: string;
  document?: StoredDocument;
}

async function request<T>(path: string, init?: RequestInit & { allowStatuses?: number[] }): Promise<T> {
  const res = await fetch(path, init);
  const body = (await res.json().catch(() => ({}))) as T & ApiErrorBody;

  if (!res.ok && !(init?.allowStatuses || []).includes(res.status)) {
    throw new Error(body?.error || `Request to ${path} failed (HTTP ${res.status})`);
  }
  return body;
}

export class DocumentRepository {
  private static instance: DocumentRepository;

  private constructor() {}

  public static getInstance(): DocumentRepository {
    if (!DocumentRepository.instance) {
      DocumentRepository.instance = new DocumentRepository();
    }
    return DocumentRepository.instance;
  }

  /** All non-superseded documents (fields + chunks included). */
  public async getAllDocuments(): Promise<StoredDocument[]> {
    return request<StoredDocument[]>('/api/documents');
  }

  public async getDocumentById(id: number): Promise<StoredDocument | undefined> {
    return request<StoredDocument | undefined>(`/api/documents/${id}`, { allowStatuses: [404] });
  }

  /**
   * Upload raw file bytes; the server hashes, dedupes, extracts, chunks and
   * persists. Duplicate uploads resolve as `{ status: 'duplicate' }` (HTTP 409).
   */
  public async uploadDocument(payload: {
    fileName: string;
    fileType: string;
    base64Data: string;
    size: number;
  }): Promise<{ status: 'success' | 'duplicate'; document?: StoredDocument; error?: string }> {
    return request<{ status: 'success' | 'duplicate'; document?: StoredDocument; error?: string }>(
      '/api/documents',
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
        allowStatuses: [409]
      }
    );
  }

  /** Re-run OCR/field extraction server-side from the stored original bytes. */
  public async reprocessDocument(id: number): Promise<{ status: string; document?: StoredDocument; error?: string }> {
    return request<{ status: string; document?: StoredDocument; error?: string }>(
      `/api/documents/${id}/reprocess`,
      { method: 'POST' }
    );
  }

  /**
   * "Remove from SafeDocAI" contract: server removes the registry row AND purges
   * managed artifacts (original, stage-2 output, understood JSON).
   */
  public async removeDocument(id: number): Promise<{ success: boolean; removedId: number; error?: string }> {
    return request<{ success: boolean; removedId: number; error?: string }>(`/api/documents/${id}`, {
      method: 'DELETE',
      allowStatuses: [404]
    });
  }

  /** Permanent wipe: every document row + managed artifacts. Returns count removed. */
  public async clearAllDocuments(): Promise<number> {
    const result = await request<{ success: boolean; removed: number }>('/api/documents', {
      method: 'DELETE'
    });
    return result.removed || 0;
  }

  /** Re-chunk all success documents server-side (Settings → Rebuild Vector Index). */
  public async rebuildIndex(): Promise<number> {
    const result = await request<{ success: boolean; rebuilt: number }>('/api/index/rebuild', {
      method: 'POST'
    });
    return result.rebuilt || 0;
  }

  /** Real diagnostics computed by the server (SQLite counts, bytes on disk, engine states). */
  public async runDiagnostics(): Promise<RuntimeDiagnostics> {
    return request<RuntimeDiagnostics>('/api/diagnostics');
  }
}

export const db = DocumentRepository.getInstance();
