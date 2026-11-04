import { DocumentChunk, StoredDocument, SourceEvidence } from '../types';

export class LocalVectorEngine {
  private static instance: LocalVectorEngine;

  private constructor() {}

  public static getInstance(): LocalVectorEngine {
    if (!LocalVectorEngine.instance) {
      LocalVectorEngine.instance = new LocalVectorEngine();
    }
    return LocalVectorEngine.instance;
  }

  /**
   * Split raw text into nominal 500-char chunks with 50-char overlap as specified in TRD Section 6.3
   */
  public chunkText(docId: number, fileName: string, text: string): DocumentChunk[] {
    const chunkSize = 500;
    const overlap = 50;
    const chunks: DocumentChunk[] = [];

    // Clean and normalize text
    const cleanText = text.replace(/\r\n/g, '\n').trim();
    if (!cleanText) return [];

    let start = 0;
    let chunkIndex = 0;

    while (start < cleanText.length) {
      let end = Math.min(start + chunkSize, cleanText.length);
      // Try not to cut words in half if possible
      if (end < cleanText.length) {
        const nextSpace = cleanText.lastIndexOf(' ', end);
        if (nextSpace > start + 300) {
          end = nextSpace;
        }
      }

      const chunkSlice = cleanText.substring(start, end).trim();
      if (chunkSlice.length > 20) {
        // Approximate page number (each 1200 characters ~ 1 page)
        const pageNumber = Math.max(1, Math.floor(start / 1200) + 1);

        chunks.push({
          chunk_id: `doc_${docId}_chunk_${chunkIndex}`,
          document_id: docId,
          file_name: fileName,
          page_number: pageNumber,
          chunk_index: chunkIndex,
          text: chunkSlice,
          token_count: Math.ceil(chunkSlice.length / 4)
        });
        chunkIndex++;
      }

      if (end >= cleanText.length) break;
      start = end - overlap;
    }

    return chunks;
  }

  /**
   * Vector representation and Cosine Similarity calculation across stored chunks
   */
  public searchChunks(
    docs: StoredDocument[],
    query: string,
    topK = 3,
    documentIdFilter?: number
  ): { chunk: DocumentChunk; score: number }[] {
    const allChunks: DocumentChunk[] = [];
    for (const doc of docs) {
      if (doc.status !== 'success') continue;
      if (documentIdFilter && doc.id !== documentIdFilter) continue;
      if (doc.chunks && doc.chunks.length > 0) {
        allChunks.push(...doc.chunks);
      }
    }

    if (allChunks.length === 0) return [];

    const queryTokens = this.tokenize(query);
    if (queryTokens.length === 0) return [];

    const scored = allChunks.map(chunk => {
      const chunkTokens = this.tokenize(chunk.text);
      const similarity = this.calculateCosineSimilarity(queryTokens, chunkTokens);

      // Intent bonus if chunk's filename or key topic appears in query
      let bonus = 0;
      const lowerFileName = chunk.file_name.toLowerCase();
      const lowerQuery = query.toLowerCase();

      if (lowerQuery.includes('bill') && lowerFileName.includes('bill')) bonus += 0.25;
      if (lowerQuery.includes('marksheet') && lowerFileName.includes('marksheet')) bonus += 0.25;
      if (lowerQuery.includes('aadhaar') && lowerFileName.includes('aadhaar')) bonus += 0.25;
      if (lowerQuery.includes('passbook') && lowerFileName.includes('passbook')) bonus += 0.25;
      if (lowerQuery.includes('policy') && lowerFileName.includes('policy')) bonus += 0.25;

      return {
        chunk,
        score: similarity + bonus
      };
    });

    // Sort descending by score
    scored.sort((a, b) => b.score - a.score);

    // Return top K with minimum threshold
    return scored.filter(item => item.score > 0.08).slice(0, topK);
  }

  private tokenize(text: string): string[] {
    return text
      .toLowerCase()
      .replace(/[^a-z0-9\u0900-\u097F\s]/gi, ' ')
      .split(/\s+/)
      .filter(w => w.length > 1);
  }

  private calculateCosineSimilarity(queryTokens: string[], docTokens: string[]): number {
    const qFreq = new Map<string, number>();
    const dFreq = new Map<string, number>();

    for (const t of queryTokens) qFreq.set(t, (qFreq.get(t) || 0) + 1);
    for (const t of docTokens) dFreq.set(t, (dFreq.get(t) || 0) + 1);

    let dotProduct = 0;
    for (const [t, count] of qFreq.entries()) {
      if (dFreq.has(t)) {
        dotProduct += count * dFreq.get(t)!;
      }
    }

    let qMag = 0;
    for (const c of qFreq.values()) qMag += c * c;
    qMag = Math.sqrt(qMag);

    let dMag = 0;
    for (const c of dFreq.values()) dMag += c * c;
    dMag = Math.sqrt(dMag);

    if (qMag === 0 || dMag === 0) return 0;
    return dotProduct / (qMag * dMag);
  }
}

export const vectorEngine = LocalVectorEngine.getInstance();
