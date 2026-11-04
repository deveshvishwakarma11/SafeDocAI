/**
 * Server-authoritative text chunker (TRD §6.3):
 * nominal 500-char chunks with 50-char overlap, approximate page numbers
 * (~1200 chars/page), token_count ≈ chars/4.
 */
import type { DocumentChunk } from '../src/types/index';
import { hasStructure, parseStructure, scopeLabel, type Structure } from './structure';

export const CHUNK_SIZE = 500;
export const CHUNK_OVERLAP = 50;
export const CHARS_PER_PAGE = 1200;

export function chunkText(docId: number, fileName: string, text: string): DocumentChunk[] {
  const cleanText = (text || '').replace(/\r\n/g, '\n').trim();
  if (!cleanText) return [];

  // Result sheets / statements: one chunk per section (with its heading), never
  // a table cut in half at 500 characters.
  const structure = parseStructure(cleanText);
  if (hasStructure(structure)) return structuredChunks(docId, fileName, structure);

  const chunks: DocumentChunk[] = [];
  let start = 0;
  let chunkIndex = 0;

  while (start < cleanText.length) {
    let end = Math.min(start + CHUNK_SIZE, cleanText.length);
    // Try not to cut words in half when it is cheap to avoid
    if (end < cleanText.length) {
      const nextSpace = cleanText.lastIndexOf(' ', end);
      if (nextSpace > start + 300) {
        end = nextSpace;
      }
    }

    const chunkSlice = cleanText.substring(start, end).trim();
    if (chunkSlice.length > 20) {
      const pageNumber = Math.max(1, Math.floor(start / CHARS_PER_PAGE) + 1);
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
    start = end - CHUNK_OVERLAP;
  }

  return chunks;
}

/** Upper bound for one section chunk; longer sections split at line boundaries. */
export const SECTION_CHUNK_MAX = 1400;

function structuredChunks(docId: number, fileName: string, structure: Structure): DocumentChunk[] {
  const chunks: DocumentChunk[] = [];
  for (const block of structure.blocks) {
    const label = scopeLabel(block.scope);
    const prefix = `[${label || 'Document header'}] `;
    let current: string[] = [];
    let length = 0;
    const flush = () => {
      const body = current.join('\n').trim();
      if (body.length > 20) {
        const index = chunks.length;
        chunks.push({
          chunk_id: `doc_${docId}_chunk_${index}`,
          document_id: docId,
          file_name: fileName,
          page_number: block.page,
          chunk_index: index,
          text: `${prefix}${body}`,
          token_count: Math.ceil((prefix.length + body.length) / 4)
        });
      }
      current = [];
      length = 0;
    };
    for (const line of block.lines) {
      if (length + line.length > SECTION_CHUNK_MAX && current.length > 0) flush();
      current.push(line);
      length += line.length + 1;
    }
    flush();
  }
  return chunks;
}
