import { test } from 'node:test';
import assert from 'node:assert/strict';
import { chunkText, CHUNK_SIZE } from '../server/chunker';

test('empty or whitespace-only text yields no chunks', () => {
  assert.deepEqual(chunkText(1, 'a.pdf', ''), []);
  assert.deepEqual(chunkText(1, 'a.pdf', '   \n  '), []);
});

test('long text chunks at ~500 chars with no content loss', () => {
  const words = Array.from({ length: 400 }, (_, i) => `word${i}`).join(' ');
  const chunks = chunkText(7, 'test.pdf', words);

  assert.ok(chunks.length > 1);
  for (const c of chunks) {
    assert.ok(c.text.length <= CHUNK_SIZE + 1, `chunk ${c.chunk_index} too long: ${c.text.length}`);
    assert.ok(c.text.length > 20);
    assert.equal(c.document_id, 7);
    assert.equal(c.file_name, 'test.pdf');
    assert.equal(c.chunk_id, `doc_7_chunk_${c.chunk_index}`);
    assert.ok(c.page_number >= 1);
    assert.ok((c.token_count || 0) > 0);
  }

  const joined = chunks.map(c => c.text).join(' ');
  for (const probe of ['word0', 'word200', 'word399']) {
    assert.ok(joined.includes(probe), `${probe} missing from chunk stream`);
  }
});

test('chunk indices are sequential from 0', () => {
  const chunks = chunkText(1, 'a.pdf', 'x'.repeat(1300));
  assert.ok(chunks.length >= 2);
  chunks.forEach((c, i) => assert.equal(c.chunk_index, i));
});

test('short text yields a single page-1 chunk', () => {
  const chunks = chunkText(2, 'b.pdf', 'Short but valid document text for testing.');
  assert.equal(chunks.length, 1);
  assert.equal(chunks[0].page_number, 1);
});

test('page numbers increase across long documents (~1200 chars/page)', () => {
  const text = Array.from({ length: 120 }, (_, i) => `sentence number ${i} with some padding words`).join(' ');
  const chunks = chunkText(3, 'long.pdf', text);
  const maxPage = Math.max(...chunks.map(c => c.page_number));
  assert.ok(maxPage >= 2, `expected multiple pages, got max page ${maxPage}`);
});
