import { test } from 'node:test';
import assert from 'node:assert/strict';
import { QueryRouter } from '../src/utils/queryRouter';

const doc: any = {
  id: 1,
  status: 'success',
  file_name: 'Result Sheet.pdf',
  file_type: 'pdf',
  category: 'Marksheet / Degree',
  raw_text: 'x',
  fields: [{ field_name: 'cgpa', display_label: 'CGPA / Grade Point', field_value: '6.09', evidence_text: 'SGPA : 6.09', page_number: 1, confidence: 0.9 }],
  chunks: []
};

function stubFetch(handler: (url: string, body: any) => Promise<Response>) {
  const original = globalThis.fetch;
  globalThis.fetch = (async (url: any, init: any) => handler(String(url), JSON.parse(init?.body || '{}'))) as any;
  return () => {
    globalThis.fetch = original;
  };
}

const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });

test('the server answer is shown as-is; the client no longer answers from flat fields first', async () => {
  let sent: any = null;
  const restore = stubFetch(async (_u, body) => {
    sent = body;
    return json({ route: 'exact_field', intent: 'structured_sgpa', answer: 'Semester 3 ka SGPA: 7.12', evidence: [], is_insufficient_evidence: false });
  });
  try {
    // Old behaviour: the client matched "sgpa" to the flat CGPA field (6.09) and never asked the server.
    const r = await QueryRouter.executeQuery('Semester 3 ka SGPA?', { documents: [doc] });
    assert.ok(sent, 'server must be asked');
    assert.equal(sent.query, 'Semester 3 ka SGPA?');
    assert.equal(sent.documents, undefined, 'the library is never uploaded per question');
    assert.equal(r.answer, 'Semester 3 ka SGPA: 7.12');
    assert.ok(!r.answer.includes('6.09'));
  } finally {
    restore();
  }
});

test('a server failure is reported honestly, not replaced by a passage from a second search', async () => {
  const restore = stubFetch(async () => json({ error: 'boom' }, 500));
  try {
    const r = await QueryRouter.executeQuery('who is the prime minister of india', { documents: [doc] });
    assert.equal(r.intent, 'server_error');
    assert.equal(r.is_insufficient_evidence, true);
  } finally {
    restore();
  }
});

test('Stop (abort) stays stopped', async () => {
  const restore = stubFetch(async () => {
    const e = new Error('aborted');
    e.name = 'AbortError';
    throw e;
  });
  try {
    await assert.rejects(() => QueryRouter.executeQuery('mera address kya hai', { documents: [doc] }), { name: 'AbortError' });
  } finally {
    restore();
  }
});
