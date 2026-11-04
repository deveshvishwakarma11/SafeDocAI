// Temporary: measure /api/ask-query latency for the user's actual complaint queries.
const BASE = process.env.BASE || 'http://127.0.0.1:3100';

const docs = await (await fetch(`${BASE}/api/documents`)).json();
console.log(`docs in inventory: ${docs.length} (${docs.map(d => `${d.file_name}:${(d.raw_text || '').length}ch`).join(', ')})`);

async function ask(query) {
  const t0 = Date.now();
  const res = await fetch(`${BASE}/api/ask-query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query })
  });
  const body = await res.json();
  const ms = Date.now() - t0;
  console.log(`\n[${ms}ms] route=${body.route} intent=${body.intent || '-'} evidence=${(body.evidence || []).length}`);
  console.log(`  Q: ${query}`);
  console.log(`  A: ${String(body.answer || '').slice(0, 220).replace(/\n/g, ' | ')}`);
  return ms;
}

const t1 = await ask('you have any kind of document');
const t2 = await ask('whose name is in the document ?');
const t3 = await ask('mera roll number kya hai?');
const t4 = await ask('explain what this scholarship reimbursement document contains');

console.log(`\nsummary ms: inventory=${t1} whose-name=${t2} roll=${t3} open=${t4}`);
