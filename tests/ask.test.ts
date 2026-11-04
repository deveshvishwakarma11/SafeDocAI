import { test } from 'node:test';
import assert from 'node:assert/strict';
import { retrieveChunks, contentTerms, isUsableChunk } from '../server/retrieval';
import { checkGrounding, criticalTokens, buildAnswerPrompt, detectAnswerLang } from '../server/answer';
import { askQuestion, type AskDeps } from '../server/ask';

const longText = (s: string) => `${s} ${'This document is issued by the competent authority for official records. '.repeat(6)}`;

const billDoc = {
  id: 1,
  status: 'success',
  file_name: 'Electricity Bill.pdf',
  category: 'Utility Bill',
  raw_text: '',
  fields: [],
  chunks: [
    {
      chunk_id: 'doc_1_chunk_0',
      document_id: 1,
      file_name: 'Electricity Bill.pdf',
      page_number: 1,
      chunk_index: 0,
      text: longText('Consumer Name: Ramesh Kumar. Billing period March 2026. Total amount payable Rs 1,250.00 due by 15/04/2026.')
    }
  ]
};

const policyDoc = {
  id: 2,
  status: 'success',
  file_name: 'Health Insurance Policy.pdf',
  category: 'Insurance Policy',
  raw_text: '',
  fields: [],
  chunks: [
    {
      chunk_id: 'doc_2_chunk_0',
      document_id: 2,
      file_name: 'Health Insurance Policy.pdf',
      page_number: 1,
      chunk_index: 0,
      text: longText('The policy provides hospitalisation coverage up to Rs 5,00,000 per year including cashless treatment and day care procedures.')
    }
  ]
};

const hindiDoc = {
  id: 3,
  status: 'success',
  file_name: 'Form.pdf',
  category: 'General Document',
  raw_text: '',
  fields: [],
  chunks: [
    {
      chunk_id: 'doc_3_chunk_0',
      document_id: 3,
      file_name: 'Form.pdf',
      page_number: 1,
      chunk_index: 0,
      text: 'आवेदक का नाम देवेश विश्वकर्मा पिता का नाम राम विश्वकर्मा जन्म तिथि 06/06/2006 पता ग्राम धस्का जनपद गोरखपुर'
    }
  ]
};

function deps(overrides: Partial<AskDeps> = {}): AskDeps {
  return {
    synthesis: true,
    ollamaReady: async () => true,
    generate: async () => 'The total amount payable is Rs 1,250.00.',
    ...overrides
  };
}

/* ---------------- retrieval ---------------- */

test('content terms drop English and Hinglish filler words', () => {
  assert.deepEqual(contentTerms('what is my total amount'), ['total', 'amount']);
  assert.deepEqual(contentTerms('mera roll number kya hai'), ['roll', 'number']);
});

test('retrieval ranks the chunk that answers the question first', () => {
  const hits = retrieveChunks('what coverage does the policy provide', [billDoc, policyDoc]);
  assert.ok(hits.length > 0);
  assert.equal(hits[0].doc.id, 2);
});

test('retrieval matches Devanagari terms', () => {
  const hits = retrieveChunks('जन्म तिथि क्या है', [billDoc, hindiDoc]);
  assert.equal(hits[0]?.doc.id, 3);
});

test('retrieval returns nothing when no content term overlaps', () => {
  assert.deepEqual(retrieveChunks('who won the cricket world cup', [billDoc, policyDoc]), []);
});

test('retrieval skips non-success documents and garbled chunks', () => {
  const failed = { ...billDoc, id: 9, status: 'failed' };
  assert.deepEqual(retrieveChunks('total amount payable', [failed]), []);
  assert.equal(isUsableChunk('~~ ## @@ ** ^^ && %% $$ !! ?? ~~ ## @@ ** ^^'), false);
});

/* ---------------- grounding validator ---------------- */

test('critical tokens capture ids, dates and amounts only', () => {
  const t = criticalTokens('Roll No 2023CS001, DOB 06/06/2006, paid Rs 1,250.00 on page 2');
  assert.ok(t.includes('2023cs001'));
  assert.ok(t.includes('06062006'));
  assert.ok(t.includes('125000'));
  assert.ok(!t.includes('2')); // single digit "page 2" is not critical
});

test('answer with values found in the context is accepted', () => {
  const hits = retrieveChunks('total amount payable', [billDoc]);
  assert.deepEqual(checkGrounding('Total payable is Rs 1,250.00 due 15/04/2026.', hits), { ok: true });
});

test('answer with an invented value is rejected', () => {
  const hits = retrieveChunks('total amount payable', [billDoc]);
  const v = checkGrounding('Total payable is Rs 9,999.00.', hits);
  assert.equal(v.ok, false);
  if (!v.ok) assert.equal(v.reason, 'ungrounded_value');
});

test('generic "typically contains" knowledge is rejected, NOT_FOUND is honoured', () => {
  const hits = retrieveChunks('total amount payable', [billDoc]);
  const generic = checkGrounding('A bill typically contains the consumer name and amount.', hits);
  assert.equal(generic.ok, false);
  if (!generic.ok) assert.equal(generic.reason, 'generic_knowledge');
  const nf = checkGrounding('NOT_FOUND', hits);
  assert.equal(nf.ok, false);
  if (!nf.ok) assert.equal(nf.reason, 'model_not_found');
});

test('prompt stays small and contains only retrieved chunks, not whole documents', () => {
  const hits = retrieveChunks('total amount payable', [billDoc, policyDoc]);
  const prompt = buildAnswerPrompt('total amount payable', hits, 'en');
  assert.ok(prompt.length < 2200, `prompt too large: ${prompt.length}`);
  assert.ok(prompt.includes('NOT_FOUND'));
  assert.ok(!prompt.includes('JSON'));
});

test('language detection for prompt', () => {
  assert.equal(detectAnswerLang('mera bill kitna hai'), 'hinglish');
  assert.equal(detectAnswerLang('what is the bill amount'), 'en');
});

/* ---------------- answer ladder ---------------- */

const billWithFields = {
  ...billDoc,
  fields: [
    { field_name: 'consumer_name', display_label: 'Consumer Name', field_value: 'Ramesh Kumar', evidence_text: 'Consumer Name: Ramesh Kumar', confidence: 0.9 },
    { field_name: 'amount_due', display_label: 'Amount Due', field_value: 'Rs 1,250.00', evidence_text: 'Total amount payable Rs 1,250.00', confidence: 0.9 }
  ]
};

const secondBill = {
  ...billDoc,
  id: 4,
  file_name: 'Water Bill.pdf',
  chunks: [
    {
      chunk_id: 'doc_4_chunk_0',
      document_id: 4,
      file_name: 'Water Bill.pdf',
      page_number: 1,
      chunk_index: 0,
      text: longText('Water charges for the quarter. Total amount payable Rs 480.00 due by 10/05/2026.')
    }
  ]
};

test('content (not file name) picks the document: no "which document?" when only one matches', async () => {
  const r = await askQuestion('is cashless treatment available', [billDoc, policyDoc], undefined,
    deps({ generate: async () => 'Yes, cashless treatment is covered.' }));
  assert.equal(r.intent, 'llm_grounded_answer');
  assert.equal(r.evidence![0].file_name, 'Health Insurance Policy.pdf');
});

test('grounded LLM answer carries our own evidence', async () => {
  const r = await askQuestion('how much is the total amount payable', [billDoc, policyDoc], undefined, deps());
  assert.equal(r.intent, 'llm_grounded_answer');
  assert.equal(r.is_insufficient_evidence, false);
  assert.ok(r.answer!.includes('1,250.00'));
  assert.equal(r.evidence![0].file_name, 'Electricity Bill.pdf');
});

test('genuinely ambiguous question asks, but only between the matching files', async () => {
  const r = await askQuestion('what is the total amount payable', [billDoc, secondBill, policyDoc], undefined, deps());
  assert.equal(r.intent, 'which_document');
  const names = (r.clarification_options || []).map(o => o.label).sort();
  assert.deepEqual(names, ['Electricity Bill.pdf', 'Water Bill.pdf']);
});

test('hallucinated LLM answer is NOT shown; user gets the document\'s own words', async () => {
  const r = await askQuestion('how much is the total amount payable', [billDoc, policyDoc], undefined,
    deps({ generate: async () => 'The total payable is Rs 77,777.00.' }));
  assert.equal(r.intent, 'retrieved_passage');
  assert.ok(!(r.answer || '').includes('77,777'));
  assert.ok(r.answer!.includes('1,250.00'));
});

test('model says NOT_FOUND => honest "could not verify", no guess', async () => {
  const r = await askQuestion('how much is the total amount payable', [billDoc, policyDoc], undefined,
    deps({ generate: async () => 'NOT_FOUND' }));
  assert.equal(r.intent, 'not_clear_in_document');
  assert.equal(r.is_insufficient_evidence, true);
});

test('Ollama down => closest verbatim passage with source', async () => {
  const r = await askQuestion('how much is the total amount payable', [billDoc, policyDoc], undefined,
    deps({ ollamaReady: async () => false }));
  assert.equal(r.intent, 'retrieved_passage');
  assert.ok(r.answer!.includes('Electricity Bill.pdf'));
  assert.ok(r.answer!.includes('1,250.00'));
});

test('LLM timeout/error falls back to the closest passage', async () => {
  const r = await askQuestion('how much is the total amount payable', [billDoc, policyDoc], undefined,
    deps({ generate: async () => { throw new Error('timeout'); } }));
  assert.equal(r.intent, 'retrieved_passage');
});

test('synthesis disabled never calls the model', async () => {
  let called = false;
  const r = await askQuestion('how much is the total amount payable', [billDoc, policyDoc], undefined,
    deps({ synthesis: false, generate: async () => { called = true; return 'x'; } }));
  assert.equal(called, false);
  assert.equal(r.intent, 'retrieved_passage');
});

test('already-aborted request never reaches the model', async () => {
  let called = false;
  const ac = new AbortController();
  ac.abort();
  await askQuestion('how much is the total amount payable', [billDoc, policyDoc], undefined,
    deps({ generate: async () => { called = true; return 'x'; } }), ac.signal);
  assert.equal(called, false);
});

test('unrelated question gets an honest "nothing found", not "which document?"', async () => {
  const r = await askQuestion('who is the prime minister of india', [billDoc, policyDoc], undefined, deps());
  assert.equal(r.intent, 'no_match_in_library');
  assert.equal(r.is_insufficient_evidence, true);
});

test('"explain this bill" is summarised by the model from fields + chunks, verified', async () => {
  const r = await askQuestion('explain what this bill contains', [billWithFields, policyDoc], undefined,
    deps({ generate: async () => '- Consumer: Ramesh Kumar\n- Total payable: Rs 1,250.00' }));
  assert.equal(r.intent, 'llm_document_summary');
  assert.equal(r.evidence![0].file_name, 'Electricity Bill.pdf');
});

test('summary without a model falls back to verified fields only', async () => {
  const r = await askQuestion('explain what this bill contains', [billWithFields, policyDoc], undefined,
    deps({ synthesis: false }));
  assert.equal(r.intent, 'fields_summary');
  assert.ok(r.answer!.includes('Ramesh Kumar'));
});

test('a summary that invents a value is rejected in favour of verified fields', async () => {
  const r = await askQuestion('explain what this bill contains', [billWithFields, policyDoc], undefined,
    deps({ generate: async () => '- Total payable: Rs 55,555.00' }));
  assert.equal(r.intent, 'fields_summary');
  assert.ok(!r.answer!.includes('55,555'));
});

test('no documents => honest empty state', async () => {
  const r = await askQuestion('anything', [], undefined, deps());
  assert.equal(r.route, 'out_of_scope');
  assert.equal(r.is_insufficient_evidence, true);
});
