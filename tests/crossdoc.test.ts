import { test } from 'node:test';
import assert from 'node:assert/strict';
import { extractRealEntities } from '../server/extract';
import { looksLikePersonName, JUNK_VALUE } from '../server/text-utils';
import { askQuestion, answerFieldAcrossDocuments, type AskDeps } from '../server/ask';
import { chunkText } from '../server/chunker';

const pad = ' This document is issued by the competent authority for official records.'.repeat(5);

function makeDoc(id: number, name: string, category: string, text: string): any {
  const entities = extractRealEntities(text, name, 'pdf');
  return { id, status: 'success', file_name: name, file_type: 'pdf', category: entities.category, raw_text: text, fields: entities.fields.map((f: any, i: number) => ({ ...f, id: i + 1 })), chunks: chunkText(id, name, text) };
}

const deps: AskDeps = { synthesis: false, ollamaReady: async () => false, generate: async () => '' };

const idCard = makeDoc(1, 'ID Card.pdf', 'Identity Document', `STUDENT IDENTITY CARD\nName : ASHA RANI\nDOB : 06-06-2006\nBlood Group : A+\nContact No. : 9876543210\nHome Address : Vill. Demo Po. Sample Distt.\nCityville\n${pad}`);
const form = makeDoc(2, 'Application Form.pdf', 'Application Form', `Application Form\nASHA RANI\n(Full Name of the Applicant)\n(D.O.B)\n06/06/2006 Female\nEnrollment ID : M123X45\n${pad}`);
const ticket = makeDoc(3, 'Train Ticket.pdf', 'Travel Ticket', `Passenger Details\nName : NA Address : NA\nAddress : Indian Railways New Delhi\nTicket Fare 1,200.00\n${pad}`);

test('text helpers: labels and placeholders are not names', () => {
  assert.equal(looksLikePersonName('NA Address'), false);
  assert.equal(looksLikePersonName('Date Of Birth'), false);
  assert.equal(looksLikePersonName('ASHA RANI'), true);
  assert.ok(JUNK_VALUE.test('NA') && JUNK_VALUE.test('null') && JUNK_VALUE.test('--') && !JUNK_VALUE.test('Delhi'));
});

test('value-first forms: "(Full Name of the Applicant)" under the name gives the candidate name', () => {
  assert.equal(form.fields.find((f: any) => f.field_name === 'candidate_name')?.field_value, 'ASHA RANI');
});

test('"NA" is never stored as a field value, and an organisation is not a home address', () => {
  assert.equal(ticket.fields.some((f: any) => /^na$/i.test(String(f.field_value))), false);
  assert.equal(ticket.fields.find((f: any) => f.field_name === 'residential_address'), undefined);
});

test('the same date in two documents is answered once with both sources, not "which document?"', async () => {
  const r = await askQuestion('What is my date of birth?', [idCard, form, ticket], undefined, deps);
  assert.equal(r.intent, 'field_across_documents');
  assert.ok(r.answer!.includes('ID Card.pdf') && r.answer!.includes('Application Form.pdf'));
  assert.ok(r.answer!.includes('06-06-2006') && r.answer!.includes('06/06/2006'));
  assert.ok(!r.answer!.includes('Train Ticket'));
});

test('different values in different documents are listed as different', () => {
  const other = makeDoc(4, 'Other.pdf', 'Identity Document', `STUDENT IDENTITY CARD\nName : MEERA ROY\nDOB : 01-01-1999\n${pad}`);
  const r = answerFieldAcrossDocuments('What is my date of birth?', [idCard, other], 'en');
  assert.ok(r!.answer!.includes('differs'));
  assert.ok(r!.answer!.includes('06-06-2006') && r!.answer!.includes('01-01-1999'));
});

test('the same name in two documents is one answer (not the first document\'s accident)', async () => {
  const r = await askQuestion('Mera name kya hai?', [ticket, form, idCard], undefined, deps);
  assert.equal(r.intent, 'name_lookup');
  assert.ok(r.answer!.includes('ASHA RANI'));
  assert.ok(!/NA Address/.test(r.answer!));
});

test('a single field in only one document is simply answered', async () => {
  const r = await askQuestion('Contact number kya hai?', [idCard, form, ticket], undefined, deps);
  assert.ok(r.answer!.includes('9876543210'));
});

test('no document holds the answer: honest, no invented value', async () => {
  const r = await askQuestion('Who is the prime minister of India?', [idCard, form, ticket], undefined, deps);
  assert.equal(r.is_insufficient_evidence, true);
});
