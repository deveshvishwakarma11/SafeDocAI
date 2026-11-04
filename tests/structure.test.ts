import { test } from 'node:test';
import assert from 'node:assert/strict';
import { parseStructure, hasStructure, headerFacts, findFacts, scopeLabel } from '../server/structure';
import { answerStructured, detectSemester } from '../server/structured-answer';
import { applyStructuredHeader, extractRealEntities } from '../server/extract';
import { chunkText } from '../server/chunker';
import { askQuestion, type AskDeps } from '../server/ask';

/**
 * Synthetic result sheet (fake student) in the layout PDF text layers produce:
 * cells separated by tabs, repeated sessions and semesters, a wrapped header
 * label, a back-paper session and an empty semester 4.
 */
const T = '\t';
const SHEET = [
  'University Student Result',
  'Institute Code &',
  `Name ${T}: (123 ) DEMO INSTITUTE OF TECHNOLOGY,CITYVILLE`,
  'Course Code &',
  `Name ${T}: (04) B.TECH Branch Code &`,
  `Name ${T}: (10) COMPUTER SCIENCE AND`,
  'ENGINEERING',
  `RollNo ${T}: 2501234500011 ${T}EnrollmentNo ${T}: 250123450099999`,
  `Name ${T}: ASHA RANI ${T}Hindi Name ${T}: आशा रानी`,
  `Father's Name ${T}: RAMESH KUMAR ${T}Gender ${T}: F`,
  'One View Result',
  'Session : 2024-25(REGULAR) Semesters : 1,2 Result : PCP Marks : 1100/1800 COP : BAS103 Audit 1 : Cleared',
  `Semester ${T}: 1 ${T}Even/Odd ${T}: Odd`,
  `Total Subjects ${T}: 3 ${T}Theory Subjects ${T}: 2`,
  `Practical Subjects ${T}: 1 ${T}Total Marks Obt. ${T}: 410`,
  `Result Status ${T}: CP( 1) ${T}SGPA ${T}: 6.10`,
  `Date of Declaration ${T}: 30/06/25`,
  `Code ${T}Name ${T}Type ${T}Internal External Back Paper Grade`,
  `BAS101 Engineering Physics ${T}Theory ${T}23 ${T}45 ${T}-- ${T}B`,
  `BAS103 Engineering Mathematics-I ${T}Theory ${T}23 ${T}09 ${T}-- ${T}F`,
  `BAS151 Engineering Physics Lab ${T}Practical 41 ${T}39 ${T}-- ${T}A`,
  `Semester ${T}: 2 ${T}Even/Odd ${T}: Even`,
  `Result Status ${T}: PCP ${T}SGPA ${T}: 6.05`,
  `Code ${T}Name ${T}Type ${T}Internal External Back Paper Grade`,
  `BAS203 Engineering Mathematics-II ${T}Theory ${T}20 ${T}21 ${T}-- ${T}E`,
  `BVA251 Sports and Yoga ${T}CA ${T}86 ${T}--`,
  '-- 1 of 2 --',
  'Session : 2025-26(BACK) Semesters : 1 Result : PASS Marks : 1212/1800 Audit 1 : Cleared',
  `Semester ${T}: 1 ${T}Even/Odd ${T}: Odd`,
  `Result Status ${T}: CP(0) ${T}SGPA ${T}: 7.00`,
  `Code ${T}Name ${T}Type ${T}Internal External Back Paper Grade`,
  `BAS101 Engineering Physics ${T}Theory ${T}23 ${T}45 ${T}-- ${T}B`,
  `BAS103 Engineering Mathematics-I ${T}Theory ${T}23 ${T}09 ${T}23* ${T}D`,
  `BAS151 Engineering Physics Lab ${T}Practical 41 ${T}39 ${T}-- ${T}A`,
  'Session : 2025-26(REGULAR) Semesters : 3 Result : Marks : 686/1000 COP :',
  `Semester ${T}: 3 ${T}Even/Odd ${T}: Odd`,
  `Result Status ${T}: CP( 0) ${T}SGPA ${T}: 7.12`,
  `Date of Declaration ${T}: 17/02/26`,
  `Code ${T}Name ${T}Type ${T}Internal External Back Paper Grade`,
  `BCS301 Data Structure ${T}Theory ${T}20 ${T}41 ${T}-- ${T}B`,
  `BCS351 Data Structure Lab ${T}Practical 32 ${T}45 ${T}-- ${T}B+`,
  `Semester ${T}: ${T}Even/Odd ${T}: Even`,
  `Result Status ${T}: ${T}SGPA ${T}: 0`,
  'No Result found for the above semester.',
  '-- 2 of 2 --'
].join('\n');

const doc = (): any => {
  const entities = extractRealEntities(SHEET, 'Result Sheet.pdf', 'pdf');
  return {
    id: 1,
    status: 'success',
    file_name: 'Result Sheet.pdf',
    file_type: 'pdf',
    category: entities.category,
    raw_text: SHEET,
    fields: entities.fields,
    chunks: chunkText(1, 'Result Sheet.pdf', SHEET)
  };
};

const deps: AskDeps = { synthesis: false, ollamaReady: async () => false, generate: async () => '' };
const ask = async (q: string) => (await askQuestion(q, [doc()], undefined, deps)).answer || '';

/* ---------------- parser ---------------- */

test('parser reads header facts including wrapped labels and wrapped values', () => {
  const s = parseStructure(SHEET);
  assert.ok(hasStructure(s));
  const h = Object.fromEntries(headerFacts(s).map(f => [f.key, f.value]));
  assert.equal(h['RollNo'], '2501234500011');
  assert.equal(h['EnrollmentNo'], '250123450099999');
  assert.equal(h['Institute Code & Name'], '(123 ) DEMO INSTITUTE OF TECHNOLOGY,CITYVILLE');
  assert.equal(h['Branch Code & Name'], '(10) COMPUTER SCIENCE AND ENGINEERING');
  assert.equal(h["Father's Name"], 'RAMESH KUMAR');
});

test('SGPA is kept per section, never collapsed to the first value', () => {
  const s = parseStructure(SHEET);
  const sgpa = findFacts(s, ['SGPA']).map(f => `${scopeLabel(f.scope)}=${f.value}`);
  assert.deepEqual(sgpa, [
    'Session 2024-25 (REGULAR), Semester 1=6.10',
    'Session 2024-25 (REGULAR), Semester 2=6.05',
    'Session 2025-26 (BACK), Semester 1=7.00',
    'Session 2025-26 (REGULAR), Semester 3=7.12',
    'Session 2025-26 (REGULAR), Semester 4=0'
  ]);
});

test('subject rows parse marks, back paper and grade (and single-mark rows)', () => {
  const s = parseStructure(SHEET);
  const maths = s.rows.filter(r => r.code === 'BAS103');
  assert.equal(maths.length, 2);
  assert.deepEqual([maths[0].internal, maths[0].external, maths[0].grade, maths[0].back], ['23', '09', 'F', '--']);
  assert.deepEqual([maths[1].back, maths[1].grade], ['23*', 'D']);
  const yoga = s.rows.find(r => r.code === 'BVA251')!;
  assert.equal(yoga.marks, '86');
  assert.equal(yoga.grade, undefined);
});

test('an empty semester block gets its number from Even/Odd and is flagged as no result', () => {
  const s = parseStructure(SHEET);
  const sem4 = s.blocks.find(b => b.scope.semester === '4');
  assert.ok(sem4);
  assert.equal(sem4!.noResult, true);
});

test('page numbers follow the "-- n of m --" markers', () => {
  const s = parseStructure(SHEET);
  assert.equal(s.blocks.find(b => b.scope.semester === '3')!.page, 2);
  assert.equal(s.rows.find(r => r.code === 'BAS101')!.page, 1);
});

test('plain prose is not treated as a structured document', () => {
  assert.equal(hasStructure(parseStructure('Dear Sir,\nThis letter confirms your appointment.\nRegards')), false);
});

/* ---------------- answers ---------------- */

test('semester detection in English and Hinglish', () => {
  assert.equal(detectSemester('semester 3 ka sgpa'), '3');
  assert.equal(detectSemester('sem 1 result'), '1');
  assert.equal(detectSemester('3rd semester marks'), '3');
  assert.equal(detectSemester('pehla semester'), '1');
  assert.equal(detectSemester('what is my roll number'), null);
});

test('SGPA of a semester that exists once is a single exact value', async () => {
  const a = await ask('Semester 3 ka SGPA?');
  assert.ok(a.includes('7.12'));
  assert.ok(!a.includes('6.10'));
});

test('SGPA of a semester present in two sessions shows BOTH, labelled', async () => {
  const a = await ask('Semester 1 ka SGPA?');
  assert.ok(a.includes('6.10') && a.includes('2024-25 (REGULAR)'));
  assert.ok(a.includes('7.00') && a.includes('2025-26 (BACK)'));
});

test('Maths-I resolves to Mathematics-I (not -II) with both attempts', async () => {
  const a = await ask('Engineering Maths-I mein kitne marks?');
  assert.ok(a.includes('Engineering Mathematics-I'));
  assert.ok(!a.includes('Mathematics-II'));
  assert.ok(a.includes('External 09') && a.includes('Grade **F**'));
  assert.ok(a.includes('23*') && a.includes('Grade **D**'));
});

test('"Data Structure" returns the theory and the lab', async () => {
  const a = await ask('Data Structure ka grade?');
  assert.ok(a.includes('BCS301') && a.includes('Grade **B**'));
  assert.ok(a.includes('BCS351') && a.includes('Grade **B+**'));
});

test('failed subject is found, with the later back-paper clearance', async () => {
  const a = await ask('Kis subject mein fail hua tha?');
  assert.ok(a.includes('Engineering Mathematics-I'));
  assert.ok(/later cleared|baad me clear/.test(a) && a.includes('**D**')); // Hinglish question => Hinglish wording
  assert.ok(!a.includes('Engineering Physics'));
});

test('semester without a result says so instead of dumping text', async () => {
  const a = await ask('Semester 4 ka result?');
  assert.ok(/no result/i.test(a) || /result nahi/i.test(a));
  assert.ok(a.includes('Semester 4'));
});

test('a semester that is not in the document is reported with the ones that are', async () => {
  const a = await ask('Semester 7 ka result?');
  assert.ok(a.includes('not in this document') || a.includes('nahi hai'));
  assert.ok(a.includes('1, 2, 3, 4'));
});

test('enrollment number is NOT answered with the roll number', async () => {
  const a = await ask('Enrollment number?');
  assert.ok(a.includes('250123450099999'));
  assert.ok(!a.includes('2501234500011'));
});

test('institute, father and roll number are exact and clean', async () => {
  assert.ok((await ask('College ka naam?')).includes('DEMO INSTITUTE OF TECHNOLOGY,CITYVILLE (code 123)'));
  assert.ok((await ask('Father ka naam?')).includes('RAMESH KUMAR'));
  assert.ok((await ask('mera roll number kya hai')).includes('2501234500011'));
});

test('overall result is listed per session', async () => {
  const a = await ask('Mera result kya hai?');
  assert.ok(a.includes('PCP') && a.includes('PASS'));
});

test('unrelated question is not answered from the table', async () => {
  const r = await askQuestion('Who is the prime minister of India?', [doc()], undefined, deps);
  assert.ok(!(r.intent || '').startsWith('structured_'));
  assert.equal(r.is_insufficient_evidence, true);
});

test('answers carry evidence with the real page number', async () => {
  const r = await askQuestion('Semester 3 ka SGPA?', [doc()], undefined, deps);
  assert.equal(r.evidence![0].page_number, 2);
  assert.ok(r.evidence![0].snippet!.includes('7.12'));
});

test('Hinglish questions get a Hinglish heading', async () => {
  const r = await askQuestion('Semester 3 ka SGPA?', [doc()], undefined, deps);
  assert.equal(r.detected_language, 'hinglish');
  assert.ok((r.answer || '').startsWith('Semester 3 ka SGPA'));
});

/* ---------------- flat fields and chunks ---------------- */

test('flat fields come from the header, not from guesses', () => {
  const fields = doc().fields;
  const get = (n: string) => fields.find((f: any) => f.field_name === n)?.field_value;
  assert.equal(get('candidate_name'), 'ASHA RANI');
  assert.equal(get('school_name'), 'DEMO INSTITUTE OF TECHNOLOGY,CITYVILLE');
  assert.equal(get('course_branch'), 'B.TECH · COMPUTER SCIENCE AND ENGINEERING');
  assert.equal(get('enrollment_id'), '250123450099999');
  assert.equal(get('cgpa'), undefined); // repeats per semester: never flattened
  assert.equal(get('result_status'), undefined);
});

test('applyStructuredHeader leaves ordinary documents untouched', () => {
  const fields = [{ field_name: 'cgpa', display_label: 'CGPA', field_value: '8.1' }];
  assert.deepEqual(applyStructuredHeader(fields, 'Just a short note.'), fields);
});

test('chunks follow the sections and carry their heading', () => {
  const chunks = chunkText(1, 'Result Sheet.pdf', SHEET);
  const labels = chunks.map(c => c.text.slice(0, c.text.indexOf(']') + 1));
  assert.ok(labels.includes('[Session 2024-25 (REGULAR), Semester 1]'));
  assert.ok(labels.includes('[Session 2025-26 (BACK), Semester 1]'));
  assert.ok(labels.includes('[Session 2025-26 (REGULAR), Semester 4]'));
  const sem3 = chunks.find(c => c.text.startsWith('[Session 2025-26 (REGULAR), Semester 3]'))!;
  assert.ok(sem3.text.includes('BCS301') && sem3.text.includes('BCS351')); // table not cut in half
  assert.equal(sem3.page_number, 2);
});

/* ---------------- additions: superlatives, counts, session kind ---------------- */

test('highest and lowest SGPA are picked from declared semesters only (empty sem 4 ignored)', async () => {
  const hi = await ask('Mera SGPA sabse zyada kis semester mein hai?');
  assert.ok(hi.includes('7.12') && hi.includes('Semester 3'));
  const lo = await ask('lowest sgpa?');
  assert.ok(lo.includes('6.05'));
  assert.ok(!/\*\*0\*\*/.test(lo)); // the "0" of the undeclared semester is not a score
});

test('"how many subjects have grade A" lists distinct subjects and says entries repeat', async () => {
  const a = await ask('Kitne subjects mein A grade mila?');
  assert.ok(a.includes('Grade **A**'));
  assert.ok(a.includes('BAS151'));
  assert.ok(a.includes('entries'));
});

test('total subjects of a semester', async () => {
  const a = await ask('Semester 1 mein total kitne subjects the?');
  assert.ok(a.includes('**3**') && a.includes('theory 2') && a.includes('practical 1'));
});

test('"back session" restricts the answer to the back session', async () => {
  const a = await ask('2025-26 back session ka result?');
  assert.ok(a.includes('PASS') && a.includes('(BACK)'));
  assert.ok(!a.includes('PCP') && !a.includes('(REGULAR)'));
  const b = await ask('Semester 1 ka SGPA back session?');
  assert.ok(b.includes('7.00') && !b.includes('6.10'));
});
