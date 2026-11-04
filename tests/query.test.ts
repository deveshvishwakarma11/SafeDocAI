import { test } from 'node:test';
import assert from 'node:assert/strict';
import { answerQueryLocally, answerFromRawText, buildGroundingPrompt, looksSuspect } from '../server/query';

const docWithFields = {
  id: 1,
  file_name: 'College Marksheet.pdf',
  category: 'Marksheet / Degree',
  raw_text: 'Candidate Name: Rahul Sharma Roll No: 2023CS001 Total Marks: 456 / 500',
  fields: [
    {
      field_name: 'roll_number',
      display_label: 'Roll Number',
      field_value: '2023CS001',
      evidence_text: 'Roll No: 2023CS001',
      confidence: 0.99
    }
  ]
};

const docGarbled = {
  id: 2,
  file_name: 'amount.pdf',
  category: 'General Document',
  raw_text: 'BART SiR Yew wfaryfef sires yomedt SAKSHAM *5 IWE Scholarship And Fee Reimbursement',
  fields: []
};

test('inventory question answers instantly without the LLM', () => {
  const r = answerQueryLocally('you have any kind of document', [docWithFields, docGarbled]);
  assert.equal(r.intent, 'document_inventory');
  assert.equal(r.is_insufficient_evidence, false);
  assert.ok(r.answer!.includes('College Marksheet.pdf'));
  assert.ok(r.answer!.includes('amount.pdf'));
});

test('inventory phrasings (english + hinglish) all match', () => {
  for (const q of ['What documents do you have?', 'list your files', 'mere paas kya kya documents hai']) {
    const r = answerQueryLocally(q, [docWithFields]);
    assert.equal(r.intent, 'document_inventory', `expected inventory for: ${q}`);
  }
});

test('content questions are NOT hijacked by the inventory intent', () => {
  for (const q of [
    'explain what this scholarship reimbursement document contains',
    'what is in this document',
    'whose name is in the document ?'
  ]) {
    const r = answerQueryLocally(q, [docGarbled]);
    assert.notEqual(r.intent, 'document_inventory', `inventory misfire for: ${q}`);
  }
});

test('field question on a real doc stays grounded and not insufficient', () => {
  const r = answerQueryLocally('mera roll number kya hai', [docWithFields]);
  assert.equal(r.route, 'exact_field');
  assert.ok(r.answer!.includes('2023CS001'));
  assert.notEqual(r.is_insufficient_evidence, true);
});

test('empty-field doc NEVER dumps raw OCR noise — honest insufficient answer instead', () => {
  const r = answerQueryLocally('whose name is in the document', [docGarbled]);
  assert.equal(r.is_insufficient_evidence, true);
  assert.ok(!r.answer!.includes('BART SiR'), 'garbled OCR must not be presented as details');
  assert.ok(r.answer!.toLowerCase().includes('ocr'));
});

test('garbled OCR is flagged suspect, real prose is not', () => {
  assert.equal(looksSuspect(docGarbled.raw_text), true);
  assert.equal(looksSuspect(docWithFields.raw_text), false);
  assert.equal(looksSuspect(''), true);
});

const docContract = {
  id: 3,
  file_name: 'msa_contract.pdf',
  category: 'Contracts / Agreements',
  raw_text:
    'MASTER SERVICES AGREEMENT. This Agreement is entered into on 14 March 2026 between Greenfield Utilities Pvt Ltd (Provider) and Rahul Sharma (Customer). The Provider shall supply electricity distribution services at 42 Shanti Nagar Indore. Payment is due within thirty days of the invoice date. Either party may terminate this agreement by giving sixty days written notice. The liability of either party shall not exceed the total fees paid in the twelve months preceding the claim. This agreement is governed by the laws of India and the courts at Indore shall have exclusive jurisdiction. Confidential information disclosed under this agreement shall remain confidential for a period of five years after termination.',
  fields: []
};

test('readable field-less doc gets an INSTANT verbatim excerpt answer (no LLM)', () => {
  const r = answerFromRawText('what are the termination terms in this contract?', [docContract]);
  assert.ok(r, 'expected an excerpt answer');
  assert.equal(r!.intent, 'raw_excerpt_answer');
  assert.equal(r!.is_insufficient_evidence, false);
  // The quote must be verbatim from stored raw_text — nothing generated.
  const ev = r!.evidence?.[0];
  assert.ok(ev?.snippet && docContract.raw_text.includes(ev.snippet), 'evidence must be verbatim');
  assert.ok(r!.answer!.includes('terminate') || r!.answer!.includes('termination'));
});

test('garbled/short text is NEVER quoted as an excerpt', () => {
  assert.equal(answerFromRawText('whose name is in this document', [docGarbled]), null);
  assert.equal(answerFromRawText('what does this say', [{ ...docContract, raw_text: 'x' }]), null);
});

test('no query-term overlap → null (falls through to honest not-found)', () => {
  assert.equal(
    answerFromRawText('weather forecast for tomorrow', [docContract]),
    null
  );
  // Queries with only stopwords carry no content terms to match on
  assert.equal(answerFromRawText('kya hai ye', [docContract]), null);
});

/**
 * Regression: every unlisted question used to fall through to the
 * "here are the details" dump, so four different questions returned one
 * identical answer and the real values were never given.
 */
const docIdCard = {
  id: 4,
  file_name: 'ID Card.pdf',
  category: 'Identity Document',
  raw_text: 'Student ID: KIPM/761/CSE/067\nName : DEVESH VISHWAKARMA\nBlood Group: A+ 008 : 06-06-2006',
  fields: [
    {
      field_name: 'blood_group',
      display_label: 'Blood Group',
      field_value: 'A+',
      evidence_text: 'Blood Group: A+',
      confidence: 0.96
    },
    {
      field_name: 'date_of_birth',
      display_label: 'Date of Birth',
      field_value: '06-06-2006',
      evidence_text: 'Blood Group: A+ 008 : 06-06-2006',
      confidence: 0.92
    },
    {
      field_name: 'candidate_name',
      display_label: 'Candidate Name',
      field_value: 'DEVESH VISHWAKARMA',
      evidence_text: 'Name : DEVESH VISHWAKARMA',
      confidence: 0.94
    }
  ]
};

test('"which kind of id it is" answers the document type instead of dumping fields', () => {
  const r = answerQueryLocally('which kind of id it is', [docIdCard]);
  assert.equal(r.intent, 'document_type_profile');
  assert.ok(r.answer!.includes('identity card'));
  assert.ok(!r.answer!.includes('Here are the details'), 'must not be the generic dump');
});

test('"is this a marksheet?" gets a direct verdict', () => {
  const r = answerQueryLocally('kya ye marksheet hai', [docIdCard]);
  assert.equal(r.intent, 'document_type_profile');
  assert.ok(r.answer!.startsWith('Nahi'));
});

test('blood group and date of birth questions return the real values', () => {
  const blood = answerQueryLocally('what is his blood group', [docIdCard]);
  assert.equal(blood.intent, 'exact_lookup_blood_group');
  assert.ok(blood.answer!.includes('A+'));

  const dob = answerQueryLocally('date of birth of the candidate', [docIdCard]);
  assert.equal(dob.intent, 'exact_lookup_date_of_birth');
  assert.ok(dob.answer!.includes('06-06-2006'));
});

test('stored-transcription fallback answers a field the old extractor missed', () => {
  // docIdCard has no student_id field (stored before the extractor change) —
  // the answer must quote the real line, not dump unrelated fields.
  const r = answerQueryLocally('what is my student id', [docIdCard]);
  assert.equal(r.intent, 'raw_line_lookup_student_id');
  assert.ok(r.answer!.includes('KIPM/761/CSE/067'));
});

test('an extracted field wins over the transcription fallback', () => {
  const withField = {
    ...docIdCard,
    fields: [
      ...docIdCard.fields,
      {
        field_name: 'student_id',
        display_label: 'Student ID',
        field_value: 'KIPM/761/CSE/067',
        evidence_text: 'Student ID: KIPM/761/CSE/067',
        confidence: 0.98
      }
    ]
  };
  const r = answerQueryLocally('what is my student id', [withField]);
  assert.equal(r.intent, 'exact_lookup_student_id');
});

test('an unanswerable question is honest, not a field dump', () => {
  const r = answerQueryLocally('what is his favourite colour', [docIdCard]);
  assert.equal(r.is_insufficient_evidence, true);
  assert.ok(!r.answer!.includes('DEVESH'), 'must not leak the field dump');
});

test('explicit "show all details" still returns the full field summary', () => {
  const r = answerQueryLocally('show all details', [docIdCard]);
  assert.equal(r.is_insufficient_evidence, false);
  assert.ok(r.answer!.includes('Blood Group'));
  assert.ok(r.answer!.includes('DEVESH VISHWAKARMA'));
});

/**
 * One document, one chat: with a scope id, content answers must come from that
 * file only — the bug was an ID-card question answered from a train ticket.
 */
const docTicket = {
  id: 5,
  file_name: 'Train Ticket.pdf',
  category: 'Travel Ticket',
  raw_text:
    'Electronic Reservation Slip (ERS). PNR 4567891234 Train No. 12345 Coach B4 Seat 36. Passenger Name: DEVESH VISHWAKARMA. Departure 14:20 from Kanpur Central. Booking Amount Rs 845. IR recovers only 57% of cost.',
  fields: [
    {
      field_name: 'candidate_name',
      display_label: 'Candidate Name',
      field_value: 'DEVESH VISHWAKARMA',
      evidence_text: 'Passenger Name: DEVESH VISHWAKARMA',
      confidence: 0.95
    }
  ]
};

test('scoped chat answers ONLY from the open document', () => {
  const r = answerQueryLocally('what is his blood group', [docTicket, docIdCard], docIdCard.id);
  assert.equal(r.intent, 'exact_lookup_blood_group');
  assert.ok(r.answer!.includes('A+'));
  assert.ok(r.answer!.includes('ID Card.pdf'), 'must name the scoped file');
});

test('scoped chat reports a missing field for THAT document, not another file', () => {
  const r = answerQueryLocally('what is his blood group', [docTicket, docIdCard], docTicket.id);
  assert.equal(r.is_insufficient_evidence, true);
  assert.ok(r.answer!.includes('Train Ticket.pdf'));
  assert.ok(!r.answer!.includes('A+'), 'must not borrow the ID card value');
});

/**
 * Regression: "<qualifier> name" is not a request for the holder's name, and a
 * single-field question is not a request for the whole field dump.
 */
test('"principal name" is not answered with the student name', () => {
  const r = answerQueryLocally('what is the principal name', [docIdCard], docIdCard.id);
  assert.equal(r.is_insufficient_evidence, true);
  assert.ok(!r.answer!.includes('DEVESH'), 'must not quote the unrelated name line');
});

test('"what is my percentage" on a doc without marks is not a field dump', () => {
  const r = answerQueryLocally('what is my percentage', [docIdCard], docIdCard.id);
  assert.equal(r.is_insufficient_evidence, true);
  assert.ok(!r.answer!.includes('10 verified') && !/Document Summary/.test(r.answer!));
});

test('an Enrollment ID question reads the enrollment_id field, not Student ID', () => {
  const withEnrollment = {
    ...docWithFields,
    fields: [
      ...docWithFields.fields,
      {
        field_name: 'enrollment_id',
        display_label: 'Enrollment ID',
        field_value: 'M103B71',
        evidence_text: '(Enrollment ID) M103B71',
        confidence: 0.94
      }
    ]
  };
  const r = answerQueryLocally('what is the enrollment id', [withEnrollment], withEnrollment.id);
  assert.equal(r.intent, 'exact_lookup_enrollment_id');
  assert.ok(r.answer!.includes('M103B71'));
  assert.equal(r.is_insufficient_evidence, false);
});

test('"aadhaar number kya hai" asks for the number, not a type verdict', () => {
  const r = answerQueryLocally('aadhaar number kya hai', [docIdCard], docIdCard.id);
  assert.notEqual(r.intent, 'document_type_profile');
  assert.equal(r.is_insufficient_evidence, true);
});

test('"what is the receipt number" is a field question, not a document-type quiz', () => {
  const r = answerQueryLocally('what is the receipt number', [docTicket], docTicket.id);
  assert.notEqual(r.intent, 'document_type_profile');
});

const docIdCardFull = {
  ...docIdCard,
  raw_text:
    'KIPM COLLEGE OF ENGINEERING & TECHNOLOGY\nStudent ID: KIPM/761/CSE/067\nName : DEVESH VISHWAKARMA\nBlood Group: A+ 008 : 06-06-2006\nSession: 2024-2028'
};

test('a labelled field that DO exist is answered from the scoped file', () => {
  const r = answerQueryLocally('what is the college name', [docIdCardFull], docIdCardFull.id);
  assert.equal(r.is_insufficient_evidence, false);
  assert.ok(r.answer!.includes('KIPM COLLEGE OF ENGINEERING'));
});

test('scope does not hide the library: inventory still lists every document', () => {
  const r = answerQueryLocally('what documents do you have', [docTicket, docIdCard], docIdCard.id);
  assert.equal(r.intent, 'document_inventory');
  assert.ok(r.answer!.includes('Train Ticket.pdf'));
  assert.ok(r.answer!.includes('ID Card.pdf'));
});

test('an unknown scope id degrades to the whole library instead of "no documents"', () => {
  const r = answerQueryLocally('what is his blood group', [docIdCard], 999);
  assert.ok(r.answer!.includes('A+'));
});

test('grounding prompt stays bounded even for a huge document', () => {
  const hugeDoc = { ...docWithFields, raw_text: 'x'.repeat(500_000) };
  const prompt = buildGroundingPrompt('what is this about?', [hugeDoc]);
  assert.ok(prompt.length < 10_000, `prompt too large: ${prompt.length} chars`);
  assert.ok(prompt.includes('excerpt truncated'));
});

/* ------------------------------------------------------------------ */
/* Whole-document questions and false-positive field routing           */
/* ------------------------------------------------------------------ */

// Exactly the transcription the user reported: a real ID card with OCR
// noise, header missing, and no "college" word anywhere.
const docIdCardNoSchool = {
  id: 7,
  file_name: 'ID Card.pdf',
  category: 'Identity Document',
  raw_text:
    'PLOT No. BL 1 & 2, SECTOR-9, GIDA, GORAKHPUR-273 209\nStudent ID: KIPM/761/CSE/067 Session: 2024-2028\n| § Name : DEVESH VISHWAKARMA gery “जा\nFather\'s Name : Mr. Ashok Vishwakarma bs 2\nCourse/Branch : B.Tech | CSE is Si\nRoll No : 2407510100067 [| yn\nBlood Group: A+ 008 : 06-06-2006 he. RE',
  fields: [
    { field_name: 'candidate_name', display_label: 'Candidate Name', field_value: 'DEVESH VISHWAKARMA', evidence_text: 'Name : DEVESH VISHWAKARMA', confidence: 0.98 },
    { field_name: 'student_id', display_label: 'Student ID', field_value: 'KIPM/761/CSE/067', evidence_text: 'Student ID: KIPM/761/CSE/067', confidence: 0.97 },
    { field_name: 'session', display_label: 'Session / Validity', field_value: '2024-2028', evidence_text: 'Session: 2024-2028', confidence: 0.95 }
  ]
};

test('"what is written in this document" summarises instead of "not recorded"', () => {
  for (const q of [
    'what is written in this document',
    'what does this document say',
    'what is in this document',
    'isme kya likha hai',
    'is document me kya kya hai',
    'yeh document kis baare me hai'
  ]) {
    const r = answerQueryLocally(q, [docIdCardNoSchool], docIdCardNoSchool.id);
    assert.equal(r.is_insufficient_evidence, false, `falsely insufficient: ${q}`);
    assert.ok(
      /document_details|raw_ocr_summary/.test(String(r.intent)),
      `expected a summary for "${q}", got intent=${r.intent}`
    );
    assert.ok(r.answer!.includes('DEVESH VISHWAKARMA'), `summary lost fields for: ${q}`);
  }
});

test('"what is this document about" is answered, not refused', () => {
  const r = answerQueryLocally('what is this document about', [docIdCardNoSchool], docIdCardNoSchool.id);
  assert.equal(r.is_insufficient_evidence, false);
  // Either the field summary or the document-type profile is a real answer.
  assert.ok(/document_details|raw_ocr_summary|document_type_profile/.test(String(r.intent)));
});

test('"is this a valid id card" is not answered as a session/validity lookup', () => {
  const r = answerQueryLocally('is this a valid id card', [docIdCardNoSchool], docIdCardNoSchool.id);
  assert.ok(!String(r.intent).includes('session'), `session misfire: ${r.intent}`);
  // Listing the document's real fields is fine; claiming a validity value is not.
  assert.ok(!r.answer!.includes('2024-2028'), 'answered a validity question nobody asked');
});

test('a missing field never quotes an unrelated line as the answer', () => {
  const r = answerQueryLocally('college name kya hai', [docIdCardNoSchool], docIdCardNoSchool.id);
  assert.equal(r.is_insufficient_evidence, true);
  assert.ok(!/Closest transcribed line|Sabse kareeb/.test(r.answer!), 'quoted a random line for a missing field');
  assert.ok(!r.answer!.includes('DEVESH VISHWAKARMA'), 'student name presented as college answer');
});
