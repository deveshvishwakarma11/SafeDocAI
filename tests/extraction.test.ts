/**
 * Deterministic extractor tests — category detection and the ID-card field set.
 *
 * Real case: a student ID card was labelled "Marksheet / Degree" because its
 * text contains "Roll No", which then ran marksheet-only patterns on it. The
 * card's DOB (OCR'd as `008 : 06-06-2006`), blood group, contact number, course
 * and student ID were never extracted, so honest questions about them returned
 * a generic "here are the details" dump instead of the real values.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { extractRealEntities, cleanPersonName, trimOcrTail } from '../server/extract';

const ID_CARD_FIXTURE = path.resolve('tests', 'fixtures', 'id-card.raw.txt');
const ID_CARD_TEXT = fs.readFileSync(ID_CARD_FIXTURE, 'utf8');

const field = (fields: any[], name: string) => fields.find(f => f.field_name === name);

test('a student ID card is an Identity Document, not a Marksheet', () => {
  const r = extractRealEntities(ID_CARD_TEXT, 'ID Card.pdf', 'pdf');
  assert.equal(r.category, 'Identity Document');
});

test('an actual marksheet is still categorised as Marksheet / Degree', () => {
  const marksheet = [
    'BOARD OF HIGH SCHOOL AND INTERMEDIATE EDUCATION',
    'MARKSHEET — 12th Standard',
    'Roll No: 2407510100067',
    'Passing Year: 2024',
    'Total Marks: 456 / 500'
  ].join('\n');
  const r = extractRealEntities(marksheet, 'marksheet.pdf', 'pdf');
  assert.equal(r.category, 'Marksheet / Degree');
  assert.equal(field(r.fields, 'year_of_passing')?.field_value, '2024');
});

test('ID-card fields are extracted, including a DOB whose label OCR mangled to "008"', () => {
  const { fields } = extractRealEntities(ID_CARD_TEXT, 'ID Card.pdf', 'pdf');
  assert.equal(field(fields, 'date_of_birth')?.field_value, '06-06-2006');
  assert.equal(field(fields, 'blood_group')?.field_value, 'A+');
  assert.equal(field(fields, 'contact_number')?.field_value, '9163791582');
  assert.equal(field(fields, 'student_id')?.field_value, 'KIPM/761/CSE/067');
  assert.equal(field(fields, 'course_branch')?.field_value, 'B.Tech | CSE');
  assert.equal(field(fields, 'session')?.field_value, '2024-2028');
  assert.equal(field(fields, 'roll_number')?.field_value, '2407510100067');
  assert.equal(field(fields, 'candidate_name')?.field_value, 'DEVESH VISHWAKARMA');
  assert.equal(field(fields, 'father_name')?.field_value, 'Mr. Ashok Vishwakarma');
});

test('a session range is never reported as a passing year', () => {
  const { fields } = extractRealEntities(ID_CARD_TEXT, 'ID Card.pdf', 'pdf');
  assert.equal(field(fields, 'year_of_passing'), undefined);
});

test('no field value carries an OCR junk tail', () => {
  const { fields } = extractRealEntities(ID_CARD_TEXT, 'ID Card.pdf', 'pdf');
  for (const f of fields) {
    assert.ok(!/gery|bs 2|is Si/.test(f.field_value), `junk left in ${f.field_name}: ${f.field_value}`);
  }
});

test('values cleaned from OCR noise are reported with lower confidence', () => {
  const { fields } = extractRealEntities(ID_CARD_TEXT, 'ID Card.pdf', 'pdf');
  assert.ok(field(fields, 'candidate_name')!.confidence < 0.99);
  assert.ok(field(fields, 'father_name')!.confidence < 0.99);
});

test('cleanPersonName keeps the name and drops Devanagari / stub noise', () => {
  assert.equal(cleanPersonName('DEVESH VISHWAKARMA gery “जा'), 'DEVESH VISHWAKARMA');
  assert.equal(cleanPersonName('Mr. Ashok Vishwakarma bs 2'), 'Mr. Ashok Vishwakarma');
  assert.equal(cleanPersonName('Rahul Sharma'), 'Rahul Sharma');
});

/**
 * Layouts that print the label in brackets with the value on another line
 * (GATE-style application forms) used to yield zero fields — the text was all
 * there, but every pattern expected "Label: value".
 */
test('bracketed labels with the value on the next line are extracted', () => {
  const gateForm = [
    'GATE 2027 APPLICATION FORM',
    '(Enrollment ID)\tM103B71',
    'E-Signature : Devesh Vishwakarma',
    '(D.O.B)',
    '06/06/2006 Male OBC-NCL',
    'KIPM Collage of Engineering and Technology',
    'Year of Graduation: 2028'
  ].join('\n');
  const { fields } = extractRealEntities(gateForm, 'ApplicationForm.pdf', 'pdf');
  assert.equal(field(fields, 'date_of_birth')?.field_value, '06/06/2006');
  assert.equal(field(fields, 'enrollment_id')?.field_value, 'M103B71');
  assert.equal(
    field(fields, 'school_name')?.field_value,
    'KIPM Collage of Engineering and Technology'
  );
});

/**
 * False-positive guards. A grounded app must not store "retted" as a Roll
 * Number ("reg" inside "regretted") or 57% from "IR recovers only 57% of cost"
 * as a marks percentage.
 */
test('word-boundary labels stop false field values', () => {
  const text = [
    'Electronic Reservation Slip (ERS)',
    'Train No./Name Class',
    'IR recovers only 57% of cost of travel on an average.',
    'Refund is regretted if cancelled.'
  ].join('\n');
  const { fields } = extractRealEntities(text, '4143027140.pdf', 'pdf');
  assert.equal(field(fields, 'roll_number'), undefined);
  assert.equal(field(fields, 'percentage'), undefined);
});

test('a reservation slip is a Travel Ticket, not an Identity Document', () => {
  const text = 'Electronic Reservation Slip (ERS)\nIRCTC e-ticketing\nPNR 4143027140\nPassenger Details';
  assert.equal(extractRealEntities(text, '4143027140.pdf', 'pdf').category, 'Travel Ticket');
});

test('a labelled percentage is still extracted', () => {
  const { fields } = extractRealEntities('Total Marks: 456 / 500\nPercentage: 91.2%', 'm.pdf', 'pdf');
  assert.equal(field(fields, 'percentage')?.field_value, '91.2%');
});

test('trimOcrTail cuts label bleed but keeps a real abbreviation period', () => {
  assert.equal(trimOcrTail('Vill. Dhaska Po. Kauriram Distt.'), 'Vill. Dhaska Po. Kauriram Distt.');
  assert.equal(trimOcrTail('B.Tech | CSE is Si'), 'B.Tech | CSE');
  assert.equal(trimOcrTail('Central Board of Secondary Education'), 'Central Board of Secondary Education');
});
