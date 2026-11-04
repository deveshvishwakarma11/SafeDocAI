import { test } from 'node:test';
import assert from 'node:assert/strict';
import { matchFieldGroup, PARENT_CANDIDATE_FIELDS } from '../src/utils/fieldMatcher';

test("father's name query routes to father_name, not generic candidate_name", () => {
  const m = matchFieldGroup('what is my father name in 10th marksheet?');
  assert.ok(m);
  assert.equal(m!.group.canonical, 'father_name');
});

test("mother's name query routes to mother_name (hinglish)", () => {
  const m = matchFieldGroup('meri maa ka naam kya hai?');
  assert.ok(m);
  assert.equal(m!.group.canonical, 'mother_name');
});

test('parent/guardian query maps to parent_name which expands to father|mother fields', () => {
  const m = matchFieldGroup('what is my parent name?');
  assert.ok(m);
  assert.equal(m!.group.canonical, 'parent_name');
  assert.ok(PARENT_CANDIDATE_FIELDS.includes('father_name'));
  assert.ok(PARENT_CANDIDATE_FIELDS.includes('mother_name'));
});

test("'total marks' is no longer stolen by the amount group (bare 'total' removed)", () => {
  const m = matchFieldGroup('marksheet ke total marks kitne hain?');
  assert.ok(m);
  assert.equal(m!.group.canonical, 'total_marks');
});

test('bill amount query still routes to total_amount', () => {
  const m = matchFieldGroup('what is the total amount due on my electricity bill?');
  assert.ok(m);
  assert.equal(m!.group.canonical, 'total_amount');
});

test('roll number queries are stable (hinglish + english)', () => {
  assert.equal(matchFieldGroup('mera roll number kya hai')!.group.canonical, 'roll_number');
  assert.equal(matchFieldGroup('what is my roll no?')!.group.canonical, 'roll_number');
});

test('generic name query falls to candidate_name', () => {
  const m = matchFieldGroup('kiske naam pe hai ye document');
  assert.ok(m);
  assert.equal(m!.group.canonical, 'candidate_name');
});

test('dob and ifsc queries resolve to their own groups', () => {
  assert.equal(matchFieldGroup('date of birth dikhao')!.group.canonical, 'date_of_birth');
  assert.equal(matchFieldGroup('ifsc code kya hai')!.group.canonical, 'ifsc_code');
});

test('unrelated query matches no field group', () => {
  assert.equal(matchFieldGroup('what is the capital of france'), null);
});

test('longer/more specific keyword beats a shorter generic one', () => {
  // 'bank balance' (12) + 'balance' should beat anything account-ish
  const m = matchFieldGroup('mera bank balance kitna hai');
  assert.ok(m);
  assert.equal(m!.group.canonical, 'available_balance');
});
