/**
 * Extractor tests: receipt/invoice fields, category ordering (a shop receipt is
 * not a bank statement) and the conservative OCR text normalizer.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { extractRealEntities, normalizeOcrText } from '../server/extract';

const fieldValue = (
  fields: { field_name: string; field_value: string }[],
  name: string
): string | undefined => fields.find(f => f.field_name === name)?.field_value;

const receiptText = [
  'SHARMA GENERAL STORE',
  '12 MG Road, Indore',
  'Tax Invoice No: INV/2026/0781',
  'Invoice Date: 12/08/2026',
  'Paid by UPI',
  'Sub Total: 1,200.00',
  'CGST 9%: 108.00',
  'Grand Total: 1,416.00',
  'GSTIN 23ABCDE1234F1Z5'
].join('\n');

test('a receipt yields receipt number, date, total, tax and payment method', () => {
  const { category, fields } = extractRealEntities(receiptText, 'shop-receipt.pdf', 'pdf');
  assert.equal(category, 'Receipt / Invoice');
  assert.equal(fieldValue(fields, 'receipt_number'), 'INV/2026/0781');
  assert.equal(fieldValue(fields, 'receipt_date'), '12/08/2026');
  assert.equal(fieldValue(fields, 'total_amount'), '1,416.00');
  assert.equal(fieldValue(fields, 'tax_amount'), '108.00');
  assert.equal(fieldValue(fields, 'payment_method'), 'UPI');
});

test('a strong receipt signal outranks a coincidental bank word', () => {
  const text = `${receiptText}\nAccount: SAVINGS`;
  const { category } = extractRealEntities(text, 'receipt-2026.pdf', 'pdf');
  assert.equal(category, 'Receipt / Invoice');
});

test('a real bank passbook is still a bank statement', () => {
  const text =
    'State Bank of India Passbook. Account Number: 12345678901. IFSC: SBIN0001234. Branch: Indore Main.';
  const { category, fields } = extractRealEntities(text, 'sbi-passbook.pdf', 'pdf');
  assert.equal(category, 'Bank Statement / Passbook');
  assert.equal(fieldValue(fields, 'account_number'), '12345678901');
});

test('a utility bill is not misfiled as a receipt', () => {
  const text =
    'Madhya Pradesh Paschim Kshetra Vidyut Vitaran Company. Electricity Bill. Consumer Number: 1234567890. Units consumed 214 kWh. Total Amount Due: Rs 1,840. Bill No: 88231';
  const { category } = extractRealEntities(text, 'electricity-bill.pdf', 'pdf');
  assert.equal(category, 'Utility Bill');
});

test('a GSTIN is never stored as a tax amount', () => {
  const { fields } = extractRealEntities('GSTIN 23ABCDE1234F1Z5', 'tax.pdf', 'pdf');
  assert.equal(fieldValue(fields, 'tax_amount'), undefined);
});

test('normalizeOcrText strips invisible junk but keeps wording verbatim', () => {
  const raw = 'Name :\u00a0DEVESH\u200b VISHWAKARMA  \r\n\r\n\r\n\r\nRoll No: 067\u00ad\n\nAge:\t22';
  const clean = normalizeOcrText(raw);
  assert.equal(clean, 'Name : DEVESH VISHWAKARMA\n\n\nRoll No: 067\n\nAge:\t22');
  assert.equal(normalizeOcrText('plain text'), 'plain text');
  assert.equal(normalizeOcrText(''), '');
});
