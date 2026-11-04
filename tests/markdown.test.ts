/** Chat markdown renderer: escaping first, then only our own tokens. */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { escapeHtml, renderChatMarkdown } from '../src/utils/markdown';

test('escapeHtml neutralizes markup characters', () => {
  assert.equal(escapeHtml('<img src=x onerror=alert(1)>'), '&lt;img src=x onerror=alert(1)&gt;');
  assert.equal(escapeHtml('a & b'), 'a &amp; b');
});

test('**bold** becomes <strong>, not literal asterisks', () => {
  const html = renderChatMarkdown('Amount is **₹1,840** due');
  assert.ok(html.includes('<strong>₹1,840</strong>'), html);
  assert.ok(!html.includes('**'), html);
});

test('backticks become <code>', () => {
  const html = renderChatMarkdown('Roll Number is `2023CS001`');
  assert.ok(html.includes('<code>2023CS001</code>'), html);
});

test('### heading markers are dropped, content kept', () => {
  const html = renderChatMarkdown('### **Verified Extracted Information:**\n• Roll: 1');
  assert.ok(!html.includes('#'), html);
  assert.ok(html.includes('<strong>Verified Extracted Information:</strong>'), html);
});

test('document-derived HTML is escaped before markup promotion', () => {
  const html = renderChatMarkdown('<b>x</b> **real**');
  assert.ok(html.includes('&lt;b&gt;x&lt;/b&gt;'), html);
  assert.ok(html.includes('<strong>real</strong>'), html);
  assert.ok(!/<b>/.test(html), html);
});

test('newlines survive for whitespace-pre-wrap layout', () => {
  const html = renderChatMarkdown('line1\nline2');
  assert.equal(html, 'line1\nline2');
});
