/**
 * Minimal, safe markdown renderer for chat answers.
 *
 * Answer text produced by the grounded routers contains **bold**, `code`,
 * ### heading markers and bullet lines. Rendering it as plain text showed raw
 * asterisks to the user. This escapes ALL HTML first, then promotes only the
 * few markup tokens we emit ourselves — so document-derived content can never
 * inject markup into the chat.
 */

export function escapeHtml(text: string): string {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

export function renderChatMarkdown(source: string): string {
  return escapeHtml(source)
    // "### Heading" → heading text (bold), marker dropped
    .replace(/^[ \t]*#{1,6}[ \t]+/gm, '')
    // **bold** → <strong>
    .replace(/\*\*([^*\n]+)\*\*/g, '<strong>$1</strong>')
    // `code` → <code>
    .replace(/`([^`\n]+)`/g, '<code>$1</code>');
}
