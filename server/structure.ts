/**
 * Structure-aware parsing for documents that repeat sections and tables
 * (university result sheets, statements, multi-period reports).
 *
 * Why: a flat "label -> single value" model cannot represent "SGPA" when it
 * appears once per semester, or a subject grade that appears in two sessions.
 * It silently returned the FIRST value, which is a confident wrong answer.
 *
 * This module is pure and deterministic (no model). It reads the extracted
 * text (cells separated by tabs, as PDF text layers produce) and returns:
 *   - facts: "Label : Value" pairs, each tagged with its section scope
 *            (session / semester)
 *   - rows:  subject-table rows (code, name, type, marks, back paper, grade)
 *   - blocks: the text of each section, used for section-aware chunking
 */

export interface Scope {
  session?: string;
  semester?: string;
}

export interface Fact {
  key: string;
  value: string;
  scope: Scope;
  line: string;
  page: number;
}

export interface Row {
  code: string;
  name: string;
  type: string;
  internal?: string;
  external?: string;
  /** Single mark (e.g. continuous-assessment subjects with one score). */
  marks?: string;
  back?: string;
  grade?: string;
  scope: Scope;
  line: string;
  page: number;
}

export interface Block {
  scope: Scope;
  lines: string[];
  noResult: boolean;
  page: number;
}

export interface Structure {
  facts: Fact[];
  rows: Row[];
  blocks: Block[];
}

const PAGE_MARKER = /^--\s*\d+\s+of\s+\d+\s*--$/;
const SESSION_LINE = /^Session\s*:\s*(\S+)/i;
const ROW_LINE = /^([A-Z]{2,5}\d{2,4}[A-Z]?)\s+(.+?)\s+(Theory|Practical|CA)\s+(.*)$/;
const NO_RESULT = /no\s+result\s+found/i;

/** "Total Marks Obt." -> "totalmarksobt" (lookup key). */
export function normKey(key: string): string {
  return key.toLowerCase().replace(/[^a-z0-9]+/g, '');
}

export function scopeLabel(scope: Scope): string {
  const parts: string[] = [];
  if (scope.session) parts.push(`Session ${scope.session}`);
  if (scope.semester) parts.push(`Semester ${scope.semester}`);
  return parts.join(', ');
}

function normSession(raw: string): string {
  return raw.replace(/\s*\(/, ' (').trim(); // "2025-26(BACK)" -> "2025-26 (BACK)"
}

/**
 * Pairs from one tab-separated line: `Label \t: Value \tLabel2 \t: Value2`.
 * A label may wrap onto the previous line ("Institute Code &" / "Name \t: ..."),
 * and a value may end with the prefix of the next wrapped label.
 */
function parseCells(
  line: string,
  pendingPrefix: string | null
): { pairs: Array<[string, string]>; trailingPrefix: string | null } {
  const cells = line.split('\t').map(c => c.trim());
  const pairs: Array<[string, string]> = [];
  let trailingPrefix: string | null = null;
  let prefix = pendingPrefix;
  let i = 0;
  while (i < cells.length) {
    const next = cells[i + 1];
    if (cells[i] && next !== undefined && next.startsWith(':')) {
      let label = cells[i];
      if (prefix && i === 0) {
        label = `${prefix} ${label}`;
        prefix = null;
      }
      let value = next.slice(1).trim();
      const wrapped = value.match(/^(.*?)\s+([A-Z][A-Za-z ]*&)$/);
      if (wrapped) {
        value = wrapped[1].trim();
        trailingPrefix = wrapped[2].trim();
      }
      pairs.push([label, value]);
      i += 2;
    } else {
      i++;
    }
  }
  return { pairs, trailingPrefix };
}

/** "Result : PCP Marks : 1198/1800 COP : BAS103 Audit 1 : Cleared" (space separated). */
function parseSessionLine(line: string): Array<[string, string]> {
  const out: Array<[string, string]> = [];
  const re = /(Semesters|Result|Marks|COP|Audit\s*\d)\s*:\s*([^\s:]*)/g;
  for (const m of line.matchAll(re)) {
    const value = m[2];
    // "Result : Marks : 686/1000": an empty value is followed directly by the next key.
    out.push([m[1].replace(/\s+/g, ' '), /^(Semesters|Result|Marks|COP|Audit)$/i.test(value) ? '' : value]);
  }
  return out;
}

function parseRow(line: string, scope: Scope, page: number): Row | null {
  const m = line.replace(/\t/g, ' ').replace(/\s+/g, ' ').match(ROW_LINE);
  if (!m) return null;
  const tokens = m[4].trim().split(' ').filter(Boolean);
  let grade: string | undefined;
  let back: string | undefined;
  if (tokens.length > 0 && /^[A-Z]\+?$/.test(tokens[tokens.length - 1])) grade = tokens.pop();
  if (tokens.length > 0 && /^(--|\d+\*)$/.test(tokens[tokens.length - 1])) back = tokens.pop();
  const row: Row = { code: m[1], name: m[2].trim(), type: m[3], back, grade, scope: { ...scope }, line: line.replace(/\t/g, ' ').replace(/\s+/g, ' '), page };
  if (tokens.length >= 2) {
    row.internal = tokens[0];
    row.external = tokens[1];
  } else if (tokens.length === 1) {
    row.marks = tokens[0];
  }
  return row;
}

export function parseStructure(text: string): Structure {
  const facts: Fact[] = [];
  const rows: Row[] = [];
  const blocks: Block[] = [];
  let scope: Scope = {};
  let page = 1;
  let block: Block = { scope: {}, lines: [], noResult: false, page: 1 };
  blocks.push(block);
  let pendingPrefix: string | null = null;
  let lastHeaderFact: Fact | null = null;
  let lastSemester: number | null = null;

  const startBlock = (s: Scope) => {
    block = { scope: { ...s }, lines: [], noResult: false, page };
    blocks.push(block);
  };
  const addFact = (key: string, value: string, line: string) => {
    const f: Fact = { key, value, scope: { ...scope }, line: line.replace(/\t/g, ' ').replace(/\s+/g, ' ').trim(), page };
    facts.push(f);
    return f;
  };

  for (const rawLine of (text || '').replace(/\r\n/g, '\n').split('\n')) {
    const line = rawLine.replace(/\s+$/, '');
    const trimmed = line.trim();
    if (!trimmed) continue;
    if (PAGE_MARKER.test(trimmed)) {
      // "-- 1 of 3 --" closes page 1: what follows is on the next page.
      page = Number(trimmed.match(/(\d+)\s+of/)?.[1] || page) + 1;
      continue;
    }

    const sessionMatch = trimmed.match(SESSION_LINE);
    if (sessionMatch && !trimmed.includes('\t')) {
      scope = { session: normSession(sessionMatch[1]) };
      lastSemester = null;
      lastHeaderFact = null;
      startBlock(scope);
      block.lines.push(trimmed);
      for (const [k, v] of parseSessionLine(trimmed)) addFact(k, v, trimmed);
      continue;
    }

    if (NO_RESULT.test(trimmed)) {
      block.noResult = true;
      block.lines.push(trimmed);
      addFact('Note', trimmed, trimmed);
      continue;
    }

    if (trimmed.includes('\t')) {
      const { pairs, trailingPrefix } = parseCells(line, pendingPrefix);
      if (pairs.length > 0) {
        pendingPrefix = trailingPrefix;
        // A new "Semester" pair opens a semester block.
        const semPair = pairs.find(([k]) => normKey(k) === 'semester');
        if (semPair) {
          let sem: string | undefined = semPair[1].match(/\d+/)?.[0];
          if (!sem) {
            // Empty semester number: infer it from Even/Odd and the previous semester.
            const parity = pairs.find(([k]) => normKey(k) === 'evenodd')?.[1].toLowerCase();
            if (lastSemester !== null && parity) {
              const following: number = lastSemester + 1;
              if ((parity === 'even') === (following % 2 === 0)) sem = String(following);
            }
          }
          if (sem) lastSemester = Number(sem);
          scope = { session: scope.session, semester: sem };
          // The session line already opened an (empty) block for this session.
          if (block.scope.session === scope.session && !block.scope.semester && block.lines.length === 1) {
            block.scope = { ...scope };
          } else {
            startBlock(scope);
          }
        }
        block.lines.push(trimmed.replace(/\t/g, ' ').replace(/\s+/g, ' '));
        for (const [k, v] of pairs) {
          const f = addFact(k, v, line);
          if (!scope.session) lastHeaderFact = f;
        }
        continue;
      }
      const row = parseRow(trimmed, scope, page);
      if (row) {
        rows.push(row);
        block.lines.push(row.line);
        continue;
      }
      block.lines.push(trimmed.replace(/\t/g, ' ').replace(/\s+/g, ' '));
      continue;
    }

    // Lines without tabs.
    if (/&$/.test(trimmed) && !trimmed.includes(':')) {
      pendingPrefix = trimmed; // wrapped label, e.g. "Institute Code &"
      block.lines.push(trimmed);
      continue;
    }
    const row = parseRow(trimmed, scope, page);
    if (row) {
      rows.push(row);
      block.lines.push(row.line);
      continue;
    }
    // Wrapped header value, e.g. "COMPUTER SCIENCE AND" + "ENGINEERING".
    if (lastHeaderFact && !scope.session && /^[A-Z][A-Z &.,-]{2,40}$/.test(trimmed)) {
      lastHeaderFact.value = `${lastHeaderFact.value} ${trimmed}`.trim();
      lastHeaderFact.line = `${lastHeaderFact.line} ${trimmed}`;
      block.lines.push(trimmed);
      continue;
    }
    block.lines.push(trimmed);
  }

  return { facts, rows, blocks: blocks.filter(b => b.lines.length > 0) };
}

/** True when the text really has repeating sections / tables worth special handling. */
export function hasStructure(s: Structure): boolean {
  const scoped = s.facts.filter(f => f.scope.session).length;
  return s.facts.length >= 6 && (s.rows.length >= 3 || scoped >= 4);
}

/** Facts before the first section (student/institute header). */
export function headerFacts(s: Structure): Fact[] {
  return s.facts.filter(f => !f.scope.session);
}

export function findFacts(s: Structure, keys: string[], predicate?: (f: Fact) => boolean): Fact[] {
  const wanted = new Set(keys.map(normKey));
  return s.facts.filter(f => wanted.has(normKey(f.key)) && (!predicate || predicate(f)));
}
