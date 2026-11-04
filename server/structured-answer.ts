/**
 * Deterministic answers for structured documents (see structure.ts).
 *
 * Rules:
 *  - Values come straight from the parsed text: no model, no guessing.
 *  - A value that exists in several sections (SGPA of semester 1 in a regular
 *    and a back session) is shown for EVERY section with its label, never
 *    collapsed to the first one.
 *  - If the question does not match anything structured, return null and let
 *    the normal ladder (retrieval + LLM) handle it.
 */
import type { GroundedResult } from './query';
import {
  findFacts,
  hasStructure,
  headerFacts,
  normKey,
  parseStructure,
  scopeLabel,
  type Fact,
  type Row,
  type Structure
} from './structure';

type Lang = 'en' | 'hinglish';

const cache = new Map<string, Structure>();

export function getStructure(doc: any): Structure | null {
  const text: string = doc?.raw_text || '';
  if (text.length < 200) return null;
  const key = `${doc.id}:${text.length}`;
  let s = cache.get(key);
  if (!s) {
    s = parseStructure(text);
    if (cache.size > 50) cache.clear();
    cache.set(key, s);
  }
  return hasStructure(s) ? s : null;
}

/* ------------------------------------------------------------------ */
/* Question understanding                                              */
/* ------------------------------------------------------------------ */

function normQuery(q: string): string {
  return q
    .toLowerCase()
    .replace(/[’`]/g, "'")
    .replace(/[^\p{L}\p{M}\p{N}'\-+.\s]/gu, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

const SEM_WORDS: Record<string, string> = {
  first: '1', pehla: '1', pehle: '1', pahla: '1',
  second: '2', dusra: '2', doosra: '2',
  third: '3', teesra: '3', tisra: '3',
  fourth: '4', chautha: '4', chauthha: '4',
  fifth: '5', panchva: '5', sixth: '6', seventh: '7', eighth: '8'
};

export function detectSemester(q: string): string | null {
  let m = q.match(/\b(?:sem(?:ester|ister)?|trimester)\s*[-:#]?\s*(\d)\b/);
  if (m) return m[1];
  m = q.match(/\b(\d)\s*(?:st|nd|rd|th)?\s*[-]?\s*(?:sem(?:ester|ister)?)\b/);
  if (m) return m[1];
  m = q.match(/\b(first|pehla|pehle|pahla|second|dusra|doosra|third|teesra|tisra|fourth|chautha|chauthha|fifth|panchva|sixth|seventh|eighth)\s+(?:sem(?:ester|ister)?)\b/);
  if (m) return SEM_WORDS[m[1]];
  return null;
}

function detectSession(q: string): string | null {
  const m = q.match(/\b(20\d\d)\s*[-/]\s*(\d\d)\b/);
  return m ? `${m[1]}-${m[2]}` : null;
}

const STOP = new Set(
  ('marks mark grade grades kitne kitna kitni mein me ka ki ke ko hai hain kya mila mile subject subjects paper papers ' +
    'number numbers score scores result results sem semester sgpa what is my in the of and for tell show batao bata btao ' +
    'dikhao konsa kaun which how many much got obtained internal external theory practical a an to me mera meri mere ' +
    'please about with from this that are was were did do does').split(/\s+/)
);

const ROMAN = /^(i{1,3}|iv|v|vi{0,3}|ix|x)$/;

function words(text: string): string[] {
  return (text.toLowerCase().match(/[\p{L}\p{M}\p{N}]+/gu) || []).filter(Boolean);
}

/** Meaningful subject words left in the question after removing filler. */
function subjectTokens(q: string): string[] {
  return words(q).filter(w => !STOP.has(w) && !/^\d+$/.test(w) && (w.length >= 3 || ROMAN.test(w)));
}

function tokenMatches(qt: string, nameTokens: string[], code: string): boolean {
  if (qt === code.toLowerCase()) return true;
  if (ROMAN.test(qt)) return nameTokens.includes(qt);
  return nameTokens.some(nt => {
    if (nt === qt) return true;
    // "maths" ~ "mathematics", "structure" ~ "structures": same first 4 letters.
    return qt.length >= 4 && nt.length >= 4 && qt.slice(0, 4) === nt.slice(0, 4) && !ROMAN.test(nt);
  });
}

function matchSubjectRows(q: string, rows: Row[]): Row[] {
  const tokens = subjectTokens(q);
  const meaningful = tokens.filter(t => !ROMAN.test(t));
  if (meaningful.length === 0) return [];
  const hits = rows.filter(r => {
    const nameTokens = words(r.name);
    return tokens.every(t => tokenMatches(t, nameTokens, r.code));
  });
  const distinctSubjects = new Set(hits.map(r => r.code));
  // A question that matches half the table ("engineering") is not about one subject.
  return distinctSubjects.size > 6 ? [] : hits;
}

/* ------------------------------------------------------------------ */
/* Formatting helpers                                                  */
/* ------------------------------------------------------------------ */

function tr(lang: Lang, en: string, hi: string): string {
  return lang === 'hinglish' ? hi : en;
}

function evidenceOf(doc: any, items: Array<{ line: string; field?: string; value?: string; page?: number }>) {
  return items.slice(0, 6).map(i => ({
    document_id: doc.id,
    file_name: doc.file_name,
    field_name: i.field,
    value: i.value,
    snippet: i.line.length > 200 ? `${i.line.slice(0, 200)}…` : i.line,
    page_number: i.page ?? 1,
    confidence: 0.97
  }));
}

function result(doc: any, intent: string, lang: Lang, answer: string, evidence: ReturnType<typeof evidenceOf>): GroundedResult {
  return {
    route: 'exact_field',
    detected_language: lang,
    intent,
    answer: `${answer}\n\n${tr(lang, `📌 Source: **${doc.file_name}**`, `📌 Source: **${doc.file_name}**`)}`,
    evidence,
    is_insufficient_evidence: false
  };
}

/** "(751 ) KIPM COLLEGE" -> "KIPM COLLEGE (code 751)". */
function splitCode(value: string): string {
  const m = value.match(/^\(\s*(\d+)\s*\)\s*(.+)$/);
  return m ? `${m[2].trim()} (code ${m[1]})` : value;
}

function rowDetail(r: Row, lang: Lang): string {
  const parts: string[] = [];
  if (r.internal !== undefined) parts.push(`${tr(lang, 'Internal', 'Internal')} ${r.internal}`);
  if (r.external !== undefined) parts.push(`${tr(lang, 'External', 'External')} ${r.external}`);
  if (r.marks !== undefined) parts.push(`${tr(lang, 'Marks', 'Marks')} ${r.marks}`);
  if (r.back && r.back !== '--') parts.push(`${tr(lang, 'Back paper', 'Back paper')} ${r.back}`);
  if (r.grade) parts.push(`${tr(lang, 'Grade', 'Grade')} **${r.grade}**`);
  return parts.join(' · ');
}

function semesterList(s: Structure): string {
  const sems = [...new Set(s.blocks.map(b => b.scope.semester).filter(Boolean))] as string[];
  return sems.sort().join(', ');
}

/* ------------------------------------------------------------------ */
/* Answer engine                                                       */
/* ------------------------------------------------------------------ */

export function answerStructured(query: string, doc: any, lang: Lang): GroundedResult | null {
  const s = getStructure(doc);
  if (!s) return null;
  const q = normQuery(query);
  if (!q) return null;

  const semester = detectSemester(q);
  const sessionYear = detectSession(q);
  // "back session" / "regular session" (but not "back paper").
  const sessionKind = /\bback\b(?!\s*papers?)/.test(q) ? 'BACK' : /\bregular\b/.test(q) ? 'REGULAR' : null;
  const inScope = (f: { scope: { session?: string; semester?: string } }) =>
    (!semester || f.scope.semester === semester) &&
    (!sessionYear || (f.scope.session || '').startsWith(sessionYear)) &&
    (!sessionKind || (f.scope.session || '').includes(`(${sessionKind})`));

  // ---- 1. Subject questions (grade / marks of a named subject) ----
  const subjectRows = matchSubjectRows(q, s.rows.filter(inScope));
  if (subjectRows.length > 0 && !/\b(sgpa|cgpa|gpa)\b/.test(q)) {
    const bySubject = new Map<string, Row[]>();
    for (const r of subjectRows) {
      const key = `${r.code}`;
      bySubject.set(key, [...(bySubject.get(key) || []), r]);
    }
    const blocks: string[] = [];
    for (const rows of bySubject.values()) {
      const head = `**${rows[0].name}** (${rows[0].code}, ${rows[0].type})`;
      const lines = rows.map(r => `- ${scopeLabel(r.scope)}: ${rowDetail(r, lang)}`);
      blocks.push(`${head}\n${lines.join('\n')}`);
    }
    return result(doc, 'structured_subject', lang, blocks.join('\n\n'), evidenceOf(doc, subjectRows.map(r => ({ line: r.line, field: r.name, value: r.grade, page: r.page }))));
  }

  // ---- 2. Failed subjects / back papers ----
  if (/\b(fail|failed|fails|backlogs?|back\s*papers?|carry|supplementary|fail\s+hua|fail\s+hue)\b/.test(q) || /\bkis\s+subject\b.*\bfail/.test(q)) {
    const failed = s.rows.filter(r => r.grade === 'F' && inScope(r));
    if (failed.length === 0) {
      return result(
        doc,
        'structured_failed',
        lang,
        tr(lang, 'No subject with grade **F** is listed in this document.', 'Is document me kisi subject me grade **F** nahi hai.'),
        evidenceOf(doc, [])
      );
    }
    const lines = failed.map(f => {
      // Same subject later passed through a back paper?
      const later = s.rows.find(r => r.code === f.code && r !== f && r.back && r.back !== '--' && r.grade && r.grade !== 'F');
      const cleared = later
        ? tr(lang, ` → later cleared: ${scopeLabel(later.scope)}, back paper ${later.back}, grade **${later.grade}**`, ` → baad me clear: ${scopeLabel(later.scope)}, back paper ${later.back}, grade **${later.grade}**`)
        : '';
      return `- **${f.name}** (${f.code}) — ${scopeLabel(f.scope)}: ${rowDetail(f, lang)}${cleared}`;
    });
    const note = tr(lang, '_Only grade F is treated as a fail here._', '_Yahan sirf grade F ko fail maana gaya hai._');
    return result(doc, 'structured_failed', lang, `${tr(lang, 'Subjects with grade F:', 'Grade F wale subjects:')}\n${lines.join('\n')}\n\n${note}`, evidenceOf(doc, failed.map(f => ({ line: f.line, field: f.name, value: f.grade, page: f.page }))));
  }

  // ---- 2b. "How many subjects / which subjects got grade X" ----
  const gradeAsked = q.match(/\bgrade\s+([a-f]\+?)(?![a-z])|\b([a-f]\+?)\s+grade\b/);
  if (gradeAsked && /\b(subjects?|papers?|kitne|konse|kaun|which|how many|list)\b/.test(q)) {
    const g = (gradeAsked[1] || gradeAsked[2]).toUpperCase();
    const matches = s.rows.filter(r => r.grade === g && inScope(r));
    if (matches.length === 0) {
      return result(doc, 'structured_grade_list', lang, tr(lang, `No subject with grade **${g}** is listed.`, `Kisi subject me grade **${g}** nahi hai.`), evidenceOf(doc, []));
    }
    const distinct = new Set(matches.map(r => r.code)).size;
    const lines = matches.map(r => `- **${r.name}** (${r.code}) — ${scopeLabel(r.scope)}`);
    const head = tr(
      lang,
      `Grade **${g}**: ${distinct} subject${distinct === 1 ? '' : 's'}${matches.length > distinct ? ` (${matches.length} entries; repeated sessions list the same subject again)` : ''}`,
      `Grade **${g}**: ${distinct} subject${matches.length > distinct ? ` (${matches.length} entries; repeat sessions me wahi subject dobara aata hai)` : ''}`
    );
    return result(doc, 'structured_grade_list', lang, `${head}\n${lines.join('\n')}`, evidenceOf(doc, matches.map(r => ({ line: r.line, field: r.name, value: r.grade, page: r.page }))));
  }

  if (semester && /\b(kitne|how many|total|count|number of)\b/.test(q) && /\b(subjects?|papers?)\b/.test(q)) {
    const blocks = s.blocks.filter(b => b.scope.semester === semester && inScope(b));
    const out: string[] = [];
    const ev: Array<{ line: string; field?: string; value?: string; page?: number }> = [];
    for (const b of blocks) {
      const f = (k: string) => s.facts.find(x => normKey(x.key) === normKey(k) && x.scope.session === b.scope.session && x.scope.semester === b.scope.semester);
      const total = f('Total Subjects');
      if (!total || b.noResult) continue;
      const th = f('Theory Subjects');
      const pr = f('Practical Subjects');
      out.push(`- ${scopeLabel(b.scope)}: **${total.value}**${th && pr ? ` (${tr(lang, 'theory', 'theory')} ${th.value}, ${tr(lang, 'practical', 'practical')} ${pr.value})` : ''}`);
      ev.push({ line: total.line, field: 'Total Subjects', value: total.value, page: total.page });
    }
    if (out.length > 0) {
      return result(doc, 'structured_subject_count', lang, `${tr(lang, `Total subjects, Semester ${semester}:`, `Semester ${semester} ke total subjects:`)}\n${out.join('\n')}`, evidenceOf(doc, ev));
    }
  }

  // ---- 3. SGPA ----
  if (/\b(sgpa|cgpa|gpa)\b/.test(q)) {
    const facts = findFacts(s, ['SGPA'], inScope);
    if (facts.length === 0) {
      return result(
        doc,
        'structured_sgpa',
        lang,
        tr(lang, `This document has no SGPA for ${semester ? `Semester ${semester}` : 'that selection'}. Semesters present: ${semesterList(s)}.`, `Is document me ${semester ? `Semester ${semester}` : 'is selection'} ka SGPA nahi hai. Maujood semesters: ${semesterList(s)}.`),
        evidenceOf(doc, [])
      );
    }
    const noResultBlocks = new Set(s.blocks.filter(b => b.noResult).map(b => scopeLabel(b.scope)));
    const wantsMax = /\b(sabse\s+(?:zyada|jyada|acha|accha|best)|highest|maximum|max|best|topper)\b/.test(q);
    const wantsMin = /\b(sabse\s+kam|lowest|minimum|min|worst)\b/.test(q);
    if (wantsMax || wantsMin) {
      const declared = facts.filter(f => !noResultBlocks.has(scopeLabel(f.scope)) && Number.isFinite(Number(f.value)));
      if (declared.length > 0) {
        const pick = declared.reduce((a, b) => ((wantsMax ? Number(b.value) > Number(a.value) : Number(b.value) < Number(a.value)) ? b : a));
        const title = wantsMax ? tr(lang, 'Highest SGPA', 'Sabse zyada SGPA') : tr(lang, 'Lowest SGPA', 'Sabse kam SGPA');
        return result(
          doc,
          'structured_sgpa_extreme',
          lang,
          `${title}: **${pick.value}** — ${scopeLabel(pick.scope)}\n\n${tr(lang, `(compared across ${declared.length} declared semester results)`, `(${declared.length} declared semester results me se)`)}`,
          evidenceOf(doc, declared.map(f => ({ line: f.line, field: 'SGPA', value: f.value, page: f.page })))
        );
      }
    }
    const lines = facts.map(f => {
      const empty = noResultBlocks.has(scopeLabel(f.scope));
      return `- ${scopeLabel(f.scope)}: ${empty ? tr(lang, '_no result declared_', '_result declare nahi hua_') : `**${f.value}**`}`;
    });
    const title = semester ? tr(lang, `SGPA, Semester ${semester}:`, `Semester ${semester} ka SGPA:`) : tr(lang, 'SGPA by semester:', 'Semester-wise SGPA:');
    return result(doc, 'structured_sgpa', lang, `${title}\n${lines.join('\n')}`, evidenceOf(doc, facts.map(f => ({ line: f.line, field: 'SGPA', value: f.value, page: f.page }))));
  }

  // ---- 4. Result of a semester ----
  if (semester && /\b(result|pass|passed|status|declared|declaration|marks?|total|kaisa|kaisi|natija)\b/.test(q)) {
    const blocks = s.blocks.filter(b => b.scope.semester === semester && inScope(b));
    if (blocks.length === 0) {
      return result(
        doc,
        'structured_semester',
        lang,
        tr(lang, `Semester ${semester} is not in this document. Semesters present: ${semesterList(s)}.`, `Semester ${semester} is document me nahi hai. Maujood semesters: ${semesterList(s)}.`),
        evidenceOf(doc, [])
      );
    }
    const out: string[] = [];
    const ev: Array<{ line: string; field?: string; value?: string; page?: number }> = [];
    for (const b of blocks) {
      const label = scopeLabel(b.scope);
      if (b.noResult) {
        out.push(`**${label}**\n${tr(lang, '- No result found for this semester in the document ("No Result found for the above semester").', '- Document me is semester ka result nahi mila ("No Result found for the above semester").')}`);
        ev.push({ line: b.lines.find(l => /no result found/i.test(l)) || label });
        continue;
      }
      const get = (k: string) => s.facts.find(f => normKey(f.key) === normKey(k) && f.scope.session === b.scope.session && f.scope.semester === b.scope.semester);
      const status = get('Result Status');
      const sg = get('SGPA');
      const tot = get('Total Marks Obt.');
      const date = get('Date of Declaration');
      const lines = [
        status && `- ${tr(lang, 'Result status', 'Result status')}: **${status.value}**`,
        sg && `- SGPA: **${sg.value}**`,
        tot && `- ${tr(lang, 'Total marks obtained', 'Total marks')}: **${tot.value}**`,
        date && `- ${tr(lang, 'Declared on', 'Declaration date')}: ${date.value}`
      ].filter(Boolean);
      out.push(`**${label}**\n${lines.join('\n')}`);
      for (const f of [status, sg, tot, date]) if (f) ev.push({ line: f.line, field: f.key, value: f.value, page: f.page });
    }
    return result(doc, 'structured_semester', lang, out.join('\n\n'), evidenceOf(doc, ev));
  }

  // ---- 5. Header facts (student / institute) ----
  const header = headerFacts(s);
  const one = (key: string) => header.find(f => normKey(f.key) === normKey(key));
  const single = (intent: string, label: string, fact: Fact | undefined, value?: string) =>
    fact ? result(doc, intent, lang, `${label}: **${value ?? fact.value}**`, evidenceOf(doc, [{ line: fact.line, field: fact.key, value: fact.value, page: fact.page }])) : null;

  if (/\benrol+ment\b/.test(q)) return single('structured_enrollment', tr(lang, 'Enrollment number', 'Enrollment number'), one('EnrollmentNo'));
  if (/\broll\b/.test(q)) return single('structured_roll', tr(lang, 'Roll number', 'Roll number'), one('RollNo'));
  if (/\b(father|pita|papa)\b/.test(q)) return single('structured_father', tr(lang, "Father's name", 'Father ka naam'), one("Father's Name"));
  if (/\b(college|institute|institution|university|vishwavidyalaya)\b/.test(q)) {
    const f = one('Institute Code & Name');
    return single('structured_institute', tr(lang, 'Institute', 'Institute'), f, f && splitCode(f.value));
  }
  if (/\b(course|branch|program(?:me)?|stream)\b/.test(q)) {
    const c = one('Course Code & Name');
    const b = one('Branch Code & Name');
    if (!c && !b) return null;
    const lines = [c && `- ${tr(lang, 'Course', 'Course')}: **${splitCode(c.value)}**`, b && `- ${tr(lang, 'Branch', 'Branch')}: **${splitCode(b.value)}**`].filter(Boolean);
    return result(doc, 'structured_course', lang, lines.join('\n'), evidenceOf(doc, [c, b].filter(Boolean).map(f => ({ line: f!.line, field: f!.key, value: f!.value }))));
  }
  if (/\bgender\b/.test(q)) return single('structured_gender', tr(lang, 'Gender', 'Gender'), one('Gender'));
  if (/\b(naam|name)\b/.test(q) && !/hindi/.test(q) && q.split(' ').length <= 6) {
    return single('structured_name', tr(lang, 'Name', 'Naam'), one('Name'));
  }

  // ---- 6. Overall result of a session (no semester named) ----
  if (/\b(result|pass|passed|natija)\b/.test(q) && !semester) {
    const sessions = s.facts.filter(f => normKey(f.key) === 'result' && f.scope.session && !f.scope.semester && inScope(f));
    if (sessions.length === 0) return null;
    const lines = sessions.map(f => {
      const marks = s.facts.find(m => normKey(m.key) === 'marks' && m.scope.session === f.scope.session && !m.scope.semester);
      return `- ${scopeLabel(f.scope)}: ${f.value ? `**${f.value}**` : tr(lang, '_not declared_', '_declare nahi_')}${marks && marks.value ? `, ${tr(lang, 'marks', 'marks')} ${marks.value}` : ''}`;
    });
    return result(doc, 'structured_overall', lang, `${tr(lang, 'Result by session:', 'Session-wise result:')}\n${lines.join('\n')}`, evidenceOf(doc, sessions.map(f => ({ line: f.line, field: 'Result', value: f.value, page: f.page }))));
  }

  return null;
}
