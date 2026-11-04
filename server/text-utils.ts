/** Small text helpers shared by the extractor and the answer engines. */

/** Placeholder values forms print for "nothing here". Never a real answer. */
export const JUNK_VALUE = /^(?:na|n\/a|nil|null|none|not\s+available|not\s+applicable|--?|\.+)$/i;

/** A person's name: 2-5 alphabetic words, none of them a form label or "NA". */
export function looksLikePersonName(value: string): boolean {
  const v = (value || '').trim();
  if (!/^[A-Za-z][A-Za-z.'\s-]{3,60}$/.test(v)) return false;
  const tokens = v.split(/\s+/);
  if (tokens.length < 2 || tokens.length > 5) return false;
  if (tokens.some(t => JUNK_VALUE.test(t))) return false;
  return !/\b(address|date|number|no|class|sex|gender|age|dob|code|name|type|status|fare|ticket|train|total)\b/i.test(v);
}

/** Organisation words: an "address" containing these is the operator's, not a person's. */
export const ORG_WORDS = /\b(railways?|office|ltd|pvt|limited|corporation|bank|department|ministry|authority|university|institute)\b/i;
