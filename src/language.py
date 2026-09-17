"""
Phase 8: Deterministic user-language detection.
===============================================

Classifies a user query as English, Hindi (Devanagari script) or
Hinglish (Roman Hindi mixed with English) using lightweight, fully
deterministic heuristics -- the LLM is never involved in detection.

Design rules:

* Zero false positives on ordinary English is the top priority: a query
  is only ever marked Hinglish/Hindi when an explicit Hindi marker is
  present. Every marker is a closed list of common Hindi function words
  (pronouns, question words, verbs) -- NOT an exhaustive dictionary --
  so the detector works for arbitrary documents without a schema.
* Script detection is trivially reliable: any Devanagari block character
  means Hindi (possibly mixed with English -> still "hindi" per spec,
  which lists Devanagari+English as the Hindi example).
* Hinglish = Latin script + >= 1 Hindi marker. Pure English = neither.
* Word-boundary matching everywhere: "hero" must not trigger on
  "ro", "an" must not fire inside "answer".

Markers are intentionally generic Hindi vocabulary, not domain terms,
so they apply to any document corpus.
"""

from __future__ import annotations

import re

__all__ = [
    "LANG_ENGLISH",
    "LANG_HINGLISH",
    "LANG_HINDI",
    "LANG_UNKNOWN",
    "detect_language",
    "language_label",
    "transliterate_for_matching",
]

#: Detected-language values used across the pipeline.
LANG_ENGLISH = "english"
LANG_HINGLISH = "hinglish"
LANG_HINDI = "hindi"
LANG_UNKNOWN = "unknown"

# ---------------------------------------------------------------------------
# Deterministic marker lists (closed, generic, high-precision)
# ---------------------------------------------------------------------------

#: Devanagari Unicode block (U+0900..U+097F).
_DEVANAGARI = re.compile(r"[\u0900-\u097f]")

#: Common Hindi/Hinglish function words and question verbs, Latin script.
#: Closed list, matched on word boundaries, lowercase.
_LATIN_HINDI_MARKERS: frozenset[str] = frozenset(
    {
        # pronouns / possessives
        "mera", "meri", "mere", "apna", "apni", "tumhara", "aapka", "aapki",
        "tera", "teri", "iska", "uska", "unka", "unki",
        # question words
        "kya", "kyun", "kyo", "kyu", "kaise", "kahan", "kaun", "kitna",
        "kitni", "kab", "kis", "kiska", "kin",
        # common verbs / auxiliaries
        "hai", "hain", "hona", "hota", "hoti", "hote", "tha", "thi",
        "batao", "bata", "bataiye", "batayiye", "dikha", "dikhaao",
        "chahiye", "chahiye", "sakta", "sakte", "sakta",
        "kar", "karo", "karna", "kare", "karta", "karti", "karte",
        "mil", "milega", "mila", "mili", "mile", "mila",
        "raha", "rahi", "rahe", "liye", "liya", "lene", "dena", "dedo",
        "nahi", "nahin", "bhi", "hi", "se", "mein",
        "par", "wala", "wali", "wale", "ho", "hoga", "hogi", "honge",
        "hua", "hui", "hue", "lijiye", "dijiye", "kijiye",
        # demonstratives
        "ye", "yeh", "wo", "woh", "yaha", "waha", "idhar", "udhar",
        # common connectives
        "aur", "ya", "lekin", "magar", "jab", "tab", "agar", "warna",
        "kyunki", "aur",
    }
)

#: Short connectives that ALSO occur in ordinary English ("hi", "se",
#: "par", "ya", "ho") are risky on their own; require at least one
#: unambiguous marker OR two distinct markers before declaring
#: Hinglish, keeping false positives on plain English near zero.
#: Deliberately NOT markers (identical to ordinary English words):
#: "to", "me", "the" (article), "mat" (doormat).
_SHORT_AMBIGUOUS = {"hi", "se", "ya", "par", "ho", "kar"}


#: Devanagari -> Latin transliteration, per character. Vowels carry an
#: inherent 'a' in Devanagari; consonant clusters are approximated by
#: plain concatenation -- good enough for marker matching ("क्या है" ->
#: "kyaa haai" whose tokens still hit the kya/hai markers).
_DEVANAGARI_TO_LATIN: dict[str, str] = {
    "\u0900": "", "\u0901": "", "\u0902": "n", "\u0903": "h",
    "\u0905": "a", "\u0906": "aa", "\u0907": "i", "\u0908": "ee",
    "\u0909": "u", "\u090a": "oo", "\u090b": "ri", "\u090f": "e",
    "\u0910": "ai", "\u0913": "o", "\u0914": "au",
    "\u0915": "k", "\u0916": "kh", "\u0917": "g", "\u0918": "gh",
    "\u0919": "ng", "\u091a": "ch", "\u091b": "chh", "\u091c": "j",
    "\u091d": "jh", "\u091e": "ny", "\u091f": "t", "\u0920": "th",
    "\u0921": "d", "\u0922": "dh", "\u0923": "n", "\u0924": "t",
    "\u0925": "th", "\u0926": "d", "\u0927": "dh", "\u0928": "n",
    "\u0929": "n", "\u092a": "p", "\u092b": "ph", "\u092c": "b",
    "\u092d": "bh", "\u092e": "m", "\u092f": "y", "\u0930": "r",
    "\u0931": "r", "\u0932": "l", "\u0933": "l", "\u0935": "v",
    "\u0936": "sh", "\u0937": "sh", "\u0938": "s", "\u0939": "h",
    "\u093c": "", "\u093e": "a", "\u093f": "i", "\u0940": "ee",
    "\u0941": "u", "\u0942": "oo", "\u0943": "ri", "\u0947": "e",
    "\u0948": "ai", "\u0949": "o", "\u094a": "o", "\u094b": "o",
    "\u094c": "au", "\u094d": "", "\u0950": "om",
    "\u0964": ".", "\u0965": ".",
}
def transliterate_for_matching(text: str) -> str:
    """Public helper: Devanagari -> Latin for downstream matching.

    Used by the query router so Devanagari queries hit the same
    deterministic Latin-script classification path. NEVER applied to
    document values or answers (those are always preserved exactly).
    Non-Devanagari input is returned unchanged.
    """

    text = str(text)
    if not _DEVANAGARI.search(text):
        return text
    return "".join(
        _DEVANAGARI_TO_LATIN.get(char, "" if "\u0900" <= char <= "\u097f" else char)
        for char in text
    )


def detect_language(text: str) -> str:
    """Return one of ``hindi`` / ``hinglish`` / ``english`` / ``unknown``.

    Deterministic and O(n):

    1. Devanagari script present -> ``hindi`` (Devanagari mixed with
       English identifiers is still Hindi, matching the spec example).
    2. Otherwise the Latin-script text (Devanagari-free) is matched
       against the closed Hindi marker list -> ``hinglish`` on a hit.
    3. Otherwise ``english``. Empty/whitespace input -> ``unknown``.
    """

    if not text or not str(text).strip():
        return LANG_UNKNOWN

    text = str(text)

    # 1) Devanagari script anywhere -> Hindi (mixed English is fine:
    #    the spec's Hindi example is Devanagari + English identifiers).
    if _DEVANAGARI.search(text):
        return LANG_HINDI

    # 2) Pure-Latin text: count distinct Hindi markers on word boundaries.
    tokens = re.findall(r"[a-zA-Z]+", text.lower())
    markers = [tok for tok in tokens if tok in _LATIN_HINDI_MARKERS]

    if not markers:
        return LANG_ENGLISH

    distinct = set(markers)
    strong = distinct - _SHORT_AMBIGUOUS
    if strong:
        return LANG_HINGLISH
    # Only short ambiguous markers: require >= 2 distinct ones to avoid
    # false positives ("to me", "hi there", "par for the course").
    if len(distinct) >= 2:
        return LANG_HINGLISH
    return LANG_ENGLISH


def language_label(lang: str) -> str:
    """Human-readable label for a detected language code."""

    return {
        LANG_ENGLISH: "English",
        LANG_HINGLISH: "Hinglish",
        LANG_HINDI: "Hindi",
        LANG_UNKNOWN: "Unknown",
    }.get(lang, lang.title() if lang else "Unknown")
