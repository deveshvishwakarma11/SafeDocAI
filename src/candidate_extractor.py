"""
Deterministic label:value candidate extraction for the approved
"selection instead of transcription" Phase 3 architecture.

This module is PURE and DETERMINISTIC:

* no LLM / API / network dependency (imports: re, dataclasses, logging only)
* no filesystem access, no randomness, no time dependence
* same input text always yields the same candidates, ids, and spans

It is NOT wired into the Phase 3 pipeline yet; document_understanding still
runs the transcription flow until integration is approved.

Approved candidate schema (exactly what ``Candidate.to_dict`` returns)::

    {
      "id": 12,
      "label": "Roll Number",
      "value": "2407510100067",
      "line_no": 18,
      "char_span": [1234, 1250]
    }

Semantics
---------
* ``line_no``   -- 0-based index of the source line in the text passed in.
* ``char_span`` -- ``[start, end)`` character offsets into the SAME text,
                   covering the original OCR region from the start of the
                   label through the end of the value (noise between them is
                   included; the next pair's label is never included). This
                   is what the post-hoc evidence-snippet attacher will slice.
* ``id``        -- 0-based, assigned AFTER capping, in document order
                   (line_no, then char offset). Deterministic and unique
                   within the extracted scope (document or window).

Tolerated OCR corruption (grounded in the real AKTU One View OCR)
-----------------------------------------------------------------
* clean ``:`` and full-width ``：`` separators
* Devanagari-digit separators (``SGPA ३ 6.05``, ``Name ४ DEVESH``)
* isolated ASCII-digit separators from Devanagari corruption
  (``Gender 3M``, ``Total Subjects 7 9``, ``Practical Subjects 3 4``)
* stray glyphs as separators (``Date of Declaration s 30/06/25``,
  attached ``Course Code&``, loose ``Institute Code &``)
* OCR spacing noise, multiple pairs on one line, table-noise lines skipped

Non-goals / hard limits
-----------------------
* Bare standalone numbers are NEVER promoted to candidates.
* No document-type-specific field schemas; patterns are generic shape rules
  so the extractor works on arbitrary document types.
* Caps: 25 candidates per window, 40 per document. When more exist, the
  highest-quality candidates are retained by deterministic scoring (never a
  silent first-N), and the cap hit is reported (logger + stats).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Approved limits (isolated constants)
# ---------------------------------------------------------------------------

#: Maximum candidates retained for one window.
MAX_CANDIDATES_PER_WINDOW = 25

#: Maximum candidates retained for one document.
MAX_CANDIDATES_PER_DOCUMENT = 40

#: Quality gate: a candidate set is "usable" when at least this many
#: candidates carry non-empty values (approved plan, fallback decision).
USABLE_CANDIDATE_MIN_COUNT = 3

_MAX_VALUE_CHARS = 160
_MAX_LABEL_CHARS = 48
_MAX_LABEL_WORDS = 6

# ---------------------------------------------------------------------------
# Separator model
# ---------------------------------------------------------------------------

_SEP_KIND_COLON = "colon"
_SEP_KIND_DEV_DIGIT = "devanagari_digit"
_SEP_KIND_ASCII_DIGIT = "ascii_digit"
_SEP_KIND_STRAY = "stray_glyph"

#: Deterministic separator-quality component of the candidate score:
#: clean colons beat Devanagari-digit corruption beats stray glyphs.
_SEP_SCORE = {
    _SEP_KIND_COLON: 2.0,
    _SEP_KIND_DEV_DIGIT: 1.5,
    _SEP_KIND_ASCII_DIGIT: 1.0,
    _SEP_KIND_STRAY: 0.5,
}

#: Candidate separator matches, most-specific alternative first.
#: Colon runs; isolated Devanagari digits; isolated ASCII digits;
#: ``&``/``*`` (attached or loose); loose stray glyphs (s/S/§/~/°).
_SEPARATOR_RE = re.compile(
    r"[:：]+"
    r"|[\u0966-\u096F]"
    r"|[0-9]"
    r"|[&*]"
    r"|(?<![A-Za-z0-9'’])s(?![A-Za-z0-9])"
    r"|(?<![A-Za-z0-9'’])S(?![A-Za-z0-9])"
    r"|§|~|°"
)

# ---------------------------------------------------------------------------
# Label model
# ---------------------------------------------------------------------------

#: A label word: any Unicode letter first (Latin, Devanagari, CJK, ...), then
#: a run of non-separator characters (covers combining vowel signs and
#: viramas inside Devanagari words, apostrophes, dots, parens). Excluded from
#: the run: whitespace and every separator-class character, so a word can
#: never swallow the separator or adjacent Devanagari digits. Generic shape
#: rule -- no script- or document-type-specific behavior. The first char is
#: a letter (never a digit), so bare numbers are never absorbed as labels.
_WORD_RE = re.compile(r"[^\W\d_][^\s:：\u0966-\u096F&*§~°]*")

#: Noise allowed between a label run and its separator: a whitespace-led run
#: of spaces/digits (``Audit 1:`` -> ``Audit``). Must START with whitespace so
#: digits belonging to the label itself (``Field000``, ``Important0``) are
#: never eaten, and it is bounded so a value word is never absorbed.
_GAP_TAIL_RE = re.compile(r"(?:\s[\d\s]{0,3})$")

#: Trailing label tokens dropped as noise (``Audit 1`` -> ``Audit``).
_NOISE_TOKENS = {"s", "S", "&", "*", "§"}

# ---------------------------------------------------------------------------
# Value model
# ---------------------------------------------------------------------------

#: Chars strippable at the start of a value when followed by whitespace or
#: more noise (``s CP( 1)`` -> ``CP( 1)``, ``| (04) ...`` -> ``(04) ...``).
#: A value consisting only of such chars (``§``) is kept intact if it still
#: fails the alnum rule downstream; stripping never empties a value.
_NOISE_LEAD_CHARS = set("|[]{}'\"~,^°*§&sS,.;:_/")

#: Trailing junk stripped from values (never ``)`` or ``.`` inside numbers).
_TRAIL_JUNK_RE = re.compile(r"[|[\]{}~^§*&\s]+$")

_DATE_LIKE_RE = re.compile(r"\b\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}\b")
_DECIMAL_LIKE_RE = re.compile(r"\d+\.\d+\b")

_WHITESPACE_RUN_RE = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    """One deterministic label:value candidate (approved schema)."""

    id: int
    label: str
    value: str
    line_no: int
    char_span: tuple[int, int]
    #: Deterministic quality score (used only for cap retention order).
    score: float = 0.0
    #: Which separator class produced this pair.
    separator_kind: str = _SEP_KIND_COLON

    def to_dict(self) -> dict:
        """Exactly the approved candidate schema (JSON-friendly)."""
        return {
            "id": self.id,
            "label": self.label,
            "value": self.value,
            "line_no": self.line_no,
            "char_span": [self.char_span[0], self.char_span[1]],
        }


@dataclass(frozen=True)
class ExtractionStats:
    """Counts for reporting; ``cap_hit`` is the approved cap signal."""

    raw_pairs: int = 0
    after_dedup: int = 0
    returned: int = 0
    dropped_by_cap: int = 0
    cap_hit: bool = False
    scope: str = "document"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_dev_digit(ch: str) -> bool:
    return "\u0966" <= ch <= "\u096F"


def _kind_of(match_text: str) -> str:
    if ":" in match_text or "：" in match_text:
        return _SEP_KIND_COLON
    if _is_dev_digit(match_text[0]):
        return _SEP_KIND_DEV_DIGIT
    if match_text[0].isdigit() and match_text[0].isascii():
        return _SEP_KIND_ASCII_DIGIT
    return _SEP_KIND_STRAY


def _separator_candidates(line: str) -> list[tuple[int, int, str]]:
    """Find separator candidates with kind-specific context validation."""
    out: list[tuple[int, int, str]] = []
    for m in _SEPARATOR_RE.finditer(line):
        start, end = m.start(), m.end()
        kind = _kind_of(m.group())
        if kind == _SEP_KIND_COLON:
            out.append((start, end, kind))
        elif kind == _SEP_KIND_DEV_DIGIT:
            # Isolated from other Devanagari digits so Hindi numerals in
            # values (१२३) are never shredded.
            if start > 0 and _is_dev_digit(line[start - 1]):
                continue
            if end < len(line) and _is_dev_digit(line[end]):
                continue
            left_ok = start == 0 or line[start - 1].isspace() or line[start - 1].isalpha()
            if not left_ok:
                continue
            out.append((start, end, kind))
        elif kind == _SEP_KIND_ASCII_DIGIT:
            # Space-separated single digits only: "Gender 3M", "Subjects 7 9".
            # Values like 610 / 2407510100067 / 30/06/25 never qualify.
            left_ok = start > 0 and line[start - 1].isspace()
            right_ok = end >= len(line) or line[end].isspace() or line[end].isalpha()
            if not (left_ok and right_ok):
                continue
            out.append((start, end, kind))
        else:  # stray glyph
            ch = line[start]
            if ch in ("&", "*"):
                left_ok = start == 0 or line[start - 1].isspace() or line[start - 1].isalpha()
                right_ok = end >= len(line) or line[end].isspace() or line[end].isalnum()
                if not (left_ok and right_ok):
                    continue
            else:
                left_ok = start == 0 or line[start - 1].isspace()
                right_ok = end >= len(line) or line[end].isspace()
                if not (left_ok and right_ok):
                    continue
            out.append((start, end, kind))
    return out


def _letter_count(text: str) -> int:
    return sum(1 for ch in text if ch.isalpha())


def _has_lowercase(text: str) -> bool:
    return any(ch.islower() for ch in text if ch.isascii() and ch.isalpha())


def _is_all_caps_word(word: str) -> bool:
    letters = [ch for ch in word if ch.isascii() and ch.isalpha()]
    return len(letters) >= 3 and all(ch.isupper() for ch in letters)


def _is_code_like(word: str) -> bool:
    """Words carrying 3+ digits (BAS103, IBEE101) end label-run extension."""
    return sum(1 for ch in word if ch.isdigit() and ch.isascii()) >= 3


def _walk_back_label(
    line: str,
    sep_start: int,
    words: list[re.Match],
) -> tuple[int, str] | None:
    """Find the label run immediately before a separator.

    Returns ``(label_start, label)`` or ``None``. The run is built backward
    from the separator and stops at: non single-space joins, code-like words,
    all-caps words when the run tail is mixed-case (values like
    ``DEVESH VISHWAKARMA`` are not absorbed into labels), size limits, or
    line start.
    """
    prefix = line[:sep_start]
    trimmed = prefix.rstrip()
    gap_m = _GAP_TAIL_RE.search(trimmed)
    pos = (len(trimmed) - len(gap_m.group())) if gap_m else len(trimmed)

    tail = None
    for w in words:
        if w.end() == pos and w.start() < pos:
            tail = w
            break
    if tail is None:
        return None

    entries = [tail]
    tail_has_lower = _has_lowercase(tail.group())
    run_chars = len(tail.group())

    while True:
        cur_start = entries[0].start()
        if cur_start == 0:
            break
        # Label words join across a whitespace run (OCR spacing noise gives
        # "Roll   No"); the run must exist, otherwise the word is attached
        # to something that is not a label join.
        gap_start = cur_start
        while gap_start > 0 and line[gap_start - 1] == " ":
            gap_start -= 1
        if gap_start == cur_start:
            break
        prev = None
        for w in words:
            if w.end() == gap_start:
                prev = w
                break
        if prev is None:
            break
        word = prev.group()
        if run_chars + 1 + len(word) > _MAX_LABEL_CHARS:
            break
        if len(entries) >= _MAX_LABEL_WORDS:
            break
        if _is_code_like(word):
            break
        if _is_all_caps_word(word) and tail_has_lower:
            break
        entries.insert(0, prev)
        run_chars += 1 + len(word)

    tokens = [w.group() for w in entries]
    while len(tokens) > 1 and (
        len(tokens[-1]) == 1 or tokens[-1] in _NOISE_TOKENS
    ):
        tokens.pop()
    label = " ".join(tokens)
    if _letter_count(label) < 2 or len(label) > _MAX_LABEL_CHARS:
        return None
    return entries[0].start(), label


def _finish_value(raw: str) -> str:
    """Normalize spacing and strip stray glyphs around a value."""
    value = _WHITESPACE_RUN_RE.sub(" ", raw).strip()
    # Leading noise: strip only while a non-noise remainder stays, and only
    # when the noise char is followed by whitespace or more noise. This
    # removes the stray "s" in "s CP( 1)" but never touches "Science" or "§".
    idx = 0
    while (
        idx < len(value)
        and value[idx] in _NOISE_LEAD_CHARS
        and idx + 1 < len(value)
        and (value[idx + 1] in _NOISE_LEAD_CHARS or value[idx + 1] == " ")
    ):
        idx += 1
    value = value[idx:].lstrip()
    value = _TRAIL_JUNK_RE.sub("", value).strip()
    return _WHITESPACE_RUN_RE.sub(" ", value)


def _is_table_noise(line: str) -> bool:
    """Marksheet table rows (``Code [.. | .. | ..``) were never reliably
    extractable; skip lines with 2+ pipe/bracket cells."""
    return sum(1 for ch in line if ch in "|[]") >= 2


def _pairs_from_line(
    line: str,
    line_no: int,
    offset: int,
) -> list[tuple[str, str, str, int, int]]:
    """Extract (label, value, sep_kind, span_start, span_end) pairs."""
    if _is_table_noise(line):
        return []

    seps = _separator_candidates(line)
    if not seps:
        return []

    words = list(_WORD_RE.finditer(line))

    paired: list[tuple[int, int, str, int, str]] = []
    last_paired_end = -1
    for sep_start, sep_end, kind in seps:
        walked = _walk_back_label(line, sep_start, words)
        if walked is None:
            continue
        label_start, label = walked
        # Monotonic-region guard: a later pair's label must start after the
        # previous pair's separator ended. Without it, "Practical Subjects
        # 3 4 Total..." would let the "4" separator steal the "3" value.
        if label_start <= last_paired_end:
            continue
        paired.append((sep_start, sep_end, kind, label_start, label))
        last_paired_end = sep_end

    pairs: list[tuple[str, str, str, int, int]] = []
    for i, (sep_start, sep_end, kind, label_start, label) in enumerate(paired):
        next_label_start = paired[i + 1][3] if i + 1 < len(paired) else len(line)
        region = line[sep_end:next_label_start]
        value = _finish_value(region)
        if not value:
            continue
        if not any(ch.isalnum() for ch in value):
            continue  # single-glyph garbage like "§"
        if len(value) > _MAX_VALUE_CHARS:
            continue
        region_end = sep_end + len(region.rstrip())
        pairs.append(
            (label, value, kind, offset + label_start, offset + region_end)
        )
    return pairs


def _norm(text: str) -> str:
    return _WHITESPACE_RUN_RE.sub(" ", text).strip().casefold()


# ---------------------------------------------------------------------------
# Scoring (deterministic, document-type agnostic)
# ---------------------------------------------------------------------------


def _score_candidate(
    label: str,
    value: str,
    kind: str,
    span_text: str,
) -> float:
    score = _SEP_SCORE.get(kind, 0.0)
    if _letter_count(label) >= 3:
        score += 1.0
    if any(ch.isupper() for ch in label if ch.isascii()) or any(
        "\u0900" <= ch <= "\u097F" for ch in label
    ):
        score += 0.5
    if label.islower() and " " not in label and _letter_count(label) == len(label):
        score -= 1.5  # junk single lowercase word ("about")
    if any(ch.isdigit() for ch in value):
        score += 1.0
    if any(ch.isalpha() for ch in value):
        score += 0.5
    if _DATE_LIKE_RE.search(value) or _DECIMAL_LIKE_RE.search(value):
        score += 0.75
    if len(value) <= 1:
        score -= 1.0
    if (
        "://" in span_text
        or span_text.casefold().startswith("about:")
        or value.casefold().startswith(("www.", "http"))
    ):
        score -= 2.5
    if len(value) > 60:
        score -= 0.5
    return score


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------


def _extract_scoped(
    raw_text: str,
    lo: int,
    hi: int,
    max_candidates: int,
    scope: str,
) -> tuple[list[Candidate], ExtractionStats]:
    if not raw_text:
        return [], ExtractionStats(scope=scope)

    all_pairs: list[tuple[str, str, str, int, int, int]] = []
    offset = 0
    for line_no, line in enumerate(raw_text.split("\n")):
        stripped_line = line.rstrip("\r")
        for label, value, kind, span_start, span_end in _pairs_from_line(
            stripped_line, line_no, offset
        ):
            all_pairs.append((label, value, kind, span_start, span_end, line_no))
        offset += len(line) + 1

    # Keep only candidates whose span starts inside the requested range.
    scoped = [
        p for p in all_pairs if lo <= p[3] < hi
    ]

    raw_pairs = len(scoped)

    # Dedup exact (normalized label, normalized value) pairs -- the same
    # label:value on another line/span (page repeats, overlapping windows)
    # is one logical field; same label with DIFFERENT values is always kept.
    seen: set[tuple[str, str]] = set()
    deduped: list[tuple[str, str, str, int, int, int]] = []
    for label, value, kind, span_start, span_end, line_no in scoped:
        key = (_norm(label), _norm(value))
        if key in seen:
            continue
        seen.add(key)
        deduped.append((label, value, kind, span_start, span_end, line_no))

    scored = [
        (
            _score_candidate(
                label, value, kind, raw_text[span_start:span_end]
            ),
            label,
            value,
            kind,
            span_start,
            span_end,
            line_no,
        )
        for label, value, kind, span_start, span_end, line_no in deduped
    ]

    # Quality-based retention under the cap: highest score first, ties broken
    # deterministically by document position. Never a silent first-N.
    ranked = sorted(scored, key=lambda item: (-item[0], item[6], item[4]))
    kept = ranked[:max_candidates] if len(ranked) > max_candidates else ranked
    dropped = len(ranked) - len(kept)
    if dropped > 0:
        logger.warning(
            "candidate cap reached (%s): kept %d, dropped %d lowest-quality "
            "candidates deterministically",
            scope,
            len(kept),
            dropped,
        )

    # Final order is document order; ids are assigned last so they are
    # deterministic and unique within the scope.
    kept.sort(key=lambda item: (item[6], item[4]))
    candidates = [
        Candidate(
            id=idx,
            label=item[1],
            value=item[2],
            line_no=item[6],
            char_span=(item[4], item[5]),
            score=item[0],
            separator_kind=item[3],
        )
        for idx, item in enumerate(kept)
    ]

    stats = ExtractionStats(
        raw_pairs=raw_pairs,
        after_dedup=len(deduped),
        returned=len(candidates),
        dropped_by_cap=dropped,
        cap_hit=dropped > 0,
        scope=scope,
    )
    return candidates, stats


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_candidates(
    raw_text: str,
    *,
    max_candidates: int = MAX_CANDIDATES_PER_DOCUMENT,
) -> list[Candidate]:
    """Extract deterministic label:value candidates from full OCR text.

    Caps retention at ``max_candidates`` (approved document cap: 40) using
    deterministic quality scoring, and reports cap hits via logging and
    :func:`extract_candidates_with_stats`.
    """
    candidates, _ = extract_candidates_with_stats(
        raw_text, max_candidates=max_candidates
    )
    return candidates


def extract_candidates_with_stats(
    raw_text: str,
    *,
    max_candidates: int = MAX_CANDIDATES_PER_DOCUMENT,
) -> tuple[list[Candidate], ExtractionStats]:
    """Like :func:`extract_candidates` but also returns cap/counts stats."""
    return _extract_scoped(
        raw_text, 0, len(raw_text), max_candidates, "document"
    )


def extract_candidates_for_window(
    raw_text: str,
    start: int,
    end: int,
    *,
    max_candidates: int = MAX_CANDIDATES_PER_WINDOW,
) -> list[Candidate]:
    """Candidates whose span starts within ``[start, end)`` of ``raw_text``.

    Pass the FULL raw text plus the window bounds so ``line_no`` and
    ``char_span`` stay absolute (what the post-hoc evidence attacher needs).
    Ids are 0-based within the window; the approved window cap is 25.
    """
    candidates, _ = extract_candidates_for_window_with_stats(
        raw_text, start, end, max_candidates=max_candidates
    )
    return candidates


def extract_candidates_for_window_with_stats(
    raw_text: str,
    start: int,
    end: int,
    *,
    max_candidates: int = MAX_CANDIDATES_PER_WINDOW,
) -> tuple[list[Candidate], ExtractionStats]:
    return _extract_scoped(
        raw_text, start, end, max_candidates, "window"
    )


def deduplicate_candidates(
    candidates: list[Candidate],
) -> list[Candidate]:
    """Merge exact (normalized label, normalized value) duplicates.

    Used when candidate lists from overlapping windows are combined: the
    same label:value pair found twice (same or different span) collapses
    into its first occurrence; different values under the same label, or
    the same value under different labels, never collapse. Order is
    preserved and ids are reassigned 0-based.
    """
    seen: set[tuple[str, str]] = set()
    out: list[Candidate] = []
    for cand in candidates:
        key = (_norm(cand.label), _norm(cand.value))
        if key in seen:
            continue
        seen.add(key)
        out.append(cand)
    return [
        Candidate(
            id=idx,
            label=c.label,
            value=c.value,
            line_no=c.line_no,
            char_span=c.char_span,
            score=c.score,
            separator_kind=c.separator_kind,
        )
        for idx, c in enumerate(out)
    ]


def usable_candidate_count(candidates: list[Candidate]) -> int:
    """Number of candidates carrying a non-empty value."""
    return sum(1 for c in candidates if c.value.strip())


def has_usable_candidates(candidates: list[Candidate]) -> bool:
    """Approved quality gate: enough labeled candidates to skip heuristics."""
    return usable_candidate_count(candidates) >= USABLE_CANDIDATE_MIN_COUNT
