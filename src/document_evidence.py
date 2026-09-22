"""
SafeDocAI - Generic Document Evidence Layer (Step 2)
====================================================

A document-type-agnostic evidence layer for Phase 3 understanding. It
improves the EXISTING dynamic architecture (OCR -> type detection ->
candidate discovery -> field selection -> evidence validation -> storage)
without introducing per-document-type branches anywhere in the pipeline.

Three generic capabilities:

1. TYPE-EVIDENCE REGISTRY (``NEW_DOCUMENT_TYPE_EVIDENCE``)
   Strong, multi-token evidence combinations for document-type FAMILIES
   the legacy heuristic table does not cover yet (railway ticket,
   application form, ...). Adding a future family (flight ticket,
   electricity bill, insurance policy, ...) means adding one registry
   entry -- no pipeline code changes. Classification is combination
   based: a family wins only with >= 2 strong evidence hits (or 1
   strong + 2 medium), never on one generic word.

2. STRUCTURAL IDENTIFIER DISCOVERY (``discover_identifier_fields``)
   Generic structural layouts that the line-based label:value extractor
   cannot see:
   * header-row-above / values-below  (``PNR Train No./Name Class`` on
     one line, ``4143027140 03252/...`` on the next)
   * caption-below-value              (``M103B71`` directly above
     ``(Enrollment ID)``)
   Labels are matched through a COMPACT, extensible identifier-family
   vocabulary (``IDENTIFIER_FAMILIES``) with reasonable label variants
   (Enrollment ID / Enrolment No / Enrollment Number, ...). Families are
   language-level vocabulary -- NOT a universal mandatory field list:
   a family only produces a candidate when its label is actually present
   in the OCR. Absent labels produce nothing; nothing is ever invented.

3. EVIDENCE CONTRADICTION GUARD (``claim_is_contradicted``)
   Used by the evidence validator: a document-type claim is unverified
   when the claimed family's own evidence is weak while ANOTHER family's
   evidence is strong in the same text. This blocks the real-data
   failure where ticket boilerplate ("Voter Identity Card / Passport /
   PAN Card ...") or form boilerplate ("... GATE 2027 results ...")
   satisfied a single-word indicator of the wrong family.

Pure and deterministic: regex only, no LLM, no network, no filesystem.
"""

from __future__ import annotations

import re
from typing import Any


__all__ = [
    "NEW_DOCUMENT_TYPE_EVIDENCE",
    "IDENTIFIER_FAMILIES",
    "normalize_family_name",
    "score_evidence_patterns",
    "classify_new_families",
    "family_evidence_score",
    "claim_is_contradicted",
    "discover_identifier_fields",
    "priority_identifier_fields",
]


# ---------------------------------------------------------------------------
# 1. Type-evidence registry (new families; same scoring model as heuristics)
# ---------------------------------------------------------------------------
# Each family: strong patterns are document-specific multi-token evidence
# combinations; medium patterns are supporting single signals. Patterns are
# case-insensitive regex against the raw OCR text.

NEW_DOCUMENT_TYPE_EVIDENCE: dict[str, dict[str, tuple[str, ...]]] = {
    "RAILWAY_TICKET": {
        "strong_patterns": (
            r"electronic reservation (slip|ticket)",
            r"\bIRCTC\b",
            r"\bPNR\b\s+Train",
            r"\bTrain No",
            r"\bBooking Status\b",
            r"\bCurrent Status\b",
            r"Reservation Slip",
            r"\bChart(ing| Preparation)\b",
        ),
        "medium_patterns": (
            r"\bPNR\b",
            r"\bwaitlist",
            r"\bberth\b",
            r"\bquota\b",
            r"\be-ticket\b",
            r"\bpassenger",
            r"\bRAC\b",
        ),
    },
    "APPLICATION_FORM": {
        "strong_patterns": (
            r"\bapplication form\b",
            r"\(\s*enrollment\s+(id|no|number)\s*\)",
            r"full name of the applicant",
            r"\(?\s*enrolment\s+(id|no|number)\s*\)?",
            r"\bexamination body\b",
            r"provisional application",
            r"\bdeclaration\b",
        ),
        "medium_patterns": (
            r"\bapplicant\b",
            r"\bscrutiny\b",
            r"\bpaper code\b",
            r"\beligibility\b",
            r"\bexamination\b",
            r"\be-?signature\b",
        ),
    },
}

# Reserved for future families (flight ticket, electricity bill, ...):
# add one entry here; no pipeline code changes are required.


def score_evidence_patterns(
    text_normalized: str,
    patterns: tuple[str, ...] | list[str] | list[tuple[str, Any]],
) -> int:
    """Count distinct pattern hits (accepts (regex, desc) tuples too)."""

    matches = 0
    for pattern_info in patterns:
        pattern = (
            pattern_info[0] if isinstance(pattern_info, tuple) else pattern_info
        )
        if re.search(pattern, text_normalized, re.IGNORECASE | re.UNICODE):
            matches += 1
    return matches


def _classify_family(
    text_normalized: str,
    family: dict[str, tuple[str, ...]],
) -> tuple[int, int] | None:
    """(strong, medium) hits for one family, or None when unclassified.

    Classification policy mirrors heuristics: >= 2 strong, or 1 strong
    plus 2 medium. Single weak keywords alone never classify.
    """

    strong = score_evidence_patterns(text_normalized, family["strong_patterns"])
    medium = score_evidence_patterns(text_normalized, family["medium_patterns"])
    if strong >= 2 or (strong >= 1 and medium >= 2):
        return (strong, medium)
    return None


def classify_new_families(text: str) -> dict[str, Any] | None:
    """Classify against the NEW-family registry only.

    Returns the winning family result (document_type, confidence,
    strong_matches, medium_matches) or None when no family classifies.
    Deterministic: highest evidence score wins; ties broken by name.
    """

    if not text or not text.strip():
        return None

    try:  # dual-path import (repo convention); lazy avoids import cycles
        from heuristics import normalize_text
    except ImportError:  # pragma: no cover - src.-mode imports
        from src.heuristics import normalize_text

    normalized = normalize_text(text)

    best: tuple[str, int, int] | None = None
    for family_name in sorted(NEW_DOCUMENT_TYPE_EVIDENCE):
        hits = _classify_family(normalized, NEW_DOCUMENT_TYPE_EVIDENCE[family_name])
        if hits is None:
            continue
        strong, medium = hits
        score = strong + 0.4 * medium
        if best is None or score > (best[1] + 0.4 * best[2]):
            best = (family_name, strong, medium)

    if best is None:
        return None

    family_name, strong, medium = best
    confidence = min(1.0, (strong + 0.4 * medium) / 3.0)
    return {
        "document_type": family_name,
        "confidence": round(confidence, 3),
        "strong_matches": strong,
        "medium_matches": medium,
        "evidence_source": "evidence_registry",
    }


def family_evidence_score(text: str, family_key: str) -> tuple[int, int]:
    """(strong, medium) evidence hits for one family key (no threshold).

    Accepts any registry family name; unknown names score (0, 0).
    """

    if not text:
        return (0, 0)

    try:  # dual-path import (repo convention); lazy avoids import cycles
        from heuristics import normalize_text
    except ImportError:  # pragma: no cover - src.-mode imports
        from src.heuristics import normalize_text

    normalized = normalize_text(text)
    family = NEW_DOCUMENT_TYPE_EVIDENCE.get(family_key)
    if family is None:
        return (0, 0)
    return (
        score_evidence_patterns(normalized, family["strong_patterns"]),
        score_evidence_patterns(normalized, family["medium_patterns"]),
    )


def normalize_family_name(claimed_type: str | None) -> str | None:
    """Map a free-form document-type string to a registry family name.

    "Railway Ticket", "RAILWAY_TICKET", "railway ticket" and
    "railway-ticket" all map to RAILWAY_TICKET: underscores, hyphens and
    whitespace are equivalent on both sides of the comparison. Returns
    None when no registry family matches.
    """

    if not claimed_type:
        return None
    normalized = re.sub(r"[_\-]+", " ", str(claimed_type)).strip().lower()
    for family_name in NEW_DOCUMENT_TYPE_EVIDENCE:
        family_normalized = re.sub(r"[_\-]+", " ", family_name.lower())
        if family_normalized == normalized:
            return family_name
    return None


def claim_is_contradicted(text: str, claimed_type: str | None) -> bool:
    """True when a type claim is contradicted by stronger rival evidence.

    Generic rules (no per-type logic):

    * A claim that normalizes to a REGISTRY family is contradicted when
      that family's own strong evidence is weak (< 2 hits) while another
      registry family has >= 2 strong hits.
    * Any OTHER claim (legacy family or free-form) is contradicted when
      some registry family has >= 2 strong hits in the same text: the
      document is demonstrably that registry family, so a different
      claim resting on one boilerplate indicator word ("voter",
      "results", ...) must not verify.

    Claims with >= 2 strong hits of their OWN registry family are never
    contradicted; text without registry-family evidence never triggers
    the guard, so all legacy behavior is unchanged.
    """

    if not claimed_type or not text:
        return False

    claimed = normalize_family_name(claimed_type)

    strongest_rival: tuple[str, int] | None = None
    for family_name in NEW_DOCUMENT_TYPE_EVIDENCE:
        strong, _ = family_evidence_score(text, family_name)
        if strong >= 2 and (
            strongest_rival is None or strong > strongest_rival[1]
        ):
            strongest_rival = (family_name, strong)

    if strongest_rival is None:
        return False

    if claimed is not None:
        claimed_strong, _ = family_evidence_score(text, claimed)
        if claimed_strong >= 2:
            return False
        return claimed != strongest_rival[0]

    # Claim of a non-registry family: contradicted whenever a registry
    # family dominates the text (the claim is not that family).
    return True


# ---------------------------------------------------------------------------
# 2. Structural identifier discovery (header-above / caption-below layouts)
# ---------------------------------------------------------------------------

#: Compact identifier-family vocabulary with reasonable label variants.
#: (canonical_key, label regex variants). Language-level vocabulary, not a
#: per-document-type field list: a family produces a candidate ONLY when its
#: label is actually present in the OCR text.
IDENTIFIER_FAMILIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("PNR", (r"\bPNR\b",)),
    (
        "Enrollment ID",
        (
            r"\benrollment\s+id\b",
            r"\benrolment\s+id\b",
            r"\benrollment\s+no\b",
            r"\benrolment\s+no\b",
            r"\benrollment\s+number\b",
            r"\benrolment\s+number\b",
        ),
    ),
    ("Transaction ID", (r"\btransaction\s+id\b", r"\btransaction\s+number\b", r"\btxn\s+id\b")),
    ("Roll Number", (r"\broll\s+(?:no|number|num)\b",)),
    ("Registration Number", (r"\bregistration\s+(?:no|number)\b",)),
    ("Application Number", (r"\bapplication\s+(?:no|number)\b",)),
    ("Passport Number", (r"\bpassport\s+(?:no|number)\b",)),
    ("Aadhaar Number", (r"\baadhaar\s*(?:no|number)?\b", r"\baadhar\s*(?:no|number)?\b")),
    ("Consumer Number", (r"\bconsumer\s+(?:no|number|id)\b",)),
    ("Account Number", (r"\baccount\s+(?:no|number)\b",)),
    ("Policy Number", (r"\bpolicy\s+(?:no|number)\b",)),
    ("Booking ID", (r"\bbooking\s+(?:id|no|number)\b",)),
    ("Receipt Number", (r"\breceipt\s+(?:no|number)\b",)),
    ("Order Number", (r"\border\s+(?:no|number)\b",)),
    ("Vehicle Number", (r"\bvehicle\s+(?:no|number)\b",)),
    ("Driving Licence No", (r"\bdriving\s+licen[cs]e\s*(?:no|number)?\b",)),
    ("Certificate Number", (r"\bcertificate\s+(?:no|number)\b",)),
)

#: A value-like token: contains a digit, 5-40 chars, bounded by non-word
#: chars. Covers identifiers (4143027140, M103B71, ABCDE1234F), dates and
#: codes -- and rejects prose (no digits) and tiny fragments.
_VALUE_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9])(?=[A-Za-z0-9]*\d)[A-Za-z0-9][A-Za-z0-9/\-\.]{3,39}(?![A-Za-z0-9])"
)

#: Trailing punctuation stripped from discovered values.
_VALUE_EDGE_TRIM_RE = re.compile(r"^[\.\-:]+|[\.\-:,]+$")

_LABEL_WRAPPER_RE = re.compile(r"^\((.*)\)$")


def _is_caption_only(line: str) -> bool:
    """True when the line is only a parenthesized caption, e.g. '(D.O.B)'."""

    stripped = line.strip()
    return bool(_LABEL_WRAPPER_RE.match(stripped)) and not _VALUE_TOKEN_RE.search(
        stripped
    )


def _first_value_token(line: str) -> tuple[str, int, int] | None:
    """First value-like token in ``line`` -> (value, start, end)."""

    match = _VALUE_TOKEN_RE.search(line)
    if match is None:
        return None
    value = _VALUE_EDGE_TRIM_RE.sub("", match.group(0))
    if not value or len(value) < 5 or not any(ch.isdigit() for ch in value):
        return None
    return (value, match.start(), match.start() + len(value))


def _last_value_token(line: str) -> tuple[str, int, int] | None:
    """Last value-like token in ``line`` -> (value, start, end)."""

    best: tuple[str, int, int] | None = None
    for match in _VALUE_TOKEN_RE.finditer(line):
        candidate = _VALUE_EDGE_TRIM_RE.sub("", match.group(0))
        if not candidate or len(candidate) < 5 or not any(
            ch.isdigit() for ch in candidate
        ):
            continue
        best = (candidate, match.start(), match.start() + len(candidate))
    return best


def _line_family_labels(line: str) -> list[tuple[str, re.Match]]:
    """All identifier-family labels found on one line."""

    found: list[tuple[str, re.Match]] = []
    lowered = line.lower()
    for canonical, variants in IDENTIFIER_FAMILIES:
        for variant in variants:
            match = re.search(variant, lowered, re.IGNORECASE)
            if match is not None:
                found.append((canonical, match))
                break
    return found


def discover_identifier_fields(raw_text: str) -> list[dict[str, Any]]:
    """Discover identifier fields from generic structural layouts.

    Handles, for every identifier family whose LABEL is present:
    * ``Label: value`` / ``Label value`` on one line (also covered by the
      line-based extractor; duplicates are removed by the caller);
    * header-row layout: label line with NO value on it, value on the
      NEXT non-empty line (first value token);
    * caption-below layout: parenthesized label caption, value taken from
      the nearest preceding non-caption line (last value token).

    Every candidate carries the EXACT char_span of the value inside
    ``raw_text`` (``[start, end)``) so evidence can be sliced verbatim.

    Nothing is invented: no label match -> no candidate. Returns a list
    of ``{family, value, char_span, line_no, layout}``.
    """

    if not raw_text or not raw_text.strip():
        return []

    lines = raw_text.split("\n")
    results: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    def _remember(family: str, value: str, span: tuple[int, int],
                  line_no: int, layout: str) -> None:
        key = (family, re.sub(r"\s+", "", value).casefold())
        if key in seen:
            return
        seen.add(key)
        results.append(
            {
                "family": family,
                "value": value,
                "char_span": [span[0], span[1]],
                "line_no": line_no,
                "layout": layout,
            }
        )

    offset = 0
    for line_no, line in enumerate(lines):
        stripped = line.strip()
        labels = _line_family_labels(stripped)
        offset += len(line) + 1

        if not labels:
            continue

        for family, label_match in labels:
            label_end = label_match.end()

            # -- Layout A: value on the SAME line after the label -------
            remainder = stripped[label_end:]
            same_line = _first_value_token(remainder)
            if same_line is not None:
                value, rel_start, rel_end = same_line
                # Absolute span of the value inside the full text.
                abs_start = sum(len(l) + 1 for l in lines[:line_no]) + \
                    len(line) - len(line.lstrip()) + \
                    line.find(remainder) + rel_start
                # Recompute precisely: find the value inside the raw text.
                abs_span = _locate(raw_text, value, offset - len(line) - 1)
                if abs_span is not None:
                    _remember(family, value, abs_span, line_no, "same_line")
                continue

            # -- Layout B: header row -> value on the NEXT non-empty line
            caption_wrapped = _LABEL_WRAPPER_RE.match(stripped) is not None
            if not caption_wrapped:
                next_line_no = line_no + 1
                while next_line_no < len(lines) and not lines[next_line_no].strip():
                    next_line_no += 1
                if next_line_no < len(lines):
                    next_line = lines[next_line_no]
                    if not _is_caption_only(next_line):
                        next_value = _first_value_token(next_line)
                        if next_value is not None:
                            value, _s, _e = next_value
                            abs_span = _locate(
                                raw_text, value,
                                sum(len(l) + 1 for l in lines[: next_line_no]),
                            )
                            if abs_span is not None:
                                _remember(
                                    family, value, abs_span, next_line_no,
                                    "header_row",
                                )
                continue

            # -- Layout C: caption below -> value on a PREVIOUS line ----
            prev_line_no = line_no - 1
            while prev_line_no >= 0:
                prev_line = lines[prev_line_no]
                if prev_line.strip() and not _is_caption_only(prev_line):
                    break
                prev_line_no -= 1
            if prev_line_no >= 0:
                prev_value = _last_value_token(lines[prev_line_no])
                if prev_value is not None:
                    value, _s, _e = prev_value
                    abs_span = _locate(
                        raw_text, value,
                        sum(len(l) + 1 for l in lines[:prev_line_no]),
                    )
                    if abs_span is not None:
                        _remember(
                            family, value, abs_span, prev_line_no,
                            "caption_below",
                        )

    return results


def _locate(raw_text: str, value: str, search_from: int) -> tuple[int, int] | None:
    """Locate ``value`` in ``raw_text`` at/after ``search_from``."""

    idx = raw_text.find(value, max(0, search_from))
    if idx < 0:
        idx = raw_text.find(value)
    if idx < 0:
        return None
    return (idx, idx + len(value))


def priority_identifier_fields(
    raw_text: str,
    existing_fields: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Evidence-validated priority fields ready to merge into Phase 3 output.

    Takes every discovered identifier candidate and keeps ONLY those whose
    value is supported by the ORIGINAL OCR text (via the authoritative
    evidence validator, strict identifier matching). Existing fields with
    the same normalized family key are respected: a family already present
    (any value) is NOT overridden; a family absent entirely is added.

    Returns Phase-3 field dicts:
        {"key", "value", "evidence_snippet", "verified": True, "priority": True}
    """

    from evidence_validator import is_value_supported_by_text

    existing_keys: set[str] = set()
    for field in existing_fields or []:
        key = re.sub(r"[\s_\-]+", " ", str(field.get("key", ""))).strip().casefold()
        if key:
            existing_keys.add(key)

    boosted: list[dict[str, Any]] = []
    for candidate in discover_identifier_fields(raw_text):
        canonical = candidate["family"]
        canonical_norm = re.sub(r"[\s_\-]+", " ", canonical).strip().casefold()

        # Respect already-extracted knowledge: never duplicate or override.
        if any(existing == canonical_norm or existing.startswith(canonical_norm)
               for existing in existing_keys):
            continue

        start, end = candidate["char_span"]
        snippet = raw_text[start:end]
        value = candidate["value"]

        if not is_value_supported_by_text(
            value=value,
            evidence_text=raw_text,
            evidence_snippet=snippet,
            require_exact_snippet_match=True,
        ):
            continue  # validation failure -> DROP, never store unverified

        boosted.append(
            {
                "key": canonical,
                "value": value,
                "evidence_snippet": snippet,
                "verified": True,
                "priority": True,
                "layout": candidate["layout"],
            }
        )
    return boosted
