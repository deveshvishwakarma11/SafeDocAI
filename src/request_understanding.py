"""
SafeDocAI - Unified Request Understanding (Step 3)
==================================================

ONE central, deterministic-first layer that turns::

    raw user query + optional Step 1 conversation context

into ONE normalized structured representation (:class:`StructuredRequest`)
describing what the user is asking. This becomes the single
interpretation contract for the clarification engine (Step 4), constraint
execution (Step 5) and future layers -- they build on this structure
instead of re-parsing the query.

Design principles
-----------------
* **Reuse, don't duplicate**: language comes from the existing Phase 8
  detector (:mod:`language`), conversation/small-talk recognition from
  :mod:`conversation`, scope/relevance signals from :mod:`query_relevance`,
  field synonyms from the router's generic synonym groups
  (:mod:`query_router._SYNONYM_GROUPS`) and the Step 2 identifier
  vocabulary (:mod:`document_evidence.IDENTIFIER_FAMILIES`), and the
  summary decision from the answer engine's own cue matcher. This module
  only NORMALIZES their outputs into one structure.
* **Deterministic-first**: regex + synonym matching + context. No new LLM
  call is ever made here; everything stays 100% local/offline.
* **Never guess**: when several plausible fields/targets exist the layer
  reports ``ambiguity=True`` with ``candidate_fields`` instead of picking
  one. Clarification QUESTION GENERATION is NOT here (Step 4 owns it) --
  this layer only flags structured ambiguity.
* **Read-only context**: a passed Step 1 context dict is read, never
  mutated; :mod:`conversation_state` stays the sole owner of state.
* **No retrieval changes**: this layer does not route, retrieve or rank.
  It only describes the request.

Extensibility: the intent/operation/completeness vocabularies are small,
stable tuples; future operations (open_document, list_documents,
compare_documents) are already representable in the schema without any
parser change, and adding a document family or field synonym is one
tuple entry (no per-document-type branches anywhere).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field as dataclass_field
from typing import Any


__all__ = [
    "INTENT_DOCUMENT_INFORMATION",
    "INTENT_CONVERSATION",
    "INTENT_OUT_OF_SCOPE",
    "INTENT_DOCUMENT_ACTION",
    "OPERATION_GET_VALUE",
    "OPERATION_SUMMARY",
    "OPERATION_RETRIEVE_DETAILS",
    "FUTURE_OPERATIONS",
    "COMPLETENESS_SINGLE_VALUE",
    "COMPLETENESS_MULTI_FIELD",
    "COMPLETENESS_IMPORTANT_DETAILS",
    "COMPLETENESS_FULL_SUMMARY",
    "CONSTRAINT_KEYS",
    "StructuredRequest",
    "understand_request",
]


# ---------------------------------------------------------------------------
# Stable controlled vocabularies (small, extensible; no giant taxonomy)
# ---------------------------------------------------------------------------

INTENT_DOCUMENT_INFORMATION = "document_information"
INTENT_CONVERSATION = "conversation"
INTENT_OUT_OF_SCOPE = "out_of_scope"
#: Representable for future steps; never emitted by this layer yet because
#: there is no evidence signal for actions in the current product.
INTENT_DOCUMENT_ACTION = "document_action"

OPERATION_GET_VALUE = "get_value"
OPERATION_SUMMARY = "summary"
OPERATION_RETRIEVE_DETAILS = "retrieve_details"

#: Future operations the SCHEMA can represent. The parser does not emit
#: them yet (no document opening/listing/comparison exists); keeping them
#: here documents the extension point without changing any behavior.
FUTURE_OPERATIONS = (
    "open_document",
    "list_documents",
    "compare_documents",
)

COMPLETENESS_SINGLE_VALUE = "single_value"
COMPLETENESS_MULTI_FIELD = "multi_field"
COMPLETENESS_IMPORTANT_DETAILS = "important_details"
COMPLETENESS_FULL_SUMMARY = "full_summary"

#: The only keys this layer may put into ``constraints``. Extensible, but
#: every key here must be produced deterministically from the query text.
CONSTRAINT_KEYS = ("semester", "year", "month", "date", "owner")


# ---------------------------------------------------------------------------
# Reused vocabularies (normalized)
# ---------------------------------------------------------------------------

#: Document-family synonyms -> normalized family name. ORDER MATTERS:
#: longer/more-specific variants first. This is reference vocabulary for
#: clearly-identifiable mentions; the Step 2 understanding pipeline stays
#: the sole authority for what a document actually IS.
_DOCUMENT_FAMILY_SYNONYMS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("railway_ticket", ("railway ticket", "train ticket", "railway", "ticket")),
    ("application_form", ("application form", "application", "form")),
    ("marksheet", ("marksheet", "marks sheet", "marksheet document")),
    ("ration_card", ("ration card", "rashan card")),
    ("voter_id_card", ("voter id card", "voter card", "voter id")),
    ("aadhaar_card", ("aadhaar card", "aadhar card", "aadhaar", "aadhar")),
    ("pan_card", ("pan card", "pan")),
    ("passport", ("passport",)),
    ("bank_statement", ("bank statement", "account statement", "statement")),
    ("driving_licence", ("driving licence", "driving license", "licence", "license")),
    ("id_card", ("id card", "identity card")),
)

#: Follow-up / reference markers that make the PREVIOUS turn's target
#: relevant ("aur fare?", "uska number", "same ticket ka fare", "is form
#: mein"). Deterministic closed vocabulary, extensible.
_FOLLOW_UP_MARKERS: tuple[str, ...] = (
    "aur", "and also", "also", "bhi", "iska", "iski", "iske", "isme",
    "is form", "is ticket", "uska", "uski", "uske", "usme", "us ticket",
    "us form", "same", "wahi", "vahi", "it ka", "that ticket", "that form",
)

#: Generic numeric-value words that must NOT map to one field on their own.
_GENERIC_NUMBER_WORDS = ("number", "no", "numbar", "ank")

#: Plausible numeric fields for an unqualified "number" request. A fixed,
#: vocabulary-level list (NOT a universal field schema): it only feeds the
#: ambiguity report for Step 4; nothing here is auto-selected.
_GENERIC_NUMBER_CANDIDATES = (
    "roll",
    "pnr",
    "enrollment_id",
    "transaction_id",
    "application_number",
    "phone",
)

#: Generic result-ish fields for an unqualified "result" request.
_RESULT_CANDIDATES = ("sgpa", "marks", "total_marks")

#: Additional generic field groups beyond the router's synonym groups.
_EXTRA_FIELD_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("fare", ("fare", "ticket fare", "price", "cost", "rate", "kiraya")),
    ("result_status", ("result status",)),
)

#: Summary-completeness cues (full/important). "detail(s)"-style cues are
#: handled by _DETAIL_CUES below (multi_field), NOT here.
_FULL_SUMMARY_CUES: tuple[str, ...] = (
    "poora", "pura", "whole document", "entire document", "full document",
    "explain the document", "explain this document", "samjhao",
    "document ke baare", "kya likha", "kya likha hai",
)

_DETAIL_CUES: tuple[str, ...] = (
    "details", "detail", "full detail", "sab kuch", "all details",
    "information", "jankari", "jaankari",
)

_IMPORTANT_DETAILS_CUES: tuple[str, ...] = (
    "important details", "important information", "important points",
    "zaruri details", "zaroori details",
)

_WORD_RE = re.compile(r"[a-z0-9]+")


def _contains_phrase(text_norm: str, phrase: str) -> bool:
    """Word-bounded containment on already-normalized text."""

    return re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", text_norm) is not None


# ---------------------------------------------------------------------------
# Field synonym groups (REUSED from query_router + document_evidence)
# ---------------------------------------------------------------------------

def _snake(name: str) -> str:
    return re.sub(r"[\s_\-]+", "_", name.strip()).strip("_").lower()


def _build_field_groups() -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Canonical field -> variants, built from EXISTING vocabularies.

    Primary source: the router's generic language synonym groups
    (canonical = first member, matching that module's convention).
    Secondary: Step 2's identifier-family vocabulary, only for families
    whose canonical key is not already covered (e.g. Application Number,
    Consumer Number, Account Number...). Last: the small generic extras
    above (fare, result_status). No document-type schema is implied.
    """

    from query_router import _SYNONYM_GROUPS

    groups: list[tuple[str, tuple[str, ...]]] = []
    covered: set[str] = set()
    for members in _SYNONYM_GROUPS:
        # Deterministic canonical: shortest member, alphabetical tie-break
        # (frozenset iteration order is process-dependent and MUST NOT be
        # used for a stable schema), then normalized through the spelling
        # map so query-side keys match Step 2 storage keys.
        raw_canonical = _snake(min(members, key=lambda m: (len(m), m)))
        canonical = _CANONICAL_SPELLING_MAP.get(raw_canonical, raw_canonical)
        variants = tuple(sorted(members))
        variants = variants + tuple(
            v for v in _FIELD_VARIANT_SUPPLEMENTS.get(canonical, ())
            if v not in variants
        )
        groups.append((canonical, variants))
        covered.add(canonical)
        covered.update(_snake(v) for v in variants)

    try:
        from document_evidence import IDENTIFIER_FAMILIES
    except ImportError:  # pragma: no cover - src.-mode imports
        from src.document_evidence import IDENTIFIER_FAMILIES

    for canonical_key, _variants in IDENTIFIER_FAMILIES:
        canonical = _snake(canonical_key)
        if canonical not in covered:
            groups.append(
                (canonical, (_canonical_label_regex_words(canonical_key),))
            )
            covered.add(canonical)

    for canonical, variants in _EXTRA_FIELD_GROUPS:
        if canonical not in covered:
            groups.append((canonical, variants))

    return tuple(groups)


#: Deterministic canonical-name normalization across spelling/label
#: variants (both spellings are legitimate; the schema picks ONE, always).
#: Values match the Step 2 storage keys (IDENTIFIER_FAMILIES) so query-side
#: and document-side vocabulary agree.
_CANONICAL_SPELLING_MAP = {
    "enrolment_id": "enrollment_id",
    "enrolment_no": "enrollment_no",
    "txn_id": "transaction_id",
    "consumer_id": "consumer_number",
}

#: Practical label variants the router's groups do not list but real
#: queries use (task-required "Enrollment No" forms). Vocabulary-level,
#: generic -- not a per-document-type schema.
_FIELD_VARIANT_SUPPLEMENTS = {
    "enrollment_id": ("enrollment no", "enrolment no"),
}


def _canonical_label_regex_words(canonical_key: str) -> str:
    """Human label for an identifier family key (used as its variant)."""

    return canonical_key.lower()


FIELD_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = _build_field_groups()


def _normalize_query(text: str) -> str:
    """Lowercase, collapse punctuation/whitespace (keep alphanumerics)."""

    lowered = str(text or "").lower()
    lowered = re.sub(r"[^a-z0-9\u0900-\u097f]+", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def _find_field(text_norm: str) -> tuple[str | None, list[str]]:
    """Canonical field for the query, or (None, matched_group_names).

    A field is recognized ONLY through its synonym group: an unqualified
    generic word ("number") intentionally matches NOTHING here so the
    request stays ambiguous instead of guessing.
    """

    matched: list[str] = []
    for canonical, variants in FIELD_GROUPS:
        for variant in variants:
            variant_norm = _normalize_query(variant)
            if variant_norm and _contains_phrase(text_norm, variant_norm):
                matched.append(
                    _CANONICAL_SPELLING_MAP.get(canonical, canonical)
                )
                break
    if not matched:
        return None, []
    # Longest canonical name wins ("total_marks" over "marks"-ish overlap).
    matched.sort(key=lambda c: (-len(c), c))
    return matched[0], matched


def _find_target_document(text_norm: str) -> str | None:
    """Normalized document family for a clearly-identifiable mention."""

    for family, variants in _DOCUMENT_FAMILY_SYNONYMS:
        for variant in variants:
            if _contains_phrase(text_norm, variant):
                return family
    return None


_SEMESTER_RES = (
    re.compile(r"\b(\d{1,2})\s*(?:st|nd|rd|th)?\s*(?:sem|semester|seme)\b"),
    re.compile(r"\b(?:sem|semester)\s*(\d{1,2})\b"),
)
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_MONTH_RE = re.compile(
    r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t|tember)?|oct(?:ober)?|nov(?:ember)?|"
    r"dec(?:ember)?)\b"
)
_DATE_RE = re.compile(r"\b(\d{1,2})[-/](\d{1,2})[-/](\d{4})\b")

_MONTH_INDEX = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7,
    "july": 7, "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10, "nov": 11, "november": 11, "dec": 12,
    "december": 12,
}

_PERSONAL_REFS = ("mera", "meri", "mere", "my", "apna", "apni")


def _extract_constraints(text_norm: str, raw_text: str) -> dict[str, Any]:
    """Obvious constraints only -- never fabricated, never guessed."""

    constraints: dict[str, Any] = {}

    for pattern in _SEMESTER_RES:
        match = pattern.search(text_norm)
        if match:
            value = int(match.group(1))
            if 1 <= value <= 12:
                constraints["semester"] = value
            break

    year_match = _YEAR_RE.search(text_norm)
    if year_match:
        constraints["year"] = int(year_match.group(0))

    month_match = _MONTH_RE.search(text_norm)
    if month_match:
        constraints["month"] = _MONTH_INDEX[month_match.group(1)]

    date_match = _DATE_RE.search(raw_text.lower())
    if date_match:
        day, month, year = date_match.groups()
        constraints["date"] = f"{int(day):02d}-{int(month):02d}-{year}"

    if any(_contains_phrase(text_norm, ref) for ref in _PERSONAL_REFS):
        constraints["owner"] = "self"

    return constraints


def _has_generic_number_word(text_norm: str, field: str | None) -> bool:
    """True for an UNQUALIFIED generic number word ("number batao").

    Qualified forms ("roll number", "PNR number", "mobile number") belong
    to their field's synonym group and are never generic here. The check
    strips every known field variant first; a leftover bare number word
    means the user did not say WHICH number.
    """

    if field is not None:
        return False
    stripped = text_norm
    for _canonical, variants in FIELD_GROUPS:
        for variant in variants:
            variant_norm = _normalize_query(variant)
            if variant_norm:
                stripped = re.sub(
                    rf"(?<![a-z0-9]){re.escape(variant_norm)}(?![a-z0-9])",
                    " ",
                    stripped,
                )
    return any(_contains_phrase(stripped, word) for word in _GENERIC_NUMBER_WORDS)


def _is_summary_request_reused(query: str) -> bool:
    """Reuse the answer engine's own summary cue matcher (lazy import)."""

    try:
        from answer_engine import is_summary_request
    except ImportError:  # pragma: no cover - src.-mode imports
        from src.answer_engine import is_summary_request
    try:
        return bool(is_summary_request(query))
    except Exception:  # noqa: BLE001 - reuse must never break this layer
        return False


def _conversation_intent_reused(query: str) -> dict[str, str] | None:
    """Reuse the conversation matcher, with one deterministic repair.

    The matcher's phrase table has "tum kya kar sakte ho" but not the
    equally common doubled form "tum kya kya kar sakte ho". When the
    original phrase misses, immediate duplicate tokens are collapsed
    ("kya kya" -> "kya") and the matcher is retried once. Pure text
    normalization for matching; the original query is never changed.
    """

    try:
        from conversation import check_conversation_intent
    except ImportError:  # pragma: no cover - src.-mode imports
        from src.conversation import check_conversation_intent

    conversation = check_conversation_intent(query)
    if conversation is not None:
        return conversation

    tokens = str(query or "").split()
    collapsed = [tok for i, tok in enumerate(tokens) if i == 0 or tok != tokens[i - 1]]
    if len(collapsed) != len(tokens):
        return check_conversation_intent(" ".join(collapsed))
    return None


def _last_turn_target(context: dict[str, Any] | None) -> str | None:
    """Target document of the previous turn (READ-ONLY context use)."""

    if not isinstance(context, dict):
        return None
    last_query = context.get("last_query")
    if not last_query:
        return None
    return _find_target_document(_normalize_query(last_query))


# ---------------------------------------------------------------------------
# Structured request
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StructuredRequest:
    """The single structured interpretation contract (Step 3 schema).

    Extensible by adding fields with defaults; existing keys are stable
    so Steps 4/5 and downstream layers can rely on them.
    """

    original_query: str
    language: str
    intent: str
    operation: str | None
    target_document: str | None
    field: str | None
    candidate_fields: tuple[str, ...] = ()
    constraints: dict[str, Any] = dataclass_field(default_factory=dict)
    completeness: str | None = None
    ambiguity: bool = False
    clarification_needed: bool = False
    context_reference: str | None = None
    is_follow_up: bool = False
    confidence: float = 0.0
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Stable JSON-safe dict (tuples become lists)."""

        return {
            "original_query": self.original_query,
            "language": self.language,
            "intent": self.intent,
            "operation": self.operation,
            "target_document": self.target_document,
            "field": self.field,
            "candidate_fields": list(self.candidate_fields),
            "constraints": dict(self.constraints),
            "completeness": self.completeness,
            "ambiguity": self.ambiguity,
            "clarification_needed": self.clarification_needed,
            "context_reference": self.context_reference,
            "is_follow_up": self.is_follow_up,
            "confidence": self.confidence,
            "reason": self.reason,
        }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def understand_request(
    query: str,
    context: dict[str, Any] | None = None,
) -> StructuredRequest:
    """Understand one user request deterministically (no LLM, no I/O).

    Args:
        query: Raw user query.
        context: Optional Step 1 conversation context (the stable dict
            emitted by ``ConversationState.to_context()``). READ-ONLY:
            used to interpret follow-ups, never mutated.

    Returns:
        A :class:`StructuredRequest` (never raises for malformed input;
        malformed/empty queries produce an explicit low-confidence
        out-of-scope request instead).
    """

    if not isinstance(query, str) or not query.strip():
        return StructuredRequest(
            original_query="" if not isinstance(query, str) else query,
            language="unknown",
            intent=INTENT_OUT_OF_SCOPE,
            operation=None,
            target_document=None,
            field=None,
            completeness=None,
            ambiguity=False,
            clarification_needed=False,
            confidence=0.0,
            reason="empty_or_malformed_query",
        )

    raw_text = query
    text_norm = _normalize_query(raw_text)

    # -- Language: existing Phase 8 detector (single source of truth) ----
    try:
        from language import detect_language
    except ImportError:  # pragma: no cover - src.-mode imports
        from src.language import detect_language
    language = detect_language(raw_text)

    # -- Field / target / constraints (pure text evidence) ----------------
    field, matched_groups = _find_field(text_norm)
    target_document = _find_target_document(text_norm)
    constraints = _extract_constraints(text_norm, raw_text)

    # -- Conversation intent: existing matcher decides FIRST --------------
    conversation = _conversation_intent_reused(raw_text)
    if conversation is not None:
        return StructuredRequest(
            original_query=raw_text,
            language=language,
            intent=INTENT_CONVERSATION,
            operation=None,
            target_document=None,
            field=None,
            constraints={},
            completeness=None,
            ambiguity=False,
            clarification_needed=False,
            confidence=0.95,
            reason=f"conversation_intent:{conversation.get('intent')}",
        )
    # -- Follow-up interpretation via READ-ONLY context --------------------
    is_follow_up = any(
        _contains_phrase(text_norm, marker) for marker in _FOLLOW_UP_MARKERS
    )
    context_reference: str | None = None
    if is_follow_up and isinstance(context, dict) and context.get("last_query"):
        if target_document is None:
            previous_target = _last_turn_target(context)
            if previous_target is not None:
                target_document = previous_target
                context_reference = "previous_turn"
        if context_reference is None:
            context_reference = "previous_turn"

    # -- Scope decision: reuse the relevance gate, with a generic,
    #    vocabulary-based correction: a query that explicitly names one of
    #    OUR fields/families is document-scope even when the lexical gate
    #    finds no signal ("4 sem ka result batao"). The gate itself is
    #    untouched -- this only normalizes the structured interpretation.
    try:
        from query_relevance import check_query_relevance
    except ImportError:  # pragma: no cover - src.-mode imports
        from src.query_relevance import check_query_relevance
    gate = check_query_relevance(raw_text)
    # Document-scope vocabulary beyond the gate's lexical signals: a query
    # naming one of OUR fields/families ("result", "sgpa", ...) is
    # document-scope even when the gate finds no personal/doc-noun signal
    # ("4 sem ka result batao"). "result" is itself in the gate's support
    # vocabulary, so this only normalizes the structured interpretation.
    document_vocab_hit = bool(matched_groups) or target_document is not None or (
        _contains_phrase(text_norm, "result")
    )

    if not gate.get("related") and not document_vocab_hit:
        return StructuredRequest(
            original_query=raw_text,
            language=language,
            intent=INTENT_OUT_OF_SCOPE,
            operation=None,
            target_document=None,
            field=None,
            constraints=constraints,
            completeness=None,
            ambiguity=False,
            clarification_needed=False,
            confidence=0.9,
            reason=f"relevance_gate:{gate.get('method')}",
        )

    # -- Intent ------------------------------------------------------------
    intent = INTENT_DOCUMENT_INFORMATION
    reason = f"relevance_gate:{gate.get('method')}"
    confidence = 0.85

    # -- Operation + completeness -------------------------------------------
    detail_cue = any(_contains_phrase(text_norm, cue) for cue in _DETAIL_CUES)
    full_summary_cue = any(
        _contains_phrase(text_norm, cue) for cue in _FULL_SUMMARY_CUES
    )
    important_cue = any(
        _contains_phrase(text_norm, cue) for cue in _IMPORTANT_DETAILS_CUES
    )
    engine_summary = _is_summary_request_reused(raw_text)

    if full_summary_cue:
        operation = OPERATION_SUMMARY
        completeness = COMPLETENESS_FULL_SUMMARY
    elif important_cue or (engine_summary and not detail_cue):
        operation = OPERATION_SUMMARY
        completeness = COMPLETENESS_IMPORTANT_DETAILS
    elif detail_cue:
        operation = OPERATION_RETRIEVE_DETAILS
        completeness = COMPLETENESS_MULTI_FIELD
    elif field is not None:
        operation = OPERATION_GET_VALUE
        completeness = COMPLETENESS_SINGLE_VALUE
    else:
        operation = OPERATION_RETRIEVE_DETAILS
        completeness = COMPLETENESS_MULTI_FIELD

    # -- Ambiguity (report, never guess) ------------------------------------
    candidate_fields: tuple[str, ...] = ()
    ambiguity = False
    ambiguity_reasons: list[str] = []

    if _has_generic_number_word(text_norm, field):
        ambiguity = True
        ambiguity_reasons.append("unqualified_number")
        candidate_fields = _GENERIC_NUMBER_CANDIDATES
    elif (
        field is None
        and _contains_phrase(text_norm, "result")
        and operation in (OPERATION_GET_VALUE, OPERATION_RETRIEVE_DETAILS)
    ):
        ambiguity = True
        ambiguity_reasons.append("unqualified_result")
        candidate_fields = _RESULT_CANDIDATES
    elif field is None and target_document is None and not detail_cue and operation == OPERATION_RETRIEVE_DETAILS:
        ambiguity = True
        ambiguity_reasons.append("unresolved_target_and_field")

    # "X ki details" with a RESOLVED target is well-defined (multi_field),
    # so it is deliberately NOT ambiguous.

    return StructuredRequest(
        original_query=raw_text,
        language=language,
        intent=intent,
        operation=operation,
        target_document=target_document,
        field=field,
        candidate_fields=candidate_fields,
        constraints=constraints,
        completeness=completeness,
        ambiguity=ambiguity,
        clarification_needed=ambiguity,
        context_reference=context_reference,
        is_follow_up=is_follow_up,
        confidence=confidence,
        reason=reason,
    )
