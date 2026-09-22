"""
SafeDocAI - Clarification Engine (Step 4)
=========================================

Turns the structured ambiguity reported by the Step 3 Unified Request
Understanding layer (:mod:`request_understanding`) into ONE concise,
deterministic clarification question.

Position in the architecture (Step 4 contract)::

    User Query
        ↓
    Unified Request Understanding (Step 3)  ->  StructuredRequest
        ↓
    Clarification Engine  (HERE)
        ├── resolved  -> needed=False (existing routing/retrieval/answer)
        └── ambiguous -> ONE question for the user

Core rules
----------
* **Never guess**: when several plausible interpretations exist the engine
  emits a question carrying the ACTUAL candidates from the structured
  request (and, for documents, the ACTUAL stored document names). It never
  selects one, never ranks candidates as "best", and never invents an
  option it did not see.
* **No re-parsing**: the engine consumes :class:`StructuredRequest`. It
  does NOT re-detect language, intent, document, field, summary cues or
  constraints -- Step 3 already owns all interpretation. This module is
  question generation only.
* **No new parser for replies**: a user's clarification answer is just the
  next user turn and flows through ``request_understanding.
  understand_request`` again (with Step 1 conversation context). The
  clarification itself is recorded as an ordinary conversation turn, so no
  parallel parser is created.
* **Deterministic-first**: fixed per-language templates, no LLM call, no
  network. Everything stays 100% local/offline.
* **Read-only context**: the optional Step 1 context dict is only READ --
  to recognize that a follow-up is already resolved by the prior turn --
  and never mutated; :mod:`conversation_state` remains the sole owner of
  state.

Clarification types (small, closed set)
--------------------------------------
* ``ambiguous_field``      - several plausible fields ("number batao").
* ``ambiguous_document``   - a details-style request whose target cannot
  be resolved while the store holds several documents ("meri details
  batao") -- only when ACTUAL document names are supplied; without real
  names nothing is invented and no question is emitted.
* ``ambiguous_reference``  - a follow-up pronoun ("iska number batao")
  that the conversation context cannot resolve.
* ``ambiguous_constraint`` - a named field missing its required constraint
  ("semester ka SGPA batao" -- WHICH semester?). Filtering/reranking is
  NOT done here (Step 5 owns constraint execution).

Only a category whose ambiguity ACTUALLY exists is emitted; every other
request returns ``needed=False`` with ``question=None``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field as dataclass_field
from typing import Any


__all__ = [
    "CLARIFICATION_FIELD",
    "CLARIFICATION_DOCUMENT",
    "CLARIFICATION_REFERENCE",
    "CLARIFICATION_CONSTRAINT",
    "Clarification",
    "build_clarification",
    "available_document_names",
    "MAX_CLARIFICATION_OPTIONS",
]

# Clarification categories (stable closed set).
CLARIFICATION_FIELD = "ambiguous_field"
CLARIFICATION_DOCUMENT = "ambiguous_document"
CLARIFICATION_REFERENCE = "ambiguous_reference"
CLARIFICATION_CONSTRAINT = "ambiguous_constraint"

#: Practical presentation cap (task: "preferably 2-5 useful choices").
#: Order is presentation-stable only; no candidate is ranked as "best".
MAX_CLARIFICATION_OPTIONS = 5

# Constraint fields that may be asked about (WHICH one?) -- stable set.
_ASKABLE_CONSTRAINTS: tuple[str, ...] = ("semester", "year", "month")

#: Words that NAME a constraint dimension. A bare mention (dimension word
#: directly followed by a postposition, no value) is the generic evidence of
#: constraint ambiguity, e.g. "semester ka SGPA" -- WHICH semester? The
#: postposition requirement keeps field names like "semester GPA" (where the
#: dimension word is part of the field synonym) from ever matching.
_CONSTRAINT_WORDS: dict[str, tuple[str, ...]] = {
    "semester": ("semester", "sem"),
    "year": ("year", "saal", "varsh"),
    "month": ("month", "mahina", "maheena"),
}

_BARE_CONSTRAINT_RE: dict[str, re.Pattern[str]] = {
    key: re.compile(
        rf"(?<![a-z0-9])(?:{'|'.join(re.escape(w) for w in words)})"
        rf"(?![a-z0-9])\s+(?:ka|ki|ke)(?![a-z0-9])"
    )
    for key, words in _CONSTRAINT_WORDS.items()
}

#: Function words (ownership, question/postposition, operation verbs, field/
#: detail/generic-document words) in both scripts. A details-style request
#: whose tokens are ALL in this closed vocabulary is a BARE "my details"
#: request ("meri details batao", "मेरा नंबर बताओ") — the user is asking
#: about their documents themselves, so asking WHICH document is right. A
#: query with subject matter left over ("quantum harmonica in my papers")
#: is NOT a document-selection question: the existing flow answers it
#: honestly (typically insufficient), exactly as before Step 4. This is a
#: narrow presentation guard — it does NOT re-parse intent/field/summary,
#: which remain Step 3's decisions.
_BARE_DETAILS_FILLERS: frozenset[str] = frozenset({
    # ownership / pronouns
    "my", "mine", "our", "me", "mera", "meri", "mere", "hamara", "hamari",
    "hamare", "apna", "apni", "mer", "mera_",
    "मेरा", "मेरी", "मेरे", "हमारा", "हमारी", "हमारे", "अपना", "अपनी",
    # question words / copulas / postpositions / articles
    "what", "which", "is", "are", "the", "a", "an", "in", "of", "on", "to",
    "kya", "hai", "hain", "ka", "ki", "ke", "ko", "se",
    "क्या", "है", "हैं", "का", "की", "के", "को", "से",
    # operation verbs
    "batao", "bata", "batana", "batao_", "de", "do", "dena", "dijiye",
    "show", "give", "tell", "samjhao", "explain",
    "बताओ", "बता", "दो", "दीजिए", "समझाओ",
    # field / detail / summary words
    "details", "detail", "information", "info", "summary", "number", "no",
    "numbar", "ank",
    "विवरण", "जानकारी", "सारांश", "नंबर",
    # generic document words (not family-specific enough to resolve a target)
    "document", "documents", "doc", "docs", "file", "files", "paper",
    "papers",
    "दस्तावेज़", "दस्तावेज", "फ़ाइल", "फाइल", "कागज़", "कागज़ात",
})

# Word tokens: the full Devanagari block FIRST (so combining marks stay
# glued to their consonants — "\w" does not match matras, and a second-
# position alternative would split "मेरा" into "म" + "ेरा"), then "\w"
# minus underscore for everything else.
_WORD_TOKEN_RE = re.compile(r"[\u0900-\u097F]+|[^\W_]+", re.UNICODE)


def _is_bare_details_request(query: str) -> bool:
    """True when every word token is a function/field/document filler.

    See ``_BARE_DETAILS_FILLERS`` for why this guard exists.
    """

    tokens = [match.group(0).lower() for match in _WORD_TOKEN_RE.finditer(query)]
    return bool(tokens) and all(token in _BARE_DETAILS_FILLERS for token in tokens)

#: Human display labels for candidate fields (presentation only; the
#: stable machine-readable ``options`` remain canonical snake_case).
FIELD_OPTION_LABELS: dict[str, str] = {
    "roll": "roll number",
    "pnr": "PNR",
    "enrollment_id": "enrollment ID",
    "transaction_id": "transaction ID",
    "application_number": "application number",
    "phone": "phone number",
    "sgpa": "SGPA",
    "marks": "marks",
    "total_marks": "total marks",
    "fare": "fare",
}

# ---------------------------------------------------------------------------
# Deterministic question templates (no LLM). Latin-script field labels stay
# as identifiers in every language, matching the Phase 8 rule that document
# values/identifiers are never transliterated. Per the Step 4 spec, the
# Hindi and Hinglish question text is the same Roman-Hindi style.
# ---------------------------------------------------------------------------

_TEMPLATES: dict[str, dict[str, str]] = {
    "field_number": {
        "english": "Which number do you need — {options}?",
        "hinglish": "Kaunsa number chahiye — {options}?",
        "hindi": "Kaunsa number chahiye — {options}?",
    },
    "field": {
        "english": "Which one do you need — {options}?",
        "hinglish": "Kaunsa chahiye — {options}?",
        "hindi": "Kaunsa chahiye — {options}?",
    },
    "document": {
        "english": "Which document do you need — {options}?",
        "hinglish": "Kaunse document ki details chahiye — {options}?",
        "hindi": "Kaunse document ki details chahiye — {options}?",
    },
    "reference": {
        "english": "Which document do you mean — {options}?",
        "hinglish": "Kaunse document ki baat kar rahe ho — {options}?",
        "hindi": "Kaunse document ki baat kar rahe ho — {options}?",
    },
    "reference_no_options": {
        "english": "Which document do you mean? Please name the document.",
        "hinglish": "Kaunse document ki baat kar rahe ho? Document ka naam batao.",
        "hindi": "Kaunse document ki baat kar rahe ho? Document ka naam batao.",
    },
    "constraint": {
        "english": "Which {constraint} do you need?",
        "hinglish": "Kaunse {constraint} ka chahiye?",
        "hindi": "Kaunse {constraint} ka chahiye?",
    },
}

_CONSTRAINT_LABELS = {
    "semester": "semester",
    "year": "year (saal)",
    "month": "month (mahina)",
}


def _field_option_label(canonical: str) -> str:
    """Stable human label for one candidate field (never invented)."""

    return FIELD_OPTION_LABELS.get(canonical, str(canonical).replace("_", " "))


def _join_options(labels: list[str], language: str) -> str:
    """Deterministic option list: "a, b, or c" (English) / "a, b, ya c"."""

    if len(labels) <= 1:
        return labels[0] if labels else ""
    joiner = " or " if language == "english" else " ya "
    return ", ".join(labels[:-1]) + joiner + labels[-1]


def _make_question(
    kind: str,
    options: tuple[str, ...],
    language: str,
    field_labeler=_field_option_label,
    **fmt: Any,
) -> str:
    """Render one template deterministically (unknown language -> hinglish)."""

    templates = _TEMPLATES[kind]
    template = templates.get(language) or templates["hinglish"]
    if "{options}" in template:
        fmt["options"] = _join_options(
            [field_labeler(option) for option in options], language
        )
    return template.format(**fmt)


# ---------------------------------------------------------------------------
# Clarification result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Clarification:
    """Stable, JSON-safe clarification result (Step 4 contract)."""

    needed: bool
    question: str | None
    options: tuple[str, ...] = ()
    reason: str | None = None
    language: str | None = None
    target_document: str | None = None
    candidates: tuple[str, ...] = dataclass_field(default_factory=tuple)
    meta: dict[str, Any] = dataclass_field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Stable JSON-safe dict (tuples become lists)."""

        return {
            "needed": self.needed,
            "question": self.question,
            "options": list(self.options),
            "reason": self.reason,
            "language": self.language,
            "target_document": self.target_document,
            "candidates": list(self.candidates),
            "meta": dict(self.meta),
        }


_NO_CLARIFICATION = Clarification(needed=False, question=None)


# ---------------------------------------------------------------------------
# Structured-request normalization (dataclass OR dict; never re-parses text)
# ---------------------------------------------------------------------------


def _normalize_request(structured_request: Any) -> dict[str, Any] | None:
    """Read the Step 3 contract in either form; None when unusable."""

    if structured_request is None:
        return None
    if isinstance(structured_request, dict):
        return {
            "intent": structured_request.get("intent"),
            "operation": structured_request.get("operation"),
            "target_document": structured_request.get("target_document"),
            "field": structured_request.get("field"),
            "candidate_fields": tuple(
                structured_request.get("candidate_fields") or ()
            ),
            "constraints": dict(structured_request.get("constraints") or {}),
            "completeness": structured_request.get("completeness"),
            "ambiguity": bool(structured_request.get("ambiguity")),
            "clarification_needed": bool(
                structured_request.get("clarification_needed")
            ),
            "context_reference": structured_request.get("context_reference"),
            "is_follow_up": bool(structured_request.get("is_follow_up")),
            "language": structured_request.get("language") or "hinglish",
            "reason": str(structured_request.get("reason") or ""),
            "original_query": str(structured_request.get("original_query") or ""),
        }
    try:
        return {
            "intent": structured_request.intent,
            "operation": structured_request.operation,
            "target_document": structured_request.target_document,
            "field": structured_request.field,
            "candidate_fields": tuple(structured_request.candidate_fields or ()),
            "constraints": dict(structured_request.constraints or {}),
            "completeness": structured_request.completeness,
            "ambiguity": bool(structured_request.ambiguity),
            "clarification_needed": bool(structured_request.clarification_needed),
            "context_reference": structured_request.context_reference,
            "is_follow_up": bool(structured_request.is_follow_up),
            "language": structured_request.language or "hinglish",
            "reason": str(structured_request.reason or ""),
            "original_query": str(structured_request.original_query or ""),
        }
    except Exception:  # noqa: BLE001 - never break the answer flow
        return None


# ---------------------------------------------------------------------------
# Read-only context helpers
# ---------------------------------------------------------------------------

_REFERENCE_PRONOUNS: tuple[str, ...] = ("iska", "iski", "isme", "it ka", "ismein")

_CONTEXT_FAMILY_VARIANTS: tuple[tuple[str, ...], ...] = (
    ("railway ticket", "train ticket", "railway", "ticket"),
    ("application form", "application", "form"),
    ("marksheet", "marks sheet"),
    ("ration card", "rashan card"),
    ("voter id card", "voter card", "voter id"),
    ("aadhaar card", "aadhar card", "aadhaar", "aadhar"),
    ("pan card",),
    ("passport",),
    ("bank statement", "account statement", "statement"),
    ("driving licence", "driving license", "licence", "license"),
    ("id card", "identity card"),
)


def _word_contains(text_norm: str, phrase: str) -> bool:
    return (
        re.search(
            rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", text_norm
        )
        is not None
    )


def _context_resolves_reference(context: dict[str, Any] | None) -> bool:
    """True when the prior turn already names a target document family.

    Read-only use of the Step 1 context dict (a ConversationState object is
    also accepted via its ``to_context()``). Step 3 already resolves the
    target when it can; this secondary check covers contexts whose last
    query names a document family without producing a resolvable target.
    """

    if context is None:
        return False
    if not isinstance(context, dict):
        to_context = getattr(context, "to_context", None)
        if callable(to_context):
            try:
                context = to_context()
            except Exception:  # noqa: BLE001 - read-only helper
                return False
        if not isinstance(context, dict):
            return False
    last_query = re.sub(r"\s+", " ", str(context.get("last_query") or "").lower())
    if not last_query:
        return False
    return any(
        _word_contains(last_query, variant)
        for family in _CONTEXT_FAMILY_VARIANTS
        for variant in family
    )


def _has_reference_pronoun(request: dict[str, Any]) -> bool:
    query_norm = re.sub(
        r"\s+", " ", request.get("original_query", "").lower()
    )
    return any(_word_contains(query_norm, p) for p in _REFERENCE_PRONOUNS)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def _resolve_document_names(
    available_documents: Any,
) -> list[str] | None:
    """Resolve the document-name source (list/tuple/callable/None).

    A LAZY callable is supported so production callers never touch the
    store unless a document clarification actually needs the names.
    Returns None when no real names are available (nothing is invented).
    """

    source = available_documents
    if callable(source):
        try:
            source = source()
        except Exception:  # noqa: BLE001 - never break the answer flow
            return None
    if not source:
        return None
    names = [str(n) for n in source if str(n or "").strip()]
    return names or None


def build_clarification(
    structured_request: Any,
    context: dict[str, Any] | None = None,
    available_documents: (
        list[str] | tuple[str, ...] | Any | None
    ) = None,
) -> Clarification:
    """Build ONE clarification question from a Step 3 StructuredRequest.

    Args:
        structured_request: The Step 3 :class:`StructuredRequest` (or its
            ``to_dict()`` form). CONSUMED, never re-parsed.
        context: Optional Step 1 conversation context (READ-ONLY). Used only
            to recognize that a follow-up is ALREADY resolved by the prior
            turn, so no unnecessary question is asked.
        available_documents: ACTUAL document names from the real store — a
            list/tuple OR a zero-arg callable returning them (LAZY: production
            callers pass a callable so clear queries never touch the store).
            Required for document clarification: candidates are never
            invented, so without real names no document question is emitted.

    Returns:
        A :class:`Clarification`. ``needed=False`` whenever the request is
        clear (or no supported ambiguity category has evidence). Never
        raises for malformed input.
    """

    request = _normalize_request(structured_request)
    if request is None:
        return _NO_CLARIFICATION

    # Only a document-information request can need clarification:
    # conversation/small-talk, out-of-scope and empty/malformed requests are
    # always clear (the engine never asks "what do you mean?" for them).
    if request["intent"] != "document_information":
        return _NO_CLARIFICATION

    language = request["language"]
    candidates: tuple[str, ...] = request["candidate_fields"]
    field = request["field"]
    target = request["target_document"]
    flagged = bool(request["ambiguity"] and request["clarification_needed"])

    # ---- 1) Ambiguous REFERENCE: unresolvable follow-up pronoun ---------
    # ("iska number batao" with no prior turn naming a document). Step 3
    # already inherits the target when the context resolves it; a missing
    # target on a pronoun follow-up means the reference itself is unclear.
    # This check comes first: one question per turn.
    if (
        request["is_follow_up"]
        and _has_reference_pronoun(request)
        and target is None
        and not _context_resolves_reference(context)
    ):
        names = _resolve_document_names(available_documents) or []
        options = tuple(names[:MAX_CLARIFICATION_OPTIONS])
        meta: dict[str, Any] = {"type": CLARIFICATION_REFERENCE}
        if len(names) > len(options):
            meta["omitted_candidates"] = names[len(options):]
        question = (
            _make_question("reference", options, language)
            if options
            else _make_question("reference_no_options", (), language)
        )
        return Clarification(
            needed=True,
            question=question,
            options=options,
            reason=CLARIFICATION_REFERENCE,
            language=language,
            target_document=None,
            candidates=candidates,
            meta=meta,
        )

    # ---- 4) Ambiguous CONSTRAINT: named field, bare constraint word ------
    # ("semester ka SGPA batao" -- WHICH semester?). The field itself is
    # resolved (no field ambiguity), but the query NAMES a constraint
    # dimension with a postposition and NO value. Evidence is self-contained
    # in the StructuredRequest (original_query + parsed constraints + field),
    # so this check runs BEFORE the Step-3-flag gate: Step 3 does not flag
    # such requests because their field resolves cleanly. Clarification ONLY
    # -- no filtering/reranking happens here (Step 5 owns execution).
    if field and not candidates:
        query_norm = re.sub(r"\s+", " ", request["original_query"].lower())
        asked = next(
            (
                key
                for key in _ASKABLE_CONSTRAINTS
                if key not in request["constraints"]
                and _BARE_CONSTRAINT_RE[key].search(query_norm)
            ),
            None,
        )
        if asked:
            return Clarification(
                needed=True,
                question=_make_question(
                    "constraint",
                    (),
                    language,
                    field_labeler=str,
                    constraint=_CONSTRAINT_LABELS.get(asked, asked),
                ),
                options=(),
                reason=CLARIFICATION_CONSTRAINT,
                language=language,
                target_document=target,
                candidates=(),
                meta={"type": CLARIFICATION_CONSTRAINT, "constraint": asked},
            )

    # ---- 2) Ambiguous DOCUMENT: unresolved target + real names -----------
    # ("meri details batao" while the store holds several documents). A
    # details-style request with NO field and NO resolvable target is
    # document-ambiguous when the user referred to their own documents
    # (owner=self) or Step 3 flagged the request. This check runs BEFORE the
    # Step-3-flag gate because its evidence is self-contained: Step 3 leaves
    # such requests unflagged when only the owner constraint is present.
    # Fires only with ACTUAL document names (resolved lazily): without them
    # nothing is invented and the existing answer flow handles the request
    # honestly as before. Resolved-target requests ("ticket ki details
    # batao") never enter.
    if (
        not candidates
        and field is None
        and target is None
        and request["operation"] in ("retrieve_details", "summary")
        and not _context_resolves_reference(context)
        and _is_bare_details_request(request["original_query"])
        and (
            flagged
            or request["constraints"].get("owner") == "self"
        )
    ):
        names = _resolve_document_names(available_documents)
        if names and len(names) >= 2:
            options = tuple(names[:MAX_CLARIFICATION_OPTIONS])
            meta: dict[str, Any] = {"type": CLARIFICATION_DOCUMENT}
            if len(names) > len(options):
                meta["omitted_candidates"] = names[len(options):]
            return Clarification(
                needed=True,
                question=_make_question("document", options, language),
                options=options,
                reason=CLARIFICATION_DOCUMENT,
                language=language,
                target_document=None,
                candidates=candidates,
                meta=meta,
            )

    # Step 3 must have flagged ambiguity for the remaining category (field):
    # if not, the existing answer flow handles the request honestly.
    if not flagged:
        return _NO_CLARIFICATION

    # ---- 3) Ambiguous FIELD: Step 3's actual candidates ------------------
    # ("number batao", "4 sem ka number batao", "mera result"). The question
    # carries only ACTUAL candidates; nothing is invented or chosen.
    if candidates:
        options = candidates[:MAX_CLARIFICATION_OPTIONS]
        meta = {"type": CLARIFICATION_FIELD}
        if len(candidates) > len(options):
            meta["omitted_candidates"] = list(candidates[len(options):])
        kind = "field_number" if any(
            word in request["original_query"].lower()
            for word in ("number", "no ", "numbar", "ank")
        ) else "field"
        return Clarification(
            needed=True,
            question=_make_question(kind, options, language),
            options=options,
            reason=CLARIFICATION_FIELD,
            language=language,
            target_document=target,
            candidates=candidates,
            meta=meta,
        )

    return _NO_CLARIFICATION


# ---------------------------------------------------------------------------
# Production helper (read-only store access, kept out of the pure engine)
# ---------------------------------------------------------------------------


def available_document_names() -> list[str] | None:
    """Actual document names from the real store (READ-ONLY, deduped).

    Returns ``None`` when the store cannot be opened so the caller skips
    document clarification instead of inventing options. Deterministic
    ordering (document id). Never raises.
    """

    try:
        from storage_engine import DB_PATH
    except ImportError:  # pragma: no cover - src.-mode imports
        try:
            from src.storage_engine import DB_PATH
        except Exception:  # noqa: BLE001
            return None
    try:
        import sqlite3

        conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
        try:
            rows = conn.execute(
                "SELECT file_name FROM documents ORDER BY id"
            ).fetchall()
        finally:
            conn.close()
        names: list[str] = []
        seen: set[str] = set()
        for (name,) in rows:
            if name and name not in seen:
                seen.add(name)
                names.append(name)
        return names or None
    except Exception:  # noqa: BLE001 - read-only helper must never raise
        return None
