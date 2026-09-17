"""
Phase 9: Deterministic query-relevance gate.
============================================

A lightweight pre-check that runs BEFORE any retrieval or LLM work and
decides whether a user query plausibly asks for information that could
exist in the user's locally stored documents. Clearly unrelated queries
(weather, general knowledge, coding, math, jokes) are rejected in
microseconds with zero SQLite/Chroma/Ollama cost.

Design rules:

* Fully deterministic: no LLM, no embeddings, no network. Same input ->
  same decision, always.
* Allowlist-first: a small set of well-known general-knowledge /
  chit-chat / coding / math / weather patterns is rejected outright
  even when the query contains a document-y word (e.g. "capital of
  France" contains "of"... no document word, but "who is the prime
  minister of India and where is his office file?" would otherwise
  sneak through the document heuristics).
* Otherwise PASS: any query containing a personal reference ("my",
  "mera"), a document-type noun ("ticket", "form"), support vocabulary
  ("roll number", "details"), an explicit document name / ID, or an
  explicit long digit value passes. The gate's purpose is to block the
  obviously-unrelated tail (weather, jokes, math, coding, general
  knowledge), not to understand the query.
* Fail-open where it matters: queries mentioning a stored document
  name / document ID always pass. Everything else defaults through the
  same lexical checks -- the gate is deliberately conservative about
  REJECTING only clearly-unrelated queries so genuine document
  questions are never lost.
"""

from __future__ import annotations

import re

__all__ = ["check_query_relevance", "is_document_related"]


# ---------------------------------------------------------------------------
# Lexical resources (closed, generic, language-level)
# ---------------------------------------------------------------------------

#: Personal-reference words -- the user asking about THEIR OWN documents.
_PERSONAL_REFS: frozenset[str] = frozenset(
    {"my", "mera", "meri", "mere", "apna", "apni", "hamara", "hamari", "mer"}
)

#: Question / information-seeking verbs and auxiliaries (EN + Hinglish).
_INFO_SEEKING: frozenset[str] = frozenset(
    {
        "what", "which", "where", "when", "who", "how", "is", "are", "was",
        "were", "does", "do", "did", "can", "could", "tell", "show", "find",
        "list", "give", "explain", "summarize", "summarise", "summary",
        "contain", "contains", "written", "batao", "bata", "bataiye",
        "likha", "likhi", "hai", "hain", "kya", "konsa", "konsi",
    }
)

#: Document-type nouns (open generic English/Hinglish vocabulary, NOT a
#: document schema). Matched as tokens so "ticket" matches but
#: "tick" does not.
_DOC_NOUNS: frozenset[str] = frozenset(
    {
        "document", "documents", "doc", "docs", "file", "files", "pdf",
        "card", "ticket", "form", "bill", "statement", "certificate",
        "letter", "receipt", "invoice", "pass", "licence", "license",
        "permit", "report", "record", "marksheet", "application", "id",
    }
)

#: Generic support vocabulary: field-like words and page furniture that
#: plausibly reference document contents. Open set at the language level.
_SUPPORT_VOCAB: frozenset[str] = frozenset(
    {
        "roll", "number", "no", "enrollment", "enrolment", "transaction",
        "txn", "consumer", "pnr", "sgpa", "cgpa", "marks", "dob", "birth",
        "name", "date", "phone", "mobile", "contact", "email", "amount",
        "due", "total", "fee", "address", "city", "train", "coach",
        "quota", "berth", "seat", "exam", "candidate", "student",
        "applicant", "details", "detail", "important", "information",
        "page", "type", "issued", "valid", "expiry", "fields", "field",
        "metadata", "value", "values", "contents", "content",
    }
)

#: Devanagari block (Hindi queries pass the script check directly).
_DEVANAGARI = re.compile(r"[\u0900-\u097f]")

#: Explicit long digit run (>= 8 digits) -- plausibly a document
#: identifier/value lookup (PNR, roll number, transaction id, ...).
_VALUE_HINT = re.compile(r"\b\d{8,}\b")

#: Allowlisted REJECT patterns (matched on the lowercased, stripped
#: query). These are clearly-unrelated categories that could otherwise
#: collide with the pass heuristics (e.g. "who is the prime minister of
#: India" contains "india" ... which no doc noun matches, but "who is
#: my prime minister" style phrasings with a personal ref must still
#: lose to general knowledge). First match wins.
_REJECT_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern)
    for pattern in (
        # weather
        r"\bweather\b", r"\btemperature\b", r"\bforecast\b",
        r"\brain(ing)?\b", r"\bmausam\b",
        # general knowledge / people
        r"^who is\b", r"^who was\b", r"^who are\b",
        r"\bprime minister\b", r"\bpresident of\b", r"\bchief minister\b",
        r"\bcapital of\b", r"\bcapital city\b",
        r"\bnational (animal|bird|flower|anthem|game)\b",
        r"\bcurrency of\b", r"\bpopulation of\b",
        # chit-chat
        r"\bjoke\b", r"\bfunny\b", r"\bstory\b", r"\bpoem\b", r"\bquote\b",
        r"\bshayari\b", r"\bgaana\b", r"\bsing\b",
        # math
        r"\bwhat is\s+[\d(]", r"\bcalculate\b", r"\bsquare root\b",
        r"\bpercentage of\b", r"\d+\s*[\+\-\*x\/%]\s*\d+",
        # coding
        r"\bwrite (me )?(a |an )?(python|java|c\+\+|javascript|code|program|script|function)\b",
        r"\bwrite code\b", r"\bdebug\b", r"\bcompile\b",
        r"\b(python|java|javascript) program\b",
        # cooking / how-to
        r"\bcook\b", r"\brecipe\b", r"\bboil\b", r"\bbake\b",
        r"\bhow (do|to) (i |you )?(cook|make|prepare|fix|build)\b",
        # time / system
        r"\btoday'?s date\b", r"\bcurrent time\b", r"\bwhat time\b",
        r"\bweather today\b",
    )
)


def _tokens(text: str) -> list[str]:
    """Lowercase alphabetic tokens (digits kept for value checks)."""

    return re.findall(r"[a-z0-9]+", text.lower())


def is_document_related(query: str, stored_file_names: list[str] | None = None) -> bool:
    """True when the query plausibly concerns the stored documents.

    Deterministic lexical decision -- see module docstring. Optional
    ``stored_file_names`` lets callers force-pass explicit document
    references even when the file name is not generic vocabulary.
    """

    text = str(query or "").strip()
    if not text:
        return False

    # 1) Explicit allowlisted rejects win over everything (general
    #    knowledge phrased with document-ish words must not sneak in).
    lowered = text.lower()
    for pattern in _REJECT_PATTERNS:
        if pattern.search(lowered):
            return False

    # 2) Explicit document name / ID reference -> pass.
    if stored_file_names:
        for name in stored_file_names:
            stem = str(name).rsplit(".", 1)[0].strip().lower()
            if len(stem) >= 4 and re.search(re.escape(stem), lowered):
                return True

    # 3) Devanagari script -> Hindi document question, pass.
    if _DEVANAGARI.search(text):
        return True

    tokens = _tokens(text)

    # 4) Explicit long digit value -> pass (value lookup).
    if _VALUE_HINT.search(text):
        return True

    token_set = set(tokens)

    # 5) Personal reference ("my" / "mera") -> pass: the user is asking
    #    about their own stored material.
    if token_set & _PERSONAL_REFS:
        return True

    # 6) Document-type noun -> pass.
    if token_set & _DOC_NOUNS:
        return True

    # 7) Support vocabulary ("roll number", "transaction id", bare
    #    field fragments) -> pass: a field-name fragment is plausibly
    #    a document-scope query even without a question word.
    if token_set & _SUPPORT_VOCAB:
        return True

    # 8) Information-seeking + Hindi/Devanagari transliteration handled
    #    above; anything else is outside document scope.
    return False


def check_query_relevance(
    query: str,
    stored_file_names: list[str] | None = None,
) -> dict:
    """Structured relevance decision for the answer engine.

    Returns::

        {
            "related": bool,
            "reason": "document_related" | "query_outside_document_scope",
            "method": "allowlist_reject" | "doc_name" | "script"
                      | "value_hint" | "personal_ref" | "doc_noun"
                      | "support_vocab" | "no_document_signal",
        }

    ``method`` records WHY the decision was made (debug/provenance
    only; never shown in the main UI answer area).
    """

    text = str(query or "").strip()
    if not text:
        return {
            "related": False,
            "reason": "query_outside_document_scope",
            "method": "empty_query",
        }

    lowered = text.lower()
    for pattern in _REJECT_PATTERNS:
        if pattern.search(lowered):
            return {
                "related": False,
                "reason": "query_outside_document_scope",
                "method": "allowlist_reject",
            }

    if stored_file_names:
        for name in stored_file_names:
            stem = str(name).rsplit(".", 1)[0].strip().lower()
            if len(stem) >= 4 and re.search(re.escape(stem), lowered):
                return {
                    "related": True,
                    "reason": "document_related",
                    "method": "doc_name",
                }

    if _DEVANAGARI.search(text):
        return {
            "related": True,
            "reason": "document_related",
            "method": "script",
        }

    if _VALUE_HINT.search(text):
        return {
            "related": True,
            "reason": "document_related",
            "method": "value_hint",
        }

    tokens = set(_tokens(text))

    if tokens & _PERSONAL_REFS:
        return {
            "related": True,
            "reason": "document_related",
            "method": "personal_ref",
        }

    if tokens & _DOC_NOUNS:
        return {
            "related": True,
            "reason": "document_related",
            "method": "doc_noun",
        }

    if tokens & _SUPPORT_VOCAB:
        return {
            "related": True,
            "reason": "document_related",
            "method": "support_vocab",
        }

    return {
        "related": False,
        "reason": "query_outside_document_scope",
        "method": "no_document_signal",
    }
