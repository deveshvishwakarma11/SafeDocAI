"""
Phase 4: Local Query Router and Retrieval Layer.
================================================

Routes a natural-language query to one of two LOCAL retrieval backends:

* EXACT    -> SQLite ``extracted_metadata`` (deterministic field lookup)
* SEMANTIC -> ChromaDB ``safedoc_documents`` vector search (local
  all-MiniLM-L6-v2 embeddings)

Design rules (Phase 4 contract):

* Deterministic wherever practical: the same query always produces the
  same route and the same results for the same database state.
* No hardcoded document-type field schema. Field recognition combines
  structural query patterns with the DYNAMIC set of field names actually
  present in ``extracted_metadata``, plus a small generic-English
  synonym map (e.g. "date of birth" <-> "dob"). Arbitrary future field
  names are matched without code changes.
* No answer generation: this module is retrieval only. Results carry
  full provenance (source store, document identity, field/chunk) so a
  future answer-generation layer never has to guess.
* No invention: if nothing matches, an explicit empty/insufficient
  state is returned instead of a fabricated value.
* Fallback: an EXACT-looking query with no metadata match may fall back
  to semantic retrieval; the fallback is always recorded.
* 100% local: SQLite + ChromaDB + local embeddings. No cloud/API/network
  dependency.

Main entry point: :func:`route_query`.
"""

from __future__ import annotations

import re
import sqlite3
import time
from typing import Any

try:  # src/ on sys.path (pipeline style)
    from language import transliterate_for_matching
    from storage_engine import (
        get_chroma_collection,
        get_db_connection,
        get_embedding_model,
    )
except ImportError:  # project root on sys.path (test/tooling style)
    from src.language import transliterate_for_matching
    from src.storage_engine import (
        get_chroma_collection,
        get_db_connection,
        get_embedding_model,
    )


__all__ = [
    "ROUTE_EXACT",
    "ROUTE_SEMANTIC",
    "CLASS_EXACT",
    "CLASS_SEMANTIC",
    "CLASS_UNKNOWN",
    "SEMANTIC_USEFUL_MAX_DISTANCE",
    "INTENT_QUALIFIED_MAX_DISTANCE",
    "INTENT_STRONG_MIN_SCORE",
    "INTENT_WEAK_MIN_SCORE",
    "INTENT_WEIGHT",
    "INTENT_SCORE_MIN",
    "classify_query",
    "detect_document_intent",
    "route_query",
]


# ============================================================
# Route / classification constants
# ============================================================

ROUTE_EXACT = "exact"
ROUTE_SEMANTIC = "semantic"

CLASS_EXACT = "exact"
CLASS_SEMANTIC = "semantic"
CLASS_UNKNOWN = "unknown"

#: Chroma stores normalized embeddings with the default L2 metric, so the
#: reported distance is squared L2 in [0, 4] (= 2 - 2*cosine). Related
#: chunks land well below ~1.2; unrelated text lands near 2.0. Tunable;
#: only used to mark the result state, never to fabricate content.
SEMANTIC_USEFUL_MAX_DISTANCE = 1.2

#: A token of this many digits or more is treated as an explicit value
#: lookup signal (PNR, roll number, transaction id, ...).
_VALUE_HINT_MIN_DIGITS = 8

# ------------------------------------------------------------
# Phase 6: document-intent detection / reranking configuration
# ------------------------------------------------------------

#: "head noun" endings that typically denote a document reference in a
#: query. Generic English morphology, NOT a document-type schema: any
#: noun with these endings is treated as a *candidate* intent phrase and
#: then matched against the user's actual document names and metadata.
_DOCUMENT_HEAD_ENDINGS: tuple[str, ...] = (
    "card",
    "ticket",
    "form",
    "bill",
    "statement",
    "certificate",
    "letter",
    "receipt",
    "invoice",
    "pass",
    "licence",
    "license",
    "permit",
    "report",
    "record",
    "marksheet",
    "application",
    "id",
)

#: Intent words that should not anchor a phrase on their own.
#: Hinglish possessives/postpositions ("mera/meri/mere", "ka/ki/ke",
#: "mein") behave like "my"/"the" (Phase 8): without them, "Meri
#: railway ticket" anchors as "meri railway ticket" and never matches
#: the actual document.
_INTENT_STOP_TOKENS = {
    "my", "the", "a", "an", "this", "that", "my", "in", "on", "from",
    "mera", "meri", "mere", "ka", "ki", "ke", "mein", "me", "of",
}

#: Minimum Dice coefficient for a phrase to be considered a STRONG
#: document-intent match (file-name stems are compared token-wise).
INTENT_STRONG_MIN_SCORE = 0.6

#: Minimum Dice coefficient for a WEAK match (routed via metadata/lexical
#: agreement, weighted down during reranking).
INTENT_WEAK_MIN_SCORE = 0.34

#: Weight of the document-intent component in the final rerank score.
INTENT_WEIGHT = 0.40

#: Minimum combined rerank score for a chunk to be preferred over a
#: non-intent chunk when a document intent was detected.
INTENT_SCORE_MIN = 0.0

#: Intent-qualified semantic threshold. The BASELINE threshold above stays
#: the default for all unrestricted searches. When a STRONG document intent
#: exists and the restricted search finds no chunk within the baseline, the
#: search may be retried with this wider bound -- but ONLY inside the
#: intent-matching documents (the restriction already guarantees no other
#: document can leak in). Real scanned OCR embeds farther than clean text;
#: 1.6 stays well below the unrelated-text region (~2.0). Recorded in the
#: document_intent provenance block whenever applied.
INTENT_QUALIFIED_MAX_DISTANCE = 1.6

#: Rerank component weights (semantic + lexical + phrase + identifier
#: must sum to 1.0 - INTENT_WEIGHT; checked at import time).
_INTENT_COMPONENT = INTENT_WEIGHT
_SEMANTIC_COMPONENT = 0.30
_LEXICAL_COMPONENT = 0.20
_PHRASE_COMPONENT = 0.05
_IDENTIFIER_COMPONENT = 0.05

assert abs(
    (_INTENT_COMPONENT + _SEMANTIC_COMPONENT + _LEXICAL_COMPONENT
     + _PHRASE_COMPONENT + _IDENTIFIER_COMPONENT) - 1.0
) < 1e-9, "rerank component weights must sum to 1.0"

#: Generic English field vocabulary shared across document types. These
#: are LANGUAGE synonyms, not document-specific schema: the actual fields
#: come dynamically from the database.
_SYNONYM_GROUPS: tuple[frozenset[str], ...] = (
    frozenset({"date of birth", "dob", "birth date", "d o b"}),
    frozenset({"roll", "roll number", "roll no", "rollno", "registration number"}),
    frozenset({"phone", "phone number", "contact number", "mobile", "mobile number", "contact"}),
    frozenset({"email", "email id", "e mail", "mail id"}),
    frozenset({"sgpa", "semester sgpa", "semester gpa"}),
    frozenset({"pnr", "pnr number", "pnr no"}),
    frozenset({"consumer number", "consumer no", "consumer id"}),
    frozenset({"enrollment id", "enrolment id", "enrollment number", "enrolment number"}),
    frozenset({"transaction id", "transaction number", "txn id"}),
    frozenset({"name", "full name", "student name", "candidate name"}),
    frozenset({"total marks", "marks total", "total score"}),
)

#: Tokens ignored at the edges of an extracted field hint.
#: Hinglish edge fillers ("kya", "hai", "batao", ...) added in Phase 8
#: so "roll number kya hai" cleans to the pure field hint "roll number".
_HINT_EDGE_FILLER = {
    "my", "the", "a", "an", "please", "thanks", "thank",
    "kya", "hai", "hain", "h", "batao", "bata", "bataiye",
    "thi", "tha", "me", "mein", "ka", "ki", "ke",
}

# Semantic-intent cues: open-ended, exploratory questions. These are
# intent patterns, deliberately NOT field names. Hinglish/Hindi word
# forms are included so language-mixed queries classify correctly
# (Phase 8): "batao"/"bataiye" = tell, "detail" = details, "likha" =
# written, "baare" = about.
_SEMANTIC_CUES: tuple[str, ...] = (
    r"\bsummar(y|ise|ize)\b",
    r"\bexplain\b",
    r"\bdescribe\b",
    r"\boverview\b",
    r"\bwhat\b[^?]*\b(says?|contain(s|ed)?|cover(s|ed)?|mention(s|ed)?|state(s|d)?)\b",
    r"\btell me (about|what)\b",
    r"\bimportant details\b",
    r"\bkey (points|details|information)\b",
    r"\bterms?\b",
    r"\bcontents?\b",
    r"\bmeaning\b",
    r"\bwhat.+(is|are).+(about|covered|included)\b",
    r"\b(show|give) me (the|an?) (summary|overview|details)\b",
    # Hinglish exploratory cues (Roman script)
    r"\bbata(o|iye|ayiye|ayye)\b",
    r"\bbatao\b",
    r"\b(bat\w*) detail\w*\b",
    r"\bimportant detail\w*\b",
    r"\bsab(\s+\w+)? (baare|bare)\b",
    r"\bke baare\b",
    r"\bme\w* kya likha\b",
    r"\bme(in)? kya\b",
    r"\bkya likha\b",
    r"\bkya hai (is|is document)\b",
    r"\bmujhe\b[^?]*\bbata\w*\b",
)

# Exact-intent cues: a specific attribute of a known document is asked for.
# Hinglish "kya hai" ("what is") added in Phase 8 so mixed-language
# field lookups stay on the deterministic SQLite path.
_EXACT_CUES: tuple[str, ...] = (
    r"\bwhat('s| is| was| are| were)\b",
    r"\bwhich\b",
    r"\b(who|when|where) (is|was|are|were)\b",
    r"\b(find|show|get|give)\b",
    r"\bhow much\b",
    r"\bmy\s+[a-z0-9]",
    # Hinglish exact cues (Roman script)
    r"\bkya hai\b",
    r"\bkya h\b",
    r"\bkya tha\b",
    r"\bmer(a|i|e)\s+[a-z0-9]",
    r"\bkonsa\b",
    r"\bkitna\b",
)

# Exact-intent cues handled above (Phase 8 Hinglish included).

_HINT_PATTERN_MY = re.compile(r"\bmy\s+([a-z0-9][a-z0-9 &'/\-\.]*)", re.IGNORECASE)

# Hinglish possessive "mera/meri/mere" behaves like "my" (Phase 8):
# "Mera roll number kya hai?" -> hint "roll number".
_HINT_PATTERN_MERA = re.compile(
    r"\bmer(a|i|e)\s+([a-z0-9][a-z0-9 &'/\-\.]*)", re.IGNORECASE
)
_HINT_PATTERN_THE = re.compile(
    r"\bthe\s+([a-z0-9][a-z0-9 &'/\-\.]*?)(?=\s+(?:in|of|from|on|for)\b|\s*$)",
    re.IGNORECASE,
)
_HINT_PATTERN_VERB = re.compile(
    r"\b(?:find|show|get|give)\s+(?:me\s+)?(?:my\s+|the\s+)?([a-z0-9][a-z0-9 &'/\-\.]*)",
    re.IGNORECASE,
)
_VALUE_HINT_PATTERN = re.compile(rf"\b\d{{{_VALUE_HINT_MIN_DIGITS},}}\b")
_DOC_ID_PATTERN = re.compile(r"\b(?:document|doc)\s*#?\s*(\d{1,6})\b", re.IGNORECASE)


# ============================================================
# Normalization helpers
# ============================================================


def _normalize_text(text: str) -> str:
    """Lowercase, drop punctuation, collapse whitespace (keep letters/digits)."""

    lowered = str(text).lower()
    lowered = re.sub(r"[^a-z0-9\s&/]+", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def _stem(token: str) -> str:
    """Tiny deterministic stem: strip a trailing plural 's' (not 'ss')."""

    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _tokens(text: str) -> frozenset[str]:
    return frozenset(_stem(tok) for tok in _normalize_text(text).split() if tok)


# ------------------------------------------------------------
# Phase 6: document-intent detection + lexical reranking
# ------------------------------------------------------------


def _dice(a: frozenset[str], b: frozenset[str]) -> float:
    """Dice coefficient between two token sets (1.0 == identical)."""

    if not a or not b:
        return 0.0
    return 2.0 * len(a & b) / (len(a) + len(b))


def _file_name_tokens(name: str) -> frozenset[str]:
    """Tokens of a file name with camel-case split, e.g.
    "ID Card.pdf" -> {"id", "card"}, "4143027140.pdf" -> {"4143027140"}."""

    stem = str(name).rsplit(".", 1)[0]
    stem = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", stem)
    return frozenset(tok for tok in _normalize_text(stem).split() if tok)


def _intent_phrases(query: str) -> list[str]:
    """Deterministically extract candidate document-intent phrases.

    Anchors on the LAST head-noun token of each candidate phrase so both
    "ID card" (head=card) and "railway ticket" (head=ticket) anchor
    correctly. Preceding tokens are prepended greedily until a stop word
    or a long digit run, yielding phrases like "railway ticket",
    "application form", "4143027140 pdf". Purely structural: no fixed
    document-type vocabulary.
    """

    tokens = _normalize_text(query).split()
    phrases: list[str] = []
    for index, token in enumerate(tokens):
        if token in _INTENT_STOP_TOKENS:
            continue
        if re.fullmatch(r"\d{8,}", token):
            # An explicit long identifier acts as its own file-stem intent.
            phrases.append(token)
            continue
        if not token.endswith(_DOCUMENT_HEAD_ENDINGS):
            continue
        start = index
        while start > 0:
            prev = tokens[start - 1]
            if prev in _INTENT_STOP_TOKENS or re.fullmatch(r"\d{8,}", prev):
                break
            start -= 1
        phrase = " ".join(tokens[start : index + 1])
        if phrase and phrase not in phrases:
            phrases.append(phrase)
    return phrases


def detect_document_intent(
    query: str,
    connection: sqlite3.Connection,
) -> dict[str, Any]:
    """Detect which stored documents the query is about (deterministic).

    Candidate phrases come from generic head-noun morphology; they are
    scored against the user's ACTUAL document file names and stored
    metadata field names -- never against a fixed document-type schema.

    Returns::

        {
          "intent_detected": bool,
          "phrases": [...],
          "candidates": [
              {"document_id", "file_name", "score", "strength",
               "match_via"}, ...
          ],
          "strong": [...], "weak": [...],
          "restrict_to": [document_id, ...] | None,
        }
    """

    phrases = _intent_phrases(query)
    if not phrases:
        return {
            "intent_detected": False,
            "phrases": [],
            "candidates": [],
            "strong": [],
            "weak": [],
            "restrict_to": None,
        }

    # A phrase that belongs to a generic field-synonym group ("enrollment
    # id", "transaction id", "consumer number", ...) names a FIELD the
    # query asks about, not a document -- excluding it prevents false
    # document restrictions (e.g. "enrollment id" must not restrict to
    # "ID Card.pdf" just because the file name contains the token "id").
    field_like = {
        phrase for phrase in phrases if len(_synonym_variants(phrase)) > 1
    }
    phrases = [phrase for phrase in phrases if phrase not in field_like]
    if not phrases:
        return {
            "intent_detected": False,
            "phrases": [],
            "field_like_phrases": sorted(field_like),
            "candidates": [],
            "strong": [],
            "weak": [],
            "restrict_to": None,
        }

    rows = connection.execute(
        "SELECT id, file_name FROM documents ORDER BY id"
    ).fetchall()

    # Pre-compute per-document token sets (file name + stored fields).
    doc_tokens: dict[int, frozenset[str]] = {}
    doc_field_tokens: dict[int, frozenset[str]] = {}
    for doc_id, file_name in rows:
        doc_id = int(doc_id)
        name_toks = _file_name_tokens(file_name)
        doc_tokens[doc_id] = name_toks
        field_toks: set[str] = set()
        for (field_name,) in connection.execute(
            "SELECT DISTINCT field_name FROM extracted_metadata WHERE document_id = ?",
            (doc_id,),
        ):
            field_toks.update(_normalize_text(field_name).split())
        doc_field_tokens[doc_id] = frozenset(field_toks)

    scored: dict[int, dict[str, Any]] = {}
    for phrase in phrases:
        phrase_tokens = frozenset(phrase.split())
        for doc_id, file_name in rows:
            doc_id = int(doc_id)
            name_score = _dice(phrase_tokens, doc_tokens[doc_id])
            field_score = _dice(phrase_tokens, doc_field_tokens[doc_id])

            # Structural match: the phrase's head noun appears in the file
            # name ("card" in "ID Card") or the phrase covers the whole
            # name ("application form" vs "ApplicationForm.pdf").
            name_head_hit = any(
                tok in doc_tokens[doc_id] and len(tok) >= 3
                for tok in phrase_tokens
            )
            name_cover = (
                phrase_tokens and phrase_tokens <= doc_tokens[doc_id]
            )
            combined = max(name_score, field_score)
            if name_head_hit:
                combined = max(combined, 0.5 + 0.5 * name_score)
            if name_cover:
                combined = max(combined, 0.85)

            if combined <= 0.0:
                continue
            prior = scored.get(doc_id)
            if prior is None or combined > prior["score"]:
                via = (
                    "file_name"
                    if name_score >= field_score or name_head_hit or name_cover
                    else "metadata"
                )
                scored[doc_id] = {
                    "document_id": doc_id,
                    "file_name": file_name,
                    "score": round(min(combined, 1.0), 4),
                    "strength": "strong" if combined >= INTENT_STRONG_MIN_SCORE else "weak",
                    "match_via": via,
                }

    candidates = sorted(
        scored.values(), key=lambda c: (-c["score"], c["document_id"])
    )
    strong = [c for c in candidates if c["strength"] == "strong"]
    weak = [c for c in candidates if c["strength"] == "weak"]

    # Restriction policy: strong candidates dominate (they keep generic
    # phrase collisions like "photo id" from outranking the requested
    # document); weak-only candidates still restrict but are recorded as
    # such. No candidates -> global retrieval, byte-identical behavior.
    restrict_ids: list[int] | None = None
    if strong:
        restrict_ids = [c["document_id"] for c in strong]
    elif weak:
        restrict_ids = [c["document_id"] for c in weak]

    return {
        "intent_detected": bool(candidates),
        "phrases": phrases,
        "candidates": candidates,
        "strong": strong,
        "weak": weak,
        "restrict_to": restrict_ids,
    }


def _rerank_chunks(
    query: str,
    chunks: list[dict[str, Any]],
    intent: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Deterministically combine semantic distance with intent + lexical signals.

    Components (weights above):
    * semantic  : normalized usefulness of the Chroma distance
    * intent    : document-intent score of the chunk's document
    * lexical   : Dice token overlap between query and chunk text
    * phrase    : exact normalized-substring match of the query in the chunk
    * identifier: shared >= 6-char alphanumeric tokens (PNR/roll style)

    Input order (Chroma distance order) breaks ties; the returned list is
    always a NEW list (chunks are copied, never mutated).
    """

    if not chunks:
        return []

    max_distance = max(
        (float(c["distance"]) for c in chunks if c.get("distance") is not None),
        default=None,
    )
    min_distance = min(
        (float(c["distance"]) for c in chunks if c.get("distance") is not None),
        default=None,
    )

    query_tokens = _tokens(query)
    query_norm = _normalize_text(query)
    query_identifiers = {
        tok
        for tok in re.findall(r"\S+", _normalize_text(query))
        if len(tok) >= 6 and any(ch.isdigit() for ch in tok) and any(ch.isalpha() or ch.isdigit() for ch in tok)
    }

    intent_scores: dict[int, float] = {}
    intent_candidates: dict[int, dict[str, Any]] = {}
    intent_present = bool(intent and intent.get("intent_detected"))
    if intent_present:
        for cand in intent.get("candidates", []):
            intent_scores[int(cand["document_id"])] = float(cand["score"])
            intent_candidates[int(cand["document_id"])] = cand

    active_weights = {
        "intent": _INTENT_COMPONENT if intent_present else 0.0,
        "semantic": _SEMANTIC_COMPONENT,
        "lexical": _LEXICAL_COMPONENT,
        "phrase": _PHRASE_COMPONENT,
        "identifier": _IDENTIFIER_COMPONENT,
    }
    # Renormalize when no intent was detected so scores stay comparable.
    weight_sum = sum(active_weights.values())

    scored: list[tuple[float, int, dict[str, Any]]] = []
    for position, chunk in enumerate(chunks):
        distance = chunk.get("distance")
        if distance is None or max_distance is None or max_distance == min_distance:
            semantic_score = 1.0 if max_distance is None else 1.0
        else:
            semantic_score = 1.0 - (float(distance) - min_distance) / (
                max_distance - min_distance
            )

        raw_id = chunk.get("document_id")
        raw_id_int = (
            int(raw_id)
            if raw_id is not None and str(raw_id).strip().isdigit()
            else None
        )
        intent_score = (
            intent_scores.get(raw_id_int, 0.0)
            if intent_present and raw_id_int is not None
            else 0.0
        )

        text_tokens = _tokens(str(chunk.get("text", "")))
        lexical_score = _dice(query_tokens, text_tokens)
        phrase_score = (
            1.0
            if len(query_norm) >= 8 and query_norm in _normalize_text(str(chunk.get("text", "")))
            else 0.0
        )
        chunk_identifiers = {
            tok
            for tok in re.findall(r"\S+", _normalize_text(str(chunk.get("text", ""))))
            if len(tok) >= 6 and any(ch.isdigit() for ch in tok)
        }
        identifier_score = (
            1.0 if query_identifiers and (query_identifiers & chunk_identifiers) else 0.0
        )

        total = (
            active_weights["intent"] * intent_score
            + active_weights["semantic"] * semantic_score
            + active_weights["lexical"] * lexical_score
            + active_weights["phrase"] * phrase_score
            + active_weights["identifier"] * identifier_score
        ) / weight_sum

        enriched = dict(chunk)
        enriched["intent_match"] = (
            dict(intent_candidates[raw_id_int])
            if intent_present and raw_id_int is not None and raw_id_int in intent_candidates
            else None
        )
        enriched["rerank_score"] = round(total, 4)
        enriched["rerank_components"] = {
            "intent": round(intent_score, 4),
            "semantic": round(semantic_score, 4),
            "lexical": round(lexical_score, 4),
            "phrase": round(phrase_score, 4),
            "identifier": round(identifier_score, 4),
        }
        enriched["rerank_weights"] = {k: round(v / weight_sum, 4) for k, v in active_weights.items()}
        scored.append((total, position, enriched))

    # Stable: equal scores keep the original (semantic-distance) order.
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [item[2] for item in scored]


def _filename_match_variants(name: str) -> set[str]:
    """Case/punctuation-tolerant variants of a file name for filter matching."""

    low = str(name).strip().lower()
    variants = {low}
    spaced = re.sub(r"[_\-]+", " ", low)
    variants.add(spaced)
    for candidate in (low, spaced):
        if "." in candidate:
            stem = candidate.rsplit(".", 1)[0].strip()
            if len(stem) >= 4:
                variants.add(stem)
    return {variant for variant in variants if variant}


# ============================================================
# Field hint extraction and dynamic field resolution
# ============================================================


def _clean_hint(raw: str) -> str | None:
    """Strip edge filler tokens from a raw field hint."""

    tokens = _normalize_text(raw).split()
    while tokens and tokens[0] in _HINT_EDGE_FILLER:
        tokens.pop(0)
    while tokens and tokens[-1] in _HINT_EDGE_FILLER:
        tokens.pop()
    if not tokens:
        return None
    return " ".join(tokens)


def extract_field_hint(query: str) -> str | None:
    """Extract the attribute phrase a query asks about (normalized).

    "What was my 4th semester SGPA?"       -> "4th semester sgpa"
    "What is the PNR in 4143027140.pdf?"   -> "pnr"
    """

    # _HINT_PATTERN_MERA reports group(2) (group 1 is mera/meri/mere).
    for pattern, group_index in (
        (_HINT_PATTERN_MERA, 2),
        (_HINT_PATTERN_MY, 1),
        (_HINT_PATTERN_VERB, 1),
        (_HINT_PATTERN_THE, 1),
    ):
        match = pattern.search(query)
        if match:
            hint = _clean_hint(match.group(group_index))
            if hint:
                return hint
    return None


def _synonym_variants(hint: str) -> list[str]:
    """Expand a hint with its generic synonym group (if any)."""

    hint_tokens = _tokens(hint)
    variants = [hint]
    for group in _SYNONYM_GROUPS:
        for phrase in group:
            phrase_tokens = _tokens(phrase)
            if (
                _normalize_text(phrase) == _normalize_text(hint)
                or phrase_tokens <= hint_tokens
                or hint_tokens <= phrase_tokens
            ):
                variants.extend(sorted(group))
                break
    return variants


def _resolve_field_variants(
    field_hint: str | None,
    known_fields: list[str],
) -> list[str]:
    """Map a field hint onto concrete field names present in the database.

    Matching is structural (normalized equality, then token-subset) over the
    DYNAMIC field names plus the generic synonym groups, so arbitrary future
    field names work without code changes. Most specific matches come first.
    """

    if not field_hint:
        return []

    hint_tokens = _tokens(field_hint)
    if not hint_tokens:
        return []

    scored: dict[str, int] = {}

    for variant in _synonym_variants(field_hint):
        variant_norm = _normalize_text(variant)
        variant_tokens = _tokens(variant)

        for field in known_fields:
            field_norm = _normalize_text(field)
            field_tokens = _tokens(field)
            if not field_tokens:
                continue

            if field_norm == variant_norm:
                scored[field] = max(scored.get(field, 0), 1000)
            elif field_tokens <= variant_tokens or variant_tokens <= field_tokens:
                overlap = len(field_tokens & variant_tokens)
                scored[field] = max(scored.get(field, 0), 100 + overlap)

    if not scored:
        return []

    return sorted(scored, key=lambda field: (-scored[field], field.lower()))


def _value_hint(query: str) -> str | None:
    """First long digit run in the query (PNR / roll / transaction id...)."""

    match = _VALUE_HINT_PATTERN.search(query)
    return match.group(0) if match else None


# ============================================================
# Query classification
# ============================================================


def classify_query(
    query: str,
    known_fields: list[str] | None = None,
) -> dict[str, Any]:
    """Deterministically classify a query as EXACT / SEMANTIC / UNKNOWN.

    Order of precedence (strongest signal wins):

    1. An explicit long value token (>= 8 digits) is a value lookup -> EXACT.
    2. A semantic-intent cue (summarize / explain / what does it say ...) -> SEMANTIC.
    3. An exact-intent cue (what is / which / find my ...) -> EXACT.
    4. A field hint that resolves against known/dynamic field names -> EXACT.
    4b. A BARE field query ("roll number", "date of birth", "roll number
        kya hai") whose filler-stripped text resolves against the dynamic
        stored field names -> EXACT (same lookup path as case 4). Semantic
        cues keep absolute precedence, and queries naming no stored field
        ("PAN", "IFSC", "weather") keep their current route.
    5. Otherwise UNKNOWN (the router still attempts retrieval, recorded as such).
    """

    if known_fields is None:
        known_fields = []

    matched_cue: str | None = None
    field_hint = extract_field_hint(query)
    value = _value_hint(query)

    if value:
        classification = CLASS_EXACT
        matched_cue = "explicit value token"
    else:
        for cue in _SEMANTIC_CUES:
            if re.search(cue, query, re.IGNORECASE):
                classification = CLASS_SEMANTIC
                matched_cue = cue
                break
        else:
            for cue in _EXACT_CUES:
                if re.search(cue, query, re.IGNORECASE):
                    classification = CLASS_EXACT
                    matched_cue = cue
                    break
            else:
                hint_resolves = bool(
                    field_hint and _resolve_field_variants(field_hint, known_fields)
                )
                if hint_resolves:
                    classification = CLASS_EXACT
                    matched_cue = "field hint matches stored metadata"
                else:
                    classification = CLASS_UNKNOWN
                    matched_cue = None

    # Phase 10.1: bare field queries -------------------------------------
    # A short query like "roll number", "date of birth" or "roll number
    # kya hai" carries no personal/verb anchor ("my", "mera", "find",
    # "the"), so extract_field_hint() returns None for it even though
    # the WHOLE filler-stripped query names a stored metadata field.
    # When the dynamic resolver (live extracted_metadata fields + the
    # generic synonym groups) recognizes the cleaned query, the query
    # goes through the SAME deterministic exact path as the anchored
    # forms. Semantic-intent cues keep absolute precedence: an
    # open-ended query ("summarize my document") is never upgraded, and
    # queries naming no stored field ("PAN", "IFSC", "weather") keep
    # their current route.
    if classification in (CLASS_EXACT, CLASS_UNKNOWN) and field_hint is None:
        bare_hint = _clean_hint(query)
        if bare_hint and _resolve_field_variants(bare_hint, known_fields):
            field_hint = bare_hint
            if classification == CLASS_UNKNOWN:
                classification = CLASS_EXACT
                matched_cue = "bare field query matches stored metadata"

    return {
        "classification": classification,
        "matched_cue": matched_cue,
        "field_hint": field_hint,
        "value_hint": value,
    }


# ============================================================
# Document filtering
# ============================================================


def _extract_document_filter(
    query: str,
    connection: sqlite3.Connection,
) -> dict[str, Any] | None:
    """Detect a document restriction in the query (file name or document id)."""

    rows = connection.execute(
        "SELECT id, file_name FROM documents ORDER BY id"
    ).fetchall()

    # Strip ALL punctuation from the padded query: a trailing sentence
    # period ("...my ID card.") must not prevent matching the file-name
    # variant "id card". File names containing dots/dashes still match via
    # their spaced/stem variants ("4143027140.pdf" -> "4143027140 pdf").
    padded = " " + re.sub(r"[^\w\s]+", " ", str(query).lower()) + " "

    for _doc_id, file_name in rows:
        for variant in _filename_match_variants(file_name):
            if f" {variant} " in padded:
                return {"file_name": str(file_name)}

    match = _DOC_ID_PATTERN.search(query)
    if match:
        doc_id = int(match.group(1))
        if doc_id in {int(row[0]) for row in rows}:
            return {"document_id": doc_id}

    return None


# ============================================================
# Exact retrieval (SQLite)
# ============================================================


def _exact_lookup(
    connection: sqlite3.Connection,
    field_variants: list[str],
    value_hint: str | None,
    document_filter: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Deterministic SQLite lookup. Field and value signals are OR-ed; the
    document filter (if any) is AND-ed. Repeated values are never collapsed."""

    conditions: list[str] = []
    params: list[Any] = []

    field_clause: list[str] = []
    if field_variants:
        placeholders = ", ".join("?" for _ in field_variants)
        field_clause.append(f"LOWER(m.field_name) IN ({placeholders})")
        params.extend(variant.lower() for variant in field_variants)
    if value_hint:
        field_clause.append("LOWER(m.field_value) = LOWER(?)")
        params.append(value_hint)
    if field_clause:
        conditions.append("(" + " OR ".join(field_clause) + ")")

    if document_filter and "file_name" in document_filter:
        conditions.append("LOWER(d.file_name) = LOWER(?)")
        params.append(str(document_filter["file_name"]))
    elif document_filter and "document_id" in document_filter:
        conditions.append("d.id = ?")
        params.append(int(document_filter["document_id"]))

    if not conditions:
        return []

    sql = (
        "SELECT d.id, d.file_name, d.file_type, d.file_path, "
        "d.upload_timestamp, d.status, m.field_name, m.field_value "
        "FROM extracted_metadata AS m "
        "JOIN documents AS d ON d.id = m.document_id "
        "WHERE "
        + " AND ".join(conditions)
        + " ORDER BY m.document_id, m.id"
    )

    results: list[dict[str, Any]] = []
    for row in connection.execute(sql, params):
        results.append(
            {
                "source": "sqlite.extracted_metadata",
                "document_id": int(row[0]),
                "file_name": row[1],
                "file_type": row[2],
                "file_path": row[3],
                "upload_timestamp": row[4],
                "status": row[5],
                "field_name": row[6],
                "field_value": row[7],
            }
        )
    return results


# ============================================================
# Semantic retrieval (ChromaDB)
# ============================================================


def _semantic_lookup(
    query: str,
    top_k: int,
    document_filter: dict[str, Any] | None,
    max_distance: float = SEMANTIC_USEFUL_MAX_DISTANCE,
    intent_restriction: list[int] | None = None,
) -> tuple[list[dict[str, Any]], bool, str | None]:
    """Local ChromaDB vector search.

    Returns ``(results, available, note)``. ``available`` is False when the
    local vector store or embedding model cannot be used at all. Results are
    ordered by ascending distance and carry full document provenance.

    ``intent_restriction`` (Phase 6) restricts the search to the given
    document ids via a Chroma ``$in`` metadata filter; it is composed with
    (never replaces) any explicit document_filter.
    """

    try:
        collection = get_chroma_collection()
        total = collection.count()
        if total == 0:
            return [], True, "vector store is empty"

        model = get_embedding_model()
        embedding = model.encode(
            [query], normalize_embeddings=True
        )[0].tolist()

        query_kwargs: dict[str, Any] = {
            "query_embeddings": [embedding],
            "n_results": min(top_k, total),
            "include": ["documents", "metadatas", "distances"],
        }

        # Build the where clause: explicit filter (exact filename/doc-id)
        # and intent restriction are AND-composed.
        where_clause: dict[str, Any] | None = None
        if document_filter:
            if "file_name" in document_filter:
                where_clause = {"file_name": str(document_filter["file_name"])}
            else:
                where_clause = {"document_id": str(int(document_filter["document_id"]))}
        if intent_restriction:
            ids = sorted({str(int(doc_id)) for doc_id in intent_restriction})
            id_clause = {"document_id": {"$in": ids}}
            where_clause = (
                id_clause
                if where_clause is None
                else {"$and": [where_clause, id_clause]}
            )
        if where_clause is not None:
            query_kwargs["where"] = where_clause

        note: str | None = None
        if where_clause is not None:
            try:
                raw = collection.query(**query_kwargs)
            except Exception as exc:  # noqa: BLE001
                return [], True, f"document filter could not be applied: {exc}"
        else:
            raw = collection.query(**query_kwargs)

        documents = (raw.get("documents") or [[]])[0]
        metadatas = (raw.get("metadatas") or [[]])[0]
        distances = (raw.get("distances") or [[]])[0]

        results: list[dict[str, Any]] = []
        for index, text in enumerate(documents):
            metadata = metadatas[index] if index < len(metadatas) else {}
            distance = distances[index] if index < len(distances) else None
            raw_document_id = (metadata or {}).get("document_id")
            results.append(
                {
                    "source": "chroma.safedoc_documents",
                    "document_id": (
                        int(raw_document_id)
                        if raw_document_id is not None
                        and str(raw_document_id).strip().isdigit()
                        else None
                    ),
                    "file_name": (metadata or {}).get("file_name"),
                    "chunk_id": (metadata or {}).get("chunk_id"),
                    "text": text,
                    "distance": distance,
                    "useful": (
                        isinstance(distance, (int, float))
                        and distance <= max_distance
                    ),
                }
            )
        return results, True, note

    except Exception as exc:  # noqa: BLE001
        return [], False, str(exc)


def _intent_aware_semantic(
    query: str,
    top_k: int,
    document_filter: dict[str, Any] | None,
    max_distance: float,
    intent: dict[str, Any] | None,
    intent_restriction: list[int] | None,
) -> dict[str, Any]:
    """Semantic retrieval with document-intent restriction + reranking.

    Deterministic policy:

    * No intent restriction -> global search, reranked with the intent
      weight renormalized away (pre-Phase-6 behavior plus annotations).
    * Restriction yields useful chunks -> those, reranked, kept restricted.
    * Restriction yields chunks but NONE within the relevance threshold ->
      the restriction is KEPT and an explicit note is returned; global
      results are NOT substituted, because that substitution is exactly
      how a generic phrase in an unrelated document ("photo ID cards")
      used to outrank the requested document.
    * Restriction is vacuous (no chunks at all for those documents) ->
      relaxed to a global search, with the relaxation recorded.

    Returns::

        {"results": useful reranked chunks,
         "available": bool, "note": str | None, "raw_count": int,
         "restriction_applied": bool, "relaxed_to_global": bool,
         "restriction_note": str | None}
    """

    if not intent_restriction:
        results, available, note = _semantic_lookup(
            query, top_k, document_filter, max_distance
        )
        if not available:
            return {
                "results": [], "available": False, "note": note,
                "raw_count": 0, "restriction_applied": False,
                "relaxed_to_global": False, "restriction_note": None,
            }
        useful = [item for item in results if item["useful"]]
        return {
            "results": _rerank_chunks(query, useful, intent),
            "available": True, "note": note, "raw_count": len(results),
            "restriction_applied": False,
            "relaxed_to_global": False, "restriction_note": None,
        }

    # ---- Restricted search first ------------------------------------------
    results, available, note = _semantic_lookup(
        query, top_k, document_filter, max_distance, intent_restriction
    )
    if not available:
        return {
            "results": [], "available": False, "note": note,
            "raw_count": 0, "restriction_applied": False,
            "relaxed_to_global": False, "restriction_note": None,
        }

    useful = [item for item in results if item["useful"]]
    if useful:
        return {
            "results": _rerank_chunks(query, useful, intent),
            "available": True, "note": note, "raw_count": len(results),
            "restriction_applied": True,
            "relaxed_to_global": False, "restriction_note": None,
        }

    if results:
        # Chunks exist in the intent-matching documents but none is within
        # the baseline threshold. For a STRONG intent, retry the SAME
        # restricted search with the intent-qualified margin: the where
        # clause is unchanged, so no unrelated document can leak in; only
        # the distance bound widens for the user's own requested document.
        strong_ids = {
            int(cand["document_id"])
            for cand in (intent or {}).get("strong", [])
        }
        if strong_ids and set(intent_restriction) <= strong_ids:
            margin_results, margin_available, _margin_note = _semantic_lookup(
                query, top_k, document_filter, INTENT_QUALIFIED_MAX_DISTANCE,
                intent_restriction,
            )
            margin_useful = [c for c in margin_results if c["useful"]]
            if margin_available and margin_useful:
                return {
                    "results": _rerank_chunks(query, margin_useful, intent),
                    "available": True, "note": note,
                    "raw_count": len(margin_results),
                    "restriction_applied": True,
                    "relaxed_to_global": False,
                    "restriction_note": (
                        f"intent-qualified threshold {INTENT_QUALIFIED_MAX_DISTANCE} "
                        f"applied (baseline {SEMANTIC_USEFUL_MAX_DISTANCE} had no "
                        "useful chunk); search stayed restricted to the "
                        "intent-matching document(s)"
                    ),
                }

        # No (or weak-only) intent, or the margin found nothing either: keep
        # the restriction and report honestly rather than letting unrelated
        # documents dominate on generic words.
        restriction_note = (
            f"restricted to {len(intent_restriction)} intent-matching "
            f"document(s); {len(results)} chunk(s) found, none within the "
            "relevance threshold"
        )
        return {
            "results": [], "available": True, "note": note,
            "raw_count": len(results), "restriction_applied": True,
            "relaxed_to_global": False, "restriction_note": restriction_note,
        }

    # ---- Vacuous restriction -> recorded relaxation to global -------------
    relax_reason = (
        "intent restriction matched no stored chunks; "
        "relaxed to global retrieval"
    )
    global_results, global_available, global_note = _semantic_lookup(
        query, top_k, document_filter, max_distance
    )
    if not global_available:
        return {
            "results": [], "available": False, "note": global_note,
            "raw_count": 0, "restriction_applied": True,
            "relaxed_to_global": True, "restriction_note": relax_reason,
        }
    global_useful = [item for item in global_results if item["useful"]]
    return {
        "results": _rerank_chunks(query, global_useful, intent),
        "available": True, "note": global_note,
        "raw_count": len(global_results), "restriction_applied": True,
        "relaxed_to_global": True, "restriction_note": relax_reason,
    }


# ============================================================
# Main entry point
# ============================================================


def route_query(
    query: str,
    top_k: int = 3,
    allow_fallback: bool = True,
    connection: sqlite3.Connection | None = None,
    max_distance: float | None = None,
    intent: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Route a natural-language query and retrieve grounded results.

    Returns a provenance-complete dict:

        {
          "route": "exact" | "semantic",
          "classification": "exact" | "semantic" | "unknown",
          "query": ...,
          "field_hint": ..., "field_variants": [...], "value_hint": ...,
          "document_filter": {...} | None,
          "fallback": {"occurred": bool, "reason": str | None},
          "insufficient": bool, "insufficient_reason": str | None,
          "results": [...],
          "timings_ms": {"classification": ..., "retrieval": ..., "total": ...}
        }

    No value is ever invented: empty matches produce an explicit
    insufficient state instead of fabricated content.

    ``max_distance`` optionally overrides the semantic usefulness
    threshold for this call only (defaults to the module constant; the
    classification logic is identical either way).

    ``intent`` (Phase 6) allows a pre-computed document-intent payload to
    be injected for testing; when None, intent is detected internally.
    """

    effective_max_distance = (
        SEMANTIC_USEFUL_MAX_DISTANCE
        if max_distance is None
        else float(max_distance)
    )

    query = str(query).strip()
    if not query:
        raise ValueError("query cannot be empty.")
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0.")

    own_connection = connection is None
    started = time.perf_counter()

    # Phase 8: Devanagari queries are transliterated to Latin BEFORE any
    # deterministic classification/matching, so Hindi queries hit the
    # same cues/synonym machinery as Hinglish ones. Detection of the
    # user's language itself happens in answer_engine (on the ORIGINAL
    # text); document values are never transliterated.
    query_for_matching = transliterate_for_matching(query)

    try:
        if own_connection:
            connection = get_db_connection()

        known_fields = [
            row[0]
            for row in connection.execute(
                "SELECT DISTINCT field_name FROM extracted_metadata"
            )
        ]

        t_after_known_fields = time.perf_counter()
        signals = classify_query(query_for_matching, known_fields)
        t_after_classify = time.perf_counter()

        classification = signals["classification"]
        document_filter = _extract_document_filter(query_for_matching, connection)

        # ---- Phase 6: document-intent detection ------------------------------
        # Restricts semantic retrieval to intent-matching documents BEFORE
        # the vector search (strong/weak policy below); pure-exact filename
        # filters still take precedence in exact lookups.
        if intent is None:
            intent = detect_document_intent(query_for_matching, connection)
        intent_restriction = (
            intent.get("restrict_to") if isinstance(intent, dict) else None
        )

        field_variants = _resolve_field_variants(
            signals["field_hint"], known_fields
        )
        value_hint = signals["value_hint"]

        results: list[dict[str, Any]] = []
        fallback = {"occurred": False, "reason": None}
        insufficient = False
        insufficient_reason: str | None = None
        restriction_applied = False
        relaxed_to_global = False
        restriction_note: str | None = None

        exact_signals_present = bool(field_variants or value_hint)
        use_exact_first = (
            classification == CLASS_EXACT
            or (classification == CLASS_UNKNOWN and exact_signals_present)
        )

        if use_exact_first:
            # ---- Exact first -------------------------------------------------
            # An EXACT-classified query attempts the deterministic lookup even
            # when its field hint matches no stored field: the miss must be
            # observable (and, when enabled, fall back with a recorded reason)
            # rather than silently rerouted.
            results = _exact_lookup(connection, field_variants, value_hint, document_filter)

            if results:
                route = ROUTE_EXACT
            elif allow_fallback:
                # ---- Recorded semantic fallback -----------------------------
                semantic = _intent_aware_semantic(
                    query, top_k, document_filter, effective_max_distance,
                    intent, intent_restriction,
                )
                restriction_applied = semantic["restriction_applied"]
                relaxed_to_global = semantic["relaxed_to_global"]
                restriction_note = semantic["restriction_note"]
                fallback = {
                    "occurred": True,
                    "reason": "no exact metadata match"
                    + (f"; {semantic['note']}" if semantic["note"] else ""),
                }
                if not semantic["available"]:
                    route = ROUTE_EXACT
                    insufficient = True
                    insufficient_reason = (
                        "no exact metadata match and semantic store unavailable: "
                        f"{semantic['note']}"
                    )
                else:
                    results = semantic["results"]
                    route = ROUTE_SEMANTIC if results else ROUTE_EXACT
                    if not results:
                        insufficient = True
                        extra_notes = "; ".join(
                            n
                            for n in (semantic["note"], restriction_note)
                            if n
                        )
                        insufficient_reason = (
                            "no exact metadata match; semantic fallback returned "
                            f"{semantic['raw_count']} chunk(s), none useful"
                            + (f"; {extra_notes}" if extra_notes else "")
                        )
            else:
                route = ROUTE_EXACT
                insufficient = True
                insufficient_reason = "no exact metadata match (fallback disabled)"

        else:
            # ---- Semantic route ---------------------------------------------
            semantic = _intent_aware_semantic(
                query, top_k, document_filter, effective_max_distance,
                intent, intent_restriction,
            )
            restriction_applied = semantic["restriction_applied"]
            relaxed_to_global = semantic["relaxed_to_global"]
            restriction_note = semantic["restriction_note"]
            route = ROUTE_SEMANTIC
            if not semantic["available"]:
                insufficient = True
                insufficient_reason = f"semantic store unavailable: {semantic['note']}"
            else:
                results = semantic["results"]
                if not results:
                    insufficient = True
                    extra_notes = "; ".join(
                        n for n in (semantic["note"], restriction_note) if n
                    )
                    insufficient_reason = (
                        "semantic retrieval returned "
                        f"{semantic['raw_count']} chunk(s), none useful"
                        + (f"; {extra_notes}" if extra_notes else "")
                    )

        t_end = time.perf_counter()

        # ---- Phase 6 provenance: document-intent block -----------------------
        if isinstance(intent, dict):
            intent_block: dict[str, Any] = {
                "detected": bool(intent.get("intent_detected")),
                "phrases": list(intent.get("phrases") or []),
                "candidates": list(intent.get("candidates") or []),
                "strong_document_ids": [
                    cand["document_id"] for cand in intent.get("strong") or []
                ],
                "weak_document_ids": [
                    cand["document_id"] for cand in intent.get("weak") or []
                ],
            }
        else:
            intent_block = {
                "detected": False,
                "phrases": [],
                "candidates": [],
                "strong_document_ids": [],
                "weak_document_ids": [],
            }
        intent_block.update(
            {
                "restrict_to": intent_restriction,
                "restriction_applied": restriction_applied,
                "relaxed_to_global": relaxed_to_global,
                "relax_reason": restriction_note if relaxed_to_global else None,
                "restriction_note": restriction_note,
            }
        )

        return {
            "route": route,
            "classification": classification,
            "query": query,
            "field_hint": signals["field_hint"],
            "field_variants": field_variants,
            "value_hint": value_hint,
            "document_filter": document_filter,
            "document_intent": intent_block,
            "fallback": fallback,
            "insufficient": insufficient,
            "insufficient_reason": insufficient_reason,
            "results": results,
            "timings_ms": {
                "classification": round((t_after_classify - t_after_known_fields) * 1000, 3),
                "retrieval": round((t_end - t_after_classify) * 1000, 3),
                "total": round((t_end - started) * 1000, 3),
            },
        }

    finally:
        if own_connection and connection is not None:
            connection.close()
