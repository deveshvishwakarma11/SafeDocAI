"""
Phase 5: Local Grounded Answer Generation.
==========================================

Consumes a :func:`query_router.route_query` result and produces a concise,
grounded natural-language answer using ONLY local resources:

* EXACT route  -> deterministic answer built from SQLite rows. The LLM is
  deliberately NOT called: exact identifiers must not be paraphrased.
* SEMANTIC route -> the retrieved Chroma chunks are passed to the local
  Ollama qwen2.5:3b model with a strict "answer only from the supplied
  context" instruction, and the response is then validated
  deterministically against that same context before it is returned.

Grounding validation (the core of this module):

* every factual value in the answer (numbers, dates, identifiers, names)
  must be traceable to the supplied retrieval context;
* fabricated numbers / ids / names are rejected;
* the model's self-reported ``confidence`` is recorded but NEVER trusted
  as evidence of grounding;
* if grounding cannot be established, an explicit insufficient-context
  result is returned instead of a guess.

Provenance: every result carries source documents (with document_id,
file_name and, for semantic answers, chunk ids and retrieval distances),
the retrieval route (including any router fallback) and the grounded flag.

100% local: Ollama on localhost + SQLite + ChromaDB. No cloud/API/network
calls, and no document content leaves the machine.
"""

from __future__ import annotations

import re
import time
from typing import Any

try:  # src/ on sys.path (pipeline style)
    from language import (
        LANG_ENGLISH,
        LANG_HINGLISH,
        LANG_HINDI,
        detect_language,
    )
    from llm_engine import extract_json, generate_with_meta
    from query_relevance import check_query_relevance
    from query_router import route_query
except ImportError:  # project root on sys.path (test/tooling style)
    from src.language import (
        LANG_ENGLISH,
        LANG_HINGLISH,
        LANG_HINDI,
        detect_language,
    )
    from src.llm_engine import extract_json, generate_with_meta
    from src.query_relevance import check_query_relevance
    from src.query_router import route_query


__all__ = [
    "INSUFFICIENT_CONTEXT_MESSAGE",
    "ANSWER_NUM_PREDICT",
    "answer_query",
    "build_answer_from_exact",
    "build_semantic_prompt",
    "parse_semantic_response",
    "validate_grounding",
]


# ============================================================
# Configuration
# ============================================================

#: Safe user-facing message when retrieval found nothing usable.
INSUFFICIENT_CONTEXT_MESSAGE = (
    "I could not find enough information in your stored documents "
    "to answer this."
)

#: Output cap for the local model. The schema is one short prose answer
#: plus a tiny sources array; 256 tokens is ample headroom on CPU.
ANSWER_NUM_PREDICT = 256

# ------------------------------------------------------------
# Phase 8: language-aware fixed strings (deterministic, no LLM).
# Document VALUES are never translated -- only the connective wording
# around them changes with the user's detected language.
# ------------------------------------------------------------

#: Insufficient-context message per detected language.
INSUFFICIENT_CONTEXT_BY_LANGUAGE: dict[str, str] = {
    LANG_ENGLISH: INSUFFICIENT_CONTEXT_MESSAGE,
    LANG_HINGLISH: (
        "Aapke stored documents mein is sawal ka jawab nahi mila."
    ),
    LANG_HINDI: (
        "आपके stored documents में इस सवाल का जवाब नहीं मिला।"
    ),
}

#: Single-value exact answer templates: field/value/source stay verbatim.
_EXACT_SINGLE_SUFFIX = {
    LANG_ENGLISH: " (from {file_name})",
    LANG_HINGLISH: " ({file_name} se)",
    LANG_HINDI: " ({file_name} से)",
}

#: Multi-value exact answer headers.
_EXACT_MULTI_HEADER = {
    LANG_ENGLISH: "Found matching values:",
    LANG_HINGLISH: "Yeh matching values mili:",
    LANG_HINDI: "ये matching values मिलीं:",
}

# ------------------------------------------------------------
# Phase 9: answer-focused phrasing. The user sees the FACT, not the
# raw "field: value" row. Values/identifiers stay verbatim.
# ------------------------------------------------------------

#: Single-value sentence templates. {subject} = the cleaned requested-
#: fact phrase ("roll number"), {value} = the grounded value verbatim.
_EXACT_SINGLE_SENTENCE = {
    LANG_ENGLISH: "Your {subject} is {value}.",
    LANG_HINGLISH: "Aapka {subject} {value} hai.",
    LANG_HINDI: "आपका {subject} {value} है।",
}

#: Multi-value templates. {count} = number of distinct values,
#: {subject} = requested-fact phrase.
_EXACT_MULTI_INTRO = {
    LANG_ENGLISH: "I found {count} matching values for {subject}:",
    LANG_HINGLISH: "Mujhe {count} {subject} values mili:",
    LANG_HINDI: "मुझे {count} {subject} values मिलीं:",
}

#: Edge fillers stripped from the requested-fact phrase (question words
#: and Hinglish postpositions; deterministic, language-level).
_SUBJECT_EDGE_FILLERS = frozenset(
    {
        "my", "the", "a", "an", "my?", "kya", "hai", "hain", "tha", "thi",
        "batao", "bata", "bataiye", "ka", "ki", "ke", "mein", "me", "par",
        "in", "on", "of", "please", "what", "which",
    }
)


def _clean_subject(hint: str | None, fallback: str | None) -> str:
    """Deterministic requested-fact phrase for answer-focused phrasing."""

    tokens = re.findall(r"[a-z0-9 .&'/\-]+", str(hint or "").lower())
    words = (tokens[0] if tokens else "").split()
    while words and words[0] in _SUBJECT_EDGE_FILLERS:
        words.pop(0)
    while words and words[-1] in _SUBJECT_EDGE_FILLERS:
        words.pop()
    subject = " ".join(words).strip()
    return subject or str(fallback or "value")

#: Hard cap on the context length handed to the model (characters). The
#: router already returns bounded chunks; this is a defensive second cap.
MAX_CONTEXT_CHARS = 3000

#: Minimum length of a standalone factual token worth checking. Short
#: fragments ("RS", "ID") are not identifiers.
_MIN_FACTUAL_TOKEN_LEN = 3


# ============================================================
# Factual-span extraction (deterministic grounding primitives)
# ============================================================

#: Numbers with separators/decimals (6.61, 1,269, 12,345.67). The lookbehind
#: allows a digit right after a period ("Rs.60") but never inside a word.
_NUMBER_RE = re.compile(
    r"(?<![\w])(?:\d[\d,]*)(?:\.\d+)?(?![\w.])"
)
#: Mixed alphanumeric identifiers (M103B71, ABCDE1234F, XZ99999).
_ALNUM_RE = re.compile(r"[A-Za-z0-9]{4,}")


def _normalize_sentence_periods(text: str) -> str:
    """Turn sentence-final periods into spaces so numbers/identifiers at the
    end of a sentence are still extractable ("was 9.99." -> "was 9.99 ").
    Internal periods (decimals, "Rs.60") are untouched."""

    return re.sub(r"\.(?=\s|$)", " ", text)
#: Dates in the shapes the corpus actually stores.
_DATE_RE = re.compile(
    r"\b\d{1,4}[-/]\d{1,2}[-/]\d{1,4}\b|\b\d{1,2}[- ][A-Za-z]{3,9}[- ]\d{2,4}\b"
)
#: Long alphanumeric identifiers (roll numbers, PNRs, transaction ids).
_IDENTIFIER_RE = re.compile(
    r"(?<![\w])\d[\dA-Za-z]{7,}(?![\w])"
)


def _context_facts(context_text: str) -> set[str]:
    """Extract the factual tokens present in the supplied context."""

    facts: set[str] = set()
    if not context_text:
        return facts

    context_text = _normalize_sentence_periods(context_text)

    for match in _DATE_RE.finditer(context_text):
        facts.add(match.group(0))
    for match in _NUMBER_RE.finditer(context_text):
        facts.add(match.group(0))
    for match in _IDENTIFIER_RE.finditer(context_text):
        facts.add(match.group(0))

    # Meaningful words (>= 3 chars) so proper names can be checked too.
    for token in re.findall(r"[A-Za-z]{3,}", context_text):
        facts.add(token)

    # Mixed alphanumeric identifiers (letters AND digits, e.g. M103B71).
    for token in _ALNUM_RE.findall(context_text):
        if any(ch.isdigit() for ch in token) and any(ch.isalpha() for ch in token):
            facts.add(token)

    return facts


def _answer_facts(answer: str) -> list[str]:
    """Extract factual tokens from a candidate answer, longest first."""

    found: set[str] = set()

    answer = _normalize_sentence_periods(answer)

    for pattern in (_DATE_RE, _NUMBER_RE, _IDENTIFIER_RE):
        for match in pattern.finditer(answer):
            found.add(match.group(0))
    for token in re.findall(r"[A-Za-z]{3,}", answer):
        found.add(token)

    # Mixed alphanumeric identifiers (letters AND digits, e.g. M103B71).
    for token in _ALNUM_RE.findall(answer):
        if any(ch.isdigit() for ch in token) and any(ch.isalpha() for ch in token):
            found.add(token)

    # Longest first so overlapping spans resolve to the most specific.
    return sorted(found, key=len, reverse=True)


def validate_grounding(
    answer: str,
    context_text: str,
) -> dict[str, Any]:
    """Deterministically verify the answer against the supplied context.

    A numeric/date/identifier token in the answer that does not exist in
    the context is a fabricated fact. Capitalized name-like words are
    checked conservatively (only as a reported warning signal; prose
    function words are ignored). The result never modifies the answer.
    """

    context_facts = _context_facts(context_text)
    normalized = {fact.lower() for fact in context_facts}
    # Comma-grouped and plain forms are equivalent (1,269 vs 1269).
    for fact in list(context_facts):
        if "," in fact:
            normalized.add(fact.replace(",", "").lower())

    unsupported_numbers: list[str] = []
    unsupported_names: list[str] = []

    for fact in _answer_facts(answer):
        if fact.lower() in normalized:
            continue
        stripped = fact.replace(",", "")
        if stripped.lower() in normalized:
            continue
        is_alnum_id = (
            any(ch.isdigit() for ch in fact)
            and any(ch.isalpha() for ch in fact)
            and _ALNUM_RE.fullmatch(fact) is not None
        )
        if (
            _DATE_RE.fullmatch(fact)
            or _NUMBER_RE.fullmatch(fact)
            or _IDENTIFIER_RE.fullmatch(fact)
            or is_alnum_id
        ):
            unsupported_numbers.append(fact)
        elif len(fact) >= _MIN_FACTUAL_TOKEN_LEN and fact[0].isupper():
            unsupported_names.append(fact)

    grounded = not unsupported_numbers
    return {
        "grounded": grounded,
        "unsupported_numbers": unsupported_numbers,
        "unsupported_names": unsupported_names,
    }


# ============================================================
# EXACT answers (deterministic, no LLM)
# ============================================================


def build_answer_from_exact(result: dict[str, Any]) -> dict[str, Any]:
    """Build the deterministic answer payload for an exact route result.

    Multiple matching values are ALL preserved with their source documents
    (never collapsed, never arbitrarily chosen). No LLM is involved.

    Phase 8: the connective wording follows the user's DETECTED language
    (English / Hinglish / Hindi); document VALUES stay verbatim.
    """

    rows = result.get("results", [])
    language = detect_language(str(result.get("query") or ""))
    if language not in _EXACT_MULTI_HEADER:  # unknown/missing query -> English
        language = LANG_ENGLISH

    sources: list[dict[str, Any]] = []
    seen_display: set[tuple[str, str]] = set()
    bullets: list[str] = []

    for row in rows:
        document_id = row.get("document_id")
        file_name = row.get("file_name")
        field_name = row.get("field_name")
        field_value = row.get("field_value")

        source: dict[str, Any] = {
            "document_id": document_id,
            "file_name": file_name,
            "file_path": row.get("file_path"),
            "field_name": field_name,
            "field_value": field_value,
        }
        if source not in sources:
            sources.append(source)

        # Display de-duplication: the SAME value from the SAME document
        # (e.g. an OCR field stored twice) appears once. Different values
        # and/or different documents are all kept (never collapsed).
        key = (str(field_value).strip().lower(), str(file_name))
        if key in seen_display:
            continue
        seen_display.add(key)
        bullets.append(f"• {field_value} — {file_name}")

    subject = _clean_subject(
        result.get("field_hint"),
        rows[0].get("field_name") if rows else None,
    )

    if len(bullets) == 1:
        value = str(rows[0].get("field_value"))
        answer = _EXACT_SINGLE_SENTENCE[language].format(
            subject=subject, value=value
        )
    elif bullets:
        answer = _EXACT_MULTI_INTRO[language].format(
            count=len(bullets), subject=subject
        ) + "\n" + "\n".join(bullets)
    else:
        answer = INSUFFICIENT_CONTEXT_BY_LANGUAGE.get(
            language, INSUFFICIENT_CONTEXT_MESSAGE
        )

    return {
        "query": result.get("query"),
        "answer": answer,
        "language": language,
        "sources": sources,
        "source_documents": [
            {"document_id": s["document_id"], "file_name": s["file_name"]}
            for s in sources
        ],
        "retrieval_route": result.get("route", "exact"),
        "classification": result.get("classification"),
        "fallback": result.get("fallback", {"occurred": False, "reason": None}),
        "grounded": bool(rows),
        "llm_called": False,
        "insufficient": not bool(rows),
    }


# ============================================================
# SEMANTIC prompt / parsing / generation
# ============================================================


# ------------------------------------------------------------
# Phase 9: answer-length styles. A fact lookup ("What is my transaction
# ID?") must produce ONE short sentence; a summary/explanation request
# ("What does my railway ticket contain?") may produce a concise
# paragraph. Deterministic cue matching decides the style -- never the
# LLM.
# ------------------------------------------------------------

#: Cues that mark an explicit summary/explanation request (EN + Hinglish).
_SUMMARY_CUES: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bsummar(y|ise|ize)\b",
        r"\bexplain\b",
        r"\bwhat does\b[^?]*\b(says?|contain(s|ed)?|state(s|d)?)\b",
        r"\btell me (about|what)\b",
        r"\bimportant (details|information|points)\b",
        r"\bke baare\b",
        r"\bbaare (mein|me)\b",
        r"\b(kya|kaisa|kaisi) likha\b",
        r"\bmujhe\b[^?]*\bbata\w*\b",
        r"\bkya details\b",
        r"\bwhat is (this|the) document\b",
        r"\bwhat (information|details) (is|are)\b",
        r"\bdocument (about|contains)\b",
        r"\bshow me the (summary|overview|details)\b",
        r"\b(brief|short) (note|description|overview)\b",
    )
)

#: Prompt length rules per style.
_LENGTH_RULE_FACT = (
    "- The user asked for a specific fact. Answer with ONE short sentence "
    "containing ONLY the requested value(s). Do not describe the document, "
    "do not add background, and do not list unrelated details."
)
_LENGTH_RULE_SUMMARY = (
    "- The user asked for an overview/summary. Give a concise paragraph "
    "(max 4 sentences)."
)


def is_summary_request(question: str) -> bool:
    """Deterministically decide fact vs summary/explanation intent."""

    text = str(question or "")
    return any(pattern.search(text) for pattern in _SUMMARY_CUES)


def build_semantic_prompt(
    question: str,
    chunks: list[dict[str, Any]],
    language: str | None = None,
    style: str | None = None,
) -> str:
    """Build the strict grounded-QA prompt from retrieved chunks.

    The prompt states the contract explicitly: answer ONLY from the
    supplied context, never invent facts, preserve identifiers exactly,
    say when the context is insufficient, and cite sources in the
    structured output.

    Phase 8: when ``language`` is provided, one deterministic extra rule
    pins the answer language to the user's detected language (Hinglish
    queries get natural Hinglish, Hindi gets Hindi). The document VALUES
    are explicitly excluded from translation. ``language=None`` keeps the
    original English-only prompt byte-identical (backward compatible).

    Phase 9: ``style`` selects the answer-length rule ("fact" = one
    short sentence with only the requested value; "summary" = concise
    paragraph). ``None`` auto-detects from the question deterministically.
    """

    numbered_chunks = "\n\n".join(
        f"[Context {index + 1} | document: {chunk.get('file_name', 'unknown')}"
        + (
            f" | document_id: {chunk['document_id']}"
            if chunk.get("document_id") is not None
            else ""
        )
        + f"]\n{chunk.get('text', '')}"
        for index, chunk in enumerate(chunks)
    )

    if style not in {"fact", "summary"}:
        style = "summary" if is_summary_request(question) else "fact"
    length_rule = _LENGTH_RULE_SUMMARY if style == "summary" else _LENGTH_RULE_FACT

    language_rule = ""
    if language == LANG_HINGLISH:
        language_rule = (
            "- The user asked in Hinglish (Roman Hindi mixed with English). "
            "Write the answer in natural conversational Hinglish (e.g. "
            "'Aapka roll number 12345 hai.'), not in formal English.\n"
        )
    elif language == LANG_HINDI:
        language_rule = (
            "- The user asked in Hindi. Write the answer in Hindi "
            "(Devanagari script), keeping English technical terms/identifiers "
            "as they appear in the context.\n"
        )
    elif language == LANG_ENGLISH:
        language_rule = "- Write the answer in English.\n"

    return f"""You are a document question-answering assistant.

Rules:
- Answer ONLY from the supplied context below. Never use outside knowledge.
- Never invent facts, numbers, names or identifiers.
- Copy important identifiers, numbers and dates EXACTLY as they appear.
- If the context does not contain the answer, say so explicitly.
{length_rule}
{language_rule}- Do NOT translate document names, identifiers, numbers or values; keep them exactly as in the context.
- Cite the source document(s) you used in the sources array.

Respond ONLY with JSON:
{{"answer": "...", "sources": [{{"document_id": <id>, "file_name": "<name>"}}], "confidence": "high|medium|low"}}

Question:
{question}

Context:
{numbered_chunks}""".strip()


def parse_semantic_response(
    response_text: str | None,
    chunks: list[dict[str, Any]],
    context_text: str,
) -> dict[str, Any] | None:
    """Defensively parse and ground-validate a semantic LLM response.

    Returns a validated payload dict, or None when the response is
    unusable (empty, invalid/truncated JSON, missing answer, unknown
    source references, or ungrounded content). The model's confidence is
    recorded but never trusted.

    Unknown source references are dropped from the payload (recorded in
    ``unknown_sources``), not silently accepted: a citation must point to
    a chunk that was actually supplied.
    """

    if not response_text or not response_text.strip():
        return None

    parsed = extract_json(response_text)
    if parsed is None:
        return None

    answer = str(parsed.get("answer", "")).strip()
    if not answer:
        return None

    grounding = validate_grounding(answer, context_text)
    if not grounding["grounded"]:
        return None

    valid_keys = {
        (chunk.get("document_id"), chunk.get("file_name")) for chunk in chunks
    }

    sources: list[dict[str, Any]] = []
    unknown_sources: list[dict[str, Any]] = []
    raw_sources = parsed.get("sources", [])
    if isinstance(raw_sources, list):
        for entry in raw_sources:
            if not isinstance(entry, dict):
                continue
            document_id = entry.get("document_id")
            file_name = entry.get("file_name")
            key = (document_id, file_name)
            if key in valid_keys:
                if key not in {(s["document_id"], s["file_name"]) for s in sources}:
                    sources.append({"document_id": document_id, "file_name": file_name})
            elif key not in {(s["document_id"], s["file_name"]) for s in unknown_sources}:
                unknown_sources.append({"document_id": document_id, "file_name": file_name})

    confidence = str(parsed.get("confidence", "")).strip().lower()
    if confidence not in {"high", "medium", "low"}:
        confidence = ""

    return {
        "answer": answer,
        "sources": sources,
        "unknown_sources": unknown_sources,
        "model_confidence": confidence,
        "grounding": grounding,
    }


def _build_context_text(chunks: list[dict[str, Any]]) -> str:
    """The exact context string the model saw (same concatenation as prompt)."""

    return "\n\n".join(str(chunk.get("text", "")) for chunk in chunks)


def _generate_semantic_answer(
    question: str,
    chunks: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Call the local model once and return its validated payload or None.

    Transport failure, invalid JSON, truncation (done_reason == "length")
    and grounding failure all return None; callers treat that as
    insufficient rather than storing a guess.

    Phase 8: the user's language is detected deterministically (never by
    the LLM) and pinned in the prompt so the answer arrives in the
    language the question was asked in.
    """

    context_text = _build_context_text(chunks)[:MAX_CONTEXT_CHARS]

    prompt = build_semantic_prompt(
        question,
        chunks,
        language=detect_language(question),
        style="summary" if is_summary_request(question) else "fact",
    )

    meta = generate_with_meta(
        prompt,
        num_predict=ANSWER_NUM_PREDICT,
    )

    if meta is None:
        return None

    if meta.get("done_reason") == "length":
        # Truncated mid-object: the JSON contract cannot be trusted.
        return None

    return parse_semantic_response(meta.get("text"), chunks, context_text)


# ============================================================
# Public entry point
# ============================================================


def _public_payload(
    answer: str,
    sources: list[dict[str, Any]],
    route_result: dict[str, Any],
    grounded: bool,
    llm_called: bool,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the final structured result with full provenance."""

    payload: dict[str, Any] = {
        "query": route_result.get("query"),
        "answer": answer,
        "sources": sources,
        "source_documents": [
            {"document_id": s.get("document_id"), "file_name": s.get("file_name")}
            for s in sources
        ],
        "retrieval_route": route_result.get("route"),
        "classification": route_result.get("classification"),
        "fallback": route_result.get("fallback", {"occurred": False, "reason": None}),
        "grounded": grounded,
        "llm_called": llm_called,
        "insufficient": not grounded and not sources,
    }
    if extra:
        payload.update(extra)
    return payload


def answer_query(
    query: str,
    top_k: int = 3,
    max_distance: float | None = None,
    llm_fn=None,
) -> dict[str, Any]:
    """Answer a natural-language question with full grounding + provenance.

    EXACT route  -> deterministic answer from SQLite; no LLM call.
    SEMANTIC route -> one local Ollama call over the retrieved chunks,
    then deterministic grounding validation against those same chunks.

    Args:
        query: The user's question.
        top_k: Passed through to the router for semantic retrieval.
        max_distance: Optional semantic usefulness override (defaults to
            the router's own threshold). Passing a larger value admits
            lower-ranked chunks for LLM context; classification logic is
            unchanged.
        llm_fn: Testing hook. Defaults to :func:`_generate_semantic_answer`.
            Any replacement must keep the same contract: return a
            validated payload dict or None (never raise, never fabricate).

    Never invents content: anything unusable becomes an explicit
    insufficient result. Fallback information from the router is
    preserved in the final result.
    """

    started = time.perf_counter()

    # ---- Phase 9: relevance gate ---------------------------------------
    # Deterministic pre-check BEFORE any retrieval/LLM work. Unrelated
    # queries (weather, general knowledge, math, jokes, coding) short-
    # circuit here with zero SQLite/Chroma/Ollama cost.
    gate = check_query_relevance(query)
    if not gate["related"]:
        language = detect_language(query)
        gate_ms = round((time.perf_counter() - started) * 1000, 3)
        return {
            "query": query,
            "answer": INSUFFICIENT_CONTEXT_BY_LANGUAGE.get(
                language, INSUFFICIENT_CONTEXT_MESSAGE
            ),
            "sources": [],
            "source_documents": [],
            "retrieval_route": "none",
            "classification": "rejected",
            "fallback": {"occurred": False, "reason": None},
            "grounded": False,
            "llm_called": False,
            "insufficient": True,
            "retrieval_performed": False,
            "reason": gate["reason"],
            "relevance": gate,
            "language": language,
            "timings_ms": {
                "router": 0.0,
                "retrieval": None,
                "llm_generation": None,
                "total": gate_ms,
            },
        }

    route_result = route_query(query, top_k=top_k, max_distance=max_distance)

    router_ms = round((time.perf_counter() - started) * 1000, 3)

    if route_result["route"] == "exact":
        payload = build_answer_from_exact(route_result)
        payload["retrieval_performed"] = True
        payload["reason"] = "document_related"
        payload["timings_ms"] = {
            "router": router_ms,
            "retrieval": route_result.get("timings_ms", {}).get("retrieval"),
            "llm_generation": None,
            "total": round((time.perf_counter() - started) * 1000, 3),
        }
        return payload

    chunks = route_result.get("results", [])
    language = detect_language(query)

    if not chunks:
        return _public_payload(
            INSUFFICIENT_CONTEXT_BY_LANGUAGE.get(
                language, INSUFFICIENT_CONTEXT_MESSAGE
            ),
            [],
            route_result,
            grounded=False,
            llm_called=False,
            extra={
                "language": language,
                "retrieval_performed": True,
                "reason": "document_related",
                "insufficient_reason": route_result.get("insufficient_reason"),
                "timings_ms": {
                    "router": router_ms,
                    "retrieval": route_result.get("timings_ms", {}).get("retrieval"),
                    "llm_generation": None,
                    "total": round((time.perf_counter() - started) * 1000, 3),
                },
            },
        )

    llm_started = time.perf_counter()

    if llm_fn is not None:
        validated = llm_fn(query, chunks)
    else:
        validated = _generate_semantic_answer(query, chunks)

    llm_ms = round((time.perf_counter() - llm_started) * 1000, 3)

    if validated is None:
        return _public_payload(
            INSUFFICIENT_CONTEXT_BY_LANGUAGE.get(
                language, INSUFFICIENT_CONTEXT_MESSAGE
            ),
            [],
            route_result,
            grounded=False,
            llm_called=True,
            extra={
                "language": language,
                "insufficient_reason": (
                    "the local model's answer could not be validated against "
                    "the retrieved context"
                ),
                "model_confidence": None,
                "timings_ms": {
                    "router": router_ms,
                    "retrieval": route_result.get("timings_ms", {}).get("retrieval"),
                    "llm_generation": llm_ms,
                    "total": round((time.perf_counter() - started) * 1000, 3),
                },
            },
        )

    # Provenance guarantee: a grounded answer must always carry sources.
    # If the model validated but cited nothing, derive sources from the
    # retrieved chunks — the exact context the answer was grounded against.
    final_sources = validated["sources"] or [
        {
            "document_id": chunk.get("document_id"),
            "file_name": chunk.get("file_name"),
            "chunk_id": chunk.get("chunk_id"),
            "distance": chunk.get("distance"),
        }
        for chunk in chunks
    ]

    chunk_sources = [
        {
            "document_id": chunk.get("document_id"),
            "file_name": chunk.get("file_name"),
            "chunk_id": chunk.get("chunk_id"),
            "distance": chunk.get("distance"),
        }
        for chunk in chunks
    ]

    return _public_payload(
        validated["answer"],
        final_sources,
        route_result,
        grounded=True,
        llm_called=True,
        extra={
            "language": language,
            "retrieval_performed": True,
            "reason": "document_related",
            "model_confidence": validated.get("model_confidence"),
            "grounding": validated.get("grounding"),
            "unknown_sources": validated.get("unknown_sources", []),
            "retrieved_chunks": chunk_sources,
            "timings_ms": {
                "router": router_ms,
                "retrieval": route_result.get("timings_ms", {}).get("retrieval"),
                "llm_generation": llm_ms,
                "total": round((time.perf_counter() - started) * 1000, 3),
            },
        },
    )
