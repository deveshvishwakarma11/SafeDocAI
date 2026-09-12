"""
SafeDocAI - Local LLM Engine
============================

Non-blocking client for Ollama's local HTTP API
(http://localhost:11434/api/generate).

Uses dynamic document understanding - the LLM is NOT restricted
to a fixed document type enum. It discovers document structure
dynamically based on evidence.

Why HTTP instead of `subprocess.run(["ollama", "run", ...])`:
- Subprocess calls block the caller with no usable timeout and
  freeze CMD for the whole generation.
- The HTTP endpoint returns a clean JSON body in one shot with
  "stream": false, and lets us cap the output with num_predict
  so a CPU-only machine (2016-era laptop, 8GB RAM) never
  chokes on huge OCR payloads.

Everything runs against localhost - 100% offline, no document
data ever leaves the machine.

Usage
-----
    from llm_engine import understand_document, health_check, build_context

    if not health_check():
        print("Ollama is not running.")

    result = understand_document(ocr_text)
    print(result["document_type"], result["fields"])
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any

import requests


# ============================================================
# Configuration
# ============================================================

OLLAMA_BASE_URL = "http://localhost:11434"
GENERATE_URL = f"{OLLAMA_BASE_URL}/api/generate"
TAGS_URL = f"{OLLAMA_BASE_URL}/api/tags"

DEFAULT_MODEL = "qwen2.5:3b"

# Single-shot response: wait for the complete answer.
STREAM = False

# Ask Ollama to enforce valid JSON output.
JSON_FORMAT = "json"

# Generation caps for CPU-only hardware.
DEFAULT_NUM_PREDICT = 320
DEFAULT_TEMPERATURE = 0.0

# Explicit context window (tokens) for the Ollama runner. Setting it
# prevents silent prompt truncation inside Ollama if the default ever
# changes. 2048 tokens comfortably fits a 1300-char window plus the
# compact instruction block and a ~384-token JSON response.
DEFAULT_NUM_CTX = 2048

# Context window for LLM (not just head - use head + tail)
DEFAULT_MAX_CONTEXT_CHARS = 2500
DEFAULT_HEADER_CHARS = 1500
DEFAULT_TAIL_CHARS = 1000

# Non-blocking behaviour: short connect timeout, bounded read timeout
# so the CLI/UI never freezes for minutes. A window within the context
# budget (see llm_engine.build_context / document_understanding window
# budgets) normally completes well inside this time on CPU; if it does
# not, the request fails gracefully and processing continues.
CONNECT_TIMEOUT = 5.0
# Read timeout is configurable (SAFEDOC_LLM_READ_TIMEOUT). With bounded
# windows (see document_understanding) a request normally completes in
# ~2-3 minutes on slow CPU-only hardware (~3 tok/s generation); the
# timeout is a safety net that fails gracefully, never the expected
# duration. The old pathology (9 oversized windows x 300s) is fixed by
# window bounding + compact prompts, not by the timeout value.
READ_TIMEOUT = float(os.environ.get("SAFEDOC_LLM_READ_TIMEOUT", "300.0"))
REQUEST_TIMEOUT = (CONNECT_TIMEOUT, READ_TIMEOUT)

# Retry only when Ollama answered but the answer was unusable.
MAX_RETRIES = 2
RETRY_BACKOFF_SECONDS = 2.0


logger = logging.getLogger("SafeDocAI.LLM")


# ============================================================
# Context Building
# ============================================================

def build_context(
    raw_text: str,
    max_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
    header_chars: int = DEFAULT_HEADER_CHARS,
    tail_chars: int = DEFAULT_TAIL_CHARS,
) -> str:
    """
    Build controlled context for LLM from raw OCR text.

    Strategy:
    - Use header (first N chars) + tail (last M chars)
    - This preserves document beginning AND important footer info
      (dates, totals, signatures, final results)
    - Do NOT truncate the original OCR text - only the LLM context

    Args:
        raw_text: Complete OCR text from document
        max_chars: Maximum total characters to send to LLM
        header_chars: Characters from the beginning
        tail_chars: Characters from the end

    Returns:
        Bounded context string suitable for LLM prompt
    """
    if not raw_text:
        return ""

    raw_text = raw_text.strip()
    text_len = len(raw_text)

    # If document is short enough, use it all
    if text_len <= max_chars:
        return raw_text

    # Calculate actual sizes
    actual_header = min(header_chars, text_len)
    remaining = max_chars - actual_header

    # If remaining is less than tail_chars, adjust
    actual_tail = min(tail_chars, remaining, text_len - actual_header)

    if actual_tail <= 0:
        # Can't fit both header and tail, just use header
        return raw_text[:actual_header]

    head = raw_text[:actual_header].rstrip()
    tail = raw_text[-actual_tail:].lstrip()

    omitted = text_len - (actual_header + actual_tail)

    # Build context with clear markers
    context = (
        f"{head}\n\n"
        f"[... {omitted} characters omitted between header and tail ...]\n\n"
        f"{tail}"
    )

    return context


# ============================================================
# JSON parsing
# ============================================================

def extract_json(raw: str) -> dict[str, Any] | None:
    """
    Parse JSON out of an LLM response.

    Handles ```json code fences, surrounding prose, and any
    stray text after the closing brace.
    """

    if not raw:
        return None

    candidate = raw.strip()

    # Strip markdown code fences if the model added them.
    fenced = re.search(
        r"```(?:json)?\s*(.*?)```",
        candidate,
        re.DOTALL | re.IGNORECASE,
    )

    if fenced:
        candidate = fenced.group(1).strip()

    # Direct parse first.
    try:
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    # Fall back to the outermost {...} block.
    start = candidate.find("{")
    end = candidate.rfind("}")

    if start == -1 or end == -1 or end <= start:
        return None

    try:
        parsed = json.loads(candidate[start:end + 1])
    except json.JSONDecodeError:
        return None

    return parsed if isinstance(parsed, dict) else None


# ============================================================
# Prompt building - DYNAMIC document understanding
# ============================================================

def build_prompt(document_text: str) -> str:
    """
    Build prompt for dynamic document understanding.

    The LLM is NOT restricted to a fixed document type enum.
    It should dynamically determine the document type based on
    evidence and extract relevant fields.

    The instruction block is deliberately COMPACT: on slow CPU-only
    machines every prompt token costs prompt-eval time on every call,
    and the output must stay well inside num_predict so the JSON
    never gets truncated mid-object.

    Rules enforced by this prompt:
    - Do not invent information
    - Do not guess missing values
    - Do not create fields without evidence
    - Preserve document-specific information
    - Different documents can have completely different fields
    - evidence_snippet must be copied from supplied OCR text

    Args:
        document_text: Bounded context text for LLM

    Returns:
        Prompt string
    """

    return f"""You are a Document Analyst. Extract structured info from the document text.

Rules:
- Only extract what is visible. No invention, no guessing.
- Mark unknown values as "UNKNOWN".
- Document type and fields are dynamic - any document kind is possible.
- evidence_snippet must be an EXACT substring of the document text.
- If the text contains "[..." omission markers, use only visible text.
- summary: max 20 words. fields: max 10 most important. Snippets: max 100 chars.

Respond ONLY with JSON:
{{"document_type": "type based on evidence", "summary": "...", "fields": [{{"key": "...", "value": "...", "evidence_snippet": "..."}}]}}

Document text:
{document_text}""".strip()


# ============================================================
# Connectivity checks
# ============================================================

def health_check(timeout: float = CONNECT_TIMEOUT) -> bool:
    """
    True if the local Ollama server is reachable.

    Cheap enough to call before every batch run.
    """

    try:
        response = requests.get(TAGS_URL, timeout=timeout)
        return response.ok
    except requests.exceptions.RequestException:
        return False


def is_model_available(model: str = DEFAULT_MODEL) -> bool:
    """
    True if the model is already pulled locally.

    Never triggers a download - this only reads /api/tags.
    """

    try:
        response = requests.get(TAGS_URL, timeout=CONNECT_TIMEOUT)
        response.raise_for_status()

        models = (response.json() or {}).get("models", [])

        return any(
            model in str(item.get("name", ""))
            for item in models
        )

    except (requests.exceptions.RequestException, ValueError):
        return False


# ============================================================
# Core generation
# ============================================================

def generate(
    prompt: str,
    *,
    model: str = DEFAULT_MODEL,
    num_predict: int = DEFAULT_NUM_PREDICT,
    temperature: float = DEFAULT_TEMPERATURE,
    num_ctx: int = DEFAULT_NUM_CTX,
) -> str | None:
    """
    Single-shot, non-blocking generation against Ollama.

    stream=false + format=json + capped num_predict + explicit num_ctx
    keep the local CPU model fast, deterministic and JSON-shaped.

    Returns the raw response text, or None on any failure.
    Never raises and never blocks longer than the timeouts.
    """

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": STREAM,
        "format": JSON_FORMAT,
        "options": {
            "num_predict": num_predict,
            "temperature": temperature,
            "num_ctx": num_ctx,
        },
    }

    attempt = 0

    while attempt <= MAX_RETRIES:

        attempt += 1

        try:
            response = requests.post(
                GENERATE_URL,
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )

            response.raise_for_status()

            response_text = (
                (response.json() or {}).get("response", "").strip()
            )

            if response_text:
                return response_text

            logger.warning(
                "Ollama returned an empty response "
                "(attempt %d/%d).",
                attempt,
                MAX_RETRIES + 1,
            )

        except requests.exceptions.ConnectionError:
            logger.error(
                "Cannot connect to Ollama at %s. "
                "Make sure Ollama is running.",
                OLLAMA_BASE_URL,
            )
            return None

        except requests.exceptions.Timeout:
            logger.error(
                "Ollama request timed out after %.0f seconds.",
                READ_TIMEOUT,
            )
            return None

        except requests.exceptions.RequestException as exc:
            logger.error("Ollama request failed: %s", exc)
            return None

        except Exception as exc:  # defensive - never crash
            logger.error("Unexpected Ollama error: %s", exc)
            return None

        if attempt <= MAX_RETRIES:
            time.sleep(RETRY_BACKOFF_SECONDS)

    return None


# ============================================================
# Document classification
# ============================================================

def _empty_understanding(error: str) -> dict[str, Any]:
    """Schema-shaped failure result (never None, never raises)."""

    return {
        "document_type": "UNKNOWN",
        "summary": "",
        "fields": [],
        "error": error,
    }


def _normalize_understanding(
    parsed: dict[str, Any],
) -> dict[str, Any]:
    """Coerce whatever the model returned into the expected schema."""

    fields = parsed.get("fields")
    if not isinstance(fields, list):
        fields = []

    # Normalize fields to ensure they have required keys
    normalized_fields = []
    for field in fields:
        if isinstance(field, dict):
            normalized_field = {
                "key": str(field.get("key", "")).strip(),
                "value": str(field.get("value", "")).strip(),
                "evidence_snippet": str(
                    field.get("evidence_snippet", "")
                ).strip(),
            }
            normalized_fields.append(normalized_field)

    return {
        "document_type": str(
            parsed.get("document_type", "UNKNOWN")
        ).strip(),
        "summary": str(
            parsed.get("summary", "")
        ).strip(),
        "fields": normalized_fields,
        "error": None,
    }


def understand_document(
    document_text: str,
    *,
    model: str = DEFAULT_MODEL,
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
    num_predict: int = DEFAULT_NUM_PREDICT,
) -> dict[str, Any]:
    """
    Perform dynamic document understanding using local LLM.

    The LLM analyzes the document and dynamically determines:
    - Document type based on evidence (NOT restricted to enum)
    - Summary of the document
    - Structured fields with evidence snippets

    IMPORTANT:
    - Uses build_context() to select header + tail, NOT just head
    - Does NOT blindly truncate to first 2500 chars
    - Preserves footer info (dates, totals, signatures)

    Args:
        document_text: Complete OCR text from document
        model: Ollama model name
        max_context_chars: Max characters to send to LLM
        num_predict: Max tokens to generate

    Returns:
        Dict with:
        - document_type: str (dynamically determined)
        - summary: str
        - fields: list of {key, value, evidence_snippet}
        - error: str or None

    Never raises. On failure returns empty structure with error.
    """

    text = (document_text or "").strip()

    if not text:
        return _empty_understanding(
            "Document text is empty."
        )

    # Build controlled context (header + tail, not just head)
    context = build_context(
        text,
        max_chars=max_context_chars,
        header_chars=DEFAULT_HEADER_CHARS,
        tail_chars=DEFAULT_TAIL_CHARS,
    )

    if len(context) < len(text):
        logger.warning(
            "LLM context truncated from %d to %d chars "
            "(max_context_chars=%d); middle content was dropped.",
            len(text),
            len(context),
            max_context_chars,
        )

    prompt = build_prompt(context)

    logger.info(
        "Sending %d characters to %s "
        "(num_predict=%d, stream=%s)...",
        len(context),
        model,
        num_predict,
        STREAM,
    )

    raw_response = generate(
        prompt,
        model=model,
        num_predict=num_predict,
    )

    if raw_response is None:
        return _empty_understanding(
            "LLM request failed or timed out."
        )

    parsed = extract_json(raw_response)

    if parsed is None:
        logger.error(
            "Ollama returned invalid JSON: %.200s",
            raw_response,
        )
        return _empty_understanding(
            "LLM returned invalid JSON."
        )

    return _normalize_understanding(parsed)


# ============================================================
# Selection mode ("selection instead of transcription")
# ============================================================
#
# The transcription path above asks the model to re-type fields and
# evidence snippets. The selection path instead hands the model a
# deterministic candidate list (from candidate_extractor) and asks it
# to SELECT relevant candidate ids. Values are never transcribed by
# the model: the caller verifies each selection against the exact
# candidate record and derives evidence post-hoc from char_span.

#: Compact output contract for selection calls (keys deliberately short
#: so a K-sized response stays far inside the approved num_predict).
_SELECTION_SCHEMA_REMINDER = (
    'Respond ONLY with JSON: '
    '{"s":[{"i":<candidate id>,"k":"<max 3 words>","v":"<max 36 chars>"}],'
    '"t":"document type","m":"brief meaning"}'
)


def build_selection_prompt(
    window_text: str,
    candidates: list[dict[str, Any]],
    doc_type_hint: str | None = None,
    max_selections: int | None = None,
) -> str:
    """Build the selection-mode prompt for one bounded OCR window.

    Args:
        window_text: The bounded OCR window (already windowed upstream).
        candidates: Candidate dicts with keys ``id``, ``label``, ``value``
            (schema of candidate_extractor.Candidate.to_dict). ``line_no``
            and ``char_span`` are deliberately NOT shown to the model.
        doc_type_hint: Optional heuristic document type (hint only, never
            authoritative; omit or pass UNKNOWN freely).
        max_selections: Optional planner K for this call; stated to the
            model so the selection count stays within the plan.

    Returns:
        Prompt string. The candidate ids/labels/values and the OCR window
        are always included; the type hint only when available.
    """

    lines: list[str] = []

    lines.append(
        "You are a Document Analyst. Below are OCR text and numbered "
        "candidate fields found in it."
    )
    lines.append("")
    lines.append("Rules:")
    lines.append("- Select ONLY candidate ids that hold important document data.")
    lines.append("- i MUST be an existing candidate id from the list. Never invent ids.")
    lines.append("- v MUST copy that candidate's value. Never invent or change values.")
    lines.append("- k: short label, max 3 words. v: max 36 characters.")
    lines.append("- No evidence snippets. No keys outside the schema.")
    if max_selections is not None:
        lines.append(f"- Select at most {int(max_selections)} candidates.")
    else:
        lines.append("- Select only as many candidates as are relevant.")
    if doc_type_hint and str(doc_type_hint).strip().upper() not in {
        "",
        "UNKNOWN",
    }:
        lines.append(
            f"- Likely document type (hint only, verify from text): {doc_type_hint}"
        )
    lines.append(_SELECTION_SCHEMA_REMINDER)

    lines.append("")
    lines.append("Candidates:")
    for cand in candidates:
        lines.append(
            f"{cand.get('id')}. {cand.get('label', '')}: {cand.get('value', '')}"
        )

    lines.append("")
    lines.append("OCR text:")
    lines.append(window_text)

    return "\n".join(lines)


class SelectionParseResult:
    """Outcome of defensively parsing one selection response.

    ``ok`` is True only for a complete, valid, schema-conforming response.
    Any malformed/truncated/over-limit response sets ``ok=False`` with a
    ``reason``; the caller must then retry or fall back. Truncated JSON is
    NEVER partially accepted.
    """

    __slots__ = (
        "ok",
        "reason",
        "selections",
        "document_type",
        "summary",
        "rejected_reasons",
        "repairs",
    )

    def __init__(
        self,
        ok: bool,
        reason: str = "",
        selections: list[dict[str, Any]] | None = None,
        document_type: str = "",
        summary: str = "",
        rejected_reasons: list[str] | None = None,
        repairs: list[str] | None = None,
    ) -> None:
        self.ok = ok
        self.reason = reason
        self.selections = selections or []
        self.document_type = document_type
        self.summary = summary
        self.rejected_reasons = rejected_reasons or []
        self.repairs = repairs or []


def _values_consistent(candidate_value: str, model_value: str) -> bool:
    """True when the model's echoed value is faithful to the candidate.

    Either an exact match or a prefix of the candidate value (the compact
    schema asks the model to clip values to 36 characters; long identifiers
    such as enrollment numbers arrive clipped). The candidate value always
    stays the authoritative one.
    """
    if not candidate_value or not model_value:
        return False
    return model_value == candidate_value or candidate_value.startswith(
        model_value
    )


def parse_selection_response(
    response_text: str | None,
    id_to_candidate: dict[int, dict[str, Any]],
    *,
    max_selections: int,
) -> SelectionParseResult:
    """Defensively parse and validate a selection-mode response.

    Validation (all must pass, else ok=False):
    - JSON parses as a complete object (truncated JSON is rejected whole).
    - Every selection has an integer ``i`` that exists in id_to_candidate.
    - ``v`` matches the candidate's value (exact, or a prefix of it -- the
      model may clip long values to 36 chars; the CANDIDATE value is then
      used as authoritative, so no value is ever invented).
    - Selection count <= max_selections (the planned K). Unknown ids,
      malformed entries, and over-K responses are rejected as a whole.
    - ``k`` is trimmed to at most 3 words (defensive; the candidate label
      remains available for evidence).

    Args:
        response_text: Raw model output (JSON-forced).
        id_to_candidate: Mapping of candidate id -> candidate dict.
        max_selections: Planned K for this call.
    """

    if not response_text or not response_text.strip():
        return SelectionParseResult(False, reason="empty response")

    parsed = extract_json(response_text)

    if parsed is None:
        return SelectionParseResult(
            False, reason="invalid or truncated JSON"
        )

    raw_selections = parsed.get("s")
    if not isinstance(raw_selections, list):
        return SelectionParseResult(
            False, reason="missing or non-list 's'"
        )

    if len(raw_selections) > max_selections:
        return SelectionParseResult(
            False,
            reason=(
                f"{len(raw_selections)} selections exceeds planned K="
                f"{max_selections}"
            ),
        )

    selections: list[dict[str, Any]] = []
    rejected_entries: list[str] = []
    repairs: list[str] = []
    claimed_ids: set[int] = set()

    for entry in raw_selections:
        if not isinstance(entry, dict):
            rejected_entries.append("non-object selection entry")
            continue

        raw_id = entry.get("i")

        # bool is a subclass of int; exclude it explicitly. Digit-string and
        # integral-float ids ("1" / 1.0 -- common JSON-forced model quirks)
        # are safely coerced BEFORE lookup: the id must still exist in the
        # candidate map and its value must still match, so coercion can never
        # invent a selection. Truly malformed ids are rejected.
        if isinstance(raw_id, bool):
            rejected_entries.append(
                f"non-integer candidate id: {raw_id!r}"
            )
            continue

        if isinstance(raw_id, int):
            candidate_id = raw_id
        elif isinstance(raw_id, float) and float(raw_id).is_integer():
            candidate_id = int(raw_id)
        elif isinstance(raw_id, str) and raw_id.strip().isdigit():
            candidate_id = int(raw_id.strip())
        else:
            rejected_entries.append(
                f"non-integer candidate id: {raw_id!r}"
            )
            continue

        model_value = str(entry.get("v", "")).strip()

        if not model_value:
            rejected_entries.append(
                f"empty value for candidate id {raw_id}"
            )
            continue

        claimed = id_to_candidate.get(candidate_id)
        claimed_value = "" if claimed is None else str(claimed.get("value", "")).strip()

        if claimed is not None and _values_consistent(claimed_value, model_value):
            final_id = candidate_id
        else:
            # The model mispaired id and value (observed on real qwen2.5:3b
            # output: 1-based ids against the 0-based candidate list, so
            # every value belongs to the neighbouring candidate). The VALUE
            # is the authoritative anchor, so the entry is re-paired only
            # when exactly ONE offered candidate carries a consistent value;
            # ambiguous or unmatched values are rejected. This never invents
            # a value and never re-pairs silently (repairs are reported).
            matching_ids: list[int] = [
                cand_id
                for cand_id, cand in id_to_candidate.items()
                if _values_consistent(
                    str(cand.get("value", "")).strip(), model_value
                )
            ]
            unique_match: int | None = (
                matching_ids[0] if len(matching_ids) == 1 else None
            )

            if unique_match is None:
                if claimed is None:
                    if matching_ids:
                        rejected_entries.append(
                            f"unknown candidate id {raw_id}: model said "
                            f"{model_value!r}, which matches "
                            f"{len(matching_ids)} offered candidates "
                            f"ambiguously"
                        )
                    else:
                        rejected_entries.append(
                            f"unknown candidate id {raw_id}: model said "
                            f"{model_value!r}, which matches no offered candidate"
                        )
                else:
                    rejected_entries.append(
                        f"value mismatch for candidate id {raw_id}: "
                        f"model said {model_value!r}, candidate is "
                        f"{claimed_value!r}"
                    )
                continue

            repairs.append(f"claimed id {raw_id} -> candidate {unique_match}")
            final_id = unique_match
            claimed = id_to_candidate[final_id]
            claimed_value = str(claimed.get("value", "")).strip()

        if final_id in claimed_ids:
            rejected_entries.append(
                f"duplicate selection of candidate id {final_id}"
            )
            continue

        claimed_ids.add(final_id)
        candidate = claimed
        candidate_value = claimed_value

        # Defensive k normalization: max 3 words. The authoritative label
        # for evidence remains the candidate's own label.
        key_words = str(entry.get("k", "")).split()
        key = " ".join(key_words[:3]).strip()

        if not key:
            key = str(candidate.get("label", "")).strip()

        selections.append(
            {
                "id": final_id,
                "key": key,
                "value": candidate_value,
                "line_no": candidate.get("line_no"),
                "char_span": candidate.get("char_span"),
            }
        )

    # A response whose EVERY entry was rejected is a failed response (the
    # model produced nothing trustworthy); the caller retries or falls back.
    # A mixed response keeps only its verified selections; the rejections
    # are reported in ``rejected_reasons`` for telemetry, never stored.
    if not selections and rejected_entries:
        return SelectionParseResult(
            False,
            reason="; ".join(rejected_entries[:3]),
        )

    document_type = str(parsed.get("t", "")).strip()
    summary = str(parsed.get("m", "")).strip()

    return SelectionParseResult(
        True,
        selections=selections,
        document_type=document_type,
        summary=summary,
        rejected_reasons=rejected_entries,
        repairs=repairs,
    )


def generate_with_meta(
    prompt: str,
    *,
    model: str = DEFAULT_MODEL,
    num_predict: int = DEFAULT_NUM_PREDICT,
    temperature: float = DEFAULT_TEMPERATURE,
    num_ctx: int = DEFAULT_NUM_CTX,
) -> dict[str, Any] | None:
    """Like :func:`generate` but also reports done_reason and token usage.

    Returns ``{"text", "done_reason", "eval_count", "latency_seconds"}``
    or None on transport failure. done_reason is what Ollama reported
    ("stop" normally, "length" when num_predict truncated the output) and
    drives the selection path's truncation retry decision.
    """

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": STREAM,
        "format": JSON_FORMAT,
        "options": {
            "num_predict": num_predict,
            "temperature": temperature,
            "num_ctx": num_ctx,
        },
    }

    start = time.time()

    try:
        response = requests.post(
            GENERATE_URL,
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )

        response.raise_for_status()

    except requests.exceptions.RequestException as exc:
        logger.error("Ollama selection request failed: %s", exc)
        return None

    except Exception as exc:  # defensive - never crash
        logger.error("Unexpected Ollama selection error: %s", exc)
        return None

    try:
        body = response.json() or {}
    except ValueError:
        logger.error("Ollama returned a non-JSON body.")
        return None

    response_text = str(body.get("response", "")).strip()

    return {
        "text": response_text,
        "done_reason": body.get("done_reason"),
        "eval_count": body.get("eval_count"),
        "latency_seconds": round(time.time() - start, 3),
    }


# ============================================================
# Module test
# ============================================================

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    print("========================================")
    print(" SafeDocAI - llm_engine.py Test")
    print("========================================")

    if not health_check():
        print("Ollama is NOT reachable at", OLLAMA_BASE_URL)
        print("Start it with: ollama serve")
        raise SystemExit(1)

    print("Ollama is reachable at", OLLAMA_BASE_URL)

    print(
        "Model available:",
        is_model_available(DEFAULT_MODEL),
    )

    sample_text = (
        "GOVERNMENT OF INDIA\n"
        "INCOME TAX DEPARTMENT\n"
        "Permanent Account Number Card\n"
        "Name: ASHA KUMARI\n"
        "Father's Name: RAM KUMAR\n"
        "Date of Birth: 15/08/1992\n"
        "Permanent Account Number: ABCDE1234F\n"
    )

    result = understand_document(
        sample_text,
        max_context_chars=5000,
        num_predict=512,
    )

    print()
    print(json.dumps(result, indent=4, ensure_ascii=False))
    print()
    print("LLM engine test complete.")