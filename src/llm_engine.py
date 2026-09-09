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
DEFAULT_NUM_PREDICT = 384
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