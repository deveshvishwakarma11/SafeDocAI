"""
SafeDocAI - Dynamic Document Understanding (Phase 3)

Orchestration layer for document understanding.

Processing order:
1. Load Phase 1 JSON (OCR output)
2. Get raw OCR text (complete, not truncated)
3. Run heuristics.py (deterministic pre-classifier)
4. Build LLM context windows (single or multi-window for long docs)
5. Call llm_engine.py for dynamic document understanding
6. Combine heuristic evidence + LLM result
7. Validate LLM fields against ORIGINAL OCR text
8. Save verified understanding to data/understood/

All inference runs 100% offline against http://localhost:11434
over HTTP - no subprocess calls, no terminal freezing.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from pathlib import Path
from typing import Any

from llm_engine import (
    DEFAULT_MAX_CONTEXT_CHARS,
    DEFAULT_MODEL,
    DEFAULT_NUM_PREDICT,
    health_check,
    is_model_available,
    understand_document,
)

from heuristics import (
    get_document_type_confidence,
    is_high_confidence_classification,
)

from evidence_validator import (
    calculate_overall_confidence,
    determine_classification_source,
    validate_document_type,
    validate_fields,
)

from document_evidence import (
    claim_is_contradicted,
    priority_identifier_fields,
)

from storage_engine import (
    store_understanding_results,
    get_db_connection,
)

from selection_engine import (
    SelectionUnavailable,
    attach_evidence,
    position_aware_merge,
    select_window,
)


def _normalize_field_key(key: Any) -> str:
    """Step 2: same key normalization as the evidence layer's family keys."""

    return re.sub(r"[\s_\-]+", " ", str(key)).strip().casefold()


def _merge_priority_identifier_fields(
    validated_fields: list[dict[str, Any]],
    raw_text: str,
) -> list[dict[str, Any]]:
    """Step 2: merge evidence-validated structural identifiers (generic).

    ``document_evidence.priority_identifier_fields`` discovers identifier
    fields from structural layouts the line-based candidate extractor cannot
    see (header-row tables, caption-below values), then re-validates every
    candidate against the ORIGINAL OCR text with the authoritative evidence
    validator: validation failure means the field is dropped, never stored.

    Merge policy (no per-document-type logic):
    * a family already present AND verified is never duplicated/overridden;
    * a same-family field that failed validation (unverified) is replaced by
      the evidence-verified candidate (strictly better grounding);
    * absent families are added with ``verified: True``.
    """

    verified_fields = [
        field for field in validated_fields if field.get("verified", False)
    ]

    try:
        priority_fields = priority_identifier_fields(
            raw_text,
            existing_fields=verified_fields,
        )
    except Exception as exc:  # defensive: never break the main pipeline
        logger.warning("Priority identifier merge failed: %s", exc)
        return validated_fields

    if not priority_fields:
        return validated_fields

    priority_keys = {
        _normalize_field_key(field["key"]) for field in priority_fields
    }
    merged = [
        field
        for field in validated_fields
        if not (
            _normalize_field_key(field.get("key", "")) in priority_keys
            and not field.get("verified", False)
        )
    ]
    merged.extend(priority_fields)

    logger.info(
        "Priority identifier fields merged: %s",
        [(f["key"], f["value"]) for f in priority_fields],
    )
    return merged


def get_document_id_from_db(file_path: str | Path) -> int | None:
    """Get the SQLite document_id for an original document path."""

    with get_db_connection() as connection:
        existing = connection.execute(
            "SELECT id FROM documents WHERE file_path = ? ORDER BY id DESC LIMIT 1",
            (str(file_path),),
        ).fetchone()

        if existing:
            return int(existing[0])

    return None


PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_DIR = PROJECT_ROOT / "data" / "output"
UNDERSTANDING_DIR = PROJECT_ROOT / "data" / "understood"

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)

logger = logging.getLogger("SafeDocAI.Understanding")


# ------------------------------------------------------------------
# Long-document LLM context configuration
# ------------------------------------------------------------------

DEFAULT_LLM_WINDOW_CHARS = 1800
DEFAULT_MAX_LLM_WINDOWS = 5
DEFAULT_LLM_CHUNKS_PER_WINDOW = 2
DEFAULT_MIN_CHUNK_RELEVANCE = 0.0
DEFAULT_LLM_WINDOW_STRIDE = 1

# ------------------------------------------------------------------
# Bounded-window configuration (CPU-laptop friendly)
# ------------------------------------------------------------------
#
# Hard invariant: a window handed to the LLM must ALWAYS fit inside
# llm_engine's context budget (DEFAULT_MAX_CONTEXT_CHARS). The old
# behavior built ~4300-char windows and then silently truncated them
# again inside understand_document(), losing evidence.

# Prompt overhead in build_prompt(): rules + JSON schema template.
# Measured empirically (~950 chars); padded for safety.
LLM_PROMPT_OVERHEAD_CHARS = 1200

# Total character budget per LLM request: window text + prompt overhead
# must stay under this. Defaults to llm_engine's context limit.
DEFAULT_LLM_REQUEST_BUDGET_CHARS = max(
    600,
    DEFAULT_MAX_CONTEXT_CHARS - LLM_PROMPT_OVERHEAD_CHARS,
)

# How much overlap between consecutive windows (fraction of budget), so
# fields split across a window boundary are still seen by the LLM.
DEFAULT_LLM_WINDOW_OVERLAP_CHARS = 200

# Ceiling on LLM calls for ONE document. Prevents accidental explosion
# on CPU-only hardware. If a document needs more windows than this to
# cover every character, the windows are spread head/tail-anchored over
# the text with clearly marked omissions (same contract as
# llm_engine.build_context) instead of making more calls.
DEFAULT_MAX_LLM_CALLS_PER_DOCUMENT = 4

# ---------------------------------------------------------------------------
# Selection-mode configuration (approved "selection instead of transcription")
# ---------------------------------------------------------------------------

#: Master switch for the candidate-selection path. True = the per-window
#: pipeline extracts deterministic candidates and the LLM selects ids; any
#: selection failure falls back to the existing legacy extraction below.
SELECTION_MODE_ENABLED = True

#: Bounded re-asks per candidate slice before falling back (the approved
#: retry rule lives in selection_budget.truncation_retry_plan; this only
#: bounds the outer loop).
SELECTION_MAX_ATTEMPTS_PER_SLICE = 2


# Optional runtime override of the per-document LLM call cap.
_max_llm_calls_override: int | None = None


def _set_max_llm_calls(max_calls: int) -> None:
    """Runtime override for the per-document LLM call cap (CLI/ops use)."""
    global _max_llm_calls_override
    _max_llm_calls_override = max(1, int(max_calls))


def load_json_files() -> list[Path]:
    """Find all Phase 1 OCR JSON files."""
    return sorted(OUTPUT_DIR.glob("*.json"))


def load_document(json_path: Path) -> dict[str, Any]:
    """Load one OCR JSON file."""
    with open(json_path, "r", encoding="utf-8") as file:
        return json.load(file)


def get_raw_text(data: dict[str, Any]) -> str:
    """
    Extract complete OCR text from Phase 1 JSON.
    Returns the FULL text, not truncated.
    """
    raw_text = data.get("raw_text")
    if isinstance(raw_text, str) and raw_text.strip():
        return raw_text
    return ""


def get_extraction_method(data: dict[str, Any]) -> str:
    """Get the extraction method used for this document."""
    return data.get("extraction_method", "unknown")


def _chunker_should_use_multi_window(text_length: int) -> bool:
    """Decide whether a document should use multiple LLM windows.

    A document only needs multi-window when it cannot fit into ONE
    bounded LLM request. Everything that fits within a single request
    budget is sent as a single call (fast path for short documents).
    """
    return text_length > DEFAULT_LLM_REQUEST_BUDGET_CHARS


def _build_windows_for_document(raw_text: str) -> list[str]:
    """Build controlled, bounded LLM context windows for a document.

    Guarantees (document-type agnostic):

    1. Every returned window is STRICTLY within the per-request budget
       (DEFAULT_LLM_REQUEST_BUDGET_CHARS), so llm_engine never has to
       re-truncate and silently drop evidence.

    2. Short documents produce exactly ONE window (one LLM call).

    3. At most DEFAULT_MAX_LLM_CALLS_PER_DOCUMENT windows are produced
       for long documents - no accidental explosion to 9/15 calls.

    3. At most DEFAULT_MAX_LLM_CALLS_PER_DOCUMENT windows are produced
       for long documents - no accidental explosion to 9/15 calls.

    4. Consecutive long-document windows overlap where possible so
       values split across a window boundary are still seen by the LLM.
       When the call cap forces gaps, the gap is marked with an explicit
       "[... N characters omitted ...]" marker inside the window.

    5. Coverage: window 1 starts at the beginning of the OCR text and
       the last window ends at the end of it (head+tail anchored), so
       headers AND footers survive.
    """
    if not raw_text or not raw_text.strip():
        return []

    text = raw_text.strip()
    text_length = len(text)
    budget = DEFAULT_LLM_REQUEST_BUDGET_CHARS

    # --- Fast path: whole document fits in a single bounded request ---
    if not _chunker_should_use_multi_window(text_length):
        return [build_prompt_context_for_window(text)]

    # --- Long path: bounded number of overlapping, budget-sized windows ---
    max_calls = max(
        1,
        _max_llm_calls_override or DEFAULT_MAX_LLM_CALLS_PER_DOCUMENT,
    )

    # Number of windows needed to cover the text, given the overlap.
    # Window w covers [w * step, w * step + budget), where step = budget
    # minus overlap. This guarantees full coverage with bounded calls.
    coverage_per_window = budget - DEFAULT_LLM_WINDOW_OVERLAP_CHARS
    windows_needed = max(
        1,
        -(-text_length // max(1, coverage_per_window)),
    )

    # Cap at max_calls: wider windows, fewer calls. Windows must stay
    # within the budget, so if the text is longer than max_calls can
    # cover, we prioritize head (headers carry document type) and tail
    # (totals/signatures) instead of making more calls.
    if windows_needed > max_calls:
        logger.info(
            "Long document: %d chars would need %d windows without "
            "overlap; capped to %d LLM calls with overlapping or "
            "omission-marked windows.",
            text_length,
            windows_needed,
            max_calls,
        )
        windows_needed = max_calls

    if windows_needed == 1:
        # Forced single call (cap=1) for an over-budget document:
        # single window with head+tail anchors.
        return [_build_single_over_budget_window(text, budget)]

    # Distribute text over windows_needed windows. step =
    # (text_length - budget) / (windows_needed - 1) anchors the LAST
    # window exactly at the end of the text. While windows_needed fits
    # under the cap the windows overlap; if the cap forced fewer windows
    # than needed, step exceeds the budget and gaps appear - those gaps
    # are then made explicit with omission markers.
    step = (text_length - budget) // (windows_needed - 1)

    if step > budget:
        logger.warning(
            "Document too long for full LLM coverage in %d calls: "
            "gaps of ~%d chars will be marked as omitted "
            "(raise DEFAULT_MAX_LLM_CALLS_PER_DOCUMENT for full coverage).",
            windows_needed,
            step - budget,
        )

    starts = [w * step for w in range(windows_needed)]
    starts[-1] = max(starts[-1], text_length - budget)

    windows: list[str] = []
    for index, start in enumerate(starts):
        end = min(start + budget, text_length)
        content = text[start:end]

        if index + 1 < len(starts):
            next_start = starts[index + 1]

            if next_start > end:
                # The call cap forced a gap - mark it so the LLM does
                # not treat a mid-document cut as the document end.
                omitted = next_start - end
                marker = f"\n[... {omitted} characters omitted ...]\n"
                content = (
                    content[: max(0, budget - len(marker))] + marker
                )

        if content.strip():
            windows.append(build_prompt_context_for_window(content))

        if end >= text_length:
            break

    return windows


def _build_single_over_budget_window(text: str, budget: int) -> str:
    """Single window for a text slightly over budget.

    Keeps the head (document type usually appears at the top) and the
    tail (totals/dates/signatures at the bottom), with a clearly marked
    omission in between - the same contract as llm_engine.build_context
    but applied ONCE, visibly, instead of silently inside the engine.
    """
    head_chars = max(0, int(budget * 0.6))
    tail_chars = max(0, budget - head_chars)

    head = text[:head_chars].rstrip()
    tail = text[-tail_chars:].lstrip() if tail_chars else ""
    omitted = len(text) - (len(head) + len(tail))

    if omitted <= 0:
        return build_prompt_context_for_window(text)

    return build_prompt_context_for_window(
        f"{head}\n\n[... {omitted} characters omitted ...]\n\n{tail}"
    )


def build_prompt_context_for_window(window_text: str) -> str:
    """Prepare a bounded LLM context string for one window."""
    return window_text.strip()


def _merge_llm_windows(
    window_results: list[dict[str, Any]],
    raw_text: str,
) -> dict[str, Any]:
    """Combine multiple LLM window results into one understanding.

    Rules:
    - Never let one weak window overwrite a stronger result.
    - For duplicate keys, prefer the result whose snippet is better
      supported by the FULL raw OCR text.
    - Preserve useful fields discovered in different windows.
    """
    combined: dict[str, dict[str, Any]] = {}

    for result in window_results:
        if not isinstance(result, dict):
            continue

        fields = result.get("fields")
        if not isinstance(fields, list):
            continue

        for field in fields:
            if not isinstance(field, dict):
                continue

            key = str(field.get("key", "")).strip()
            value = str(field.get("value", "")).strip()
            snippet = str(field.get("evidence_snippet", "")).strip()

            if not key:
                continue

            current = combined.get(key)

            replacement = {
                "key": key,
                "value": value,
                "evidence_snippet": snippet,
                "verified": False,
            }

            if current is None:
                combined[key] = replacement
                continue

            current_score = _field_evidence_score(
                current["value"],
                current["evidence_snippet"],
                raw_text,
            )
            candidate_score = _field_evidence_score(
                value,
                snippet,
                raw_text,
            )

            # Only replace when the candidate is clearly better supported.
            if candidate_score > current_score:
                combined[key] = replacement

    fields_out = list(combined.values())

    # Prefer the most common non-UNKNOWN document type across windows.
    type_votes: dict[str, int] = {}
    for result in window_results:
        doc_type = (
            str(result.get("document_type", ""))
            if isinstance(result, dict)
            else ""
        )

        if not doc_type or doc_type.strip().upper() in {
            "UNKNOWN",
            "",
            "NONE",
            "N/A",
        }:
            continue

        type_votes[doc_type] = type_votes.get(doc_type, 0) + 1

    chosen_type = ""
    if type_votes:
        chosen_type = max(type_votes, key=lambda t: (type_votes[t], len(t)))

    summary_parts: list[str] = []
    for result in window_results:
        summary = str(result.get("summary", "")).strip()
        if summary and summary not in summary_parts:
            summary_parts.append(summary)

    summary = " ".join(summary_parts).strip()

    return {
        "document_type": chosen_type or "UNKNOWN",
        "summary": summary,
        "fields": fields_out,
    }


def _field_evidence_score(value: str, snippet: str, raw_text: str) -> float:
    """Lightweight score for how well a field appears to be supported.

    This is intentionally simple and document-type agnostic. It is used
    only for merge preference, not as a final verification step.
    """
    if not raw_text:
        return 0.0

    base = 0.0

    if value and value.strip().upper() not in {
        "UNKNOWN",
        "",
        "NONE",
        "N/A",
    }:
        if _normalize_for_merge(value) in _normalize_for_merge(raw_text):
            base += 1.0
        elif snippet and _normalize_for_merge(snippet) in _normalize_for_merge(
            raw_text
        ):
            base += 0.5

    return base


def _normalize_for_merge(text: str) -> str:
    """Normalize text for simple merge-time evidence comparison."""
    if not text:
        return ""

    normalized = text.strip().lower()
    normalized = re.sub(r"[\s,\.]+(\d)", r" \1", normalized)
    normalized = re.sub(r"\s+", " ", normalized)

    return normalized


def save_understanding(
    json_path: Path,
    original_data: dict[str, Any],
    understanding: dict[str, Any],
    model: str,
) -> Path:
    """Save verified understanding to data/understood/."""

    UNDERSTANDING_DIR.mkdir(parents=True, exist_ok=True)

    output_path = UNDERSTANDING_DIR / json_path.name

    result = {
        "source_file": original_data.get("file_name", json_path.stem),
        "source_json": json_path.name,
        "ai_model": model,
        "understanding": understanding,
    }

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)

    return output_path


# ---------------------------------------------------------------------------
# Selection-mode helpers (approved candidate-selection path)
# ---------------------------------------------------------------------------

_WINDOW_MARKER_RE = re.compile(r"\[\.\.\..*?\.\.\.\]", re.DOTALL)


def _strip_window_markers(window_text: str) -> str:
    """Remove synthetic omission markers, leaving only real OCR content."""
    return _WINDOW_MARKER_RE.sub("", window_text)


def _window_bounds(
    windows: list[str],
    raw_text: str,
) -> list[tuple[int, int, int]]:
    """Locate each window's real OCR content as absolute spans.

    Returns ``(window_index, start, end)`` triples. A window whose text
    contains an omission marker (head+tail join) yields TWO contiguous
    regions - head and tail - so footer candidates are never silently
    excluded from candidate extraction.

    Each region is located by matching its first line from a forward cursor
    and walking its lines to find the end. Slight end imprecision is
    harmless: candidate spans only need the right [start, end) neighborhood,
    and position-aware merge collapses genuine cross-window duplicates.
    """
    regions: list[tuple[int, int, int]] = []
    cursor = 0
    text_length = len(raw_text)

    for window_index, window_text in enumerate(windows, start=1):
        # Split on omission markers FIRST: each piece is a contiguous region
        # of the original OCR (head, tail, or a plain single-region window).
        pieces = [
            piece.strip()
            for piece in _WINDOW_MARKER_RE.split(window_text)
            if piece.strip()
        ]

        search_from = cursor
        last_start = cursor

        for piece_index, piece in enumerate(pieces):
            lines = [ln.strip() for ln in piece.split("\n") if ln.strip()]
            if not lines:
                continue

            # Pieces after the first (a joined tail) live near the document
            # end, far from the running cursor - search the whole text.
            base = search_from if piece_index == 0 else 0

            probe = lines[0][:120]
            start = raw_text.find(probe, base)
            if start < 0:
                start = raw_text.find(probe)
            if start < 0:
                start = base

            # Region end: walk the piece's lines to find where its content
            # ends (good precision on real, non-repetitive OCR text).
            pos = start
            for line in lines:
                idx = raw_text.find(line, pos)
                if idx < 0:
                    break
                pos = idx + len(line)

            end = min(text_length, max(start + 1, pos))
            if end > start:
                regions.append((window_index, max(0, start), end))
                last_start = start

        # Next window overlaps this one, so its content starts at or after
        # this window's first matched position (not after its end).
        cursor = max(0, min(last_start + 1, text_length))

    # Tail-anchor correction: the window builder guarantees the LAST window
    # ends at the end of the OCR text (head+tail anchoring). Extend the final
    # region accordingly so footer candidates are never excluded.
    if regions:
        region_index, region_start, _ = regions[-1]
        regions[-1] = (region_index, region_start, text_length)

    return regions


def _run_selection_mode(
    raw_text: str,
    windows: list[str],
    heuristic_type: str,
    model: str,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Run the approved selection pipeline over every window.

    Returns ``(selection_result, call_log)``. ``selection_result`` is None
    when NO window produced a valid selection (caller falls back to legacy).
    A single window's failure never destroys the document: remaining windows
    still contribute, and the failure is reported in the telemetry.
    """
    call_log: list[dict[str, Any]] = []
    window_regions = _window_bounds(windows, raw_text)
    window_results: list[dict[str, Any]] = []
    window_stats: list[dict[str, Any]] = []

    doc_type_hint = (
        heuristic_type if heuristic_type and heuristic_type != "UNKNOWN" else None
    )

    for window_index, w_start, w_end in window_regions:
        window_stat: dict[str, Any] = {
            "window": window_index,
            "window_start": w_start,
            "window_end": w_end,
        }

        try:
            result = select_window(
                raw_text,
                windows[window_index - 1],
                w_start,
                w_end,
                model=model,
                doc_type_hint=doc_type_hint,
                call_log=call_log,
                max_attempts_per_slice=SELECTION_MAX_ATTEMPTS_PER_SLICE,
            )
        except SelectionUnavailable as exc:
            logger.warning(
                "Selection unavailable for window region %d [%d:%d): %s",
                window_index,
                w_start,
                w_end,
                exc,
            )
            window_stat["status"] = "unusable"
            window_stat["reason"] = str(exc)
            window_stats.append(window_stat)
            continue

        accepted, rejected = attach_evidence(
            result["selected_fields"], raw_text
        )

        result["accepted_fields"] = accepted
        result["rejected_fields"] = rejected
        result["window_index"] = window_index
        window_results.append(result)

        window_stat.update(
            {
                "status": "ok",
                "candidate_count": result["candidate_count"],
                "selected": len(result["selected_fields"]),
                "accepted": len(accepted),
                "rejected_span": len(rejected),
                "plan": result["plan"],
            }
        )
        window_stats.append(window_stat)

    if not window_results:
        return None, call_log

    merged_fields, merge_info = position_aware_merge(window_results, raw_text)

    # Document type: majority vote across this window's selection calls,
    # heuristic remains only a downstream cross-check (never authoritative).
    type_votes: dict[str, int] = {}
    for result in window_results:
        for vote in result.get("doc_type_votes", []):
            vote_clean = str(vote).strip()
            if vote_clean and vote_clean.upper() not in {
                "UNKNOWN",
                "NONE",
                "N/A",
            }:
                type_votes[vote_clean] = type_votes.get(vote_clean, 0) + 1

    chosen_type = ""
    if type_votes:
        chosen_type = max(type_votes, key=lambda t: (type_votes[t], len(t)))

    summary_parts: list[str] = []
    for result in window_results:
        for part in result.get("summary_parts", []):
            part_clean = str(part).strip()
            if part_clean and part_clean not in summary_parts:
                summary_parts.append(part_clean)

    total_selected = sum(
        len(r.get("selected_fields", [])) for r in window_results
    )
    total_accepted = sum(
        len(r.get("accepted_fields", [])) for r in window_results
    )
    total_rejected = sum(
        len(r.get("rejected_fields", [])) for r in window_results
    )
    failed_windows = sum(
        1 for s in window_stats if s.get("status") != "ok"
    )

    valid_calls = [c for c in call_log if c.get("json_valid")]
    truncated_calls = [
        c for c in call_log if c.get("done_reason") == "length"
    ]

    telemetry = {
        "path": "selection",
        "windows_total": len(windows),
        "windows_ok": len(window_results),
        "windows_unusable": failed_windows,
        "candidate_counts": [
            s.get("candidate_count")
            for s in window_stats
            if s.get("status") == "ok"
        ],
        "selection_calls": len(call_log),
        "valid_selection_calls": len(valid_calls),
        "truncated_calls": len(truncated_calls),
        "calls": call_log,
        "selected_total": total_selected,
        "evidence_accepted": total_accepted,
        "evidence_rejected": total_rejected,
        "merge": merge_info,
        "window_stats": window_stats,
        "type_votes": type_votes,
    }

    fields_out = [
        {
            "key": f["key"],
            "value": f["value"],
            "evidence_snippet": f["evidence_snippet"],
        }
        for f in merged_fields
    ]

    selection_result = {
        "document_type": chosen_type,
        "summary": " ".join(summary_parts).strip(),
        "fields": fields_out,
        "telemetry": telemetry,
    }

    return selection_result, call_log


def process_document(
    json_path: Path,
    model: str = DEFAULT_MODEL,
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
    num_predict: int = DEFAULT_NUM_PREDICT,
) -> dict[str, Any] | None:
    """Process one document through the full Phase 3 pipeline.

    Returns understanding dict or None on failure.
    """

    logger.info("Processing: %s", json_path.name)

    start_time = time.time()

    # Load Phase 1 JSON
    data = load_document(json_path)

    # Get complete raw OCR text (NOT truncated)
    raw_text = get_raw_text(data)

    if not raw_text:
        logger.warning("No raw_text found. Skipping: %s", json_path.name)
        return None

    extraction_method = get_extraction_method(data)
    text_length = len(raw_text)

    logger.info(
        "OCR text loaded: %d characters (method: %s)",
        text_length,
        extraction_method,
    )

    # Step 1: Run heuristic classifier
    heuristic_start = time.time()
    heuristic_result = get_document_type_confidence(raw_text)
    heuristic_time = time.time() - heuristic_start

    heuristic_type = heuristic_result["document_type"]
    heuristic_confidence = heuristic_result["confidence"]

    logger.info(
        "Heuristic classification: %s (confidence: %.2f, time: %.3fs)",
        heuristic_type,
        heuristic_confidence,
        heuristic_time,
    )

    # Step 2: Build LLM context windows (single or multi-window)
    windows = _build_windows_for_document(raw_text)

    if not windows:
        logger.error(
            "No LLM contexts could be built for %s", json_path.name
        )
        return None

    logger.info(
        "LLM windows: %d (from %d total OCR chars)",
        len(windows),
        text_length,
    )

    # Step 3: Document understanding. Preferred path is the approved
    # candidate-selection architecture (deterministic candidates -> bounded
    # LLM selection -> post-hoc evidence -> position-aware merge); the
    # existing legacy transcription path is the fallback and stays intact.
    selection_result: dict[str, Any] | None = None
    selection_call_log: list[dict[str, Any]] = []
    window_results: list[dict[str, Any]] = []
    llm_start = time.time()

    if SELECTION_MODE_ENABLED:
        selection_result, selection_call_log = _run_selection_mode(
            raw_text,
            windows,
            heuristic_type,
            model,
        )

    if selection_result is not None:
        llm_result = {
            "document_type": selection_result["document_type"],
            "summary": selection_result["summary"],
            "fields": selection_result["fields"],
            "error": None,
        }
        llm_time = time.time() - llm_start
    else:
        if SELECTION_MODE_ENABLED:
            logger.info(
                "Selection path unavailable for %s; falling back to "
                "legacy extraction.",
                json_path.name,
            )

        for window_index, window_text in enumerate(windows, start=1):
            logger.info(
                "LLM window %d/%d: %d chars",
                window_index,
                len(windows),
                len(window_text),
            )

            window_result = understand_document(
                window_text,
                model=model,
                max_context_chars=max_context_chars,
                num_predict=num_predict,
            )

            if window_result.get("error"):
                logger.warning(
                    "LLM window %d failed for %s: %s",
                    window_index,
                    json_path.name,
                    window_result.get("error"),
                )
                continue

            window_results.append(window_result)

        llm_time = time.time() - llm_start

    if selection_result is None and not window_results:
        logger.error("All LLM windows failed for %s", json_path.name)

        if heuristic_type != "UNKNOWN":
            logger.info("Falling back to heuristic classification only.")
            merged = {
                "document_type": heuristic_type,
                "summary": "LLM unavailable - heuristic classification used",
                "fields": [],
            }
        else:
            return None

        validated_fields = validate_fields(
            [],
            raw_text,
            require_all_verified=False,
        )

        # Step 2: generic structural identifier merge (evidence-verified
        # only) -- the heuristic-only fallback still reports identifiers
        # such as PNR when the OCR evidence supports them.
        validated_fields = _merge_priority_identifier_fields(
            validated_fields,
            raw_text,
        )

        final_understanding = {
            "document_type": merged["document_type"],
            "summary": merged["summary"],
            "fields": validated_fields,
            "classification_source": "HEURISTIC",
            "confidence": "LOW",
            "processing": {
                "extraction_method": extraction_method,
                "ocr_text_length": text_length,
                "llm_context_length": sum(len(w) for w in windows),
                "llm_window_count": len(windows),
                "heuristic_time_seconds": round(heuristic_time, 3),
                "llm_time_seconds": round(llm_time, 3),
                "total_time_seconds": round(time.time() - start_time, 3),
            },
        }

        if merged.get("error"):
            final_understanding["error"] = merged["error"]

        output_path = save_understanding(
            json_path,
            data,
            final_understanding,
            model,
        )

        logger.info(
            "Understanding saved: %s", output_path
        )

        original_doc_path = data.get("file_path", json_path)

        if validated_fields:
            storage_result = store_understanding_results(
                output_path,
                final_understanding,
                original_document_path=original_doc_path,
            )

            if storage_result.get("error"):
                logger.warning(
                    "Failed to store verified metadata in SQLite: %s",
                    storage_result["error"],
                )
            else:
                logger.info(
                    "Stored %d verified fields in SQLite (document_id=%d)",
                    storage_result.get("stored", 0),
                    storage_result.get("document_id"),
                )
                final_understanding["document_id"] = storage_result.get(
                    "document_id"
                )

        if "document_id" in final_understanding:
            _rewrite_understanding_file(
                output_path,
                data,
                model,
                final_understanding,
            )

        return final_understanding


    # Merge window results before downstream validation (legacy path only;
    # the selection path already merged position-aware).
    if selection_result is None:
        merged = _merge_llm_windows(window_results, raw_text)
        llm_result = {
            "document_type": merged["document_type"],
            "summary": merged["summary"],
            "fields": merged["fields"],
            "error": None,
        }

    if llm_result.get("error"):
        logger.error(
            "LLM understanding failed for %s: %s",
            json_path.name,
            llm_result["error"],
        )
        # Still save heuristic result if available
        if heuristic_type != "UNKNOWN":
            logger.info("Falling back to heuristic classification only.")
            llm_result = {
                "document_type": heuristic_type,
                "summary": "LLM unavailable - heuristic classification used",
                "fields": [],
                "error": "LLM unavailable",
            }
        else:
            return None

    logger.info(
        "LLM classification: %s (time: %.3fs)",
        llm_result.get("document_type", "UNKNOWN"),
        llm_time,
    )

    # Step 4: Validate document type
    llm_type = llm_result.get("document_type", "UNKNOWN")
    validated_llm_type, llm_type_verified = validate_document_type(
        llm_type,
        raw_text,
        heuristic_type if heuristic_type != "UNKNOWN" else None,
    )

    logger.info(
        "Document type verification: %s (verified: %s)",
        validated_llm_type,
        llm_type_verified,
    )

    # Step 5: Validate fields against ORIGINAL OCR text
    llm_fields = llm_result.get("fields", [])
    validated_fields = validate_fields(
        llm_fields,
        raw_text,  # Use ORIGINAL full text, not truncated context
        require_all_verified=False,  # Keep unverified fields but mark them
    )

    # Step 2: generic structural identifier merge (evidence-verified only).
    # Runs AFTER the authoritative validate_fields so every merged value is
    # itself evidence-validated against the ORIGINAL OCR text.
    validated_fields = _merge_priority_identifier_fields(
        validated_fields,
        raw_text,
    )

    verified_count = sum(
        1 for f in validated_fields if f.get("verified", False)
    )
    logger.info(
        "Field validation: %d/%d verified",
        verified_count,
        len(validated_fields),
    )

    # Step 6: Determine final classification
    # Conservative document type selection:
    # 1. If LLM type is verified -> trust it
    # 2. If heuristic is high-confidence and evidence-backed -> can use it
    # 3. If LLM unverified and heuristic not reliable -> use UNKNOWN

    final_document_type = "UNKNOWN"
    classification_source = determine_classification_source(
        heuristic_type if heuristic_type != "UNKNOWN" else None,
        validated_llm_type
        if validated_llm_type != "UNKNOWN"
        else None,
        heuristic_confidence,
        llm_type_verified,
    )

    # Determine final document type based on verification status
    if llm_type_verified and validated_llm_type != "UNKNOWN":
        # LLM classification is verified - trust it
        final_document_type = validated_llm_type
    elif (
        is_high_confidence_classification(heuristic_confidence)
        and heuristic_type != "UNKNOWN"
    ):
        # Heuristic is high-confidence - can use as fallback.
        # Usable when the LLM claim is UNKNOWN, agrees with the heuristic,
        # or was actively CONTRADICTED by strong evidence of another family
        # (Step 2: e.g. a ticket's boilerplate "Voter Identity Card" claim
        # vs. the registry family that actually dominates the text). This
        # stays fully generic: the rule is evidence-based, not per-type.
        llm_claim_contradicted = (
            validated_llm_type != "UNKNOWN"
            and not llm_type_verified
            and claim_is_contradicted(raw_text, validated_llm_type)
        )
        if (
            validated_llm_type == "UNKNOWN"
            or validated_llm_type.upper() == heuristic_type.upper()
            or llm_claim_contradicted
        ):
            final_document_type = heuristic_type
            if classification_source == "LLM":
                classification_source = "HYBRID"

    # Step 7: Calculate overall confidence
    overall_confidence = calculate_overall_confidence(
        validated_fields,
        llm_type_verified,
        heuristic_confidence,
        heuristic_type != "UNKNOWN",
    )

    # Step 8: Build final understanding
    final_understanding = {
        "document_type": final_document_type,
        "summary": llm_result.get("summary", ""),
        "fields": validated_fields,
        "classification_source": classification_source,
        "confidence": overall_confidence,
        "processing": {
            "extraction_method": extraction_method,
            "ocr_text_length": text_length,
            "llm_context_length": sum(len(w) for w in windows),
            "llm_window_count": len(windows),
            "heuristic_time_seconds": round(heuristic_time, 3),
            "llm_time_seconds": round(llm_time, 3),
            "total_time_seconds": round(time.time() - start_time, 3),
        },
    }

    # Selection-path telemetry: full provenance of every selection call
    # (eval_count, done_reason, latency, JSON validity) and the rejected-
    # evidence audit trail. The legacy path simply omits this block.
    if selection_result is not None:
        final_understanding["selection_telemetry"] = selection_result[
            "telemetry"
        ]

    # Add error info if present
    if llm_result.get("error"):
        final_understanding["error"] = llm_result["error"]

    # Save understanding
    output_path = save_understanding(
        json_path, data, final_understanding, model
    )

    logger.info("Understanding saved to %s", output_path)

    logger.info(
        "Final classification: %s [%s]",
        final_document_type,
        classification_source,
    )
    logger.info(
        "Confidence: %s | Verified fields: %d/%d",
        overall_confidence,
        verified_count,
        len(validated_fields),
    )

    # Store metadata in SQLite using the original document path.
    original_doc_path = data.get("file_path", json_path)

    if verified_count > 0:
        storage_result = store_understanding_results(
            output_path,  # Phase 3 JSON path (for reference)
            final_understanding,
            original_document_path=original_doc_path,  # ORIGINAL document path
        )

        if storage_result.get("error"):
            logger.warning(
                "Failed to store verified metadata in SQLite: %s",
                storage_result["error"],
            )
        else:
            logger.info(
                "Stored %d verified fields in SQLite (document_id=%d)",
                storage_result.get("stored", 0),
                storage_result.get("document_id"),
            )
            # Add document_id to understanding for future reference
            final_understanding["document_id"] = storage_result.get(
                "document_id"
            )

    # Update the saved understanding with document_id if available
    if "document_id" in final_understanding:
        _rewrite_understanding_file(
            output_path,
            data,
            model,
            final_understanding,
        )

    return final_understanding


def _rewrite_understanding_file(
    output_path: Path,
    original_data: dict[str, Any],
    model: str,
    understanding: dict[str, Any],
) -> None:
    """Atomically rewrite the saved Phase 3 understanding JSON."""
    UNDERSTANDING_DIR.mkdir(parents=True, exist_ok=True)

    payload = {
        "source_file": original_data.get("file_name", output_path.stem),
        "source_json": output_path.name,
        "ai_model": model,
        "understanding": understanding,
    }

    temp_path = output_path.with_suffix(".tmp")

    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    temp_path.replace(output_path)


def process_file(
    json_path: Path,
    model: str = DEFAULT_MODEL,
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
    num_predict: int = DEFAULT_NUM_PREDICT,
) -> dict[str, Any] | None:
    """Process one OCR JSON document."""
    return process_document(
        json_path,
        model=model,
        max_context_chars=max_context_chars,
        num_predict=num_predict,
    )


def process_all(
    model: str = DEFAULT_MODEL,
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
    num_predict: int = DEFAULT_NUM_PREDICT,
) -> list[dict[str, Any]]:
    """Process every OCR JSON automatically."""

    json_files = load_json_files()

    if not json_files:
        logger.warning("No JSON files found in data/output.")
        return []

    logger.info("Found %d document(s).", len(json_files))

    results = []
    for json_path in json_files:
        result = process_file(
            json_path,
            model=model,
            max_context_chars=max_context_chars,
            num_predict=num_predict,
        )
        if result:
            results.append(result)

    return results


def main():
    parser = argparse.ArgumentParser(
        description=(
            "SafeDocAI Dynamic Document Understanding "
            "(local Ollama HTTP engine)"
        )
    )

    parser.add_argument("--file", help="Process one OCR JSON file")

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Ollama model (default: {DEFAULT_MODEL})",
    )

    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=DEFAULT_MAX_CONTEXT_CHARS,
        help=(
            "Max OCR characters sent to the model "
            f"(default: {DEFAULT_MAX_CONTEXT_CHARS})"
        ),
    )

    parser.add_argument(
        "--num-predict",
        type=int,
        default=DEFAULT_NUM_PREDICT,
        help=(
            "Max tokens generated per document "
            f"(default: {DEFAULT_NUM_PREDICT})"
        ),
    )

    parser.add_argument(
        "--max-llm-windows",
        type=int,
        default=None,
        help=(
            "Hard cap on LLM calls per document "
            f"(default: {DEFAULT_MAX_LLM_CALLS_PER_DOCUMENT})"
        ),
    )

    parser.add_argument(
        "--check",
        action="store_true",
        help="Check Ollama connectivity/model and exit",
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help="Run test suite on sample documents",
    )

    args = parser.parse_args()

    if args.check:
        if not health_check():
            logger.error(
                "Ollama is not reachable at "
                "http://localhost:11434. "
                "Start it with: ollama serve"
            )
            return

        logger.info("Ollama is reachable.")
        logger.info(
            "Model %s available: %s",
            args.model,
            is_model_available(args.model),
        )
        return

    if args.test:
        run_test_suite(model=args.model)
        return

    if args.file:
        json_path = Path(args.file)

        if not json_path.is_absolute():
            json_path = PROJECT_ROOT / json_path

        if not json_path.exists():
            logger.error("File not found: %s", json_path)
            return

        if args.max_llm_windows is not None:
            _set_max_llm_calls(max(1, args.max_llm_windows))

        process_file(
            json_path,
            model=args.model,
            max_context_chars=args.max_context_chars,
            num_predict=args.num_predict,
        )
    else:
        if args.max_llm_windows is not None:
            _set_max_llm_calls(max(1, args.max_llm_windows))

        process_all(
            model=args.model,
            max_context_chars=args.max_context_chars,
            num_predict=args.num_predict,
        )


def run_test_suite(model: str = DEFAULT_MODEL):
    """Run test suite on all sample documents."""

    print("=" * 70)
    print("SafeDocAI - Phase 3 Test Suite")
    print("=" * 70)

    if not health_check():
        print("ERROR: Ollama is not reachable at localhost:11434")
        print("Start it with: ollama serve")
        return

    print(f"\nModel: {model}")
    print(f"Model available: {is_model_available(model)}\n")

    json_files = load_json_files()

    if not json_files:
        print("No JSON files found in data/output/")
        return

    print(f"Found {len(json_files)} document(s)\n")

    results_summary = []

    for json_path in json_files:
        print("-" * 70)
        print(f"Document: {json_path.stem}")
        print("-" * 70)

        data = load_document(json_path)
        raw_text = get_raw_text(data)
        extraction_method = get_extraction_method(data)

        print(f"\n1. OCR STATUS")
        print(f"   Extraction method: {extraction_method}")
        print(f"   Text length: {len(raw_text)} characters")

        if extraction_method == "ocr_fallback" or "ocr" in extraction_method.lower():
            print("   OCR fallback: YES (embedded text was corrupt/empty)")
        else:
            print("   OCR fallback: NO (embedded text used)")

        # Run heuristic
        print(f"\n2. HEURISTIC CLASSIFICATION")
        heuristic_result = get_document_type_confidence(raw_text)
        print(f"   Document type: {heuristic_result['document_type']}")
        print(f"   Confidence: {heuristic_result['confidence']:.2f}")
        print(f"   Strong matches: {heuristic_result['strong_matches']}")
        print(f"   Medium matches: {heuristic_result['medium_matches']}")
        if heuristic_result["evidence_details"]:
            print("   Evidence:")
            for evidence in heuristic_result["evidence_details"][:5]:
                print(
                    f"     - [{evidence['type']}] {evidence['description']}"
                )

        # Build windows preview
        windows = _build_windows_for_document(raw_text)
        print(f"\n3. WINDOWING")
        print(f"   OCR chars: {len(raw_text)}")
        print(f"   Context windows: {len(windows)}")
        for idx, w in enumerate(windows, start=1):
            print(f"     Window {idx}: {len(w)} chars")

        # Run LLM
        print(f"\n4. LLM CLASSIFICATION")
        llm_start = time.time()
        llm_result = understand_document(
            raw_text,
            model=model,
            max_context_chars=DEFAULT_MAX_CONTEXT_CHARS,
            num_predict=DEFAULT_NUM_PREDICT,
        )
        llm_time = time.time() - llm_start

        if llm_result.get("error"):
            print(f"   ERROR: {llm_result['error']}")
        else:
            print(
                f"   Document type: {llm_result.get('document_type', 'UNKNOWN')}"
            )
            print(f"   Processing time: {llm_time:.2f}s")
            print(f"   Fields extracted: {len(llm_result.get('fields', []))}")
            if llm_result.get("fields"):
                print("   Extracted fields:")
                for field in llm_result["fields"][:5]:
                    verified_mark = (
                        "\u2713" if field.get("verified") else "\u2717"
                    )
                    print(
                        f"     {verified_mark} {field['key']}: "
                        f"{field['value'][:50]}"
                    )

        # Validate
        print(f"\n5. VALIDATION")
        validated_fields = validate_fields(
            llm_result.get("fields", []),
            raw_text,
        )
        verified = sum(1 for f in validated_fields if f.get("verified"))
        print(f"   Verified fields: {verified}/{len(validated_fields)}")

        llm_type = llm_result.get("document_type", "UNKNOWN")
        type_verified, _ = validate_document_type(
            llm_type, raw_text, heuristic_result["document_type"]
        )
        print(f"   Document type verified: {type_verified}")

        print()
        results_summary.append(
            {
                "file": json_path.stem,
                "extraction_method": extraction_method,
                "heuristic_type": heuristic_result["document_type"],
                "heuristic_confidence": heuristic_result["confidence"],
                "llm_type": llm_type,
                "llm_time": llm_time,
                "verified_fields": verified,
                "total_fields": len(validated_fields),
                "type_verified": type_verified,
            }
        )

    # Summary
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    for result in results_summary:
        print(f"\n{result['file']}:")
        print(f"  Extraction: {result['extraction_method']}")
        print(
            f"  Heuristic: {result['heuristic_type']} "
            f"(conf: {result['heuristic_confidence']:.2f})"
        )
        print(f"  LLM: {result['llm_type']} ({result['llm_time']:.2f}s)")
        print(
            f"  Verified: {result['verified_fields']}/"
            f"{result['total_fields']} fields, "
            f"type={result['type_verified']}"
        )

    print("\n" + "=" * 70)
    print("Test suite complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
