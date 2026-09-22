"""
SafeDocAI - Selection Engine (approved "selection instead of transcription" path)
==================================================================================

First production integration glue for:

    Candidate Extractor -> LLM Selection -> Evidence Attachment/Validation
    -> Position-Aware Merge -> existing storage contract

Responsibilities (all local; the only non-deterministic step is the LLM call):

1.  For one OCR window: extract candidates (candidate_extractor), plan the
    calls (selection_budget planner), make the bounded Ollama calls
    (llm_engine.generate_with_meta), parse/validate each response defensively
    (llm_engine.parse_selection_response), and retry on truncation
    (done_reason == "length") with halved K per the approved retry rule.
2.  Attach evidence post-hoc by slicing the candidate's original char_span
    out of the ORIGINAL raw OCR text (never paraphrased or normalized).
3.  Position-aware merge across windows: fields merge by (normalized key,
    source position bucket), so repeated values from different sections
    (multi-semester SGPA, multiple dates) never collapse.
4.  Any selection failure (invalid response after retries, transport error,
    no usable candidates) raises SelectionUnavailable; the caller falls back
    to the existing legacy extraction path. An invalid LLM output is never
    converted into valid-looking data.

Nothing here talks to SQLite or Chroma; storage stays in storage_engine.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from candidate_extractor import (
    Candidate,
    extract_candidates_for_window_with_stats,
    has_usable_candidates,
)
from llm_engine import (
    build_selection_prompt,
    generate_with_meta,
    parse_selection_response,
)
from selection_budget import (
    SelectionPlan,
    plan_selection_calls,
    split_candidate_ids,
    truncation_retry_plan,
)
try:  # src/ on sys.path (pipeline style)
    from document_evidence import discover_identifier_fields
except ImportError:  # project root on sys.path (test/tooling style)
    from src.document_evidence import discover_identifier_fields

logger = logging.getLogger("SafeDocAI.Selection")


# ---------------------------------------------------------------------------
# Position-aware merge configuration
# ---------------------------------------------------------------------------

#: Two same-key selections farther apart than this (in OCR characters) come
#: from different document sections and stay separate fields even when their
#: values are identical (repeated "Session" blocks on different pages).
#: One marksheet page is roughly 800-1200 chars, so 800 is a conservative
#: section boundary.
_MERGE_SAME_VALUE_POSITION_SPAN_CHARS = 800


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class SelectionUnavailable(Exception):
    """Raised when the selection path cannot produce a usable result.

    The caller must fall back to the existing legacy extraction path; this
    exception never wraps stored data.
    """


# ---------------------------------------------------------------------------
# Window helpers
# ---------------------------------------------------------------------------

_OMISSION_MARKER_RE = re.compile(r"\[\.\.\..*?\.\.\.\]", re.DOTALL)


def _slice_text(window: str) -> str:
    """Strip omission markers so candidates come only from real OCR text.

    Marker text is synthetic; without this the deterministic extractor would
    treat "characters omitted" as a label.
    """
    return _OMISSION_MARKER_RE.sub("", window)


def _usable_window_text(window: str) -> bool:
    """A window is usable when it carries real OCR text (not only markers)."""
    return bool(_slice_text(window).strip())


def _identifier_candidates(
    raw_text: str,
    window_start: int,
    window_end: int,
) -> list[dict[str, Any]]:
    """Structural identifier candidates for this window's region (Step 2).

    Uses the generic structural-layout discovery in
    ``document_evidence.discover_identifier_fields`` (header-row tables and
    caption-below values, which the line-based label:value extractor cannot
    see) and keeps only discoveries whose value starts inside
    ``[window_start, window_end)``. ``char_span`` values are absolute.
    """

    if window_end <= window_start or window_end > len(raw_text):
        return []

    discovered = discover_identifier_fields(raw_text)
    return [
        record
        for record in discovered
        if window_start <= record["char_span"][0] < window_end
    ]

# ---------------------------------------------------------------------------
# Per-window selection
# ---------------------------------------------------------------------------


def select_window(
    raw_text: str,
    window_text: str,
    window_start: int,
    window_end: int,
    *,
    model: str,
    doc_type_hint: str | None,
    call_log: list[dict[str, Any]] | None = None,
    max_attempts_per_slice: int = 2,
) -> dict[str, Any]:
    """Run candidate extraction + bounded LLM selection for ONE window.

    Args:
        raw_text: Full original OCR text (evidence source of truth).
        window_text: The window as handed to the LLM (may contain omission
            markers; markers are excluded from candidate extraction).
        window_start: Absolute start of the window in raw_text (inclusive).
        window_end: Absolute end of the window in raw_text (exclusive);
            candidates must start inside [window_start, window_end).
        model: Ollama model name.
        doc_type_hint: Heuristic type hint (None/UNKNOWN allowed).
        call_log: Optional list; every LLM call appends telemetry here.
        max_attempts_per_slice: Max parse attempts per candidate slice
            (includes the approved truncation retries).

    Returns:
        Dict with ``selected_fields`` (defensively validated selections
        carrying candidate id/key/value/line_no/char_span), ``candidates``,
        ``candidate_count``, ``doc_type_votes``, ``summary_parts`` and the
        plan telemetry.

    Raises:
        SelectionUnavailable: when candidates are unusable or any planned
            call cannot be completed validly (invalid JSON, unknown ids,
            value mismatch, over-K, truncation after retries, transport
            error). The caller then falls back to the legacy path.
    """
    if not _usable_window_text(window_text):
        raise SelectionUnavailable("empty window text")

    if not isinstance(window_start, int) or not isinstance(window_end, int):
        raise SelectionUnavailable("invalid window bounds")

    if window_start < 0 or window_end <= window_start:
        raise SelectionUnavailable("invalid window bounds")

    if window_end > len(raw_text) or window_start >= len(raw_text):
        raise SelectionUnavailable("window outside raw text")

    # --- Deterministic candidates for exactly this window -------------------
    candidates, stats = extract_candidates_for_window_with_stats(
        raw_text,
        window_start,
        window_end,
    )

    if not has_usable_candidates(candidates):
        raise SelectionUnavailable(
            f"only {stats.returned} usable candidates in window"
        )

    candidate_dicts = [c.to_dict() for c in candidates]
    id_to_candidate: dict[int, dict[str, Any]] = {
        c.id: c.to_dict() for c in candidates
    }

    # --- Step 2: deterministic identifier candidates (priority offer) ------
    # Generic structural layouts (header-row tables, caption-below values)
    # are invisible to the line-based extractor above. Structural identifier
    # candidates are APPENDED with higher ids and offered FIRST, so a
    # bounded budget always spends on them before line-based candidates;
    # every candidate is still offered exactly once. Both pools pass through
    # the same defensive response validation, and the evidence validator
    # remains the sole authority for what gets stored.
    identifier_candidates = _identifier_candidates(raw_text, window_start, window_end)
    priority_ids: list[str] = []
    if identifier_candidates:
        next_id = len(candidate_dicts)
        for record in identifier_candidates:
            entry = {
                "id": next_id,
                "label": record["family"],
                "value": record["value"],
                "line_no": record["line_no"],
                "char_span": record["char_span"],
            }
            candidate_dicts.append(entry)
            id_to_candidate[next_id] = entry
            priority_ids.append(str(next_id))
            next_id += 1
        logger.info(
            "Identifier candidates discovered in window [%d:%d): %s",
            window_start,
            window_end,
            [(c["family"], c["value"]) for c in identifier_candidates],
        )

    # --- Approved planner: slices, budget, K --------------------------------
    plan: SelectionPlan = plan_selection_calls(len(candidate_dicts))

    if plan.total_calls == 0:
        raise SelectionUnavailable("planner produced no calls")

    offered_ids = priority_ids + [str(c.id) for c in candidates]
    slices = split_candidate_ids(offered_ids, plan.slice_sizes)

    # --- Bounded selection calls with truncation retry ----------------------
    collected: list[dict[str, Any]] = []
    doc_type_votes: list[str] = []
    summary_parts: list[str] = []
    call_index = 0
    last_parse_reason = "no attempts left"

    for slice_index, slice_ids in enumerate(slices):
        slice_candidates = [candidate_dicts[int(cid)] for cid in slice_ids]
        k = plan.slice_sizes[slice_index]
        attempts_left = max(1, int(max_attempts_per_slice))
        parsed = None

        while attempts_left > 0:
            call_index += 1
            prompt = build_selection_prompt(
                window_text,
                slice_candidates,
                doc_type_hint=doc_type_hint,
                max_selections=k,
            )

            meta = generate_with_meta(
                prompt,
                model=model,
                num_predict=plan.num_predict,
            )

            if meta is None:
                if call_log is not None:
                    call_log.append(
                        {
                            "window_start": window_start,
                            "slice": slice_index,
                            "call": call_index,
                            "num_predict": plan.num_predict,
                            "offered": k,
                            "error": "transport failure",
                            "latency_seconds": None,
                            "eval_count": None,
                            "done_reason": None,
                            "json_valid": False,
                        }
                    )
                raise SelectionUnavailable("LLM transport failure")

            response_text = meta["text"]
            done_reason = meta.get("done_reason")

            parsed = parse_selection_response(
                response_text,
                id_to_candidate,
                max_selections=k,
            )

            if call_log is not None:
                call_log.append(
                    {
                        "window_start": window_start,
                        "slice": slice_index,
                        "call": call_index,
                        "num_predict": plan.num_predict,
                        "offered": k,
                        "eval_count": meta.get("eval_count"),
                        "done_reason": done_reason,
                        "latency_seconds": meta.get("latency_seconds"),
                        "json_valid": bool(parsed.ok),
                        "accepted_selections": len(parsed.selections)
                        if parsed.ok
                        else 0,
                        "rejected_entries": len(parsed.rejected_reasons),
                        "repairs": len(parsed.repairs),
                        "parse_reason": None if parsed.ok else parsed.reason,
                    }
                )

            if parsed.ok:
                break

            # Approved truncation rule: same slice, halved K, bounded tries.
            if done_reason == "length":
                decision = truncation_retry_plan(
                    attempted_k=k,
                    attempts_left=attempts_left,
                )
                if decision.should_retry:
                    k = decision.retry_k
                    slice_candidates = slice_candidates[:k]
                    attempts_left -= 1
                    parsed = None
                    continue
                # Floor reached: give up on this window.
                attempts_left = 0
                break

            # Non-truncation failure (invalid JSON, unknown ids, over-K,
            # value mismatch): one bounded re-ask, then give up.
            last_parse_reason = parsed.reason
            attempts_left -= 1
            parsed = None

        if parsed is None or not parsed.ok:
            raise SelectionUnavailable(
                "selection failed after retries: "
                f"{parsed.reason if parsed and not parsed.ok else last_parse_reason}"
            )

        collected.extend(parsed.selections)

        if parsed.document_type:
            doc_type_votes.append(parsed.document_type)

        if parsed.summary:
            summary_parts.append(parsed.summary)

    return {
        "selected_fields": collected,
        "candidates": candidate_dicts,
        "candidate_count": len(candidate_dicts),
        "doc_type_votes": doc_type_votes,
        "summary_parts": summary_parts,
        "plan": {
            "num_predict": plan.num_predict,
            "slice_sizes": list(plan.slice_sizes),
            "total_calls": plan.total_calls,
        },
        "window_start": window_start,
        "window_end": window_end,
    }


# ---------------------------------------------------------------------------
# Evidence attachment (post-hoc, from ORIGINAL OCR)
# ---------------------------------------------------------------------------


def attach_evidence(
    selected_fields: list[dict[str, Any]],
    raw_text: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Slice evidence snippets from the ORIGINAL raw OCR via char_span.

    Returns ``(accepted, rejected)``. A selection is rejected when its span
    is invalid (wrong shape, out of bounds, inverted) or slices to an empty
    snippet; it is never given invented evidence. The stored snippet
    preserves the original OCR characters exactly (no paraphrase, no
    whitespace normalization).
    """
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for field in selected_fields or []:
        span = field.get("char_span")

        if (
            not isinstance(span, (list, tuple))
            or len(span) != 2
            or not all(isinstance(x, int) for x in span)
            or span[0] < 0
            or span[1] > len(raw_text)
            or span[0] >= span[1]
        ):
            rejected.append({**field, "rejection_reason": "invalid span"})
            continue

        start, end = span
        snippet = raw_text[start:end]

        if not snippet.strip():
            rejected.append({**field, "rejection_reason": "empty snippet"})
            continue

        accepted.append(
            {
                "key": str(field.get("key", "")),
                "value": str(field.get("value", "")),
                "evidence_snippet": snippet,
                "candidate_id": field.get("id"),
                "line_no": field.get("line_no"),
                "char_span": [start, end],
            }
        )

    return accepted, rejected


# ---------------------------------------------------------------------------
# Position-aware merge
# ---------------------------------------------------------------------------


def _position_bucket(char_span: list[int] | tuple[int, int] | None) -> int:
    """Bucket a field's source position for merge identity."""
    if (
        not isinstance(char_span, (list, tuple))
        or len(char_span) != 2
        or not all(isinstance(x, int) for x in char_span)
    ):
        return -1
    return char_span[0] // _MERGE_SAME_VALUE_POSITION_SPAN_CHARS


def position_aware_merge(
    window_results: list[dict[str, Any]],
    raw_text: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Merge accepted fields across windows WITHOUT collapsing repeats.

    Identity rule: two accepted fields are the SAME field only when their
    normalized keys match AND their source positions are close
    (|span_start difference| < _MERGE_SAME_VALUE_POSITION_SPAN_CHARS) AND
    their values match. Everything else is kept as a separate record.

    Consequences (all intended):
    - SGPA 6.09 / 6.05 / 7.00 / 7.12 / 6.61 stay separate (values differ).
    - Two identical "Session 2024-25" values from different sections stay
      separate (positions far apart).
    - The same field re-found in overlapping windows collapses into one
      record (same key, same value, close positions).

    Returns ``(fields, merge_info)`` where merge_info reports kept/collapsed
    counts for telemetry. Output preserves first-seen (document) order.
    """
    merged: list[dict[str, Any]] = []
    collapsed_count = 0

    def _norm_key(key: str) -> str:
        return re.sub(r"\s+", " ", str(key or "")).strip().casefold()

    for result in window_results or []:
        for field in result.get("accepted_fields", []):
            key_norm = _norm_key(field.get("key", ""))
            value_norm = re.sub(
                r"\s+", " ", str(field.get("value", ""))
            ).strip().casefold()
            span = field.get("char_span") or [-1, -1]
            start = span[0]

            duplicate_of = None
            for existing in merged:
                ex_span = existing.get("char_span") or [-1, -1]
                if (
                    _norm_key(existing.get("key", "")) == key_norm
                    and _position_bucket(existing.get("char_span"))
                    == _position_bucket(span)
                    and abs(ex_span[0] - start)
                    < _MERGE_SAME_VALUE_POSITION_SPAN_CHARS
                    and _norm_key_value(existing.get("value", ""))
                    == value_norm
                ):
                    duplicate_of = existing
                    break

            if duplicate_of is not None:
                collapsed_count += 1
                continue

            merged.append(field)

    merge_info = {
        "accepted_input": sum(
            len(r.get("accepted_fields", [])) for r in window_results or []
        ),
        "kept": len(merged),
        "collapsed_duplicates": collapsed_count,
        "position_span_chars": _MERGE_SAME_VALUE_POSITION_SPAN_CHARS,
    }
    return merged, merge_info


def _norm_key_value(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()
