"""
Shared lightweight OCR chunking helpers.

This module is intentionally small and dependency-light so it can be reused
by both the storage pipeline and the document-understanding pipeline without
creating circular imports.

It does NOT try to be a full semantic chunker. It focuses on:
- keeping numeric values such as 6.61 and 1200.50 intact
- keeping lines/rows together where possible
- producing chunks small enough for local LLM windows
"""

from __future__ import annotations

import re
from typing import Sequence


# ------------------------------------------------------------------
# Decimal-safe helpers
# ------------------------------------------------------------------


_DECIMAL_LIKE_RE = re.compile(r"[0-9]+(?:[.,][0-9]+)*$")


def _split_points(text: str, chunk_size: int) -> list[int]:
    """
    Return candidate split positions that avoid breaking decimal-like tokens.

    We never split inside a numeric token such as `6.61`, `1200.50`,
    `1,20,000`. For non-decimal text we still prefer line boundaries first.
    """

    if not text:
        return []

    positions: list[int] = []
    length = len(text)

    # Prefer newline boundaries first.
    for idx, ch in enumerate(text):
        if ch in "\n\r":
            positions.append(idx + 1)

    if not positions:
        # Fallback: space/punctuation boundaries.
        for idx, ch in enumerate(text):
            if ch in " \t,;:!?\u2013\u2014":
                positions.append(idx + 1)

    if not positions:
        # Absolute last resort: every character boundary.
        positions = list(range(1, length))

    kept: list[int] = []

    for pos in positions:
        # Keep the split point only if it does not split a decimal-like run.
        start = max(0, pos - 6)
        window = text[start:pos]
        if _DECIMAL_LIKE_RE.search(window) and _DECIMAL_LIKE_RE.search(text[pos:pos + 6]):
            continue
        kept.append(pos)

    return kept


def chunk_text_safe(
    text: str,
    *,
    chunk_size: int = 1200,
    overlap: int = 180,
) -> list[str]:
    """
    Split OCR text into controlled chunks without breaking decimals.

    Strategy:
    - Prefer line breaks as split points
    - Avoid splitting numeric tokens like 6.61 or 1200.50
    - Add overlap so boundary context is not lost
    - Keep chunk sizes bounded and predictable
    """

    text = text.strip()
    if not text:
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    if len(text) <= chunk_size:
        return [text]

    split_positions = _split_points(text, chunk_size)

    chunks: list[str] = []
    start = 0

    for pos in split_positions:
        if pos <= start:
            continue

        chunk_end = min(pos, start + chunk_size)
        chunk = text[start:chunk_end].strip()
        if chunk:
            chunks.append(chunk)

        start = max(0, chunk_end - overlap)

        if start >= len(text):
            break

    if start < len(text):
        tail = text[start:].strip()
        if tail:
            chunks.append(tail)

    # Merge tiny stragglers with the previous chunk when possible.
    merged: list[str] = []
    for chunk in chunks:
        if not merged:
            merged.append(chunk)
            continue

        prev = merged[-1]
        if len(prev) + 1 + len(chunk) <= chunk_size * 1.25:
            merged[-1] = f"{prev} {chunk}".strip()
        else:
            merged.append(chunk)

    return merged


def chunk_windows(
    chunks: Sequence[str],
    *,
    window_size: int = 3,
    stride: int = 1,
) -> list[list[str]]:
    """
    Create overlapping windows over an ordered chunk list.

    Each window is a small list of consecutive chunks.
    This is useful for multi-window LLM processing where each window
    gets one bounded LLM call.
    """

    if not chunks:
        return []

    if window_size < 1:
        window_size = 1

    if stride < 1:
        stride = 1

    windows: list[list[str]] = []
    idx = 0
    n = len(chunks)

    while idx < n:
        window = chunks[idx: idx + window_size]
        if not window:
            break
        windows.append(list(window))
        idx += stride

    return windows


def score_chunk_relevance(chunk: str) -> float:
    """
    Lightweight local relevance heuristic for a chunk.

    It is intentionally document-type agnostic. It simply rewards chunks
    that contain:
    - numbers
    - names/labels
    - dates
    - structured-looking text (colons, pipes, tables, indentation, etc.)
    """

    if not chunk:
        return 0.0

    has_number = bool(re.search(r"[0-9]", chunk))
    has_date = bool(
        re.search(
            r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b|\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{4}\b",
            chunk,
            re.IGNORECASE,
        )
    )
    has_label = bool(re.search(r"[A-Za-z\u0900-\u097F\u0B80-\u0BFF]{2,}:", chunk))
    has_structure = bool(
        re.search(r"[|:;\u0964\u0965]|[-]{2,}|[=]{2,}", chunk)
    )
    has_upper_name = bool(re.search(r"\b[A-Z][A-Z ]{3,}\b", chunk))

    score = 0.0
    if has_number:
        score += 1.0
    if has_date:
        score += 2.0
    if has_label:
        score += 1.5
    if has_structure:
        score += 1.0
    if has_upper_name:
        score += 0.75

    return score


def select_relevant_windows(
    chunks: Sequence[str],
    *,
    max_windows: int = 6,
    window_size: int = 3,
    min_score: float = 0.0,
    stride: int = 1,
) -> list[list[str]]:
    """
    Choose a controlled set of relevant chunk windows.

    Rules:
    - Score each window by the best chunk inside it
    - Keep the highest scoring windows first
    - Always keep neighbors of selected windows so boundary context survives
    - Cap total windows for local hardware
    - If uncertain, keep neighboring chunks rather than risk data loss
    """

    if not chunks:
        return []

    windows = chunk_windows(chunks, window_size=window_size, stride=stride)
    if not windows:
        return []

    # Score each window by its best chunk.
    scored: list[tuple[float, int, list[str]]] = []
    for index, window in enumerate(windows):
        best = max(score_chunk_relevance(chunk) for chunk in window)
        scored.append((best, index, window))

    # Keep only windows above the relevance threshold when possible.
    interesting = [(s, i, w) for s, i, w in scored if s >= min_score]

    if not interesting:
        # If nothing looks obviously relevant, keep a few spread windows.
        step = max(1, len(windows) // max_windows)
        interesting = [(s, i, w) for s, i, w in scored[::step]][:max_windows]

    # Sort by score desc, then keep top N windows.
    interesting.sort(key=lambda item: (-item[0], item[1]))
    selected_indices = {item[1] for item in interesting[:max_windows]}

    # Always include immediate neighbors of selected windows.
    for idx in list(selected_indices):
        if idx - 1 >= 0:
            selected_indices.add(idx - 1)
        if idx + 1 < len(windows):
            selected_indices.add(idx + 1)

    ordered: list[list[str]] = []
    for idx in sorted(selected_indices):
        window = windows[idx]
        if window and window not in ordered:
            ordered.append(window)

    return ordered
