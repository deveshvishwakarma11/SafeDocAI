"""
Phase 7: User Interface Layer (Streamlit).
==========================================

A lightweight, 100%-local web UI for SafeDocAI. Renders documents from
SQLite and answers natural-language questions by delegating EVERYTHING
retrieval/LLM-related to :func:`answer_engine.answer_query` — the UI
implements no SQL, no Chroma search, no LLM prompting, no grounding
validation of its own.

Privacy:
* Local filesystem paths only; Ollama on localhost only.
* No external frontend assets, analytics, or remote APIs.
* No upload/import functionality in this phase.

Failure handling: every degradation path (Ollama/SQLite/Chroma
unavailable, malformed result, empty query, no documents, insufficient
context) becomes a friendly banner or info message — never a raw
stack trace. The full exception is logged (console) for diagnosis.

Launch:
    streamlit run src/ui.py
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any

# Make `src` importable when launched via `streamlit run src/ui.py`.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

try:  # src/ on sys.path (pipeline style)
    import storage_engine
    from answer_engine import INSUFFICIENT_CONTEXT_MESSAGE, answer_query
except ImportError:  # project root on sys.path
    from src.answer_engine import INSUFFICIENT_CONTEXT_MESSAGE, answer_query
    from src import storage_engine

import streamlit as st


__all__ = [
    "render_error",
    "render_result",
    "load_documents",
    "build_payload",
    "run_app",
    "main",
]


logger = logging.getLogger("SafeDocAI.UI")


# ============================================================
# Configuration
# ============================================================

APP_TITLE = "SafeDocAI"
APP_TAGLINE = "Private. Local. Your Documents."

#: Answer latency for the local model is high on CPU; make that
#: explicit and honest in the loading state (Phase 9): a fast exact
#: query gets a simple "checking" status; a semantic query gets the
#: two-stage explanation (search = seconds, local generation = 1–2 min).
EXACT_LOADING_TEXT = "Checking your documents…"
SEMANTIC_LOADING_TEXT = (
    "Searching your documents… then generating the answer locally.  \n"
    "Searching takes seconds — generating with the local model can take "
    "1–2 minutes on CPU. Please keep this tab open."
)

#: Caption suffix appended to each answer with its measured total time.
ELAPSED_CAPTION = "answered in {seconds:.1f} s"

#: Lightweight deterministic retry limit for transient transport errors.
MAX_TRANSIENT_RETRIES = 2

#: Chat-style history kept in the session (bounded, in-memory only).
MAX_HISTORY_ENTRIES = 10


# ============================================================
# Document listing (read-only, reuses storage engine)
# ============================================================


def load_documents() -> list[dict[str, Any]]:
    """Read-only list of stored documents for the sidebar.

    Returns a list of dicts (id, file_name, file_type, status,
    upload_timestamp, fields) ordered by id. Returns [] when SQLite is
    unavailable; callers render a friendly banner instead of a trace.
    """

    try:
        connection = storage_engine.get_db_connection()
    except Exception as exc:  # noqa: BLE001 — degraded mode, never crash
        logger.error("SQLite unavailable while listing documents: %s", exc)
        st.sidebar.error("Database unavailable — cannot load documents.")
        return []

    try:
        rows = connection.execute(
            """
            SELECT id, file_name, file_type, file_path, upload_timestamp, status
            FROM documents
            ORDER BY id
            """
        ).fetchall()
    except Exception as exc:  # noqa: BLE001
        logger.error("Document query failed: %s", exc)
        st.sidebar.error("Could not read documents from the database.")
        return []
    finally:
        connection.close()

    return [
        {
            "id": int(row[0]),
            "file_name": str(row[1]),
            "file_type": str(row[2]),
            "file_path": str(row[3]),
            "upload_timestamp": str(row[4]),
            "status": str(row[5]),
        }
        for row in rows
    ]


# ============================================================
# answer_query() invocation + normalization
# ============================================================


def build_payload(query: str) -> dict[str, Any]:
    """Normalize any answer_query() outcome into a UI-friendly payload.

    Never raises and never fabricates: every failure mode (transport
    down, malformed result, unexpected shape) degrades to a payload
    with ok=False or insufficient=True and a friendly message.
    """

    started = time.perf_counter()

    try:
        result = answer_query(query)
    except ConnectionError:
        # llm_engine raises ConnectionError when Ollama is unreachable.
        logger.error("Ollama unreachable for query: %r", query)
        return _failure_payload(
            "The local AI service (Ollama) is not reachable. "
            "Make sure Ollama is running locally, then try again."
        )
    except Exception as exc:  # noqa: BLE001 — log full detail, show friendly text
        logger.exception("answer_query failed for query: %r", query)
        return _failure_payload(
            "Something went wrong while processing your question. "
            "Please try again."
        )

    total_ms = round((time.perf_counter() - started) * 1000, 3)

    if not isinstance(result, dict):
        logger.error("Malformed answer_query result (not a dict): %r", result)
        return _failure_payload(
            "Received an unexpected internal response. Please try again."
        )

    answer = result.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        logger.error("Malformed answer_query result (no answer): %r", result)
        return _failure_payload(
            "Received an unexpected internal response. Please try again."
        )

    sources = result.get("sources")
    if not isinstance(sources, list):
        sources = []

    normalized: dict[str, Any] = {
        "ok": True,
        "answer": answer,
        "sources": sources,
        "source_documents": result.get("source_documents", []),
        "retrieval_route": result.get("retrieval_route"),
        "classification": result.get("classification"),
        "fallback": result.get("fallback") or {"occurred": False, "reason": None},
        "grounded": bool(result.get("grounded")),
        "llm_called": bool(result.get("llm_called")),
        "insufficient": bool(result.get("insufficient")),
        "insufficient_reason": result.get("insufficient_reason"),
        "retrieved_chunks": result.get("retrieved_chunks"),
        "timings_ms": result.get("timings_ms") or {},
        "document_intent": None,
        "ui_total_ms": total_ms,
    }

    # Intent provenance (Phase 6) may live on the router result inside
    # exact payloads or as a top-level key; tolerate both shapes.
    if isinstance(result.get("document_intent"), dict):
        normalized["document_intent"] = result["document_intent"]

    return normalized


def _failure_payload(message: str) -> dict[str, Any]:
    return {
        "ok": False,
        "error_message": message,
        "answer": None,
        "sources": [],
        "source_documents": [],
        "retrieval_route": None,
        "classification": None,
        "fallback": {"occurred": False, "reason": None},
        "grounded": False,
        "llm_called": False,
        "insufficient": False,
        "insufficient_reason": None,
        "retrieved_chunks": None,
        "timings_ms": {},
        "document_intent": None,
        "ui_total_ms": None,
    }


# ============================================================
# Rendering helpers (pure-ish; session_state for the query box)
# ============================================================

_EXAMPLE_QUERIES = [
    "What is my roll number?",
    "What is the transaction ID in 4143027140.pdf?",
    "What is my phone number?",
    "Tell me the important information on my ID card.",
    "What does my railway ticket contain?",
]


def _route_label(payload: dict[str, Any]) -> str:
    route = payload.get("retrieval_route") or "unknown"
    label = {
        "exact": "Exact match (database field)",
        "semantic": "Semantic search (document content)",
    }.get(str(route), str(route))
    if payload.get("fallback", {}).get("occurred"):
        label += " · fallback"
    return label


def _fallback_note(payload: dict[str, Any]) -> str | None:
    """Human explanation when the router fell back from exact to semantic."""

    fallback = payload.get("fallback") or {}
    if not fallback.get("occurred"):
        return None

    route = str(payload.get("retrieval_route") or "")
    if route == "semantic" and payload.get("llm_called"):
        return (
            "Exact field match was not found, so SafeDocAI searched the "
            "relevant document content."
        )
    if route == "semantic":
        return (
            "Exact field match was not found, so SafeDocAI searched the "
            "relevant document content."
        )
    return "Fallback occurred during retrieval."


def _source_files(payload: dict[str, Any]) -> list[str]:
    """Unique source file names for the compact one-line Source view."""

    names: list[str] = []
    for source in payload.get("sources") or []:
        if isinstance(source, dict):
            name = source.get("file_name")
            if name and name not in names:
                names.append(str(name))
    return names


def render_result(payload: dict[str, Any]) -> None:
    """Render one normalized answer payload into the answer area.

    Phase 9: the main area shows ONLY the answer, a one-line Source,
    a grounded/elapsed status line and (when applicable) the fallback
    explainer. All technical provenance (document IDs, field names,
    chunk IDs, distances, route metadata) lives inside a collapsed
    "Details" expander. This is the single rendering path for all
    result kinds; it performs no retrieval and no LLM work.
    """

    if not payload.get("ok", False):
        st.error(payload.get("error_message") or "Something went wrong.")
        return

    answer = payload.get("answer") or ""

    # -- Insufficient context (never invent an answer in the UI) ----
    if payload.get("insufficient") or (
        answer.strip() == INSUFFICIENT_CONTEXT_MESSAGE
    ):
        st.info(
            "I couldn't find enough information in your stored documents "
            "to answer that."
        )
        if payload.get("insufficient_reason"):
            with st.expander("Details"):
                st.caption(f"Reason: {payload['insufficient_reason']}")
                _render_sources(payload, context="Available context")
        return

    # -- Grounded answer ------------------------------------------------
    st.markdown("### Answer")
    st.markdown(f"> {answer}")

    files = _source_files(payload)
    if files:
        st.markdown(f"**Source:** {', '.join(files)}")

    grounded_line = "Yes ✅" if payload.get("grounded") else "No ⚠️"
    total_ms = (payload.get("timings_ms") or {}).get("total")
    elapsed_line = (
        f"   ·   {ELAPSED_CAPTION.format(seconds=total_ms / 1000)}"
        if isinstance(total_ms, (int, float)) and total_ms
        else ""
    )
    st.caption(f"Grounded: {grounded_line}{elapsed_line}")

    note = _fallback_note(payload)
    if note:
        st.info(note)

    if files:
        with st.expander("Details (sources & provenance)"):
            st.caption(f"Route: {_route_label(payload)}")
            _render_sources(payload)


def _render_sources(payload: dict[str, Any], context: str = "Sources") -> None:
    """Render DETAILED provenance (lives inside the collapsed expander)."""

    sources = payload.get("sources") or []
    if not sources:
        return

    st.markdown(f"#### {context}")
    for source in sources:
        if not isinstance(source, dict):
            continue
        file_name = source.get("file_name") or "Unknown document"
        document_id = source.get("document_id")

        if "field_name" in source:
            # Exact-route source: field-level provenance.
            field = source.get("field_name") or "?"
            value = source.get("field_value")
            st.markdown(f"- **{file_name}** — field: `{field}`")
            if value is not None:
                st.markdown(f"  - value: `{value}`")
        else:
            # Semantic-route source: chunk-level provenance.
            parts = [f"**{file_name}**"]
            if document_id is not None:
                parts.append(f"document id: {document_id}")
            if source.get("chunk_id") is not None:
                parts.append(f"chunk: {source['chunk_id']}")
            if source.get("distance") is not None:
                try:
                    parts.append(f"distance: {float(source['distance']):.3f}")
                except (TypeError, ValueError):
                    pass
            st.markdown("- " + " · ".join(parts))

    retrieved = payload.get("retrieved_chunks")
    if retrieved:
        st.caption("Retrieved chunks (provenance detail):")
        for chunk in retrieved:
            if not isinstance(chunk, dict):
                continue
            st.caption(
                f"{chunk.get('file_name', '?')} · chunk {chunk.get('chunk_id', '?')}"
                f" · distance {chunk.get('distance', '?')}"
            )


# ============================================================
# Streamlit app
# ============================================================


def run_app() -> None:
    """Build the Streamlit page. Called by main() and the test harness."""

    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    _inject_css()

    # ---------------- Sidebar: documents ----------------
    with st.sidebar:
        st.header("📄 Documents")
        documents = load_documents()

        if documents:
            st.caption(f"{len(documents)} stored")
            for doc in documents:
                st.markdown(
                    f"**{doc['file_name']}**  \n"
                    f"<span style='color:var(--text-color-secondary);font-size:0.8rem'>"
                    f"id {doc['id']} · {doc['file_type']} · {doc['status']}"
                    f"</span>",
                    unsafe_allow_html=True,
                )
        else:
            st.info("No documents stored yet.")

        st.divider()
        st.caption(
            "🔒 **Private & local.** All processing happens on this machine. "
            "No document content leaves your computer."
        )

    # ---------------- Header ----------------
    st.title(APP_TITLE)
    st.caption(f"**{APP_TAGLINE}**  ·  offline · evidence-grounded answers")

    # ---------------- Ask area ----------------
    st.markdown("#### Ask SafeDocAI")

    default_query = st.session_state.get("pending_query", "")
    query = st.text_area(
        "Your question",
        value=default_query,
        height=90,
        placeholder="e.g. What is my roll number?",
        label_visibility="collapsed",
        key="query_input",
    )

    col1, col2, _ = st.columns([1, 1, 4])
    ask_clicked = col1.button("Ask", type="primary", use_container_width=True)
    clear_clicked = col2.button("Clear", use_container_width=True)

    if clear_clicked:
        for key in ("query_input", "pending_query", "history"):
            st.session_state.pop(key, None)
        st.rerun()

    if not documents:
        st.warning("No documents are stored yet. Run the pipeline first.")
        return

    if not ask_clicked:
        if not query and not default_query:
            _render_examples()
        _render_history()
        return

    # ---------------- Query handling ----------------
    if not query.strip():
        st.warning("Please enter a question before asking.")
        return

    # Deterministic transient retry for transport hiccups.
    payload: dict[str, Any] | None = None
    for attempt in range(1, MAX_TRANSIENT_RETRIES + 1):
        loading = SEMANTIC_LOADING_TEXT if _looks_semantic(query) else EXACT_LOADING_TEXT
        with st.spinner(loading):
            payload = build_payload(query)
        if payload.get("ok") or attempt == MAX_TRANSIENT_RETRIES:
            break

    # Chat-style history (most recent first); rendered on every run.
    history: list[dict[str, Any]] = st.session_state.setdefault("history", [])
    if _should_record_query(history, query):
        history.insert(0, {"query": query, "payload": payload})
        history[:] = history[:MAX_HISTORY_ENTRIES]

    st.session_state["pending_query"] = query
    st.rerun()


def _should_record_query(history: list[dict[str, Any]] | None, query: str) -> bool:
    """True when this query is new and belongs in the visible history.

    The identical query asked again in a row is NOT re-recorded: with a
    slow local model this prevents accidental duplicate LLM runs from
    spamming the history with identical entries.
    """

    if not history:
        return True
    return history[0].get("query") != query


def _render_history() -> None:
    """Render the stored Q/A history (chat-style, newest first)."""

    history: list[dict[str, Any]] = st.session_state.get("history") or []
    for index, entry in enumerate(history):
        if not isinstance(entry, dict) or "payload" not in entry:
            continue
        st.markdown(f"**Q: {entry.get('query', '')}**")
        render_result(entry["payload"])
        if index < len(history) - 1:
            st.divider()


def _looks_semantic(query: str) -> bool:
    """Cheap heuristic purely for the loading message (not routing).

    Phase 8: includes Hinglish exploratory cues ("batao", "ke baare",
    "likha") so mixed-language queries get the honest long-run message.
    """

    lowered = query.lower()
    return any(
        cue in lowered
        for cue in (
            "what does", "tell me", "explain", "summarize", "about",
            "batao", "bataiye", "bata ", "ke baare", "likha",
            "important details", "kya likha", "mujhe",
        )
    )


def _render_examples() -> None:
    st.caption("Try one of these:")
    columns = st.columns(min(3, len(_EXAMPLE_QUERIES)))
    for index, example in enumerate(_EXAMPLE_QUERIES):
        with columns[index % len(columns)]:
            if st.button(example, key=f"example_{index}"):
                st.session_state["pending_query"] = example
                st.rerun()


def _inject_css() -> None:
    """Dark, minimal, privacy-focused styling. No external assets."""

    st.markdown(
        """
        <style>
            .stApp { background: #0e1117; color: #e8eaed; }
            section[data-testid="stSidebar"] {
                background: #171b23; border-right: 1px solid #2a2f3a;
            }
            h1 { color: #f5f7fa; letter-spacing: 0.5px; }
            .stCaption, [data-testid="stCaptionContainer"] {
                color: #9aa3b2 !important;
            }
            h4 { color: #c9d1dc; margin-top: 1rem; }
            blockquote {
                border-left: 3px solid #4c8dff; padding: 0.4rem 1rem;
                background: rgba(76, 141, 255, 0.06);
            }
            div[data-testid="stButton"] > button {
                border-radius: 8px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    run_app()


if __name__ == "__main__":
    main()
