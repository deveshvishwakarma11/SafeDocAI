"""
Phase 7: User Interface Layer (Streamlit).
==========================================

A lightweight, 100%-local web UI for SafeDocAI. Renders documents from
SQLite and answers natural-language questions by delegating EVERYTHING
retrieval/LLM-related to :func:`answer_engine.answer_query` — the UI
implements no SQL, no Chroma search, no LLM prompting, no grounding
validation of its own.

Phase 10: the ask area is a real chat conversation. Every submission
(Enter key or the composer's send arrow — one shared code path) becomes
a visible USER message immediately and clears the composer; the answer
is appended below it as an assistant bubble. Previous turns stay
visible for the whole UI session and new queries append instead of
replacing anything.

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
    from answer_engine import (
        INSUFFICIENT_CONTEXT_MESSAGE,
        LLMUnavailableError,
        answer_query,
    )
except ImportError:  # project root on sys.path
    from src.answer_engine import (
        INSUFFICIENT_CONTEXT_MESSAGE,
        LLMUnavailableError,
        answer_query,
    )
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
    except (ConnectionError, LLMUnavailableError):
        # The local model transport is unreachable (llm_engine raises
        # ConnectionError via health paths; the semantic generation path
        # raises LLMUnavailableError after exhausting transport retries).
        # A service-down query must NOT read as "not found in documents".
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
        if st.button("🗑️ Clear conversation", use_container_width=True):
            for key in (
                "chat_history", "history",  # current + legacy history keys
                "pending_query", "query_input", "_scroll_pending",
            ):
                st.session_state.pop(key, None)
            st.rerun()

    # ---------------- Header ----------------
    st.title(APP_TITLE)
    st.caption(f"**{APP_TAGLINE}**  ·  offline · evidence-grounded answers")

    # ---------------- Ask area (chat) ----------------
    st.markdown("#### Ask SafeDocAI")

    history: list[dict[str, Any]] = st.session_state.get("chat_history")
    if not isinstance(history, list):
        history = []
        st.session_state["chat_history"] = history

    if not documents:
        st.warning("No documents are stored yet. Run the pipeline first.")

    # -- 1) Conversation (oldest → newest), then examples when empty ----
    # A just-submitted query is already in the history with payload=None,
    # so its user bubble (with a searching placeholder) is visible for
    # the whole processing run — the chat never loses the question.
    _render_conversation(history)
    if not history and documents:
        _render_examples()

    # -- 2) Scroll toward the newest message when one was just added ----
    if st.session_state.pop("_scroll_pending", False):
        _scroll_to_bottom()

    # -- 3) Answer a query submitted on the previous run ----------------
    # Processed exactly once, below the conversation so the spinner and
    # the pending user bubble are both visible; then rerun so the
    # finished answer renders inside the conversation above the composer
    # (which stays hidden while busy — a natural duplicate-submit guard).
    pending = str(st.session_state.pop("pending_query", "") or "").strip()
    if pending and documents:
        _answer_pending(history, pending)
        st.session_state["_scroll_pending"] = True
        st.rerun()
    elif pending:
        # Composer is disabled without documents; drop any orphaned
        # request and repaint the cleaned conversation.
        while history and history[-1].get("payload") is None:
            history.pop()
        st.rerun()

    # -- 4) Composer -----------------------------------------------------
    # st.chat_input auto-clears after submission; Enter and the send
    # arrow both land here, so there is exactly ONE submission path.
    prompt = st.chat_input("e.g. What is my roll number?", disabled=not documents)
    if prompt is None:
        return

    action, entry = _prepare_submission(history, prompt)
    if action == "empty":
        st.warning("Please enter a question before asking.")
        return
    if action == "duplicate":
        st.info("You just asked that — the answer is above.")
        return
    st.session_state["pending_query"] = (entry or {}).get("query", "")
    st.session_state["_scroll_pending"] = True
    st.rerun()


def _should_record_query(history: list[dict[str, Any]] | None, query: str) -> bool:
    """True when this query is new and belongs at the end of the history.

    History is oldest-first, so only the LAST entry matters: re-asking
    the identical query immediately after it was answered is treated as
    an accidental duplicate (with a slow local model, a double-Enter
    would otherwise spam identical LLM runs) and is NOT recorded.
    """

    if not history:
        return True
    last = history[-1]
    if not isinstance(last, dict):
        return True
    return last.get("query") != query


def _prepare_submission(
    history: list[dict[str, Any]], raw_query: str
) -> tuple[str, dict[str, Any] | None]:
    """Validate + record a submitted query as a pending USER message.

    Pure state-transition helper for the chat composer (no Streamlit
    rendering, no engine calls). Returns one of:

    * ``("empty", None)``     — blank/whitespace query; nothing recorded
    * ``("duplicate", None)`` — identical to the just-answered query
    * ``("accepted", entry)`` — ``entry`` appended to ``history`` with
      ``payload=None`` (assistant answer still pending)

    The oldest-first history stays bounded by MAX_HISTORY_ENTRIES.
    """

    query = (raw_query or "").strip()
    if not query:
        return "empty", None
    if not _should_record_query(history, query):
        return "duplicate", None
    entry: dict[str, Any] = {"query": query, "payload": None}
    history.append(entry)
    del history[:-MAX_HISTORY_ENTRIES]
    return "accepted", entry


def _answer_pending(history: list[dict[str, Any]], pending: str) -> None:
    """Process ``pending`` and attach the assistant payload to its entry.

    Delegates to :func:`build_payload` (and therefore answer_query)
    unchanged, with the same spinner and transient-retry behavior as
    before. Never raises: every failure mode (including Ollama being
    unreachable) degrades to a friendly payload that renders as the
    assistant bubble below the user's question.
    """

    entry: dict[str, Any] | None = None
    for candidate in reversed(history):
        if (
            isinstance(candidate, dict)
            and candidate.get("query") == pending
            and candidate.get("payload") is None
        ):
            entry = candidate
            break
    if entry is None:
        entry = {"query": pending, "payload": None}
        history.append(entry)
        del history[:-MAX_HISTORY_ENTRIES]

    # Deterministic transient retry for transport hiccups (unchanged).
    loading = SEMANTIC_LOADING_TEXT if _looks_semantic(pending) else EXACT_LOADING_TEXT
    payload: dict[str, Any] | None = None
    for attempt in range(1, MAX_TRANSIENT_RETRIES + 1):
        with st.spinner(loading):
            payload = build_payload(pending)
        if payload.get("ok") or attempt == MAX_TRANSIENT_RETRIES:
            break
    entry["payload"] = payload


def _render_conversation(history: list[dict[str, Any]]) -> None:
    """Render the whole conversation, oldest → newest, as chat bubbles.

    Each recorded turn shows the user's question and, once available,
    the assistant answer via :func:`render_result` — the single
    rendering path that keeps the grounded-answer view and the Details
    (sources & provenance) expander identical to previous phases.
    """

    for entry in history:
        if not isinstance(entry, dict) or "query" not in entry:
            continue
        with st.chat_message("user", avatar="🧑"):
            st.markdown(str(entry.get("query", "")))
        with st.chat_message("assistant", avatar="📄"):
            payload = entry.get("payload")
            if payload is None:
                st.caption("Searching your documents…")
            else:
                render_result(payload)


def _scroll_to_bottom() -> None:
    """Best-effort scroll toward the newest message (inline JS only).

    Implemented as a zero-height inline component so the app stays
    100% local — no external assets are loaded. Every failure mode is
    silently ignored: this is cosmetic, never load-bearing.
    """

    try:
        from streamlit.components.v1 import html as _components_html

        _components_html(
            "<script>(()=>{try{"
            "const d=window.parent.document;"
            "const main=d.querySelector('section.main')"
            "||d.querySelector('[data-testid=\"stMain\"]')"
            "||d.querySelector('[data-testid=\"stAppViewContainer\"]');"
            "if(main){main.scrollTop=main.scrollHeight;}"
            "window.parent.scrollTo(0,window.parent.document.body.scrollHeight);"
            "}catch(e){}})();</script>",
            height=0,
        )
    except Exception:  # noqa: BLE001 — cosmetic only, never break the app
        logger.debug("Auto-scroll skipped", exc_info=True)


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
            "important details", "kya likha", "kya details", "mujhe",
        )
    )


def _render_examples() -> None:
    st.caption("Try one of these:")
    columns = st.columns(min(3, len(_EXAMPLE_QUERIES)))
    for index, example in enumerate(_EXAMPLE_QUERIES):
        with columns[index % len(columns)]:
            if st.button(example, key=f"example_{index}"):
                # Examples go through the SAME submission path as the
                # composer (record user message → pending → process).
                history = st.session_state.setdefault("chat_history", [])
                action, entry = _prepare_submission(history, example)
                if action == "accepted" and entry is not None:
                    st.session_state["pending_query"] = entry["query"]
                    st.session_state["_scroll_pending"] = True
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
            [data-testid="stChatMessage"] {
                background: rgba(23, 27, 35, 0.75);
                border: 1px solid #2a2f3a;
                border-radius: 10px;
            }
            [data-testid="stChatInput"] textarea {
                background: #171b23;
                color: #e8eaed;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    run_app()


if __name__ == "__main__":
    main()
