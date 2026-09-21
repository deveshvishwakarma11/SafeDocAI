"""
Phase 7 UI test suite (plain-Python runner, no pytest).
========================================================

Uses a mocked ``answer_query`` transport only inside this file — no mock
data ever enters the real database/vector store. A post-run guard
verifies the real ``data/safedoc.db`` was untouched (size + mtime).

Run:  PYTHONIOENCODING=utf-8 python tests/test_ui.py
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

REAL_DB = PROJECT_ROOT / "data" / "safedoc.db"
REAL_DB_SIGNATURE = (
    REAL_DB.stat().st_size,
    REAL_DB.stat().st_mtime_ns,
)

import ui  # noqa: E402
from ui import build_payload, load_documents, render_result  # noqa: E402


# ============================================================
# Mock answer_query results (shape-identical to the real payload)
# ============================================================

_BASE_RESULT = {
    "query": "q",
    "answer": "roll: 2407510100067",
    "sources": [
        {
            "document_id": 9,
            "file_name": "ID Card.pdf",
            "file_path": "data/samples/ID Card.pdf",
            "field_name": "roll",
            "field_value": "2407510100067",
        }
    ],
    "source_documents": [{"document_id": 9, "file_name": "ID Card.pdf"}],
    "retrieval_route": "exact",
    "fallback": {"occurred": False, "reason": None},
    "grounded": True,
    "llm_called": False,
    "insufficient": False,
    "insufficient_reason": None,
    "timings_ms": {"router": 1.0, "retrieval": None, "llm_generation": None, "total": 2.0},
}


def make_result(**overrides):
    return {**_BASE_RESULT, **overrides}


class _Stream:
    """Minimal recorder standing in for st.error/st.info/st.markdown/...

    Nested access (st.sidebar.error) shares one calls list so the
    harness sees every recorded call regardless of nesting depth.
    """

    def __init__(self, calls: list | None = None):
        self.calls: list[tuple[str, tuple, dict]] = (
            calls if calls is not None else []
        )

    def __call__(self, *args, **kwargs):
        self.calls.append((self._name, args, kwargs))
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __getattr__(self, name):
        proxy = _Stream(self.calls)  # share the recorder
        proxy._name = name
        return proxy

    _name = "?"


class _StreamlitHarness:
    """Patch ui.st with a recorder and expose what was rendered."""

    def __init__(self):
        self.recorded = _Stream()

    def __enter__(self):
        self._patcher = mock.patch.object(ui, "st", self.recorded)
        self._patcher.start()
        return self

    def __exit__(self, *exc):
        self._patcher.stop()

    def text_of(self, method: str) -> str:
        # "expander" is a context-manager marker in the flat recorder:
        # return everything recorded AFTER the first expander() call
        # (its collapsed content), regardless of the inner method.
        if method == "expander":
            calls = self.recorded.calls
            start = next(
                (i for i, (name, _a, _k) in enumerate(calls) if name == "expander"),
                None,
            )
            if start is None:
                return ""
            chunks = [
                args[0]
                for name, args, _kwargs in calls[start + 1 :]
                if args and isinstance(args[0], str)
            ]
            return "\n".join(chunks)
        chunks = []
        for name, args, _kwargs in self.recorded.calls:
            if name == method and args and isinstance(args[0], str):
                chunks.append(args[0])
        return "\n".join(chunks)

    def called(self, method: str) -> bool:
        return any(name == method for name, _a, _k in self.recorded.calls)


# ============================================================
# Tests
# ============================================================


class TestBuildPayload(unittest.TestCase):
    """build_payload() normalization over the full failure matrix."""

    def _run(self, result=None, side_effect=None):
        with mock.patch.object(ui, "answer_query", side_effect=side_effect, return_value=result):
            return build_payload("What is my roll number?")

    def test_01_exact_result_normalized(self):
        payload = self._run(result=make_result())
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["answer"], "roll: 2407510100067")
        self.assertEqual(payload["retrieval_route"], "exact")
        self.assertTrue(payload["grounded"])
        self.assertFalse(payload["llm_called"])
        self.assertIn("ui_total_ms", payload)

    def test_02_malformed_result_not_a_dict(self):
        payload = self._run(result="not a dict")
        self.assertFalse(payload["ok"])
        self.assertIn("unexpected internal response", payload["error_message"])

    def test_03_malformed_result_missing_answer(self):
        payload = self._run(result={"sources": []})
        self.assertFalse(payload["ok"])
        self.assertIn("unexpected internal response", payload["error_message"])

    def test_04_empty_answer_rejected(self):
        payload = self._run(result=make_result(answer="   "))
        self.assertFalse(payload["ok"])

    def test_05_ollama_unreachable_connection_error(self):
        payload = self._run(side_effect=ConnectionError("refused"))
        self.assertFalse(payload["ok"])
        self.assertIn("Ollama", payload["error_message"])
        self.assertNotIn("Traceback", payload["error_message"])

    def test_06_generic_exception_handled(self):
        payload = self._run(side_effect=RuntimeError("boom"))
        self.assertFalse(payload["ok"])
        self.assertIn("Something went wrong", payload["error_message"])
        self.assertNotIn("boom", payload["error_message"])  # no internals leaked

    def test_07_sources_none_tolerated(self):
        payload = self._run(result=make_result(sources=None))
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["sources"], [])

    def test_08_fallback_metadata_preserved(self):
        payload = self._run(
            result=make_result(
                retrieval_route="semantic",
                fallback={"occurred": True, "reason": "no exact match"},
            )
        )
        self.assertTrue(payload["fallback"]["occurred"])
        self.assertEqual(payload["fallback"]["reason"], "no exact match")

    def test_09_insufficient_result_normalized(self):
        payload = self._run(
            result=make_result(
                answer="I could not find enough information in your stored documents to answer this.",
                grounded=False,
                insufficient=True,
                sources=[],
                insufficient_reason="no useful chunks",
            )
        )
        self.assertTrue(payload["insufficient"])
        self.assertEqual(payload["insufficient_reason"], "no useful chunks")


class TestRenderResult(unittest.TestCase):
    """render_result() display paths for every result kind."""

    def test_10_exact_display(self):
        harness = _StreamlitHarness()
        with harness, mock.patch.object(ui, "answer_query", return_value=make_result()):
            render_result(build_payload("What is my roll number?"))
        self.assertIn("2407510100067", harness.text_of("markdown"))
        # Phase 9: compact main view — file name in the Source line, the
        # detailed field/value provenance hidden in the expander.
        self.assertIn("ID Card.pdf", harness.text_of("markdown"))
        self.assertIn("roll", harness.text_of("expander"))
        self.assertIn("Grounded: Yes", harness.text_of("caption"))
        self.assertIn("Exact match", harness.text_of("expander"))

    def test_11_semantic_display_shows_chunks_and_llm(self):
        result = make_result(
            answer="The ticket shows refund rules.",
            retrieval_route="semantic",
            llm_called=True,
            sources=[{"document_id": 7, "file_name": "4143027140.pdf"}],
            retrieved_chunks=[
                {"document_id": 7, "file_name": "4143027140.pdf", "chunk_id": 3, "distance": 0.95}
            ],
        )
        harness = _StreamlitHarness()
        with harness, mock.patch.object(ui, "answer_query", return_value=result):
            render_result(build_payload("q"))
        self.assertIn("Grounded: Yes", harness.text_of("caption"))
        # Phase 9: chunk internals live in the collapsed Details expander.
        self.assertIn("chunk 3", harness.text_of("expander"))
        self.assertIn("4143027140.pdf", harness.text_of("markdown"))

    def test_12_fallback_display_is_human_readable(self):
        result = make_result(
            answer="M103B71",
            retrieval_route="semantic",
            llm_called=True,
            fallback={"occurred": True, "reason": "exact miss"},
            sources=[{"document_id": 8, "file_name": "ApplicationForm.pdf"}],
        )
        harness = _StreamlitHarness()
        with harness, mock.patch.object(ui, "answer_query", return_value=result):
            render_result(build_payload("q"))
        info = harness.text_of("info")
        self.assertIn("Exact field match was not found", info)
        self.assertIn("searched the relevant document content", info)

    def test_13_insufficient_display(self):
        result = make_result(
            answer="I could not find enough information in your stored documents to answer this.",
            grounded=False,
            insufficient=True,
            sources=[],
        )
        harness = _StreamlitHarness()
        with harness, mock.patch.object(ui, "answer_query", return_value=result):
            render_result(build_payload("q"))
        info = harness.text_of("info")
        self.assertIn("couldn't find enough information", info)

    def test_14_error_display_no_stack_trace(self):
        harness = _StreamlitHarness()
        with harness:
            render_result(
                {
                    "ok": False,
                    "error_message": "The local AI service (Ollama) is not reachable.",
                    "sources": [],
                    "answer": None,
                }
            )
        err = harness.text_of("error")
        self.assertIn("Ollama", err)
        self.assertNotIn("Traceback", err)


class TestUIGuards(unittest.TestCase):
    """Structural/privacy guarantees."""

    def test_15_module_imports(self):
        self.assertTrue(hasattr(ui, "run_app"))
        self.assertTrue(hasattr(ui, "main"))

    def test_16_entry_point_exists(self):
        entry = PROJECT_ROOT / "run_ui.py"
        self.assertTrue(entry.exists(), "run_ui.py entry point missing")
        self.assertIn("streamlit", entry.read_text(encoding="utf-8").lower())

    def test_17_no_duplicate_retrieval_logic(self):
        """The UI must not embed its own SQL/Chroma/LLM/grounding code."""
        source = (PROJECT_ROOT / "src" / "ui.py").read_text(encoding="utf-8")
        banned = [
            ("chroma query", "query_embeddings"),
            ("chroma collection access", "get_or_create_collection"),
            ("sqlite metadata select", "FROM extracted_metadata"),
            ("llm prompt construction", "num_predict"),
            ("grounding validation", "validate_grounding"),
            ("semantic json contract", '"s": ['),
        ]
        for label, needle in banned:
            self.assertNotIn(needle, source, f"UI must not implement {label}")

    def test_18_ui_delegates_to_answer_query(self):
        self.assertIn("answer_query", (PROJECT_ROOT / "src" / "ui.py").read_text(encoding="utf-8"))


class TestLoadDocuments(unittest.TestCase):
    """Document-listing path (real DB read-only, no writes)."""

    def test_19_real_documents_listed_readonly(self):
        docs = load_documents()
        # The real store has documents; listing must never mutate anything.
        self.assertIsInstance(docs, list)
        for doc in docs:
            self.assertIn("id", doc)
            self.assertIn("file_name", doc)

    def test_20_empty_document_list_handled(self):
        harness = _StreamlitHarness()
        with harness, mock.patch.object(ui.storage_engine, "get_db_connection", side_effect=RuntimeError("no db")):
            docs = load_documents()
        self.assertEqual(docs, [])
        self.assertTrue(harness.called("error"))


# ============================================================
# Runner + real-DB safety guard
# ============================================================


class TestAskFlow(unittest.TestCase):
    """Query-history behavior for repeated identical asks."""

    def test_21_identical_consecutive_ask_not_duplicated(self):
        """Re-asking the exact same committed query must not double history."""
        history = [{"query": "What is my roll number?", "payload": {"ok": True}}]
        self.assertFalse(
            ui._should_record_query(history, "What is my roll number?"),
            "identical consecutive query must not be re-recorded",
        )

    def test_22_new_query_after_previous_is_recorded(self):
        history = [{"query": "What is my roll number?", "payload": {"ok": True}}]
        self.assertTrue(ui._should_record_query(history, "What is my phone number?"))

    def test_23_empty_history_records_any_query(self):
        self.assertTrue(ui._should_record_query([], "What is my roll number?"))
        self.assertTrue(ui._should_record_query(None, "What is my roll number?"))


class TestOllamaUnavailableHandling(unittest.TestCase):
    """Ollama-down must read as 'service not reachable', never as 'not found'."""

    def test_24_llm_unavailable_error_shows_ollama_message(self):
        """The semantic path raises LLMUnavailableError when transport is down."""
        import answer_engine as ae

        harness = _StreamlitHarness()
        with harness, mock.patch.object(
            ui, "answer_query", side_effect=ae.LLMUnavailableError("down")
        ):
            payload = build_payload("What does my railway ticket contain?")
        self.assertFalse(payload["ok"])
        self.assertIn("Ollama", payload["error_message"])
        self.assertIn("not reachable", payload["error_message"])

    def test_25_connection_error_shows_ollama_message(self):
        harness = _StreamlitHarness()
        with harness, mock.patch.object(
            ui, "answer_query", side_effect=ConnectionError("refused")
        ):
            payload = build_payload("What is my roll number?")
        self.assertFalse(payload["ok"])
        self.assertIn("Ollama", payload["error_message"])

    def test_26_transport_down_is_not_reported_as_insufficient(self):
        """The distinguishing regression: service-down != insufficient-context."""
        import answer_engine as ae

        harness = _StreamlitHarness()
        with harness, mock.patch.object(
            ui, "answer_query", side_effect=ae.LLMUnavailableError("down")
        ):
            payload = build_payload("What does my railway ticket contain?")
        self.assertFalse(payload.get("insufficient"))
        self.assertIsNone(payload.get("answer"))

    def test_27_hinglish_summary_loading_message(self):
        """Hinglish summary phrasings get the honest long-run loading state."""
        self.assertTrue(ui._looks_semantic("Meri railway ticket mein kya details hain?"))
        self.assertTrue(ui._looks_semantic("ID card ke important details batao"))
        self.assertTrue(ui._looks_semantic("Meri application form ki important details batao"))
        # Fact-ish exact queries keep the quick message.
        self.assertFalse(ui._looks_semantic("Mera roll number kya hai?"))


class TestChatFlow(unittest.TestCase):
    """Phase 10 chat state machine: submit → user bubble → answer below.

    Covers the required interaction cases (first query, second query,
    consecutive queries, duplicate submission, empty query, fast exact
    vs slow semantic loading state, Ollama unavailable, previous
    messages stay visible) at the state-management layer.
    """

    # -- submission state machine (pure helpers, no Streamlit run) ----

    def test_28_first_query_appends_pending_user_message(self):
        history: list = []
        action, entry = ui._prepare_submission(history, "What is my roll number?")
        self.assertEqual(action, "accepted")
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["query"], "What is my roll number?")
        self.assertIsNone(history[0]["payload"])  # answer pending
        self.assertIs(entry, history[0])

    def test_29_empty_query_rejected_without_recording(self):
        history: list = []
        for raw in ("", "   ", None):
            action, entry = ui._prepare_submission(history, raw)
            self.assertEqual(action, "empty")
            self.assertIsNone(entry)
        self.assertEqual(history, [])

    def test_30_consecutive_identical_query_blocked(self):
        history = [
            {"query": "What is my roll number?", "payload": {"ok": True}},
        ]
        action, entry = ui._prepare_submission(history, "What is my roll number?")
        self.assertEqual(action, "duplicate")
        self.assertIsNone(entry)
        self.assertEqual(len(history), 1)  # history untouched

    def test_31_second_different_query_appends_after_previous(self):
        history = [
            {"query": "What is my roll number?", "payload": {"ok": True}},
        ]
        action, _ = ui._prepare_submission(history, "What is my phone number?")
        self.assertEqual(action, "accepted")
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["query"], "What is my roll number?")
        self.assertEqual(history[1]["query"], "What is my phone number?")
        self.assertEqual(
            [e["query"] for e in history],
            ["What is my roll number?", "What is my phone number?"],
        )  # oldest-first, nothing replaced

    def test_32_history_stays_bounded_oldest_dropped(self):
        history: list = []
        for i in range(ui.MAX_HISTORY_ENTRIES + 2):
            action, _ = ui._prepare_submission(history, f"question {i}")
            self.assertEqual(action, "accepted")
        self.assertEqual(len(history), ui.MAX_HISTORY_ENTRIES)
        self.assertEqual(history[0]["query"], "question 2")
        self.assertEqual(history[-1]["query"], f"question {ui.MAX_HISTORY_ENTRIES + 1}")

    # -- processing the pending query (spinner + payload attachment) --

    def _answer(self, history, pending, **answer_query_kwargs):
        harness = _StreamlitHarness()
        with harness, mock.patch.object(ui, "answer_query", **answer_query_kwargs):
            ui._answer_pending(history, pending)
        return harness

    def test_33_fast_exact_query_attaches_payload_exact_loading(self):
        history = [{"query": "What is my roll number?", "payload": None}]
        harness = self._answer(
            history, "What is my roll number?", return_value=make_result()
        )
        entry = history[0]
        self.assertTrue(entry["payload"]["ok"])
        self.assertIn("2407510100067", entry["payload"]["answer"])
        self.assertIn("Checking your documents", harness.text_of("spinner"))

    def test_34_slow_semantic_query_gets_honest_loading_message(self):
        query = "Tell me the important information on my ID card."
        history = [{"query": query, "payload": None}]
        harness = self._answer(history, query, return_value=make_result())
        self.assertIn("Searching your documents", harness.text_of("spinner"))
        self.assertTrue(history[0]["payload"]["ok"])

    def test_35_ollama_unavailable_becomes_assistant_error_bubble(self):
        history = [{"query": "What does my railway ticket contain?", "payload": None}]
        harness = self._answer(
            history,
            "What does my railway ticket contain?",
            side_effect=ConnectionError("refused"),
        )
        payload = history[0]["payload"]
        self.assertFalse(payload["ok"])
        self.assertIn("Ollama", payload["error_message"])
        # The user message stays visible above the failed answer.
        self.assertEqual(history[0]["query"], "What does my railway ticket contain?")

    def test_35b_pending_without_entry_creates_its_own_user_message(self):
        history: list = []
        self._answer(history, "What is my roll number?", return_value=make_result())
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["query"], "What is my roll number?")
        self.assertTrue(history[0]["payload"]["ok"])

    # -- conversation rendering (previous messages stay visible) ------

    def test_36_conversation_shows_user_and_assistant_messages(self):
        with mock.patch.object(ui, "answer_query", return_value=make_result()):
            payload = build_payload("What is my roll number?")
        history = [{"query": "What is my roll number?", "payload": payload}]
        harness = _StreamlitHarness()
        with harness:
            ui._render_conversation(history)
        self.assertTrue(harness.called("chat_message"))
        markdown = harness.text_of("markdown")
        self.assertIn("What is my roll number?", markdown)  # user bubble
        self.assertIn("2407510100067", markdown)  # assistant answer below
        # Details (sources & provenance) expander preserved.
        self.assertIn("roll", harness.text_of("expander"))

    def test_37_pending_turn_renders_user_bubble_with_waiting_state(self):
        history = [{"query": "What is my phone number?", "payload": None}]
        harness = _StreamlitHarness()
        with harness:
            ui._render_conversation(history)
        markdown = harness.text_of("markdown")
        self.assertIn("What is my phone number?", markdown)  # user bubble first
        self.assertNotIn("### Answer", markdown)  # no answer yet
        self.assertIn("Searching your documents", harness.text_of("caption"))

    def test_38_multiple_turns_render_in_order(self):
        with mock.patch.object(ui, "answer_query", return_value=make_result()):
            first = build_payload("What is my roll number?")
        history = [
            {"query": "What is my roll number?", "payload": first},
            {"query": "What is my phone number?", "payload": None},
        ]
        harness = _StreamlitHarness()
        with harness:
            ui._render_conversation(history)
        markdown = harness.text_of("markdown")
        self.assertLess(
            markdown.index("What is my roll number?"),
            markdown.index("What is my phone number?"),
        )  # oldest turn first, newest last

    # -- structural guards for the composer rework ---------------------

    def test_39_single_chat_submission_path(self):
        """Enter key and the send arrow share one code path (st.chat_input);
        the old text_area + separate Ask-button flow is gone."""
        source = (PROJECT_ROOT / "src" / "ui.py").read_text(encoding="utf-8")
        self.assertIn("st.chat_input(", source)
        self.assertIn("st.chat_message(", source)
        self.assertNotIn("st.text_area(", source)
        self.assertNotIn('button("Ask"', source)
        self.assertEqual(source.count("st.chat_input("), 1)

    def test_40_submitted_query_never_refills_the_input(self):
        """Regression: the old flow wrote the query back into the widget
        state (pending_query = query) so text stayed in the box."""
        source = (PROJECT_ROOT / "src" / "ui.py").read_text(encoding="utf-8")
        self.assertNotIn('"pending_query"] = query', source)
        # Both the composer and the example buttons route through the
        # one shared submission helper.
        self.assertGreaterEqual(source.count("_prepare_submission("), 3)


class TestConversationRendering(unittest.TestCase):
    """Phase 10.1: conversation + scope replies render naturally."""

    def test_41_conversation_renders_without_grounded_line(self):
        result = make_result(
            answer="Hello! I'm SafeDocAI — ask me anything about the documents "
            "stored on this machine.",
            classification="conversation",
            retrieval_route="none",
            grounded=True,
            llm_called=False,
            sources=[],
            source_documents=[],
        )
        harness = _StreamlitHarness()
        with harness, mock.patch.object(ui, "answer_query", return_value=result):
            render_result(build_payload("hello"))
        self.assertIn("SafeDocAI", harness.text_of("markdown"))
        # Ordinary assistant message: no Grounded line, no provenance
        # expander, no error — nothing was retrieved or validated.
        self.assertNotIn("Grounded:", harness.text_of("caption"))
        self.assertFalse(harness.called("expander"))
        self.assertFalse(harness.called("error"))

    def test_42_conversation_payload_normalized(self):
        result = make_result(
            classification="conversation",
            retrieval_route="none",
            sources=[],
            source_documents=[],
        )
        with mock.patch.object(ui, "answer_query", return_value=result):
            payload = build_payload("hi")
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["classification"], "conversation")
        self.assertFalse(payload["insufficient"])
        self.assertFalse(payload["scope_reply"])

    def test_43_scope_reply_shown_as_info_without_provenance(self):
        result = make_result(
            answer=ui.OUT_OF_SCOPE_MESSAGE,
            classification="rejected",
            retrieval_route="none",
            grounded=False,
            llm_called=False,
            sources=[],
            source_documents=[],
            scope_reply=True,
        )
        harness = _StreamlitHarness()
        with harness, mock.patch.object(ui, "answer_query", return_value=result):
            render_result(build_payload("weather today"))
        info = harness.text_of("info")
        self.assertIn("documents stored on this machine", info)
        self.assertIn("out of", info)
        self.assertNotIn("Grounded:", harness.text_of("caption"))
        self.assertFalse(harness.called("expander"))


def main() -> int:
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for case in (
        TestBuildPayload,
        TestRenderResult,
        TestUIGuards,
        TestLoadDocuments,
        TestAskFlow,
        TestOllamaUnavailableHandling,
        TestChatFlow,
        TestConversationRendering,
    ):
        suite.addTests(loader.loadTestsFromTestCase(case))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    if (REAL_DB.stat().st_size, REAL_DB.stat().st_mtime_ns) != REAL_DB_SIGNATURE:
        print("SAFETY FAILURE: real data/safedoc.db was modified during UI tests!")
        return 1

    passed = result.testsRun - len(result.failures) - len(result.errors)
    print(f"\nReal data/safedoc.db untouched: OK")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
