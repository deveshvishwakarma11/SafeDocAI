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
        self.assertIn("ID Card.pdf", harness.text_of("markdown"))
        self.assertIn("roll", harness.text_of("markdown"))
        self.assertIn("Grounded: Yes", harness.text_of("caption"))
        self.assertIn("Exact match", harness.text_of("caption"))

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
        self.assertIn("Local AI used: Yes", harness.text_of("caption"))
        self.assertIn("4143027140.pdf", harness.text_of("markdown"))
        self.assertIn("chunk 3", harness.text_of("caption"))

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


def main() -> int:
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for case in (TestBuildPayload, TestRenderResult, TestUIGuards, TestLoadDocuments, TestAskFlow):
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
