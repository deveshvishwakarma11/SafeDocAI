"""
Phase 9.1: Summary-query retrieval/context/prompt policy test suite.
=====================================================================

Verifies the isolated summary policy WITHOUT touching real data:

* detection     - summary detection unchanged, fact queries unaffected;
* retrieval     - summary requests widen top_k to SUMMARY_TOP_K while
                  document-intent restriction still holds; explicit
                  caller top_k wins; fact queries keep top_k=3;
* context       - near-duplicate chunks are dropped, coverage is
                  round-robined across documents, budget bounded;
* prompt        - summary prompt carries the strict context-only rules;
* grounding     - validator untouched: unsupported facts still rejected;
* payload       - provenance and route unchanged; gate rejects unchanged.

LLM transport is mocked at the ``generate_with_meta`` seam; storage is
redirected into a temp store by tests/_harness.py.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for entry in (str(PROJECT_ROOT), str(PROJECT_ROOT / "src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from _harness import redirect_storage_to_temp  # noqa: E402

_TEMP_DIR = redirect_storage_to_temp("safedocai-summary-tests-")

import answer_engine  # noqa: E402
from answer_engine import (  # noqa: E402
    MAX_CONTEXT_CHARS,
    SUMMARY_DEDUPE_JACCARD,
    SUMMARY_MAX_CONTEXT_CHARS,
    SUMMARY_TOP_K,
    build_semantic_prompt,
    is_summary_request,
    validate_grounding,
)

_REAL_DB = PROJECT_ROOT / "data" / "safedoc.db"


def _chunk(doc_id: int, file_name: str, chunk_id: int, text: str, distance: float = 0.9) -> dict:
    return {
        "source": "chroma.safedoc_documents",
        "document_id": doc_id,
        "file_name": file_name,
        "chunk_id": chunk_id,
        "text": text,
        "distance": distance,
        "useful": True,
    }


# ===========================================================================
# 1. Detection
# ===========================================================================


class TestSummaryDetection(unittest.TestCase):
    def test_summary_detection_unchanged(self) -> None:
        self.assertTrue(is_summary_request("What does my railway ticket contain?"))
        self.assertTrue(is_summary_request("Meri application form ki important details batao"))
        self.assertTrue(is_summary_request("ID card ke important details batao"))

    def test_fact_queries_not_summary(self) -> None:
        self.assertFalse(is_summary_request("What is my roll number?"))
        self.assertFalse(is_summary_request("Mera phone number kya hai?"))
        self.assertFalse(is_summary_request("What is the transaction ID in 4143027140.pdf?"))


# ===========================================================================
# 2-5. Context assembly: dedupe, coverage, budget
# ===========================================================================


class TestContextAssembly(unittest.TestCase):
    def test_near_duplicates_dropped(self) -> None:
        base = "Enrollment ID M103B71 name DEVESH VISHWAKARMA dob 06/06/2006"
        # >80% token overlap (same content, two extra tokens) -> duplicate.
        near = base + " x y"
        chunks = [
            _chunk(8, "ApplicationForm.pdf", 0, base),
            _chunk(8, "ApplicationForm.pdf", 1, near),
            _chunk(8, "ApplicationForm.pdf", 2, "Refund policy terms and legal liability text."),
        ]
        result = answer_engine._diverse_chunks(chunks)
        self.assertEqual([c["chunk_id"] for c in result], [0, 2])

    def test_documents_round_robined(self) -> None:
        chunks = [
            _chunk(7, "ticket.pdf", 0, "alpha text one"),
            _chunk(7, "ticket.pdf", 1, "alpha text two"),
            _chunk(8, "form.pdf", 0, "beta text one"),
        ]
        result = answer_engine._diverse_chunks(chunks)
        self.assertEqual([c["document_id"] for c in result], [7, 8, 7])

    def test_identical_text_jaccard_boundary(self) -> None:
        self.assertTrue(
            answer_engine._is_near_duplicate(
                frozenset({"a", "b", "c"}), frozenset({"a", "b", "c"})
            )
        )
        self.assertFalse(
            answer_engine._is_near_duplicate(
                frozenset({"a", "b", "c"}), frozenset({"d", "e", "f"})
            )
        )

    def test_dedupe_threshold_constant(self) -> None:
        self.assertEqual(SUMMARY_DEDUPE_JACCARD, 0.8)

    def test_provenance_preserved_through_diversity(self) -> None:
        chunks = [
            _chunk(7, "ticket.pdf", 0, "one two three four five"),
            _chunk(8, "form.pdf", 0, "six seven eight nine ten"),
        ]
        result = answer_engine._diverse_chunks(chunks)
        for original, processed in zip(chunks, result):
            for key in ("document_id", "file_name", "chunk_id", "text", "distance"):
                self.assertEqual(original[key], processed[key])


# ===========================================================================
# 6. Summary retrieval policy (intent preserved, top_k widened)
# ===========================================================================


class TestSummaryRetrievalPolicy(unittest.TestCase):
    def test_summary_query_uses_wider_top_k(self) -> None:
        """Summary route_query call uses SUMMARY_TOP_K when caller left default."""

        captured: dict = {}

        def fake_route(query, top_k=3, **kwargs):
            captured["top_k"] = top_k
            return {
                "route": "semantic",
                "classification": "semantic",
                "query": query,
                "results": [_chunk(8, "ApplicationForm.pdf", 0, "Enrollment ID M103B71 fact.")],
                "fallback": {"occurred": False, "reason": None},
                "timings_ms": {"retrieval": 1.0},
                "insufficient": False,
            }

        with mock.patch.object(answer_engine, "route_query", fake_route), \
                mock.patch.object(answer_engine, "_generate_semantic_answer") as gen:
            gen.return_value = {
                "answer": "Enrollment ID M103B71 fact.",
                "sources": [{"document_id": 8, "file_name": "ApplicationForm.pdf"}],
                "model_confidence": "high",
                "grounding": {"grounded": True},
                "unknown_sources": [],
            }
            payload = answer_engine.answer_query(
                "Meri application form ki important details batao"
            )

        self.assertEqual(captured["top_k"], SUMMARY_TOP_K)
        self.assertTrue(payload["grounded"])

    def test_fact_query_keeps_default_top_k(self) -> None:
        captured: dict = {}

        def fake_route(query, top_k=3, **kwargs):
            captured["top_k"] = top_k
            return {
                "route": "exact",
                "classification": "exact",
                "query": query,
                "results": [{
                    "document_id": 9, "file_name": "ID Card.pdf",
                    "field_name": "roll", "field_value": "12345678",
                }],
                "fallback": {"occurred": False, "reason": None},
                "timings_ms": {},
            }

        with mock.patch.object(answer_engine, "route_query", fake_route):
            answer_engine.answer_query("What is my roll number?")

        self.assertEqual(captured["top_k"], 3)

    def test_explicit_caller_top_k_wins(self) -> None:
        captured: dict = {}

        def fake_route(query, top_k=3, **kwargs):
            captured["top_k"] = top_k
            return {
                "route": "semantic", "classification": "semantic", "query": query,
                "results": [_chunk(8, "ApplicationForm.pdf", 0, "fact one.")],
                "fallback": {"occurred": False, "reason": None},
                "timings_ms": {},
            }

        with mock.patch.object(answer_engine, "route_query", fake_route), \
                mock.patch.object(answer_engine, "_generate_semantic_answer") as gen:
            gen.return_value = {
                "answer": "fact one.",
                "sources": [{"document_id": 8, "file_name": "ApplicationForm.pdf"}],
                "model_confidence": "high", "grounding": {"grounded": True},
                "unknown_sources": [],
            }
            answer_engine.answer_query(
                "Meri application form ki important details batao", top_k=4
            )

        self.assertEqual(captured["top_k"], 4)


# ===========================================================================
# 7. Summary prompt rules
# ===========================================================================


class TestSummaryPrompt(unittest.TestCase):
    def test_summary_prompt_has_context_only_rules(self) -> None:
        prompt = build_semantic_prompt(
            "What does my railway ticket contain?",
            [_chunk(7, "4143027140.pdf", 0, "PNR 4143027140 train 03252.")],
            style="summary",
        )
        self.assertIn("ONLY information present in the supplied context", prompt)
        self.assertIn("do NOT use outside knowledge", prompt)
        self.assertIn("say so", prompt)

    def test_fact_prompt_unchanged(self) -> None:
        prompt = build_semantic_prompt(
            "What is my roll number?",
            [_chunk(9, "ID Card.pdf", 0, "Roll 12345678.")],
            style="fact",
        )
        self.assertIn("ONE short sentence", prompt)
        self.assertNotIn("bullet", prompt)

    def test_language_rules_still_pinned(self) -> None:
        from language import LANG_HINGLISH

        prompt = build_semantic_prompt(
            "Meri application form ki important details batao",
            [_chunk(8, "ApplicationForm.pdf", 0, "Enrollment ID M103B71.")],
            language=LANG_HINGLISH,
            style="summary",
        )
        self.assertIn("Hinglish", prompt)


# ===========================================================================
# 8-9. Grounding: unchanged strictness
# ===========================================================================


class TestGroundingUnchanged(unittest.TestCase):
    def test_unsupported_identifier_still_rejected(self) -> None:
        context = "Enrollment ID M103B71 for DEVESH VISHWAKARMA."
        result = validate_grounding("The enrollment ID is XYZ99999.", context)
        self.assertFalse(result["grounded"])
        self.assertIn("XYZ99999", result["unsupported_numbers"])

    def test_supported_values_accepted(self) -> None:
        context = "Enrollment ID M103B71 dob 06/06/2006."
        result = validate_grounding("Enrollment ID M103B71, DOB 06/06/2006.", context)
        self.assertTrue(result["grounded"])

    def test_summary_budget_constants_bounded(self) -> None:
        self.assertEqual(SUMMARY_TOP_K, 6)
        self.assertEqual(SUMMARY_MAX_CONTEXT_CHARS, 4000)
        # Fact budget untouched.
        self.assertEqual(MAX_CONTEXT_CHARS, 3000)


# ===========================================================================
# 10-12. Unrelated-query gate unchanged
# ===========================================================================


class TestGateUnchanged(unittest.TestCase):
    def test_unrelated_summary_styled_query_still_rejected(self) -> None:
        mock_llm = mock.Mock()
        payload = answer_engine.answer_query(
            "Tell me a story with important details", llm_fn=mock_llm
        )
        mock_llm.assert_not_called()
        self.assertFalse(payload["retrieval_performed"])
        self.assertEqual(payload["reason"], "query_outside_document_scope")

    def test_gate_allows_real_summary_queries(self) -> None:
        from query_relevance import is_document_related

        self.assertTrue(is_document_related("What does my railway ticket contain?"))
        self.assertTrue(is_document_related("Meri application form ki important details batao"))


# ===========================================================================
# 10. Bounded retry policy (transport failures only, grounding final)
# ===========================================================================


class TestSummaryRetryPolicy(unittest.TestCase):
    """Phase 9.1: summaries retry ONCE on transport-level failure; a valid
    JSON answer rejected by validate_grounding is NEVER retried; fact
    queries get no retry at all."""

    GOOD_TICKET = (
        '{"answer": "PNR 4143027140 train 03252 sleeper.", '
        '"sources": [{"document_id": 7, "file_name": "4143027140.pdf"}], '
        '"confidence": "high"}'
    )
    UNGROUNDED_TICKET = (
        '{"answer": "PNR 9999999999 train 03252.", '
        '"sources": [{"document_id": 7, "file_name": "4143027140.pdf"}], '
        '"confidence": "high"}'
    )

    @staticmethod
    def _transport(script):
        state = {"n": 0}

        def transport(prompt, num_predict):
            state["n"] += 1
            return script(state["n"])

        return transport, state

    def setUp(self) -> None:
        self._original = answer_engine.generate_with_meta

    def tearDown(self) -> None:
        answer_engine.generate_with_meta = self._original

    def _chunks(self):
        return [
            _chunk(7, "4143027140.pdf", 0, "PNR 4143027140 train 03252 sleeper class.")
        ]

    def test_truncated_then_good_is_retried_once(self) -> None:
        transport, state = self._transport(
            lambda n: (
                {"done_reason": "length", "text": '{"answer": "partial'}
                if n == 1
                else {"done_reason": "stop", "text": self.GOOD_TICKET}
            )
        )
        answer_engine.generate_with_meta = transport
        result = answer_engine._generate_semantic_answer(
            "What does my railway ticket contain?", self._chunks()
        )
        self.assertIsNotNone(result)
        self.assertIn("PNR 4143027140", result["answer"])
        self.assertEqual(state["n"], 2)

    def test_bad_json_then_good_is_retried_once(self) -> None:
        transport, state = self._transport(
            lambda n: (
                {"done_reason": "stop", "text": "not json at all"}
                if n == 1
                else {"done_reason": "stop", "text": self.GOOD_TICKET}
            )
        )
        answer_engine.generate_with_meta = transport
        result = answer_engine._generate_semantic_answer(
            "What does my railway ticket contain?", self._chunks()
        )
        self.assertIsNotNone(result)
        self.assertEqual(state["n"], 2)

    def test_ungrounded_valid_json_is_never_retried(self) -> None:
        transport, state = self._transport(
            lambda n: {"done_reason": "stop", "text": self.UNGROUNDED_TICKET}
        )
        answer_engine.generate_with_meta = transport
        result = answer_engine._generate_semantic_answer(
            "What does my railway ticket contain?", self._chunks()
        )
        self.assertIsNone(result)  # validate_grounding verdict is final
        self.assertEqual(state["n"], 1)

    def test_fact_query_gets_no_retry(self) -> None:
        transport, state = self._transport(
            lambda n: {"done_reason": "length", "text": '{"answer": "partial'}
        )
        answer_engine.generate_with_meta = transport
        result = answer_engine._generate_semantic_answer(
            "What is my PNR number?", self._chunks()
        )
        self.assertIsNone(result)
        self.assertEqual(state["n"], 1)

    def test_summary_retry_budget_is_bounded(self) -> None:
        self.assertEqual(answer_engine.SUMMARY_MAX_RETRIES, 1)

    def test_transport_failure_raises_unavailable_summary(self) -> None:
        """Ollama down must surface as LLMUnavailableError, never None->insufficient."""

        transport, state = self._transport(lambda n: None)
        answer_engine.generate_with_meta = transport
        with self.assertRaises(answer_engine.LLMUnavailableError):
            answer_engine._generate_semantic_answer(
                "What does my railway ticket contain?", self._chunks()
            )
        self.assertEqual(state["n"], 1)  # transport None is not retried

    def test_transport_failure_raises_unavailable_fact(self) -> None:
        transport, state = self._transport(lambda n: None)
        answer_engine.generate_with_meta = transport
        with self.assertRaises(answer_engine.LLMUnavailableError):
            answer_engine._generate_semantic_answer(
                "What is my PNR number?", self._chunks()
            )
        self.assertEqual(state["n"], 1)

    def test_exact_route_never_raises_transport_error(self) -> None:
        """Exact SQLite answers must not depend on the model at all."""

        def exploding_transport(prompt, num_predict=0):
            raise AssertionError("transport must not be called for exact route")

        original = answer_engine.route_query
        answer_engine.route_query = lambda *a, **k: {
            "query": "Mera roll number kya hai?",
            "route": "exact",
            "results": [
                {
                    "document_id": 7,
                    "file_name": "ID Card.pdf",
                    "field_name": "roll_number",
                    "field_value": "2407510100067",
                    "file_path": "samples/ID Card.pdf",
                }
            ],
            "field_hint": "roll number",
            "classification": "exact_field",
            "fallback": {"occurred": False, "reason": None},
            "timings_ms": {},
        }
        try:
            payload = answer_engine.answer_query("Mera roll number kya hai?")
        finally:
            answer_engine.route_query = original
        self.assertEqual(payload.get("retrieval_route"), "exact")
        self.assertFalse(payload.get("llm_called"))
        self.assertIn("2407510100067", payload.get("answer", ""))


# ===========================================================================
# Runner
# ===========================================================================


def main() -> int:
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for cls in (
        TestSummaryDetection,
        TestContextAssembly,
        TestSummaryRetrievalPolicy,
        TestSummaryPrompt,
        TestGroundingUnchanged,
        TestGateUnchanged,
        TestSummaryRetryPolicy,
    ):
        suite.addTests(loader.loadTestsFromTestCase(cls))

    from _harness import guard_real_db, real_db_state, restore_storage

    before = real_db_state()

    runner = unittest.TextTestRunner(verbosity=1)
    try:
        result = runner.run(suite)
    finally:
        guard_real_db(before)
        restore_storage()

    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
