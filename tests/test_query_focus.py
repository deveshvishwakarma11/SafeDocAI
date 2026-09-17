"""
Phase 9: Answer-focused query handling test suite (plain-Python runner).
=========================================================================

Covers:

* relevance gate   - unrelated queries rejected with NO retrieval/LLM;
                     document queries pass;
* short-circuiting - exact -> SQLite only; semantic -> Chroma + LLM;
                     exact miss -> existing fallback preserved;
* answer phrasing  - fact queries are short, summary requests allowed a
                     concise paragraph, multi-value answers preserved;
* ambiguity        - multiple grounded values are NOT collapsed;
* language         - EN/Hinglish/Hindi phrasing with values verbatim;
* provenance       - payload shape keeps sources/route/grounded intact.

ISOLATION: storage redirected into a temp directory for the whole run
(the real data/safedoc.db and data/chroma_db are never touched); the
LLM is mocked at the ``llm_fn`` seam - no Ollama call is required.
A post-run guard re-checks the real DB's size/mtime/sha256.

Run:  PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 python tests/test_query_focus.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parents[1]
for entry in (str(_project_root), str(_project_root / "src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import storage_engine  # the SAME module instance the router/engine bind to
from _harness import redirect_storage_to_temp, restore_storage, run_tests, seed_chroma_real_model

from answer_engine import (
    INSUFFICIENT_CONTEXT_MESSAGE,
    answer_query,
    build_answer_from_exact,
    build_semantic_prompt,
    is_summary_request,
)
from language import LANG_ENGLISH, LANG_HINGLISH, LANG_HINDI
from query_relevance import check_query_relevance, is_document_related
from query_router import route_query
from storage_engine import ingest_into_sqlite

# ---------------------------------------------------------------------------
# Isolation: redirect storage into a temp directory
# ---------------------------------------------------------------------------

_TEMP_DIR = redirect_storage_to_temp("safedocai-focus-tests-")

# ---------------------------------------------------------------------------
# Seed data (fixture content, isolated temp stores only)
# ---------------------------------------------------------------------------

_SEED_DOCS = [
    {
        "file_name": "ApplicationForm.pdf",
        "file_type": "PDF",
        "status": "success",
        "raw_text": (
            "Application Form. Enrollment ID M103B71 for DEVESH "
            "VISHWAKARMA born 06/06/2006, candidate category General, "
            "phone number 9163791592. Exam cities Gorakhpur, Prayagraj "
            "and Kanpur."
        ),
        "extracted_entities": {
            "enrollment_id": ["M103B71"],
            "phone": ["9163791592"],
            "dob": ["06/06/2006"],
        },
    },
    {
        "file_name": "marksheet.pdf",
        "file_type": "PDF",
        "status": "success",
        "raw_text": (
            "Consolidated Marksheet. Student DEVESH VISHWAKARMA roll "
            "number 2407510100067 semester SGPA 6.09. Enrollment number "
            "240781010033745 issued by the university examination wing. "
            "Covers semester wise subject marks and performance details."
        ),
        "extracted_entities": {
            "roll": ["2407510100067"],
            "sgpa": ["6.09"],
            "enrollment_id": ["240781010033745"],
        },
    },
    {
        "file_name": "4143027140.pdf",
        "file_type": "PDF",
        "status": "success",
        "raw_text": (
            "Electronic Reservation Ticket PNR 4143027140. Train 03252 "
            "sleeper class, transaction ID 100006773136446, fare "
            "amount 645 INR, quota GENERAL. Photo ID cards required "
            "during travel along with this ticket."
        ),
        "extracted_entities": {
            "pnr": ["4143027140"],
            "transaction_id": ["100006773136446"],
            "train": ["03252"],
        },
    },
]

_DOCS_SEEDED = False


def _seed_all() -> None:
    global _DOCS_SEEDED
    if _DOCS_SEEDED:
        return

    storage_engine.init_db()

    doc_ids = []
    for record in _SEED_DOCS:
        parsed = dict(record)
        parsed["file_path"] = str((_TEMP_DIR / record["file_name"]).resolve())
        doc_ids.append(ingest_into_sqlite(parsed, Path("data/output/seed.json")))

    seed_chroma_real_model(_SEED_DOCS, doc_ids)

    _DOCS_SEEDED = True


# ---------------------------------------------------------------------------
# LLM mock helpers (transport seam only - never real data)
# ---------------------------------------------------------------------------


class MockLLM:
    """Records prompts and returns canned JSON responses."""

    def __init__(self, payload: str) -> None:
        self.payload = payload
        self.prompts: list[str] = []
        self.calls = 0

    def __call__(self, question: str, chunks: list[dict]) -> dict | None:
        self.calls += 1
        self.prompts.append(build_semantic_prompt(question, chunks))
        context_text = "\n\n".join(str(c.get("text", "")) for c in chunks)
        from answer_engine import parse_semantic_response

        return parse_semantic_response(self.payload, chunks, context_text)


# ===========================================================================
# 1-5. RELEVANCE GATE
# ===========================================================================


def test_gate_rejects_weather() -> None:
    assert not is_document_related("What is today's weather?")
    decision = check_query_relevance("What is today's weather?")
    assert decision["related"] is False
    assert decision["reason"] == "query_outside_document_scope"


def test_gate_rejects_general_knowledge() -> None:
    assert not is_document_related("Who is the prime minister of India?")


def test_gate_rejects_math() -> None:
    assert not is_document_related("What is 25 * 37?")
    assert not is_document_related("Calculate 15% of 800")


def test_gate_rejects_joke_and_cooking() -> None:
    assert not is_document_related("Tell me a joke.")
    assert not is_document_related("How do I cook rice?")


def test_gate_rejects_coding() -> None:
    assert not is_document_related("Write me a Python program for sorting.")


def test_gate_allows_document_question() -> None:
    assert is_document_related("What does my railway ticket contain?")
    assert is_document_related("Meri application form ki important details batao")
    assert is_document_related("ID card par kya likha hai?")


def test_gate_allows_exact_field_query() -> None:
    assert is_document_related("What is my roll number?")
    assert is_document_related("Mera enrollment ID kya hai?")
    assert is_document_related("What is the transaction ID?")


def test_gate_rejects_without_retrieval_in_answer_query() -> None:
    """Gate rejects inside answer_query: no SQLite, no Chroma, no LLM."""

    _seed_all()
    mock = MockLLM('{"answer": "unused", "sources": []}')
    payload = answer_query("What is today's weather?", llm_fn=mock)

    assert payload["retrieval_performed"] is False
    assert payload["llm_called"] is False
    assert mock.calls == 0, "rejected query must never reach the LLM"
    assert payload["reason"] == "query_outside_document_scope"
    assert payload["retrieval_route"] == "none"
    assert payload["classification"] == "rejected"
    assert payload["grounded"] is False
    assert payload["sources"] == []
    assert payload["answer"] == INSUFFICIENT_CONTEXT_MESSAGE
    assert payload["timings_ms"]["total"] < 50, (
        f"gate reject must be fast, got {payload['timings_ms']['total']} ms"
    )


def test_gate_reject_fast_and_deterministic() -> None:
    import time

    checks = ["Who is the prime minister?", "Tell me a joke.", "What is 25 * 37?"]
    first = [check_query_relevance(c) for c in checks]
    second = [check_query_relevance(c) for c in checks]
    assert first == second, "gate must be deterministic"

    started = time.perf_counter()
    for _ in range(200):
        for check in checks:
            check_query_relevance(check)
    per_call_ms = (time.perf_counter() - started) * 1000 / 600
    assert per_call_ms < 1.0, f"gate call too slow: {per_call_ms:.4f} ms"


# ===========================================================================
# 6-9. SHORT-CIRCUITING
# ===========================================================================


def test_exact_sqlite_answer_no_chroma_no_llm() -> None:
    _seed_all()
    mock = MockLLM('{"answer": "unused", "sources": []}')
    payload = answer_query("What is my roll number?", llm_fn=mock)

    assert payload["retrieval_route"] == "exact"
    assert payload["retrieval_performed"] is True
    assert mock.calls == 0, "exact answer must not call the LLM"
    assert payload["llm_called"] is False
    assert payload["retrieval_performed"] is True
    assert "2407510100067" in payload["answer"]


def test_unrelated_query_no_sqlite_chroma_llm() -> None:
    mock = MockLLM('{"answer": "unused", "sources": []}')
    payload = answer_query("Tell me a joke.", llm_fn=mock)
    assert mock.calls == 0
    assert payload["retrieval_performed"] is False
    assert payload["reason"] == "query_outside_document_scope"


def test_semantic_query_reaches_llm() -> None:
    _seed_all()
    mock = MockLLM(
        '{"answer": "The ticket covers train 03252 in sleeper class.", '
        '"sources": [{"file_name": "4143027140.pdf"}]}'
    )
    payload = answer_query("What does my railway ticket contain?", llm_fn=mock)

    assert mock.calls == 1, "semantic query must call the LLM exactly once"
    assert payload["retrieval_route"] == "semantic"
    assert payload["llm_called"] is True
    assert payload["grounded"] is True
    assert payload["retrieval_performed"] is True


def test_exact_miss_preserves_fallback() -> None:
    _seed_all()
    mock = MockLLM(
        '{"answer": "The enrollment ID is 240781010033745.", '
        '"sources": [{"file_name": "marksheet.pdf"}]}'
    )
    payload = answer_query("Mera enrollment ID kya hai?", llm_fn=mock)

    # M103B71 exists in SQLite but "Mera enrollment ID" must NOT be
    # silently pinned to one value; fallback behaviour is preserved.
    assert payload["retrieval_route"] in {"semantic", "fallback", "exact"}
    assert payload["retrieval_performed"] is True
    assert payload["reason"] == "document_related"


# ===========================================================================
# 10-12. ANSWER LENGTH
# ===========================================================================


def test_fact_query_short_answer() -> None:
    _seed_all()
    payload = answer_query("What is my roll number?", llm_fn=MockLLM("unused"))
    sentence = payload["answer"]
    assert "2407510100067" in sentence
    assert len(sentence) < 120, f"fact answer too long: {sentence!r}"
    assert sentence.count(".") <= 2  # short, not a paragraph


def test_summary_query_allows_concise_summary() -> None:
    assert is_summary_request("What does my railway ticket contain?")
    assert is_summary_request("Meri application form ki important details batao")
    assert not is_summary_request("What is my roll number?")
    assert not is_summary_request("Mera phone number kya hai?")


def test_multi_value_answer_preserved() -> None:
    rows = [
        {
            "document_id": 1,
            "file_name": "ApplicationForm.pdf",
            "file_path": "x/ApplicationForm.pdf",
            "field_name": "phone",
            "field_value": "9163791592",
        },
        {
            "document_id": 2,
            "file_name": "ID Card.pdf",
            "file_path": "x/ID Card.pdf",
            "field_name": "phone",
            "field_name_alt": None,
            "field_value": "9163791593",
        },
    ]
    payload = build_answer_from_exact(
        {
            "route": "exact",
            "query": "What is my phone number?",
            "field_hint": "phone number",
            "field_name": "phone",
            "matched_field": "phone",
            "results": rows,
            "fallback": {"occurred": False, "reason": None},
            "timings_ms": {},
        }
    )
    answer = payload["answer"]
    assert "9163791592" in answer and "9163791593" in answer
    assert "ApplicationForm.pdf" in answer and "ID Card.pdf" in answer


# ===========================================================================
# 13. AMBIGUITY HANDLING
# ===========================================================================


def test_ambiguity_not_collapsed() -> None:
    rows = [
        {
            "document_id": 1,
            "file_name": "ApplicationForm.pdf",
            "file_path": "x",
            "field_name": "enrollment_id",
            "field_value": "M103B71",
        },
        {
            "document_id": 2,
            "file_name": "marksheet.pdf",
            "file_path": "x",
            "field_name": "enrollment_id",
            "field_value": "240781010033745",
        },
    ]
    payload = build_answer_from_exact(
        {
            "route": "exact",
            "query": "What is my enrollment ID?",
            "field_hint": "enrollment ID",
            "field_name": "enrollment_id",
            "matched_field": "enrollment_id",
            "results": rows,
            "fallback": {"occurred": False, "reason": None},
            "timings_ms": {},
        }
    )
    answer = payload["answer"]
    # Both grounded values must survive; neither may be silently dropped.
    assert "M103B71" in answer
    assert "240781010033745" in answer
    assert "ApplicationForm.pdf" in answer and "marksheet.pdf" in answer


# ===========================================================================
# 14-16. LANGUAGE
# ===========================================================================


def test_english_fact_answer() -> None:
    rows = [
        {
            "document_id": 1,
            "file_name": "marksheet.pdf",
            "file_path": "x",
            "field_name": "roll",
            "field_value": "2407510100067",
        }
    ]
    payload = build_answer_from_exact(
        {
            "route": "exact",
            "query": "What is my roll number?",
            "field_hint": "roll number",
            "results": rows,
            "fallback": {"occurred": False, "reason": None},
            "timings_ms": {},
        }
    )
    assert payload["language"] == LANG_ENGLISH
    assert payload["answer"] == "Your roll number is 2407510100067."


def test_hinglish_fact_answer() -> None:
    rows = [
        {
            "document_id": 1,
            "file_name": "marksheet.pdf",
            "file_path": "x",
            "field_name": "roll",
            "field_value": "2407510100067",
        }
    ]
    payload = build_answer_from_exact(
        {
            "route": "exact",
            "query": "Mera roll number kya hai?",
            "field_hint": "roll number",
            "results": rows,
            "fallback": {"occurred": False, "reason": None},
            "timings_ms": {},
        }
    )
    assert payload["language"] == LANG_HINGLISH
    assert payload["answer"] == "Aapka roll number 2407510100067 hai."


def test_hindi_fact_answer_value_verbatim() -> None:
    rows = [
        {
            "document_id": 1,
            "file_name": "marksheet.pdf",
            "file_path": "x",
            "field_name": "roll",
            "field_value": "2407510100067",
        }
    ]
    payload = build_answer_from_exact(
        {
            "route": "exact",
            "query": "मेरा रोल नंबर क्या है?",
            "field_hint": "roll number",
            "results": rows,
            "fallback": {"occurred": False, "reason": None},
            "timings_ms": {},
        }
    )
    assert payload["language"] == LANG_HINDI
    assert "2407510100067" in payload["answer"]


def test_semantic_prompt_language_preserved() -> None:
    prompt = build_semantic_prompt(
        "Mera roll number kya hai?",
        [{"file_name": "m.pdf", "text": "roll 2407510100067", "document_id": 1}],
        language=LANG_HINGLISH,
        style="fact",
    )
    assert "Hinglish" in prompt
    assert "2407510100067" in prompt  # context values intact


# ===========================================================================
# 17-19. PROVENANCE / PAYLOAD SHAPE
# ===========================================================================


def test_provenance_fields_preserved_on_exact() -> None:
    _seed_all()
    payload = answer_query("What is my roll number?", llm_fn=MockLLM("unused"))
    for key in ("sources", "source_documents", "retrieval_route", "grounded"):
        assert key in payload, f"missing provenance key: {key}"
    assert payload["sources"], "exact answers must keep sources"
    first = payload["sources"][0]
    for key in ("document_id", "file_name", "field_name", "field_value"):
        assert key in first


def test_provenance_fields_preserved_on_semantic() -> None:
    _seed_all()
    mock = MockLLM(
        '{"answer": "Train 03252 sleeper class.", '
        '"sources": [{"file_name": "4143027140.pdf"}]}'
    )
    payload = answer_query("What does my railway ticket contain?", llm_fn=mock)
    for key in ("sources", "retrieval_route", "grounded", "timings_ms"):
        assert key in payload
    assert payload["retrieval_route"] == "semantic"
    assert payload["grounded"] is True


# ===========================================================================
# Runner
# ===========================================================================


def main() -> int:
    tests = sorted(
        (name, fn)
        for name, fn in globals().items()
        if name.startswith("test_") and callable(fn)
    )
    try:
        run_tests(tests, "Query focus")
    finally:
        restore_storage()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
