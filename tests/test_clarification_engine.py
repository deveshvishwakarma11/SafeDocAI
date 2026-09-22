"""
Clarification Engine tests (Step 4, plain-Python runner, no pytest).
====================================================================

Covers the approved Step 4 contract:

 1. Ambiguous field                      13. Deterministic question generation
 2. Ambiguous document                   14. Stable option ordering
 3. Ambiguous reference                  15. No invented candidates
 4. Ambiguous constraint                 16. No candidate explosion
 5. No clarification for clear field     17. No LLM calls
 6. No clarification for clear document  18. Malformed/empty input
 7. No clarification for clear summary   19. Structured payload contract
 8. Context-resolved follow-up           20. answer_engine integration
 9. Unresolved follow-up                 21. No retrieval on clarification
10. English clarification                22. No expensive semantic LLM call
11. Hindi clarification                  23. Existing clear queries unchanged
12. Hinglish clarification               24. ConversationState not mutated

The clarification engine is deterministic: no LLM, no network. Storage is
redirected to temp paths for the engine-integration tests (the real
data/safedoc.db and data/chroma_db are never touched; a post-run guard
verifies it). The LLM is MOCKED at the module seam.
"""

from __future__ import annotations

import copy
import json
import sys
import unittest.mock
from pathlib import Path

_project_root = Path(__file__).resolve().parents[1]
for _entry in (str(_project_root), str(_project_root / "src")):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

import storage_engine  # noqa: F401  (bound before any redirect)
from _harness import redirect_storage_to_temp, seed_chroma_real_model

import answer_engine
from answer_engine import build_semantic_prompt, parse_semantic_response
from clarification_engine import (
    CLARIFICATION_CONSTRAINT,
    CLARIFICATION_DOCUMENT,
    CLARIFICATION_FIELD,
    CLARIFICATION_REFERENCE,
    MAX_CLARIFICATION_OPTIONS,
    Clarification,
    available_document_names,
    build_clarification,
)
from conversation_state import ConversationState
from request_understanding import understand_request
from storage_engine import ingest_into_sqlite

# ---------------------------------------------------------------------------
# Isolation: redirect storage into a temp directory (engine integration tests)
# ---------------------------------------------------------------------------

_TEMP_DIR = redirect_storage_to_temp("safedocai-clarify-tests-")

_SEED_DOCS = [
    {
        "file_name": "marksheet.pdf",
        "file_type": "PDF",
        "status": "success",
        "raw_text": (
            "Consolidated Marksheet. Semester 1 SGPA 6.09. Student TEST "
            "USER roll number 12345678 born on 06-06-2006."
        ),
        "extracted_entities": {
            "sgpa": ["6.09"],
            "roll": ["12345678"],
            "dob": ["06-06-2006"],
        },
    },
    {
        "file_name": "idcard.pdf",
        "file_type": "PDF",
        "status": "success",
        "raw_text": (
            "College identity card for TEST USER, roll number 12345678, "
            "course B.Tech CSE, session 2024 to 2028."
        ),
        "extracted_entities": {
            "roll": ["12345678"],
            "Course": ["B.Tech | CSE"],
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
        doc_id = ingest_into_sqlite(parsed, Path("data/output/seed.json"))
        record["doc_id"] = doc_id
        doc_ids.append(doc_id)

    seed_chroma_real_model(_SEED_DOCS, doc_ids)
    _DOCS_SEEDED = True


class MockLLM:
    """Records prompts; any call means the LLM path was (wrongly) used."""

    def __init__(self, payload: str = '{"answer": "unused", "sources": []}') -> None:
        self.payload = payload
        self.prompts: list[str] = []

    def __call__(self, question: str, chunks: list[dict]) -> dict | None:
        self.prompts.append(build_semantic_prompt(question, chunks))
        context_text = "\n\n".join(str(c.get("text", "")) for c in chunks)
        return parse_semantic_response(self.payload, chunks, context_text)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DOCS = ["ApplicationForm.pdf", "4143027140.pdf", "ID Card.pdf"]


def _sr(query: str, context=None):
    return understand_request(query, context=context)


def _clarify(query: str, context=None, docs=_DOCS) -> Clarification:
    return build_clarification(_sr(query, context), context=context, available_documents=docs)


def _ticket_context() -> dict:
    state = ConversationState()
    state.add_turn(
        "meri railway ticket ki details batao",
        "Ticket details: PNR 4143027140, fare 545.",
        "hinglish",
    )
    return state.to_context()


# ---------------------------------------------------------------------------
# 1) Ambiguous field
# ---------------------------------------------------------------------------

def test_ambiguous_field_number_batao() -> None:
    c = _clarify("number batao")
    assert c.needed is True
    assert c.reason == CLARIFICATION_FIELD
    assert c.options, "field clarification must carry the ACTUAL candidates"
    assert c.question and "Kaunsa number chahiye" in c.question
    for option in c.options:
        assert option in c.candidates  # only real candidates are listed


def test_ambiguous_field_four_sem_number() -> None:
    c = _clarify("4 sem ka number batao")
    assert c.needed is True
    assert c.reason == CLARIFICATION_FIELD
    assert "number" in c.question


def test_ambiguous_field_form_ka_number() -> None:
    c = _clarify("form ka number batao")
    assert c.needed is True
    assert c.reason == CLARIFICATION_FIELD


def test_ambiguous_field_mera_result() -> None:
    c = _clarify("mera result")
    assert c.needed is True
    assert c.reason == CLARIFICATION_FIELD
    assert set(c.options) == {"sgpa", "marks", "total_marks"}


def test_ambiguous_field_result_ki_detail_de() -> None:
    c = _clarify("result ki detail de")
    assert c.needed is True
    assert c.reason == CLARIFICATION_FIELD


def test_ambiguous_field_four_sem_result() -> None:
    c = _clarify("4 sem ka result batao")
    assert c.needed is True
    assert c.reason == CLARIFICATION_FIELD


# ---------------------------------------------------------------------------
# 2) Ambiguous document (only with ACTUAL document names)
# ---------------------------------------------------------------------------

def test_ambiguous_document_with_real_names() -> None:
    c = _clarify("meri details batao", docs=["Application Form", "Railway Ticket", "ID Card"])
    assert c.needed is True
    assert c.reason == CLARIFICATION_DOCUMENT
    assert c.options == ("Application Form", "Railway Ticket", "ID Card")
    assert "Kaunse document" in c.question


def test_ambiguous_document_lazy_callable() -> None:
    c = build_clarification(
        _sr("meri details batao"),
        available_documents=lambda: ["A.pdf", "B.pdf"],
    )
    assert c.needed is True
    assert c.reason == CLARIFICATION_DOCUMENT
    assert c.options == ("A.pdf", "B.pdf")


def test_no_document_clarification_without_names() -> None:
    # Nothing is invented: without real names NO question is emitted.
    c = _clarify("meri details batao", docs=None)
    assert c.needed is False and c.question is None


def test_no_document_clarification_for_single_document() -> None:
    c = _clarify("meri details batao", docs=["Only One.pdf"])
    assert c.needed is False


def test_no_document_clarification_for_resolved_target() -> None:
    c = _clarify("ticket ki details batao")
    assert c.needed is False


def test_document_names_callable_failure_is_safe() -> None:
    def _boom() -> list[str]:
        raise RuntimeError("store down")

    c = build_clarification(_sr("meri details batao"), available_documents=_boom)
    assert c.needed is False


# ---------------------------------------------------------------------------
# 3) Ambiguous reference
# ---------------------------------------------------------------------------

def test_ambiguous_reference_unresolved_pronoun() -> None:
    c = _clarify("iska number batao", context=None)
    assert c.needed is True
    assert c.reason == CLARIFICATION_REFERENCE
    assert c.options == tuple(_DOCS)  # actual names only


def test_ambiguous_reference_no_options_still_asks() -> None:
    c = _clarify("iska number batao", docs=None)
    assert c.needed is True
    assert c.reason == CLARIFICATION_REFERENCE
    assert c.options == ()
    assert c.question  # a name-the-document question still helps the user


# ---------------------------------------------------------------------------
# 4) Ambiguous constraint
# ---------------------------------------------------------------------------

def test_ambiguous_constraint_semester() -> None:
    c = _clarify("semester ka SGPA batao")
    assert c.needed is True
    assert c.reason == CLARIFICATION_CONSTRAINT
    assert c.question == "Kaunse semester ka chahiye?"
    assert c.options == ()


def test_no_constraint_clarification_when_value_present() -> None:
    for query in ("4 sem ka SGPA batao", "4th sem ka SGPA batao", "2025 ka result batao"):
        c = _clarify(query)
        assert c.reason != CLARIFICATION_CONSTRAINT, query


def test_constraint_word_part_of_field_name_never_asks() -> None:
    # "semester" inside the field synonym "semester GPA" is not a constraint.
    c = _clarify("semester GPA batao")
    assert c.reason != CLARIFICATION_CONSTRAINT


# ---------------------------------------------------------------------------
# 5-7) No clarification for clear requests
# ---------------------------------------------------------------------------

def test_no_clarification_clear_field() -> None:
    for query in (
        "mera roll number kya hai",
        "roll number",
        "PNR kya hai",
        "enrollment id batao",
        "application number batao",
        "4th sem ka SGPA batao",
        "ticket ka fare batao",
    ):
        c = _clarify(query)
        assert c.needed is False, (query, c.reason)
        assert c.question is None, query


def test_no_clarification_clear_document() -> None:
    for query in ("ticket ki details batao", "meri application ki important details batao"):
        c = _clarify(query)
        assert c.needed is False, (query, c.reason)


def test_no_clarification_conversation_and_out_of_scope() -> None:
    for query in ("hello", "weather today", "tum kya kya kar sakte ho", ""):
        c = _clarify(query)
        assert c.needed is False, (query, c.reason)


# ---------------------------------------------------------------------------
# 8-9) Context-resolved vs unresolved follow-ups
# ---------------------------------------------------------------------------

def test_context_resolved_followup_no_clarification() -> None:
    context = _ticket_context()
    for query in ("iska fare batao", "aur PNR?", "uska date batao"):
        c = _clarify(query, context=context)
        assert c.needed is False, (query, c.reason)


def test_unresolved_followup_after_unrelated_context() -> None:
    state = ConversationState()
    state.add_turn("hello", "Hello! How can I help?", "english")
    c = _clarify("iska number batao", context=state.to_context())
    assert c.needed is True
    assert c.reason == CLARIFICATION_REFERENCE


def test_state_object_accepted_directly() -> None:
    """A ConversationState (not just a dict) is accepted as read-only
    context. The ticket context resolves the 'iska' REFERENCE, but a
    follow-up asking for a 'number' remains FIELD-ambiguous (PNR?
    transaction ID? phone?) -- clarifying it is the never-guess contract.
    A single-field follow-up resolves fully and never clarifies."""
    state = ConversationState()
    state.add_turn(
        "meri railway ticket ki details batao",
        "Ticket details: PNR 4143027140, fare 545.",
        "hinglish",
    )
    # Field still ambiguous -> WHICH number? (document is resolved, the
    # field is not; the engine does not pick one)
    c = build_clarification(
        _sr("iska number batao", context=state),
        context=state,
        available_documents=_DOCS,
    )
    assert c.needed is True
    assert c.reason == CLARIFICATION_FIELD
    # Single-field follow-up -> fully resolved, no clarification.
    c_fare = build_clarification(
        _sr("iska fare batao", context=state),
        context=state,
        available_documents=_DOCS,
    )
    assert c_fare.needed is False


def test_field_only_context_cannot_resolve_reference() -> None:
    """A prior turn naming a FIELD (roll number) does NOT resolve which
    document 'iska' refers to -- the reference stays ambiguous."""
    state = ConversationState()
    state.add_turn("mera roll number batao", "Roll number: 12345678", "hinglish")
    c = build_clarification(
        _sr("iska number batao", context=state),
        context=state,
        available_documents=_DOCS,
    )
    assert c.needed is True
    assert c.reason == CLARIFICATION_REFERENCE


# ---------------------------------------------------------------------------
# 10-12) Language of the clarification question
# ---------------------------------------------------------------------------

def test_hinglish_question() -> None:
    c = _clarify("number batao")
    assert c.language == "hinglish"
    assert c.question.startswith("Kaunsa number chahiye")


def test_english_question() -> None:
    c = _clarify("What is my number?")
    assert c.language == "english"
    assert c.question.startswith("Which number do you need")


def test_hindi_devanagari_ambiguous_without_candidates() -> None:
    """Devanagari query: Step 3 flags ambiguity but supplies no Latin-
    vocabulary field candidates. With real documents available the engine
    asks WHICH DOCUMENT instead of letting retrieval silently pick one
    number (never guess); with no names available nothing is invented and
    the existing evidence-grounded flow handles the request."""
    c = _clarify("मेरा नंबर बताओ")
    assert c.language == "hindi"
    assert c.needed is True
    assert c.reason == CLARIFICATION_DOCUMENT
    assert c.options == tuple(_DOCS)  # real names only, nothing invented
    # Without real names: no question (never invent options).
    c_none = _clarify("मेरा नंबर बताओ", docs=None)
    assert c_none.needed is False and c_none.question is None


# ---------------------------------------------------------------------------
# 13-16) Determinism, ordering, no invention, no explosion
# ---------------------------------------------------------------------------

def test_deterministic_question_generation() -> None:
    first = _clarify("number batao")
    for _ in range(5):
        again = _clarify("number batao")
        assert again.to_dict() == first.to_dict()


def test_stable_option_ordering() -> None:
    orders = {tuple(_clarify("number batao").options) for _ in range(5)}
    assert len(orders) == 1


def test_no_invented_candidates() -> None:
    c = _clarify("number batao")
    listed = set(c.options)
    omitted = set(c.meta.get("omitted_candidates", []))
    assert listed | omitted == set(c.candidates)
    assert listed <= set(c.candidates)  # every listed option is a real candidate


def test_no_candidate_explosion() -> None:
    for query in ("number batao", "mera result", "4 sem ka number batao"):
        c = _clarify(query)
        assert len(c.options) <= MAX_CLARIFICATION_OPTIONS


def test_option_cap_with_omitted_list() -> None:
    docs = [f"Document {i}.pdf" for i in range(1, 8)]  # 7 names
    c = _clarify("meri details batao", docs=docs)
    assert c.needed is True
    assert len(c.options) == MAX_CLARIFICATION_OPTIONS
    assert c.meta.get("omitted_candidates") == ["Document 6.pdf", "Document 7.pdf"]


# ---------------------------------------------------------------------------
# 17) No LLM calls (structural + behavioral)
# ---------------------------------------------------------------------------

def test_module_has_no_llm_or_network_imports() -> None:
    source = Path(__file__).resolve().parents[1].joinpath(
        "src", "clarification_engine.py"
    ).read_text(encoding="utf-8")
    # Whole-token import/module-name scan (substring matching would false-
    # positive on words like "requests" inside prose comments).
    for banned in (
        "import llm_engine", "from llm_engine", "import requests",
        "import urllib", "import http", "import socket", "import subprocess",
    ):
        assert banned not in source, banned


def test_no_llm_call_during_build() -> None:
    with unittest.mock.patch.object(
        answer_engine, "_generate_semantic_answer", side_effect=AssertionError
    ):
        _clarify("number batao")


# ---------------------------------------------------------------------------
# 18) Malformed / empty input
# ---------------------------------------------------------------------------

def test_malformed_inputs_never_raise() -> None:
    for bad in (None, "", 123, {}, {"nonsense": True}, [], "just a string", 3.14):
        c = build_clarification(bad)
        assert isinstance(c, Clarification)
        assert c.needed is False


def test_empty_query_understanding_never_clarifies() -> None:
    c = _clarify("   ")
    assert c.needed is False


# ---------------------------------------------------------------------------
# 19) Structured payload contract
# ---------------------------------------------------------------------------

def test_clarification_to_dict_json_safe() -> None:
    c = _clarify("number batao")
    payload = c.to_dict()
    for key in ("needed", "question", "options", "reason"):
        assert key in payload
    assert isinstance(payload["options"], list)
    json.dumps(payload)  # must not raise
    empty = build_clarification(None).to_dict()
    json.dumps(empty)
    assert empty["needed"] is False and empty["question"] is None


def test_result_type_is_frozen_dataclass() -> None:
    c = _clarify("number batao")
    assert isinstance(c, Clarification)
    try:
        c.needed = False  # type: ignore[misc]
        raised = False
    except Exception:  # noqa: BLE001 - frozen dataclass raises
        raised = True
    assert raised, "Clarification must be immutable"


# ---------------------------------------------------------------------------
# 20-23) answer_engine integration (seeded temp store)
# ---------------------------------------------------------------------------

def test_engine_returns_clarification_without_retrieval_or_llm() -> None:
    _seed_all()
    mock = MockLLM()
    with unittest.mock.patch.object(
        answer_engine, "_generate_semantic_answer", side_effect=AssertionError
    ):
        result = answer_engine.answer_query("number batao", llm_fn=mock)
    assert result["classification"] == "clarification"
    assert result["retrieval_route"] == "none"
    assert result["retrieval_performed"] is False
    assert result["llm_called"] is False
    assert result["sources"] == []
    assert result["clarification"]["needed"] is True
    assert result["clarification"]["reason"] == CLARIFICATION_FIELD
    assert "Kaunsa number chahiye" in result["answer"]
    assert mock.prompts == []  # the LLM was never invoked
    assert result["request_understanding"] is not None


def test_engine_constraint_clarification() -> None:
    _seed_all()
    mock = MockLLM()
    result = answer_engine.answer_query("semester ka SGPA batao", llm_fn=mock)
    assert result["classification"] == "clarification"
    assert result["clarification"]["reason"] == CLARIFICATION_CONSTRAINT
    assert result["clarification"]["question"] == "Kaunse semester ka chahiye?"
    assert mock.prompts == []


def test_engine_reference_clarification_without_context() -> None:
    _seed_all()
    result = answer_engine.answer_query("iska number batao")
    assert result["classification"] == "clarification"
    assert result["clarification"]["reason"] == CLARIFICATION_REFERENCE
    # Options come from the REAL (temp) store, deduped and deterministic.
    assert set(result["clarification"]["options"]) == {"marksheet.pdf", "idcard.pdf"}


def test_engine_document_clarification_with_real_store_names() -> None:
    _seed_all()
    result = answer_engine.answer_query("meri details batao")
    assert result["classification"] == "clarification"
    assert result["clarification"]["reason"] == CLARIFICATION_DOCUMENT
    assert set(result["clarification"]["options"]) == {"marksheet.pdf", "idcard.pdf"}


def test_engine_context_resolved_followup_still_answers() -> None:
    _seed_all()
    mock = MockLLM()
    state = ConversationState()
    state.add_turn(
        "mera roll number batao",
        "Roll number: 12345678",
        "hinglish",
    )
    result = answer_engine.answer_query(
        "roll number", llm_fn=mock, context=state.to_context()
    )
    assert result["retrieval_route"] == "exact"
    assert result["grounded"] is True
    assert mock.prompts == []
    assert "12345678" in result["answer"]


def test_engine_clear_queries_remain_unchanged() -> None:
    _seed_all()
    mock = MockLLM()
    exact = answer_engine.answer_query("roll number", llm_fn=mock)
    assert exact["retrieval_route"] == "exact"
    assert exact["grounded"] is True
    assert exact["llm_called"] is False
    assert exact.get("clarification") is None  # no clarification key/None payload
    assert exact["request_understanding"] is not None  # Step 3 contract kept

    conversation = answer_engine.answer_query("hello")
    assert conversation["classification"] == "conversation"
    assert conversation.get("clarification") is None

    rejected = answer_engine.answer_query("weather today")
    assert rejected["classification"] == "rejected"
    assert rejected.get("clarification") is None


def test_engine_document_clarification_lazy_no_db_for_clear_queries() -> None:
    """Clear queries must not even resolve document names (lazy contract)."""
    _seed_all()
    calls: list[int] = []

    def counting() -> list[str] | None:
        calls.append(1)
        return ["marksheet.pdf", "idcard.pdf"]

    import clarification_engine

    with unittest.mock.patch.object(
        clarification_engine, "available_document_names", counting
    ):
        answer_engine.answer_query("roll number")
        assert calls == []  # lazy: clear query never resolved names
        answer_engine.answer_query("mera roll number kya hai")
        assert calls == []
        answer_engine.answer_query("meri details batao")
        assert len(calls) == 1  # only the document-ambiguous query asks


# ---------------------------------------------------------------------------
# 24) ConversationState is not mutated
# ---------------------------------------------------------------------------

def test_conversation_state_never_mutated_by_clarification() -> None:
    state = ConversationState()
    state.add_turn("meri railway ticket ki details batao", "...", "hinglish")
    context = state.to_context()
    snapshot = copy.deepcopy(context)

    _clarify("iska fare batao", context=context)
    _clarify("iska number batao", context=context)
    build_clarification(_sr("number batao"), context=context, available_documents=_DOCS)

    assert context == snapshot
    assert len(state.to_context()["turns"]) == 1  # state object itself intact


def test_module_never_imports_conversation_state_for_writing() -> None:
    """The engine reads context dicts only; it never owns state mutation."""
    source = Path(__file__).resolve().parents[1].joinpath(
        "src", "clarification_engine.py"
    ).read_text(encoding="utf-8")
    assert "add_turn" not in source
    assert "ConversationState(" not in source.replace("ConversationState) ", "")


# ---------------------------------------------------------------------------
# UI payload pass-through
# ---------------------------------------------------------------------------

def test_ui_payload_passes_clarification_through() -> None:
    import ui

    engine_payload = {
        "query": "number batao",
        "answer": "Kaunsa number chahiye — roll number ya PNR?",
        "sources": [],
        "source_documents": [],
        "retrieval_route": "none",
        "classification": "clarification",
        "clarification": {
            "needed": True,
            "question": "Kaunsa number chahiye — roll number ya PNR?",
            "options": ["roll", "pnr"],
            "reason": "ambiguous_field",
        },
        "fallback": {"occurred": False, "reason": None},
        "grounded": False,
        "llm_called": False,
        "insufficient": False,
        "timings_ms": {"total": 1.0},
    }
    with unittest.mock.patch.object(ui, "answer_query", return_value=engine_payload):
        normalized = ui.build_payload("number batao")
    assert normalized["ok"] is True
    assert normalized["classification"] == "clarification"
    assert normalized["clarification"]["needed"] is True
    assert normalized["clarification"]["options"] == ["roll", "pnr"]


def test_ui_failure_payload_has_null_clarification() -> None:
    import ui

    payload = ui._failure_payload("boom")
    assert payload["clarification"] is None


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _TESTS: list[tuple[str, object]] = [
        (name, fn)
        for name, fn in sorted(globals().items())
        if name.startswith("test_") and callable(fn)
    ]
    from _harness import run_tests

    run_tests(_TESTS, "Step 4 clarification engine")
