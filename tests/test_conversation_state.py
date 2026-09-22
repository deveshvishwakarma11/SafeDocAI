"""
Conversation State Layer tests (Step 1, plain-Python runner, no pytest).
========================================================================

Covers the Step 1 contract ONLY:

* ConversationState can be created; empty state works;
* turns are stored immutably and preserve order;
* the bounded-history cap still drops the oldest turn;
* ``to_context()`` emits the stable read-only context dict;
* context (state object or dict) can be passed into ``answer_query``
  and is echoed in every payload;
* ``answer_query(query)`` and ``answer_query(query, context=None)``
  behave EXACTLY as before (byte-for-byte payload equivalence);
* passing context does NOT change routing, answers, or language;
* the caller's state object is never mutated by the engine;
* no SQLite/Chroma/data files are modified (harness isolation guard).

Run:  PYTHONIOENCODING=utf-8 python tests/test_conversation_state.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parents[1]
for entry in (str(_project_root), str(_project_root / "src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import storage_engine  # the SAME module instance the engine binds to
from _harness import redirect_storage_to_temp, run_tests, seed_chroma_real_model  # noqa: E402

from answer_engine import answer_query  # noqa: E402
from conversation_state import (  # noqa: E402
    MAX_STATE_TURNS,
    ConversationState,
    ConversationTurn,
)
from language import LANG_HINGLISH  # noqa: E402
from query_router import route_query  # noqa: E402
from storage_engine import ingest_into_sqlite  # noqa: E402

# ---------------------------------------------------------------------------
# Isolation: redirect storage into a temp directory (real stores untouched)
# ---------------------------------------------------------------------------

_TEMP_DIR = redirect_storage_to_temp("safedocai-conversation-state-tests-")

# ---------------------------------------------------------------------------
# Seed data (fixture content, isolated temp stores only)
# ---------------------------------------------------------------------------

_SEED_DOC = {
    "file_name": "marksheet.pdf",
    "file_type": "PDF",
    "status": "success",
    "raw_text": (
        "Consolidated Marksheet. Student TEST USER roll number 2407510100067 "
        "born on 06-06-2006. The marksheet covers semester wise subject marks."
    ),
    "extracted_entities": {
        "roll": ["2407510100067"],
        "dob": ["06-06-2006"],
    },
}

_DOCS_SEEDED = False


def _seed_all() -> None:
    global _DOCS_SEEDED
    if _DOCS_SEEDED:
        return

    storage_engine.init_db()

    parsed = dict(_SEED_DOC)
    parsed["file_path"] = str((_TEMP_DIR / _SEED_DOC["file_name"]).resolve())
    doc_id = ingest_into_sqlite(parsed, Path("data/output/seed.json"))
    _SEED_DOC["doc_id"] = doc_id
    seed_chroma_real_model([_SEED_DOC], [doc_id])

    _DOCS_SEEDED = True


def _require_seed() -> None:
    _seed_all()


# ---------------------------------------------------------------------------
# 1. State creation / emptiness / turn storage / ordering
# ---------------------------------------------------------------------------


def test_state_can_be_created() -> None:
    state = ConversationState()
    assert state.is_empty()
    assert len(state) == 0
    assert state.max_turns == MAX_STATE_TURNS
    assert state.last is None


def test_turn_can_be_stored() -> None:
    state = ConversationState()
    turn = state.add_turn("mera roll number kya hai", "Aapka roll number 2407510100067 hai.")
    assert isinstance(turn, ConversationTurn)
    assert turn.user_query == "mera roll number kya hai"
    assert turn.assistant_response == "Aapka roll number 2407510100067 hai."
    assert turn.language is None
    assert state.last is turn


def test_turn_with_language_preserved() -> None:
    state = ConversationState()
    state.add_turn("mera roll number kya hai", "Aapka roll number X hai.", language=LANG_HINGLISH)
    assert state.last.language == LANG_HINGLISH


def test_multiple_turns_preserve_order() -> None:
    state = ConversationState()
    state.add_turn("mera roll number kya hai", "Aapka roll number 1 hai.")
    state.add_turn("aur phone number?", "Aapka phone number 2 hai.")
    state.add_turn("ticket ki details batao", "Ticket details 3.")

    queries = [turn.user_query for turn in state.turns]
    assert queries == [
        "mera roll number kya hai",
        "aur phone number?",
        "ticket ki details batao",
    ]
    assert state.last.user_query == "ticket ki details batao"
    assert len(state) == 3


def test_turns_are_immutable_records() -> None:
    state = ConversationState()
    turn = state.add_turn("q1", "a1")
    try:
        turn.user_query = "mutated"  # type: ignore[misc]
        raise AssertionError("ConversationTurn must be immutable")
    except AttributeError:
        pass


def test_empty_and_whitespace_queries_stripped() -> None:
    state = ConversationState()
    state.add_turn("   mera roll number   ", None)
    assert state.last.user_query == "mera roll number"
    assert state.last.assistant_response == ""


def test_clear_empties_state() -> None:
    state = ConversationState()
    state.add_turn("q", "a")
    state.clear()
    assert state.is_empty()
    assert state.to_context()["last_query"] is None


# ---------------------------------------------------------------------------
# 2. Bounded-history behavior
# ---------------------------------------------------------------------------


def test_bounded_history_respected() -> None:
    state = ConversationState(max_turns=3)
    for index in range(6):
        state.add_turn(f"q{index}", f"a{index}")
    assert len(state) == 3
    assert [turn.user_query for turn in state.turns] == ["q3", "q4", "q5"]


def test_default_cap_matches_max_state_turns() -> None:
    state = ConversationState()
    for index in range(MAX_STATE_TURNS + 3):
        state.add_turn(f"q{index}", f"a{index}")
    assert len(state) == MAX_STATE_TURNS
    assert state.turns[0].user_query == f"q{3}"


def test_invalid_max_turns_rejected() -> None:
    try:
        ConversationState(max_turns=0)
        raise AssertionError("max_turns=0 must be rejected")
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# 3. to_context() shape
# ---------------------------------------------------------------------------


def test_empty_state_context_shape() -> None:
    context = ConversationState().to_context()
    assert context == {
        "turns": [],
        "last_query": None,
        "last_response": None,
        "last_language": None,
    }


def test_context_contains_ordered_turn_dicts() -> None:
    state = ConversationState()
    state.add_turn("q1", "a1", language="hinglish")
    state.add_turn("q2", "a2")

    context = state.to_context()
    assert context["last_query"] == "q2"
    assert context["last_response"] == "a2"
    assert context["last_language"] is None
    assert context["turns"] == [
        {"user_query": "q1", "assistant_response": "a1", "language": "hinglish"},
        {"user_query": "q2", "assistant_response": "a2", "language": None},
    ]


# ---------------------------------------------------------------------------
# 4. answer_query behavior: unchanged without context / with context=None
# ---------------------------------------------------------------------------


def test_answer_query_without_context_exact() -> None:
    _require_seed()
    result = answer_query("What is my roll number?")
    assert result["classification"] == "exact"
    assert result["retrieval_route"] == "exact"
    assert result["grounded"] is True
    assert result["llm_called"] is False
    assert "2407510100067" in result["answer"]
    # Step 1 provenance: a None context is echoed as None.
    assert result["context"] is None


def test_answer_query_context_none_identical_to_no_context() -> None:
    _require_seed()
    first = answer_query("What is my roll number?")
    second = answer_query("What is my roll number?", context=None)
    first.pop("timings_ms")
    second.pop("timings_ms")
    assert first == second


def test_answer_query_out_of_scope_unchanged() -> None:
    result = answer_query("weather today", context=None)
    assert result["classification"] == "rejected"
    assert result["scope_reply"] is True
    assert result["context"] is None


def test_answer_query_conversation_unchanged() -> None:
    result = answer_query("hello", context=None)
    assert result["classification"] == "conversation"
    assert result["context"] is None


# ---------------------------------------------------------------------------
# 5. Context can be passed; it is echoed but changes NO decision
# ---------------------------------------------------------------------------


def _build_two_turn_state() -> ConversationState:
    state = ConversationState()
    state.add_turn(
        "mera roll number kya hai",
        "Aapka roll number 2407510100067 hai.",
        language=LANG_HINGLISH,
    )
    state.add_turn("aur phone number?", "Aapka phone number 9163791592 hai.")
    return state


def test_context_object_passed_into_engine() -> None:
    _require_seed()
    state = _build_two_turn_state()
    snapshot = state.to_context()

    result = answer_query("What is my roll number?", context=state)

    assert result["context"] == snapshot
    # The engine must never mutate the caller's state object.
    assert len(state) == 2
    assert state.last.user_query == "aur phone number?"


def test_context_dict_passed_into_engine() -> None:
    _require_seed()
    context = _build_two_turn_state().to_context()
    result = answer_query("What is my roll number?", context=context)
    assert result["context"] == context


def test_context_does_not_change_exact_answer() -> None:
    _require_seed()
    state = _build_two_turn_state()

    without = answer_query("What is my roll number?")
    with_ctx = answer_query("What is my roll number?", context=state)

    assert without["answer"] == with_ctx["answer"]
    assert without["retrieval_route"] == with_ctx["retrieval_route"]
    assert without["classification"] == with_ctx["classification"]
    assert without["language"] == with_ctx["language"]
    assert without["grounded"] == with_ctx["grounded"]
    assert without["sources"] == with_ctx["sources"]
    # The only allowed difference is the echoed context itself.
    assert with_ctx["context"] is not None
    without.pop("context")
    with_ctx.pop("context")
    without.pop("timings_ms")
    with_ctx.pop("timings_ms")
    assert without == with_ctx


def test_context_does_not_change_out_of_scope() -> None:
    without = answer_query("weather today")
    with_ctx = answer_query("weather today", context=_build_two_turn_state())
    assert without["answer"] == with_ctx["answer"]
    assert without["classification"] == with_ctx["classification"]
    assert with_ctx["context"]["turns"]


def test_context_does_not_change_conversation_reply() -> None:
    without = answer_query("namaste")
    with_ctx = answer_query("namaste", context=_build_two_turn_state())
    assert without["answer"] == with_ctx["answer"]
    assert without["language"] == with_ctx["language"]
    assert without["classification"] == "conversation"


def test_context_does_not_change_router_classification() -> None:
    _require_seed()
    state = _build_two_turn_state()
    assert route_query("What is my roll number?")["route"] == "exact"
    assert answer_query("What is my roll number?", context=state)["retrieval_route"] == "exact"


def test_empty_state_context_is_none_echo() -> None:
    _require_seed()
    result = answer_query("What is my roll number?", context=ConversationState())
    # An empty state emits a truthy dict with empty turns; the engine
    # echoes it verbatim (no interpretation of emptiness either).
    assert result["context"] == ConversationState().to_context()


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_tests(
        [
            ("state_can_be_created", test_state_can_be_created),
            ("turn_can_be_stored", test_turn_can_be_stored),
            ("turn_with_language_preserved", test_turn_with_language_preserved),
            ("multiple_turns_preserve_order", test_multiple_turns_preserve_order),
            ("turns_are_immutable_records", test_turns_are_immutable_records),
            ("empty_and_whitespace_queries_stripped", test_empty_and_whitespace_queries_stripped),
            ("clear_empties_state", test_clear_empties_state),
            ("bounded_history_respected", test_bounded_history_respected),
            ("default_cap_matches_max_state_turns", test_default_cap_matches_max_state_turns),
            ("invalid_max_turns_rejected", test_invalid_max_turns_rejected),
            ("empty_state_context_shape", test_empty_state_context_shape),
            ("context_contains_ordered_turn_dicts", test_context_contains_ordered_turn_dicts),
            ("answer_query_without_context_exact", test_answer_query_without_context_exact),
            ("answer_query_context_none_identical_to_no_context", test_answer_query_context_none_identical_to_no_context),
            ("answer_query_out_of_scope_unchanged", test_answer_query_out_of_scope_unchanged),
            ("answer_query_conversation_unchanged", test_answer_query_conversation_unchanged),
            ("context_object_passed_into_engine", test_context_object_passed_into_engine),
            ("context_dict_passed_into_engine", test_context_dict_passed_into_engine),
            ("context_does_not_change_exact_answer", test_context_does_not_change_exact_answer),
            ("context_does_not_change_out_of_scope", test_context_does_not_change_out_of_scope),
            ("context_does_not_change_conversation_reply", test_context_does_not_change_conversation_reply),
            ("context_does_not_change_router_classification", test_context_does_not_change_router_classification),
            ("empty_state_context_is_none_echo", test_empty_state_context_is_none_echo),
        ],
        "Conversation state",
    )
