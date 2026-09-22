"""
Unified Request Understanding tests (Step 3, plain-Python runner).
==================================================================

Covers the approved Step 3 contract:

 1. Exact field identification          12. Ambiguity detection
 2. Field synonym normalization         13. No-guess behavior
 3. Document target detection           14. Out-of-scope detection
 4. Intent normalization                15. Conversation detection
 5. Operation detection                 16. English queries
 6. Summary detection                   17. Hindi queries
 7. Completeness detection              18. Hinglish queries
 8. Obvious semester parsing            19. Existing query compatibility
 9. Obvious year parsing                20. Optional context compatibility
10. Conversation-context interpretation 21. Malformed/empty input behavior
11. Pronoun/follow-up interpretation    22. Schema stability

The layer is deterministic: no LLM, no network, no filesystem writes.
Real-data behavior is validated separately (read-only).
"""

from __future__ import annotations

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parents[1]
for _entry in (str(_project_root), str(_project_root / "src")):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

from src.request_understanding import (
    COMPLETENESS_FULL_SUMMARY,
    COMPLETENESS_IMPORTANT_DETAILS,
    COMPLETENESS_MULTI_FIELD,
    COMPLETENESS_SINGLE_VALUE,
    INTENT_CONVERSATION,
    INTENT_DOCUMENT_INFORMATION,
    INTENT_OUT_OF_SCOPE,
    OPERATION_GET_VALUE,
    OPERATION_RETRIEVE_DETAILS,
    OPERATION_SUMMARY,
    StructuredRequest,
    understand_request,
)


def _d(query, context=None):
    return understand_request(query, context).to_dict()


# ---------------------------------------------------------------------------
# 1. Exact field identification
# ---------------------------------------------------------------------------

def test_exact_field_roll_number():
    d = _d("mera roll number kya hai")
    assert d["field"] == "roll"
    assert d["operation"] == OPERATION_GET_VALUE
    assert d["completeness"] == COMPLETENESS_SINGLE_VALUE
    assert d["intent"] == INTENT_DOCUMENT_INFORMATION


def test_exact_field_pnr():
    assert _d("PNR kya hai")["field"] == "pnr"
    assert _d("mera PNR kya hai")["field"] == "pnr"


def test_exact_field_enrollment_id():
    d = _d("enrollment id batao")
    assert d["field"] == "enrollment_id"


# ---------------------------------------------------------------------------
# 2. Field synonym normalization
# ---------------------------------------------------------------------------

def test_synonym_pnr_number_variants():
    for q in ("pnr number batao", "PNR No", "pnr"):
        assert _d(q)["field"] == "pnr", q


def test_synonym_enrollment_variants():
    for q in ("enrollment no batao", "enrolment id batao",
              "enrollment number", "enrolment no"):
        assert _d(q)["field"] == "enrollment_id", q


def test_synonym_sgpa_and_semester_gpa():
    assert _d("SGPA batao")["field"] == "sgpa"
    assert _d("semester GPA kya hai")["field"] == "sgpa"


def test_synonym_roll_variants():
    for q in ("roll no", "roll number", "mera roll batao"):
        assert _d(q)["field"] == "roll", q


def test_synonym_mobile_and_email():
    assert _d("mobile number batao")["field"] == "phone"
    assert _d("email id kya hai")["field"] == "email"


def test_synonym_transaction_id():
    assert _d("transaction id batao")["field"] == "transaction_id"


def test_fare_field():
    assert _d("ticket ka fare batao")["field"] == "fare"


# ---------------------------------------------------------------------------
# 3. Document target detection
# ---------------------------------------------------------------------------

def test_target_railway_ticket():
    assert _d("ticket ki details batao")["target_document"] == "railway_ticket"
    assert _d("mera railway ticket")["target_document"] == "railway_ticket"


def test_target_application_form():
    assert _d("meri application ka detail batao")["target_document"] == "application_form"
    assert _d("application form ki details")["target_document"] == "application_form"


def test_target_marksheet():
    assert _d("marksheet ka SGPA")["target_document"] == "marksheet"


def test_target_not_invented_without_document_word():
    assert _d("mera roll number kya hai")["target_document"] is None


# ---------------------------------------------------------------------------
# 4. Intent normalization
# ---------------------------------------------------------------------------

def test_intent_document_information():
    assert _d("mera roll number kya hai")["intent"] == INTENT_DOCUMENT_INFORMATION


def test_intent_conversation_greeting():
    assert _d("hello")["intent"] == INTENT_CONVERSATION


def test_intent_conversation_capabilities():
    assert _d("tum kya kya kar sakte ho")["intent"] == INTENT_CONVERSATION


def test_intent_out_of_scope():
    assert _d("weather today")["intent"] == INTENT_OUT_OF_SCOPE


# ---------------------------------------------------------------------------
# 5. Operation detection
# ---------------------------------------------------------------------------

def test_operation_get_value():
    assert _d("PNR kya hai")["operation"] == OPERATION_GET_VALUE


def test_operation_retrieve_details():
    assert _d("ticket ki details batao")["operation"] == OPERATION_RETRIEVE_DETAILS


def test_operation_summary():
    d = _d("meri application ki important details batao")
    assert d["operation"] == OPERATION_SUMMARY
    assert d["completeness"] == COMPLETENESS_IMPORTANT_DETAILS


# ---------------------------------------------------------------------------
# 6. Summary detection
# ---------------------------------------------------------------------------

def test_summary_important_details():
    d = _d("meri application ki important details batao")
    assert d["operation"] == OPERATION_SUMMARY
    assert d["target_document"] == "application_form"


def test_summary_english_cue():
    assert _d("summarize my document")["operation"] == OPERATION_SUMMARY


# ---------------------------------------------------------------------------
# 7. Completeness detection
# ---------------------------------------------------------------------------

def test_completeness_single_value():
    assert _d("roll number")["completeness"] == COMPLETENESS_SINGLE_VALUE


def test_completeness_multi_field():
    assert _d("ticket ki details batao")["completeness"] == COMPLETENESS_MULTI_FIELD


def test_completeness_full_summary():
    d = _d("poora document samjhao")
    assert d["completeness"] == COMPLETENESS_FULL_SUMMARY
    assert d["operation"] == OPERATION_SUMMARY


# ---------------------------------------------------------------------------
# 8. Obvious semester parsing
# ---------------------------------------------------------------------------

def test_semester_ordinal_forms():
    for q in ("4th sem ka SGPA batao", "4 sem ka SGPA", "semester 4 sgpa",
              "sem 4 ka result"):
        assert _d(q)["constraints"].get("semester") == 4, q


def test_no_semester_constraint_when_absent():
    assert "semester" not in _d("SGPA batao")["constraints"]


# ---------------------------------------------------------------------------
# 9. Obvious year parsing
# ---------------------------------------------------------------------------

def test_year_constraint():
    assert _d("2025 ka result")["constraints"].get("year") == 2025


def test_month_constraint():
    assert _d("august wali ticket")["constraints"].get("month") == 8


def test_constraints_never_fabricated():
    assert _d("roll number batao")["constraints"] == {}


# ---------------------------------------------------------------------------
# 10. Conversation-context interpretation
# ---------------------------------------------------------------------------

def test_follow_up_aur_fare_with_ticket_context():
    context = {
        "turns": [{"user_query": "ticket ki details batao",
                   "assistant_response": "...", "language": "hinglish"}],
        "last_query": "ticket ki details batao",
        "last_response": "...",
        "last_language": "hinglish",
    }
    d = _d("aur fare?", context)
    assert d["is_follow_up"] is True
    assert d["context_reference"] == "previous_turn"
    assert d["target_document"] == "railway_ticket"  # inherited, read-only
    assert d["field"] == "fare"


def test_follow_up_without_context_stays_unresolved_not_guessed():
    d = _d("aur fare?")
    assert d["is_follow_up"] is True
    assert d["target_document"] is None
    assert d["field"] == "fare"  # explicit field survives
    assert d["ambiguity"] is False  # field is unambiguous


def test_context_dict_is_read_only():
    context = {
        "turns": [{"user_query": "ticket ki details batao",
                   "assistant_response": "...", "language": "hinglish"}],
        "last_query": "ticket ki details batao",
        "last_response": None,
        "last_language": "hinglish",
    }
    snapshot = repr(context)
    _d("aur fare?", context)
    assert repr(context) == snapshot  # never mutated


# ---------------------------------------------------------------------------
# 11. Pronoun/follow-up interpretation
# ---------------------------------------------------------------------------

def test_pronoun_uska_with_context():
    context = {"last_query": "ticket ki details batao"}
    d = _d("uska number batao", context)
    assert d["is_follow_up"] is True
    assert d["target_document"] == "railway_ticket"
    assert d["field"] is None
    assert d["ambiguity"] is True  # "number" stays ambiguous (Step 4 decides)


def test_same_ticket_reference():
    context = {"last_query": "ticket ka fare batao"}
    d = _d("same ticket ka PNR", context)
    assert d["target_document"] == "railway_ticket"
    assert d["field"] == "pnr"


# ---------------------------------------------------------------------------
# 12. Ambiguity detection
# ---------------------------------------------------------------------------

def test_ambiguous_number_query():
    d = _d("4 sem ka number batao")
    assert d["field"] is None
    assert d["ambiguity"] is True
    assert d["clarification_needed"] is True
    assert d["candidate_fields"], "candidate fields must be reported"
    assert "sgpa" not in d["candidate_fields"]  # number != sgpa guess


def test_ambiguous_bare_number():
    d = _d("number batao")
    assert d["ambiguity"] is True
    assert d["field"] is None


def test_ambiguous_result_query():
    d = _d("mera result")
    assert d["ambiguity"] is True
    assert set(d["candidate_fields"]) == {"sgpa", "marks", "total_marks"}


def test_qualified_number_not_ambiguous():
    assert _d("application number batao")["ambiguity"] is False
    assert _d("form ka number batao")["ambiguity"] is True  # form + bare number


def test_resolved_details_not_ambiguous():
    assert _d("ticket ki details batao")["ambiguity"] is False


# ---------------------------------------------------------------------------
# 13. No-guess behavior
# ---------------------------------------------------------------------------

def test_ambiguous_queries_never_pick_a_field():
    for q in ("number batao", "4 sem ka number batao", "form ka number batao",
              "mera result", "result ki detail de"):
        d = _d(q)
        if d["ambiguity"]:
            assert d["field"] is None, f"{q} guessed field {d['field']!r}"


# ---------------------------------------------------------------------------
# 14. Out-of-scope detection
# ---------------------------------------------------------------------------

def test_weather_out_of_scope():
    d = _d("weather today")
    assert d["intent"] == INTENT_OUT_OF_SCOPE
    assert d["operation"] is None


def test_document_words_do_not_hijack_out_of_scope():
    d = _d("tell me a joke about documents")
    assert d["intent"] == INTENT_OUT_OF_SCOPE


# ---------------------------------------------------------------------------
# 15. Conversation detection
# ---------------------------------------------------------------------------

def test_greetings_and_thanks():
    assert _d("hello")["intent"] == INTENT_CONVERSATION
    assert _d("thank you")["intent"] == INTENT_CONVERSATION
    assert _d("dhanyavaad")["intent"] == INTENT_CONVERSATION


# ---------------------------------------------------------------------------
# 16-18. English / Hindi / Hinglish
# ---------------------------------------------------------------------------

def test_english_query():
    d = _d("what is my roll number")
    assert d["language"] == "english"
    assert d["field"] == "roll"
    assert d["intent"] == INTENT_DOCUMENT_INFORMATION


def test_hindi_query():
    d = _d("मेरा रोल नंबर बताओ")
    assert d["language"] == "hindi"
    assert d["intent"] == INTENT_DOCUMENT_INFORMATION


def test_hinglish_query():
    d = _d("mera roll number kya hai")
    assert d["language"] == "hinglish"
    assert d["constraints"].get("owner") == "self"


def test_language_uses_existing_detector():
    from src.language import detect_language
    for q in ("hello", "weather today", "mera PNR kya hai",
              "मेरा रोल नंबर बताओ", "roll number"):
        assert _d(q)["language"] == detect_language(q), q


# ---------------------------------------------------------------------------
# 19. Existing query compatibility (answer engine unchanged)
# ---------------------------------------------------------------------------

def _sandbox_answer(query):
    import tempfile
    from pathlib import Path
    import storage_engine
    sandbox = Path(tempfile.mkdtemp(prefix="step3_test_"))
    storage_engine.DATA_DIR = sandbox
    storage_engine.DB_PATH = sandbox / "safedoc.db"
    storage_engine.CHROMA_PATH = sandbox / "chroma_db"
    storage_engine._chroma_client = None
    storage_engine._chroma_available = None

    from src.answer_engine import answer_query
    return answer_query(query)


def test_answer_engine_classification_unchanged():
    out = _sandbox_answer("weather today")
    assert out["classification"] == "rejected"
    assert out["request_understanding"].intent == INTENT_OUT_OF_SCOPE

    conv = _sandbox_answer("hello")
    assert conv["classification"] == "conversation"
    assert conv["request_understanding"].intent == INTENT_CONVERSATION


def test_answer_engine_attaches_structured_request():
    out = _sandbox_answer("weather today")
    ru = out["request_understanding"]
    # Cross-module isinstance is unreliable under the repo's dual-path
    # imports; verify the stable SCHEMA instead.
    assert type(ru).__name__ == "StructuredRequest"
    assert ru.original_query == "weather today"


def test_all_real_example_queries_produce_requests():
    examples = [
        "mera roll number kya hai", "roll number", "4th sem ka SGPA batao",
        "4 sem ka number batao", "4 sem ka result batao", "result ki detail de",
        "mera result", "meri application ka detail batao",
        "application number batao", "form ka number batao",
        "ticket ki details batao", "PNR kya hai", "enrollment id batao",
        "details batao", "weather today", "tum kya kya kar sakte ho",
    ]
    for q in examples:
        request = understand_request(q)
        assert isinstance(request, StructuredRequest), q
        assert request.intent in (
            INTENT_DOCUMENT_INFORMATION, INTENT_CONVERSATION, INTENT_OUT_OF_SCOPE,
        ), q


# ---------------------------------------------------------------------------
# 20. Optional context compatibility
# ---------------------------------------------------------------------------

def test_context_optional_and_none_equivalent():
    q = "roll number"
    assert _d(q) == _d(q, None)


def test_context_accepts_step1_shape():
    context = {
        "turns": [{"user_query": "mera PNR kya hai", "assistant_response":
                   "4143027140", "language": "hinglish"}],
        "last_query": "mera PNR kya hai",
        "last_response": "4143027140",
        "last_language": "hinglish",
    }
    d = _d("aur fare?", context)
    assert d["is_follow_up"] is True
    assert d["field"] == "fare"


# ---------------------------------------------------------------------------
# 21. Malformed/empty input behavior
# ---------------------------------------------------------------------------

def test_empty_and_whitespace_queries():
    for q in ("", "   "):
        d = _d(q)
        assert d["intent"] == INTENT_OUT_OF_SCOPE
        assert d["reason"] == "empty_or_malformed_query"


def test_none_and_non_string_queries_never_raise():
    for bad in (None, 123, ["x"], {"a": 1}):
        d = _d(bad)  # must not raise
        assert d["intent"] == INTENT_OUT_OF_SCOPE


# ---------------------------------------------------------------------------
# 22. Schema stability
# ---------------------------------------------------------------------------

def test_schema_keys_and_types():
    d = _d("4th sem ka SGPA batao")
    expected_keys = {
        "original_query", "language", "intent", "operation", "target_document",
        "field", "candidate_fields", "constraints", "completeness",
        "ambiguity", "clarification_needed", "context_reference",
        "is_follow_up", "confidence", "reason",
    }
    assert expected_keys <= set(d)
    assert isinstance(d["candidate_fields"], list)
    assert isinstance(d["constraints"], dict)
    assert isinstance(d["ambiguity"], bool)
    assert isinstance(d["clarification_needed"], bool)
    assert isinstance(d["confidence"], float)


def test_schema_json_safe_and_frozen():
    import dataclasses
    import json
    request = understand_request("4th sem ka SGPA batao")
    encoded = json.dumps(request.to_dict())  # must not raise
    assert "sgpa" in encoded
    assert dataclasses.is_dataclass(request)
    with __import__("contextlib").suppress(Exception):
        # frozen dataclasses raise on normal attribute assignment
        request.field = "hacked"
    assert request.field == "sgpa", "StructuredRequest must be immutable"


def test_deterministic_repeats():
    for q in ("4 sem ka number batao", "ticket ki details batao", "mera result"):
        assert _d(q) == _d(q), q


def test_clarification_flag_matches_ambiguity():
    for q in ("number batao", "roll number", "weather today", "hello"):
        d = _d(q)
        assert d["clarification_needed"] == d["ambiguity"], q


# ---------------------------------------------------------------------------

def run_all() -> None:
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_") and callable(value)]

    failed: list[str] = []

    for test in tests:
        try:
            test()
        except AssertionError as exc:
            failed.append(f"{test.__name__}: {exc}")
        except Exception as exc:
            failed.append(f"{test.__name__}: RAISED {type(exc).__name__}: {exc}")

    if failed:
        print("Request understanding checks FAILED:")
        for line in failed:
            print(" -", line)
        raise SystemExit(1)

    print(f"Request understanding checks PASSED: {len(tests)}")


if __name__ == "__main__":
    run_all()
