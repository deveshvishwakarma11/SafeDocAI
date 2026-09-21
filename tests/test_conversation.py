"""
Conversation intent layer tests (plain-Python runner, no pytest).
=================================================================

Covers the Phase 10.1 local conversation layer:

* every required conversational phrase is recognized with the right
  intent, in English / Hinglish / Hindi;
* responses are localized per detected language;
* document-scope queries are NEVER hijacked (hijack guard);
* general/unsupported queries do NOT match (they get the scope reply
  from the answer engine instead);
* matching is deterministic, microsecond-fast and whole-phrase.

ISOLATION: this layer is pure string logic -- no SQLite, no Chroma, no
Ollama, no network. The real data/safedoc.db is never touched (the
harness guard still verifies that).

Run:  PYTHONIOENCODING=utf-8 python tests/test_conversation.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_project_root = Path(__file__).resolve().parents[1]
for entry in (str(_project_root), str(_project_root / "src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from _harness import run_tests  # noqa: E402

from conversation import (  # noqa: E402
    INTENT_CAPABILITIES,
    INTENT_FAREWELL,
    INTENT_GREETING,
    INTENT_HOW_ARE_YOU,
    INTENT_THANKS,
    check_conversation_intent,
    out_of_scope_response,
)
from language import LANG_ENGLISH, LANG_HINGLISH, LANG_HINDI  # noqa: E402


# ---------------------------------------------------------------------------
# 1. Greetings
# ---------------------------------------------------------------------------


def test_greetings_english() -> None:
    for query in ("hello", "hello there", "Hello!", "Hi", "hey"):
        result = check_conversation_intent(query)
        assert result is not None, query
        assert result["intent"] == INTENT_GREETING, (query, result)


def test_greetings_hinglish_hindi() -> None:
    for query in ("namaste", "namaskar", "Namaste ji"):
        result = check_conversation_intent(query)
        assert result is not None, query
        assert result["intent"] == INTENT_GREETING, (query, result)


def test_greeting_hindi_script_localized() -> None:
    result = check_conversation_intent("नमस्ते")
    assert result is not None
    assert result["intent"] == INTENT_GREETING
    assert result["language"] == LANG_HINDI
    assert "SafeDocAI" in result["response"]


def test_greeting_response_localized_english() -> None:
    result = check_conversation_intent("hello")
    assert result is not None
    assert result["language"] == LANG_ENGLISH
    assert "SafeDocAI" in result["response"]


# ---------------------------------------------------------------------------
# 2. How-are-you / thanks / capabilities / farewell
# ---------------------------------------------------------------------------


def test_how_are_you_en_hinglish() -> None:
    for query in ("how are you?", "how are you", "kaise ho?", "kya haal hai?"):
        result = check_conversation_intent(query)
        assert result is not None, query
        assert result["intent"] == INTENT_HOW_ARE_YOU, (query, result)


def test_how_are_you_hinglish_response_localized() -> None:
    result = check_conversation_intent("kaise ho?")
    assert result is not None
    assert result["language"] == LANG_HINGLISH
    assert "theek" in result["response"].lower()


def test_thanks_variants() -> None:
    for query in ("thanks", "thank you", "shukriya", "dhanyavaad", "Thank you so much"):
        result = check_conversation_intent(query)
        assert result is not None, query
        assert result["intent"] == INTENT_THANKS, (query, result)


def test_thanks_hinglish_response_localized() -> None:
    result = check_conversation_intent("shukriya")
    assert result is not None
    assert result["language"] == LANG_HINGLISH
    assert "documents" in result["response"].lower()


def test_capabilities_and_help() -> None:
    for query in (
        "what can you do?", "what can you help with?", "help",
        "kya kar sakte ho?", "tum kya kar sakte ho?",
    ):
        result = check_conversation_intent(query)
        assert result is not None, query
        assert result["intent"] == INTENT_CAPABILITIES, (query, result)


def test_capabilities_response_explains_scope() -> None:
    result = check_conversation_intent("what can you do?")
    assert result is not None
    assert "documents" in result["response"].lower()
    assert "roll number" in result["response"].lower()


def test_farewells() -> None:
    for query in ("bye", "goodbye", "good bye", "alvida", "bye bye"):
        result = check_conversation_intent(query)
        assert result is not None, query
        assert result["intent"] == INTENT_FAREWELL, (query, result)


def test_informal_elongation_still_matches() -> None:
    result = check_conversation_intent("hiiii")
    assert result is not None
    assert result["intent"] == INTENT_GREETING


# ---------------------------------------------------------------------------
# 3. Document queries must NEVER be hijacked
# ---------------------------------------------------------------------------


def test_document_queries_not_conversation() -> None:
    for query in (
        "What is my roll number?",
        "roll number",
        "What is my date of birth?",
        "meri application form ki important details batao",
        "What does my railway ticket contain?",
        "hey what is my roll number",
        "thanks, what is my dob?",
    ):
        result = check_conversation_intent(query)
        assert result is None, f"hijacked a document query: {query!r} -> {result}"


def test_mixed_conversation_with_document_signal_not_hijacked() -> None:
    """Thanks/help followed by a document question stays a doc query."""

    for query in (
        "thanks, what is my roll number?",
        "hello, meri application number kya hai?",
        "help me find my transaction id",
    ):
        result = check_conversation_intent(query)
        assert result is None, f"hijacked: {query!r} -> {result}"


# ---------------------------------------------------------------------------
# 4. Unsupported/general queries do not match (scope reply handles them)
# ---------------------------------------------------------------------------


def test_general_queries_not_conversation() -> None:
    for query in ("weather today", "2 + 2", "latest cricket score", "tell me a joke"):
        result = check_conversation_intent(query)
        assert result is None, f"matched a general query: {query!r} -> {result}"


def test_content_followup_not_conversation() -> None:
    """A conversational opener followed by real content must not match."""

    for query in ("hello darkness", "help me with weather forecasts", "thanks for the cricket score"):
        result = check_conversation_intent(query)
        assert result is None, f"matched: {query!r} -> {result}"


def test_empty_and_whitespace_queries() -> None:
    assert check_conversation_intent("") is None
    assert check_conversation_intent("   ") is None
    assert check_conversation_intent("???") is None


# ---------------------------------------------------------------------------
# 5. Out-of-scope reply (deterministic, localized, honest scope)
# ---------------------------------------------------------------------------


def test_out_of_scope_response_localized() -> None:
    english = out_of_scope_response(LANG_ENGLISH)
    hinglish = out_of_scope_response(LANG_HINGLISH)
    hindi = out_of_scope_response(LANG_HINDI)

    assert "documents" in english.lower()
    assert "roll number" in english.lower()
    assert "documents" in hinglish.lower()
    assert "roll number" in hinglish.lower()
    assert "documents" in hindi.lower()
    assert hindi != english  # Devanagari wording actually differs


def test_out_of_scope_response_defaults_to_english() -> None:
    assert out_of_scope_response(None) == out_of_scope_response(LANG_ENGLISH)
    assert out_of_scope_response("unknown-language") == out_of_scope_response(
        LANG_ENGLISH
    )


# ---------------------------------------------------------------------------
# 6. Determinism + speed
# ---------------------------------------------------------------------------


def test_deterministic_across_calls() -> None:
    checks = ["hello", "kaise ho?", "what can you do?", "thanks, what is my dob?"]
    first = [check_conversation_intent(c) for c in checks]
    second = [check_conversation_intent(c) for c in checks]
    assert first == second, "conversation detection must be deterministic"


def test_microsecond_fast() -> None:
    queries = ["hello", "thanks", "what can you do?", "bye", "kaise ho?"]
    started = time.perf_counter()
    for _ in range(200):
        for query in queries:
            check_conversation_intent(query)
    per_call_ms = (time.perf_counter() - started) * 1000 / 1000
    assert per_call_ms < 1.0, f"conversation check too slow: {per_call_ms:.4f} ms"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def main() -> int:
    tests = sorted(
        (name, fn)
        for name, fn in globals().items()
        if name.startswith("test_") and callable(fn)
    )
    run_tests(tests, "Conversation layer")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
