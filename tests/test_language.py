"""
Phase 8 language + Hinglish routing + LLM-discipline test suite.

Plain-Python runner (no pytest), consistent with the other suites.

ISOLATION: storage paths are redirected into a temp directory for the
whole run, so the REAL data/safedoc.db and data/chroma_db are never
touched (a post-run guard re-checks the real DB's sha256). The LLM is
MOCKED at the ``llm_fn`` seam inside unit tests -- no Ollama call is
required and no generation runs in tests.

Covers the Phase 8 spec:
* STEP 2  language detection: English / Hindi / Hinglish / mixed
* STEP 4  exact answers stay LLM-free, language-aware, values verbatim
* STEP 6  Hinglish/Hindi routing (exact stays exact, semantic stays
  semantic), without replacing the deterministic router
* STEP 7  performance behavior (11-13), duplicate guard, answer
  behavior (15-20)
"""

from __future__ import annotations

import atexit
import gc
import hashlib
import shutil
import sys
import tempfile
from pathlib import Path

_project_root = Path(__file__).resolve().parents[1]
for entry in (str(_project_root), str(_project_root / "src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import storage_engine  # the SAME module instance the router binds to

import answer_engine
from answer_engine import (
    INSUFFICIENT_CONTEXT_MESSAGE,
    answer_query,
    build_answer_from_exact,
    build_semantic_prompt,
)
from language import (
    LANG_ENGLISH,
    LANG_HINGLISH,
    LANG_HINDI,
    LANG_UNKNOWN,
    detect_language,
    language_label,
    transliterate_for_matching,
)
from query_router import CLASS_EXACT, CLASS_SEMANTIC, classify_query


# ---------------------------------------------------------------------------
# Isolation: redirect storage into a temp directory
# ---------------------------------------------------------------------------

_TEMP_DIR: Path | None = None
_ORIGINALS: dict[str, Path] = {}


def _redirect_storage_to_temp() -> Path:
    global _TEMP_DIR
    if _TEMP_DIR is not None:
        return _TEMP_DIR

    _TEMP_DIR = Path(tempfile.mkdtemp(prefix="safedocai-language-tests-"))
    _ORIGINALS.update(
        DATA_DIR=storage_engine.DATA_DIR,
        DB_PATH=storage_engine.DB_PATH,
        CHROMA_PATH=storage_engine.CHROMA_PATH,
    )
    storage_engine.DATA_DIR = _TEMP_DIR
    storage_engine.DB_PATH = _TEMP_DIR / "safedoc.db"
    storage_engine.CHROMA_PATH = _TEMP_DIR / "chroma_db"
    storage_engine._chroma_client = None
    storage_engine._chroma_available = None
    return _TEMP_DIR


def _restore_storage() -> None:
    global _TEMP_DIR
    if _TEMP_DIR is None:
        return
    gc.collect()
    shutil.rmtree(_TEMP_DIR, ignore_errors=True)
    storage_engine.DATA_DIR = _ORIGINALS["DATA_DIR"]
    storage_engine.DB_PATH = _ORIGINALS["DB_PATH"]
    storage_engine.CHROMA_PATH = _ORIGINALS["CHROMA_PATH"]
    storage_engine._chroma_client = None
    storage_engine._chroma_available = None
    _TEMP_DIR = None


_redirect_storage_to_temp()
atexit.register(_restore_storage)


def _real_db_sha() -> str | None:
    real_db = _ORIGINALS["DB_PATH"]
    if not real_db.exists():
        return None
    return hashlib.sha256(real_db.read_bytes()).hexdigest()


_REAL_SHA_AT_START = _real_db_sha()


def _assert_real_db_untouched() -> None:
    current = _real_db_sha()
    if _REAL_SHA_AT_START is not None:
        assert current == _REAL_SHA_AT_START, "real DB changed during tests!"


# ---------------------------------------------------------------------------
# Minimal fixture docs (isolated temp stores only)
# ---------------------------------------------------------------------------

from storage_engine import (  # noqa: E402  (after path redirect)
    init_db,
    ingest_into_sqlite,
)
from storage_engine import chunk_text, get_chroma_collection, get_embedding_model  # noqa: E402

_SEED_DOCS = [
    {
        "file_name": "id_card.pdf",
        "file_type": "PDF",
        "status": "success",
        "raw_text": (
            "Student ID Card with photograph. The roll number is 12345678 "
            "and the contact phone number is 9876543210. Date of birth "
            "01-01-1980. The identity card contains student details."
        ),
        "extracted_entities": {
            "roll": ["12345678"],
            "phone": ["9876543210"],
            "dob": ["01-01-1980"],
        },
    },
    {
        "file_name": "ticket.pdf",
        "file_type": "PDF",
        "status": "success",
        "raw_text": (
            "Indian Railways electronic reservation ticket. PNR 8425631974 "
            "train 03252. The ticket contains passenger details and "
            "journey information."
        ),
        "extracted_entities": {
            "pnr": ["8425631974"],
        },
    },
]


_DOCS_SEEDED = False
_DOC_IDS: dict[str, int] = {}


def _seed_all() -> None:
    """Ingest fixture docs into the temp SQLite DB (and Chroma)."""

    global _DOCS_SEEDED
    if _DOCS_SEEDED:
        return

    storage_engine.init_db()

    for record in _SEED_DOCS:
        parsed = dict(record)
        parsed["file_path"] = str((_TEMP_DIR / record["file_name"]).resolve())
        _DOC_IDS[record["file_name"]] = ingest_into_sqlite(
            parsed, Path("data/output/seed.json")
        )

    _DOCS_SEEDED = True


# ---------------------------------------------------------------------------
# Mock LLM (records calls; validates against the supplied context)
# ---------------------------------------------------------------------------


class MockLLM:
    """Mimics _generate_semantic_answer's contract and records invocations."""

    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[str] = []

    def __call__(self, question: str, chunks: list[dict]) -> dict | None:
        self.calls.append(question)
        from answer_engine import parse_semantic_response

        context_text = "\n\n".join(str(c.get("text", "")) for c in chunks)
        return parse_semantic_response(self.response, chunks, context_text)


# ---------------------------------------------------------------------------
# STEP 2: language detection
# ---------------------------------------------------------------------------


def test_language_english() -> None:
    assert detect_language("What is my roll number?") == LANG_ENGLISH


def test_language_hinglish() -> None:
    assert detect_language("Mera roll number kya hai?") == LANG_HINGLISH


def test_language_hindi_devanagari() -> None:
    assert detect_language("इस दस्तावेज़ में क्या लिखा है?") == LANG_HINDI


def test_language_mixed_english_hindi() -> None:
    # Devanagari + English identifier stays "hindi" per the spec example.
    assert detect_language("मेरा enrollment ID क्या है?") == LANG_HINDI


def test_language_roman_hindi() -> None:
    assert detect_language("Mujhe application form ki important details batao") == LANG_HINGLISH


def test_language_english_not_hinglish() -> None:
    # Ordinary English words must not be Hindi markers.
    assert detect_language("Tell me the important information on my ID card.") == LANG_ENGLISH
    assert detect_language("to me") == LANG_ENGLISH
    assert detect_language("hi there") == LANG_ENGLISH


def test_language_unknown_and_labels() -> None:
    assert detect_language("   ") == LANG_UNKNOWN
    assert language_label(LANG_HINGLISH) == "Hinglish"
    assert language_label(LANG_HINDI) == "Hindi"


def test_transliteration_devanagari_to_latin() -> None:
    latin = transliterate_for_matching("मेरा रोल नंबर क्या है?")
    assert "kya" in latin and "hai" in latin and "mera" in latin
    # Non-Devanagari text is returned unchanged.
    assert transliterate_for_matching("What is my roll number?") == "What is my roll number?"


# ---------------------------------------------------------------------------
# STEP 6: routing of language-mixed queries
# ---------------------------------------------------------------------------


def test_routing_hinglish_exact_queries() -> None:
    for query in (
        "Mera roll number kya hai?",
        "Mera phone number kya hai?",
        "Mera enrollment ID kya hai?",
        "meri date of birth kya hai",
    ):
        signals = classify_query(query)
        assert signals["classification"] == CLASS_EXACT, (query, signals)
        assert signals["field_hint"], query


def test_routing_hinglish_semantic_queries() -> None:
    for query in (
        "ID card ke important details batao",
        "Meri application form ki important details batao",
        "my phone number batao",
    ):
        assert classify_query(query)["classification"] == CLASS_SEMANTIC, query
    # "What's in X?" (exploratory) stays semantic even with "kya hai".
    assert classify_query("Meri railway ticket mein kya hai?")["classification"] == CLASS_SEMANTIC


def test_routing_hindi_exact_query() -> None:
    # Devanagari exact query classifies EXACT via transliteration.
    latin = transliterate_for_matching("मेरा रोल नंबर क्या है?")
    assert classify_query(latin)["classification"] == CLASS_EXACT


def test_routing_hindi_semantic_query() -> None:
    latin = transliterate_for_matching("इस दस्तावेज़ में क्या लिखा है?")
    assert classify_query(latin)["classification"] == CLASS_SEMANTIC


def test_routing_english_queries_unchanged() -> None:
    assert classify_query("What is my roll number?")["classification"] == CLASS_EXACT
    assert classify_query("What does my railway ticket contain?")["classification"] == CLASS_SEMANTIC


# ---------------------------------------------------------------------------
# STEP 4: exact answers stay LLM-free and language-aware
# ---------------------------------------------------------------------------


def _exact_result(query: str, rows: list[dict]) -> dict:
    return {
        "query": query,
        "route": "exact",
        "classification": "exact",
        "results": rows,
    }


_ROW = {
    "document_id": 1,
    "file_name": "id_card.pdf",
    "file_path": "x/id_card.pdf",
    "field_name": "roll",
    "field_value": "12345678",
}


def test_exact_answer_english_value_verbatim() -> None:
    payload = build_answer_from_exact(
        _exact_result("What is my roll number?", [_ROW]) | {"field_hint": "roll number"}
    )
    assert payload["language"] == LANG_ENGLISH
    # Phase 9: answer-focused sentence; value verbatim.
    assert payload["answer"] == "Your roll number is 12345678."
    assert payload["llm_called"] is False
    assert payload["grounded"] is True


def test_exact_answer_hinglish_value_verbatim() -> None:
    payload = build_answer_from_exact(
        _exact_result("Mera roll number kya hai?", [_ROW]) | {"field_hint": "roll number"}
    )
    assert payload["language"] == LANG_HINGLISH
    assert payload["answer"] == "Aapka roll number 12345678 hai."
    assert payload["llm_called"] is False


def test_exact_answer_hindi_value_verbatim() -> None:
    payload = build_answer_from_exact(
        _exact_result("मेरा रोल नंबर क्या है?", [_ROW]) | {"field_hint": "roll number"}
    )
    assert payload["language"] == LANG_HINDI
    assert "12345678" in payload["answer"]  # value verbatim
    assert payload["llm_called"] is False


def test_exact_multi_value_hinglish_preserves_all() -> None:
    rows = [
        dict(_ROW),
        dict(_ROW, document_id=2, file_name="ticket.pdf", field_value="9999999999", field_name="phone"),
    ]
    payload = build_answer_from_exact(
        _exact_result("Mera phone number kya hai?", rows) | {"field_hint": "phone number"}
    )
    assert payload["language"] == LANG_HINGLISH
    assert "12345678" in payload["answer"] and "9999999999" in payload["answer"]
    assert "id_card.pdf" in payload["answer"] and "ticket.pdf" in payload["answer"]


def test_exact_insufficient_message_is_localized() -> None:
    payload = build_answer_from_exact(_exact_result("Mera aadhar kya hai?", []))
    assert payload["language"] == LANG_HINGLISH
    assert payload["insufficient"] is True and payload["grounded"] is False
    assert payload["answer"] != INSUFFICIENT_CONTEXT_MESSAGE  # Hinglish variant
    assert "Aapke stored documents" in payload["answer"]


def test_prompt_language_rules() -> None:
    legacy = build_semantic_prompt("q", [])
    assert "- The user asked in" not in legacy  # backward compatible

    hinglish = build_semantic_prompt("q", [], language=LANG_HINGLISH)
    assert "Hinglish" in hinglish and "Do NOT translate document names" in hinglish

    hindi = build_semantic_prompt("q", [], language=LANG_HINDI)
    assert "Devanagari" in hindi


# ---------------------------------------------------------------------------
# STEP 7: performance behavior (LLM discipline) + answer behavior
# ---------------------------------------------------------------------------


def test_exact_query_does_not_call_llm() -> None:
    _seed_all()
    _seed_chroma()
    mock = MockLLM('{"answer": "should never be used", "sources": []}')
    payload = answer_query("Mera roll number kya hai?", llm_fn=mock)
    assert payload["retrieval_route"] == "exact"
    assert mock.calls == [], "exact query MUST NOT call the LLM"
    assert payload["llm_called"] is False
    assert payload["answer"] == "Aapka roll number 12345678 hai."
    assert payload["language"] == LANG_HINGLISH
    assert payload["retrieval_performed"] is True


def test_exact_query_timings_have_no_llm_stage() -> None:
    _seed_all()
    mock = MockLLM('{"answer": "unused", "sources": []}')
    payload = answer_query("Mera roll number kya hai?", llm_fn=mock)
    timings = payload["timings_ms"]
    assert timings["llm_generation"] is None
    assert timings["router"] < 2000  # SQLite path is sub-second
    assert timings["total"] < 2000


def _seed_chroma() -> None:
    """Seed the temp Chroma store once with real local embeddings."""

    global _CHROMA_SEEDED
    if _CHROMA_SEEDED:
        return

    collection = get_chroma_collection()
    model = get_embedding_model()
    for record in _SEED_DOCS:
        doc_id = _DOC_IDS[record["file_name"]]
        chunks = chunk_text(record["raw_text"])
        embeddings = model.encode(
            chunks, batch_size=16, show_progress_bar=False,
            normalize_embeddings=True,
        )
        ids = [f"doc_{doc_id}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "document_id": str(doc_id),
                "file_name": record["file_name"],
                "chunk_id": i,
            }
            for i in range(len(chunks))
        ]
        collection.upsert(
            ids=ids,
            documents=chunks,
            metadatas=metadatas,
            embeddings=embeddings.tolist(),
        )
    _CHROMA_SEEDED = True


_CHROMA_SEEDED = False


def test_semantic_query_reaches_llm() -> None:
    _seed_all()
    _seed_chroma()
    mock = MockLLM(
        '{"answer": "The PNR is 8425631974.", '
        '"sources": [{"document_id": 2, "file_name": "ticket.pdf"}], "confidence": "high"}'
    )
    payload = answer_query("Meri railway ticket mein kya hai?", llm_fn=mock)
    assert payload["retrieval_route"] == "semantic"
    assert len(mock.calls) == 1
    assert payload["llm_called"] is True
    assert payload["grounded"] is True


def test_hinglish_semantic_answer_preserves_identifiers() -> None:
    _seed_all()
    _seed_chroma()
    mock = MockLLM(
        '{"answer": "Ticket mein PNR 8425631974 hai aur train 03252 hai.", '
        '"sources": [{"document_id": 2, "file_name": "ticket.pdf"}], "confidence": "high"}'
    )
    payload = answer_query("Meri railway ticket mein kya hai?", llm_fn=mock)
    assert payload["grounded"] is True
    assert "8425631974" in payload["answer"] and "03252" in payload["answer"]
    assert payload["sources"][0]["file_name"] == "ticket.pdf"


def test_hinglish_semantic_prompt_pinned() -> None:
    """The REAL generation path pins the detected language in the prompt.

    ``llm_fn`` bypasses prompt construction entirely, so this test
    patches the transport seam (``generate_with_meta``) instead and lets
    ``_generate_semantic_answer`` build the genuine prompt.
    """

    _seed_all()
    _seed_chroma()
    captured_prompts: list[str] = []

    def fake_transport(prompt: str, num_predict: int = 0) -> dict:
        captured_prompts.append(prompt)
        return {
            "text": (
                '{"answer": "Ticket mein PNR 8425631974 hai.", '
                '"sources": [{"document_id": %d, "file_name": "ticket.pdf"}], '
                '"confidence": "high"}' % _DOC_IDS["ticket.pdf"]
            ),
            "done_reason": "stop",
        }

    original = answer_engine.generate_with_meta
    answer_engine.generate_with_meta = fake_transport
    try:
        payload = answer_query("Meri railway ticket mein kya hai?")
    finally:
        answer_engine.generate_with_meta = original

    assert len(captured_prompts) == 1
    assert "Hinglish" in captured_prompts[0]
    assert payload["grounded"] is True
    assert "8425631974" in payload["answer"]


def test_duplicate_submission_guard_ui() -> None:
    from ui import _should_record_query

    history = [{"query": "Mera roll number kya hai?", "payload": {}}]
    assert _should_record_query(history, "Mera roll number kya hai?") is False
    assert _should_record_query(history, "Mera phone number kya hai?") is True
    assert _should_record_query([], "anything") is True


def test_semantic_no_useful_result_localized_insufficient() -> None:
    _seed_all()
    _seed_chroma()
    # Query with a field-like phrase that matches nothing anywhere.
    mock = MockLLM('{"answer": "x", "sources": []}')
    payload = answer_query("Mera kutte ka naam kya hai?", llm_fn=mock)
    assert payload["insufficient"] is True and payload["grounded"] is False
    assert mock.calls == []  # no chunks -> no LLM call


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_all() -> None:
    tests = [
        test_language_english,
        test_language_hinglish,
        test_language_hindi_devanagari,
        test_language_mixed_english_hindi,
        test_language_roman_hindi,
        test_language_english_not_hinglish,
        test_language_unknown_and_labels,
        test_transliteration_devanagari_to_latin,
        test_routing_hinglish_exact_queries,
        test_routing_hinglish_semantic_queries,
        test_routing_hindi_exact_query,
        test_routing_hindi_semantic_query,
        test_routing_english_queries_unchanged,
        test_exact_answer_english_value_verbatim,
        test_exact_answer_hinglish_value_verbatim,
        test_exact_answer_hindi_value_verbatim,
        test_exact_multi_value_hinglish_preserves_all,
        test_exact_insufficient_message_is_localized,
        test_prompt_language_rules,
        test_exact_query_does_not_call_llm,
        test_exact_query_timings_have_no_llm_stage,
        test_semantic_query_reaches_llm,
        test_hinglish_semantic_answer_preserves_identifiers,
        test_hinglish_semantic_prompt_pinned,
        test_duplicate_submission_guard_ui,
        test_semantic_no_useful_result_localized_insufficient,
    ]

    failed: list[str] = []
    for test in tests:
        try:
            test()
        except AssertionError as exc:
            failed.append(f"{test.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failed.append(f"{test.__name__}: {type(exc).__name__}: {exc}")

    print(f"\n{len(tests) - len(failed)}/{len(tests)} tests passed.")
    if failed:
        print("FAILURES:")
        for line in failed:
            print(f"  - {line}")
        sys.exit(1)

    _assert_real_db_untouched()
    print("Real DB untouched: OK")


if __name__ == "__main__":
    run_all()
