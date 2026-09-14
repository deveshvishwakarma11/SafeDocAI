"""
Phase 5 Answer Engine test suite (plain-Python runner, no pytest).

ISOLATION: storage paths are redirected into a temp directory for the whole
run, so the REAL data/safedoc.db and data/chroma_db are never touched.
A post-run guard re-checks the real DB's size/mtime/sha256 and fails if it
changed. The temp Chroma store is seeded with the REAL local embedding
model (all-MiniLM-L6-v2). The LLM itself is MOCKED at the module seam
(``llm_fn`` / ``generate_with_meta``) inside unit tests - no Ollama call
is required to run this suite.
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

import storage_engine  # the SAME module instance the router/engine bind to

import answer_engine
from answer_engine import (
    INSUFFICIENT_CONTEXT_MESSAGE,
    build_answer_from_exact,
    build_semantic_prompt,
    parse_semantic_response,
    validate_grounding,
)
from query_router import route_query
from storage_engine import ingest_into_sqlite


# ---------------------------------------------------------------------------
# Isolation: redirect storage into a temp directory
# ---------------------------------------------------------------------------

_TEMP_DIR: Path | None = None
_ORIGINALS: dict[str, Path] = {}


def _redirect_storage_to_temp() -> Path:
    global _TEMP_DIR
    if _TEMP_DIR is not None:
        return _TEMP_DIR

    _TEMP_DIR = Path(tempfile.mkdtemp(prefix="safedocai-answer-tests-"))
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


def _real_db_state() -> tuple[int, int, str] | None:
    real_db = _ORIGINALS["DB_PATH"]
    if not real_db.exists():
        return None
    data = real_db.read_bytes()
    return (len(data), real_db.stat().st_mtime_ns, hashlib.sha256(data).hexdigest())


# ---------------------------------------------------------------------------
# Seed data (fixture content, isolated temp stores only)
# ---------------------------------------------------------------------------

_SEED_DOCS = [
    {
        "file_name": "marksheet.pdf",
        "file_type": "PDF",
        "status": "success",
        "raw_text": (
            "Consolidated Marksheet. Semester 1 SGPA 6.09 and Semester 2 "
            "SGPA 6.05. Student TEST USER roll number 12345678 born on "
            "06-06-2006. The marksheet covers semester wise subject marks "
            "and cumulative performance details."
        ),
        "extracted_entities": {
            "sgpa": ["6.09", "6.05"],
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
            "course B.Tech CSE, session 2024 to 2028. The card contains "
            "the student photograph, course branch and session validity."
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

    for record in _SEED_DOCS:
        parsed = dict(record)
        parsed["file_path"] = str((_TEMP_DIR / record["file_name"]).resolve())
        record["doc_id"] = ingest_into_sqlite(parsed, Path("data/output/seed.json"))

    collection = storage_engine.get_chroma_collection()
    model = storage_engine.get_embedding_model()

    for record in _SEED_DOCS:
        chunks = storage_engine.chunk_text(record["raw_text"])
        embeddings = model.encode(
            chunks, batch_size=16, show_progress_bar=False,
            normalize_embeddings=True,
        )
        ids = [f"doc_{record['doc_id']}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "document_id": str(record["doc_id"]),
                "file_name": record["file_name"],
                "chunk_id": i,
            }
            for i in range(len(chunks))
        ]
        collection.upsert(
            ids=ids, documents=chunks,
            embeddings=embeddings.tolist(), metadatas=metadatas,
        )

    _DOCS_SEEDED = True


def _doc(file_name: str) -> dict:
    for record in _SEED_DOCS:
        if record["file_name"] == file_name:
            return record
    raise AssertionError(f"unknown seed doc {file_name}")


# ---------------------------------------------------------------------------
# LLM mock helpers (transport seam only - never real data)
# ---------------------------------------------------------------------------


class MockLLM:
    """Records prompts and returns canned JSON responses."""

    def __init__(self, payload: str) -> None:
        self.payload = payload
        self.prompts: list[str] = []

    def __call__(self, question: str, chunks: list[dict]) -> dict | None:
        self.prompts.append(build_semantic_prompt(question, chunks))
        context_text = "\n\n".join(str(c.get("text", "")) for c in chunks)
        return parse_semantic_response(self.payload, chunks, context_text)


def _chunks_fixture() -> list[dict]:
    return [
        {
            "source": "chroma.safedoc_documents",
            "document_id": 8,
            "file_name": "ApplicationForm.pdf",
            "chunk_id": 0,
            "text": (
                "Application Form Enrollment ID M103B71 for DEVESH "
                "VISHWAKARMA born 06/06/2006, exam cities Gorakhpur, "
                "Prayagraj and Kanpur."
            ),
            "distance": 0.9,
            "useful": True,
        },
        {
            "source": "chroma.safedoc_documents",
            "document_id": 7,
            "file_name": "4143027140.pdf",
            "chunk_id": 2,
            "text": "Electronic reservation ticket PNR 4143027140 for train 03252.",
            "distance": 1.1,
            "useful": True,
        },
    ]


# ---------------------------------------------------------------------------
# Grounding validation tests
# ---------------------------------------------------------------------------


def test_grounding_accepts_supported_values() -> None:
    result = validate_grounding(
        "The SGPA was 6.09 and 6.05 in 2024.",
        "Semester 1 SGPA 6.09 and Semester 2 SGPA 6.05, session 2024.",
    )
    assert result["grounded"] is True
    assert result["unsupported_numbers"] == []


def test_grounding_rejects_fabricated_numbers() -> None:
    result = validate_grounding(
        "The SGPA was 9.99.",
        "Semester 1 SGPA 6.09.",
    )
    assert result["grounded"] is False
    assert "9.99" in result["unsupported_numbers"]


def test_grounding_rejects_fabricated_identifiers() -> None:
    result = validate_grounding(
        "Enrollment ID is XZ99999 and born 06/06/2006.",
        "Enrollment ID M103B71 for DEVESH born 06/06/2006.",
    )
    assert result["grounded"] is False


def test_grounding_comma_and_case_equivalence() -> None:
    result = validate_grounding(
        "total marks were 1,269 out of 1900.",
        "total_marks 1269/1900",
    )
    assert result["grounded"] is True


def test_model_confidence_cannot_override_grounding() -> None:
    chunks = _chunks_fixture()
    context_text = "\n\n".join(c["text"] for c in chunks)
    response = (
        '{"answer": "The candidate scored CGPA 9.87.", '
        '"sources": [{"document_id": 8, "file_name": "ApplicationForm.pdf"}], '
        '"confidence": "high"}'
    )
    parsed = parse_semantic_response(response, chunks, context_text)
    assert parsed is None, "high self-confidence must not bypass grounding"


# ---------------------------------------------------------------------------
# EXACT answer tests (deterministic, no LLM)
# ---------------------------------------------------------------------------


def test_exact_single_field_answer_deterministic() -> None:
    route_result = {
        "route": "exact",
        "fallback": {"occurred": False, "reason": None},
        "results": [
            {
                "document_id": 2,
                "file_name": "Devesh 4th sem.pdf",
                "file_path": "C:/data/Devesh 4th sem.pdf",
                "field_name": "roll",
                "field_value": "2407510100067",
            }
        ],
    }
    payload = build_answer_from_exact(route_result)
    assert payload["answer"] == "roll: 2407510100067"
    assert payload["llm_called"] is False
    assert payload["grounded"] is True
    assert payload["sources"][0]["field_value"] == "2407510100067"
    assert payload["retrieval_route"] == "exact"


def test_exact_repeated_values_all_preserved() -> None:
    route_result = {
        "route": "exact",
        "fallback": {"occurred": False, "reason": None},
        "results": [
            {"document_id": 1, "file_name": "a.pdf", "file_path": "x",
             "field_name": "sgpa", "field_value": "6.09"},
            {"document_id": 1, "file_name": "a.pdf", "file_path": "x",
             "field_name": "sgpa", "field_value": "6.05"},
            {"document_id": 2, "file_name": "b.pdf", "file_path": "y",
             "field_name": "sgpa", "field_value": "6.61"},
        ],
    }
    payload = build_answer_from_exact(route_result)
    assert "6.09" in payload["answer"]
    assert "6.05" in payload["answer"]
    assert "6.61" in payload["answer"]
    assert len(payload["sources"]) == 3  # never collapsed


def test_exact_empty_results_insufficient_without_llm() -> None:
    payload = build_answer_from_exact(
        {
            "route": "exact",
            "fallback": {"occurred": False, "reason": None},
            "results": [],
        }
    )
    assert payload["answer"] == INSUFFICIENT_CONTEXT_MESSAGE
    assert payload["llm_called"] is False
    assert payload["grounded"] is False
    assert payload["insufficient"] is True


# ---------------------------------------------------------------------------
# SEMANTIC prompt/response tests (mocked LLM transport)
# ---------------------------------------------------------------------------


def test_semantic_prompt_contains_context_and_rules() -> None:
    prompt = build_semantic_prompt("What is this?", _chunks_fixture())
    assert "ONLY from the supplied context" in prompt
    assert "M103B71" in prompt  # chunk text present
    assert "ApplicationForm.pdf" in prompt
    assert "4143027140.pdf" in prompt
    assert "4143027140" in prompt  # identifier preserved in context


def test_semantic_valid_grounded_response_accepted() -> None:
    chunks = _chunks_fixture()
    context_text = "\n\n".join(c["text"] for c in chunks)
    response = (
        '{"answer": "The enrollment ID is M103B71 for DEVESH VISHWAKARMA.", '
        '"sources": [{"document_id": 8, "file_name": "ApplicationForm.pdf"}], '
        '"confidence": "high"}'
    )
    parsed = parse_semantic_response(response, chunks, context_text)
    assert parsed is not None
    assert parsed["answer"].startswith("The enrollment ID is M103B71")
    assert parsed["model_confidence"] == "high"
    assert parsed["sources"] == [
        {"document_id": 8, "file_name": "ApplicationForm.pdf"}
    ]


def test_semantic_unsupported_claim_rejected() -> None:
    chunks = _chunks_fixture()
    context_text = "\n\n".join(c["text"] for c in chunks)
    response = (
        '{"answer": "The enrollment ID is M103B71 and the fee paid was Rs 5000.", '
        '"sources": [{"document_id": 8, "file_name": "ApplicationForm.pdf"}], '
        '"confidence": "high"}'
    )
    assert parse_semantic_response(response, chunks, context_text) is None


def test_semantic_unknown_source_citation_dropped() -> None:
    chunks = _chunks_fixture()
    context_text = "\n\n".join(c["text"] for c in chunks)
    response = (
        '{"answer": "The PNR is 4143027140 for train 03252.", '
        '"sources": ['
        '{"document_id": 7, "file_name": "4143027140.pdf"}, '
        '{"document_id": 99, "file_name": "made_up.pdf"}], '
        '"confidence": "medium"}'
    )
    parsed = parse_semantic_response(response, chunks, context_text)
    assert parsed is not None
    assert parsed["sources"] == [{"document_id": 7, "file_name": "4143027140.pdf"}]
    assert parsed["unknown_sources"] == [
        {"document_id": 99, "file_name": "made_up.pdf"}
    ]


def test_semantic_invalid_json_handled_safely() -> None:
    chunks = _chunks_fixture()
    context_text = "\n\n".join(c["text"] for c in chunks)
    broken = '{"answer": "The ID is M103B71.", "sources": ['
    assert parse_semantic_response(broken, chunks, context_text) is None
    assert parse_semantic_response(None, chunks, context_text) is None
    assert parse_semantic_response("", chunks, context_text) is None


def test_semantic_truncated_response_rejected() -> None:
    # A response cut off by num_predict: the outer object never closes,
    # so extract_json fails -> payload None -> insufficient. This mirrors
    # what _generate_semantic_answer does after done_reason == "length".
    chunks = _chunks_fixture()
    context_text = "\n\n".join(c["text"] for c in chunks)
    truncated = (
        '{"answer": "The enrollment ID is M103B71 for DEVESH '
        '"sources": [{"document_id": 8, "file_name": "ApplicationForm.pdf"'
    )
    assert parse_semantic_response(truncated, chunks, context_text) is None


def test_semantic_missing_sources_field_handled_safely() -> None:
    chunks = _chunks_fixture()
    context_text = "\n\n".join(c["text"] for c in chunks)
    response = '{"answer": "The enrollment ID is M103B71.", "confidence": "low"}'
    parsed = parse_semantic_response(response, chunks, context_text)
    assert parsed is not None
    assert parsed["sources"] == []
    assert parsed["model_confidence"] == "low"


# ---------------------------------------------------------------------------
# answer_query integration (mocked LLM, real router + temp stores)
# ---------------------------------------------------------------------------


def test_answer_query_exact_no_llm_call() -> None:
    _seed_all()
    mock = MockLLM('{"answer": "should never be used", "sources": []}')

    payload = answer_query.__wrapped__("What is my roll number?") if hasattr(
        answer_engine.answer_query, "__wrapped__"
    ) else None

    result = answer_engine.answer_query(
        "What is my roll number?", llm_fn=mock
    )
    assert result["route" if "route" in result else "retrieval_route"] in ("exact",)
    assert result["llm_called"] is False
    assert mock.prompts == []  # the LLM was never invoked
    assert "12345678" in result["answer"]
    assert result["grounded"] is True
    assert result["sources"]


def test_answer_query_exact_repeated_values_distinct_sources() -> None:
    _seed_all()
    result = answer_engine.answer_query("What is my SGPA?")
    assert result["llm_called"] is False
    values = [s["field_value"] for s in result["sources"]]
    assert values == ["6.09", "6.05"], values
    assert all(s["file_name"] == "marksheet.pdf" for s in result["sources"])


def test_answer_query_semantic_calls_mock_llm() -> None:
    _seed_all()
    mock = MockLLM(
        '{"answer": "The roll number is 12345678 for TEST USER.", '
        '"sources": [{"document_id": '
        + str(_doc("marksheet.pdf")["doc_id"])
        + ', "file_name": "marksheet.pdf"}], "confidence": "high"}'
    )
    result = answer_engine.answer_query(
        "What does the marksheet say about the student?",
        llm_fn=mock,
    )
    assert len(mock.prompts) == 1
    assert result["llm_called"] is True
    assert result["grounded"] is True
    assert "12345678" in result["answer"]
    assert result["retrieval_route"] == "semantic"
    assert result["model_confidence"] == "high"
    assert result["retrieved_chunks"][0]["document_id"] is not None


def test_answer_query_semantic_grounded_without_citations_gets_chunk_sources() -> None:
    _seed_all()
    # Model answers correctly but cites no sources: the grounded answer must
    # still carry provenance (derived from the retrieved chunk context).
    mock = MockLLM(
        '{"answer": "The roll number is 12345678 for TEST USER.", '
        '"sources": [], "confidence": "high"}'
    )
    result = answer_engine.answer_query(
        "What does the marksheet say about the student?",
        llm_fn=mock,
    )
    assert result["grounded"] is True
    assert result["llm_called"] is True
    assert result["sources"], "grounded answer must always carry sources"
    assert result["source_documents"], "source_documents must mirror sources"
    assert result["sources"][0]["file_name"] == "marksheet.pdf"


def test_answer_query_semantic_grounded_false_path() -> None:
    _seed_all()
    # Mock returns a validated-None payload (fabricated number).
    mock = MockLLM(
        '{"answer": "The fee was Rs 99999 exactly.", "sources": [], '
        '"confidence": "high"}'
    )
    result = answer_engine.answer_query(
        "What does the marksheet say about the student?",
        llm_fn=mock,
    )
    assert result["llm_called"] is True
    assert result["grounded"] is False
    assert result["insufficient"] is True
    assert result["answer"] == INSUFFICIENT_CONTEXT_MESSAGE
    assert result["sources"] == []


def test_answer_query_empty_retrieval_insufficient_no_llm() -> None:
    _seed_all()
    mock = MockLLM('{"answer": "nope", "sources": []}')
    result = answer_engine.answer_query(
        "Explain the beautiful quantum harmonica in my papers.",
        llm_fn=mock,
    )
    assert result["grounded"] is False
    assert result["insufficient"] is True
    assert result["sources"] == []
    if result["llm_called"]:
        assert mock.prompts == []  # no chunks -> no LLM call possible


def test_answer_query_fallback_metadata_preserved() -> None:
    _seed_all()
    mock = MockLLM(
        '{"answer": "The enrollment ID is M103B71 for DEVESH VISHWAKARMA.", '
        '"sources": [{"document_id": '
        + str(_doc("marksheet.pdf")["doc_id"])
        + ', "file_name": "marksheet.pdf"}], "confidence": "medium"}'
    )
    # Force a semantic answer that carries the router's fallback record.
    result = answer_engine.answer_query(
        "Summarize the contents of the documents.",
        llm_fn=mock,
    )
    assert "fallback" in result
    assert isinstance(result["fallback"], dict)
    assert "occurred" in result["fallback"]


def test_answer_query_deterministic_for_repeated_exact_queries() -> None:
    _seed_all()
    first = answer_engine.answer_query("What is my roll number?")
    second = answer_engine.answer_query("What is my roll number?")
    strip = lambda r: {k: v for k, v in r.items() if k != "timings_ms"}
    assert strip(first) == strip(second)


def test_provenance_fields_in_every_payload() -> None:
    _seed_all()
    exact = answer_engine.answer_query("What is my roll number?")
    for key in (
        "answer", "sources", "source_documents", "retrieval_route",
        "fallback", "grounded", "llm_called",
    ):
        assert key in exact, (key, exact)
    row = exact["sources"][0]
    for key in ("document_id", "file_name", "file_path", "field_name", "field_value"):
        assert key in row, (key, row)

    mock = MockLLM(
        '{"answer": "The roll number is 12345678.", '
        '"sources": [{"document_id": '
        + str(_doc("marksheet.pdf")["doc_id"])
        + ', "file_name": "marksheet.pdf"}], "confidence": "high"}'
    )
    semantic = answer_engine.answer_query(
        "What does the marksheet say?", llm_fn=mock
    )
    assert semantic["retrieved_chunks"], semantic
    chunk = semantic["retrieved_chunks"][0]
    for key in ("document_id", "file_name", "chunk_id", "distance"):
        assert key in chunk, (key, chunk)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_all() -> None:
    tests = [
        test_grounding_accepts_supported_values,
        test_grounding_rejects_fabricated_numbers,
        test_grounding_rejects_fabricated_identifiers,
        test_grounding_comma_and_case_equivalence,
        test_model_confidence_cannot_override_grounding,
        test_exact_single_field_answer_deterministic,
        test_exact_repeated_values_all_preserved,
        test_exact_empty_results_insufficient_without_llm,
        test_semantic_prompt_contains_context_and_rules,
        test_semantic_valid_grounded_response_accepted,
        test_semantic_unsupported_claim_rejected,
        test_semantic_unknown_source_citation_dropped,
        test_semantic_invalid_json_handled_safely,
        test_semantic_truncated_response_rejected,
        test_semantic_missing_sources_field_handled_safely,
        test_answer_query_exact_no_llm_call,
        test_answer_query_exact_repeated_values_distinct_sources,
        test_answer_query_semantic_calls_mock_llm,
        test_answer_query_semantic_grounded_without_citations_gets_chunk_sources,
        test_answer_query_semantic_grounded_false_path,
        test_answer_query_empty_retrieval_insufficient_no_llm,
        test_answer_query_fallback_metadata_preserved,
        test_answer_query_deterministic_for_repeated_exact_queries,
        test_provenance_fields_in_every_payload,
    ]

    failed: list[str] = []
    for test in tests:
        try:
            test()
        except AssertionError as exc:
            failed.append(f"{test.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failed.append(f"{test.__name__}: RAISED {type(exc).__name__}: {exc}")

    if failed:
        print("Answer engine tests FAILED:")
        for line in failed:
            print(" -", line)
        raise SystemExit(1)

    print(f"Answer engine tests PASSED: {len(tests)}")


def main() -> None:
    real_db_before = _real_db_state()
    try:
        run_all()
    finally:
        real_db_after = _real_db_state()
        _restore_storage()

    if real_db_before is None:
        print("Isolation guard: real data/safedoc.db does not exist (nothing to protect).")
    elif real_db_before != real_db_after:
        print("Isolation guard FAILED: real data/safedoc.db was modified by the test run!")
        raise SystemExit(1)
    else:
        print(
            "Isolation guard OK: real data/safedoc.db untouched "
            f"(size={real_db_after[0]} bytes, sha256={real_db_after[2][:12]}...)."
        )


if __name__ == "__main__":
    main()
