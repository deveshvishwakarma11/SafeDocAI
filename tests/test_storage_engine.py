"""
Storage engine focused regression checks.

These narrow checks target the Step 2 risks discussed for
src/storage_engine.py:
- decimal-aware chunking
- document mapping by original path
- SQLite + Chroma storage behavior
- duplicate / re-index handling

ISOLATION:
All storage paths (DATA_DIR / DB_PATH / CHROMA_PATH) are redirected into a
temporary directory for the whole run, so executing this suite NEVER creates,
updates, deletes, or leaves records in the real data/safedoc.db or
data/chroma_db. A post-run guard re-checks the real database's size, mtime,
and SHA-256 and fails the suite if either changed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from _harness import redirect_storage_to_temp, run_tests

import src.storage_engine as storage_engine

from src.storage_engine import (
    chunk_text,
    load_parsed_json,
    ingest_into_sqlite,
    store_understanding_results,
    resolve_document_id,
    init_db,
    query_exact,
)


# ---------------------------------------------------------------------------
# Test isolation: redirect storage into a temp directory
# ---------------------------------------------------------------------------

# Redirect immediately at import so no test can ever touch the real DB.
_TEMP_DIR = redirect_storage_to_temp("safedocai-tests-")


def _reset_db() -> None:
    init_db()


# ---------------------------------------------------------------------------
# Chunking: decimals should not be split into numeric fragments
# ---------------------------------------------------------------------------

def test_decimal_values_stay_intact_in_chunks() -> None:
    text = "Total marks obtained: 1269 out of 1900. SGPA: 6.61.\nResult: PASS"
    chunks = chunk_text(text, chunk_size=80, overlap=10)

    joined = "\n".join(chunks)

    assert "1269" in joined
    assert "6.61" in joined
    assert "1900" in joined
    assert "PASS" in joined


def test_large_paragraph_splits_without_losing_numbers() -> None:
    repeated = "Roll No: 12345678\n"
    long_text = "Header line for context.\n" + repeated * 40

    chunks = chunk_text(long_text, chunk_size=60, overlap=12)

    joined = "\n".join(chunks)
    assert "12345678" in joined


# ---------------------------------------------------------------------------
# Document mapping by exact original path
# ---------------------------------------------------------------------------

def test_resolve_document_id_uses_exact_original_path() -> None:
    _reset_db()

    parsed = {
        "file_name": "sample.pdf",
        "file_type": "PDF",
        "status": "success",
        "raw_text": "some text",
        "extracted_entities": {},
        "extracted_metadata": [],
        "file_path": str(Path("data/samples/sample.pdf").resolve()),
    }

    doc_id = ingest_into_sqlite(parsed, Path("data/output/sample.json"))
    assert doc_id > 0

    resolved = resolve_document_id(parsed["file_path"])
    assert resolved == doc_id


def test_resolve_document_id_returns_none_for_missing_path() -> None:
    _reset_db()

    missing_path = str(Path("data/samples/does_not_exist_original.pdf").resolve())
    assert resolve_document_id(missing_path) is None


# ---------------------------------------------------------------------------
# SQLite ingestion and duplicate/re-index handling
# ---------------------------------------------------------------------------

def test_sqlite_ingestion_matches_original_path() -> None:
    _reset_db()

    parsed = {
        "file_name": "sample.pdf",
        "file_type": "PDF",
        "status": "success",
        "raw_text": "sample text",
        "extracted_entities": {
            "roll": ["12345678"],
            "name": ["TEST USER"],
        },
        "file_path": str(Path("data/samples/sample.pdf").resolve()),
    }

    doc_id = ingest_into_sqlite(parsed, Path("data/output/sample.json"))
    assert doc_id > 0

    resolved = resolve_document_id(parsed["file_path"])
    assert resolved == doc_id

    rows = query_exact("roll")
    values = {row["field_value"] for row in rows if row["document_id"] == doc_id}
    assert values == {"12345678"}


def test_reindex_updates_metadata_for_same_original_path() -> None:
    _reset_db()

    original_path = str(Path("data/samples/sample.pdf").resolve())

    first = {
        "file_name": "sample.pdf",
        "file_type": "PDF",
        "status": "success",
        "raw_text": "first version",
        "extracted_entities": {
            "roll": ["11111111"],
        },
        "file_path": original_path,
    }

    second = {
        "file_name": "sample.pdf",
        "file_type": "PDF",
        "status": "success",
        "raw_text": "second version",
        "extracted_entities": {
            "roll": ["22222222"],
            "name": ["NEW USER"],
        },
        "file_path": original_path,
    }

    first_id = ingest_into_sqlite(first, Path("data/output/sample.json"))
    second_id = ingest_into_sqlite(second, Path("data/output/sample.json"))

    assert first_id == second_id

    rows = query_exact("roll")
    values = {row["field_value"] for row in rows if row["document_id"] == first_id}
    assert values == {"22222222"}

    name_rows = query_exact("name")
    name_values = {row["field_value"] for row in name_rows if row["document_id"] == first_id}
    assert name_values == {"NEW USER"}


# ---------------------------------------------------------------------------
# Phase 3 verified storage links by original path
# ---------------------------------------------------------------------------

def test_store_understanding_results_links_by_exact_path() -> None:
    _reset_db()

    original_path = str(Path("data/samples/sample.pdf").resolve())

    parsed = {
        "file_name": "sample.pdf",
        "file_type": "PDF",
        "status": "success",
        "raw_text": "sample text",
        "extracted_entities": {},
        "file_path": original_path,
    }

    doc_id = ingest_into_sqlite(parsed, Path("data/output/sample.json"))
    assert doc_id > 0

    understanding = {
        "document_type": "MARKSHEET",
        "fields": [
            {"key": "roll", "value": "12345678", "evidence_snippet": "12345678", "verified": True},
            {"key": "name", "value": "UNKNOWN", "evidence_snippet": "", "verified": True},
            {"key": "fake", "value": "FAKEVAL", "evidence_snippet": "FAKEVAL", "verified": False},
        ],
    }

    result = store_understanding_results(
        Path("data/understood/sample.json"),
        understanding,
        original_document_path=original_path,
    )

    assert result["status"] == "success"
    assert result["stored"] == 1
    assert result["document_id"] == doc_id

    rows = query_exact("roll")
    matched = [row for row in rows if row["document_id"] == doc_id]
    assert len(matched) == 1
    assert matched[0]["field_value"] == "12345678"

    fake_rows = query_exact("fake")
    assert not any(row["document_id"] == doc_id for row in fake_rows)


def test_store_understanding_results_fails_when_original_path_unknown() -> None:
    _reset_db()

    understanding = {
        "document_type": "UNKNOWN",
        "fields": [
            {"key": "roll", "value": "12345678", "evidence_snippet": "12345678", "verified": True},
        ],
    }

    result = store_understanding_results(
        Path("data/understood/sample.json"),
        understanding,
        original_document_path=str(Path("data/samples/missing_original.pdf").resolve()),
    )

    assert result["status"] == "error"
    assert result["stored"] == 0
    assert "error" in result


# ---------------------------------------------------------------------------
# Frozen storage state for integration-style checks
# ---------------------------------------------------------------------------

def test_load_parsed_json_rejects_missing_file() -> None:
    missing = Path("data/output/missing_file.json")
    try:
        load_parsed_json(missing)
    except FileNotFoundError:
        return
    raise AssertionError("expected FileNotFoundError")


def run_all() -> None:
    tests = [(fn.__name__, fn) for fn in (
        test_decimal_values_stay_intact_in_chunks,
        test_large_paragraph_splits_without_losing_numbers,
        test_resolve_document_id_uses_exact_original_path,
        test_resolve_document_id_returns_none_for_missing_path,
        test_sqlite_ingestion_matches_original_path,
        test_reindex_updates_metadata_for_same_original_path,
        test_store_understanding_results_links_by_exact_path,
        test_store_understanding_results_fails_when_original_path_unknown,
        test_load_parsed_json_rejects_missing_file,
    )]

    run_tests(tests, "Storage engine")


def main() -> None:
    run_all()


if __name__ == "__main__":
    main()
