"""
Phase 4 Query Router test suite (plain-Python runner, no pytest).

ISOLATION: storage paths are redirected into a temp directory for the whole
run, so the REAL data/safedoc.db and data/chroma_db are never touched.
A post-run guard re-checks the real DB's size/mtime/sha256 and fails if it
changed. The temp Chroma store is seeded with the REAL local embedding model
(all-MiniLM-L6-v2, no cloud APIs) so semantic tests exercise genuine
retrieval behavior. Synthetic document TEXT is test fixture data used ONLY
inside the isolated temp stores - it never enters production data.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parents[1]
for entry in (str(_project_root), str(_project_root / "src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from _harness import redirect_storage_to_temp, run_tests, seed_chroma_real_model

import storage_engine  # the SAME module instance query_router binds to
from storage_engine import ingest_into_sqlite
from src.query_router import (
    CLASS_EXACT,
    CLASS_SEMANTIC,
    CLASS_UNKNOWN,
    ROUTE_EXACT,
    ROUTE_SEMANTIC,
    classify_query,
    route_query,
)


# ---------------------------------------------------------------------------
# Isolation: redirect storage into a temp directory
# ---------------------------------------------------------------------------

_TEMP_DIR = redirect_storage_to_temp("safedocai-router-tests-")


# ---------------------------------------------------------------------------
# Seed data (fixture content, isolated temp stores only)
# ---------------------------------------------------------------------------

_SEED_DOCS = [
    {
        "file_name": "marksheet.pdf",
        "file_type": "PDF",
        "status": "success",
        "raw_text": (
            "Consolidated Marksheet Statement. Semester 1 SGPA 6.09. "
            "Semester 2 SGPA 6.05. Semester 3 SGPA 7.00. "
            "Student name TEST USER with roll number 12345678 and "
            "date of birth 06-06-2006. The marksheet covers semester "
            "wise grades, subject marks and cumulative performance."
        ),
        "extracted_entities": {
            "sgpa": ["6.09", "6.05", "7.00"],
            "roll": ["12345678"],
            "dob": ["06-06-2006"],
            "name": ["TEST USER"],
        },
    },
    {
        "file_name": "electricity_bill.pdf",
        "file_type": "PDF",
        "status": "success",
        "raw_text": (
            "Electricity bill for consumer number 1111222233 for the month "
            "of August. Amount due Rs 1450 before the due date. Units "
            "consumed 240. The bill contains tariff details, meter reading "
            "and payment instructions."
        ),
        "extracted_entities": {
            "Consumer Number": ["1111222233"],
            "dob": ["01-01-1980"],
            "amount": ["Rs. 1450"],
        },
    },
    {
        "file_name": "ticket.pdf",
        "file_type": "PDF",
        "status": "success",
        "raw_text": (
            "Indian Railways electronic reservation ticket. PNR 8425631974 "
            "train 12309 Rajdhani Express from New Delhi to Kanpur. "
            "The ticket contains passenger coach, quota and berth details "
            "along with journey date and boarding station information."
        ),
        "extracted_entities": {
            "PNR": ["8425631974"],
            "Train": ["12309 Rajdhani Express"],
        },
    },
]

_DOCS_SEEDED = False


def _seed_all() -> None:
    """Ingest fixture docs into the temp SQLite DB and Chroma store."""

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


def _doc(file_name: str) -> dict:
    for record in _SEED_DOCS:
        if record["file_name"] == file_name:
            return record
    raise AssertionError(f"unknown seed doc {file_name}")


# ---------------------------------------------------------------------------
# Classification tests
# ---------------------------------------------------------------------------


def test_classification_exact_queries() -> None:
    for query in (
        "What is my SGPA?",
        "What was my 4th semester SGPA?",
        "What is my roll number?",
        "What is my consumer number?",
        "What was my enrollment ID?",
        "What was my date of birth?",
    ):
        signals = classify_query(query)
        assert signals["classification"] == CLASS_EXACT, (query, signals)


def test_classification_semantic_queries() -> None:
    for query in (
        "What does my railway ticket say?",
        "What is covered in this document?",
        "Explain the important details in my application.",
        "What are the terms mentioned in this document?",
        "Summarize my document.",
    ):
        signals = classify_query(query)
        assert signals["classification"] == CLASS_SEMANTIC, (query, signals)


def test_classification_unknown_query() -> None:
    signals = classify_query("hmm interesting asdf")
    assert signals["classification"] == CLASS_UNKNOWN, signals


# ---------------------------------------------------------------------------
# Exact retrieval tests
# ---------------------------------------------------------------------------


def test_exact_sgpa_query_preserves_repeated_values() -> None:
    _seed_all()
    result = route_query("What is my SGPA?")
    assert result["route"] == ROUTE_EXACT
    assert result["insufficient"] is False
    values = {row["field_value"] for row in result["results"]}
    assert values == {"6.09", "6.05", "7.00"}, values
    assert len(result["results"]) == 3  # repeated values NOT collapsed


def test_exact_roll_number_query() -> None:
    _seed_all()
    result = route_query("What is my roll number?")
    assert result["route"] == ROUTE_EXACT
    matches = [row for row in result["results"] if row["field_name"] == "roll"]
    assert len(matches) == 1
    assert matches[0]["field_value"] == "12345678"
    assert matches[0]["file_name"] == "marksheet.pdf"


def test_exact_consumer_number_query() -> None:
    _seed_all()
    result = route_query("What is my consumer number?")
    assert result["route"] == ROUTE_EXACT
    matches = [
        row for row in result["results"] if row["field_value"] == "1111222233"
    ]
    assert len(matches) == 1
    assert matches[0]["file_name"] == "electricity_bill.pdf"


def test_exact_dob_query() -> None:
    _seed_all()
    result = route_query("What was my date of birth?")
    assert result["route"] == ROUTE_EXACT
    matches = [row for row in result["results"] if row["field_name"] == "dob"]
    assert {row["field_value"] for row in matches} == {"06-06-2006", "01-01-1980"}


def test_exact_value_hint_lookup() -> None:
    _seed_all()
    result = route_query("Find details for PNR 8425631974")
    assert result["classification"] == CLASS_EXACT
    assert result["value_hint"] == "8425631974"
    matches = [
        row for row in result["results"] if row["field_value"] == "8425631974"
    ]
    assert len(matches) == 1
    assert matches[0]["file_name"] == "ticket.pdf"


def test_exact_query_no_match_never_invents() -> None:
    _seed_all()
    result = route_query("What is my PAN number?")
    assert result["classification"] == CLASS_EXACT
    assert result["fallback"]["occurred"] is True
    assert result["fallback"]["reason"]
    # No fabricated value: nothing may be presented as an exact SQLite match.
    for row in result["results"]:
        assert row.get("source") != "sqlite.extracted_metadata", row
        assert row["field_name"].lower() != "pan"


def test_case_variation() -> None:
    _seed_all()
    result = route_query("WHAT IS MY ROLL NUMBER?")
    assert result["route"] == ROUTE_EXACT
    assert any(row["field_value"] == "12345678" for row in result["results"])


def test_minor_natural_language_variation() -> None:
    _seed_all()
    result = route_query("Can you show my roll no please?")
    assert result["route"] == ROUTE_EXACT
    assert any(row["field_value"] == "12345678" for row in result["results"])


def test_document_specific_file_name_query() -> None:
    _seed_all()
    result = route_query("What is the PNR in ticket.pdf?")
    assert result["document_filter"] == {"file_name": "ticket.pdf"}
    assert result["route"] == ROUTE_EXACT
    matches = [
        row for row in result["results"] if row["field_value"] == "8425631974"
    ]
    assert len(matches) == 1


def test_file_name_filtering_restricts_results() -> None:
    _seed_all()
    result = route_query("What is the dob in electricity_bill.pdf?")
    assert result["document_filter"] == {"file_name": "electricity_bill.pdf"}
    assert result["route"] == ROUTE_EXACT
    assert len(result["results"]) == 1
    assert result["results"][0]["field_value"] == "01-01-1980"
    assert result["results"][0]["document_id"] == _doc("electricity_bill.pdf")["doc_id"]


def test_document_id_filtering() -> None:
    _seed_all()
    ticket_id = _doc("ticket.pdf")["doc_id"]
    result = route_query(f"What is the PNR in document {ticket_id}?")
    assert result["document_filter"] == {"document_id": ticket_id}
    matches = [
        row for row in result["results"] if row["field_value"] == "8425631974"
    ]
    assert len(matches) == 1


def test_filter_and_repeated_values_combined() -> None:
    _seed_all()
    result = route_query("What is the SGPA in marksheet.pdf?")
    assert result["document_filter"] == {"file_name": "marksheet.pdf"}
    values = sorted(row["field_value"] for row in result["results"])
    assert values == ["6.05", "6.09", "7.00"], values


# ---------------------------------------------------------------------------
# Semantic retrieval tests
# ---------------------------------------------------------------------------


def test_open_ended_semantic_query() -> None:
    _seed_all()
    result = route_query("What does my railway ticket say?")
    assert result["route"] == ROUTE_SEMANTIC
    assert result["fallback"]["occurred"] is False
    assert result["results"], "expected useful chunks from the ticket text"
    for row in result["results"]:
        assert row["source"] == "chroma.safedoc_documents"
        assert row["file_name"] == "ticket.pdf", row
        assert row["distance"] is not None


def test_semantic_query_no_useful_result() -> None:
    _seed_all()
    result = route_query("Give me an overview of quantum knitting patterns.")
    assert result["route"] == ROUTE_SEMANTIC
    # No fabrication: either an explicit insufficient state, or only grounded
    # chroma chunks with provenance (no invented text).
    if result["insufficient"]:
        assert result["insufficient_reason"]
    for row in result["results"]:
        assert row["source"] == "chroma.safedoc_documents"


def test_unknown_query_routes_to_semantic_without_fabrication() -> None:
    _seed_all()
    result = route_query("hmm interesting asdf")
    assert result["classification"] == CLASS_UNKNOWN
    assert result["route"] == ROUTE_SEMANTIC
    if result["insufficient"]:
        assert result["results"] == []


# ---------------------------------------------------------------------------
# Phase 10.1: bare field queries (no personal/verb anchor)
# ---------------------------------------------------------------------------


def test_bare_field_query_routes_exact() -> None:
    """"roll number" (bare, no "my"/question anchor) must hit the exact
    SQLite path instead of the slow semantic LLM path."""
    _seed_all()
    result = route_query("roll number")
    assert result["classification"] == CLASS_EXACT, result["classification"]
    assert result["route"] == ROUTE_EXACT
    assert result["fallback"]["occurred"] is False
    matches = [row for row in result["results"] if row["field_name"] == "roll"]
    assert len(matches) == 1
    assert matches[0]["field_value"] == "12345678"
    assert matches[0]["file_name"] == "marksheet.pdf"


def test_bare_field_query_variants_routes_exact() -> None:
    _seed_all()
    for query in ("Roll No", "roll number kya hai?", "my roll number",
                  "Mera roll number kya hai?"):
        result = route_query(query)
        assert result["route"] == ROUTE_EXACT, (query, result["route"])
        assert any(
            row["field_value"] == "12345678" for row in result["results"]
        ), query


def test_bare_multi_document_field_lists_all_values() -> None:
    """Ambiguity rule: "dob" exists in two documents with different values.
    The bare query must list ALL values with provenance — never silently
    pick one."""
    _seed_all()
    result = route_query("date of birth")
    assert result["route"] == ROUTE_EXACT
    matches = [row for row in result["results"] if row["field_name"] == "dob"]
    assert {row["field_value"] for row in matches} == {
        "06-06-2006", "01-01-1980",
    }
    assert {row["file_name"] for row in matches} == {
        "marksheet.pdf", "electricity_bill.pdf",
    }


def test_unmapped_short_queries_stay_semantic() -> None:
    """Short queries naming NO stored field keep the current route."""
    _seed_all()
    for query in ("PAN", "IFSC", "enrollment ID"):
        result = route_query(query)
        assert result["route"] == ROUTE_SEMANTIC, (query, result["route"])
        assert result["fallback"]["occurred"] is False, query


def test_unrelated_short_queries_never_exact() -> None:
    """"weather" / "hello" / "2 + 2" must not become document lookups."""
    _seed_all()
    for query in ("weather", "hello", "2 + 2"):
        result = route_query(query)
        assert result["classification"] != CLASS_EXACT, (query, result)
        assert result["route"] == ROUTE_SEMANTIC, (query, result["route"])


# ---------------------------------------------------------------------------
# Provenance + determinism tests
# ---------------------------------------------------------------------------


def test_provenance_fields_present() -> None:
    _seed_all()
    exact = route_query("What is my roll number?")
    assert exact["query"]
    assert exact["route"]
    assert "fallback" in exact and "occurred" in exact["fallback"]
    assert set(exact["timings_ms"]) == {"classification", "retrieval", "total"}
    row = exact["results"][0]
    for key in (
        "source",
        "document_id",
        "file_name",
        "file_path",
        "field_name",
        "field_value",
    ):
        assert key in row, (key, row)

    semantic = route_query("What does my railway ticket say?")
    assert semantic["results"]
    srow = semantic["results"][0]
    for key in ("source", "document_id", "file_name", "chunk_id", "text", "distance"):
        assert key in srow, (key, srow)
    assert srow["document_id"] == _doc("ticket.pdf")["doc_id"]


def test_deterministic_for_repeated_identical_queries() -> None:
    _seed_all()
    first = route_query("What is my SGPA?")
    second = route_query("What is my SGPA?")
    strip = lambda r: {k: v for k, v in r.items() if k != "timings_ms"}
    assert strip(first) == strip(second)

    first_s = route_query("What does my railway ticket say?")
    second_s = route_query("What does my railway ticket say?")
    assert strip(first_s) == strip(second_s)


def test_fallback_disabled_returns_explicit_insufficient() -> None:
    _seed_all()
    result = route_query("What is my PAN number?", allow_fallback=False)
    assert result["route"] == ROUTE_EXACT
    assert result["fallback"]["occurred"] is False
    assert result["insufficient"] is True
    assert "no exact metadata match" in result["insufficient_reason"]
    assert result["results"] == []


def test_empty_and_invalid_inputs_rejected() -> None:
    _seed_all()
    for kwargs in ({"query": ""}, {"query": "   "}):
        try:
            route_query(**kwargs)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {kwargs}")
    try:
        route_query("What is my SGPA?", top_k=0)
    except ValueError:
        return
    raise AssertionError("expected ValueError for top_k=0")


def test_deep_copy_results_not_shared_between_calls() -> None:
    _seed_all()
    first = route_query("What is my SGPA?")
    snapshot = copy.deepcopy(first["results"])
    second = route_query("What is my SGPA?")
    assert first["results"] == snapshot  # first call's payload untouched
    assert second["results"] == snapshot


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def main() -> None:
    tests = [(fn.__name__, fn) for fn in (
        test_classification_exact_queries,
        test_classification_semantic_queries,
        test_classification_unknown_query,
        test_exact_sgpa_query_preserves_repeated_values,
        test_exact_roll_number_query,
        test_exact_consumer_number_query,
        test_exact_dob_query,
        test_exact_value_hint_lookup,
        test_exact_query_no_match_never_invents,
        test_case_variation,
        test_minor_natural_language_variation,
        test_document_specific_file_name_query,
        test_file_name_filtering_restricts_results,
        test_document_id_filtering,
        test_filter_and_repeated_values_combined,
        test_open_ended_semantic_query,
        test_semantic_query_no_useful_result,
        test_unknown_query_routes_to_semantic_without_fabrication,
        test_bare_field_query_routes_exact,
        test_bare_field_query_variants_routes_exact,
        test_bare_multi_document_field_lists_all_values,
        test_unmapped_short_queries_stay_semantic,
        test_unrelated_short_queries_never_exact,
        test_provenance_fields_present,
        test_deterministic_for_repeated_identical_queries,
        test_fallback_disabled_returns_explicit_insufficient,
        test_empty_and_invalid_inputs_rejected,
        test_deep_copy_results_not_shared_between_calls,
    )]

    run_tests(tests, "Query router")


if __name__ == "__main__":
    main()
