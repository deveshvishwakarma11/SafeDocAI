"""
Phase 6 Retrieval Relevance test suite (plain-Python runner, no pytest).

ISOLATION: storage paths are redirected into a temp directory for the whole
run, so the REAL data/safedoc.db and data/chroma_db are never touched. The
temp Chroma store is seeded with the REAL local embedding model
(all-MiniLM-L6-v2, no cloud APIs). Fixture document TEXT is test data used
ONLY inside the isolated temp stores.
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

import storage_engine  # top-level: the SAME module instance the router binds to

from storage_engine import ingest_into_sqlite
from src.query_router import (
    ROUTE_EXACT,
    ROUTE_SEMANTIC,
    SEMANTIC_USEFUL_MAX_DISTANCE,
    detect_document_intent,
    route_query,
)


# ---------------------------------------------------------------------------
# Isolation (same pattern as test_query_router.py)
# ---------------------------------------------------------------------------

_TEMP_DIR: Path | None = None
_ORIGINALS: dict[str, Path] = {}


def _redirect_storage_to_temp() -> Path:
    global _TEMP_DIR
    if _TEMP_DIR is not None:
        return _TEMP_DIR

    _TEMP_DIR = Path(tempfile.mkdtemp(prefix="safedocai-relevance-tests-"))
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
# Seed data: the Phase 6 collision scenario, with REAL-shape fixture names
# ---------------------------------------------------------------------------

_SEED_DOCS = [
    {
        # Colliding doc: boilerplate mentions "Photo ID cards" but it is a ticket.
        "file_name": "4143027140.pdf",
        "file_type": "PDF",
        "status": "success",
        "raw_text": (
            "Indian Railways electronic reservation ticket. PNR 8425631974 "
            "train 12309 Rajdhani Express. Passengers must carry original photo "
            "ID cards during the journey. Customer care 14646 and toll free "
            "1800-11-400. The ticket contains coach, quota and berth details."
        ),
        "extracted_entities": {
            "PNR": ["8425631974"],
            "Transaction ID": ["100006773136446"],
            "Customer care": ["14646"],
        },
    },
    {
        "file_name": "ApplicationForm.pdf",
        "file_type": "PDF",
        "status": "success",
        "raw_text": (
            "Graduate aptitude test application form. Candidate DEVESH "
            "VISHWAKARMA with enrollment ID M103B71 and date of birth "
            "06/06/2006. Enrollment ID M103B71 is the candidate enrollment "
            "identifier on this application form. Exam cities Gorakhpur, "
            "Agra, Prayagraj and Kanpur. The application form contains "
            "qualification details and contact details for the examination."
        ),
        "extracted_entities": {
            "dob": ["06/06/2006"],
            "email": ["gate@iitk.ac.in"],
            "Exam City": ["Gorakhpur (UP)"],
        },
    },
    {
        "file_name": "ID Card.pdf",
        "file_type": "PDF",
        "status": "success",
        "raw_text": (
            "College identity card. Name DEVESH VISHVYAICARIIA course B.Tech "
            "CSE date of birth 06-06-2006 phone 9163791592 police station "
            "Kauriram. The card is issued by the institute for identification."
        ),
        "extracted_entities": {
            "dob": ["06-06-2006"],
            "phone": ["9163791592"],
            "Name": ["DEVESH VISHVYAICARIIA"],
            "Course": ["B.Tech | CSE >"],
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
            chunks, batch_size=16, show_progress_bar=False, normalize_embeddings=True
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
            ids=ids,
            documents=chunks,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
        )
    _DOCS_SEEDED = True


def _doc(file_name: str) -> dict:
    for record in _SEED_DOCS:
        if record["file_name"] == file_name:
            return record
    raise AssertionError(f"unknown seed doc {file_name}")


def _file_names(result: dict) -> list[str]:
    return [row.get("file_name") for row in result["results"]]


# ---------------------------------------------------------------------------
# 1. Document intent detection (pure, no retrieval)
# ---------------------------------------------------------------------------


def test_intent_id_card_detects_strong_candidate() -> None:
    _seed_all()
    import sqlite3 as _sq

    con = _sq.connect(str(_TEMP_DIR / "safedoc.db"))
    try:
        intent = detect_document_intent("Tell me about my ID card", con)
        assert intent["intent_detected"] is True
        assert "id card" in intent["phrases"]
        strong_ids = {c["document_id"] for c in intent["strong"]}
        assert strong_ids == {_doc("ID Card.pdf")["doc_id"]}, intent["candidates"]
        assert intent["restrict_to"] == [_doc("ID Card.pdf")["doc_id"]]
    finally:
        con.close()


def test_intent_railway_ticket_and_application_form() -> None:
    _seed_all()
    import sqlite3 as _sq

    con = _sq.connect(str(_TEMP_DIR / "safedoc.db"))
    try:
        ticket = detect_document_intent("What does my railway ticket say?", con)
        # No stored document is named like a ticket: intent stays non-restrictive.
        assert ticket["restrict_to"] in (None, []) or all(
            c["strength"] == "weak" for c in ticket["candidates"]
        ), ticket["candidates"]

        app = detect_document_intent("What is in my application form?", con)
        strong_ids = {c["document_id"] for c in app["strong"]}
        assert strong_ids == {_doc("ApplicationForm.pdf")["doc_id"]}, app["candidates"]
    finally:
        con.close()


def test_intent_explicit_filename_digit_stem() -> None:
    _seed_all()
    import sqlite3 as _sq

    con = _sq.connect(str(_TEMP_DIR / "safedoc.db"))
    try:
        intent = detect_document_intent(
            "What is the transaction ID in 4143027140.pdf?", con
        )
        assert intent["intent_detected"] is True
        ticket_id = _doc("4143027140.pdf")["doc_id"]
        strong_ids = {c["document_id"] for c in intent["strong"]}
        assert strong_ids == {ticket_id}, intent["candidates"]
    finally:
        con.close()


def test_intent_field_like_phrases_never_restrict_documents() -> None:
    _seed_all()
    import sqlite3 as _sq

    con = _sq.connect(str(_TEMP_DIR / "safedoc.db"))
    try:
        # "enrollment id" names a FIELD, not a document: the token "id" in
        # "ID Card.pdf" must not hijack the query into a wrong restriction.
        intent = detect_document_intent("What was my enrollment ID?", con)
        assert intent["intent_detected"] is False, intent
        assert intent["restrict_to"] is None
        assert intent.get("field_like_phrases") == ["enrollment id"]

        txn = detect_document_intent("What is my transaction ID?", con)
        assert txn["restrict_to"] is None
    finally:
        con.close()


def test_intent_no_phrase_no_restriction() -> None:
    _seed_all()
    import sqlite3 as _sq

    con = _sq.connect(str(_TEMP_DIR / "safedoc.db"))
    try:
        for query in ("What's my SGPA?", "Tell me a joke", "hello there"):
            intent = detect_document_intent(query, con)
            assert intent["intent_detected"] is False, (query, intent)
            assert intent["restrict_to"] is None
    finally:
        con.close()


# ---------------------------------------------------------------------------
# 2. Reranked semantic retrieval (the core Phase 6 behavior)
# ---------------------------------------------------------------------------


def test_id_card_query_ranks_id_card_first_not_eticket() -> None:
    """THE acceptance test: generic 'Photo ID cards' boilerplate in the
    E-ticket must never produce the answer for the ID Card query.

    Either route is acceptable (both are correct): the exact route returns
    the ID Card's own stored fields; the semantic route must be restricted
    to ID Card.pdf and reranked. In NO case may an E-ticket chunk appear."""
    _seed_all()
    result = route_query("Tell me the important information on my ID card.")
    assert result["document_intent"]["detected"] is True
    assert result["document_intent"]["strong_document_ids"] == [
        _doc("ID Card.pdf")["doc_id"]
    ]
    assert result["document_intent"]["relaxed_to_global"] is False
    assert result["insufficient"] is False

    if result["route"] == ROUTE_EXACT:
        # Deterministic fields from the requested document only.
        assert result["document_filter"] == {"file_name": "ID Card.pdf"}
        assert {row["file_name"] for row in result["results"]} == {"ID Card.pdf"}
        assert result["fallback"]["occurred"] is False
    else:
        names = _file_names(result)
        assert names, "expected useful chunks"
        assert names[0] == "ID Card.pdf", names
        assert set(names) == {"ID Card.pdf"}, (
            "intent restriction should keep E-ticket chunks out entirely",
            names,
        )
        assert result["document_intent"]["restriction_applied"] is True
        top = result["results"][0]
        assert top["rerank_score"] is not None
        assert top["intent_match"]["document_id"] == _doc("ID Card.pdf")["doc_id"]


def test_application_form_query_ranks_application_first() -> None:
    _seed_all()
    result = route_query("What is in my application form?")
    assert result["route"] == ROUTE_SEMANTIC
    names = _file_names(result)
    assert names and names[0] == "ApplicationForm.pdf", names
    assert set(names) == {"ApplicationForm.pdf"}


def test_generic_collision_id_card_is_not_hijacked_by_eticket() -> None:
    _seed_all()
    # Bare "id card" with a semantic cue: the E-ticket's "Photo ID cards"
    # boilerplate must not outrank the actual ID Card document.
    result = route_query("Tell me about my id card")
    assert result["route"] == ROUTE_SEMANTIC
    names = _file_names(result)
    assert names and names[0] == "ID Card.pdf", names


def test_phase4_exact_document_filter_behavior_preserved() -> None:
    """Pre-Phase-6 behavior: a query containing a stored file name with an
    exact cue returns that document's fields via the exact route."""
    _seed_all()
    result = route_query("my id card")
    assert result["route"] == ROUTE_EXACT
    assert result["document_filter"] == {"file_name": "ID Card.pdf"}
    assert result["results"], "expected the ID Card's stored fields"
    assert {row["file_name"] for row in result["results"]} == {"ID Card.pdf"}


def test_no_intent_query_keeps_global_semantic_behavior() -> None:
    _seed_all()
    result = route_query("Give me an overview of quantum knitting patterns.")
    assert result["route"] == ROUTE_SEMANTIC
    assert result["document_intent"]["detected"] is False
    assert result["document_intent"]["restriction_applied"] is False
    assert result["document_intent"]["relaxed_to_global"] is False
    # Global behavior: chunks (if any useful) come from anywhere.
    for row in result["results"]:
        assert row["source"] == "chroma.safedoc_documents"


def test_below_threshold_restricted_results_stay_insufficient() -> None:
    """Chunks exist in the intent-matching doc but none is relevant enough:
    the restriction is KEPT (no global substitution) and the state is
    explicitly insufficient."""
    _seed_all()
    result = route_query("Tell me the important information on my ID card.")
    # Sanity: restricted retrieval works for the real collision query.
    assert result["results"]

    # Now an ID-card-intent query about an unrelated topic: the restriction
    # to ID Card.pdf must hold, and irrelevant content must NOT be pulled
    # from other documents.
    result2 = route_query("Tell me about my ID card and quantum knitting patterns")
    if result2["results"]:
        # If anything is returned it can ONLY come from the restricted doc.
        assert set(_file_names(result2)) == {"ID Card.pdf"}
    else:
        assert result2["insufficient"] is True
        assert result2["insufficient_reason"]
        assert result2["document_intent"]["relaxed_to_global"] is False


def test_exact_query_with_document_filter_still_works() -> None:
    _seed_all()
    result = route_query("What is the transaction ID in 4143027140.pdf?")
    assert result["route"] == ROUTE_EXACT
    assert result["document_filter"] == {"file_name": "4143027140.pdf"}
    matches = [
        row for row in result["results"] if row["field_value"] == "100006773136446"
    ]
    assert len(matches) == 1
    assert matches[0]["file_name"] == "4143027140.pdf"


def test_exact_miss_semantic_fallback_applies_intent() -> None:
    _seed_all()
    # 'enrollment id' is field-like (no document restriction) but the exact
    # lookup misses; fallback must still land on the application form doc.
    result = route_query("What was my enrollment ID?")
    assert result["fallback"]["occurred"] is True
    assert result["document_intent"]["detected"] is False
    assert result["document_intent"]["restriction_applied"] is False
    assert result["results"], "expected fallback chunks from the application form"
    names = _file_names(result)
    assert "ApplicationForm.pdf" in names, names
    assert any("M103B71" in row.get("text", "") for row in result["results"])


def test_exact_miss_semantic_fallback_with_document_intent() -> None:
    _seed_all()
    # "journey date" is NOT stored in the ticket's metadata -> exact miss;
    # the explicit filename intent restricts the fallback to the ticket doc.
    result = route_query("What is the journey date in 4143027140.pdf?")
    assert result["fallback"]["occurred"] is True
    assert result["document_intent"]["detected"] is True
    assert result["document_intent"]["restriction_applied"] is True
    if result["results"]:
        assert set(_file_names(result)) == {"4143027140.pdf"}
    else:
        assert result["insufficient"] is True
        # The restriction must be KEPT: no unrelated document substituted.
        assert result["document_intent"]["relaxed_to_global"] is False


def test_identifier_heavy_query_prefers_matching_document() -> None:
    _seed_all()
    result = route_query("Find details for PNR 8425631974")
    # Value-hint exact lookup wins deterministically.
    assert result["route"] == ROUTE_EXACT
    matches = [row for row in result["results"] if row["field_value"] == "8425631974"]
    assert len(matches) == 1
    assert matches[0]["file_name"] == "4143027140.pdf"


def test_case_and_whitespace_variations_rank_identically() -> None:
    _seed_all()
    base = route_query("Tell me the important information on my ID card.")
    varied = route_query("  tell   me THE IMPORTANT Information on MY id card  ")
    assert _file_names(base)[0] == "ID Card.pdf"
    assert _file_names(varied)[0] == "ID Card.pdf"


def test_provenance_fields_preserved_with_rerank_annotations() -> None:
    _seed_all()
    # Semantic-path provenance (guaranteed semantic via 'tell me about').
    result = route_query("Tell me about my id card")
    assert result["route"] == ROUTE_SEMANTIC
    for row in result["results"]:
        for key in (
            "source",
            "document_id",
            "file_name",
            "chunk_id",
            "text",
            "distance",
            "useful",
        ):
            assert key in row, (key, row)
        assert "rerank_score" in row
        assert set(row["rerank_components"]) == {
            "intent",
            "semantic",
            "lexical",
            "phrase",
            "identifier",
        }
    block = result["document_intent"]
    for key in (
        "detected",
        "phrases",
        "candidates",
        "strong_document_ids",
        "weak_document_ids",
        "restrict_to",
        "restriction_applied",
        "relaxed_to_global",
        "restriction_note",
    ):
        assert key in block, (key, block)

    # Exact-path provenance must remain complete too.
    exact = route_query("my id card")
    assert exact["route"] == ROUTE_EXACT
    row = exact["results"][0]
    for key in ("source", "document_id", "file_name", "field_name", "field_value"):
        assert key in row, (key, row)


def test_deterministic_repeated_ranking() -> None:
    _seed_all()
    strip = lambda r: {k: v for k, v in r.items() if k != "timings_ms"}
    first = route_query("Tell me the important information on my ID card.")
    second = route_query("Tell me the important information on my ID card.")
    assert strip(first) == strip(second)


def test_empty_chroma_store_returns_insufficient_not_crash() -> None:
    """With no chunks at all, a restricted search relaxes to global, which
    then finds nothing: explicit insufficient state, no crash, no invention."""
    _seed_all()
    collection = storage_engine.get_chroma_collection()
    saved = collection.get()
    if saved["ids"]:
        collection.delete(ids=saved["ids"])
    try:
        result = route_query("Tell me about my ID card")
        assert result["route"] in (ROUTE_EXACT, ROUTE_SEMANTIC)
        assert result["results"] == []
        assert result["insufficient"] is True
        assert result["insufficient_reason"]
    finally:
        # Restore the fixture chunks (re-embed deterministically).
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
                ids=ids,
                documents=chunks,
                embeddings=embeddings.tolist(),
                metadatas=metadatas,
            )


def test_semantic_threshold_not_loosened() -> None:
    """The global baseline threshold is unchanged; the only admitted extra
    distance is the intent-qualified margin, which is restricted to
    strong-intent documents and recorded in provenance."""
    _seed_all()
    from src.query_router import INTENT_QUALIFIED_MAX_DISTANCE

    # Unrestricted global search: baseline only, no margin possible.
    global_result = route_query("Give me an overview of quantum knitting patterns.")
    for row in global_result["results"]:
        assert row["useful"] is True
        assert row["distance"] <= SEMANTIC_USEFUL_MAX_DISTANCE + 1e-9

    # Strong-intent semantic search: baseline OR recorded margin, nothing else.
    result = route_query("Tell me about my id card")
    block = result["document_intent"]
    margin_used = (
        "intent-qualified threshold" in (block.get("restriction_note") or "")
    )
    for row in result["results"]:
        assert row["useful"] is True
        if row["distance"] <= SEMANTIC_USEFUL_MAX_DISTANCE + 1e-9:
            continue
        assert margin_used, "above-baseline chunk accepted without recorded margin"
        assert block["restriction_applied"] is True
        assert block["relaxed_to_global"] is False
        assert row["distance"] <= INTENT_QUALIFIED_MAX_DISTANCE + 1e-9
        assert row["file_name"] == "ID Card.pdf"


def test_weak_only_intent_restricts_and_reranks() -> None:
    """A weak-only intent (no strong file-name hit) still restricts the
    search but is explicitly recorded as weak."""
    _seed_all()
    # 'bill' head noun with no bill-named document: no candidates at all.
    result = route_query("What does my electricity bill say?")
    assert result["route"] == ROUTE_SEMANTIC
    assert result["document_intent"]["detected"] is False


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_all() -> None:
    tests = [
        test_intent_id_card_detects_strong_candidate,
        test_intent_railway_ticket_and_application_form,
        test_intent_explicit_filename_digit_stem,
        test_intent_field_like_phrases_never_restrict_documents,
        test_intent_no_phrase_no_restriction,
        test_id_card_query_ranks_id_card_first_not_eticket,
        test_application_form_query_ranks_application_first,
        test_generic_collision_id_card_is_not_hijacked_by_eticket,
        test_phase4_exact_document_filter_behavior_preserved,
        test_no_intent_query_keeps_global_semantic_behavior,
        test_below_threshold_restricted_results_stay_insufficient,
        test_exact_query_with_document_filter_still_works,
        test_exact_miss_semantic_fallback_applies_intent,
        test_exact_miss_semantic_fallback_with_document_intent,
        test_identifier_heavy_query_prefers_matching_document,
        test_case_and_whitespace_variations_rank_identically,
        test_provenance_fields_preserved_with_rerank_annotations,
        test_deterministic_repeated_ranking,
        test_empty_chroma_store_returns_insufficient_not_crash,
        test_semantic_threshold_not_loosened,
        test_weak_only_intent_restricts_and_reranks,
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
        print("Retrieval relevance tests FAILED:")
        for line in failed:
            print(" -", line)
        raise SystemExit(1)

    print(f"Retrieval relevance tests PASSED: {len(tests)}")


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
