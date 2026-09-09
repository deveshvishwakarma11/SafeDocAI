"""
Tests for SafeDocAI long-document handling upgrade.

These tests verify the new multi-window pipeline stage without
replacing the existing Phase 1/Phase 2/Phase 3 flow.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chunker import (
    chunk_text_safe,
    chunk_windows,
    score_chunk_relevance,
    select_relevant_windows,
)

from document_understanding import (
    _build_windows_for_document,
    _field_evidence_score,
    _merge_llm_windows,
    _normalize_for_merge,
    build_prompt_context_for_window,
)

from evidence_validator import validate_fields


# ---------------------------------------------------------------------------
# Chunker safety
# ---------------------------------------------------------------------------

class TestChunkSafety:
    def test_decimals_not_split(self):
        text = "A 1200.50 B 6.61 C 3.14 D"
        chunks = chunk_text_safe(text, chunk_size=12, overlap=2)
        flat = "\n".join(chunks)
        assert "1200.50" in flat
        assert "6.61" in flat
        assert "3.14" in flat

    def test_ids_and_dates_preserved(self):
        text = "Roll No 999 Date 01/02/2021 Amount 1000.00"
        chunks = chunk_text_safe(text, chunk_size=15, overlap=3)
        flat = "\n".join(chunks)
        assert "999" in flat
        assert "01/02/2021" in flat
        assert "1000.00" in flat

    def test_empty_text(self):
        assert chunk_text_safe("", chunk_size=10) == []

    def test_very_small_chunk_size(self):
        chunks = chunk_text_safe("ab cd ef", chunk_size=2, overlap=1)
        assert len(chunks) >= 1
        flat = "\n".join(chunks)
        assert "ab cd ef" == flat.replace("\n", " ")


# ---------------------------------------------------------------------------
# Window selection
# ---------------------------------------------------------------------------

class TestWindowSelection:
    def test_short_doc_uses_single_window(self):
        text = "Name: X\nRoll No: 1\nMarks: 6.61\nCollege: ABC"
        windows = _build_windows_for_document(text)
        assert len(windows) == 1
        assert "Roll No" in windows[0]

    def test_long_doc_becomes_multi_window(self):
        middle = "Subject: Maths\nMarks: 85\nRoll No: 999\nCollege: XYZ"
        text = "\n".join(
            ["header line"] * 6
            + [middle]
            + [f"footer line {i}" for i in range(200)]
        )
        windows = _build_windows_for_document(text)
        assert len(windows) >= 2
        joined = "\n".join(windows)
        assert "999" in joined

    def test_overlap_keeps_boundary_content(self):
        chunks = ["a"] * 4 + ["boundary 777"] + ["b"] * 4
        windows = select_relevant_windows(
            chunks,
            max_windows=5,
            window_size=2,
            min_score=0.0,
            stride=1,
        )
        joined = "\n".join("\n".join(w) for w in windows)
        assert "777" in joined

    def test_relevance_selection_keeps_important_chunk(self):
        chunks = ["filler"] * 6 + ["IMPORTANT 777"] + ["filler"] * 6
        windows = select_relevant_windows(
            chunks,
            max_windows=5,
            window_size=2,
            min_score=0.0,
            stride=1,
        )
        joined = "\n".join("\n".join(w) for w in windows)
        assert "777" in joined

    def test_fallback_when_no_windows_selected(self):
        chunks = ["x"] * 3
        windows = select_relevant_windows(
            chunks,
            max_windows=5,
            window_size=2,
            min_score=999,
            stride=1,
        )
        assert windows == []
        fallback = chunk_windows(
            chunks,
            window_size=2,
            stride=max(1, len(chunks) // max(1, 5)),
        )
        assert fallback


# ---------------------------------------------------------------------------
# Prompt context helper
# ---------------------------------------------------------------------------

class TestPromptContext:
    def test_strips_but_keeps_content(self):
        ctx = build_prompt_context_for_window("  line1\n\nline2  \n")
        assert ctx == "line1\n\nline2"

    def test_empty(self):
        assert build_prompt_context_for_window("") == ""


# ---------------------------------------------------------------------------
# Merge logic
# ---------------------------------------------------------------------------

class TestMergeWindows:
    def test_duplicate_field_uses_better_evidence(self):
        merged = _merge_llm_windows(
            [
                {
                    "document_type": "MARKSHEET",
                    "summary": "s1",
                    "fields": [
                        {
                            "key": "Roll No",
                            "value": "100",
                            "evidence_snippet": "Roll No 100",
                        }
                    ],
                },
                {
                    "document_type": "MARKSHEET",
                    "summary": "s2",
                    "fields": [
                        {
                            "key": "Roll No",
                            "value": "200",
                            "evidence_snippet": "Roll No 200",
                        }
                    ],
                },
            ],
            "Roll No 100 is present here and Roll No 200 is not",
        )
        combined = {f["key"]: f["value"] for f in merged["fields"]}
        assert combined["Roll No"] == "100"

    def test_both_fields_preserved_when_different_keys(self):
        merged = _merge_llm_windows(
            [
                {
                    "document_type": "MARKSHEET",
                    "fields": [
                        {"key": "Roll No", "value": "100", "evidence_snippet": "100"}
                    ],
                },
                {
                    "document_type": "MARKSHEET",
                    "fields": [
                        {"key": "Name", "value": "X", "evidence_snippet": "X"}
                    ],
                },
            ],
            "Roll No 100 Name X",
        )
        keys = {f["key"] for f in merged["fields"]}
        assert keys == {"Roll No", "Name"}

    def test_document_type_voting_ignores_unknown(self):
        merged = _merge_llm_windows(
            [
                {"document_type": "MARKSHEET"},
                {"document_type": "MARKSHEET"},
                {"document_type": "UNKNOWN"},
            ],
            "",
        )
        assert merged["document_type"] == "MARKSHEET"

    def test_all_unknown_results_in_unknown_type(self):
        merged = _merge_llm_windows(
            [
                {"document_type": "UNKNOWN"},
                {"document_type": "N/A"},
                {"document_type": ""},
            ],
            "",
        )
        assert merged["document_type"] == "UNKNOWN"


# ---------------------------------------------------------------------------
# Field evidence score
# ---------------------------------------------------------------------------

class TestFieldEvidenceScore:
    def test_value_present_in_full_ocr(self):
        score = _field_evidence_score(
            "100", "100", "Roll No 100 is present"
        )
        assert score >= 1.0

    def test_snippet_present_but_value_not(self):
        score = _field_evidence_score(
            "200", "100", "Roll No 100 is present"
        )
        assert score >= 0.5

    def test_unknown_value_not_scored(self):
        score = _field_evidence_score(
            "UNKNOWN", "UNKNOWN", "some text"
        )
        assert score == 0.0

    def test_no_raw_text(self):
        assert _field_evidence_score("100", "100", "") == 0.0


# ---------------------------------------------------------------------------
# Normalize helper
# ---------------------------------------------------------------------------

class TestNormalizeForMerge:
    def test_basic(self):
        assert _normalize_for_merge("Roll No 100") == "roll no 100"

    def test_punctuation_before_digit(self):
        assert _normalize_for_merge("1200.50") == "1200 50"

    def test_empty(self):
        assert _normalize_for_merge("") == ""


# ---------------------------------------------------------------------------
# Evidence validation still rejects fake fields against full OCR
# ---------------------------------------------------------------------------

class TestEvidenceValidationStillWorks:
    def test_real_field_verified(self):
        fields = [
            {"key": "Roll No", "value": "100", "evidence_snippet": "Roll No 100"}
        ]
        validated = validate_fields(fields, "Roll No 100 is present", require_all_verified=False)
        assert any(f["key"] == "Roll No" and f.get("verified") for f in validated)

    def test_fake_field_rejected(self):
        fields = [
            {"key": "Roll No", "value": "99999", "evidence_snippet": "Roll No 99999"}
        ]
        validated = validate_fields(fields, "Roll No 100 is present", require_all_verified=False)
        assert not any(
            f["key"] == "Roll No" and f.get("verified") for f in validated
        )


# ---------------------------------------------------------------------------
# Original document path stability
# ---------------------------------------------------------------------------

class TestOriginalDocPathStability:
    def test_process_document_keeps_original_doc_path_in_storage_call(self):
        from document_understanding import process_document, save_understanding

        ocr_json = {
            "file_name": "sample.pdf",
            "file_path": "/real/original/path/sample.pdf",
            "raw_text": "Name: X\nRoll No: 1\nMarks: 6.61",
            "extraction_method": "pdf-parse",
        }

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "output"
            understood_dir = Path(tmp) / "understood"
            out_dir.mkdir()
            understood_dir.mkdir()

            json_path = out_dir / "sample.json"
            json_path.write_text(json.dumps(ocr_json), encoding="utf-8")

            from document_understanding import (
                UNDERSTANDING_DIR as original_understood_dir,
                OUTPUT_DIR as original_output_dir,
            )
            from document_understanding import (
                UNDERSTANDING_DIR,
                OUTPUT_DIR,
                PROJECT_ROOT,
            )

            saved_output = None

            class Span:
                def __enter__(self):
                    self.prev_out = OUTPUT_DIR
                    self.prev_understood = UNDERSTANDING_DIR
                    OUTPUT_DIR.replace(out_dir)
                    UNDERSTANDING_DIR.replace(understood_dir)
                    return self

                def __exit__(self, *args):
                    OUTPUT_DIR.replace(self.prev_out)
                    UNDERSTANDING_DIR.replace(self.prev_understood)

            with Span():
                result = process_document(json_path)

            assert result is not None
            saved = understood_dir / json_path.name
            assert saved.exists()
            payload = json.loads(saved.read_text(encoding="utf-8"))
            assert payload["source_file"] == "sample.pdf"
            assert payload["source_json"] == "sample.json"
            understanding = payload["understanding"]
            assert understanding["document_type"] in {
                "MARKSHEET",
                "UNKNOWN",
            }
