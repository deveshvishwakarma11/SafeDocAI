"""
Candidate extractor focused checks (Phase 3 selection architecture).

Covers the 16 approved test areas: separator tolerance (colons, Devanagari
digits, full-width colon, spacing noise, stray glyphs, two pairs per line),
identifier/long-label preservation, bare-number rejection, dedup vs.
same-label-different-values, deterministic ids/spans, the 25/40 caps, and
quality-based retention. The extractor is NOT wired into the pipeline;
these tests exercise the pure module only.
"""

from __future__ import annotations

import sys

# Make sure we import the project copy, not any installed shadow.
_project_root = __import__("pathlib").Path(__file__).resolve().parents[1]
_sys_path_inserted = False
for _p in sys.path:
    if str(_project_root) == _p:
        _sys_path_inserted = True
        break
if not _sys_path_inserted:
    sys.path.insert(0, str(_project_root))

from src.candidate_extractor import (
    MAX_CANDIDATES_PER_DOCUMENT,
    MAX_CANDIDATES_PER_WINDOW,
    Candidate,
    deduplicate_candidates,
    extract_candidates,
    extract_candidates_for_window,
    extract_candidates_for_window_with_stats,
    extract_candidates_with_stats,
    has_usable_candidates,
    usable_candidate_count,
)


def _by_label(cands: list[Candidate], label: str) -> Candidate:
    """Single-candidate lookup by exact label text."""
    matches = [c for c in cands if c.label == label]
    assert matches, f"label {label!r} not found in {[c.label for c in cands]}"
    return matches[0]


# ---------------------------------------------------------------------------
# 1. Normal English label:value
# ---------------------------------------------------------------------------

def test_normal_english_label_value() -> None:
    cands = extract_candidates("RollNo : 2407510100067\nResult : PCP")
    assert len(cands) == 2
    assert _by_label(cands, "RollNo").value == "2407510100067"
    assert _by_label(cands, "Result").value == "PCP"


# ---------------------------------------------------------------------------
# 2. Hindi label:value (Devanagari label letters)
# ---------------------------------------------------------------------------

def test_hindi_label_value() -> None:
    cands = extract_candidates("नाम : देवेश विश्वकर्मा\nपिता का नाम : आशोक")
    assert len(cands) == 2
    assert _by_label(cands, "नाम").value == "देवेश विश्वकर्मा"
    assert _by_label(cands, "पिता का नाम").value == "आशोक"


# ---------------------------------------------------------------------------
# 3. Devanagari digits as corrupted separators
# ---------------------------------------------------------------------------

def test_devanagari_digit_separator() -> None:
    cands = extract_candidates(
        "SGPA ३ 6.05\nFather's Name ३ ASHOK VISHWAKARMA\nSession ४ 2024-25(REGULAR)"
    )
    assert _by_label(cands, "SGPA").value == "6.05"
    assert _by_label(cands, "Father's Name").value == "ASHOK VISHWAKARMA"
    assert _by_label(cands, "Session").value == "2024-25(REGULAR)"


def test_devanagari_digits_inside_values_not_shredded() -> None:
    # A Hindi numeral run in a value must never be treated as separators.
    cands = extract_candidates("Amount ३ १२३४५ रुपये")
    assert len(cands) == 1
    assert cands[0].value == "१२३४५ रुपये"


# ---------------------------------------------------------------------------
# 4. Full-width colon
# ---------------------------------------------------------------------------

def test_full_width_colon() -> None:
    cands = extract_candidates("Student Name：DEVESH VISHWAKARMA\n学号：2407510100067")
    assert _by_label(cands, "Student Name").value == "DEVESH VISHWAKARMA"
    assert _by_label(cands, "学号").value == "2407510100067"


# ---------------------------------------------------------------------------
# 5. OCR spacing noise
# ---------------------------------------------------------------------------

def test_ocr_spacing_noise() -> None:
    cands = extract_candidates(
        "  Roll   No   :    2407510100067  \n\tTotal Marks Obt.  :\t610 "
    )
    assert _by_label(cands, "Roll No").value == "2407510100067"
    assert _by_label(cands, "Total Marks Obt.").value == "610"


# ---------------------------------------------------------------------------
# 6. Stray glyph around separator (incl. attached & / loose s)
# ---------------------------------------------------------------------------

def test_stray_glyph_separators() -> None:
    cands = extract_candidates(
        "Course Code& (04) B.TECH\n"
        "Institute Code & (751)\n"
        "Date of Declaration s 30/06/25\n"
        "Result Status 3s CP( 1)\n"
        "Gender 3M"
    )
    assert _by_label(cands, "Course Code").value == "(04) B.TECH"
    assert _by_label(cands, "Institute Code").value == "(751)"
    assert _by_label(cands, "Date of Declaration").value == "30/06/25"
    assert _by_label(cands, "Result Status").value == "CP( 1)"
    assert _by_label(cands, "Gender").value == "M"


# ---------------------------------------------------------------------------
# 7. Two pairs on one line (monotonic region split)
# ---------------------------------------------------------------------------

def test_two_pairs_on_one_line() -> None:
    cands = extract_candidates(
        "RollNo : 2407510100067 EnrollmentNo : 240781010033745"
    )
    assert len(cands) == 2
    assert _by_label(cands, "RollNo").value == "2407510100067"
    assert _by_label(cands, "EnrollmentNo").value == "240781010033745"


def test_dense_multi_pair_line_with_digit_separators() -> None:
    # Real OCR: "Practical Subjects 3 4 Total Marks Obt. : 610" -- the second
    # digit separator must not steal the first pair's value.
    cands = extract_candidates(
        "Total Subjects 7 9 Theory Subjects : §\n"
        "Practical Subjects 3 4 Total Marks Obt. : 610"
    )
    practical = _by_label(cands, "Practical Subjects")
    total = _by_label(cands, "Total Marks Obt.")
    assert practical.value == "4"
    assert total.value == "610"


# ---------------------------------------------------------------------------
# 8. Long identifiers preserved (Roll / Enrollment Number)
# ---------------------------------------------------------------------------

def test_long_identifiers_preserved() -> None:
    cands = extract_candidates(
        "Roll Number: 2407510100067\nEnrollment Number: 240781010033745"
    )
    assert _by_label(cands, "Roll Number").value == "2407510100067"
    assert _by_label(cands, "Enrollment Number").value == "240781010033745"
    assert _by_label(cands, "Enrollment Number").value.isdigit()


# ---------------------------------------------------------------------------
# 9. Long label like Date of Declaration
# ---------------------------------------------------------------------------

def test_long_label_preserved() -> None:
    cands = extract_candidates("Date of Declaration : 30/06/25")
    assert _by_label(cands, "Date of Declaration").value == "30/06/25"


def test_long_multi_word_label_with_stopword_shapes() -> None:
    cands = extract_candidates("Name of the Candidate's Father : ASHOK VISHWAKARMA")
    # Multi-word labels survive; the 's possessive is kept inside a word.
    assert cands[0].value == "ASHOK VISHWAKARMA"
    assert "Father" in cands[0].label


# ---------------------------------------------------------------------------
# 10. Bare standalone numbers are NOT promoted
# ---------------------------------------------------------------------------

def test_bare_numbers_rejected() -> None:
    cands = extract_candidates(
        "12345\n30/06/25\n6.61\n2407510100067\n\nplain words only line"
    )
    assert cands == []
    assert usable_candidate_count(cands) == 0


def test_bare_number_after_clean_label_is_kept() -> None:
    # "NOT promoted" applies to label-less numbers; a labeled number is the
    # core candidate case.
    cands = extract_candidates("Total Marks Obt. : 610")
    assert _by_label(cands, "Total Marks Obt.").value == "610"


# ---------------------------------------------------------------------------
# 11. Same label with different values retained
# ---------------------------------------------------------------------------

def test_same_label_different_values_retained() -> None:
    cands = extract_candidates(
        "Semester : 1\nSGPA : 6.09\nSemester : 2\nSGPA : 6.05"
    )
    sgpas = [c.value for c in cands if c.label == "SGPA"]
    assert sgpas == ["6.09", "6.05"]
    sems = [c.value for c in cands if c.label == "Semester"]
    assert sems == ["1", "2"]


# ---------------------------------------------------------------------------
# 12. Exact duplicate candidates deduplicated
# ---------------------------------------------------------------------------

def test_exact_duplicates_deduplicated() -> None:
    cands = extract_candidates("RollNo : 2407510100067\nRollNo : 2407510100067")
    assert len(cands) == 1
    assert cands[0].value == "2407510100067"


def test_deduplicate_candidates_merges_only_exact_dups() -> None:
    a = Candidate(id=0, label="SGPA", value="6.09", line_no=3, char_span=(10, 20))
    b = Candidate(id=0, label="SGPA", value="6.09", line_no=3, char_span=(40, 50))
    c = Candidate(id=0, label="SGPA", value="6.05", line_no=9, char_span=(80, 90))
    d = Candidate(id=0, label="Roll No", value="6.09", line_no=4, char_span=(60, 70))
    merged = deduplicate_candidates([a, b, c, d])
    # a/b share (label, value) -> one; c differs in value; d in label.
    # Different spans/lines do NOT keep an otherwise identical pair alive.
    assert len(merged) == 3
    assert [m.id for m in merged] == [0, 1, 2]
    assert [(m.label, m.value) for m in merged] == [
        ("SGPA", "6.09"),
        ("SGPA", "6.05"),
        ("Roll No", "6.09"),
    ]


# ---------------------------------------------------------------------------
# 13. Deterministic ids, line numbers, and char spans
# ---------------------------------------------------------------------------

def test_ids_spans_lines_deterministic() -> None:
    text = "Alpha : 111\nBeta : 222\nGamma : 333"
    one = extract_candidates(text)
    two = extract_candidates(text)
    assert one == two  # dataclass equality covers id/label/value/line/span

    assert [c.id for c in one] == [0, 1, 2]
    assert [c.line_no for c in one] == [0, 1, 2]
    for cand in one:
        start, end = cand.char_span
        assert text[start:end].startswith(cand.label)
        assert cand.value in text[start:end]


def test_char_spans_are_absolute_into_full_text() -> None:
    text = "intro line\nRollNo : 2407510100067\ntail"
    cands = extract_candidates(text)
    cand = _by_label(cands, "RollNo")
    start, end = cand.char_span
    assert text[start:end].startswith("RollNo")
    assert "2407510100067" in text[start:end]
    assert cand.line_no == 1


def test_window_extraction_keeps_absolute_spans_and_ids() -> None:
    text = "head : x\n" + "RollNo : 2407510100067\n" + "tail : y"
    start = text.index("RollNo")
    end = text.index("tail")
    cands = extract_candidates_for_window(text, start, end)
    assert len(cands) == 1
    assert cands[0].id == 0  # ids are window-scoped
    s, e = cands[0].char_span
    assert text[s:e].startswith("RollNo")  # spans stay absolute
    assert cands[0].line_no == 1


# ---------------------------------------------------------------------------
# 14. Window cap = 25
# ---------------------------------------------------------------------------

def test_window_cap_is_25() -> None:
    text = "\n".join(f"Field{i:03d} : value{i:03d}" for i in range(60))
    cands = extract_candidates_for_window(text, 0, len(text))
    assert len(cands) == 25
    stats = extract_candidates_for_window_with_stats(text, 0, len(text))[1]
    assert stats.cap_hit is True
    assert stats.dropped_by_cap == 35
    assert stats.returned == 25


# ---------------------------------------------------------------------------
# 15. Document cap = 40
# ---------------------------------------------------------------------------

def test_document_cap_is_40() -> None:
    text = "\n".join(f"Field{i:03d} : value{i:03d}" for i in range(60))
    cands = extract_candidates(text)
    assert len(cands) == 40
    stats = extract_candidates_with_stats(text)[1]
    assert stats.cap_hit is True
    assert stats.dropped_by_cap == 20
    assert stats.raw_pairs == 60


# ---------------------------------------------------------------------------
# 16. Quality-based retention under the cap (not first-N)
# ---------------------------------------------------------------------------

def test_quality_based_retention_under_cap() -> None:
    # 45 low-quality candidates (noise values) + 5 high-quality ones
    # (numbers, decimals). The high-quality ones must survive the 40 cap
    # even though they appear last in document order.
    junk = "\n".join(f" filler{i:02d} : x" for i in range(45))
    gold = "\n".join(
        f"Important{i} : {i}.75" for i in range(5)
    )
    text = junk + "\n" + gold
    cands = extract_candidates(text)
    assert len(cands) == 40
    kept_values = {c.value for c in cands}
    for i in range(5):
        assert f"{i}.75" in kept_values


# ---------------------------------------------------------------------------
# Extra guards
# ---------------------------------------------------------------------------

def test_table_noise_lines_skipped() -> None:
    cands = extract_candidates(
        "IBEE101 [Fundamentals of Electrical Engineering [Theory [24 1 36 eB\n"
        "SGPA : 6.05"
    )
    assert [c.label for c in cands] == ["SGPA"]


def test_url_and_junk_penalized_but_not_crashing() -> None:
    text = "website : https://example.com/page\njunk : ~~~\nok : 42"
    cands = extract_candidates(text)
    assert _by_label(cands, "ok").value == "42"


def test_empty_and_garbage_inputs() -> None:
    assert extract_candidates("") == []
    assert extract_candidates("   \n\t\n") == []
    assert extract_candidates("::::::") == []
    assert extract_candidates("§ § §") == []
    assert has_usable_candidates([]) is False
    assert has_usable_candidates(
        extract_candidates("roll : 1\nname : 2\nmark : 3")
    ) is True
    # Single-character labels are deliberate quality-rejected OCR junk.
    assert extract_candidates("a : 1\nb : 2") == []


def test_pure_function_determinism_on_real_ocr_sample() -> None:
    text = (
        "RollNo : 2407510100067 EnrollmentNo : 240781010033745\n"
        "Father's Name ३ ASHOK VISHWAKARMA Gender 3M\n"
        "Session ४ 2024-25(REGULAR) Semesters : 1,2 Result : PCP Marks ३ 1198/1800\n"
        "Result Status 3s CP( 1) SGPA : 6.09\n"
        "Date of Declaration s 30/06/25\n"
    )
    one = extract_candidates_with_stats(text)
    two = extract_candidates_with_stats(text)
    assert one == two
    cands, stats = one
    labels = {c.label: c.value for c in cands}
    assert labels["RollNo"] == "2407510100067"
    assert labels["EnrollmentNo"] == "240781010033745"
    assert labels["Father's Name"] == "ASHOK VISHWAKARMA"
    assert labels["Gender"] == "M"
    assert labels["Session"] == "2024-25(REGULAR)"
    assert labels["Semesters"] == "1,2"
    assert labels["Result"] == "PCP"
    assert labels["Marks"] == "1198/1800"
    assert labels["SGPA"] == "6.09"
    assert labels["Date of Declaration"] == "30/06/25"
    assert stats.returned == len(cands)
    assert stats.cap_hit is False


def test_schema_shape_matches_approved_contract() -> None:
    cands = extract_candidates("Roll Number : 2407510100067")
    as_dict = cands[0].to_dict()
    assert set(as_dict.keys()) == {"id", "label", "value", "line_no", "char_span"}
    assert isinstance(as_dict["id"], int)
    assert isinstance(as_dict["char_span"], list) and len(as_dict["char_span"]) == 2


# ---------------------------------------------------------------------------
# Runner (project convention)
# ---------------------------------------------------------------------------

def run_all() -> None:
    tests = [
        test_normal_english_label_value,
        test_hindi_label_value,
        test_devanagari_digit_separator,
        test_devanagari_digits_inside_values_not_shredded,
        test_full_width_colon,
        test_ocr_spacing_noise,
        test_stray_glyph_separators,
        test_two_pairs_on_one_line,
        test_dense_multi_pair_line_with_digit_separators,
        test_long_identifiers_preserved,
        test_long_label_preserved,
        test_long_multi_word_label_with_stopword_shapes,
        test_bare_numbers_rejected,
        test_bare_number_after_clean_label_is_kept,
        test_same_label_different_values_retained,
        test_exact_duplicates_deduplicated,
        test_deduplicate_candidates_merges_only_exact_dups,
        test_ids_spans_lines_deterministic,
        test_char_spans_are_absolute_into_full_text,
        test_window_extraction_keeps_absolute_spans_and_ids,
        test_window_cap_is_25,
        test_document_cap_is_40,
        test_quality_based_retention_under_cap,
        test_table_noise_lines_skipped,
        test_url_and_junk_penalized_but_not_crashing,
        test_empty_and_garbage_inputs,
        test_pure_function_determinism_on_real_ocr_sample,
        test_schema_shape_matches_approved_contract,
    ]

    failed: list[str] = []

    for test in tests:
        try:
            test()
        except AssertionError as exc:
            failed.append(f"{test.__name__}: {exc}")
        except Exception as exc:
            failed.append(f"{test.__name__}: RAISED {type(exc).__name__}: {exc}")

    if failed:
        print("Candidate extractor checks FAILED:")
        for line in failed:
            print(" -", line)
        raise SystemExit(1)

    print(f"Candidate extractor checks PASSED: {len(tests)}")


if __name__ == "__main__":
    run_all()
