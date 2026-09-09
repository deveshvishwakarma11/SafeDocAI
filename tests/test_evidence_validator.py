"""
Evidence validator focused regression checks.

These are narrow checks for the fake-field / false-verification risks
discussed in the Step 1 review, not a full replacement for the existing
evidence_validator.py contract.
"""

from __future__ import annotations

import sys
import types

# Make sure we import the project copy, not any installed shadow.
_project_root = __import__("pathlib").Path(__file__).resolve().parents[1]
_sys_path_inserted = False
for _p in sys.path:
    if str(_project_root) == _p:
        _sys_path_inserted = True
        break
if not _sys_path_inserted:
    sys.path.insert(0, str(_project_root))

from src.evidence_validator import (
    is_value_supported_by_text,
    validate_fields,
    validate_document_type,
)


def _ocr(text: str) -> str:
    """Normalize a little so tests read like real OCR samples."""
    return text.strip()


# ---------------------------------------------------------------------------
# Multi-word names: verifying on a single component should fail
# ---------------------------------------------------------------------------

def test_multi_word_name_not_verified_by_partial_component() -> None:
    ocr_text = _ocr("ASHA KUMARI")
    # LLM returns full name, evidence snippet is the full name -> should verify
    assert is_value_supported_by_text(
        value="ASHA KUMARI",
        evidence_text=ocr_text,
        evidence_snippet="ASHA KUMARI",
        require_exact_snippet_match=True,
    )

    # Partial component should not verify a multi-word value
    assert not is_value_supported_by_text(
        value="ASHA KUMARI",
        evidence_text=_ocr("ASHA"),
        evidence_snippet="ASHA",
        require_exact_snippet_match=True,
    )


# ---------------------------------------------------------------------------
# Plain amounts/percentages should not be forced into identifier mode
# ---------------------------------------------------------------------------

def test_amount_values_support_flexible_matching() -> None:
    ocr_text = _ocr("Total marks obtained: 1269 out of 1900")

    # Numeric values with commas should still normalize to their compact form
    # and match the same number present in the OCR evidence.
    assert is_value_supported_by_text(
        value="1,269",
        evidence_text=ocr_text,
        evidence_snippet="1269",
        require_exact_snippet_match=True,
    )

    # Multi-part Indian-style comma amounts should also normalize.
    ocr_text_indian = _ocr("Total villagers: 120000")
    assert is_value_supported_by_text(
        value="1,20,000",
        evidence_text=ocr_text_indian,
        evidence_snippet="120000",
        require_exact_snippet_match=True,
    )

    # Decimal amounts should normalize their dot away so they can match
    # compact numeric evidence.
    assert is_value_supported_by_text(
        value="1200.50",
        evidence_text=_ocr("Balance: 120050"),
        evidence_snippet="120050",
        require_exact_snippet_match=True,
    )

    # Currency-prefixed amounts should still be treated as numeric-like and
    # match compact evidence when punctuation is normalized away.
    assert is_value_supported_by_text(
        value="1200",
        evidence_text=_ocr("Total: Rs. 1200"),
        evidence_snippet="1200",
        require_exact_snippet_match=True,
    )

    # Pure percentage values are not identifiers and may match directly.
    assert is_value_supported_by_text(
        value="85%",
        evidence_text=_ocr("SGPA: 85%"),
        evidence_snippet="85%",
        require_exact_snippet_match=True,
    )


# ---------------------------------------------------------------------------
# PAN / IFSC: must use strict match, and must not pass on unrelated text
# ---------------------------------------------------------------------------

def test_pan_strict_match() -> None:
    ocr_text = _ocr("Permanent Account Number: ABCDE1234F")

    assert is_value_supported_by_text(
        value="ABCDE1234F",
        evidence_text=ocr_text,
        evidence_snippet="ABCDE1234F",
        require_exact_snippet_match=True,
    )

    # PAN must not verify against random Indian-name text
    assert not is_value_supported_by_text(
        value="ABCDE1234F",
        evidence_text=_ocr("Name: ASHA KUMARI"),
        evidence_snippet="ASHA KUMARI",
        require_exact_snippet_match=True,
    )


def test_ifsc_strict_match() -> None:
    ocr_text = _ocr("IFSC: ABCD0123456")

    assert is_value_supported_by_text(
        value="ABCD0123456",
        evidence_text=ocr_text,
        evidence_snippet="ABCD0123456",
        require_exact_snippet_match=True,
    )

    # IFSC must not verify against a plain numeric table
    assert not is_value_supported_by_text(
        value="ABCD0123456",
        evidence_text=_ocr("Roll No: 12345678"),
        evidence_snippet="12345678",
        require_exact_snippet_match=True,
    )


# ---------------------------------------------------------------------------
# Snippet verification is meaningful, not decorative
# ---------------------------------------------------------------------------

def test_evidence_snippet_must_appear_in_text() -> None:
    ocr_text = _ocr("Roll No: 2407510100067")

    # Value present, but snippet fabricated -> should fail
    assert not is_value_supported_by_text(
        value="2407510100067",
        evidence_text=ocr_text,
        evidence_snippet="ROLL: 999999999999",
        require_exact_snippet_match=True,
    )


# ---------------------------------------------------------------------------
# UNKNOWN / empty / N/A style values should be treated as not needing evidence
# ---------------------------------------------------------------------------

def test_unknown_values_are_not_rejected() -> None:
    assert is_value_supported_by_text(
        value="UNKNOWN",
        evidence_text="",
        evidence_snippet="",
        require_exact_snippet_match=True,
    )
    assert is_value_supported_by_text(
        value="N/A",
        evidence_text="irrelevant",
        evidence_snippet="",
        require_exact_snippet_match=True,
    )
    assert is_value_supported_by_text(
        value="",
        evidence_text="some text",
        evidence_snippet="",
        require_exact_snippet_match=True,
    )


def test_missing_evidence_text_blocks_verification() -> None:
    assert not is_value_supported_by_text(
        value="Some Real Value",
        evidence_text="",
        evidence_snippet="Some Real Value",
        require_exact_snippet_match=True,
    )


# ---------------------------------------------------------------------------
# validate_fields pipeline behaviour
# ---------------------------------------------------------------------------

def test_validate_fields_keeps_unverified_when_not_required() -> None:
    fields = [
        {"key": "name", "value": "ASHA KUMARI", "evidence_snippet": "ASHA KUMARI"},
        {"key": "pan", "value": "FAKEPAN1234X", "evidence_snippet": "no evidence"},
    ]
    ocr_text = _ocr("ASHA KUMARI")

    validated = validate_fields(fields, ocr_text, require_all_verified=False)

    assert len(validated) == 2
    assert validated[0]["verified"] is True
    assert validated[1]["verified"] is False


def test_validate_fields_drops_unverified_when_required() -> None:
    fields = [
        {"key": "name", "value": "ASHA KUMARI", "evidence_snippet": "ASHA KUMARI"},
        {"key": "pan", "value": "FAKEPAN1234X", "evidence_snippet": "no evidence"},
    ]
    ocr_text = _ocr("ASHA KUMARI")

    validated = validate_fields(fields, ocr_text, require_all_verified=True)

    assert len(validated) == 1
    assert validated[0]["key"] == "name"
    assert validated[0]["verified"] is True


# ---------------------------------------------------------------------------
# Document type validation should not accept garbage indicator tokens
# ---------------------------------------------------------------------------

def test_pan_card_type_detection_uses_clean_indicators() -> None:
    ocr_text = _ocr(
        "GOVERNMENT OF INDIA\n"
        "INCOME TAX DEPARTMENT\n"
        "Permanent Account Number\n"
        "PAN: ABCDE1234F\n"
    )

    _, verified = validate_document_type(
        document_type="pan card",
        evidence_text=ocr_text,
        heuristic_type="PAN_CARD",
    )
    assert verified is True


def test_pan_card_type_rejected_when_no_supporting_evidence() -> None:
    ocr_text = _ocr("This is a completely unrelated document.")

    _, verified = validate_document_type(
        document_type="pan card",
        evidence_text=ocr_text,
        heuristic_type=None,
    )
    assert verified is False


def test_document_type_accepted_when_type_name_appears_in_evidence() -> None:
    ocr_text = _ocr("RATION CARD FAMILY DETAILS")

    _, verified = validate_document_type(
        document_type="RATION CARD",
        evidence_text=ocr_text,
        heuristic_type=None,
    )
    assert verified is True


# ---------------------------------------------------------------------------
# Heuristic cross-check should still count as evidence, not override it
# ---------------------------------------------------------------------------

def test_heuristic_match_can_verify_document_type() -> None:
    ocr_text = _ocr("Marks: 1269/1900 Semester: 4")

    _, verified = validate_document_type(
        document_type="MARKSHEET",
        evidence_text=ocr_text,
        heuristic_type="MARKSHEET",
    )
    assert verified is True


def run_all() -> None:
    tests = [
        test_multi_word_name_not_verified_by_partial_component,
        test_amount_values_support_flexible_matching,
        test_pan_strict_match,
        test_ifsc_strict_match,
        test_evidence_snippet_must_appear_in_text,
        test_unknown_values_are_not_rejected,
        test_missing_evidence_text_blocks_verification,
        test_validate_fields_keeps_unverified_when_not_required,
        test_validate_fields_drops_unverified_when_required,
        test_pan_card_type_detection_uses_clean_indicators,
        test_pan_card_type_rejected_when_no_supporting_evidence,
        test_document_type_accepted_when_type_name_appears_in_evidence,
        test_heuristic_match_can_verify_document_type,
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
        print("Evidence validator checks FAILED:")
        for line in failed:
            print(" -", line)
        raise SystemExit(1)

    print(f"Evidence validator checks PASSED: {len(tests)}")


if __name__ == "__main__":
    run_all()
