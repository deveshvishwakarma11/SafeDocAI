"""
Step 2 tests: Generic Document Evidence Layer (plain-Python runner).
====================================================================

Covers the approved Step 2 contract, generic (no per-document-type logic):

1.  Railway Ticket classification (registry family, real-layout fixture).
2.  Application Form classification (not MARKSHEET).
3.  PNR candidate discovery (header-row-above layout, exact char span).
4.  Enrollment ID candidate discovery (caption-below layout + variants).
5.  Evidence validation stays authoritative (contradiction guard + fields).
6.  No-fabrication behavior when the field/value is absent.
7.  Regression for existing document types (marksheet, PAN, ration).
8.  Dynamic-field behavior remains generic (vocabulary is language-level,
    candidates only appear when the label is present in the OCR).
9.  No broad false-positive classification.
10. Priority merge into Phase 3 output (verified respected, unverified
    same-family replaced, absent families added; storage never touched).

Real documents are NOT read here (unit-level fixtures mirror their
evidence); real-data behavior is verified separately in the sandbox runs.
"""

from __future__ import annotations

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parents[1]
for _entry in (str(_project_root), str(_project_root / "src")):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

from src.document_evidence import (
    IDENTIFIER_FAMILIES,
    NEW_DOCUMENT_TYPE_EVIDENCE,
    claim_is_contradicted,
    classify_new_families,
    discover_identifier_fields,
    family_evidence_score,
    normalize_family_name,
    priority_identifier_fields,
)
from src.evidence_validator import validate_document_type, validate_fields
from src.heuristics import classify_document

# ---------------------------------------------------------------------------
# Fixtures mirroring the REAL OCR evidence shapes (railway ticket header-row
# table, application form caption-below values, boilerplate lines).
# ---------------------------------------------------------------------------

TICKET_TEXT = """E-TICKET : ELECTRONIC RESERVATION SLIP
IRCTC e-ticketing Portal

PNR Train No./Name Class
4143027140 03252/SMVB DNR SPL SLEEPER CLASS (SL)

Waitlist #12 | Quota: GENERAL | Berth: SL
Transaction ID: 100006773136446

Note: This e-ticket is valid only with one of the following identity
cards in original: Voter Identity Card / Passport / PAN Card / Aadhaar Card
"""

FORM_TEXT = """GATE 2027 APPLICATION FORM
Provisional Application for Examination

DEVESH V
(Full Name of the Applicant)

M103B71
(Enrollment ID)

Declaration: I declare that the particulars given above are true. I am
aware that my eligibility and results are subject to scrutiny.
Signature with e-Signature
"""

MARKSHEET_TEXT = """Dr. A.P.J. Abdul Kalam Technical University
Semester: IV
SGPA: 8.2
Roll No: 2010301
Result: PASS
"""

PAN_TEXT = """INCOME TAX DEPARTMENT
Permanent Account Number Card
ABCDE1234F
Government of India
"""

RATION_TEXT = "खाद्य एवं रसद विभाग\nउचित दर दुकान\nराशन कार्ड\n"

PLAIN_TEXT = (
    "The voter list was displayed at the district office for review. "
    "Residents were requested to check their entries."
)


# ---------------------------------------------------------------------------
# 1. Railway Ticket classification
# ---------------------------------------------------------------------------

def test_railway_ticket_classifies_from_strong_evidence():
    doc_type, confidence = classify_document(TICKET_TEXT)
    assert doc_type == "RAILWAY_TICKET", f"got {doc_type}"
    assert confidence >= 0.67


def test_railway_ticket_beats_single_voter_keyword():
    # One boilerplate "Voter Identity Card" word must NOT produce a voter
    # classification for a ticket full of ticket evidence.
    doc_type, _ = classify_document(TICKET_TEXT)
    assert doc_type != "VOTER_ID_CARD"


def test_railway_ticket_registry_result_shape():
    result = classify_new_families(TICKET_TEXT)
    assert result is not None
    assert result["document_type"] == "RAILWAY_TICKET"
    assert result["strong_matches"] >= 2
    assert result["evidence_source"] == "evidence_registry"


# ---------------------------------------------------------------------------
# 2. Application Form classification
# ---------------------------------------------------------------------------

def test_application_form_classifies():
    doc_type, confidence = classify_document(FORM_TEXT)
    assert doc_type == "APPLICATION_FORM", f"got {doc_type}"
    assert confidence >= 0.67


def test_application_form_not_marksheet():
    doc_type, _ = classify_document(FORM_TEXT)
    assert doc_type != "MARKSHEET"


# ---------------------------------------------------------------------------
# 3. PNR candidate discovery (header-row layout)
# ---------------------------------------------------------------------------

def test_pnr_discovered_from_header_row_layout():
    fields = discover_identifier_fields(TICKET_TEXT)
    pnr = [f for f in fields if f["family"] == "PNR"]
    assert pnr, f"no PNR discovered: {fields}"
    assert pnr[0]["value"] == "4143027140"
    assert pnr[0]["layout"] == "header_row"
    start, end = pnr[0]["char_span"]
    assert TICKET_TEXT[start:end] == "4143027140"  # exact span into raw text


def test_transaction_id_discovered_from_same_line_label():
    fields = discover_identifier_fields(TICKET_TEXT)
    txn = [f for f in fields if f["family"] == "Transaction ID"]
    assert txn and txn[0]["value"] == "100006773136446"
    assert txn[0]["layout"] == "same_line"


def test_discovery_char_spans_are_absolute():
    for record in discover_identifier_fields(TICKET_TEXT + FORM_TEXT):
        start, end = record["char_span"]
        blob = TICKET_TEXT + FORM_TEXT
        assert blob[start:end] == record["value"]


# ---------------------------------------------------------------------------
# 4. Enrollment ID candidate discovery (caption-below layout + variants)
# ---------------------------------------------------------------------------

def test_enrollment_id_discovered_from_caption_below_layout():
    fields = discover_identifier_fields(FORM_TEXT)
    enr = [f for f in fields if f["family"] == "Enrollment ID"]
    assert enr, f"no Enrollment ID discovered: {fields}"
    assert enr[0]["value"] == "M103B71"
    assert enr[0]["layout"] == "caption_below"
    start, end = enr[0]["char_span"]
    assert FORM_TEXT[start:end] == "M103B71"


def test_enrollment_label_variants_all_recognized():
    for caption in (
        "(Enrollment ID)", "(Enrolment ID)", "(Enrollment No)",
        "(Enrolment No)", "(Enrollment Number)",
    ):
        text = "AB12CD34\n" + caption + "\n"
        fields = discover_identifier_fields(text)
        assert any(
            f["family"] == "Enrollment ID" and f["value"] == "AB12CD34"
            for f in fields
        ), f"variant not recognized: {caption} -> {fields}"


def test_id_is_not_over_generalized():
    # A bare "ID" or unrelated "Student ID" label is NOT an Enrollment ID.
    text = "STU-9911\n(Student ID)\n"
    fields = discover_identifier_fields(text)
    assert not any(f["family"] == "Enrollment ID" for f in fields)


# ---------------------------------------------------------------------------
# 5. Evidence validation stays authoritative
# ---------------------------------------------------------------------------

def test_ticket_voter_claim_is_contradicted():
    assert claim_is_contradicted(TICKET_TEXT, "VOTER_ID_CARD") is True


def test_ticket_railway_claim_is_not_contradicted():
    assert claim_is_contradicted(TICKET_TEXT, "RAILWAY_TICKET") is False


def test_wrong_claim_unverified_on_ticket():
    verified_type, verified = validate_document_type(
        "VOTER_ID_CARD", TICKET_TEXT, "RAILWAY_TICKET"
    )
    assert verified is False  # boilerplate word no longer verifies
    assert verified_type == "VOTER_ID_CARD"  # value echoed, not rewritten


def test_correct_claim_verified_on_ticket():
    _, verified = validate_document_type(
        "Railway Ticket", TICKET_TEXT, "RAILWAY_TICKET"
    )
    assert verified is True  # free-form naming normalizes to the family


def test_form_marksheet_claim_is_contradicted():
    assert claim_is_contradicted(FORM_TEXT, "MARKSHEET") is True


def test_form_marksheet_claim_unverified():
    _, verified = validate_document_type(
        "MARKSHEET", FORM_TEXT, "APPLICATION_FORM"
    )
    assert verified is False


def test_form_correct_claim_verified():
    _, verified = validate_document_type(
        "APPLICATION_FORM", FORM_TEXT, "APPLICATION_FORM"
    )
    assert verified is True


def test_validator_unchanged_without_registry_evidence():
    # No registry-family evidence -> legacy behavior fully preserved.
    assert claim_is_contradicted(PLAIN_TEXT, "VOTER_ID_CARD") is False


# ---------------------------------------------------------------------------
# 6. No-fabrication behavior when the field/value is absent
# ---------------------------------------------------------------------------

def test_no_pnr_candidate_without_value_line():
    text = "PNR\n\nStatus: Confirmed\nTransaction ID: 100006773136446\n"
    fields = discover_identifier_fields(text)
    assert not any(f["family"] == "PNR" for f in fields)


def test_no_enrollment_candidate_without_value_line():
    text = "GATE 2027 APPLICATION FORM\nProvisional Application\n(Enrollment ID)\n"
    fields = discover_identifier_fields(text)
    assert not any(f["family"] == "Enrollment ID" for f in fields)


def test_empty_text_discovers_nothing():
    assert discover_identifier_fields("") == []
    assert discover_identifier_fields("   \n  \n") == []


def test_absent_labels_produce_no_candidates():
    fields = discover_identifier_fields(PLAIN_TEXT)
    assert fields == []


def test_nothing_invented_in_ticket_discovery():
    # Every discovered value must exist verbatim in the source text.
    for record in discover_identifier_fields(TICKET_TEXT):
        assert record["value"] in TICKET_TEXT


# ---------------------------------------------------------------------------
# 7. Regression for existing document types
# ---------------------------------------------------------------------------

def test_marksheet_still_classifies():
    doc_type, _ = classify_document(MARKSHEET_TEXT)
    assert doc_type == "MARKSHEET"


def test_marksheet_claim_still_verifies():
    _, verified = validate_document_type("MARKSHEET", MARKSHEET_TEXT, "MARKSHEET")
    assert verified is True


def test_pan_card_still_classifies_and_verifies():
    doc_type, _ = classify_document(PAN_TEXT)
    assert doc_type == "PAN_CARD"
    _, verified = validate_document_type("PAN_CARD", PAN_TEXT, "PAN_CARD")
    assert verified is True


def test_ration_card_still_classifies_and_verifies():
    doc_type, _ = classify_document(RATION_TEXT)
    assert doc_type == "RATION_CARD"
    _, verified = validate_document_type("RATION_CARD", RATION_TEXT, "RATION_CARD")
    assert verified is True


def test_voter_claim_on_marksheet_not_contradicted():
    # No registry-family evidence in a marksheet -> guard stays silent.
    assert claim_is_contradicted(MARKSHEET_TEXT, "VOTER_ID_CARD") is False


# ---------------------------------------------------------------------------
# 8. Dynamic-field behavior remains generic
# ---------------------------------------------------------------------------

def test_registry_addition_requires_no_pipeline_change():
    # A future family is ONE registry entry; the pipeline reads the registry
    # generically (this is a structural contract test over the public data).
    for family_name, family in NEW_DOCUMENT_TYPE_EVIDENCE.items():
        assert set(family) == {"strong_patterns", "medium_patterns"}
        assert isinstance(family["strong_patterns"], tuple)


def test_identifier_vocabulary_is_language_level_not_per_type():
    # Families are keyed by label vocabulary, not document type; every
    # family has at least one variant regex.
    assert IDENTIFIER_FAMILIES, "vocabulary must not be empty"
    for canonical, variants in IDENTIFIER_FAMILIES:
        assert canonical and canonical.strip()
        assert variants, f"{canonical} has no label variants"


def test_score_and_normalization_helpers():
    assert family_evidence_score(TICKET_TEXT, "RAILWAY_TICKET")[0] >= 2
    assert family_evidence_score(TICKET_TEXT, "NOT_A_FAMILY") == (0, 0)
    assert normalize_family_name("Railway Ticket") == "RAILWAY_TICKET"
    assert normalize_family_name("railway-ticket") == "RAILWAY_TICKET"
    assert normalize_family_name("MARKSHEET") is None
    assert normalize_family_name(None) is None


def test_classify_new_families_none_for_plain_text():
    assert classify_new_families(PLAIN_TEXT) is None


# ---------------------------------------------------------------------------
# 9. No broad false-positive classification
# ---------------------------------------------------------------------------

def test_plain_text_does_not_classify():
    doc_type, confidence = classify_document(PLAIN_TEXT)
    assert doc_type == "UNKNOWN"
    assert confidence == 0.0


def test_marksheet_guard_returns_unknown_for_form_text():
    # The marksheet negative guard: form boilerplate without marksheet
    # evidence must not produce (or support) a marksheet classification.
    from src.heuristics import get_document_type_confidence
    result = get_document_type_confidence(FORM_TEXT)
    assert result["document_type"] != "MARKSHEET"


# ---------------------------------------------------------------------------
# 10. Priority merge into Phase 3 output (pure; no storage touched)
# ---------------------------------------------------------------------------

def _merge(validated_fields, raw_text):
    from src.document_understanding import _merge_priority_identifier_fields
    return _merge_priority_identifier_fields(validated_fields, raw_text)


def test_merge_adds_evidence_verified_priority_fields():
    merged = _merge([], TICKET_TEXT)
    keys = {f["key"] for f in merged}
    assert "PNR" in keys and "Transaction ID" in keys
    for field in merged:
        assert field["verified"] is True  # validation failure would drop it
        assert field["evidence_snippet"] == field["value"]


def test_merge_respects_existing_verified_family():
    existing = [{"key": "PNR", "value": "4143027140",
                 "evidence_snippet": "4143027140", "verified": True}]
    merged = _merge(existing, TICKET_TEXT)
    pnr_fields = [f for f in merged if f["key"] == "PNR"]
    assert len(pnr_fields) == 1  # never duplicated
    assert pnr_fields[0] is existing[0]  # untouched


def test_merge_replaces_unverified_same_family_field():
    existing = [{"key": "PNR", "value": "999", "evidence_snippet": "999",
                 "verified": False}]
    merged = _merge(existing, TICKET_TEXT)
    pnr_fields = [f for f in merged if f["key"] == "PNR"]
    assert len(pnr_fields) == 1
    assert pnr_fields[0]["value"] == "4143027140"
    assert pnr_fields[0]["verified"] is True


def test_merge_adds_nothing_for_id_card_like_text():
    assert _merge([], PLAIN_TEXT) == []


def test_merge_survives_internal_errors_defensively():
    # A broken evidence layer must never break the main pipeline.
    import src.document_understanding as du
    original = du.priority_identifier_fields
    du.priority_identifier_fields = lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("boom")
    )
    try:
        existing = [{"key": "Name", "value": "DEVESH V",
                     "evidence_snippet": "DEVESH V", "verified": True}]
        assert _merge(existing, FORM_TEXT) == existing
    finally:
        du.priority_identifier_fields = original


def test_priority_fields_pass_authoritative_validation():
    # Independent proof: every emitted priority field would be verified=True
    # under the plain validator too (validator stays the sole authority).
    boosted = priority_identifier_fields(FORM_TEXT)
    assert boosted, "Enrollment ID should be discovered on the form fixture"
    for field in boosted:
        validated = validate_fields(
            [{"key": field["key"], "value": field["value"],
              "evidence_snippet": field["evidence_snippet"]}],
            FORM_TEXT,
            require_all_verified=True,
        )
        assert len(validated) == 1 and validated[0]["verified"] is True


def test_validate_fields_drops_fabricated_value():
    fabricated = [{"key": "PNR", "value": "9999999999",
                   "evidence_snippet": "PNR: 9999999999"}]
    validated = validate_fields(fabricated, TICKET_TEXT,
                                require_all_verified=True)
    assert validated == []  # unverified -> dropped, never stored


# ---------------------------------------------------------------------------

def run_all() -> None:
    tests = [
        test_railway_ticket_classifies_from_strong_evidence,
        test_railway_ticket_beats_single_voter_keyword,
        test_railway_ticket_registry_result_shape,
        test_application_form_classifies,
        test_application_form_not_marksheet,
        test_pnr_discovered_from_header_row_layout,
        test_transaction_id_discovered_from_same_line_label,
        test_discovery_char_spans_are_absolute,
        test_enrollment_id_discovered_from_caption_below_layout,
        test_enrollment_label_variants_all_recognized,
        test_id_is_not_over_generalized,
        test_ticket_voter_claim_is_contradicted,
        test_ticket_railway_claim_is_not_contradicted,
        test_wrong_claim_unverified_on_ticket,
        test_correct_claim_verified_on_ticket,
        test_form_marksheet_claim_is_contradicted,
        test_form_marksheet_claim_unverified,
        test_form_correct_claim_verified,
        test_validator_unchanged_without_registry_evidence,
        test_no_pnr_candidate_without_value_line,
        test_no_enrollment_candidate_without_value_line,
        test_empty_text_discovers_nothing,
        test_absent_labels_produce_no_candidates,
        test_nothing_invented_in_ticket_discovery,
        test_marksheet_still_classifies,
        test_marksheet_claim_still_verifies,
        test_pan_card_still_classifies_and_verifies,
        test_ration_card_still_classifies_and_verifies,
        test_voter_claim_on_marksheet_not_contradicted,
        test_registry_addition_requires_no_pipeline_change,
        test_identifier_vocabulary_is_language_level_not_per_type,
        test_score_and_normalization_helpers,
        test_classify_new_families_none_for_plain_text,
        test_plain_text_does_not_classify,
        test_marksheet_guard_returns_unknown_for_form_text,
        test_merge_adds_evidence_verified_priority_fields,
        test_merge_respects_existing_verified_family,
        test_merge_replaces_unverified_same_family_field,
        test_merge_adds_nothing_for_id_card_like_text,
        test_merge_survives_internal_errors_defensively,
        test_priority_fields_pass_authoritative_validation,
        test_validate_fields_drops_fabricated_value,
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
        print("Step 2 document evidence checks FAILED:")
        for line in failed:
            print(" -", line)
        raise SystemExit(1)

    print(f"Step 2 document evidence checks PASSED: {len(tests)}")


if __name__ == "__main__":
    run_all()
