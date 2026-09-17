"""
SafeDocAI - Deterministic Heuristic Pre-Classifier

A fast, pure-Python document type classifier that runs BEFORE the LLM.
Uses regex/string matching with strong evidence patterns.

This is a pre-check/fallback layer, NOT the entire document-understanding system.
It provides deterministic evidence that can be combined with LLM results.

Requirements:
- Pure Python
- Regex/string matching
- Very fast
- No AI call
- No network
- No hardcoded filename dependency
"""

from __future__ import annotations

import re
from typing import Any, Tuple


# ============================================================================
# Document Type Classification
# ============================================================================

# Define document types with their evidence patterns
# Each type requires MULTIPLE strong signals to avoid false positives

DOCUMENT_TYPE_PATTERNS = {
    "RATION_CARD": {
        "strong_patterns": [
            # Official department names
            (r"खाद्य\s*एवं\s*रसद", "Food & Civil Supplies Department"),
            (r"खाद्य एवं रसद विभाग", "Food & Civil Supplies Department"),
            (r"राशन\s*कार्ड", "Ration Card"),
            (r"राशन काड", "Ration Card (variant)"),
            # Ration-specific terminology
            (r"उचित\s*दर\s*दुकान", "Fair Price Shop"),
            (r"उचित दर दुकान", "Fair Price Shop"),
            (r"ईंट\u200cकार्ड|आधार\s*कार्ड\s*संख्या", "Aadhaar/Ration card number"),
            # Family member structure typical in ration cards
            (r"पारिवारिक\s*सदस्य|RATION CARD FAMILY", "Family member section"),
        ],
        "medium_patterns": [
            r"जनपद",  # District (common in Indian govt docs)
            r"गैस\s*सब्सिडी|gas\s*subsidy",  # LPG subsidy (often on ration cards)
        ]
    },
    "PAN_CARD": {
        "strong_patterns": [
            (r"INCOME TAX DEPARTMENT", "Income Tax Department"),
            (r"Permanent Account Number", "PAN"),
            (r"पैन\s*कार्ड|PAN\s*CARD", "PAN Card"),
            (r"कर\s*विभाग|Tax\s*Department", "Tax Department"),
        ],
        "medium_patterns": [
            r"[A-Z]{5}[0-9]{4}[A-Z]",  # PAN number pattern
            r"आयकर|INCOME\s*TAX",  # Income tax
        ]
    },
    "MARKSHEET": {
        "strong_patterns": [
            (r"TECHNICAL UNIVERSITY", "Technical University"),
            (r"विश्वविद्यालय|UNIVERSITY", "University"),
            (r"BOARD OF (HIGH SCHOOL|SECONDARY)", "Education Board"),
            (r"बोर्ड\s*ऑफ\s*हाई\s*स्कूल|बोर्ड\s*ऑफ\s*सेकंडरी", "Education Board"),
            (r"SGPA|CGPA", "Grade Point Average"),
            (r"Semester|सेमेस्टर", "Semester"),
            (r"Result|परिणाम", "Result"),
            (r"Marks|अंक|मार्क्स", "Marks"),
            (r"Roll\s*No|रोल\s*नンバー", "Roll Number"),
        ],
        "medium_patterns": [
            r"Student|छात्र",
            r"Institute|कॉलेज|College",
            r"Subject|विषय",
            r"Grade|ग्रेड",
            r"Pass|पास|FAILED",
            r"Date of Declaration|परिणाम की घोषणा",
        ]
    },
    "VOTER_ID_CARD": {
        "strong_patterns": [
            (r"ELECTORAL ROLL", "Electoral Roll"),
            (r"EPIC\s*Number|VOTER\s*ID", "Voter ID"),
            (r"निर्वाचन\s*पत्र|VOTER", "Voter Document"),
            (r"Assembly\s*Constitency|निर्वाचन\s*क्षेत्र", "Constituency"),
        ],
        "medium_patterns": [
            r"Name of\s*Father|Husband",
            r"Part\s*No",
            r"Police\s*Station",
            r"File\s*No",
        ]
    },
    "AADHAAR_CARD": {
        "strong_patterns": [
            (r"आधार\s*कार्ड|AADHAAR\s*CARD", "Aadhaar Card"),
            (r"Unique Identification Authority", "UIDAI"),
            (r"मेरा\s*आधार|MY\s*AADHAAR", "My Aadhaar"),
        ],
        "medium_patterns": [
            r"12[\s-]?digit|12\s*आधार",
            r"Enrollment\s*No",
        ]
    },
    "BANK_STATEMENT": {
        "strong_patterns": [
            (r"BANK\s*STATEMENT|ACCOUNT\s*STATEMENT", "Bank Statement"),
            (r"IFSC", "IFSC Code"),
            (r"Account\s*No|खाता\s*संख्या", "Account Number"),
            (r"Transaction|लेन\u2011देन", "Transaction"),
        ],
        "medium_patterns": [
            r"Cr\.|DB\.",
            r"Balance|शेष",
            r"Cheque|चेक",
        ]
    },
    "PASSPORT": {
        "strong_patterns": [
            (r"पासपोर्ट|PASSPORT", "Passport"),
            (r"Date\s*of\s*Issue|जारी\s*की\s*तारीख", "Issue Date"),
            (r"Place\s*of\s*Issue|जारी\s*स्थान", "Place of Issue"),
        ],
        "medium_patterns": [
            r"Passport\s*No",
            r"Father's\s*Name",
            r"MANTRALAYAM|MEERUT",  # Known passport offices
        ]
    },
}


def normalize_text(text: str) -> str:
    """
    Normalize text for pattern matching.
    
    - Collapse whitespace
    - Keep Unicode characters intact
    """
    if not text:
        return ""
    
    normalized = re.sub(r'\s+', ' ', text.strip())
    
    return normalized


def _match_pattern_list(text_normalized: str, patterns: list) -> int:
    """
    Count how many patterns from a list match the text.
    
    Returns the number of distinct pattern matches.
    """
    matches = 0
    
    for pattern_info in patterns:
        if isinstance(pattern_info, tuple):
            pattern = pattern_info[0]
        else:
            pattern = pattern_info
        
        if re.search(pattern, text_normalized, re.IGNORECASE | re.UNICODE):
            matches += 1
    
    return matches


def _classify_core(text: str) -> tuple[str, float, int, int]:
    """
    Core classification logic shared by the tuple and dict APIs.

    Returns:
        (document_type, confidence, strong_matches, medium_matches)
    """
    if not text or not text.strip():
        return ("UNKNOWN", 0.0, 0, 0)

    text_normalized = normalize_text(text)

    best_type = "UNKNOWN"
    best_score = 0.0
    best_strong = 0
    best_medium = 0

    for doc_type, patterns in DOCUMENT_TYPE_PATTERNS.items():
        strong_matches = _match_pattern_list(
            text_normalized,
            patterns["strong_patterns"],
        )
        medium_matches = _match_pattern_list(
            text_normalized,
            patterns["medium_patterns"],
        )

        # Strong patterns are worth 1.0, medium patterns are worth 0.5
        evidence_score = (strong_matches * 1.0) + (medium_matches * 0.5)

        # Need at least 2 strong patterns, OR 1 strong + 2 medium
        is_classified = (
            strong_matches >= 2 or
            (strong_matches >= 1 and medium_matches >= 2)
        )

        if is_classified and evidence_score > best_score:
            best_type = doc_type
            best_score = evidence_score
            best_strong = strong_matches
            best_medium = medium_matches

    if best_type == "UNKNOWN":
        return ("UNKNOWN", 0.0, 0, 0)

    confidence = min(1.0, best_score / 3.0)

    return (best_type, confidence, best_strong, best_medium)


def classify_document(text: str) -> Tuple[str, float]:
    """
    Classify a document based on text evidence.
    
    Returns a tuple of (document_type, confidence).
    """
    doc_type, confidence, _strong, _medium = _classify_core(text)
    return (doc_type, confidence)


def _describe_pattern(pattern_info: Any) -> tuple[str, str]:
    """Extract (regex, description) from a pattern entry."""
    if isinstance(pattern_info, tuple):
        return pattern_info[0], pattern_info[1]

    return pattern_info, f"Pattern: {pattern_info}"


def _collect_evidence_details(text_normalized: str, patterns: list, kind: str) -> list[dict[str, Any]]:
    """Return matched pattern details for one evidence tier."""
    details: list[dict[str, Any]] = []

    for pattern_info in patterns:
        pattern, description = _describe_pattern(pattern_info)

        if re.search(pattern, text_normalized, re.IGNORECASE | re.UNICODE):
            details.append({
                "type": kind,
                "pattern": pattern,
                "description": description,
            })

    return details


def get_document_type_confidence(
    text: str
) -> dict[str, Any]:
    """
    Get detailed classification information.
    
    Returns a dict with:
    - document_type: The classified type
    - confidence: Float 0-1
    - strong_matches: Number of strong pattern matches
    - medium_matches: Number of medium pattern matches
    - evidence_details: List of matched patterns with descriptions
    """
    if not text or not text.strip():
        return {
            "document_type": "UNKNOWN",
            "confidence": 0.0,
            "strong_matches": 0,
            "medium_matches": 0,
            "evidence_details": [],
        }

    doc_type, confidence, strong, medium = _classify_core(text)
    text_normalized = normalize_text(text)

    if doc_type == "UNKNOWN":
        return {
            "document_type": "UNKNOWN",
            "confidence": 0.0,
            "strong_matches": 0,
            "medium_matches": 0,
            "evidence_details": [],
        }

    evidence_details = [
        * _collect_evidence_details(text_normalized, DOCUMENT_TYPE_PATTERNS[doc_type]["strong_patterns"], "strong"),
        * _collect_evidence_details(text_normalized, DOCUMENT_TYPE_PATTERNS[doc_type]["medium_patterns"], "medium"),
    ]

    return {
        "document_type": doc_type,
        "confidence": round(confidence, 3),
        "strong_matches": strong,
        "medium_matches": medium,
        "evidence_details": evidence_details,
    }


# ============================================================================
# Utility Functions
# ============================================================================

def is_high_confidence_classification(confidence: float) -> bool:
    """
    Check if classification confidence is high enough to use without LLM.
    
    Returns True if confidence >= 0.67 (roughly 2+ strong patterns).
    """
    return confidence >= 0.67


def should_skip_llm_for_type(
    document_type: str,
    confidence: float
) -> bool:
    """Determine if LLM can be skipped for this classification.
    
    Currently, we still want LLM for field extraction even if type is
    confidently identified, but this can be tuned.
    """
    return False


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    import sys

    # Use utf-8 for stdout
    sys.stdout.reconfigure(encoding='utf-8')
    
    print("=" * 60)
    print("SafeDocAI - Heuristics Pre-Classifier Test")
    print("=" * 60)
    
    # Test with sample texts
    test_cases = [
        # PAN Card
        (
            "GOVERNMENT OF INDIA\n"
            "INCOME TAX DEPARTMENT\n"
            "Permanent Account Number Card\n"
            "Name: ASHA KUMARI\n"
            "Father's Name: RAM KUMAR\n"
            "Date of Birth: 15/08/1992\n"
            "Permanent Account Number: ABCDE1234F\n"
        ),
        
        # Ration Card (with corrupted text that still has key patterns)
        (
            "khady ate varasad vibhag\n"
            "uttar pradesh\n"
            "janpad : Gorakhpur\n"
            "ration card sankhya 218840359519\n"
            "parivar ke sadasyon ka vivaran\n"
        ),
        
        # University Marksheet
        (
            "Dr. APJ Abdul Kalam Technical University\n"
            "Student Result\n"
            "Roll No: 2407510100067\n"
            "Semester: 4\n"
            "SGPA: 6.61\n"
            "Marks: 1269/1900\n"
        ),
        
        # Unknown document (should return UNKNOWN)
        (
            "This is just a random document\n"
            "with no identifiable patterns.\n"
        ),
    ]
    
    for i, test_text in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i} ---")
        result = get_document_type_confidence(test_text)
        print(f"Document type: {result['document_type']}")
        print(f"Confidence: {result['confidence']}")
        print(f"Strong matches: {result['strong_matches']}")
        print(f"Medium matches: {result['medium_matches']}")
    
    print("\n" + "=" * 60)
    print("Heuristics test complete.")
    print("=" * 60)
