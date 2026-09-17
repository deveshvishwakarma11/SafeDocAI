"""
SafeDocAI - Evidence Validation Layer

Validates LLM-extracted fields against the ORIGINAL full OCR text.
Ensures that extracted information is actually supported by evidence.

Key principles:
- Validate against ORIGINAL full OCR text, not truncated LLM context
- Normalize case, whitespace, and Unicode
- Be conservative: false information is worse than missing information
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any


def normalize_for_comparison(text: str) -> str:
    """
    Normalize text for evidence comparison.
    
    - Unicode normalization (NFC)
    - Case folding
    - Whitespace normalization
    - Remove some common OCR artifacts
    
    This makes comparison tolerant to minor OCR differences.
    """
    if not text:
        return ""
    
    # Unicode normalization
    normalized = unicodedata.normalize('NFC', text)
    
    # Case folding for case-insensitive comparison
    normalized = normalized.casefold()
    
    # Normalize whitespace (collapse multiple to single space)
    normalized = re.sub(r'\s+', ' ', normalized.strip())
    
    return normalized


def normalize_value_for_matching(value: str) -> str:
    """
    Normalize an extracted value for matching against OCR text.
    
    Handles:
    - Case differences
    - Whitespace differences
    - Common OCR substitutions
    - Punctuation variations

    Numeric punctuation commonly used in Indian documents (commas,
    periods in decimal amounts) is also normalized so values like
    "1,269" can match "1269" in the OCR text.
    """
    if not value:
        return ""
    
    # Unicode normalization
    normalized = unicodedata.normalize('NFC', value)
    
    # Case folding
    normalized = normalized.casefold()
    
    # Normalize whitespace
    normalized = re.sub(r'\s+', ' ', normalized.strip())
    
    # Normalize numeric punctuation so OCR variants of amounts still match.
    # Indian amounts often appear as "1,269", "1,20,000", "1200.50", etc.
    normalized = re.sub(
        r'[,\u2026\u2009\u200a\u202f\u00a0\u2060\u200b]+',
        '',
        normalized,
    )

    # Normalize decimal amounts so "1200.50" can match "120050" in OCR text.
    # If the value is all digits/dots after normalization, also expose a
    # dot-stripped variant used only for numeric matching.
    if re.fullmatch(r'[0-9.]+', normalized):
        normalized_without_dot = normalized.replace('.', '')
        if normalized_without_dot:
            normalized = normalized_without_dot
    
    return normalized


def extract_search_terms(value: str) -> list[str]:
    """
    Extract searchable terms from a value.
    
    For a value like "KUMKUM VISHWAKARMA", we want to search for
    both the full name and individual components.
    
    Returns list of normalized search terms (longest first).
    """
    if not value:
        return []
    
    # Normalize
    normalized = normalize_value_for_matching(value)
    
    if not normalized:
        return []
    
    terms = []
    
    # Add the full normalized value
    terms.append(normalized)
    
    # Split on whitespace and add components
    parts = normalized.split()
    for part in parts:
        if len(part) > 2:  # Avoid very short matches
            terms.append(part)
    
    # Sort by length (longest first) to try most specific matches first
    terms = sorted(set(terms), key=len, reverse=True)
    
    return terms


def is_value_supported_by_text(
    value: str,
    evidence_text: str,
    evidence_snippet: str = "",
    require_exact_snippet_match: bool = True,
) -> bool:
    """
    Check if a value is actually supported by the OCR evidence text.
    
    This is the core validation function. It checks whether the
    LLM-extracted value can be found (or reasonably inferred from)
    the original OCR text.
    
    For multi-word values: requires the COMPLETE normalized value
    to be present, not just individual components.
    
    For identifier-like values (PAN, IFSC, numbers, etc.): requires
    STRICT normalized match with NO fuzzy matching.
    
    Args:
        value: The LLM-extracted value to validate
        evidence_text: The ORIGINAL full OCR text (not truncated!)
        evidence_snippet: The snippet the LLM claimed as evidence
        require_exact_snippet_match: If True, verify the snippet
            appears in the evidence text. If False, only check value.
    
    Returns:
        True if the value is supported by evidence, False otherwise
    """
    if not value or value.strip().upper() in ("UNKNOWN", "", "NONE", "N/A"):
        # "UNKNOWN" values don't need evidence
        return True
    
    if not evidence_text:
        # No evidence text available - can't validate
        return False
    
    # Normalize for comparison
    value_normalized = normalize_value_for_matching(value)
    evidence_normalized = normalize_for_comparison(evidence_text)
    
    if not value_normalized or not evidence_normalized:
        return False
    
    # Determine if this is an identifier-like value that requires strict matching
    is_identifier = _is_identifier_value(value)
    
    # For multi-word values, require the COMPLETE value to be present
    # Do NOT accept verification based on individual components only
    # The full normalized value must appear in the evidence
    if is_identifier:
        # Identifiers require strict normalized match - NO fuzzy matching
        found = _strict_match_in_text(value_normalized, evidence_normalized)
    elif _looks_like_numeric_value(value_normalized):
        # Non-identifier numeric values (amounts, totals, counts) sometimes
        # appear in OCR with different separators, so compare both the
        # normalized form and a separator-stripped numeric form.
        found = _strict_match_in_text(value_normalized, evidence_normalized) or _strict_match_in_text(
            _normalize_numeric_like_value(value_normalized),
            _normalize_ocr_numeric_field(evidence_normalized),
        )
    else:
        # For non-identifiers, try full value first, then components if full fails
        found = _flexible_match_in_text(value_normalized, evidence_normalized)
    
    if not found:
        return False
    
    # Verify the evidence_snippet if required.
    # This check runs AFTER value support is already confirmed, so we
    # are validating BOTH the value AND the snippet the model claimed.
    if require_exact_snippet_match and evidence_snippet:
        snippet_normalized = normalize_for_comparison(evidence_snippet)
        if snippet_normalized:
            # The snippet MUST be supported by the OCR text.
            # For multi-word snippets we require the full snippet to be present,
            # not just one word, to avoid accepting loose fabricated snippets.
            if not _strict_match_in_text(snippet_normalized, evidence_normalized):
                return False
    
    return True


def _normalize_numeric_like_value(value_normalized: str) -> str:
    """Normalize a numeric-like value for flexible evidence matching.

    This is only used for non-identifier numeric values such as amounts,
    totals, counts, and percentages. It strips separators and, when the
    value is purely numeric/dot after normalization, also removes dots so
    "1,269" and "1200.50" can match compact forms found in OCR text.
    """
    collapsed = re.sub(r'[,\s\-]+', '', value_normalized)

    if re.fullmatch(r'[0-9.]+', collapsed):
        collapsed = collapsed.replace('.', '')

    return collapsed


def _normalize_ocr_numeric_field(value_normalized: str) -> str:
    """Normalize numeric-like OCR evidence in the same family as values.

    This is intentionally conservative: it only removes separators that
    the value normalizer already targets, so a snippet like "120000"
    stays "120000" while "1,20,000" becomes "120000".
    """
    return _normalize_numeric_like_value(value_normalized)


def _looks_like_numeric_value(value_normalized: str) -> bool:
    """Return True if the normalized value looks like a numeric amount/slug.

    This is used only to decide whether numeric-like non-identifier values
    should get relaxed separator handling. It deliberately does NOT include
    PAN/IFSC/Aadhaar/phone/date patterns.
    """
    if not value_normalized:
        return False

    # Pure number, optionally with commas/dots/whitespace already removed
    # by the caller's normalization.
    if re.fullmatch(r'[0-9]+', value_normalized):
        return True

    # Value that still contains %, INR, Rs, ₹ etc. is numeric-like.
    if re.search(r'[%₹\u20b9rs\.0-9]', value_normalized):
        return True

    return False


def _is_identifier_value(value: str) -> bool:
    """
    Check if a value looks like an identifier that requires strict matching.
    
    Identifiers include: PAN, IFSC, Aadhaar, account numbers, roll numbers,
    registration numbers, phone numbers, dates, other ID-like values.

    Numeric-only values are NOT treated as identifiers here, because they
    are usually amounts/slugs/numbers that should be compared flexibly.
    """
    if not value:
        return False
    
    normalized = normalize_value_for_matching(value)
    
    # PAN pattern: 5 letters + 4 digits + 1 letter (e.g., ABCDE1234F)
    if re.match(r'^[a-z]{5}\d{4}[a-z]$', normalized):
        return True
    
    # IFSC pattern: 4 letters + 0 + 6 alphanumeric (e.g., ABCD0123456)
    if re.match(r'^[a-z]{4}0[a-z0-9]{6}$', normalized):
        return True
    
    # Aadhaar pattern: 12 digits (with optional spaces)
    digits_only = re.sub(r'[\s-]', '', normalized)
    if re.match(r'^\d{12}$', digits_only):
        return True
    
    # Phone numbers: 10 digits starting with 6-9 (Indian mobile)
    if re.match(r'^[6-9]\d{9}$', digits_only):
        return True
    
    # Dates: DD/MM/YYYY, DD-MM-YYYY, DD Mon YYYY, etc.
    if re.match(r'^\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}$', normalized):
        return True
    if re.match(r'^\d{1,2}\s+[a-z]{3,9}\s+\d{4}$', normalized):
        return True
    
    # Alphanumeric IDs like roll/registration/enrollment numbers.
    # Keep them strict, but avoid treating plain numbers this way.
    if re.match(r'^[a-z][a-z0-9]{7,}$', normalized):
        return True
    
    return False


def _strict_match_in_text(term: str, text: str) -> bool:
    """
    Check if term exists in text with STRICT matching.
    
    No fuzzy matching, no character substitutions.
    Only exact normalized match or word-boundary match.
    """
    if not term or not text:
        return False
    
    # Direct substring match
    if term in text:
        return True
    
    # Word boundary match (handles values split across lines or with punctuation)
    word_pattern = r'\b' + re.escape(term) + r'\b'
    if re.search(word_pattern, text):
        return True
    
    # For hyphenated or spaced values, try without spaces/hyphens
    # This handles cases like "1234 5678" vs "12345678"
    term_no_spaces = re.sub(r'[\s-]+', '', term)
    if len(term_no_spaces) >= 4 and term_no_spaces != term:
        text_no_spaces = re.sub(r'[\s-]+', '', text)
        if term_no_spaces in text_no_spaces:
            return True
    
    return False


def _flexible_match_in_text(value_normalized: str, evidence_normalized: str) -> bool:
    """
    Check if a non-identifier value is supported by evidence.
    
    CRITICAL RULE: For multi-word values, the COMPLETE value must be present.
    Do NOT verify a multi-word value merely because one component appears.
    
    This prevents false verification like accepting "ASHA KUMARI" just because
    "ASHA" appears somewhere in the text.
    """
    if not value_normalized or not evidence_normalized:
        return False
    
    # For multi-word values, require the COMPLETE value to be present
    words = value_normalized.split()
    
    if len(words) > 1:
        # MULTI-WORD VALUE: Require complete match
        # First, try direct substring match
        if value_normalized in evidence_normalized:
            return True
        
        # Word boundary match for complete value
        word_pattern = r'\b' + re.escape(value_normalized) + r'\b'
        if re.search(word_pattern, evidence_normalized):
            return True
        
        # Check if words appear in sequence with flexible spacing
        # This handles cases like "ASHA  KUMARI" (double space) or 
        # "ASHA, KUMARI" (with punctuation)
        pattern = r'\s+'.join(re.escape(w) for w in words)
        if re.search(pattern, evidence_normalized):
            return True
        
        # DO NOT fall back to individual components for multi-word values
        # The complete value must be present
        return False
    else:
        # SINGLE-WORD VALUE: Can check the word directly
        if value_normalized in evidence_normalized:
            return True
        
        word_pattern = r'\b' + re.escape(value_normalized) + r'\b'
        if re.search(word_pattern, evidence_normalized):
            return True
        
        return False



def validate_fields(
    fields: list[dict[str, Any]],
    evidence_text: str,
    require_all_verified: bool = False,
) -> list[dict[str, Any]]:
    """
    Validate a list of LLM-extracted fields against OCR evidence.
    
    Args:
        fields: List of {key, value, evidence_snippet} from LLM
        evidence_text: ORIGINAL full OCR text
        require_all_verified: If True, drop unverified fields entirely
            instead of marking them as unverified
    
    Returns:
        List of validated fields with 'verified' boolean added
    """
    if not fields:
        return []
    
    validated_fields = []
    
    for field in fields:
        if not isinstance(field, dict):
            continue
        
        key = str(field.get("key", "")).strip()
        value = str(field.get("value", "")).strip()
        evidence_snippet = str(field.get("evidence_snippet", "")).strip()
        
        # Validate this field
        is_verified = is_value_supported_by_text(
            value=value,
            evidence_text=evidence_text,
            evidence_snippet=evidence_snippet,
            require_exact_snippet_match=True,
        )
        
        validated_field = {
            "key": key,
            "value": value,
            "evidence_snippet": evidence_snippet,
            "verified": is_verified,
        }
        
        # Optionally drop unverified fields
        if require_all_verified and not is_verified:
            continue
        
        validated_fields.append(validated_field)
    
    return validated_fields


def validate_document_type(
    document_type: str,
    evidence_text: str,
    heuristic_type: str | None = None,
) -> tuple[str, bool]:
    """
    Validate that the LLM's document_type classification is supported by evidence.
    
    This is important for catching hallucinations like the ration card
    being classified as "Voter ID Card".
    
    Args:
        document_type: LLM-extracted document type
        evidence_text: ORIGINAL full OCR text
        heuristic_type: Optional heuristic classification for comparison
    
    Returns:
        Tuple of (validated_document_type, is_verified)
    """
    if not document_type or document_type.strip().upper() == "UNKNOWN":
        return ("UNKNOWN", True)
    
    if not evidence_text:
        return (document_type, False)
    
    # Normalize
    type_normalized = normalize_value_for_matching(document_type)
    evidence_normalized = normalize_for_comparison(evidence_text)
    
    # Check if the document type appears in the evidence
    # This is a basic check - more sophisticated validation could
    # look for type-specific patterns
    
    # Direct match
    if type_normalized in evidence_normalized:
        return (document_type, True)
    
    # Word boundary match
    if re.search(r'\b' + re.escape(type_normalized) + r'\b', evidence_normalized):
        return (document_type, True)
    
    # If heuristic also classified it, and they match, that's strong evidence
    if heuristic_type and normalize_value_for_matching(heuristic_type) == type_normalized:
        return (document_type, True)
    
    # Check for key patterns that indicate the document type
    # For example, "marksheet" should have "marks", "semester", etc.
    type_indicators = {
        "marksheet": ["marks", "semester", "sgpa", "cgpa", "roll", "result", "university", "board"],
        "ration card": ["ration", "खाद्य", "रसद", "family", "unit", "fair price"],
        "pan card": ["income tax", "permanent account number", "pan"],
        "voter id": ["voter", "electoral", "epic", "constituency"],
        "aadhaar": ["aadhaar", "uidai", "unique identification"],
        "passport": ["passport", "date of issue", "place of issue"],
        "bank statement": ["bank", "ifsc", "account", "transaction", "balance"],
    }
    
    # Normalize type keywords for matching (replace spaces/underscores)
    def normalize_type_key(key: str) -> str:
        """Normalize type key for matching (handle spaces vs underscores)."""
        return re.sub(r'[\s_-]+', ' ', key.lower().strip())
    
    for type_keyword, indicators in type_indicators.items():
        # Normalize both for comparison
        norm_type_key = normalize_type_key(type_keyword)
        norm_type_normalized = normalize_type_key(type_normalized)
        
        if norm_type_key in norm_type_normalized or norm_type_normalized in norm_type_key:
            # Check if at least one indicator appears in evidence
            found_indicator = False
            for indicator in indicators:
                indicator_norm = normalize_value_for_matching(indicator)
                if indicator_norm in evidence_normalized:
                    found_indicator = True
                    break
            
            if found_indicator:
                return (document_type, True)
            else:
                # Type claims to be X but evidence doesn't support it
                return (document_type, False)
    
    # Default: if we can't verify, mark as uncertain
    # but don't reject outright - let the user decide
    return (document_type, False)


def calculate_overall_confidence(
    validated_fields: list[dict[str, Any]],
    document_type_verified: bool,
    heuristic_confidence: float = 0.0,
    heuristic_used: bool = False,
) -> str:
    """
    Calculate overall confidence level based on validation results.
    
    Confidence levels:
    - HIGH: document_type verified + most fields verified
    - MEDIUM: document_type verified OR many fields verified
    - LOW: little verification success
    
    Args:
        validated_fields: List of fields with 'verified' status
        document_type_verified: Whether document type was verified
        heuristic_confidence: Confidence from heuristic classifier (0-1)
        heuristic_used: Whether heuristic classification was used
    
    Returns:
        "HIGH", "MEDIUM", or "LOW"
    """
    if not validated_fields:
        if document_type_verified and heuristic_used:
            return "MEDIUM"
        return "LOW"
    
    total_fields = len(validated_fields)
    verified_fields = sum(1 for f in validated_fields if f.get("verified", False))
    
    verified_ratio = verified_fields / total_fields if total_fields > 0 else 0
    
    # Determine confidence
    if document_type_verified and verified_ratio >= 0.7:
        return "HIGH"
    elif document_type_verified or verified_ratio >= 0.5:
        return "MEDIUM"
    else:
        return "LOW"


def determine_classification_source(
    heuristic_type: str | None,
    llm_type: str | None,
    heuristic_confidence: float,
    llm_type_verified: bool,
) -> str:
    """
    Determine the classification source for the final output.
    
    Returns:
        "HEURISTIC" - Only heuristic was used/verified
        "LLM" - Only LLM was used/verified
        "HYBRID" - Both contributed
        "UNKNOWN" - Neither provided useful classification
    """
    if heuristic_type and heuristic_confidence >= 0.67:
        if llm_type_verified or (llm_type and llm_type.upper() == heuristic_type.upper()):
            return "HYBRID"
        return "HEURISTIC"
    
    if llm_type and llm_type_verified:
        return "LLM"
    
    if llm_type:
        return "LLM"  # LLM classified but unverified
    
    if heuristic_type:
        return "HEURISTIC"  # Heuristic only
    
    return "UNKNOWN"
