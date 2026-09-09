from __future__ import annotations

import argparse
import json
import logging
import re
import string
import unicodedata
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pdfplumber
import pytesseract
from PIL import Image
from pytesseract import Output

from utils import (
    preprocess_for_ocr,
    validate_file,
)


# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SAMPLES_DIR = PROJECT_ROOT / "data" / "samples"
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".jpg",
    ".jpeg",
    ".png",
}


# ---------------------------------------------------------
# LOGGING
# ---------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# OCR LANGUAGE
# ---------------------------------------------------------

def get_ocr_language() -> str:
    """
    Use English + Hindi when Hindi traineddata is available.
    Otherwise fall back to English.
    """

    try:
        languages = pytesseract.get_languages(config="")
    except Exception:
        languages = []

    if "hin" in languages and "eng" in languages:
        return "eng+hin"

    if "eng" in languages:
        return "eng"

    if "hin" in languages:
        return "hin"

    return "eng"


OCR_LANGUAGE = get_ocr_language()


# ---------------------------------------------------------
# EMBEDDED TEXT CORRUPTION DETECTION
# ---------------------------------------------------------

def is_text_corrupt(text: str) -> bool:
    """
    Evaluate whether PDF embedded text is unreliable/corrupt.

    Returns True if the text shows signs of:
    - Font encoding corruption (cid: patterns)
    - Malformed/non-printable characters
    - Obvious encoding garbage
    - Suspiciously short text for the content

    Criteria:
    1. "(cid:" occurs anywhere in the text
    2. High proportion of malformed characters
    3. Encoding garbage like "αñ", "\ufffd", etc.
    4. Text is suspiciously short compared to content
    """

    if not text:
        return True

    text_lower = text.lower()

    # Criterion 1: cid: patterns indicate font encoding corruption
    if "(cid:" in text_lower or "cid:" in text_lower:
        return True

    # Criterion 2: Check for high proportion of non-printable chars
    # Allow normal whitespace, common punctuation, and Unicode chars
    non_printable_count = 0
    total_chars = len(text)

    if total_chars == 0:
        return True

    for char in text:
        code_point = ord(char)

        # Allow:
        # - Printable ASCII (32-126)
        # - Common Unicode categories (letters, marks, numbers, punctuation)
        # - Whitespace (space, tab, newline, etc.)
        if (32 <= code_point <= 126):
            continue
        if char in '\t\n\r':
            continue
        # Check if it's a valid Unicode letter, mark, number, or punct
        category = unicodedata.category(char)
        if category.startswith(('L', 'M', 'N', 'P')):
            continue
        # Control characters (except tab/newline already handled)
        if code_point < 32 or code_point == 127:
            non_printable_count += 1
            continue
        # Sudden large code points (possible encoding artifacts)
        if code_point > 0x2000 and code_point not in (
            0x2018, 0x2019, 0x201C, 0x201D,  # Smart quotes
            0x2026,  # Ellipsis
            0x2013, 0x2014,  # En/em dash
            0x2022,  # Bullet
            0x2122,  # Trademark
            0x0900, 0x097F,  # Devanagari
            0x0980, 0x09FF,  # Bengali
            0x0B80, 0x0BFF,  # Tamil
            0x0C80, 0x0CFF,  # Telugu
            0x0E80, 0x0EFF,  # Thai (partial)
            0x3000, 0x303F,  # CJK symbols
            0x4E00, 0x9FFF,  # CJK Unified (partial)
            0xFF00, 0xFFEF,  # Fullwidth forms
        ):
            non_printable_count += 1

    non_printable_ratio = non_printable_count / total_chars

    # If more than 15% of characters look corrupted, flag it
    if non_printable_ratio > 0.15:
        return True

    # Criterion 3: Detect obvious encoding garbage patterns
    garbage_patterns = [
        r"[\x00-\x08\x0b\x0c\x0e-\x1f]",  # Control chars (except tab/newline)
        r"αñ",  # Example from the problem statement
        r"�",  # Unicode replacement character
        r"Ã",  # mojibake patterns (high-bit character followed by nothing)
        r"Â",  # mojibake patterns (high-bit character followed by nothing)
    ]

    for pattern in garbage_patterns:
        if re.search(pattern, text):
            return True

    return False


# ---------------------------------------------------------
# IMAGE OCR
# ---------------------------------------------------------

def ocr_image(image: Image.Image) -> dict[str, Any]:
    """
    OCR a PIL image and return text + OCR metadata.
    """

    image = image.convert("RGB")

    image_array = np.array(image)

    processed = preprocess_for_ocr(image_array)

    text = pytesseract.image_to_string(
        processed,
        lang=OCR_LANGUAGE,
        config="--psm 6",
    )

    data = pytesseract.image_to_data(
        processed,
        lang=OCR_LANGUAGE,
        config="--psm 6",
        output_type=Output.DICT,
    )

    words = []

    total = len(data.get("text", []))

    for i in range(total):
        word = data["text"][i].strip()

        if not word:
            continue

        try:
            confidence = float(data["conf"][i])
        except (ValueError, TypeError):
            confidence = -1.0

        words.append(
            {
                "text": word,
                "confidence": confidence,
                "left": data["left"][i],
                "top": data["top"][i],
                "width": data["width"][i],
                "height": data["height"][i],
            }
        )

    return {
        "text": text.strip(),
        "words": words,
    }


# ---------------------------------------------------------
# PDF EXTRACTION WITH SMART OCR FALLBACK
# ---------------------------------------------------------

def extract_pdf_text(file_path: Path) -> tuple[str, list[dict[str, Any]], str]:
    """
    Extract PDF text with smart OCR fallback.

    1. Attempt embedded text extraction first.
    2. Evaluate text quality using is_text_corrupt().
    3. If embedded text is corrupt, discard it and OCR all pages.
    4. Pages with little/no text are always OCR processed.

    Returns:
        tuple of (text, ocr_pages, extraction_method)
        - text: extracted or OCR text
        - ocr_pages: list of page info dicts
        - extraction_method: "embedded_text", "ocr_fallback", or "ocr"
    """

    all_text = []
    ocr_pages = []
    extraction_method = "embedded_text"

    try:
        import fitz

        with pdfplumber.open(file_path) as pdf:
            plumber_pages = pdf.pages

            pdf_document = fitz.open(file_path)

            # Step 1: Extract all embedded text first for quality check
            embedded_texts = []
            for page_index, page in enumerate(plumber_pages):
                embedded_text = page.extract_text() or ""
                embedded_texts.append(embedded_text)

            # Step 2: Check overall text quality
            combined_embedded = "\n".join(embedded_texts)

            if is_text_corrupt(combined_embedded):
                logger.warning(
                    "Embedded text appears corrupt for %s. "
                    "Switching to OCR fallback.",
                    file_path.name,
                )
                extraction_method = "ocr_fallback"

                # OCR all pages instead of using embedded text
                for page_index, page in enumerate(plumber_pages):
                    if page_index >= len(pdf_document):
                        continue

                    pdf_page = pdf_document[page_index]

                    pix = pdf_page.get_pixmap(
                        dpi=250,
                        alpha=False,
                    )

                    image = Image.frombytes(
                        "RGB",
                        [pix.width, pix.height],
                        pix.samples,
                    )

                    ocr_result = ocr_image(image)

                    if ocr_result["text"]:
                        all_text.append(ocr_result["text"])

                    ocr_pages.append(
                        {
                            "page": page_index + 1,
                            "text": ocr_result["text"],
                            "words": ocr_result["words"],
                            "extraction_method": "ocr",
                        }
                    )

            else:
                # Use embedded text, but OCR pages with little/no text
                for page_index, page in enumerate(plumber_pages):
                    embedded_text = embedded_texts[page_index]

                    if len(embedded_text.strip()) >= 20:
                        all_text.append(embedded_text.strip())
                        continue

                    # Page has little text, OCR it
                    if page_index >= len(pdf_document):
                        continue

                    pdf_page = pdf_document[page_index]

                    pix = pdf_page.get_pixmap(
                        dpi=250,
                        alpha=False,
                    )

                    image = Image.frombytes(
                        "RGB",
                        [pix.width, pix.height],
                        pix.samples,
                    )

                    ocr_result = ocr_image(image)

                    if ocr_result["text"]:
                        all_text.append(ocr_result["text"])

                    ocr_pages.append(
                        {
                            "page": page_index + 1,
                            "text": ocr_result["text"],
                            "words": ocr_result["words"],
                            "extraction_method": "ocr",
                        }
                    )

            pdf_document.close()

    except Exception as exc:
        logger.warning(
            "PDF extraction failed for %s: %s",
            file_path.name,
            exc,
        )

        raise

    return "\n\n".join(all_text).strip(), ocr_pages, extraction_method


# ---------------------------------------------------------
# IMAGE FILE EXTRACTION
# ---------------------------------------------------------

def extract_image_text(file_path: Path) -> tuple[str, list[dict[str, Any]], str]:
    """
    OCR a standalone image file.

    Returns:
        tuple of (text, ocr_pages, extraction_method)
        - text: OCR text
        - ocr_pages: list of page info dicts
        - extraction_method: always "ocr" for images
    """

    image = Image.open(file_path)

    result = ocr_image(image)

    return (
        result["text"],
        [
            {
                "page": 1,
                "text": result["text"],
                "words": result["words"],
                "extraction_method": "ocr",
            }
        ],
        "ocr",
    )


# ---------------------------------------------------------
# ENTITY EXTRACTION
# ---------------------------------------------------------

def extract_entities(text: str) -> dict[str, list[str]]:
    """
    Basic deterministic entity extraction.

    This remains Phase 1 extraction.
    Dynamic document-specific understanding belongs
    to document_understanding.py.
    """

    entities = {
        "dob": [],
        "pan": [],
        "ifsc": [],
        "phone": [],
        "email": [],
        "amounts": [],
    }

    if not text:
        return entities

    # Dates
    date_patterns = [
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
        r"\b\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}\b",
        r"\b[A-Za-z]{3,9}\s+\d{1,2},\s+\d{4}\b",
    ]

    for pattern in date_patterns:
        entities["dob"].extend(
            re.findall(pattern, text, flags=re.IGNORECASE)
        )

    # PAN
    pan_matches = re.findall(
        r"\b[A-Z]{5}[0-9]{4}[A-Z]\b",
        text.upper(),
    )
    entities["pan"].extend(pan_matches)

    # IFSC
    ifsc_matches = re.findall(
        r"\b[A-Z]{4}0[A-Z0-9]{6}\b",
        text.upper(),
    )
    entities["ifsc"].extend(ifsc_matches)

    # Phone
    phone_matches = re.findall(
        r"(?<!\d)(?:\+91[\s-]?)?[6-9]\d{9}(?!\d)",
        text,
    )
    entities["phone"].extend(phone_matches)

    # Email
    email_matches = re.findall(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        text,
    )
    entities["email"].extend(email_matches)

    # Amounts
    amount_matches = re.findall(
        r"(?:₹|Rs\.?|INR)\s*[\d,]+(?:\.\d{1,2})?",
        text,
        flags=re.IGNORECASE,
    )
    entities["amounts"].extend(amount_matches)

    # Remove duplicates while preserving order
    for key in entities:
        entities[key] = list(dict.fromkeys(entities[key]))

    return entities


# ---------------------------------------------------------
# SINGLE FILE PARSER
# ---------------------------------------------------------

def parse_document(file_path: str | Path) -> dict[str, Any]:
    """
    Parse one PDF/image document.
    
    Returns a dict with parsed document data including raw_text,
    extracted entities, and extraction method.
    """

    path = Path(file_path)

    if not path.exists():
        return {
            "file_name": path.name,
            "file_type": path.suffix.upper().replace(".", ""),
            "raw_text": "",
            "extracted_entities": {},
            "status": "error",
            "error": "File does not exist.",
        }

    try:
        validate_file(path)

        extension = path.suffix.lower()

        if extension not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: {extension}"
            )

        if extension == ".pdf":
            raw_text, pages, extraction_method = extract_pdf_text(path)
            file_type = "PDF"

        else:
            raw_text, pages, extraction_method = extract_image_text(path)
            file_type = "IMAGE"

        entities = extract_entities(raw_text)

        return {
            "file_name": path.name,
            "file_type": file_type,
            "file_path": str(path),
            "raw_text": raw_text,
            "pages": pages,
            "extracted_entities": entities,
            "ocr_language": OCR_LANGUAGE,
            "extraction_method": extraction_method,
            "status": "success",
            "error": None,
        }

    except Exception as exc:
        logger.exception(
            "Failed to process %s",
            path.name,
        )

        return {
            "file_name": path.name,
            "file_type": path.suffix.upper().replace(".", ""),
            "file_path": str(path),
            "raw_text": "",
            "pages": [],
            "extracted_entities": {},
            "ocr_language": OCR_LANGUAGE,
            "extraction_method": "error",
            "status": "error",
            "error": str(exc),
        }


# ---------------------------------------------------------
# JSON OUTPUT
# ---------------------------------------------------------

def save_json_output(
    result: dict[str, Any],
    output_dir: str | Path = OUTPUT_DIR,
) -> Path:

    output_path = Path(output_dir)

    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    source_name = Path(
        result["file_name"]
    ).stem

    json_path = output_path / f"{source_name}.json"

    json_path.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=4,
        ),
        encoding="utf-8",
    )

    return json_path


# ---------------------------------------------------------
# FIND ALL SAMPLE FILES
# ---------------------------------------------------------

def discover_sample_files(
    samples_dir: str | Path = SAMPLES_DIR,
) -> list[Path]:

    samples_path = Path(samples_dir)

    samples_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    files = [
        path
        for path in samples_path.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    return sorted(
        files,
        key=lambda path: path.name.lower(),
    )


# ---------------------------------------------------------
# PROCESS MULTIPLE FILES
# ---------------------------------------------------------

def process_files(
    files: list[Path],
    skip_existing: bool = False,
) -> list[Path]:

    if not files:
        logger.info(
            "No supported PDF/image files found in %s",
            SAMPLES_DIR,
        )
        return []

    generated_outputs = []

    logger.info(
        "Found %d document(s).",
        len(files),
    )

    for file_path in files:

        output_path = OUTPUT_DIR / f"{file_path.stem}.json"

        if skip_existing and output_path.exists():
            logger.info(
                "Skipping (JSON exists): %s",
                output_path.name,
            )
            generated_outputs.append(output_path)
            continue

        logger.info(
            "Processing: %s",
            file_path.name,
        )

        result = parse_document(file_path)

        output_path = save_json_output(result)

        generated_outputs.append(output_path)

        if result["status"] == "success":
            logger.info(
                "JSON created: %s",
                output_path.name,
            )
        else:
            logger.error(
                "Failed: %s",
                result.get("error"),
            )

    return generated_outputs


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------

def build_argument_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description=(
            "SafeDocAI document parser. "
            "Without --file, all supported documents "
            "inside data/samples/ are processed."
        )
    )

    parser.add_argument(
        "--file",
        type=str,
        help=(
            "Process one specific PDF/image file."
        ),
    )

    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help=(
            "Skip files that already have a JSON in "
            "data/output/ (avoids re-running OCR on CPU)."
        ),
    )

    return parser


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main() -> None:

    parser = build_argument_parser()
    args = parser.parse_args()

    logger.info(
        "OCR language: %s",
        OCR_LANGUAGE,
    )

    # ---------------------------------------------
    # SINGLE FILE MODE
    # ---------------------------------------------

    if args.file:

        file_path = Path(args.file)

        if not file_path.is_absolute():
            file_path = PROJECT_ROOT / file_path

        process_files(
            [file_path],
            skip_existing=args.skip_existing,
        )
        return

    # ---------------------------------------------
    # AUTO-SCAN MODE
    # ---------------------------------------------

    files = discover_sample_files()

    process_files(
        files,
        skip_existing=args.skip_existing,
    )


if __name__ == "__main__":
    main()