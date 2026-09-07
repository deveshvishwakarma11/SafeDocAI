from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import fitz
import numpy as np
import pdfplumber
import pytesseract

from utils import (
    is_image_file,
    is_pdf_file,
    load_image,
    preprocess_for_ocr,
    validate_file,
)


# ============================================================
# Configuration
# ============================================================

TESSERACT_WINDOWS_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

OCR_DPI = 250
OCR_MIN_TEXT_LENGTH = 20

SUPPORTED_OCR_LANGUAGES = ("eng", "hin")


# ============================================================
# Logging
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("SafeDocAI")


# ============================================================
# Data Models
# ============================================================

@dataclass
class OCRWord:
    text: str
    confidence: float
    left: int
    top: int
    width: int
    height: int


@dataclass
class OCRResult:
    text: str
    words: list[OCRWord] = field(default_factory=list)


# ============================================================
# Document Parser
# ============================================================

class DocumentParser:
    """Local document parser for SafeDocAI."""

    def __init__(self) -> None:
        self._configure_tesseract()

    # --------------------------------------------------------
    # Tesseract
    # --------------------------------------------------------

    def _configure_tesseract(self) -> None:
        """Configure Tesseract executable."""

        try:
            version = pytesseract.get_tesseract_version()
            logger.info("Tesseract detected: %s", version)
        except Exception:
            if Path(TESSERACT_WINDOWS_PATH).exists():
                pytesseract.pytesseract.tesseract_cmd = (
                    TESSERACT_WINDOWS_PATH
                )

                try:
                    version = pytesseract.get_tesseract_version()
                    logger.info("Tesseract detected: %s", version)
                except Exception as exc:
                    raise RuntimeError(
                        "Tesseract is installed but could not be started."
                    ) from exc
            else:
                raise RuntimeError(
                    "Tesseract OCR was not found. "
                    "Please install Tesseract OCR."
                )

    def _get_ocr_language(self) -> str:
        """Return available OCR language combination."""

        try:
            languages = pytesseract.get_languages(config="")
        except Exception:
            languages = ["eng"]

        has_eng = "eng" in languages
        has_hin = "hin" in languages

        if has_eng and has_hin:
            logger.info("English + Hindi OCR enabled.")
            return "eng+hin"

        if has_eng:
            logger.warning(
                "Hindi language data ('hin') not installed. "
                "Using English OCR only."
            )
            return "eng"

        if has_hin:
            logger.info("Hindi OCR enabled.")
            return "hin"

        raise RuntimeError(
            "No usable Tesseract language data found."
        )

    # --------------------------------------------------------
    # OCR
    # --------------------------------------------------------

    def _run_ocr(self, image: np.ndarray) -> OCRResult:
        """Run OCR on a preprocessed image."""

        processed = preprocess_for_ocr(image)
        language = self._get_ocr_language()

        config = "--psm 6"

        text = pytesseract.image_to_string(
            processed,
            lang=language,
            config=config,
        )

        data = pytesseract.image_to_data(
            processed,
            lang=language,
            config=config,
            output_type=pytesseract.Output.DICT,
        )

        words: list[OCRWord] = []

        count = len(data["text"])

        for i in range(count):
            word = data["text"][i].strip()

            if not word:
                continue

            try:
                confidence = float(data["conf"][i])
            except (ValueError, TypeError):
                confidence = -1.0

            words.append(
                OCRWord(
                    text=word,
                    confidence=confidence,
                    left=int(data["left"][i]),
                    top=int(data["top"][i]),
                    width=int(data["width"][i]),
                    height=int(data["height"][i]),
                )
            )

        return OCRResult(
            text=text.strip(),
            words=words,
        )

    # --------------------------------------------------------
    # PDF text extraction
    # --------------------------------------------------------

    def _extract_pdf_text(
        self,
        file_path: Path,
    ) -> tuple[str, list[int]]:
        """
        Extract embedded PDF text.

        Returns:
            complete_text,
            pages_requiring_ocr
        """

        page_texts: list[str] = []
        ocr_pages: list[int] = []

        with pdfplumber.open(file_path) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""

                page_texts.append(text.strip())

                if len(text.strip()) < OCR_MIN_TEXT_LENGTH:
                    ocr_pages.append(page_number)

        return "\n\n".join(page_texts), ocr_pages

    # --------------------------------------------------------
    # PDF OCR
    # --------------------------------------------------------

    def _ocr_pdf_page(
        self,
        document: fitz.Document,
        page_number: int,
    ) -> OCRResult:
        """Render and OCR a single PDF page."""

        page = document[page_number - 1]

        matrix = fitz.Matrix(
            OCR_DPI / 72,
            OCR_DPI / 72,
        )

        pixmap = page.get_pixmap(
            matrix=matrix,
            alpha=False,
        )

        image = np.frombuffer(
            pixmap.samples,
            dtype=np.uint8,
        )

        image = image.reshape(
            pixmap.height,
            pixmap.width,
            pixmap.n,
        )

        if pixmap.n == 4:
            image = cv2.cvtColor(
                image,
                cv2.COLOR_RGBA2BGR,
            )
        else:
            image = cv2.cvtColor(
                image,
                cv2.COLOR_RGB2BGR,
            )

        return self._run_ocr(image)

    # --------------------------------------------------------
    # Image parsing
    # --------------------------------------------------------

    def _parse_image(
        self,
        file_path: Path,
    ) -> str:
        """Parse an image document."""

        image = load_image(file_path)

        if image is None:
            raise ValueError(
                f"Unable to load image: {file_path}"
            )

        logger.info("Running OCR on image: %s", file_path.name)

        result = self._run_ocr(image)

        return result.text

    # --------------------------------------------------------
    # PDF parsing
    # --------------------------------------------------------

    def _parse_pdf(
        self,
        file_path: Path,
    ) -> str:
        """Parse PDF using text extraction + OCR fallback."""

        logger.info(
            "Processing PDF: %s",
            file_path.name,
        )

        embedded_text, ocr_pages = self._extract_pdf_text(
            file_path
        )

        logger.info(
            "OCR required for pages: %s",
            ocr_pages,
        )

        if not ocr_pages:
            return embedded_text.strip()

        with fitz.open(file_path) as document:

            page_texts = embedded_text.split("\n\n")

            while len(page_texts) < len(document):
                page_texts.append("")

            for page_number in ocr_pages:
                logger.info(
                    "Running OCR on PDF page %s",
                    page_number,
                )

                ocr_result = self._ocr_pdf_page(
                    document,
                    page_number,
                )

                page_texts[page_number - 1] = (
                    ocr_result.text
                )

            return "\n\n".join(
                text for text in page_texts if text.strip()
            ).strip()

    # --------------------------------------------------------
    # Entity extraction
    # --------------------------------------------------------

    @staticmethod
    def _extract_entities(
        text: str,
    ) -> dict[str, list[str]]:
        """Extract common structured entities."""

        entities: dict[str, list[str]] = {
            "dob": [],
            "pan": [],
            "ifsc": [],
            "phone": [],
            "email": [],
            "amounts": [],
        }

        # Dates / DOB
        date_patterns = [
            r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
            r"\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|"
            r"Oct|Nov|Dec)[a-z]*\s+\d{2,4}\b",
        ]

        for pattern in date_patterns:
            entities["dob"].extend(
                re.findall(
                    pattern,
                    text,
                    flags=re.IGNORECASE,
                )
            )

        # PAN
        entities["pan"] = re.findall(
            r"\b[A-Z]{5}[0-9]{4}[A-Z]\b",
            text.upper(),
        )

        # IFSC
        entities["ifsc"] = re.findall(
            r"\b[A-Z]{4}0[A-Z0-9]{6}\b",
            text.upper(),
        )

        # Phone numbers
        entities["phone"] = re.findall(
            r"(?<!\d)(?:\+91[\s-]?)?[6-9]\d{9}(?!\d)",
            text,
        )

        # Email
        entities["email"] = re.findall(
            r"\b[A-Za-z0-9._%+-]+"
            r"@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
            text,
        )

        # Amounts
        entities["amounts"] = re.findall(
            r"(?:₹|Rs\.?|INR)\s*"
            r"(?:\d{1,3}(?:,\d{3})+|\d+)"
            r"(?:\.\d{1,2})?",
            text,
            flags=re.IGNORECASE,
        )

        # Remove duplicates while preserving order
        for key in entities:
            entities[key] = list(
                dict.fromkeys(entities[key])
            )

        return entities

    # --------------------------------------------------------
    # Main parser
    # --------------------------------------------------------

    def parse(
        self,
        file_path: str | Path,
    ) -> dict[str, Any]:
        """Parse a document and return structured JSON data."""

        path = Path(file_path)

        try:
            validate_file(path)

            if is_pdf_file(path):
                raw_text = self._parse_pdf(path)

            elif is_image_file(path):
                raw_text = self._parse_image(path)

            else:
                raise ValueError(
                    f"Unsupported file type: {path.suffix}"
                )

            return {
                "file_name": path.name,
                "file_type": path.suffix.lstrip(".").upper(),
                "raw_text": raw_text,
                "extracted_entities": self._extract_entities(
                    raw_text
                ),
                "status": "success",
                "error": None,
            }

        except Exception as exc:
            logger.exception(
                "Document parsing failed."
            )

            return {
                "file_name": path.name,
                "file_type": path.suffix.lstrip(".").upper(),
                "raw_text": "",
                "extracted_entities": {
                    "dob": [],
                    "pan": [],
                    "ifsc": [],
                    "phone": [],
                    "email": [],
                    "amounts": [],
                },
                "status": "error",
                "error": str(exc),
            }


# ============================================================
# JSON Output
# ============================================================

def parse_document_to_json(
    file_path: str | Path,
) -> str:
    """Parse document and return formatted JSON string."""

    parser = DocumentParser()

    result = parser.parse(file_path)

    return json.dumps(
        result,
        ensure_ascii=False,
        indent=4,
    )


def save_json_output(
    result: dict[str, Any],
    output_dir: str | Path = "data/output",
) -> Path:
    """Save parser result as UTF-8 JSON."""

    output_path = Path(output_dir)

    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    source_name = Path(
        result["file_name"]
    ).stem

    json_path = (
        output_path /
        f"{source_name}.json"
    )

    json_path.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=4,
        ),
        encoding="utf-8",
    )

    return json_path


# ============================================================
# Command Line / Test
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description="SafeDocAI Local Document Parser"
    )

    parser.add_argument(
        "file",
        nargs="?",
        default="data/samples/Devesh 4th sem.pdf",
        help="Path to PDF or image file",
    )

    args = parser.parse_args()

    print()
    print("==============================")
    print("     SafeDocAI - Phase 1")
    print("          Document Parser")
    print("==============================")
    print()

    print(
        f"Input file: {args.file}"
    )

    result = DocumentParser().parse(
        args.file
    )

    print()
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=4,
        )
    )
    print()

    output_path = save_json_output(
        result
    )

    print(
        f"JSON saved to: {output_path}"
    )


if __name__ == "__main__":
    main()