# SafeDocAI

Offline document AI pipeline that turns scanned PDFs and images into **classified, evidence-verified, searchable** structured data — 100% locally, with no cloud APIs.

Drop documents into `data/samples/`, run the pipeline, and get:

- **OCR text extraction** (embedded PDF text when reliable, Tesseract OCR fallback when not)
- **Dynamic document classification** (no fixed type enum — the LLM discovers the document type from evidence)
- **Field extraction with proof** — every extracted field is validated against the *original* OCR text and marked verified/unverified
- **Local storage** — SQLite for exact queries, ChromaDB for semantic search

## Pipeline

```
PDF / Image  (data/samples/)
        │
        ▼
┌─────────────────────────────────────────────┐
│ Phase 1 — document_parser.py                │
│  • pdfplumber embedded-text extraction      │
│  • corruption detection → OCR fallback      │
│  • OpenCV preprocessing (deskew, threshold) │
│  • Tesseract OCR (eng + hin if available)   │
└─────────────────────────────────────────────┘
        │  data/output/*.json
        ▼
┌─────────────────────────────────────────────┐
│ Phase 3 — document_understanding.py         │
│  • heuristics.py  — fast regex pre-class    │
│  • chunker.py     — bounded context windows │
│  • llm_engine.py  — local Ollama HTTP call  │
│  • evidence_validator.py — verify vs OCR    │
└─────────────────────────────────────────────┘
        │  data/understood/*.json
        ▼
┌─────────────────────────────────────────────┐
│ Phase 2 — storage_engine.py                 │
│  • SQLite (data/safedoc.db)                 │
│  • ChromaDB (data/chroma_db/) with          │
│    all-MiniLM-L6-v2 embeddings              │
└─────────────────────────────────────────────┘
```

All inference runs against **http://localhost:11434** (Ollama) over HTTP. No document data ever leaves the machine.

## Requirements

- **Python 3.10+**
- **Tesseract OCR** installed and on `PATH` (`pytesseract`). Include the `eng` language pack; add `hin` for Hindi documents (the parser auto-selects `eng+hin` when available).
- **Ollama** running locally with a model (default: `qwen2.5:3b`) — required for Phase 3 only:

  ```bash
  ollama serve
  ollama pull qwen2.5:3b
  ```

Install Python dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Usage

Run each stage from the project root (modules import each other as top-level scripts):

**Phase 1 — Parse documents to OCR JSON**

```bash
python src/document_parser.py                    # parse everything in data/samples/
python src/document_parser.py --file "path/to/doc.pdf"
python src/document_parser.py --skip-existing    # skip files that already have JSON
```

Supported inputs: `.pdf`, `.jpg`, `.jpeg`, `.png` → written to `data/output/<name>.json`.

**Phase 2 — Ingest into storage**

```bash
python src/storage_engine.py    # ingests JSON, runs exact + semantic demo queries
```

**Phase 3 — Understand documents**

```bash
python src/document_understanding.py                          # process all OCR JSONs
python src/document_understanding.py --file data/output/x.json
python src/document_understanding.py --check                  # verify Ollama connectivity
python src/document_understanding.py --test                   # full test suite on samples
```

Useful flags: `--model`, `--max-context-chars`, `--num-predict`, `--max-llm-windows` (hard cap on LLM calls per document).

## How understanding works

1. **Heuristic pre-classifier** (`heuristics.py`) — pure-Python regex evidence matching runs first, giving a fast, deterministic document-type hypothesis with confidence. It is a fallback layer, not the final answer.
2. **Bounded context windows** (`chunker.py` + windowing) — long OCR text is split into windows that are *guaranteed* to fit the LLM context budget, with head+tail anchoring (headers and footers survive), overlap between windows, and explicit `[... N characters omitted ...]` markers when the call cap forces gaps. Default cap: **4 LLM calls per document**.
3. **Local LLM** (`llm_engine.py`) — calls Ollama over HTTP (never subprocess) with JSON-enforced output, `temperature=0`, and generation caps sized for CPU-only hardware.
4. **Evidence validation** (`evidence_validator.py`) — every LLM-extracted field and the document type itself are checked against the **original, full OCR text** (not the truncated window). Unverified fields are kept but marked. Final classification source is reported as `LLM`, `HEURISTIC`, `HYBRID`, or `UNKNOWN`, with an overall confidence grade.

If Ollama is unreachable, Phase 3 degrades gracefully to heuristic-only classification (clearly marked, confidence `LOW`) instead of failing.

## Data layout

```
data/
├── samples/        # input documents (you put files here)
├── output/         # Phase 1 OCR JSON (raw_text, extraction_method, ...)
├── understood/     # Phase 3 understanding JSON (type, fields, verification, timing)
├── safedoc.db      # SQLite storage (Phase 2)
└── chroma_db/      # ChromaDB vector store (Phase 2)
```

## Tests

Tests are standalone scripts (no pytest required) and run against temp directories — they never touch the real `data/safedoc.db` or `data/chroma_db/` (an isolation guard verifies the real DB's hash after the run):

```bash
python tests/test_storage_engine.py
python tests/test_evidence_validator.py
python tests/test_long_document_handling.py
```

## Design notes

- **Offline-first**: every dependency is local. OCR, embeddings, and LLM inference all happen on your machine.
- **Evidence over trust**: the LLM proposes; the validator disposes. Fields unsupported by the original OCR text are flagged, never silently accepted.
- **CPU-friendly**: bounded windows, capped LLM calls per document, and generation limits keep the pipeline usable on low-RAM laptops.
