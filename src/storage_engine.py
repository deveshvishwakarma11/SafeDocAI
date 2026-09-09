from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "safedoc.db"
CHROMA_PATH = DATA_DIR / "chroma_db"

COLLECTION_NAME = "safedoc_documents"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


# ============================================================
# Logging
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("SafeDocAI.Storage")


# ============================================================
# SQLite
# ============================================================

def get_db_connection() -> sqlite3.Connection:
    """Create a SQLite connection with foreign keys enabled."""

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(DB_PATH)

    connection.execute("PRAGMA foreign_keys = ON")

    return connection


def init_db() -> None:
    """Create SafeDocAI SQLite tables if they do not exist."""

    with get_db_connection() as connection:

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_name TEXT NOT NULL,
                file_type TEXT NOT NULL,
                file_path TEXT NOT NULL,
                upload_timestamp TEXT NOT NULL,
                status TEXT NOT NULL
            )
            """
        )

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS extracted_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL,
                field_name TEXT NOT NULL,
                field_value TEXT NOT NULL,
                FOREIGN KEY (document_id)
                    REFERENCES documents(id)
                    ON DELETE CASCADE
            )
            """
        )

        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_metadata_field
            ON extracted_metadata(field_name)
            """
        )

        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_metadata_value
            ON extracted_metadata(field_value)
            """
        )

        connection.commit()

    logger.info("SQLite database initialized: %s", DB_PATH)


# ============================================================
# ChromaDB (lazy/optional)
# ============================================================

_chroma_client: Any | None = None
_chroma_available: bool | None = None


def _get_chroma_client() -> Any | None:
    """Import and return a ChromaDB PersistentClient lazily.

    Returns None if chromadb is not installed or cannot be imported.
    """

    global _chroma_client, _chroma_available

    if _chroma_available is False:
        return None

    if _chroma_client is not None:
        return _chroma_client

    try:
        import chromadb as _chromadb
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "ChromaDB is not available: %s", exc
        )
        _chroma_available = False
        return None

    try:
        _chroma_client = _chromadb.PersistentClient(
            path=str(CHROMA_PATH)
        )
        _chroma_available = True
        return _chroma_client

    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "ChromaDB client initialization failed: %s", exc
        )
        _chroma_available = False
        return None


def is_chroma_available() -> bool:
    """True once ChromaDB import and client init have succeeded."""

    _get_chroma_client()

    return bool(_chroma_available)


def get_chroma_collection():
    """Return the persistent ChromaDB collection."""

    CHROMA_PATH.mkdir(parents=True, exist_ok=True)

    client = _get_chroma_client()

    if client is None:
        raise RuntimeError(
            "ChromaDB is not available."
        )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "description": "SafeDocAI document semantic search"
        },
    )

    return collection


def get_chroma_collection_ignoring_errors():
    """Return the persistent ChromaDB collection, or None if unavailable.

    This is used by callers that want semantic search to be optional
    instead of crashing the pipeline.
    """

    try:
        return get_chroma_collection()

    except RuntimeError:
        return None


# ============================================================
# Embedding Model
# ============================================================

_embedding_model: Any | None = None


def get_embedding_model() -> Any:
    """Load the local Sentence Transformer model once.

    sentence-transformers is optional. If it is unavailable,
    this function raises a clear error only when embedding is
    actually needed.
    """

    global _embedding_model

    if _embedding_model is not None:
        return _embedding_model

    try:
        from sentence_transformers import SentenceTransformer as _StSentenceTransformer
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "sentence-transformers is not available."
        ) from exc

    logger.info(
        "Loading embedding model: %s",
        EMBEDDING_MODEL,
    )

    _embedding_model = _StSentenceTransformer(
        EMBEDDING_MODEL,
        device="cpu",
    )

    logger.info("Embedding model loaded.")

    return _embedding_model


# ============================================================
# Text Chunking
# ============================================================

def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """
    Split text into overlapping chunks.

    The splitter first tries to preserve paragraph/sentence
    boundaries and falls back to character-based splitting.
    """

    text = text.strip()

    if not text:
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be >= 0 and smaller than chunk_size"
        )

    # Normalize excessive whitespace while preserving lines.
    paragraphs = [
        paragraph.strip()
        for paragraph in text.splitlines()
        if paragraph.strip()
    ]

    if not paragraphs:
        return []

    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:

        # If the paragraph itself fits, append it.
        if len(paragraph) <= chunk_size:

            if not current:
                current = paragraph

            elif len(current) + 1 + len(paragraph) <= chunk_size:
                current = f"{current} {paragraph}"

            else:
                chunks.append(current)

                overlap_text = current[
                    max(0, len(current) - overlap):
                ]

                current = (
                    f"{overlap_text} {paragraph}".strip()
                )

            continue

        # Flush current chunk.
        if current:
            chunks.append(current)
            current = ""

        # Split large paragraph using sentence boundaries.
        sentences = [
            sentence.strip()
            for sentence in paragraph.replace(
                "?", "."
            ).replace(
                "!", "."
            ).split(".")
            if sentence.strip()
        ]

        for sentence in sentences:

            if len(sentence) <= chunk_size:

                if not current:
                    current = sentence

                elif len(current) + 1 + len(sentence) <= chunk_size:
                    current = f"{current} {sentence}"

                else:
                    chunks.append(current)

                    overlap_text = current[
                        max(0, len(current) - overlap):
                    ]

                    current = (
                        f"{overlap_text} {sentence}".strip()
                    )

            else:
                # Hard split very long sentence.
                if current:
                    chunks.append(current)
                    current = ""

                start = 0

                while start < len(sentence):

                    end = min(
                        start + chunk_size,
                        len(sentence),
                    )

                    chunks.append(
                        sentence[start:end].strip()
                    )

                    if end >= len(sentence):
                        break

                    start = end - overlap

    if current:
        chunks.append(current)

    return [
        chunk.strip()
        for chunk in chunks
        if chunk.strip()
    ]


# ============================================================
# JSON Ingestion
# ============================================================

def load_parsed_json(
    json_path: str | Path,
) -> dict[str, Any]:
    """Load Phase 1 JSON output."""

    path = Path(json_path)

    if not path.exists():
        raise FileNotFoundError(
            f"JSON file not found: {path}"
        )

    if path.suffix.lower() != ".json":
        raise ValueError(
            f"Expected JSON file, got: {path.suffix}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(
            "Phase 1 JSON must contain a JSON object."
        )

    return data


def ingest_metadata(
    connection: sqlite3.Connection,
    document_id: int,
    extracted_entities: dict[str, Any],
) -> None:
    """Insert extracted entities into SQLite."""

    for field_name, values in extracted_entities.items():

        if values is None:
            continue

        if not isinstance(values, list):
            values = [values]

        for value in values:

            if value is None:
                continue

            value = str(value).strip()

            if not value:
                continue

            connection.execute(
                """
                INSERT INTO extracted_metadata
                (
                    document_id,
                    field_name,
                    field_value
                )
                VALUES (?, ?, ?)
                """,
                (
                    document_id,
                    field_name,
                    value,
                ),
            )


def ingest_into_sqlite(
    parsed_data: dict[str, Any],
    json_path: str | Path,
) -> int:
    """Insert a Phase 1 document into SQLite."""

    file_name = str(
        parsed_data.get(
            "file_name",
            Path(json_path).name,
        )
    )

    file_type = str(
        parsed_data.get(
            "file_type",
            Path(file_name).suffix
            .lstrip(".")
            .upper(),
        )
    )

    status = str(
        parsed_data.get(
            "status",
            "unknown",
        )
    )

    file_path = str(
        parsed_data.get(
            "file_path",
            Path(json_path).resolve(),
        )
    )

    upload_timestamp = datetime.now(
        timezone.utc
    ).isoformat()

    extracted_entities = parsed_data.get(
        "extracted_entities",
        {},
    )

    with get_db_connection() as connection:

        # Prevent duplicate document ingestion.
        existing = connection.execute(
            """
            SELECT id
            FROM documents
            WHERE file_name = ?
              AND file_path = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (
                file_name,
                file_path,
            ),
        ).fetchone()

        if existing:
            document_id = int(existing[0])

            connection.execute(
                """
                DELETE FROM extracted_metadata
                WHERE document_id = ?
                """,
                (document_id,),
            )

            connection.execute(
                """
                UPDATE documents
                SET
                    file_type = ?,
                    upload_timestamp = ?,
                    status = ?
                WHERE id = ?
                """,
                (
                    file_type,
                    upload_timestamp,
                    status,
                    document_id,
                ),
            )

        else:
            cursor = connection.execute(
                """
                INSERT INTO documents
                (
                    file_name,
                    file_type,
                    file_path,
                    upload_timestamp,
                    status
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    file_name,
                    file_type,
                    file_path,
                    upload_timestamp,
                    status,
                ),
            )

            document_id = int(
                cursor.lastrowid
            )

        ingest_metadata(
            connection,
            document_id,
            extracted_entities,
        )

        connection.commit()

    logger.info(
        "SQLite ingestion complete. document_id=%s",
        document_id,
    )

    return document_id


def resolve_document_id(
    original_document_path: str | Path,
) -> int | None:
    """Find a document by its exact original file path.

    Returns None if no matching row exists.
    """

    resolved = Path(original_document_path).resolve()

    with get_db_connection() as connection:

        row = connection.execute(
            """
            SELECT id FROM documents
            WHERE file_path = ?
            ORDER BY id DESC LIMIT 1
            """,
            (str(resolved),),
        ).fetchone()

        if not row:
            return None

        return int(row[0])


# ============================================================
# Chroma Ingestion
# ============================================================

def _ingest_into_chroma_optional(
    parsed_data: dict[str, Any],
    document_id: int,
) -> int:
    """Chunk raw text, create embeddings and store in ChromaDB.

    This step is optional. If ChromaDB or sentence-transformers
    are unavailable, this function logs a warning and returns 0
    instead of crashing the rest of the pipeline.
    """

    raw_text = str(
        parsed_data.get(
            "raw_text",
            "",
        )
    ).strip()

    if not raw_text:
        logger.warning(
            "No raw_text found. Skipping ChromaDB ingestion."
        )
        return 0

    file_name = str(
        parsed_data.get(
            "file_name",
            "unknown",
        )
    )

    chunks = chunk_text(raw_text)

    if not chunks:
        return 0

    client = _get_chroma_client()

    if client is None:
        logger.warning(
            "ChromaDB unavailable. Skipping semantic indexing."
        )
        return 0

    try:
        model = get_embedding_model()
    except RuntimeError as exc:
        logger.warning(
            "Embedding model unavailable: %s. Skipping semantic indexing.",
            exc,
        )
        return 0

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "description": "SafeDocAI document semantic search"
        },
    )

    try:
        embeddings = model.encode(
            chunks,
            batch_size=16,
            show_progress_bar=False,
            normalize_embeddings=True,
        )

        ids = [
            f"doc_{document_id}_chunk_{index}"
            for index in range(len(chunks))
        ]

        metadatas = [
            {
                "document_id": str(document_id),
                "file_name": file_name,
                "chunk_id": index,
            }
            for index in range(len(chunks))
        ]

        collection.upsert(
            ids=ids,
            documents=chunks,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
        )

        logger.info(
            "ChromaDB ingestion complete. chunks=%s",
            len(chunks),
        )

        return len(chunks)

    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "ChromaDB ingestion failed: %s", exc
        )
        return 0


def ingest_into_chroma(
    parsed_data: dict[str, Any],
    document_id: int,
) -> int:
    """Chunk raw text, create embeddings and store in ChromaDB.

    This function expects ChromaDB to already be available.
    For a tolerant version, use _ingest_into_chroma_optional().
    """

    raw_text = str(
        parsed_data.get(
            "raw_text",
            "",
        )
    ).strip()

    if not raw_text:
        logger.warning(
            "No raw_text found. Skipping ChromaDB ingestion."
        )
        return 0

    file_name = str(
        parsed_data.get(
            "file_name",
            "unknown",
        )
    )

    chunks = chunk_text(raw_text)

    if not chunks:
        return 0

    client = _get_chroma_client()

    if client is None:
        raise RuntimeError(
            "ChromaDB is not available."
        )

    model = get_embedding_model()

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "description": "SafeDocAI document semantic search"
        },
    )

    embeddings = model.encode(
        chunks,
        batch_size=16,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    ids = [
        f"doc_{document_id}_chunk_{index}"
        for index in range(len(chunks))
    ]

    metadatas = [
        {
            "document_id": str(document_id),
            "file_name": file_name,
            "chunk_id": index,
        }
        for index in range(len(chunks))
    ]

    collection.upsert(
        ids=ids,
        documents=chunks,
        embeddings=embeddings.tolist(),
        metadatas=metadatas,
    )

    logger.info(
        "ChromaDB ingestion complete. chunks=%s",
        len(chunks),
    )

    return len(chunks)


# ============================================================
# Complete Ingestion
# ============================================================

def ingest_parsed_json(
    json_path: str | Path,
) -> dict[str, Any]:
    """
    Ingest Phase 1 JSON into SQLite and, if available, ChromaDB.
    """

    init_db()

    parsed_data = load_parsed_json(
        json_path
    )

    document_id = ingest_into_sqlite(
        parsed_data,
        json_path,
    )

    chroma_chunks = _ingest_into_chroma_optional(
        parsed_data,
        document_id,
    )

    chroma_ready = is_chroma_available()

    return {
        "document_id": document_id,
        "file_name": parsed_data.get(
            "file_name"
        ),
        "sqlite": True,
        "chroma": {
            "available": chroma_ready,
            "chunks": chroma_chunks,
        },
        "status": "success",
    }


# ============================================================
# Verified Metadata Storage (Phase 3)
# ============================================================

def store_verified_metadata(
    document_id: int,
    understanding: dict[str, Any],
) -> int:
    """Store verified fields from Phase 3 understanding into SQLite.

    Only stores fields marked as 'verified': true.
    Uses the provided document_id directly - does not search by file path.
    """

    fields = understanding.get("fields", [])

    if not fields:
        return 0

    # Verify document exists before storing metadata
    with get_db_connection() as connection:

        # Check that the document_id exists in the documents table
        doc_exists = connection.execute(
            """
            SELECT id FROM documents WHERE id = ? LIMIT 1
            """,
            (document_id,),
        ).fetchone()

        if not doc_exists:
            logger.warning(
                "Document ID %d not found in database, skipping metadata storage",
                document_id,
            )
            return 0

    stored_count = 0

    with get_db_connection() as connection:

        for field in fields:
            if not isinstance(field, dict):
                continue

            if not field.get("verified", False):
                continue

            key = str(field.get("key", "")).strip()
            value = str(field.get("value", "")).strip()

            if not key or not value or value.upper() in (
                "UNKNOWN", "", "NONE", "N/A"
            ):
                continue

            # Check for duplicate to avoid redundant entries
            existing_field = connection.execute(
                """
                SELECT id FROM extracted_metadata
                WHERE document_id = ?
                  AND field_name = ?
                  AND field_value = ?
                LIMIT 1
                """,
                (document_id, key, value),
            ).fetchone()

            if existing_field:
                continue

            connection.execute(
                """
                INSERT INTO extracted_metadata
                (document_id, field_name, field_value)
                VALUES (?, ?, ?)
                """,
                (document_id, key, value),
            )

            stored_count += 1

        connection.commit()

    if stored_count > 0:
        logger.info(
            "Stored %d verified fields for document_id=%d",
            stored_count,
            document_id,
        )

    return stored_count


def store_understanding_results(
    json_path: str | Path,
    understanding: dict[str, Any],
    original_document_path: str | Path | None = None,
) -> dict[str, Any]:
    """Store Phase 3 understanding results.

    Links to an existing document by its exact original file path.
    A filename-only fallback is intentionally not used, because it
    can link verified fields to the wrong document when multiple
    files share the same name.

    Args:
        json_path: Path to the Phase 3 understanding JSON (for reference)
        understanding: The Phase 3 understanding dict
        original_document_path: Path to the ORIGINAL document (PDF/image),
            NOT the Phase 1 JSON or Phase 3 understanding JSON.
            This is used to find the document in SQLite if document_id is not available.
    
    Returns:
        dict with document_id, stored count, and status
    """

    raw_document_id = understanding.get("document_id")

    if isinstance(raw_document_id, int):
        document_id = raw_document_id

    elif isinstance(raw_document_id, str) and raw_document_id.strip().isdigit():
        document_id = int(raw_document_id)

    else:
        document_id = None

    if document_id is not None:
        stored_count = store_verified_metadata(
            document_id,
            understanding,
        )

        return {
            "document_id": document_id,
            "stored": stored_count,
            "status": "success",
        }

    if original_document_path is None:
        return {
            "stored": 0,
            "error": "Cannot determine original document path",
        }

    document_id = resolve_document_id(original_document_path)

    if document_id is None:
        resolved = Path(original_document_path).resolve()

        return {
            "stored": 0,
            "error": (
                "Document not found in database for original path: "
                f"{resolved}"
            ),
            "status": "error",
        }

    stored_count = store_verified_metadata(
        document_id,
        understanding,
    )

    return {
        "document_id": document_id,
        "stored": stored_count,
        "status": "success",
    }


# ============================================================
# Exact Search
# ============================================================

def query_exact(
    field_name: str,
    value: str | None = None,
) -> list[dict[str, Any]]:
    """
    Deterministic SQLite lookup.

    Examples:
        query_exact("pan", "ABCDE1234F")
        query_exact("dob")
    """

    field_name = field_name.strip()

    if not field_name:
        raise ValueError(
            "field_name cannot be empty."
        )

    with get_db_connection() as connection:

        if value is None:

            rows = connection.execute(
                """
                SELECT
                    d.id,
                    d.file_name,
                    d.file_type,
                    d.file_path,
                    d.upload_timestamp,
                    d.status,
                    m.field_name,
                    m.field_value
                FROM extracted_metadata AS m
                JOIN documents AS d
                    ON d.id = m.document_id
                WHERE LOWER(m.field_name) = LOWER(?)
                ORDER BY d.id DESC
                """,
                (field_name,),
            ).fetchall()

        else:

            rows = connection.execute(
                """
                SELECT
                    d.id,
                    d.file_name,
                    d.file_type,
                    d.file_path,
                    d.upload_timestamp,
                    d.status,
                    m.field_name,
                    m.field_value
                FROM extracted_metadata AS m
                JOIN documents AS d
                    ON d.id = m.document_id
                WHERE LOWER(m.field_name) = LOWER(?)
                  AND LOWER(m.field_value) = LOWER(?)
                ORDER BY d.id DESC
                """,
                (
                    field_name,
                    value.strip(),
                ),
            ).fetchall()

    columns = [
        "document_id",
        "file_name",
        "file_type",
        "file_path",
        "upload_timestamp",
        "status",
        "field_name",
        "field_value",
    ]

    return [
        dict(zip(columns, row))
        for row in rows
    ]


def lookup_exact_entity(
    field_name: str,
    value: str | None = None,
) -> list[dict[str, Any]]:
    """Alias for query_exact()."""

    return query_exact(
        field_name,
        value,
    )


# ============================================================
# Semantic Search
# ============================================================

def query_semantic(
    query: str,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """Search ChromaDB using local semantic embeddings.

    This function requires ChromaDB and sentence-transformers.
    If either is unavailable, it raises a clear RuntimeError.
    """

    query = query.strip()

    if not query:
        raise ValueError(
            "Semantic query cannot be empty."
        )

    if top_k <= 0:
        raise ValueError(
            "top_k must be greater than 0."
        )

    collection = get_chroma_collection()

    if collection.count() == 0:
        return []

    model = get_embedding_model()

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
    )[0].tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(
            top_k,
            collection.count(),
        ),
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    documents = results.get(
        "documents",
        [[]],
    )[0]

    metadatas = results.get(
        "metadatas",
        [[]],
    )[0]

    distances = results.get(
        "distances",
        [[]],
    )[0]

    output: list[dict[str, Any]] = []

    for index, document in enumerate(documents):

        output.append(
            {
                "text": document,
                "metadata": (
                    metadatas[index]
                    if index < len(metadatas)
                    else {}
                ),
                "distance": (
                    distances[index]
                    if index < len(distances)
                    else None
                ),
            }
        )

    return output


def semantic_search(
    query: str,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """Alias for query_semantic()."""

    return query_semantic(
        query,
        top_k,
    )


def query_semantic_optional(
    query: str,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """Semantic search that remains optional.

    If ChromaDB or the embedding model are unavailable, this
    returns an empty list instead of crashing the caller.
    """

    query = query.strip()

    if not query:
        raise ValueError(
            "Semantic query cannot be empty."
        )

    if top_k <= 0:
        raise ValueError(
            "top_k must be greater than 0."
        )

    collection = get_chroma_collection_ignoring_errors()

    if collection is None or collection.count() == 0:
        return []

    try:
        model = get_embedding_model()

    except RuntimeError:
        return []

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
    )[0].tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(
            top_k,
            collection.count(),
        ),
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    documents = results.get(
        "documents",
        [[]],
    )[0]

    metadatas = results.get(
        "metadatas",
        [[]],
    )[0]

    distances = results.get(
        "distances",
        [[]],
    )[0]

    output: list[dict[str, Any]] = []

    for index, document in enumerate(documents):

        output.append(
            {
                "text": document,
                "metadata": (
                    metadatas[index]
                    if index < len(metadatas)
                    else {}
                ),
                "distance": (
                    distances[index]
                    if index < len(distances)
                    else None
                ),
            }
        )

    return output


# ============================================================
# Test Harness
# ============================================================

def main() -> None:

    print()
    print("==============================")
    print("   SafeDocAI - Phase 2")
    print("      Storage Engine")
    print("==============================")
    print()

    default_json = Path(__file__).resolve().parent.parent / "data" / "output"

    candidate_files = sorted(default_json.glob("*.json"))

    print("Initializing SQLite...")
    init_db()

    if not candidate_files:
        print()
        print("No JSON files found in data/output/.")
        print("Stored documents can still be queried once ingested.")
        print()
        print("==============================")
        print("Phase 2 storage test complete.")
        print("==============================")
        return

    target_json = candidate_files[0]

    print()
    print("Ingesting Phase 1 JSON:")
    print(target_json)

    try:
        ingestion_result = ingest_parsed_json(
            target_json
        )

    except FileNotFoundError as exc:
        print()
        print("Ingestion failed:")
        print(exc)
        print()
        print("==============================")
        print("Phase 2 storage test complete.")
        print("==============================")
        return

    print()
    print("Ingestion result:")
    print(
        json.dumps(
            ingestion_result,
            indent=4,
            ensure_ascii=False,
        )
    )

    # --------------------------------------------------------
    # Exact query
    # --------------------------------------------------------

    print()
    print("=== Exact Entity Query ===")

    exact_results = query_exact(
        "dob"
    )

    print(
        json.dumps(
            exact_results,
            indent=4,
            ensure_ascii=False,
        )
    )

    # --------------------------------------------------------
    # Semantic query
    # --------------------------------------------------------

    chroma_ready = ingestion_result.get("chroma", {}).get("available", False)

    if chroma_ready:
        print()
        print("=== Semantic Search ===")

        semantic_results = query_semantic(
            "marksheet subjects",
            top_k=3,
        )

        for index, result in enumerate(
            semantic_results,
            start=1,
        ):
            print()
            print(f"Result {index}")
            print(
                f"Distance: {result['distance']}"
            )
            print(
                f"Metadata: {result['metadata']}"
            )
            print(
                f"Text: {result['text']}"
            )

    else:
        print()
        print("=== Semantic Search ===")
        print("Skipped: ChromaDB or embedding model unavailable.")

    print()
    print("==============================")
    print("Phase 2 storage test complete.")
    print("==============================")


if __name__ == "__main__":
    main()