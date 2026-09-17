"""
Shared SafeDocAI test infrastructure (plain-Python, no pytest).
===============================================================

Every storage-touching suite must never touch the REAL data/safedoc.db or
data/chroma_db. This module centralizes the boilerplate that suites used to
duplicate:

* :func:`redirect_storage_to_temp` / :func:`restore_storage` -- point
  ``storage_engine`` at isolated temp paths (and drop cached Chroma clients
  so they rebind), with atexit restore.
* :func:`real_db_state` / :func:`guard_real_db` -- capture and enforce the
  real DB's size/mtime/sha256 across a run.
* :func:`seed_chroma_real_model` -- seed a temp Chroma store using the REAL
  local embedding model (all-MiniLM-L6-v2), the same metadata shape the
  production ingester writes.
* :func:`run_tests` -- the common runner: execute test callables, print
  per-test PASS/FAIL, enforce the isolation guard, exit non-zero on failure.

Suites keep their own seed documents and test functions.
"""

from __future__ import annotations

import atexit
import gc
import hashlib
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

_project_root = Path(__file__).resolve().parents[1]
for _entry in (str(_project_root), str(_project_root / "src")):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

import storage_engine  # the SAME module instance the router/engine bind to

# ``src.storage_engine`` and ``storage_engine`` load as TWO distinct module
# objects (both import styles are used across suites). Redirect BOTH so no
# code path can reach the real store regardless of which import it binds to.
try:  # 'src' is importable from the project root
    import src.storage_engine as _src_storage_engine  # type: ignore[no-redef]
except ImportError:  # pragma: no cover - running from inside src/
    _src_storage_engine = storage_engine

_MODULES = {id(storage_engine): storage_engine, id(_src_storage_engine): _src_storage_engine}

__all__ = [
    "redirect_storage_to_temp",
    "restore_storage",
    "real_db_state",
    "guard_real_db",
    "seed_chroma_real_model",
    "run_tests",
]

# ---------------------------------------------------------------------------
# Storage isolation
# ---------------------------------------------------------------------------

_TEMP_DIR: Path | None = None
_ORIGINALS: dict[str, Path] = {}


def _patched(attr_data_dir: Path | None, db: Path | None, chroma: Path | None) -> None:
    """Apply (or revert) path globals on every storage_engine module object."""

    for module in _MODULES.values():
        if attr_data_dir is not None:
            module.DATA_DIR = attr_data_dir
            module.DB_PATH = db
            module.CHROMA_PATH = chroma
            module._chroma_client = None
            module._chroma_available = None
        else:
            module.DATA_DIR = _ORIGINALS["DATA_DIR"]
            module.DB_PATH = _ORIGINALS["DB_PATH"]
            module.CHROMA_PATH = _ORIGINALS["CHROMA_PATH"]
            module._chroma_client = None
            module._chroma_available = None


def redirect_storage_to_temp(prefix: str) -> Path:
    """Point storage_engine (and everything bound to it) at temp paths."""

    global _TEMP_DIR
    if _TEMP_DIR is not None:
        return _TEMP_DIR

    _TEMP_DIR = Path(tempfile.mkdtemp(prefix=prefix))
    _ORIGINALS.update(
        DATA_DIR=storage_engine.DATA_DIR,
        DB_PATH=storage_engine.DB_PATH,
        CHROMA_PATH=storage_engine.CHROMA_PATH,
    )
    _patched(_TEMP_DIR, _TEMP_DIR / "safedoc.db", _TEMP_DIR / "chroma_db")
    return _TEMP_DIR


def restore_storage() -> None:
    """Restore real paths and remove the temp store."""

    global _TEMP_DIR
    if _TEMP_DIR is None:
        return
    gc.collect()
    shutil.rmtree(_TEMP_DIR, ignore_errors=True)
    _patched(None, None, None)
    _TEMP_DIR = None


def real_db_state() -> tuple[int, int, str] | None:
    """(size, mtime_ns, sha256) of the REAL db, or None when absent."""

    real_db = _ORIGINALS.get("DB_PATH")
    if real_db is None or not real_db.exists():
        return None
    data = real_db.read_bytes()
    return (len(data), real_db.stat().st_mtime_ns, hashlib.sha256(data).hexdigest())


def guard_real_db(before: tuple[int, int, str] | None) -> None:
    """Print the isolation-guard verdict; exit 1 if the real DB changed."""

    after = real_db_state()
    if before is None:
        print("Isolation guard: real data/safedoc.db does not exist (nothing to protect).")
    elif before != after:
        print("Isolation guard FAILED: real data/safedoc.db was modified by the test run!")
        raise SystemExit(1)
    else:
        print(
            "Isolation guard OK: real data/safedoc.db untouched "
            f"(size={after[0]} bytes, sha256={after[2][:12]}...)."
        )


def seed_chroma_real_model(
    seed_docs: list[dict[str, Any]],
    doc_ids: list[int],
) -> None:
    """Upsert seed documents into the temp Chroma store.

    Uses the REAL local embedding model so distances match production
    behavior; metadata shape mirrors the production ingester
    (``document_id`` stored as a string).
    """

    collection = storage_engine.get_chroma_collection()
    model = storage_engine.get_embedding_model()

    for record, doc_id in zip(seed_docs, doc_ids):
        chunks = storage_engine.chunk_text(record["raw_text"])
        embeddings = model.encode(
            chunks, batch_size=16, show_progress_bar=False,
            normalize_embeddings=True,
        )
        ids = [f"doc_{doc_id}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "document_id": str(doc_id),
                "file_name": record["file_name"],
                "chunk_id": i,
            }
            for i in range(len(chunks))
        ]
        collection.upsert(
            ids=ids, documents=chunks,
            embeddings=embeddings.tolist(), metadatas=metadatas,
        )


def run_tests(
    tests: list[tuple[str, Callable[[], None]]],
    label: str,
) -> None:
    """Common runner: per-test PASS/FAIL, guard, non-zero exit on failure."""

    passed = 0
    failed: list[str] = []
    before = real_db_state()

    try:
        for name, fn in tests:
            try:
                fn()
                passed += 1
                print(f"  PASS {name}")
            except AssertionError as exc:
                failed.append(f"{name}: {exc}")
                print(f"  FAIL {name}: {exc}")
            except Exception as exc:  # noqa: BLE001
                failed.append(f"{name}: {type(exc).__name__}: {exc}")
                print(f"  FAIL {name}: {type(exc).__name__}: {exc}")

        total = passed + len(failed)
        print()
        if failed:
            print(f"{label} tests FAILED: {len(failed)}/{total}")
            for line in failed:
                print(" -", line)
        else:
            print(f"{label} tests PASSED: {total}")
    finally:
        guard_real_db(before)
        restore_storage()

    if failed:
        raise SystemExit(1)
