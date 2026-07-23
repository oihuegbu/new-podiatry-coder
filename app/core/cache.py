"""Result cache — hash(PDF bytes + logic fingerprint + provider) → stored JSON result.

True determinism: once a note is processed successfully, the same note always
returns the same codes regardless of LLM non-determinism across runs.

Cache invalidation is AUTOMATIC, not manual: the key incorporates a
fingerprint of every source file whose logic bakes into a stored result
(prompts, validator rules, compliance-store logic) plus the reference data
files they read. The previous design relied on a hand-bumped
PIPELINE_VERSION string — which predictably was NOT bumped across a large
prompt/validator overhaul, so stale cached results (produced by known-buggy
logic, e.g. the strip-all-modifiers era) kept being served as current. Note
the on-hit re-run in pipeline.py only covers the compliance scrubber, not
the validator — the validator's code/modifier mutations are frozen into the
cached arrays, so any validator change MUST invalidate the cache.
"""
import hashlib
import json
from pathlib import Path

from app.core.config import BASE_DIR, LLM_PROVIDER
from app.core.logger import get_logger

logger = get_logger(__name__)

PIPELINE_VERSION = "v4.0"  # coarse manual override; automatic fingerprint below is primary
CACHE_DIR = BASE_DIR / "data" / "result_cache"

_APP_DIR = BASE_DIR / "app"
# Source files whose behavior is baked into a cached result and NOT re-applied
# on a cache hit (pipeline re-runs only the scrubber on hits — see
# pipeline.process_note). Scrubber agents are deliberately excluded: they DO
# re-run on every hit, so their changes propagate without invalidation.
_LOGIC_SOURCES = [
    _APP_DIR / "ingestion" / "pdf_parser.py",
    _APP_DIR / "ner" / "entity_extractor.py",
    _APP_DIR / "ner" / "biomed_ner.py",
    _APP_DIR / "rag" / "retriever.py",
    _APP_DIR / "coding" / "code_assigner.py",
    _APP_DIR / "validation" / "validator.py",
    _APP_DIR / "compliance" / "datastore" / "store.py",
]
_DATA_DIRS = [BASE_DIR / "data" / "codes", BASE_DIR / "data"]


def _logic_fingerprint() -> str:
    """Hash of pipeline logic sources (content) + reference data files
    (name/size/mtime — content hashing 400MB+ of NCCI data per lookup would
    be prohibitively slow, and size+mtime is a reliable change signal for
    files only ever replaced wholesale by the refresh layer)."""
    h = hashlib.sha256()
    for src in _LOGIC_SOURCES:
        try:
            h.update(src.read_bytes())
        except OSError:
            h.update(f"missing:{src.name}".encode())
    for d in _DATA_DIRS:
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*.json")):
            st = f.stat()
            h.update(f"{f.name}:{st.st_size}:{st.st_mtime_ns}".encode())
    return h.hexdigest()


def _key(pdf_path: Path) -> str:
    content = pdf_path.read_bytes()
    raw = (
        content
        + PIPELINE_VERSION.encode()
        + LLM_PROVIDER.encode()
        + _logic_fingerprint().encode()
    )
    return hashlib.sha256(raw).hexdigest()[:20]


def get_cached(pdf_path: Path) -> dict | None:
    key = _key(pdf_path)
    cache_file = CACHE_DIR / f"{key}.json"
    if cache_file.exists():
        try:
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            logger.info(f"  Cache HIT [{key}] — returning stored result for {pdf_path.name}")
            return data
        except Exception as e:
            logger.warning(f"  Cache read failed ({e}), reprocessing")
    return None


def store(pdf_path: Path, result_dict: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = _key(pdf_path)
    cache_file = CACHE_DIR / f"{key}.json"
    try:
        cache_file.write_text(
            json.dumps(result_dict, default=str, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(f"  Cached result [{key}] for {pdf_path.name}")
    except Exception as e:
        logger.warning(f"  Cache write failed: {e}")


def invalidate(pdf_path: Path) -> None:
    key = _key(pdf_path)
    cache_file = CACHE_DIR / f"{key}.json"
    if cache_file.exists():
        cache_file.unlink()
        logger.info(f"  Cache invalidated [{key}] for {pdf_path.name}")
