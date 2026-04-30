"""Result cache — hash(PDF bytes + pipeline version + provider) → stored JSON result.

True determinism: once a note is processed successfully, the same note always
returns the same codes regardless of LLM non-determinism across runs.
Bump PIPELINE_VERSION whenever prompts, rules, or logic change.
"""
import hashlib
import json
from pathlib import Path

from app.core.config import BASE_DIR, LLM_PROVIDER
from app.core.logger import get_logger

logger = get_logger(__name__)

PIPELINE_VERSION = "v3.0"
CACHE_DIR = BASE_DIR / "data" / "result_cache"


def _key(pdf_path: Path) -> str:
    content = pdf_path.read_bytes()
    raw = content + PIPELINE_VERSION.encode() + LLM_PROVIDER.encode()
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
