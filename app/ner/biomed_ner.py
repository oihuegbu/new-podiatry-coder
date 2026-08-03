"""GLiNER-BioMed — deterministic biomedical NER layer.

Device priority on Apple Silicon M-series:
  1. MPS  (Metal Performance Shaders — Mac GPU, fastest)
  2. CUDA (if ever running on Linux/Windows with NVIDIA)
  3. CPU  (fallback)

macOS segfault fix: TOKENIZERS_PARALLELISM and OMP_NUM_THREADS must be set
BEFORE torch is imported. Setting them at module top-level is enough because
this module is imported before any other torch import in the pipeline.
"""
from __future__ import annotations
import os

# Must be set before torch / tokenizers are imported anywhere in the process
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from app.core.logger import get_logger

logger = get_logger(__name__)

_model = None
_device: str = "cpu"
_available: bool | None = None

ENTITY_LABELS = [
    "diagnosis",
    "procedure performed",
    "medication",
    "medical supply",
    "clinical finding",
    "body structure",
    "laterality",
]

_MAX_CHARS = 1500  # chunk size to stay within GLiNER token limit


def _get_device() -> str:
    """Return the best available device: mps > cuda > cpu."""
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _load() -> bool:
    global _model, _device, _available
    if _available is not None:
        return _available
    try:
        import torch
        torch.set_num_threads(1)  # extra guard for CPU threads

        _device = _get_device()
        logger.info(f"  Loading GLiNER-BioMed model on device: {_device.upper()}...")

        from gliner import GLiNER
        _model = GLiNER.from_pretrained(
            "Ihor/gliner-biomed-large-v1.0",
            local_files_only=False,
        )
        # Move model to the best available device
        if hasattr(_model, "to"):
            _model = _model.to(_device)
        elif hasattr(_model, "model") and hasattr(_model.model, "to"):
            _model.model = _model.model.to(_device)

        _model.eval()
        _available = True
        logger.info(f"  GLiNER-BioMed ready [{_device.upper()}]")
    except Exception as exc:
        logger.warning(
            f"  GLiNER-BioMed unavailable ({type(exc).__name__}: {exc}) — LLM-only NER mode"
        )
        _available = False
    return _available


def _predict_chunks(text: str, threshold: float) -> list[dict]:
    """Split long notes into chunks and run prediction on each."""
    import torch

    chunks: list[str] = []
    current = ""
    for sentence in text.replace("\n", " \n ").split(". "):
        if len(current) + len(sentence) < _MAX_CHARS:
            current += sentence + ". "
        else:
            if current.strip():
                chunks.append(current.strip())
            current = sentence + ". "
    if current.strip():
        chunks.append(current.strip())

    results: list[dict] = []
    seen: set[str] = set()
    ctx = torch.no_grad() if _device != "mps" else torch.inference_mode()
    with ctx:
        for chunk in chunks:
            if not chunk:
                continue
            try:
                hits = _model.predict_entities(chunk, ENTITY_LABELS, threshold=threshold)
                for h in hits:
                    key = h["text"].lower().strip()
                    if key not in seen:
                        seen.add(key)
                        results.append(h)
            except Exception as exc:
                logger.warning(f"  GLiNER chunk prediction failed: {exc}")
    return results


def get_confirmed_spans(text: str, threshold: float = 0.45) -> dict[str, float]:
    """Return {lowercased_span: confidence} for all entities GLiNER finds."""
    if not _load():
        return {}
    try:
        hits = _predict_chunks(text, threshold)
        return {h["text"].lower().strip(): round(h.get("score", threshold), 3) for h in hits}
    except Exception as exc:
        logger.warning(f"  GLiNER prediction error: {exc}")
        return {}


def enrich_entities(entities: list[dict], full_text: str) -> list[dict]:
    """Tag each LLM entity dict with ner_source based on GLiNER confirmation."""
    if not entities:
        return entities
    confirmed = get_confirmed_spans(full_text)
    if not confirmed:
        # Distinguish "not independently confirmed" from a measured 100%
        # result.  Downstream provenance must not silently present an
        # unavailable/empty confirmation layer as certainty.
        for e in entities:
            e["ner_source"] = "llm_only"
            try:
                e["ner_confidence"] = min(
                    float(e.get("ner_confidence", 0.7)), 0.7)
            except (TypeError, ValueError):
                e["ner_confidence"] = 0.7
        return entities
    for e in entities:
        term = e.get("clinical_term", "").lower().strip()
        text_span = e.get("text", "").lower().strip()
        matched_conf = confirmed.get(term) or confirmed.get(text_span)
        if matched_conf:
            e["ner_source"] = "gliner_confirmed"
            e["ner_confidence"] = matched_conf
        else:
            e["ner_source"] = "llm_only"
            e["ner_confidence"] = e.get("ner_confidence", 0.7)
    confirmed_count = sum(1 for e in entities if e.get("ner_source") == "gliner_confirmed")
    logger.info(f"  GLiNER confirmed {confirmed_count}/{len(entities)} entities [{_device.upper()}]")
    return entities
