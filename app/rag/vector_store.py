import hashlib
import json
import pickle
from pathlib import Path

import faiss
import numpy as np

from app.core.config import (
    VECTOR_STORE_DIR,
    EMBEDDING_DIMENSIONS,
    ICD10_FILE,
    CPT_FILE,
    HCPCS_FILE,
    RAG_TOP_K,
    RAG_SIMILARITY_THRESHOLD,
)
from app.core.llm_client import embed_texts
from app.core.logger import get_logger

logger = get_logger(__name__)

# Files whose content drives the FAISS index — any change triggers a rebuild
_INDEXED_FILES = [ICD10_FILE, CPT_FILE, HCPCS_FILE]
_CHECKSUM_FILE = VECTOR_STORE_DIR / "codes_checksum.txt"
_CODE_SYSTEMS  = ["icd10", "cpt", "hcpcs"]


def _compute_checksum() -> str:
    """Fast fingerprint: sha256 of each file's size + mtime + path.
    Detects new files, replaced files, or in-place edits without reading GBs of JSON."""
    h = hashlib.sha256()
    for p in _INDEXED_FILES:
        path = Path(p)
        if path.exists():
            stat = path.stat()
            h.update(f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}".encode())
        else:
            h.update(f"{p}:missing".encode())
    return h.hexdigest()


class MedicalCodeVectorStore:
    """FAISS-backed vector store for medical code retrieval.

    Smart rebuild logic:
    - First run (no index on disk)  → build and cache
    - Same code files as last run   → load from disk (fast)
    - Any code file replaced/edited → auto-rebuild without --rebuild-index flag
    - force_rebuild=True            → always rebuild regardless
    """

    def __init__(self):
        self.indices:  dict[str, faiss.IndexFlatIP] = {}
        self.metadata: dict[str, list[dict]] = {}
        self._loaded = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_or_load(self, force_rebuild: bool = False) -> None:
        VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

        current_checksum = _compute_checksum()
        indices_on_disk  = self._all_indices_exist()
        stored_checksum  = _CHECKSUM_FILE.read_text().strip() if _CHECKSUM_FILE.exists() else ""

        files_unchanged  = (stored_checksum == current_checksum)
        can_load         = indices_on_disk and files_unchanged and not force_rebuild

        if can_load:
            logger.info("Code files unchanged — loading pre-built FAISS indices...")
            self._load_indices()
        else:
            if not indices_on_disk:
                logger.info("No FAISS indices found — building for the first time...")
            elif not files_unchanged:
                logger.info("Code files have changed — rebuilding FAISS indices...")
            else:
                logger.info("Force rebuild requested — rebuilding FAISS indices...")
            self._build_all_indices()
            _CHECKSUM_FILE.write_text(current_checksum)

        self._loaded = True

    def search(
        self,
        query: str,
        code_system: str,
        top_k: int | None = None,
        threshold: float | None = None,
    ) -> list[dict]:
        if code_system not in self.indices:
            logger.warning(f"No index for code system: {code_system}")
            return []

        top_k     = top_k or RAG_TOP_K
        threshold = threshold or RAG_SIMILARITY_THRESHOLD

        query_emb = embed_texts([query])
        query_vec = np.array(query_emb, dtype=np.float32)
        faiss.normalize_L2(query_vec)

        scores, ids = self.indices[code_system].search(query_vec, top_k)

        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0 or score < threshold:
                continue
            entry = self.metadata[code_system][idx].copy()
            entry["similarity_score"] = round(float(score), 4)
            results.append(entry)

        return results

    def search_multi(
        self,
        query: str,
        code_systems: list[str] | None = None,
        top_k: int | None = None,
    ) -> dict[str, list[dict]]:
        code_systems = code_systems or _CODE_SYSTEMS
        return {cs: self.search(query, cs, top_k) for cs in code_systems}

    # ------------------------------------------------------------------
    # Internal — index management
    # ------------------------------------------------------------------

    def _all_indices_exist(self) -> bool:
        return all(
            (VECTOR_STORE_DIR / f"{cs}.index").exists() and
            (VECTOR_STORE_DIR / f"{cs}_meta.pkl").exists()
            for cs in _CODE_SYSTEMS
        )

    def _build_all_indices(self) -> None:
        logger.info("Building FAISS indices for all code systems — this may take a few minutes...")
        self._build_index("icd10", self._load_icd10_records())
        self._build_index("cpt",   self._load_cpt_records())
        self._build_index("hcpcs", self._load_hcpcs_records())
        logger.info("All FAISS indices built and saved")

    def _build_index(self, name: str, records: list[dict]) -> None:
        if not records:
            logger.warning(f"No records for {name} — skipping")
            return

        logger.info(f"Embedding {len(records)} {name.upper()} codes...")

        texts      = [r["embedding_text"] for r in records]
        embeddings = []
        batch_size = 2048

        for i in range(0, len(texts), batch_size):
            batch     = texts[i: i + batch_size]
            batch_emb = embed_texts(batch)
            embeddings.extend(batch_emb)
            logger.info(f"  Embedded {min(i + batch_size, len(texts))}/{len(texts)}")

        vectors = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(vectors)

        index = faiss.IndexFlatIP(EMBEDDING_DIMENSIONS)
        index.add(vectors)

        faiss.write_index(index, str(VECTOR_STORE_DIR / f"{name}.index"))

        meta = [{k: v for k, v in r.items() if k != "embedding_text"} for r in records]
        with open(VECTOR_STORE_DIR / f"{name}_meta.pkl", "wb") as f:
            pickle.dump(meta, f)

        self.indices[name]  = index
        self.metadata[name] = meta
        logger.info(f"  {name.upper()} index built: {index.ntotal} vectors")

    def _load_indices(self) -> None:
        for name in _CODE_SYSTEMS:
            idx_path  = VECTOR_STORE_DIR / f"{name}.index"
            meta_path = VECTOR_STORE_DIR / f"{name}_meta.pkl"
            if idx_path.exists() and meta_path.exists():
                self.indices[name] = faiss.read_index(str(idx_path))
                with open(meta_path, "rb") as f:
                    self.metadata[name] = pickle.load(f)
                logger.info(f"  Loaded {name.upper()}: {self.indices[name].ntotal} vectors")
            else:
                logger.warning(f"  {name.upper()} index missing — rebuilding")
                self._build_index(name, getattr(self, f"_load_{name}_records")())

    # ------------------------------------------------------------------
    # Internal — data loaders (flexible, format-agnostic)
    # ------------------------------------------------------------------

    def _load_icd10_records(self) -> list[dict]:
        """Load ICD-10-CM codes from any JSON list with 'code' + 'description' fields."""
        raw = _read_json(ICD10_FILE)
        records = []
        for entry in _as_list(raw):
            code   = _get_field(entry, ["code"]).strip().replace(".", "").upper()
            desc   = _get_field(entry, ["description", "long_description", "short_description"]).strip()
            status = _get_field(entry, ["status"]).lower()
            if not code or not desc:
                continue
            if status and status not in ("active", ""):
                continue
            dotted = f"{code[:3]}.{code[3:]}" if len(code) > 3 else code
            records.append({
                "code":        dotted,
                "code_raw":    code,
                "description": desc,
                "code_system": "ICD-10-CM",
                "embedding_text": f"ICD-10-CM {dotted}: {desc}",
            })
        logger.info(f"  Loaded {len(records)} ICD-10-CM records from {ICD10_FILE.name}")
        return records

    def _load_cpt_records(self) -> list[dict]:
        """Load CPT codes from a JSON list or a dict with a 'codes' list."""
        raw  = _read_json(CPT_FILE)
        data = _as_list(raw)
        records = []
        for entry in data:
            code       = _get_field(entry, ["code"]).strip()
            long_desc  = _get_field(entry, ["long_description"]).strip()
            short_desc = _get_field(entry, ["short_description", "description"]).strip()
            desc       = long_desc or short_desc
            if not code or not desc:
                continue
            records.append({
                "code":                code,
                "short_description":   short_desc,
                "long_description":    long_desc,
                "consumer_description": _get_field(entry, ["consumer_description"]),
                "code_system":         "CPT",
                "embedding_text":      f"CPT {code}: {desc}",
            })
        logger.info(f"  Loaded {len(records)} CPT records from {CPT_FILE.name}")
        return records

    def _load_hcpcs_records(self) -> list[dict]:
        """Load HCPCS Level II codes. Handles CMS fixed-width parse artifacts where
        the 'code' field may contain the 5-char code followed by extra text."""
        raw = _read_json(HCPCS_FILE)
        records = []
        seen: set[str] = set()

        for entry in _as_list(raw):
            raw_code = _get_field(entry, ["code"]).strip()
            code     = raw_code[:5] if len(raw_code) >= 5 else raw_code

            if len(code) != 5 or not code[0].isalpha() or not code[1:].isdigit():
                continue
            if code in seen:
                continue
            seen.add(code)

            # Description: try raw_code remainder first (CMS artifact), then named fields
            remainder = raw_code[5:].strip() if len(raw_code) > 5 else ""
            desc = (
                remainder
                or _get_field(entry, ["long_description", "short_description", "description"]).strip()
            )
            if not desc:
                continue

            records.append({
                "code":        code,
                "description": desc,
                "code_system": "HCPCS",
                "embedding_text": f"HCPCS {code}: {desc}",
            })

        logger.info(f"  Loaded {len(records)} HCPCS records from {HCPCS_FILE.name}")
        return records


# ------------------------------------------------------------------
# Helpers — keep loaders format-agnostic
# ------------------------------------------------------------------

def _read_json(path: Path) -> any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _as_list(data: any) -> list:
    """Normalise any top-level JSON shape to a flat list of code dicts."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        # Try common wrapper keys first
        for key in ("codes", "data", "items", "results", "entries"):
            if key in data and isinstance(data[key], list):
                return data[key]
        # Single-entry dict where each value is a list → concatenate
        all_lists = [v for v in data.values() if isinstance(v, list)]
        if all_lists:
            combined = []
            for lst in all_lists:
                combined.extend(lst)
            return combined
    return []


def _get_field(entry: dict, candidates: list[str]) -> str:
    """Return the first non-empty string value found among candidate field names."""
    for key in candidates:
        val = entry.get(key)
        if isinstance(val, str) and val.strip():
            return val
    return ""
