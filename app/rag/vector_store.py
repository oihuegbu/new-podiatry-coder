import json
import pickle
import numpy as np
import faiss
from pathlib import Path

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


class MedicalCodeVectorStore:
    """FAISS-backed vector store for medical code retrieval."""

    def __init__(self):
        self.indices: dict[str, faiss.IndexFlatIP] = {}
        self.metadata: dict[str, list[dict]] = {}
        self._loaded = False

    def build_or_load(self, force_rebuild: bool = False) -> None:
        VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

        index_exists = all(
            (VECTOR_STORE_DIR / f"{code_sys}.index").exists()
            for code_sys in ["icd10", "cpt", "hcpcs"]
        )

        if index_exists and not force_rebuild:
            self._load_indices()
        else:
            self._build_all_indices()

        self._loaded = True

    def _build_all_indices(self) -> None:
        logger.info("Building FAISS indices for all code systems — this may take a few minutes...")

        self._build_index("icd10", self._load_icd10_records())
        self._build_index("cpt", self._load_cpt_records())
        self._build_index("hcpcs", self._load_hcpcs_records())

        logger.info("All FAISS indices built and saved")

    def _build_index(self, name: str, records: list[dict]) -> None:
        if not records:
            logger.warning(f"No records for {name}, skipping index build")
            return

        logger.info(f"Embedding {len(records)} {name.upper()} codes...")

        texts = [r["embedding_text"] for r in records]

        embeddings = []
        batch_size = 2048
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
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

        self.indices[name] = index
        self.metadata[name] = meta
        logger.info(f"  {name.upper()} index built: {index.ntotal} vectors")

    def _load_indices(self) -> None:
        logger.info("Loading pre-built FAISS indices...")
        for name in ["icd10", "cpt", "hcpcs"]:
            idx_path = VECTOR_STORE_DIR / f"{name}.index"
            meta_path = VECTOR_STORE_DIR / f"{name}_meta.pkl"
            if idx_path.exists() and meta_path.exists():
                self.indices[name] = faiss.read_index(str(idx_path))
                with open(meta_path, "rb") as f:
                    self.metadata[name] = pickle.load(f)
                logger.info(f"  Loaded {name.upper()}: {self.indices[name].ntotal} vectors")
            else:
                logger.warning(f"  {name.upper()} index not found, will rebuild")
                self._build_index(name, getattr(self, f"_load_{name}_records")())

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

        top_k = top_k or RAG_TOP_K
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
        code_systems = code_systems or ["icd10", "cpt", "hcpcs"]
        return {cs: self.search(query, cs, top_k) for cs in code_systems}

    # --- Data loaders ---

    def _load_icd10_records(self) -> list[dict]:
        with open(ICD10_FILE) as f:
            raw = json.load(f)

        records = []
        for entry in raw:
            code = entry.get("code", "").strip()
            desc = entry.get("description", "").strip()
            if code and desc and entry.get("status") == "active":
                dotted = f"{code[:3]}.{code[3:]}" if len(code) > 3 else code
                records.append({
                    "code": dotted,
                    "code_raw": code,
                    "description": desc,
                    "code_system": "ICD-10-CM",
                    "embedding_text": f"ICD-10-CM {dotted}: {desc}",
                })
        return records

    def _load_cpt_records(self) -> list[dict]:
        with open(CPT_FILE) as f:
            raw = json.load(f)

        codes_list = raw.get("codes", raw) if isinstance(raw, dict) else raw
        records = []
        for entry in codes_list:
            code = entry.get("code", "").strip()
            long_desc = entry.get("long_description", "").strip()
            short_desc = entry.get("short_description", "").strip()
            if code and (long_desc or short_desc):
                records.append({
                    "code": code,
                    "short_description": short_desc,
                    "long_description": long_desc,
                    "consumer_description": entry.get("consumer_description", ""),
                    "code_system": "CPT",
                    "embedding_text": f"CPT {code}: {long_desc or short_desc}",
                })
        return records

    def _load_hcpcs_records(self) -> list[dict]:
        with open(HCPCS_FILE) as f:
            raw = json.load(f)

        records = []
        for entry in raw:
            raw_code = entry.get("code", "").strip()
            if len(raw_code) >= 5:
                code = raw_code[:5]
                if code[0].isalpha() and code[1:].isdigit():
                    desc = raw_code[5:].strip() if len(raw_code) > 5 else entry.get("short_description", "").strip()
                    if desc:
                        records.append({
                            "code": code,
                            "description": desc,
                            "code_system": "HCPCS",
                            "embedding_text": f"HCPCS {code}: {desc}",
                        })
        return records
