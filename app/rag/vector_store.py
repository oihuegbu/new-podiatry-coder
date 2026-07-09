import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    PointStruct,
    SparseVector,
    Prefetch,
    FusionQuery,
    Fusion,
)
from fastembed import TextEmbedding, SparseTextEmbedding

from app.core.config import (
    QDRANT_DIR,
    QDRANT_URL,
    ICD10_FILE,
    CPT_FILE,
    HCPCS_FILE,
    RAG_TOP_K,
    RAG_SIMILARITY_THRESHOLD,
)
from app.core.logger import get_logger

logger = get_logger(__name__)

# Files whose content drives the index — any change triggers a rebuild
_INDEXED_FILES = [ICD10_FILE, CPT_FILE, HCPCS_FILE]
_CHECKSUM_FILE = QDRANT_DIR / "codes_checksum.txt"
_CODE_SYSTEMS  = ["icd10", "cpt", "hcpcs"]
_DENSE_MODEL   = "BAAI/bge-base-en-v1.5"   # 768-dim, strong on specialized text
_DENSE_DIMS    = 768
_UPSERT_BATCH  = 30


def _compute_checksum() -> str:
    """SHA-256 of each indexed file's size + mtime — detects any change without reading GBs."""
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
    """Qdrant-backed hybrid vector store: dense (OpenAI cosine) + sparse (BM25 keyword).

    Hybrid search with RRF fusion recovers codes that pure dense search misses:
    - BM25 catches exact medical term matches: "hammertoe", "bunionectomy", "L3020", "11721"
    - Dense catches semantic matches: "foot pain" → plantar fasciitis codes
    - RRF fusion ranks results from both channels together

    Smart rebuild logic:
    - First run (no collections)     → build and persist to disk
    - Same code files as last run    → load from disk (fast startup)
    - Any code file replaced/edited  → auto-rebuild
    - force_rebuild=True             → always rebuild
    """

    def __init__(self):
        self._client: QdrantClient | None = None
        self._dense_model: TextEmbedding | None = None
        self._sparse_model: SparseTextEmbedding | None = None
        self._loaded = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_or_load(self, force_rebuild: bool = False) -> None:
        # Checksum dir must always exist regardless of Qdrant mode
        _CHECKSUM_FILE.parent.mkdir(parents=True, exist_ok=True)

        self._open_client()

        current_checksum  = _compute_checksum()
        stored_checksum   = _CHECKSUM_FILE.read_text().strip() if _CHECKSUM_FILE.exists() else ""
        collections_exist = self._all_collections_exist()
        files_unchanged   = (stored_checksum == current_checksum)
        can_load          = collections_exist and files_unchanged and not force_rebuild

        if can_load:
            logger.info("Qdrant hybrid collections up to date — ready for search")
        else:
            if not collections_exist:
                logger.info("No Qdrant collections found — building for the first time...")
            elif not files_unchanged:
                logger.info("Code files changed — rebuilding Qdrant collections...")
            else:
                logger.info("Force rebuild requested — rebuilding Qdrant collections...")
            self._rebuild_via_subprocesses()
            _CHECKSUM_FILE.write_text(current_checksum)

        # Load the embedding models for query-time search (one query at a time is
        # light — the memory-heavy part is the bulk index build, handled above in
        # isolated subprocesses).
        self._load_models()
        self._loaded = True

    # ------------------------------------------------------------------
    # Client / model lifecycle
    # ------------------------------------------------------------------

    def _open_client(self) -> None:
        if self._client is not None:
            return
        if QDRANT_URL:
            logger.info(f"Connecting to Qdrant server at {QDRANT_URL}")
            self._client = QdrantClient(url=QDRANT_URL, timeout=3600)
        else:
            QDRANT_DIR.mkdir(parents=True, exist_ok=True)
            logger.info(f"Using local Qdrant store at {QDRANT_DIR}")
            self._client = QdrantClient(path=str(QDRANT_DIR))

    def _close_client(self) -> None:
        """Release the client (and, for the local store, its on-disk lock)."""
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None

    def _load_models(self) -> None:
        if self._dense_model is None or self._sparse_model is None:
            logger.info(f"Loading dense model ({_DENSE_MODEL}) and BM25 sparse model...")
            self._dense_model  = TextEmbedding(_DENSE_MODEL)
            self._sparse_model = SparseTextEmbedding("Qdrant/bm25")

    def build_one(self, name: str) -> None:
        """Build a single collection end-to-end. Called inside a dedicated
        subprocess (see app/rag/build_collection.py) so all memory — including
        the onnxruntime arena — is returned to the OS when the process exits."""
        self._open_client()
        self._load_models()
        self._build_collection(name, self._records_for(name))
        self._close_client()

    def _records_for(self, name: str) -> list[dict]:
        loaders = {
            "icd10": self._load_icd10_records,
            "cpt":   self._load_cpt_records,
            "hcpcs": self._load_hcpcs_records,
        }
        if name not in loaders:
            raise ValueError(f"unknown collection '{name}' (expected one of {_CODE_SYSTEMS})")
        return loaders[name]()

    def search(
        self,
        query: str,
        code_system: str,
        top_k: int | None = None,
        threshold: float | None = None,
    ) -> list[dict]:
        if not self._loaded:
            return []

        top_k     = top_k or RAG_TOP_K
        threshold = threshold or RAG_SIMILARITY_THRESHOLD

        # Dense embedding via FastEmbed
        dense_vec = list(self._dense_model.embed([query]))[0].tolist()

        # Sparse BM25 embedding via FastEmbed
        sparse_emb = list(self._sparse_model.embed([query]))[0]
        sparse_vec = SparseVector(
            indices=sparse_emb.indices.tolist(),
            values=sparse_emb.values.tolist(),
        )

        # Hybrid search: dense prefetch (with cosine threshold) + sparse prefetch, fused with RRF
        results = self._client.query_points(
            collection_name=code_system,
            prefetch=[
                Prefetch(
                    query=dense_vec,
                    using="dense",
                    limit=top_k * 2,
                    score_threshold=threshold,  # threshold on cosine similarity
                ),
                Prefetch(
                    query=sparse_vec,
                    using="sparse",
                    limit=top_k * 2,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=top_k,
            with_payload=True,
        )

        output = []
        for point in results.points:
            entry = dict(point.payload)
            entry["similarity_score"] = round(float(point.score), 4)
            output.append(entry)
        return output

    def search_multi(
        self,
        query: str,
        code_systems: list[str] | None = None,
        top_k: int | None = None,
    ) -> dict[str, list[dict]]:
        code_systems = code_systems or _CODE_SYSTEMS
        return {cs: self.search(query, cs, top_k) for cs in code_systems}

    # ------------------------------------------------------------------
    # Internal — collection management
    # ------------------------------------------------------------------

    def _all_collections_exist(self) -> bool:
        existing = {c.name for c in self._client.get_collections().collections}
        return all(cs in existing for cs in _CODE_SYSTEMS)

    def _rebuild_via_subprocesses(self) -> None:
        """Build each collection in its OWN subprocess.

        Embedding 94K+ codes across three collections in a single process makes
        the onnxruntime memory arena grow without ever returning memory to the OS,
        which OOM-crashes lower-RAM / CPU-only machines. Running each collection
        build as a separate process that fully exits forces the OS to reclaim ALL
        of that memory (arena included) between collections, capping peak usage at
        a single collection's worth.
        """
        logger.info("Building collections — each in its OWN subprocess so memory "
                    "(incl. the onnxruntime arena) is fully reclaimed between collections...")
        # Release the local-store lock so each build subprocess can open it.
        self._close_client()
        project_root = Path(__file__).resolve().parents[2]

        for cs in _CODE_SYSTEMS:
            logger.info(f"→ building '{cs}' in a fresh subprocess")
            proc = subprocess.run(
                [sys.executable, "-m", "app.rag.build_collection", cs],
                cwd=str(project_root),
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"Collection build failed for '{cs}' (subprocess exit {proc.returncode})"
                )

        # Reopen for this (parent) process to use for querying.
        self._open_client()
        logger.info("All Qdrant collections built (memory-isolated, one process each)")

    def _build_collection(self, name: str, records: list[dict]) -> None:
        if not records:
            logger.warning(f"No records for {name} — skipping")
            return

        # Drop and recreate to guarantee clean state
        if self._client.collection_exists(name):
            self._client.delete_collection(name)

        self._client.create_collection(
            collection_name=name,
            vectors_config={
                "dense": VectorParams(size=_DENSE_DIMS, distance=Distance.COSINE),
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(),
            },
        )

        logger.info(f"Embedding {len(records)} {name.upper()} codes (dense + BM25 sparse)...")
        texts    = [r["embedding_text"] for r in records]
        point_id = 0

        for batch_start in range(0, len(records), _UPSERT_BATCH):
            batch_records = records[batch_start: batch_start + _UPSERT_BATCH]
            batch_texts   = texts[batch_start: batch_start + _UPSERT_BATCH]

            # Dense embeddings via FastEmbed
            batch_dense = [v.tolist() for v in self._dense_model.embed(batch_texts)]

            # Sparse BM25 embeddings
            batch_sparse = list(self._sparse_model.embed(batch_texts))

            points = []
            for record, dense_emb, sparse_emb in zip(batch_records, batch_dense, batch_sparse):
                payload = {k: v for k, v in record.items() if k != "embedding_text"}
                points.append(PointStruct(
                    id=point_id,
                    vector={
                        "dense": dense_emb,
                        "sparse": SparseVector(
                            indices=sparse_emb.indices.tolist(),
                            values=sparse_emb.values.tolist(),
                        ),
                    },
                    payload=payload,
                ))
                point_id += 1

            # Retry upsert up to 3 times — Qdrant's 5s REST keep-alive can
            # close the connection between batches (embedding takes 5-8s each).
            for _attempt in range(3):
                try:
                    self._client.upsert(collection_name=name, points=points, wait=True)
                    break
                except Exception as e:
                    if _attempt < 2 and "disconnected" in str(e).lower():
                        logger.warning(f"Qdrant disconnected on batch, reconnecting (attempt {_attempt+1}/3)...")
                        time.sleep(1)
                        self._close_client()
                        self._open_client()
                    else:
                        raise
            logger.info(f"  {name.upper()}: {min(batch_start + _UPSERT_BATCH, len(records))}/{len(records)}")

        logger.info(f"  {name.upper()} collection built: {point_id} points")

    # ------------------------------------------------------------------
    # Internal — data loaders (format-agnostic, same logic as before)
    # ------------------------------------------------------------------

    def _load_icd10_records(self) -> list[dict]:
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
                "code":            dotted,
                "code_raw":        code,
                "description":     desc,
                "code_system":     "ICD-10-CM",
                "embedding_text":  f"ICD-10-CM {dotted}: {desc}",
            })
        logger.info(f"  Loaded {len(records)} ICD-10-CM records from {ICD10_FILE.name}")
        return records

    def _load_cpt_records(self) -> list[dict]:
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
                "code":                 code,
                "short_description":    short_desc,
                "long_description":     long_desc,
                "consumer_description": _get_field(entry, ["consumer_description"]),
                "code_system":          "CPT",
                "embedding_text":       f"CPT {code}: {desc}",
            })
        logger.info(f"  Loaded {len(records)} CPT records from {CPT_FILE.name}")
        return records

    def _load_hcpcs_records(self) -> list[dict]:
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
            remainder = raw_code[5:].strip() if len(raw_code) > 5 else ""
            desc = (
                remainder
                or _get_field(entry, ["long_description", "short_description", "description"]).strip()
            )
            if not desc:
                continue
            records.append({
                "code":            code,
                "description":     desc,
                "code_system":     "HCPCS",
                "embedding_text":  f"HCPCS {code}: {desc}",
            })
        logger.info(f"  Loaded {len(records)} HCPCS records from {HCPCS_FILE.name}")
        return records


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _read_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _as_list(data) -> list:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("codes", "data", "items", "results", "entries"):
            if key in data and isinstance(data[key], list):
                return data[key]
        all_lists = [v for v in data.values() if isinstance(v, list)]
        if all_lists:
            combined = []
            for lst in all_lists:
                combined.extend(lst)
            return combined
    return []


def _get_field(entry: dict, candidates: list[str]) -> str:
    for key in candidates:
        val = entry.get(key)
        if isinstance(val, str) and val.strip():
            return val
    return ""
