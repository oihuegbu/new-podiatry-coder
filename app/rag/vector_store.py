import hashlib
import json
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
_UPSERT_BATCH  = 500


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

        if QDRANT_URL:
            logger.info(f"Connecting to Qdrant server at {QDRANT_URL}")
            self._client = QdrantClient(url=QDRANT_URL)
        else:
            QDRANT_DIR.mkdir(parents=True, exist_ok=True)
            logger.info(f"Using local Qdrant store at {QDRANT_DIR}")
            self._client = QdrantClient(path=str(QDRANT_DIR))

        logger.info(f"Loading dense model ({_DENSE_MODEL}) and BM25 sparse model...")
        self._dense_model  = TextEmbedding(_DENSE_MODEL)
        self._sparse_model = SparseTextEmbedding("Qdrant/bm25")

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
            self._build_all_collections()
            _CHECKSUM_FILE.write_text(current_checksum)

        self._loaded = True

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

    def _build_all_collections(self) -> None:
        logger.info("Building Qdrant hybrid collections — this may take a few minutes...")
        self._build_collection("icd10", self._load_icd10_records())
        self._build_collection("cpt",   self._load_cpt_records())
        self._build_collection("hcpcs", self._load_hcpcs_records())
        logger.info("All Qdrant collections built and persisted")

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

            self._client.upsert(collection_name=name, points=points)
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
