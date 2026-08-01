import hashlib
import json
import os
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

# The ICD-10-CM Index carries the clinical SYNONYMS and EPONYMS a note uses
# ('Haglund', 'bunionette') that a terse Tabular descriptor omits — folded
# into each diagnosis's embedding text so vocabulary-mismatched notes retrieve
# the right code.
ICD10_INDEX_TERMS_FILE = ICD10_FILE.parent / "icd10cm_index_terms.json"
# LLM-generated, grounded, provenance-tagged clinical-synonym indexes
# (tools/build_code_synonyms.py) — the eponym/clinician vocabulary the terse
# descriptors omit. CPT/HCPCS ship no synonym source at all; ICD's authoritative
# Index still misses eponyms, so its LLM file SUPPLEMENTS the Index. All
# optional — an absent file degrades to descriptor(+Index)-only embeddings.
CPT_SYNONYMS_FILE   = CPT_FILE.parent / "cpt_synonyms.json"
HCPCS_SYNONYMS_FILE = HCPCS_FILE.parent / "hcpcs_synonyms.json"
ICD10_SYNONYMS_FILE = ICD10_FILE.parent / "icd10_synonyms.json"
# Cap synonyms folded per code so a code with a long index list does not
# dominate/dilute its own embedding.
_INDEX_TERMS_PER_CODE = 12

# Files whose content drives the index — any change triggers a rebuild
_INDEXED_FILES = [ICD10_FILE, CPT_FILE, HCPCS_FILE, ICD10_INDEX_TERMS_FILE,
                  CPT_SYNONYMS_FILE, HCPCS_SYNONYMS_FILE, ICD10_SYNONYMS_FILE]


def _load_synonyms(path) -> dict:
    """{code: [synonyms]} from an LLM-synonym file, or {} if absent/malformed."""
    try:
        d = _read_json(path)
        t = d.get("terms", {}) if isinstance(d, dict) else {}
        return {str(k): v for k, v in t.items() if isinstance(v, list)}
    except Exception:
        return {}
_CHECKSUM_FILE = QDRANT_DIR / "codes_checksum.txt"
_CODE_SYSTEMS  = ["icd10", "cpt", "hcpcs"]


def _dedup_texts(texts) -> list[str]:
    """Unique, order-preserving, case-insensitive non-empty strings — so a
    code's several descriptor variants fold into one embedding text without
    redundant repetition."""
    seen: set[str] = set()
    out: list[str] = []
    for t in texts:
        t = (t or "").strip()
        k = t.lower()
        if t and k not in seen:
            seen.add(k)
            out.append(t)
    return out
_DENSE_MODEL   = "BAAI/bge-base-en-v1.5"   # 768-dim, strong on specialized text
_DENSE_DIMS    = 768
_UPSERT_BATCH  = 30
# Cross-encoder reranker: reads (query, candidate descriptor) together and
# reorders the high-recall fusion pool. DISABLED by default after measurement:
# on the recall benchmark it was NET-NEGATIVE (total recall 73%->60%, MRR
# 0.656->0.404; 28118 'Haglund resection' rank 1->17), because it scores the
# BARE DESCRIPTOR, which lacks the synonym enrichment — so it demotes exactly
# the eponym matches the synonyms just recovered ('onychomycosis' != the
# descriptor's 'dermatophytosis'). Kept behind the flag; a synonym-AWARE
# reranker (reranking against the enriched text, not the descriptor) could be
# revisited, but the bi-encoder + BM25 + synonyms already rank most answers #1,
# so the upside is marginal. Enable only with RAG_RERANK=1 after re-measuring.
_RERANK_MODEL   = os.getenv("RAG_RERANK_MODEL", "BAAI/bge-reranker-base")
_RERANK_DEPTH   = int(os.getenv("RAG_RERANK_DEPTH", "40"))
_RERANK_ENABLED = os.getenv("RAG_RERANK", "0") == "1"


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
        self._reranker = None          # lazy cross-encoder
        self._reranker_failed = False  # don't retry a broken load every query
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

        # Dense embedding via FastEmbed. bge-base-en-v1.5 is an ASYMMETRIC
        # retrieval model: queries carry a retrieval instruction prefix,
        # passages do not. query_embed() applies that prefix (passages were
        # indexed with plain embed()) — using plain embed() here under-embeds
        # every query and was silently degrading dense recall.
        dense_vec = list(self._dense_model.query_embed([query]))[0].tolist()

        # Sparse BM25 embedding via FastEmbed
        sparse_emb = list(self._sparse_model.query_embed([query]))[0]
        sparse_vec = SparseVector(
            indices=sparse_emb.indices.tolist(),
            values=sparse_emb.values.tolist(),
        )

        # Retrieve a WIDER pool than requested when a reranker is active, so
        # the cross-encoder can promote the right code even if the bi-encoder
        # fusion ranked it deep. Returned pool is reranked down to top_k below.
        pool = max(top_k, _RERANK_DEPTH) if self._reranker_enabled() else top_k
        # Hybrid search: dense prefetch (with cosine threshold) + sparse prefetch, fused with RRF
        results = self._client.query_points(
            collection_name=code_system,
            prefetch=[
                Prefetch(
                    query=dense_vec,
                    using="dense",
                    limit=pool * 2,
                    score_threshold=threshold,  # threshold on cosine similarity
                ),
                Prefetch(
                    query=sparse_vec,
                    using="sparse",
                    limit=pool * 2,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=pool,
            with_payload=True,
        )

        output = []
        for point in results.points:
            entry = dict(point.payload)
            entry["similarity_score"] = round(float(point.score), 4)
            output.append(entry)
        # Cross-encoder rerank: the bi-encoder + BM25 fusion is high-RECALL but
        # coarse on ranking; a cross-encoder reads the query and each
        # candidate's descriptor TOGETHER and reorders for precision, pushing
        # the right code toward rank 1 (measured: 28118 sat at rank 2 under
        # fusion alone). Degrades gracefully to fusion order if the reranker
        # is unavailable.
        output = self._rerank(query, output, top_k, code_system)
        return output[:top_k]

    def _reranker_enabled(self) -> bool:
        return _RERANK_ENABLED and not self._reranker_failed

    def _get_reranker(self):
        if self._reranker is None and not self._reranker_failed:
            try:
                from fastembed.rerank.cross_encoder import TextCrossEncoder
                logger.info(f"Loading cross-encoder reranker ({_RERANK_MODEL})")
                self._reranker = TextCrossEncoder(_RERANK_MODEL)
            except Exception as exc:
                logger.warning(f"Reranker unavailable ({exc}) — returning "
                               f"fusion order")
                self._reranker_failed = True
        return self._reranker

    def _syn_map(self, code_system: str) -> dict:
        """Lazy-cached {code: [synonyms]} for a system — the SAME LLM synonym
        files the embedding folds in, read here so the reranker can score the
        query against the ENRICHED text (descriptor + synonyms), not the bare
        descriptor. This is what makes reranking synonym-AWARE: scoring the
        bare descriptor demoted exactly the eponym matches the synonyms just
        recovered ('onychomycosis' != descriptor 'dermatophytosis')."""
        if not hasattr(self, "_syn_cache"):
            self._syn_cache = {}
        if code_system not in self._syn_cache:
            f = {"cpt": CPT_SYNONYMS_FILE, "hcpcs": HCPCS_SYNONYMS_FILE,
                 "icd10": ICD10_SYNONYMS_FILE}.get(code_system)
            self._syn_cache[code_system] = _load_synonyms(f) if f else {}
        return self._syn_cache[code_system]

    def _rerank_text(self, entry: dict, code_system: str) -> str:
        """The enriched text the cross-encoder scores against the query:
        descriptor variants PLUS the code's LLM synonyms — the same signal the
        embedding uses. ICD synonym files key on the UNDOTTED code, so try
        both the payload's dotted 'code' and undotted 'code_raw'."""
        code = str(entry.get("code") or "")
        smap = self._syn_map(code_system)
        syns = (smap.get(code)
                or smap.get(str(entry.get("code_raw") or ""))
                or smap.get(code.replace(".", "").upper()) or [])
        parts = [entry.get("long_description"), entry.get("description"),
                 entry.get("short_description"),
                 entry.get("consumer_description"),
                 *[str(s) for s in syns if str(s).strip()]]
        return " ".join(dict.fromkeys(
            p.strip() for p in parts if isinstance(p, str) and p.strip()))

    def _rerank(self, query: str, candidates: list[dict],
                top_k: int, code_system: str) -> list[dict]:
        """Reorder the fusion pool by cross-encoder relevance of (query,
        descriptor+synonyms). Fail-open: any error returns input order."""
        if not self._reranker_enabled() or len(candidates) <= 1:
            return candidates
        rr = self._get_reranker()
        if rr is None:
            return candidates
        try:
            docs = [self._rerank_text(c, code_system) for c in candidates]
            scores = list(rr.rerank(query, docs))
            for c, s in zip(candidates, scores):
                c["rerank_score"] = round(float(s), 4)
            return sorted(candidates, key=lambda c: c.get("rerank_score", 0.0),
                          reverse=True)
        except Exception as exc:
            logger.warning(f"Rerank failed ({exc}) — fusion order kept")
            return candidates

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
        # Index synonyms/eponyms, keyed by undotted code: {'M2161': ['bunion
        # of great toe', ...]}. Missing/malformed file degrades to
        # descriptor-only embeddings (no crash).
        index_terms: dict[str, list] = {}
        try:
            idx = _read_json(ICD10_INDEX_TERMS_FILE)
            terms = idx.get("terms", idx) if isinstance(idx, dict) else {}
            for k, v in terms.items():
                if isinstance(v, list):
                    index_terms[str(k).replace(".", "").upper()] = v
        except Exception as exc:
            logger.warning(f"ICD index terms unavailable ({exc}) — embedding "
                           f"descriptors only")
        # LLM synonym SUPPLEMENT: eponyms the authoritative Index misses
        # (measured: 'Haglund's deformity' -> M77.31 was not retrieved because
        # the Index lists only 'calcaneal spur').
        icd_syns = _load_synonyms(ICD10_SYNONYMS_FILE)
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
            # Fold in the Index's clinical synonyms/eponyms (capped) PLUS the
            # LLM supplement, so a note that names the condition the way a
            # clinician does — not the way the Tabular descriptor does — still
            # retrieves this code.
            syns = [str(s) for s in index_terms.get(code, [])
                    if str(s).strip()][:_INDEX_TERMS_PER_CODE]
            llm = [str(s) for s in icd_syns.get(code, [])
                   if str(s).strip()][:_INDEX_TERMS_PER_CODE]
            variants = _dedup_texts([desc, *syns, *llm])
            records.append({
                "code":            dotted,
                "code_raw":        code,
                "description":     desc,
                "code_system":     "ICD-10-CM",
                "embedding_text":  f"ICD-10-CM {dotted}: " + " ".join(variants),
            })
        logger.info(f"  Loaded {len(records)} ICD-10-CM records from "
                    f"{ICD10_FILE.name} ({len(index_terms)} with index terms)")
        return records

    def _load_cpt_records(self) -> list[dict]:
        raw  = _read_json(CPT_FILE)
        data = _as_list(raw)
        cpt_syns = _load_synonyms(CPT_SYNONYMS_FILE)
        records = []
        for entry in data:
            code       = _get_field(entry, ["code"]).strip()
            long_desc  = _get_field(entry, ["long_description"]).strip()
            short_desc = _get_field(entry, ["short_description", "description"]).strip()
            medium_desc   = _get_field(entry, ["medium_description"]).strip()
            consumer_desc = _get_field(entry, ["consumer_description"]).strip()
            desc       = long_desc or short_desc
            if not code or not desc:
                continue
            # Embed EVERY descriptor variant (deduped), not just the terse
            # long descriptor: the short/consumer forms use the note's
            # vocabulary ('Removal of heel bone') where the long form uses
            # CMS terminology ('Ostectomy, calcaneus;'). This is the recall
            # fix for the whole code set — a semicolon-parent or any code
            # whose surgeon-facing wording differs from its official
            # descriptor becomes retrievable on both dense and sparse.
            syns = [str(s) for s in cpt_syns.get(code, [])
                    if str(s).strip()][:_INDEX_TERMS_PER_CODE]
            variants = _dedup_texts(
                [long_desc, short_desc, medium_desc, consumer_desc, *syns])
            records.append({
                "code":                 code,
                "short_description":    short_desc,
                "long_description":     long_desc,
                "consumer_description": consumer_desc,
                "code_system":          "CPT",
                "embedding_text":       f"CPT {code}: " + " ".join(variants),
            })
        logger.info(f"  Loaded {len(records)} CPT records from {CPT_FILE.name}")
        return records

    def _load_hcpcs_records(self) -> list[dict]:
        raw = _read_json(HCPCS_FILE)
        hcpcs_syns = _load_synonyms(HCPCS_SYNONYMS_FILE)
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
            long_desc = _get_field(entry, ["long_description", "description"]).strip()
            short_desc = _get_field(entry, ["short_description"]).strip()
            desc = remainder or long_desc or short_desc
            if not desc:
                continue
            llm = [str(s) for s in hcpcs_syns.get(code, [])
                   if str(s).strip()][:_INDEX_TERMS_PER_CODE]
            variants = _dedup_texts([desc, long_desc, short_desc, *llm])
            records.append({
                "code":            code,
                "description":     desc,
                "code_system":     "HCPCS",
                "embedding_text":  f"HCPCS {code}: " + " ".join(variants),
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
