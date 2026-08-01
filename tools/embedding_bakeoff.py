#!/usr/bin/env python3
"""Measure candidate dense embedding models against the current one BEFORE
committing to an expensive full re-embed.

Swapping the embedding model means re-embedding ~95K codes (~90 min) — too
expensive to try blindly. This harness measures each candidate on the recall
benchmark over a MANAGEABLE eval corpus per code system (every benchmark answer
+ a random distractor sample), using the SAME enriched embedding_text as
production and the model's own query prefix (query_embed). It reports DENSE-ONLY
recall@k + MRR per model — the right signal for the model lever, since BM25 and
the reranker are model-independent. Only the winner earns the full re-embed.

Usage (in-container or local, downloads each model ~1GB):
  python tools/embedding_bakeoff.py [--distractors 3000] [--top-k 20]
"""
import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CANDIDATES = [
    "BAAI/bge-base-en-v1.5",              # current baseline
    "BAAI/bge-large-en-v1.5",
    "mixedbread-ai/mxbai-embed-large-v1",
    "snowflake/snowflake-arctic-embed-l",
]


def _norm(c):
    return str(c or "").replace(".", "").upper()


def _corpus_by_system(distractors: int, seed: int = 13):
    """{system: {code(norm): embedding_text}} — every benchmark answer for that
    system plus a random distractor sample, enriched exactly like production."""
    from app.rag.vector_store import MedicalCodeVectorStore
    from tools.recall_benchmark import PROBES
    vs = MedicalCodeVectorStore.__new__(MedicalCodeVectorStore)
    loaders = {"cpt": vs._load_cpt_records, "icd10": vs._load_icd10_records,
               "hcpcs": vs._load_hcpcs_records}
    answers = {}
    for _, code, cs in PROBES:
        answers.setdefault(cs, set()).add(_norm(code))
    rng = random.Random(seed)
    out = {}
    for cs, recs in ((cs, loaders[cs]()) for cs in loaders):
        by_code = {_norm(r.get("code") or r.get("code_raw")):
                   r["embedding_text"] for r in recs}
        keep = set(a for a in answers.get(cs, set()) if a in by_code)
        pool = [c for c in by_code if c not in keep]
        keep.update(rng.sample(pool, min(distractors, len(pool))))
        out[cs] = {c: by_code[c] for c in keep}
    return out


def _embed_corpus(model, corpus: dict):
    """Embed a system's corpus ONCE (documents), normalized. Returns
    (codes, matrix)."""
    import numpy as np
    codes = list(corpus)
    dv = np.array(list(model.embed([corpus[c] for c in codes])))
    dv /= (np.linalg.norm(dv, axis=1, keepdims=True) + 1e-9)
    return codes, dv


def _rank(model, codes, dv, query: str, top_k: int):
    import numpy as np
    qv = np.array(list(model.query_embed([query]))[0])
    qv /= (np.linalg.norm(qv) + 1e-9)
    order = (dv @ qv).argsort()[::-1][:max(top_k, 1)]
    return [codes[i] for i in order]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--distractors", type=int, default=3000)
    ap.add_argument("--top-k", type=int, default=20)
    args = ap.parse_args()

    from fastembed import TextEmbedding
    from tools.recall_benchmark import PROBES
    print(f"Building eval corpus ({args.distractors} distractors/system)...")
    corpus = _corpus_by_system(args.distractors)
    for cs, d in corpus.items():
        print(f"  {cs}: {len(d)} codes")

    results = {}
    for model_name in CANDIDATES:
        print(f"\n=== {model_name} ===")
        try:
            model = TextEmbedding(model_name)
        except Exception as exc:
            print(f"  load failed: {exc}")
            continue
        # embed each system's corpus ONCE, reuse across that system's probes
        embedded = {cs: _embed_corpus(model, corpus[cs]) for cs in corpus}
        rows = []
        for phrase, code, cs in PROBES:
            codes, dv = embedded[cs]
            ranked = _rank(model, codes, dv, phrase, args.top_k)
            rank = next((i + 1 for i, c in enumerate(ranked)
                         if c == _norm(code)), None)
            rows.append((rank is not None, 1.0 / rank if rank else 0.0))
            print(f"  [{cs}] {code:<8} rank={rank}")
        recall = sum(f for f, _ in rows) / len(rows)
        mrr = sum(r for _, r in rows) / len(rows)
        results[model_name] = (recall, mrr)
        print(f"  -> recall@{args.top_k} {recall:.0%}   MRR {mrr:.3f}")

    print("\n" + "=" * 62)
    print(f"{'model':<40} {'recall@k':>9} {'MRR':>7}")
    base = results.get(CANDIDATES[0], (0, 0))
    for m in CANDIDATES:
        if m in results:
            r, mrr = results[m]
            delta = "" if m == CANDIDATES[0] else \
                f"  (Δrecall {r-base[0]:+.0%}, ΔMRR {mrr-base[1]:+.3f})"
            print(f"{m:<40} {r:>8.0%} {mrr:>7.3f}{delta}")
    print("\nOnly commit to a full re-embed if a candidate clearly beats the "
          "baseline on BOTH recall and MRR.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
