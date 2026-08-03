#!/usr/bin/env python3
"""Retrieval recall@k + MRR benchmark — DATA-DERIVED, no hardcoded codes/terms/scenarios.

Recall is the retriever's job: if the correct code is not in the candidates, no
downstream layer can code it. This harness measures recall WITHOUT a hand-written
probe list — every probe (query + expected code) is DERIVED at runtime from the
authoritative, provenance-tagged clinician-synonym layers already in the repo
(data/codes/*_synonyms.json). For a seeded random sample of codes that carry a
distinctive synonym, one of that code's OWN synonym terms becomes the query and we
check whether retrieval returns that code (recall@k) and at what rank (MRR).

Because the ground truth comes from the data, this tests ANY code, scales to
thousands of cases, self-updates when the code/synonym sets change, and contains no
medical code literal, condition, or eponym anywhere in the file.

Needs the built Qdrant index — run in-container on the instance:
  docker compose run --rm --no-deps -e PYTHONPATH=/app app \
      python tools/recall_benchmark.py [--per-system 300] [--top-k 20] [--seed 13]
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import DATA_DIR

# system -> the provenance-tagged synonym layer that supplies queries + ground truth.
SYSTEM_SYNONYMS = {
    "cpt": "cpt_synonyms.json",
    "hcpcs": "hcpcs_synonyms.json",
    "icd10": "icd10_synonyms.json",
}


def _norm(code) -> str:
    return str(code or "").replace(".", "").upper()


def _load_terms(filename: str) -> dict:
    path = DATA_DIR / "codes" / filename
    if not path.exists():
        return {}
    try:
        return json.load(open(path)).get("terms", {}) or {}
    except Exception:
        return {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-system", type=int, default=300,
                    help="random codes sampled per system (capped at available)")
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--seed", type=int, default=13, help="reproducible sample")
    args = ap.parse_args()

    from app.rag.vector_store import MedicalCodeVectorStore
    store = MedicalCodeVectorStore()
    store.build_or_load()
    rng = random.Random(args.seed)

    print(f"\nData-derived recall@{args.top_k} + MRR "
          f"(sample {args.per_system}/system, seed {args.seed})\n" + "-" * 62)
    grand_hits = grand_n = 0
    grand_rr = 0.0
    for cs, fname in SYSTEM_SYNONYMS.items():
        terms = _load_terms(fname)
        pool = [c for c, syns in terms.items() if syns]   # codes with a distinctive synonym
        if not pool:
            print(f"  {cs:6}: no synonym data — skipped")
            continue
        sample = rng.sample(pool, min(args.per_system, len(pool)))
        hits = n = 0
        rr = 0.0
        worst = []
        for code in sample:
            query = rng.choice(terms[code])               # a synonym for THIS code
            results = store.search(query, cs, top_k=args.top_k)
            rank = next((i + 1 for i, h in enumerate(results)
                         if _norm(h.get("code")) == _norm(code)), None)
            n += 1
            if rank:
                hits += 1
                rr += 1.0 / rank
            elif len(worst) < 3:
                worst.append((code, query, rank))
        print(f"  {cs:6}: recall@{args.top_k} {hits}/{n} = {hits/n:.0%}   "
              f"MRR {rr/n:.3f}")
        for code, query, rank in worst:
            print(f"           ↳ rank={rank} «{query[:44]}»")
        grand_hits += hits
        grand_n += n
        grand_rr += rr
    if grand_n:
        print("-" * 62)
        print(f"  {'TOTAL':6}: recall@{args.top_k} {grand_hits}/{grand_n} = "
              f"{grand_hits/grand_n:.0%}   MRR {grand_rr/grand_n:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
