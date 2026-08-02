#!/usr/bin/env python3
"""Data-driven retrieval benchmark — NO hardcoded codes, terms, or scenarios.

Every query and its ground-truth code are DERIVED at runtime from the
authoritative data already in the repo: the code registry plus the generated,
provenance-tagged clinician-synonym layers (data/codes/*_synonyms.json). For a
random, seeded sample of codes in each system, the harness uses one of that
code's OWN clinician-vocabulary terms as the query and checks whether retrieval
returns that code (recall@k) and at what rank (MRR).

Because ground truth comes from the data, this tests ANY code, scales to
thousands of cases instead of a hand-curated few, and self-updates when the
CPT/HCPCS/ICD-10-CM sets change quarterly — there is no fixture to maintain and
no medical code literal anywhere in this file (guarded by the same principle as
the validator rule-packs).

What it measures: a code's own distinctive synonym is part of that code's
enriched embedding, so this is not an out-of-distribution generalization test —
it measures whether the enrichment actually achieves retrieval against SIBLING
COMPETITION at scale. Recall < 100% means a code's own term cannot surface it
past its neighbours (severe sibling crowding or an over-shared synonym); MRR <
1.0 means neighbours outrank it. Those are exactly the failure modes worth
tracking as the code sets and synonym layers evolve.

Usage (in-container on the box, against the built index):
  python tools/recall_benchmark.py [--per-system 300] [--top-k 20] [--seed 13]
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import DATA_DIR

# system -> the provenance-tagged synonym layer that supplies queries+truth.
SYSTEM_SYNONYMS = {
    "cpt": "cpt_synonyms.json",
    "hcpcs": "hcpcs_synonyms.json",
    "icd10": "icd10_synonyms.json",
}


def _norm(code) -> str:
    """Compare codes irrespective of ICD dotting / case."""
    return str(code or "").replace(".", "").upper()


def _load_terms(filename: str) -> dict[str, list[str]]:
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

    print(f"Data-driven recall@{args.top_k} + MRR "
          f"(sample {args.per_system}/system, seed {args.seed})")
    print("-" * 62)
    grand_hits = grand_n = 0
    grand_rr = 0.0
    for cs, fname in SYSTEM_SYNONYMS.items():
        terms = _load_terms(fname)
        # only codes that actually carry a distinctive clinician synonym
        pool = [c for c, syns in terms.items() if syns]
        if not pool:
            print(f"  {cs:6}: no synonym data — skipped")
            continue
        sample = rng.sample(pool, min(args.per_system, len(pool)))
        hits = n = 0
        rr = 0.0
        worst: list[tuple[str, str, int | None]] = []
        for code in sample:
            query = rng.choice(terms[code])          # a term for THIS code
            results = store.search(query, cs, top_k=args.top_k)
            rank = next((i + 1 for i, h in enumerate(results)
                         if _norm(h.get("code")) == _norm(code)), None)
            n += 1
            if rank:
                hits += 1
                rr += 1.0 / rank
            if rank is None or rank > 5:
                worst.append((code, query, rank))
        print(f"  {cs:6}: recall@{args.top_k} {hits}/{n} = {hits/n:.0%}   "
              f"MRR {rr/n:.3f}")
        for code, query, rank in worst[:3]:            # sample of hard cases
            print(f"           ↳ {code} rank={rank} «{query[:44]}»")
        grand_hits += hits
        grand_n += n
        grand_rr += rr
    if grand_n:
        print("-" * 62)
        print(f"  TOTAL : recall@{args.top_k} {grand_hits}/{grand_n} = "
              f"{grand_hits/grand_n:.0%}   MRR {grand_rr/grand_n:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
