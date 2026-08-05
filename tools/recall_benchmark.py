#!/usr/bin/env python3
"""Retrieval quality benchmark — DATA-DERIVED, no hardcoded codes/terms/scenarios.

Recall is the retriever's job: if the correct code is not in the candidates, no
downstream layer can code it. Every probe (query + expected code) is DERIVED at
runtime from the authoritative, provenance-tagged synonym layers already in the repo
(data/codes/*_synonyms.json) — no hand-written probe list, no code literal anywhere.

THREE measures, because a single "recall" number was misleading:

  1. INGESTION self-retrieval (UPPER BOUND, not generalization).
     Query = one of the code's OWN synonym terms. Those synonyms are folded into the
     code's indexed embedding text, so a hit largely confirms INGESTION worked — it
     does NOT measure held-out clinician phrasing. Reported, but explicitly as a
     ceiling, so it is never mistaken for real-world recall.

  2. PERTURBED-query recall (GENERALIZATION PROXY).
     Query = a PERTURBED synonym (token subset + reorder) that is NOT verbatim in the
     indexed text, approximating the phrasing drift a real note introduces. The gap
     (upper bound − perturbed) is the benchmark's leakage sensitivity: a large gap
     means the ceiling was inflated by verbatim overlap.

  3. HARD-NEGATIVE discrimination (PRECISION PROXY).
     For a perturbed query that retrieves the target, does a code from a DIFFERENT
     family (data-derived: a different category prefix) rank at #1 above it? This is
     the wrong-neighbour / wrong-laterality confusion that shares descriptor language
     but denotes a different concept. Reported as a top-1-displacement rate.

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


def _family(code) -> str:
    """Data-derived family key: the code stem minus its most-specific trailing char
    (for ICD, the laterality/character axis; for CPT/HCPCS, adjacent-code family).
    Two codes with the same family are near-siblings; a different family is a genuine
    hard negative. No code or range is named — purely structural."""
    c = _norm(code)
    return c[:-1] if len(c) > 3 else c


def _perturb(term: str, rng: random.Random) -> str:
    """A phrasing-drift version of a synonym that is NOT verbatim-indexed: keep a
    shuffled ~70% token subset (min 2 tokens). Returns '' when the term is too short
    to perturb without collapsing to the original."""
    toks = [t for t in term.replace("/", " ").split() if t]
    if len(toks) < 3:
        return ""
    k = max(2, int(round(len(toks) * 0.7)))
    keep = rng.sample(toks, k)
    rng.shuffle(keep)
    out = " ".join(keep)
    return "" if out.lower() == term.lower() else out


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

    print(f"\nRetrieval quality (sample {args.per_system}/system, top-{args.top_k}, "
          f"seed {args.seed})")
    print("  [1] ingestion self-retrieval = UPPER BOUND (not generalization)")
    print("  [2] perturbed-query recall   = generalization proxy (held-out phrasing)")
    print("  [3] top-1 hard-neg displacement = precision proxy (wrong-family at #1)")
    print("-" * 70)

    for cs, fname in SYSTEM_SYNONYMS.items():
        terms = _load_terms(fname)
        pool = [c for c, syns in terms.items() if syns]
        if not pool:
            print(f"  {cs:6}: no synonym data — skipped")
            continue
        sample = rng.sample(pool, min(args.per_system, len(pool)))
        up_hits = pt_hits = pt_n = disp = n = 0
        for code in sample:
            n += 1
            verbatim = rng.choice(terms[code])
            res = store.search(verbatim, cs, top_k=args.top_k)
            if any(_norm(h.get("code")) == _norm(code) for h in res):
                up_hits += 1

            pq = _perturb(verbatim, rng)
            if not pq:
                continue
            pt_n += 1
            pres = store.search(pq, cs, top_k=args.top_k)
            rank = next((i for i, h in enumerate(pres)
                         if _norm(h.get("code")) == _norm(code)), None)
            if rank is not None:
                pt_hits += 1
                # hard-negative: a DIFFERENT-family code sitting at #1 above the target
                if rank > 0 and _family(pres[0].get("code")) != _family(code):
                    disp += 1

        ub = up_hits / n if n else 0.0
        pr = pt_hits / pt_n if pt_n else 0.0
        dr = disp / pt_hits if pt_hits else 0.0
        print(f"  {cs:6}: [1] upper {ub:.0%}   [2] perturbed {pr:.0%}   "
              f"(leakage gap {ub - pr:+.0%})   [3] displaced {dr:.0%}   "
              f"(n={n}, pert={pt_n})")
    print("-" * 70)
    print("  Interpretation: [2] approximates real recall; a large [1]-[2] gap means")
    print("  the old self-retrieval number was inflated by verbatim index overlap.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
