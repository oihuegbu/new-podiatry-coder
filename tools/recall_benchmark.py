#!/usr/bin/env python3
"""Retrieval recall@k benchmark — the measurement the RAG layer never had.

Recall is the retriever's job: if the correct code is not in the candidates,
no downstream layer can code it (measured: the Haglund ostectomy 28118 was
never retrieved, so it was never coded). This harness runs the LIVE hybrid
store over a curated set of (clinical phrase -> expected code) probes that
exercise the vocabulary-mismatch cases the embedding enrichment targets, and
reports recall@k per code system.

The probes are deliberately phrased the way a NOTE reads (surgeon/clinician
vocabulary, eponyms), NOT the way the official descriptor reads — that gap is
exactly what recall must bridge. Codes here are TEST EXPECTATIONS, not coding
logic (this file makes no claim decision), the same way benchmark gold files
carry codes.

Needs the built Qdrant index — run in-container on the instance:
  docker compose run --rm --no-deps -e PYTHONPATH=/app app \
      python tools/recall_benchmark.py [--top-k N]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# (clinical phrase as a note would write it, expected code, code_system).
# Deliberately note-vocabulary / eponym phrasing — the gap recall must bridge.
PROBES = [
    ("retrocalcaneal exostectomy, Haglund resection of the calcaneus",
     "28118", "cpt"),
    ("removal of prominent heel bone", "28118", "cpt"),
    ("Haglund resection", "28118", "cpt"),                 # pure eponym
    ("Achilles tendon debridement with reattachment using suture anchors",
     "27654", "cpt"),
    ("bunionette correction, fifth metatarsal head resection", "28110", "cpt"),
    ("hammertoe correction with proximal interphalangeal arthrodesis",
     "28285", "cpt"),
    ("Morton's neuroma excision, intermetatarsal nerve", "28080", "cpt"),
    ("Haglund's deformity of the right heel", "M77.31", "icd10"),
    ("pump bump of the heel", "M77.31", "icd10"),          # eponym/lay term
    ("hallux valgus, bunion of the great toe", "M20.11", "icd10"),
    ("plantar fasciitis", "M72.2", "icd10"),
    ("onychomycosis of the toenail", "B35.1", "icd10"),
    ("diabetic foot ulcer of the heel", "L97.4", "icd10"),
    ("compression burn garment, bodysuit", "A6501", "hcpcs"),
    ("therapeutic diabetic shoe, custom molded", "A5501", "hcpcs"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=None)
    args = ap.parse_args()

    from app.rag.vector_store import MedicalCodeVectorStore
    store = MedicalCodeVectorStore()
    store.build_or_load()

    def _norm(c):
        return str(c or "").replace(".", "").upper()

    by_system: dict[str, list[tuple[bool, float]]] = {}
    print(f"\nRecall@k + MRR benchmark (top_k={args.top_k or 'default'})\n"
          + "-" * 62)
    for phrase, expected, cs in PROBES:
        hits = store.search(phrase, cs, top_k=args.top_k)
        rank = next((i + 1 for i, h in enumerate(hits)
                     if _norm(h.get("code")) == _norm(expected)), None)
        found = rank is not None
        rr = 1.0 / rank if rank else 0.0
        by_system.setdefault(cs, []).append((found, rr))
        mark = "✅" if found else "❌"
        print(f"  {mark} [{cs}] {expected:<8} rank={str(rank):<4} «{phrase[:48]}»")

    print("-" * 62)
    allv = [v for vs in by_system.values() for v in vs]

    def _report(label, rows):
        r = sum(f for f, _ in rows) / len(rows)
        mrr = sum(rr for _, rr in rows) / len(rows)
        print(f"  {label:<14}: recall@k {sum(f for f,_ in rows)}/{len(rows)} "
              f"= {r:.0%}   MRR = {mrr:.3f}")
    for cs, rows in sorted(by_system.items()):
        _report(cs, rows)
    _report("TOTAL", allv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
