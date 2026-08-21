#!/usr/bin/env python3
"""Round-trip validate `data/codes/cpt_synonyms.json` against the authoritative
retrieval index, writing `data/codes/cpt_verified_synonyms.json` (issue #6,
compiled-semantic-layer plan item 3: "verified learned mappings that invalidate
when CPT descriptors change").

`cpt_synonyms.json`'s own provenance field says outright: "llm-generated...
RETRIEVAL AID ONLY -- NOT an authoritative source and never a coding decision
input". An LLM's self-assertion that term T is a synonym of code C is not
evidence of anything on its own. What IS real evidence: whether an INDEPENDENT,
authoritative mechanism -- the SAME hybrid retrieval index every other candidate
lookup in this pipeline already goes through, built entirely from licensed CPT
descriptors -- agrees, by actually returning C when searched with T alone. A
synonym that cannot find its own code through the real index is not grounded in
the authoritative data; a synonym that can has been INDEPENDENTLY corroborated
by data the LLM never had a hand in.

This is intentionally NOT run at request time: the retrieval index is a real
question against a live Qdrant collection, expensive per term, and this file
verifies ~10K codes worth of candidate synonyms once, producing a versioned
artifact `AuthoritativeSource.concept_lookup` can read cheaply per-process
(matching `tools/build_snomed_concept_terms.py`'s existing pattern of compiling
a big one-time check into a small runtime-consumable table). Re-run this
whenever `cpt_synonyms.json` or the compiled CPT retrieval index changes --
a synonym that no longer round-trips (because the descriptor changed, or the
index was rebuilt) is dropped on the NEXT run, not silently kept.

Usage (needs a running app stack with QDRANT_URL configured — NOT the isolated
test image, which has no network):
    docker compose run --rm app python tools/verify_cpt_synonyms.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.config import DATA_DIR
from claude_coder.data_access import AuthoritativeSource

#: A verified synonym must retrieve its own code within this many top results.
#: Not rank 1 -- the authoritative descriptor for the CORRECT code often loses to
#: a closely related sibling on pure lexical/embedding similarity even when the
#: synonym is genuinely valid (the same reason `resolution.py`'s own retrieval
#: uses top_k recall rather than a rank-1 requirement everywhere else). Within
#: the top 5 is real corroboration; anywhere else is not.
_TOP_K = 5


def main() -> int:
    raw_path = DATA_DIR / "codes" / "cpt_synonyms.json"
    with open(raw_path) as f:
        raw = json.load(f)
    terms: dict[str, list[str]] = raw.get("terms") or {}
    if not terms:
        print(f"{raw_path}: no terms to verify", file=sys.stderr)
        return 1

    source = AuthoritativeSource()
    verified: dict[str, list[str]] = {}
    checked = 0
    kept = 0
    for code, synonyms in terms.items():
        good = []
        for term in synonyms:
            checked += 1
            hits = source.retrieve(term, "cpt", top_k=_TOP_K)
            if any(h.code == code for h in hits):
                good.append(term)
                kept += 1
        if good:
            verified[code] = good
        if checked % 2000 == 0:
            print(f"  checked {checked} candidate synonyms, {kept} verified so far",
                  file=sys.stderr)

    out = {
        "provenance": (
            f"Round-trip validated against the authoritative CPT retrieval index "
            f"(top-{_TOP_K}): each kept synonym independently retrieves its own "
            f"originating code through the SAME hybrid search every candidate "
            f"lookup in this pipeline uses, built from licensed CPT descriptors "
            f"only -- not the LLM's self-assertion in {raw_path.name}. A synonym "
            f"present in {raw_path.name} but absent here failed that check."),
        "verified_against": "cpt_synonyms.json + the compiled CPT retrieval index",
        "top_k": _TOP_K,
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "candidate_count": checked,
        "verified_count": kept,
        "terms": verified,
    }
    out_path = DATA_DIR / "codes" / "cpt_verified_synonyms.json"
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True))
    print(f"wrote {out_path}: {kept}/{checked} candidate synonyms verified "
          f"across {len(verified)} codes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
