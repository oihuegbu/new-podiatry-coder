#!/usr/bin/env python3
"""Promote the learned verified-resolution observations into the deterministic
crosswalk -> data/codes/learned_cpt_index.json.

Reads the append-only observation log written by propose-then-verify
(learned_observations.jsonl) and promotes a phrase->code mapping to deterministic
trust ONLY when it is confirmed across >= PROMOTE_AT distinct encounters and is
unambiguous (the winning code dominates any competitor). No human sign-off — cross-
encounter agreement is the automated gate. Run it on a schedule / in the data
refresh so newly-confirmed mappings become deterministic.

  python tools/build_learned_index.py [--promote-at N]

No medical code is authored here; entries are DATA accreted from verified
resolutions, each carrying the descriptor + evidence it was confirmed on.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    from claude_coder import learned
    ap = argparse.ArgumentParser()
    ap.add_argument("--promote-at", type=int, default=learned.PROMOTE_AT,
                    help="distinct encounters that must agree to promote a mapping")
    args = ap.parse_args()

    obs = learned.load_observations()
    payload = learned.build_index(promote_at=args.promote_at)
    n_obs = len(obs)
    n_entries = len(payload["entries"])
    n_phrases = len({o.get("phrase") for o in obs if o.get("phrase")})
    print(f"{n_obs} observations across {n_phrases} distinct phrases -> "
          f"{n_entries} promoted (>= {args.promote_at} distinct encounters, "
          f"unambiguous)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
