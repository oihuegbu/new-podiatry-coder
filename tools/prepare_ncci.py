#!/usr/bin/env python3
"""Deterministically prepare the pinned authoritative NCCI PTP snapshot for BOTH the
production `app` image and the `test` image, from committed source only.

It rebuilds data/codes/ncci_data.json from CMS via tools/build_ncci_ptp.py and verifies the
result against the checked-in lock (data/sources/ncci_ptp.lock.json). A checksum mismatch or
a build failure exits non-zero so the image build FAILS LOUDLY rather than shipping
unverified or stale authoritative data. The builder is byte-deterministic for a given CMS
release, so the pinned checksum makes the artifact reproducible and provenance-verifiable.
(Codex F6-R8.)
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOCK = ROOT / "data" / "sources" / "ncci_ptp.lock.json"
OUT = ROOT / "data" / "codes" / "ncci_data.json"


def main() -> int:
    lock = json.loads(LOCK.read_text())
    want = str(lock["output_sha256"]).lower()
    rc = subprocess.call([sys.executable, str(ROOT / "tools" / "build_ncci_ptp.py")])
    if rc != 0 or not OUT.exists():
        print("prepare_ncci: NCCI build failed", file=sys.stderr)
        return 1
    got = hashlib.sha256(OUT.read_bytes()).hexdigest()
    if got != want:
        print(f"prepare_ncci: CHECKSUM MISMATCH -- built {got}, pinned {want}. The CMS NCCI "
              f"release has drifted from {LOCK.relative_to(ROOT)}; review provenance and "
              f"update the lock deliberately.", file=sys.stderr)
        return 1
    print(f"prepare_ncci: NCCI verified against lock ({got}, {lock['pairs']} pairs, "
          f"effective {lock['effective_from']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
