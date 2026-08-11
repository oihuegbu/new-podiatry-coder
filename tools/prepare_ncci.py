#!/usr/bin/env python3
"""Deterministically prepare the pinned authoritative NCCI PTP snapshot for BOTH the
production `app` image and the `test` image, from committed source only.

It rebuilds data/codes/ncci_data.json via tools/build_ncci_ptp.py from the PINNED release
INPUTS recorded in data/sources/ncci_ptp.lock.json — exact URLs with byte sizes and SHA-256
checksums, optionally served from a controlled immutable copy via NCCI_INPUT_DIR — and then
verifies the built OUTPUT against the same lock. Pinning both ends is what makes the build
reproducible: the builder no longer resolves whichever quarter CMS happens to expose today,
so a clean rebuild of a reviewed commit cannot silently change (or fail) because CMS rotated
a quarter. An input checksum mismatch, an output checksum mismatch, or a build failure exits
non-zero so the image build FAILS LOUDLY rather than shipping unverified or stale
authoritative data. Intentional version upgrades go through `build_ncci_ptp.py --refresh`,
which writes a PROPOSED lock for review instead of redefining this build. (Codex F6-R8.)
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
    inputs = lock.get("inputs")
    if not isinstance(inputs, list) or not inputs:
        print("prepare_ncci: the lock declares no pinned release inputs; refusing to build "
              "from unidentified upstream files.", file=sys.stderr)
        return 1
    print(f"prepare_ncci: building from {len(inputs)} pinned input(s) "
          f"({', '.join(str(i.get('name')) for i in inputs)})")
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
