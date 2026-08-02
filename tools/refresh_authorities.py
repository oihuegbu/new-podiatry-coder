#!/usr/bin/env python3
"""Refresh stale automated authorities and report remaining blockers."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.compliance.refresh.preflight import refresh_stale_sources


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-current", action="store_true")
    args = parser.parse_args()
    try:
        report = refresh_stale_sources(require_current=args.require_current)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, default=str))
    return 0 if report["current"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
