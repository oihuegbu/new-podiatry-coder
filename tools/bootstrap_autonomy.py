#!/usr/bin/env python3
"""Materialize the signed autonomy scope from practice configuration."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.release.scope_bootstrap import ScopeBootstrapError, bootstrap_scope


def main() -> int:
    try:
        print(json.dumps(bootstrap_scope(), indent=2))
        return 0
    except ScopeBootstrapError as exc:
        print(f"autonomy bootstrap failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
