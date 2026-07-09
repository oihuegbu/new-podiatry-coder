"""Guard against re-introducing hardcoded medical code lists in agent logic.

The client requires the scrubber to be fully data-driven — all CPT/ICD/HCPCS
knowledge must come from the data store, never from literals in the agents.
This guard fails if an agent file contains a cluster of CPT-like (5-digit) or
ICD-like ([A-Z]NN...) quoted codes — the signature of a hardcoded list.

Modifiers (RT/LT/T1–T9/XE…), single structural references (e.g. the no-charge
code 99024), and CPT section *ranges* are allowed.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

AGENTS_DIR = Path(__file__).resolve().parent.parent / "app" / "compliance" / "agents"

_CPT_RE = re.compile(r"[\"'](\d{5})[\"']")            # "11055"
_ICD_RE = re.compile(r"[\"']([A-Z]\d{2}\.?\d{1,2})[\"']")  # "E11.42"
THRESHOLD = 3  # 1–2 structural references are fine; a real list has many


def main() -> int:
    violations = []
    for py in sorted(AGENTS_DIR.glob("*.py")):
        text = py.read_text()
        cpts = set(_CPT_RE.findall(text))
        icds = set(_ICD_RE.findall(text))
        if len(cpts) >= THRESHOLD:
            violations.append(f"{py.name}: {len(cpts)} hardcoded CPT-like codes {sorted(cpts)}")
        if len(icds) >= THRESHOLD:
            violations.append(f"{py.name}: {len(icds)} hardcoded ICD-like codes {sorted(icds)}")

    if violations:
        print("❌ HARDCODING GUARD FAILED — move these into the data store:")
        for v in violations:
            print(f"   - {v}")
        return 1
    print(f"✅ No hardcoded code lists found in {AGENTS_DIR.name}/ "
          f"({len(list(AGENTS_DIR.glob('*.py')))} files scanned)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
