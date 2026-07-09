"""Regression harness — run the ClaimScrubber over every existing result fixture.

Uses the already-produced output/results/*.json files so we can validate the
scrubber deterministically and for free (no Vision/LLM/API calls). Run after
every change:  python -m tests.scrub_fixtures
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from app.compliance.datastore.store import ComplianceDataStore
from app.compliance.engine import ClaimScrubber, build_claim
from app.compliance.agents import build_default_agents

RESULTS_DIR = Path(__file__).resolve().parent.parent / "output" / "results"


def load_fixtures() -> list[dict]:
    out = []
    for p in sorted(RESULTS_DIR.glob("*_results.json")):
        try:
            data = json.loads(p.read_text())
        except Exception as exc:
            print(f"  ! could not load {p.name}: {exc}")
            continue
        if isinstance(data, dict) and data.get("document_id"):
            out.append(data)
    return out


def main() -> int:
    store = ComplianceDataStore()
    store.build_or_load()
    scrubber = ClaimScrubber(store, agents=build_default_agents(store))

    fixtures = load_fixtures()
    print(f"\nLoaded {len(fixtures)} fixtures | {len(scrubber.agents)} agents active\n")
    print(f"{'Document':<52} {'DOS':<11} {'CPT':>3} {'ICD':>3} {'Disp':<7} Fails")
    print("-" * 100)

    total_findings = 0
    for fx in fixtures:
        claim = build_claim(fx)
        res = scrubber.scrub(fx)
        total_findings += len(res.findings)
        fails = res.blocking_findings
        fail_str = ", ".join(sorted({f.filter_id for f in fails})) or "-"
        dos = claim.date_of_service.isoformat() if claim.date_of_service else "?"
        print(f"{fx['document_id'][:51]:<52} {dos:<11} {len(claim.lines):>3} "
              f"{len(claim.diagnoses):>3} {res.disposition.value:<7} {fail_str}")

    print("-" * 100)
    clean = sum(1 for fx in fixtures if scrubber.scrub(fx).clean)
    print(f"\nCLEAN: {clean}/{len(fixtures)}  |  total findings emitted: {total_findings}")

    # Detailed findings dump
    if "--verbose" in sys.argv:
        for fx in fixtures:
            res = scrubber.scrub(fx)
            if res.findings:
                print(f"\n### {fx['document_id']}")
                for f in res.findings:
                    print(f"  [{f.status.value}] {f.filter_id} ({f.denial_risk.value}) "
                          f"{f.codes} — {f.reason}")
                    if f.source_rule:
                        print(f"        ↳ {f.source_rule}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
