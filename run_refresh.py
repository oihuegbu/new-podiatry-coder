"""CLI for the compliance data refresh layer.

  python run_refresh.py --all                 # refresh every source due this month
  python run_refresh.py --source ncci_ptp     # refresh one source
  python run_refresh.py --source mue --file path/to/MUE.csv   # ingest a local file (offline)
  python run_refresh.py --all --dry-run       # download+parse, report counts, don't write
  python run_refresh.py --history             # show ingested-snapshot provenance

Schedule (cron) per the Excel cadence — e.g.:
  weekly  : MCD articles
  quarterly (Jan/Apr/Jul/Oct): NCCI PTP, MUE, PFS
  annual  : ICD-10 (Oct), CPT (Jan)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.compliance.datastore.store import ComplianceDataStore
from app.compliance.refresh.runner import refresh_all, refresh_source


def main() -> int:
    ap = argparse.ArgumentParser(description="Compliance data refresh")
    ap.add_argument("--source", help="refresh a single source id")
    ap.add_argument("--all", action="store_true", help="refresh all sources due this month")
    ap.add_argument("--file", help="ingest a local file for --source (offline)")
    ap.add_argument("--effective-from", help="effective date for the snapshot (YYYY-MM-DD)")
    ap.add_argument("--month", type=int, help="override calendar month for --all cadence")
    ap.add_argument("--dry-run", action="store_true", help="parse only; do not write")
    ap.add_argument("--history", action="store_true", help="show snapshot provenance and exit")
    args = ap.parse_args()

    store = ComplianceDataStore()
    store.build_or_load()

    if args.history:
        for v in store.version_history():
            print(f"  {v['ingested_at']}  {v['source_id']:<14} eff={v['effective_from']:<12} "
                  f"rows={v['row_count']:<7} {v['file_name']}")
        return 0

    if args.source:
        local = Path(args.file).read_bytes() if args.file else None
        res = refresh_source(store, args.source, effective_from=args.effective_from,
                             local_bytes=local, dry_run=args.dry_run)
        print(json.dumps(res, indent=2))
        return 0 if res.get("ok") else 1

    if args.all:
        results = refresh_all(store, month=args.month, dry_run=args.dry_run)
        print(json.dumps(results, indent=2))
        return 0 if all(r.get("ok") for r in results) else 1

    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
