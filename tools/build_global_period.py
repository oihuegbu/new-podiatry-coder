#!/usr/bin/env python3
"""Build data/codes/global_period.json as a PROJECTION of the authoritative
data/global_periods.json (issue #6 F9-R11-H-C), not a second, independently
downloaded CMS extract.

Global-surgical-package days (GLOB DAYS), the bilateral-surgery indicator
(BILAT SURG), and the CMS payment-status indicator (STATUS CODE) are all
published in the same CMS Physician Fee Schedule Relative Value File. This
project ALREADY ingests that file once, into data/global_periods.json (the
"global_periods" declared source, kept current by the compliance-refresh
pipeline) -- this tool used to download its OWN separate copy of the same CMS
zip from a hardcoded URL, which meant two independently-refreshed extracts of
the same underlying data could (and did: independently verified 67 codes and
30 status differences between an April copy this tool had downloaded and the
July copy already ingested) silently disagree, with the one THIS tool wrote
being the stale one on a claim-affecting classification path.

Fixed by projecting from the single authoritative source instead of a second
download: read data/global_periods.json, re-key its glob_days/bilat_surg/
status fields into the shape claude_coder.data_access already reads, and
write it back to the SAME declared "pfs_indicators" path -- so every existing
consumer, guard, and required-source declaration is unchanged; only the INPUT
this tool projects FROM changed. Provenance (source, version, url, and a
content hash of the exact parent bytes projected from) is carried into the
output so a stale projection is detectable against its own parent, never
silently possible again.

Values: 000 (0-day minor), 010 (10-day minor), 090 (90-day major), XXX (concept
does not apply), YYY (carrier-priced), ZZZ (add-on, global tied to the primary),
MMM (maternity).

Honest limitation, inherited from the parent source, not introduced here:
neither data/global_periods.json nor this projection is ingested into a
versioned effective-window table (see app/release/source_manifest.py's own
"release_metadata_exemption" for BOTH "global_periods" and "pfs_indicators" --
"not ingested into a versioned effective-window table; identity rests on the
content digest alone"). Genuine date-of-service-aware historical PFS
selection (choosing last quarter's vs. this quarter's status/global/bilateral
values by the encounter's own date) does not exist anywhere in this codebase
today, for either file -- this tool makes the two extracts CONSISTENT with
each other, it does not add a capability neither one has.

Usage (run on the box; no internet needed -- reads the already-ingested file):
  python tools/build_global_period.py
"""
import json
import sys
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.core.config import GLOBAL_PERIODS_FILE
from app.release.source_manifest import declared_source_path

GLOBAL_VALUES = {"000", "010", "090", "XXX", "YYY", "ZZZ", "MMM"}
BILAT_VALUES = {"0", "1", "2", "3", "9"}


def main() -> int:
    if not GLOBAL_PERIODS_FILE.exists():
        raise SystemExit(
            f"authoritative source {GLOBAL_PERIODS_FILE} does not exist -- run the "
            f"compliance-refresh pipeline (or ingest a PPRRVU file into it) before "
            f"projecting from it; this tool no longer downloads its own copy")
    raw_bytes = GLOBAL_PERIODS_FILE.read_bytes()
    parent = json.loads(raw_bytes)
    parent_codes = parent.get("codes") or {}
    if not parent_codes:
        raise SystemExit(f"{GLOBAL_PERIODS_FILE} declares no non-empty 'codes' table")

    codes: dict[str, dict] = {}
    for code, rec in parent_codes.items():
        if not isinstance(rec, dict):
            continue
        gval = str(rec.get("global_days") or "").strip().upper()
        if gval not in GLOBAL_VALUES:
            continue                              # base code rows only, same filter
                                                    # the old direct-CSV parse applied
        bval = str(rec.get("bilat_surg") or "").strip()
        sval = str(rec.get("status") or "").strip().upper()
        codes[str(code).strip().upper()] = {
            "global": gval,
            "bilat": bval if bval in BILAT_VALUES else "9",
            "status": sval,
        }

    out = declared_source_path("pfs_indicators")
    payload = {
        "source": "Projected from the authoritative 'global_periods' source "
                  "(data/global_periods.json) -- glob_days/bilat_surg/status "
                  "fields, re-keyed for claude_coder.data_access. Not a second "
                  "independently downloaded CMS extract (issue #6 F9-R11-H-C).",
        "parent_source": parent.get("source", ""),
        "parent_version": parent.get("version", ""),
        "parent_source_url": parent.get("source_url", ""),
        "parent_sha256": sha256(raw_bytes).hexdigest(),
        "generated": datetime.now(timezone.utc).isoformat(),
        "count": len(codes),
        "codes": dict(sorted(codes.items())),
    }
    out.write_text(json.dumps(payload, indent=1))
    print(f"Projected {len(codes)} codes from {GLOBAL_PERIODS_FILE.name} "
         f"(parent version: {parent.get('version', '?')}) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
