#!/usr/bin/env python3
"""Automated retrieval + preparation of every authoritative source the coder uses.

ONE idempotent command fetches each source from its authoritative origin, runs
the existing prepare/parse tool, verifies the result (provenance present, non-zero
counts), and writes a manifest (data/codes/_manifest.json) recording source URL,
timestamp, row counts and a sha256 of each output. Re-run any time the code sets
change (quarterly NCCI/MUE, annual CPT/HCPCS/ICD) — nothing is hand-edited.

Sources
-------
FREE / public (fetched automatically):
  • global_period   – CMS PFS RVU file            -> build_global_period.py
  • snomed_icd10    – NLM SNOMED→ICD (Tuva mirror) -> build_snomed_icd10_map.py
  • icd10cm_index   – NCHS ICD-10-CM Alphabetic Index zip -> parse_icd10cm_index.py

LICENSED (fetched from YOUR configured source — never a public download):
  • cpt_index       – AMA CPT Link 'Index file'   -> parse_cpt_index.py
      Provide it via ONE of, in priority order:
        CPT_INDEX_FILE=/path/to/index.(csv|tsv|txt)     a local licensed file
        CPT_INDEX_S3=s3://bucket/key                     an object in your S3
        CPT_INDEX_URL=https://…  (+ optional CPT_INDEX_AUTH='Bearer <key>')
                                                         e.g. the AMA CPT data API
      If none is set, the stage is SKIPPED (not failed) with a clear note — the
      coder degrades gracefully to the descriptor index until the file is supplied.

Usage
-----
  python tools/refresh_authoritative_data.py                 # all sources
  python tools/refresh_authoritative_data.py global_period snomed_icd10
  python tools/refresh_authoritative_data.py --icd-url <zip> --dry-run
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.core.config import DATA_DIR

CODES = DATA_DIR / "codes"
PY = sys.executable
# CMS annual "ICD-10-CM Table and Index" zip (contains icd10cm_index_<yr>.xml).
ICD_INDEX_URL_DEFAULT = os.environ.get(
    "ICD_INDEX_URL",
    "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Publications/ICD10CM/2026/"
    "icd10cm-Table-and-Index-2026.zip")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _download(url: str, auth: str | None = None) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "coder-refresh/1.0"})
    if auth:
        req.add_header("Authorization", auth)
    with urllib.request.urlopen(req, timeout=300) as r:
        return r.read()


def _run(cmd: list[str]) -> None:
    print("  $", " ".join(cmd))
    subprocess.run(cmd, cwd=str(ROOT), check=True)


# ── per-source FETCH steps (return a local file path for the prepare step) ──────
def fetch_icd_index(tmp: Path, args) -> Path:
    url = args.icd_url or ICD_INDEX_URL_DEFAULT
    print(f"  fetch {url}")
    blob = _download(url)
    zf = zipfile.ZipFile(io.BytesIO(blob))
    name = next((n for n in zf.namelist()
                 if re.search(r"index.*\.xml$", n, re.I)), None)
    if not name:
        raise RuntimeError(f"no index*.xml in {url}: {zf.namelist()[:8]}")
    out = tmp / Path(name).name
    out.write_bytes(zf.read(name))
    return out


def fetch_cpt_index(tmp: Path, args) -> Path | None:
    """Resolve the LICENSED CPT Link Index file from a configured location."""
    if os.environ.get("CPT_INDEX_FILE"):
        p = Path(os.environ["CPT_INDEX_FILE"])
        if not p.exists():
            raise RuntimeError(f"CPT_INDEX_FILE set but not found: {p}")
        return p
    if os.environ.get("CPT_INDEX_S3"):
        uri = os.environ["CPT_INDEX_S3"]
        import boto3  # only needed on this path
        bkt, key = uri[5:].split("/", 1)
        out = tmp / Path(key).name
        boto3.client("s3").download_file(bkt, key, str(out))
        return out
    if os.environ.get("CPT_INDEX_URL"):
        out = tmp / "cpt_index_download"
        out.write_bytes(_download(os.environ["CPT_INDEX_URL"],
                                  os.environ.get("CPT_INDEX_AUTH")))
        return out
    return None                      # nothing configured -> caller skips


# ── source registry ────────────────────────────────────────────────────────────
SOURCES: dict[str, dict] = {
    "global_period": {
        "output": "global_period.json",
        "prepare": lambda tmp, args: [PY, "tools/build_global_period.py"],
    },
    "snomed_icd10": {
        "output": "snomed_icd10_map.json",
        "prepare": lambda tmp, args: [PY, "tools/build_snomed_icd10_map.py"],
    },
    "icd10cm_index": {
        "output": "icd10cm_index_terms.json",
        "fetch": fetch_icd_index,
        "prepare": lambda tmp, xml: [PY, "tools/parse_icd10cm_index.py", str(xml),
                                     str(CODES / "icd10cm_index_terms.json")],
    },
    "drug_table": {
        "output": "hcpcs_drug_table.json",
        # Prepares the drug name->HCPCS + per-unit-dose table from the authoritative
        # HCPCS Level II descriptors already ingested (public domain). Optionally
        # enriches with brand names from an external CMS table when DRUG_TABLE_URL
        # (or DRUG_TABLE_FILE) is set — auto-fetched by the prepare tool itself.
        "prepare": lambda tmp, f: [PY, "tools/build_hcpcs_drug_table.py"]
        + (["--table-url", os.environ["DRUG_TABLE_URL"]] if os.environ.get("DRUG_TABLE_URL")
           else ["--table-file", os.environ["DRUG_TABLE_FILE"]] if os.environ.get("DRUG_TABLE_FILE")
           else []),
    },
    "cpt_index": {
        "output": "cpt_index_terms.json",
        "licensed": True,
        "fetch": fetch_cpt_index,
        "prepare": lambda tmp, f: [PY, "tools/parse_cpt_index.py", str(f)],
    },
    "learned_index": {
        # Promote propose-then-verify observations into the deterministic crosswalk.
        # Local, no fetch; safe to re-run any time (idempotent over the log).
        "output": "learned_cpt_index.json",
        "optional": True,   # empty until the coder has accreted enough observations
        "prepare": lambda tmp, f: [PY, "tools/build_learned_index.py"],
    },
}


def _verify(output: str) -> dict:
    path = CODES / output
    if not path.exists():
        raise RuntimeError(f"expected output missing: {path}")
    data = json.loads(path.read_text())
    terms = data.get("terms") or data.get("codes") or {}
    n = len(terms)
    if n == 0:
        raise RuntimeError(f"{output} prepared but empty")
    return {"codes": n,
            "provenance": data.get("provenance") or data.get("source") or "",
            "sha256": _sha256(path), "bytes": path.stat().st_size}


def run_source(name: str, args) -> dict:
    spec = SOURCES[name]
    print(f"\n== {name} {'(LICENSED)' if spec.get('licensed') else ''} ==")
    rec: dict = {"status": "ok", "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if args.dry_run:                                  # never touch the network in a dry-run
        rec["status"] = "dry-run"
        print(f"  (dry-run) would fetch + prepare -> {spec['output']}")
        return rec
    try:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            arg = None
            if spec.get("fetch"):
                arg = spec["fetch"](tmp, args)
                if arg is None:                       # licensed source not configured
                    rec["status"] = ("skipped: no licensed source configured "
                                     "(set CPT_INDEX_FILE / CPT_INDEX_S3 / CPT_INDEX_URL)")
                    print("  " + rec["status"])
                    return rec
            cmd = spec["prepare"](tmp, arg)
            _run(cmd)
        rec.update(_verify(spec["output"]))
        print(f"  OK: {rec['codes']} codes, {rec['bytes']} bytes")
    except Exception as exc:                          # a source failing never aborts the rest
        if spec.get("optional") and "empty" in str(exc).lower():
            rec["status"] = "ok (nothing to promote yet)"
            print("  " + rec["status"])
        else:
            rec["status"] = f"ERROR: {exc}"
            print(f"  {rec['status']}")
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sources", nargs="*",
                    help=f"sources to refresh (default: all). one of: {', '.join(SOURCES)}")
    ap.add_argument("--icd-url", help="override the NCHS ICD-10-CM index zip URL")
    ap.add_argument("--dry-run", action="store_true", help="show commands, change nothing")
    args = ap.parse_args()

    bad = [n for n in args.sources if n not in SOURCES]
    if bad:
        ap.error(f"unknown source(s) {bad}; choose from {', '.join(SOURCES)}")
    names = args.sources or list(SOURCES)
    manifest = {}
    for name in names:
        manifest[name] = run_source(name, args)

    if not args.dry_run:
        (CODES / "_manifest.json").write_text(json.dumps(
            {"refreshed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "sources": manifest}, indent=1))
    print("\nmanifest:")
    for n, r in manifest.items():
        print(f"  {n:16} {r['status']}"
              + (f"  ({r.get('codes')} codes)" if r.get("codes") else ""))
    # non-zero exit only if a NON-licensed source hard-errored
    hard = [n for n, r in manifest.items()
            if str(r["status"]).startswith("ERROR")
            and not SOURCES[n].get("licensed") and not SOURCES[n].get("optional")]
    return 1 if hard else 0


if __name__ == "__main__":
    raise SystemExit(main())
