#!/usr/bin/env python3
"""Build data/codes/umls_cpt_snomed_crosswalk.json — the CPT/HCPCS <-> SNOMED CT
concept crosswalk derived from the UMLS Metathesaurus's own atom-to-concept table.

Neither CPT's own licensed data nor SNOMED CT's own RF2 release carries a crosswalk
between a CPT/HCPCS code and a SNOMED CT concept -- CPT has no such crosswalk at all,
and SNOMED's RF2 distribution never references CPT. The UMLS Metathesaurus is the one
artifact that clusters every source vocabulary's atoms under a shared CUI (Concept
Unique Identifier), so joining CPT/HCPCS atoms and SNOMEDCT_US atoms that share a CUI
is how this crosswalk gets built.

RECALL/AUDIT ARTIFACT ONLY -- never consulted to select or expand a CPT/HCPCS code.
A shared UMLS concept (or CUI) does not establish CPT billing-code equivalence: CPT
carves the same real-world clinical action into billing-specific categories (site,
technique, complexity tier) that do not line up 1:1 with SNOMED's or UMLS's own
concept granularity. This is the SAME boundary tools/build_snomed_procedure_terms.py's
own docstring and claude_coder/data_access.py's procedure_relation_detail() docstring
already establish for SNOMED CT Procedure concepts, stated here a third time for the
same reason: whoever consumes this artifact must not cross that line.

AUTOMATED: it self-locates the installed UMLS RRF subset (--release, else
$UMLS_RRF_DIR, else a data/sources/umls or ~/umls glob) and runs as a refresh step
(tools/refresh_authoritative_data.py umls_crosswalk). If no release is present it
writes nothing and exits 0 -- absence degrades this recall/audit aid gracefully,
exactly like snomed_crosswalk/snomed_concept_terms already do.

LICENSING: the UMLS Metathesaurus (including its bundled CPT/HCPCS/SNOMED CT US
content) requires an NLM UMLS license; confirm it applies to your use before
deploying. This tool records what release it was built from -- it never re-checks or
re-grants licensing itself.

Producing the RRF input this tool reads requires NLM's own MetamorphoSys tool, run in
its documented headless "Batch Subset" mode against a full UMLS release archive --
see tools/install_umls_release.sh, which scripts that step end to end (extract,
build the subset config, invoke MetamorphoSys, verify real output). This script only
consumes MetamorphoSys's OWN standard MRCONSO.RRF output, never the raw release
archive or its proprietary .nlm files directly.
"""
import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.core.config import DATA_DIR

# The RSAB values MRCONSO.RRF actually carries for a code billed as CPT/HCPCS in this
# system -- CPT (AMA CPT proper), HCPT (the CPT-within-HCPCS channel CMS republishes),
# HCPCS (CMS's own Level II codes). CPTSP (Spanish translation) and HCDT (dental) are
# deliberately excluded -- neither is a code system this pipeline bills against.
_BILLING_SABS = frozenset({"CPT", "HCPT", "HCPCS"})
_CONCEPT_SAB = "SNOMEDCT_US"

# MRCONSO.RRF column order, per NLM's documented, unversioned RRF schema (no header
# row ships in the file itself -- this order is the format, not a guess).
_CUI, _LAT, _SAB, _CODE, _STR, _SUPPRESS = 0, 1, 11, 13, 14, 16


def _find_release(arg: str | None) -> Path | None:
    """Locate a directory containing MRCONSO.RRF: explicit arg, then $UMLS_RRF_DIR,
    then a shallow (non-recursive-unbounded) search of the conventional source
    locations -- direct, one level down, or two levels down (matching NLM's own
    typical <install_dir>/META/MRCONSO.RRF layout)."""
    cands: list[Path] = []
    if arg:
        cands.append(Path(arg))
    if os.environ.get("UMLS_RRF_DIR"):
        cands.append(Path(os.environ["UMLS_RRF_DIR"]))
    for base in (ROOT / "data" / "sources", ROOT / "data" / "sources" / "umls",
                 Path.home(), Path.home() / "umls"):
        if not base.exists():
            continue
        if (base / "MRCONSO.RRF").is_file():
            cands.append(base)
        cands += sorted((p.parent for p in base.glob("*/MRCONSO.RRF")), reverse=True)
        cands += sorted((p.parent for p in base.glob("*/*/MRCONSO.RRF")), reverse=True)
    for c in cands:
        if c and (c / "MRCONSO.RRF").is_file():
            return c
    return None


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sab_versions(release: Path) -> dict[str, str]:
    """{RSAB -> versioned source string (e.g. "CPT" -> "CPT2026")} from MRSAB.RRF,
    when present and non-empty alongside MRCONSO.RRF. NLM's headless Batch Subset
    path does not regenerate a per-SAB MRSAB.RRF (GUI-only functionality, confirmed
    empirically) -- this degrades to {} rather than failing when it's absent, and the
    caller falls back to the release directory's own name for identity."""
    sab_file = release / "MRSAB.RRF"
    if not sab_file.is_file() or sab_file.stat().st_size == 0:
        return {}
    out: dict[str, str] = {}
    with open(sab_file, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            f = line.rstrip("\n").split("|")
            if len(f) <= 3:
                continue
            rsab, vsab = f[3], f[2]   # RSAB, VSAB per MRSAB.RRF's documented column order
            if rsab and vsab:
                out.setdefault(rsab, vsab)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", help="path to a directory containing the installed "
                                      "UMLS RRF subset's MRCONSO.RRF")
    args = ap.parse_args()

    release = _find_release(args.release)
    if not release:
        print("UMLS RRF subset not found (--release / $UMLS_RRF_DIR / data/sources/umls "
              "/ ~/umls) -- skipping umls_crosswalk build; resolver degrades gracefully.")
        return 0
    conso = release / "MRCONSO.RRF"

    billing_cuis_by_code: dict[str, set] = defaultdict(set)
    snomed_codes_by_cui: dict[str, set] = defaultdict(set)
    with open(conso, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            f = line.rstrip("\n").split("|")
            if len(f) <= _SUPPRESS:
                continue
            if f[_LAT] != "ENG" or f[_SUPPRESS] != "N":
                continue
            sab, code, cui = f[_SAB], f[_CODE].strip(), f[_CUI]
            if not code or not cui:
                continue
            if sab in _BILLING_SABS:
                billing_cuis_by_code[code].add(cui)
            elif sab == _CONCEPT_SAB:
                snomed_codes_by_cui[cui].add(code)
    if not billing_cuis_by_code:
        print(f"{release}: no CPT/HCPT/HCPCS rows parsed from MRCONSO.RRF -- skipping.")
        return 0
    if not snomed_codes_by_cui:
        print(f"{release}: no SNOMEDCT_US rows parsed from MRCONSO.RRF -- skipping.")
        return 0

    crosswalk: dict[str, dict] = {}
    for code, cuis in billing_cuis_by_code.items():
        concepts: set = set()
        for cui in cuis:
            concepts |= snomed_codes_by_cui.get(cui, set())
        if concepts:
            crosswalk[code] = {"cuis": sorted(cuis),
                               "matched_snomed_concept_ids": sorted(concepts)}
    if not crosswalk:
        print(f"{release}: no CPT/HCPCS code shared a CUI with any SNOMEDCT_US "
              f"concept -- skipping.")
        return 0

    versions = _sab_versions(release)
    out = DATA_DIR / "codes" / "umls_cpt_snomed_crosswalk.json"
    payload = {
        "source": "UMLS Metathesaurus (MRCONSO.RRF, CPT/HCPT/HCPCS x SNOMEDCT_US "
                  "sharing a CUI)",
        "release": release.name,
        "sab_versions": versions,
        "mrconso_sha256": _sha256(conso),
        "license": "UMLS Metathesaurus (NLM UMLS license, including its bundled AMA "
                   "CPT and SNOMED CT US Edition sub-licenses); confirm it applies to "
                   "your use before deploying",
        "provenance": ("MRCONSO.RRF, English/non-suppressed rows only, grouped by CUI; "
                       "a CPT/HCPT/HCPCS code links to every SNOMEDCT_US concept code "
                       "sharing at least one of its CUIs. Recall/audit artifact only -- "
                       "never consulted to select or expand a CPT/HCPCS code."),
        "generated": datetime.now(timezone.utc).isoformat(),
        "count": len(crosswalk),
        "crosswalk": dict(sorted(crosswalk.items())),
    }
    out.write_text(json.dumps(payload, indent=1))
    print(f"{release.name}: {len(billing_cuis_by_code)} CPT/HCPT/HCPCS codes -> "
          f"{len(crosswalk)} crosswalk entries -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
