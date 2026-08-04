#!/usr/bin/env python3
"""Build data/codes/snomed_icd10_map.json — the SNOMED CT -> ICD-10-CM term map that
backs the coder's second authoritative diagnosis-resolution layer.

SNOMED CT is the comprehensive clinical terminology (millions of synonyms/eponyms);
the SNOMED CT US Edition ships the authoritative SNOMED -> ICD-10-CM cross-map. This
tool ingests the OFFICIAL RF2 release (not a mirror): it JOINS the ICD-10-CM extended
map refset (SNOMED concept -> ICD target) with the description file (concept ->
clinician terms) and inverts to term -> ICD-10-CM code(s), so a documented
eponym/synonym the ICD Alphabetic Index lacks resolves deterministically.

Only UNCONDITIONAL default maps (mapRule TRUE / OTHERWISE TRUE) are kept — age/sex/
context-conditional rules need patient data resolved elsewhere — and only targets that
exist in the loaded ICD-10-CM code set are emitted. Active rows only. No code is
authored here; every mapping is read from the authoritative release.

AUTOMATED: it self-locates the RF2 release (--release, else $SNOMED_RF2_DIR, else a
data/sources / ~/snomed glob) and runs as a refresh step
(tools/refresh_authoritative_data.py snomed_icd10). If no release is present it writes
nothing and exits 0 — the resolver degrades to the ICD Index + retrieval.

LICENSING: SNOMED CT US Edition is free for US use under the NLM UMLS license; ICD-10-CM
is public domain. Confirm the SNOMED license applies to your use before deploying.
"""
import argparse
import hashlib
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.core.config import DATA_DIR

_ICD = re.compile(r"^[A-Z]\d{2}[A-Z0-9]*$")
_FSN_TAG = re.compile(r"\s*\([^)]*\)\s*$")   # trailing SNOMED semantic tag on an FSN


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ",
                  str(s).lower().replace("'", ""))).strip()


def _dot(code: str) -> str:
    c = str(code).upper().replace(".", "")
    return c if len(c) <= 3 else f"{c[:3]}.{c[3:]}"


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _find_release(arg: str | None) -> Path | None:
    """Locate the unzipped RF2 release dir: explicit arg, then $SNOMED_RF2_DIR, then a
    glob of the conventional source locations (newest first)."""
    cands: list[Path] = []
    if arg:
        cands.append(Path(arg))
    if os.environ.get("SNOMED_RF2_DIR"):
        cands.append(Path(os.environ["SNOMED_RF2_DIR"]))
    for base in (ROOT / "data" / "sources", ROOT / "data" / "sources" / "snomed",
                 Path.home(), Path.home() / "snomed", Path("/snomed")):
        if base.exists():
            cands += sorted(base.glob("SnomedCT_*"), reverse=True)
    for c in cands:
        if c and c.is_dir():
            return c
    return None


def _one(release: Path, subdir: str, pattern: str) -> Path | None:
    hits = sorted((release / subdir).glob(pattern))
    return hits[0] if hits else None


def _icd_set() -> set[str]:
    """Undotted ICD-10-CM codes we carry — a map target not in it is dropped, so the
    crosswalk can never resolve to a code the coder can't validate."""
    try:
        rows = json.load(open(DATA_DIR / "codes" / "icd10cm_codes.json"))
        return {str(r["code"]).replace(".", "").upper()
                for r in rows if isinstance(r, dict) and r.get("code")}
    except Exception:
        return set()


def _cols(fh) -> dict:
    return {name: i for i, name in enumerate(fh.readline().rstrip("\n").split("\t"))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", help="path to the unzipped SNOMED CT RF2 release dir")
    args = ap.parse_args()

    release = _find_release(args.release)
    if not release:
        print("SNOMED RF2 release not found (--release / $SNOMED_RF2_DIR / data/sources "
              "/ ~/snomed) — skipping snomed_icd10_map build; resolver degrades gracefully.")
        return 0
    ext = _one(release, "Snapshot/Refset/Map", "*ExtendedMapSnapshot*.txt")
    desc = _one(release, "Snapshot/Terminology", "sct2_Description_Snapshot-en*.txt")
    if not ext or not desc:
        print(f"{release.name}: ExtendedMap/Description Snapshot file missing — skipping.")
        return 0

    icd = _icd_set()

    # 1) ExtendedMap: SNOMED concept -> ICD-10-CM code(s); active, unconditional, in-set.
    concept_codes: dict[str, set] = defaultdict(set)
    with open(ext, encoding="utf-8") as fh:
        c = _cols(fh)
        A, CID, RULE, TGT = c["active"], c["referencedComponentId"], c["mapRule"], c["mapTarget"]
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) <= TGT or f[A] != "1":
                continue
            if f[RULE].strip().upper() not in ("TRUE", "OTHERWISE TRUE"):
                continue
            tgt = f[TGT].strip().upper().replace(".", "")
            if not _ICD.match(tgt) or (icd and tgt not in icd):
                continue
            concept_codes[f[CID]].add(_dot(tgt))
    if not concept_codes:
        print(f"{release.name}: no ICD-10-CM maps parsed — skipping.")
        return 0

    # 2) Description: SNOMED concept -> clinician terms (active English), mapped concepts only.
    terms: dict[str, set] = defaultdict(set)
    with open(desc, encoding="utf-8") as fh:
        c = _cols(fh)
        A, LANG, CID, TERM = c["active"], c["languageCode"], c["conceptId"], c["term"]
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) <= TERM or f[A] != "1" or f[LANG] != "en":
                continue
            codes = concept_codes.get(f[CID])
            if not codes:
                continue
            t = _norm(_FSN_TAG.sub("", f[TERM]))
            if t:
                terms[t].update(codes)

    out = DATA_DIR / "codes" / "snomed_icd10_map.json"
    payload = {
        "source": "SNOMED CT US Edition (official RF2) -> ICD-10-CM extended map",
        "release": release.name,
        "map_file": ext.name,
        "map_sha256": _sha256(ext),
        "license": "SNOMED CT US Edition (NLM UMLS, free for US use); ICD-10-CM public domain",
        "provenance": ("official RF2 ExtendedMap x Description; unconditional default rules; "
                       "targets restricted to the loaded ICD-10-CM set; inverted to term->code"),
        "generated": datetime.now(timezone.utc).isoformat(),
        "count": len(terms),
        "terms": {t: sorted(v) for t, v in sorted(terms.items())},
    }
    out.write_text(json.dumps(payload, indent=1))
    print(f"{release.name}: {len(concept_codes)} mapped concepts -> {len(terms)} terms -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
