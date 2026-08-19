#!/usr/bin/env python3
"""Build data/codes/snomed_concept_terms.json — versioned SNOMED CT concept identity
and hierarchy for OPEN clinical vocabulary axes (issue #6 F7-R3-C, governed
terminology integration).

`claude_coder.coreference` compares open-vocabulary axis values (anatomy, approach,
site, ...) by lexical shape alone, which can only ever answer "identical string" or
"not identical" — it cannot tell a lay term and its clinical synonym for the same
structure from a real distinguishing qualifier, so it never asserts a difference from
wording and never merges an inexact match either (a genuine, but permanent, autonomy
ceiling). This table is the authoritative alternative: SNOMED CT's Body Structure
hierarchy already IS a governed, versioned map from a term to a stable CONCEPT, and
from a concept to its parents, which is exactly the "equivalence / ancestor-descendant
/ disjoint" relation open-vocabulary comparison needs and free lexical matching cannot
supply.

SCOPE: Body Structure only (SNOMED root concept 123037004 |Body structure|), the one
hierarchy the reviewer's exact counterexample (a lay anatomy term vs its clinical
synonym for the same structure) and this codebase's anatomy axis actually need. Other
open axes (approach, site, session, objective) have no SNOMED hierarchy that obviously
fits them the way anatomy fits Body Structure, and are left unresolved (the existing
conservative lexical-identity behavior) rather than mapped to a hierarchy that would
not actually answer the question asked of it. 123037004 is SNOMED's own root concept
id — an architecture constant of the terminology, not a clinical vocabulary term, so
scoping to it is not the kind of hardcoded scenario/term this project's fixes must
avoid.

Ingests the OFFICIAL RF2 release (not a mirror), joining:
  * the Relationship snapshot — active, INFERRED (the characteristic type actually
    computed from the full stated-relationship closure, not the possibly-incomplete
    authored subset), IS_A (116680003) rows — to build the concept hierarchy;
  * the Description snapshot — active, English (Synonym + FSN) rows — to build
    concept -> clinician term(s), FSN semantic tag stripped.

Only concepts within the Body Structure descendant closure are emitted (a BFS from
the root over the child->parent edges just parsed) — millions of unrelated SNOMED
concepts never enter the table, keeping it scoped to what this axis can use and
answerable from data this project actually needs, not a claim to cover all of SNOMED.

AUTOMATED: self-locates the RF2 release exactly like build_snomed_icd10_map.py
(--release, else $SNOMED_RF2_DIR, else a data/sources / ~/snomed glob) and runs as a
refresh step (tools/refresh_authoritative_data.py snomed_concept_terms). If no release
is present it writes nothing and exits 0 — every consumer already degrades to today's
lexical-identity-only comparison when this source is absent (REVIEWED-OPTIONAL, same
disposition as snomed_icd10_map).

LICENSING: SNOMED CT US Edition is free for US use under the NLM UMLS license.
Confirm the SNOMED license applies to your use before deploying.
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

#: SNOMED's own architecture constants -- not clinical vocabulary.
_BODY_STRUCTURE_ROOT = "123037004"
_IS_A_TYPE = "116680003"
_INFERRED_CHARACTERISTIC = "900000000000011006"
_FSN_TYPE = "900000000000003001"
_SYNONYM_TYPE = "900000000000013009"
_FSN_TAG = re.compile(r"\s*\([^)]*\)\s*$")   # trailing SNOMED semantic tag on an FSN


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ",
                  str(s).lower().replace("'", ""))).strip()


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _find_release(arg: str | None) -> Path | None:
    """Locate the unzipped RF2 release dir -- identical resolution order to
    build_snomed_icd10_map.py, so both tools point at the same release by default."""
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


def _cols(fh) -> dict:
    return {name: i for i, name in enumerate(fh.readline().rstrip("\n").split("\t"))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", help="path to the unzipped SNOMED CT RF2 release dir")
    args = ap.parse_args()

    release = _find_release(args.release)
    if not release:
        print("SNOMED RF2 release not found (--release / $SNOMED_RF2_DIR / data/sources "
              "/ ~/snomed) — skipping snomed_concept_terms build; open-vocabulary axis "
              "comparison degrades gracefully to lexical-identity-only.")
        return 0
    rel = _one(release, "Snapshot/Terminology", "sct2_Relationship_Snapshot_*.txt")
    desc = _one(release, "Snapshot/Terminology", "sct2_Description_Snapshot-en*.txt")
    if not rel or not desc:
        print(f"{release.name}: Relationship/Description Snapshot file missing — skipping.")
        return 0

    # 1) Relationship: active, inferred, IS_A rows -> child concept -> parent concept(s).
    parents: dict[str, set] = defaultdict(set)
    children: dict[str, set] = defaultdict(set)
    with open(rel, encoding="utf-8") as fh:
        c = _cols(fh)
        A, SRC, DST, TYPE, CHAR = (c["active"], c["sourceId"], c["destinationId"],
                                   c["typeId"], c["characteristicTypeId"])
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) <= CHAR or f[A] != "1":
                continue
            if f[TYPE] != _IS_A_TYPE or f[CHAR] != _INFERRED_CHARACTERISTIC:
                continue
            parents[f[SRC]].add(f[DST])
            children[f[DST]].add(f[SRC])
    if not parents:
        print(f"{release.name}: no inferred IS_A relationships parsed — skipping.")
        return 0

    # 2) BFS the descendant closure of Body Structure over the child edges just built.
    closure: set[str] = set()
    frontier = [_BODY_STRUCTURE_ROOT]
    while frontier:
        nxt: list[str] = []
        for cid in frontier:
            for child in children.get(cid, ()):
                if child not in closure:
                    closure.add(child)
                    nxt.append(child)
        frontier = nxt
    if not closure:
        print(f"{release.name}: Body Structure root {_BODY_STRUCTURE_ROOT} has no "
              f"descendants in this release — skipping.")
        return 0

    # 3) Description: active English (Synonym + FSN) terms, closure concepts only.
    terms: dict[str, set] = defaultdict(set)
    with open(desc, encoding="utf-8") as fh:
        c = _cols(fh)
        A, LANG, CID, TYPE, TERM = (c["active"], c["languageCode"], c["conceptId"],
                                    c["typeId"], c["term"])
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) <= TERM or f[A] != "1" or f[LANG] != "en":
                continue
            cid = f[CID]
            if cid not in closure:
                continue
            if f[TYPE] not in (_FSN_TYPE, _SYNONYM_TYPE):
                continue
            raw = f[TERM]
            t = _norm(_FSN_TAG.sub("", raw) if f[TYPE] == _FSN_TYPE else raw)
            if t:
                terms[cid].add(t)

    concepts = {cid: {"terms": sorted(terms[cid]),
                      "parents": sorted(parents.get(cid, ()) & closure)}
               for cid in closure if terms.get(cid)}

    out = DATA_DIR / "codes" / "snomed_concept_terms.json"
    payload = {
        "source": "SNOMED CT US Edition (official RF2) — Body Structure hierarchy",
        "release": release.name,
        "relationship_file": rel.name,
        "relationship_sha256": _sha256(rel),
        "description_file": desc.name,
        "description_sha256": _sha256(desc),
        "license": "SNOMED CT US Edition (NLM UMLS, free for US use)",
        "root_concept_id": _BODY_STRUCTURE_ROOT,
        "provenance": ("official RF2 Relationship (active, inferred, IS_A) x Description "
                       "(active, English, Synonym+FSN); BFS descendant closure of the "
                       "Body Structure root; parents restricted to the same closure"),
        "generated": datetime.now(timezone.utc).isoformat(),
        "concept_count": len(concepts),
        "concepts": concepts,
    }
    out.write_text(json.dumps(payload, indent=1))
    print(f"{release.name}: {len(closure)} Body Structure concepts, "
          f"{len(concepts)} with terms -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
