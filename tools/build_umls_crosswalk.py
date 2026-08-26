#!/usr/bin/env python3
"""Build data/codes/umls_cpt_snomed_crosswalk.json (the CPT/HCPCS <-> SNOMED CT
concept crosswalk) AND data/codes/umls_term_index.json (the term->CUI->atom
recall index), both derived from the UMLS Metathesaurus's own atom-to-concept
table.

================================================================================
umls_term_index.json — issue #6 F9-R7, product-priority reset item 2
================================================================================
Unlike the crosswalk above, this artifact IS consulted at candidate-recall
time (`AuthoritativeSource.umls_candidates`, `resolution.py`'s multi-query
union) -- but only as a RECALL SEED, never as selecting authority: it turns a
note phrase into a set of CPT/HCPCS candidates for the existing
descriptor-entailment / typed-facet-uniqueness / DOS-activity / CMS-validation
machinery to accept or reject, exactly like a RAG hit or an advisory synonym
expansion. It carries three indexes:

  term_to_cuis  -- normalized exact atom term -> CUIs, built from EVERY
                   unsuppressed English atom (any source vocabulary a
                   clinician's phrase might match: SNOMED, MeSH, ICD-10-CM,
                   etc.), but PRUNED to only CUIs that also appear in
                   `cui_to_atoms` below -- so the index stays proportional to
                   the billable-code concept space, not the whole UMLS
                   universe, while still capturing every synonym phrase for a
                   billable concept regardless of which vocabulary coined it.
  cui_to_atoms  -- CUI -> [{sab, code, term, tty}], restricted to CPT/HCPT/
                   HCPCS atoms whose CODE is present in the CURRENT
                   authoritative CPT/HCPCS registries this deployment already
                   loads (`CodeReferenceDB`, the same registry
                   `AuthoritativeSource.lookup` answers from) -- a code UMLS
                   merely mentions but the authoritative registry no longer
                   carries as current never appears here.
  code_to_cuis  -- "{system}:{code}" -> CUIs, the inverse of cui_to_atoms.

No fuzzy matching. Missing/ambiguous terms simply have no entry -- recall
degrades gracefully, exactly like the crosswalk above.

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
(tools/refresh_authoritative_data.py umls_crosswalk / umls_term_index). If no
release is present it writes nothing and exits 0 -- absence degrades this
recall/audit aid gracefully, exactly like snomed_crosswalk/snomed_concept_terms
already do.

ARCHIVE-TO-RUNTIME (issue #6 F9-R7 item 5): if no RRF is found but a staged raw
release archive is (`$UMLS_ARCHIVE_PATH`, else `data/sources/umls-release.zip`),
`_ensure_installed` runs tools/install_umls_release.sh against it FIRST -- the
already-scripted, already-proven MetamorphoSys headless "Batch Subset" extraction
-- before proceeding, so a configured owner-supplied archive regenerates and
activates both artifacts through this one refresh step with no second manual
command. No license, UTS authentication, or source-location approval workflow is
touched -- the archive must already be staged.

LICENSING: the UMLS Metathesaurus (including its bundled CPT/HCPCS/SNOMED CT US
content) requires an NLM UMLS license; confirm it applies to your use before
deploying. This tool records what release it was built from -- it never re-checks or
re-grants licensing itself.

Producing the RRF input this tool reads requires NLM's own MetamorphoSys tool, run in
its documented headless "Batch Subset" mode against a full UMLS release archive --
see tools/install_umls_release.sh (invoked automatically above when an archive is
staged, or runnable by hand), which scripts that step end to end (extract, build the
subset config, invoke MetamorphoSys, verify real output). This script only consumes
MetamorphoSys's OWN standard MRCONSO.RRF output, never the raw release archive or
its proprietary .nlm files directly.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.core.config import DATA_DIR
from claude_coder.terminology import normalize_term as _norm_term

# The RSAB values MRCONSO.RRF actually carries for a code billed as CPT/HCPCS in this
# system -- CPT (AMA CPT proper), HCPT (the CPT-within-HCPCS channel CMS republishes),
# HCPCS (CMS's own Level II codes). CPTSP (Spanish translation) and HCDT (dental) are
# deliberately excluded -- neither is a code system this pipeline bills against.
_BILLING_SABS = frozenset({"CPT", "HCPT", "HCPCS"})
_CONCEPT_SAB = "SNOMEDCT_US"
#: Which of this pipeline's OWN code systems ("cpt"/"hcpcs", matching
#: `CodeSource`/`CodeReferenceDB` attribute names) each billing RSAB names.
#: HCPT rows still describe CPT (5-digit numeric) codes -- CMS's own
#: republished channel, not a distinct code system.
_SAB_TO_SYSTEM = {"CPT": "cpt", "HCPT": "cpt", "HCPCS": "hcpcs"}

# MRCONSO.RRF column order, per NLM's documented, unversioned RRF schema (no header
# row ships in the file itself -- this order is the format, not a guess).
_CUI, _LAT, _SAB, _TTY, _CODE, _STR, _SUPPRESS = 0, 1, 11, 12, 13, 14, 16


def _load_current_codes() -> dict[str, set]:
    """{"cpt": {codes...}, "hcpcs": {codes...}} from the SAME authoritative
    registry `AuthoritativeSource.lookup` answers from -- never a second,
    hand-parsed notion of "current" (issue #6 F9-R7 item 2)."""
    from app.rag.code_reference import CodeReferenceDB
    db = CodeReferenceDB()
    db.load_all()
    return {"cpt": set(db.cpt.keys()), "hcpcs": set(db.hcpcs.keys())}


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


#: Where a staged owner-supplied raw UMLS release archive is expected, absent an
#: explicit override -- the SINGLE convention point issue #6 F9-R7 item 5 wires
#: the extraction step behind, so "owner drops an archive" -> "refresh runs" needs
#: no second manual command.
_DEFAULT_ARCHIVE = ROOT / "data" / "sources" / "umls-release.zip"
#: `install_umls_release.sh <archive> <scratch>` writes `<scratch>/rrf_output/` --
#: this is also one of `_find_release`'s own searched locations (a `data/sources/
#: umls` child), so a release this step just installed is found on the very next
#: call with no further wiring.
_INSTALL_SCRATCH = ROOT / "data" / "sources" / "umls"


def _ensure_installed(arg: str | None) -> Path | None:
    """`_find_release(arg)`, but when nothing is installed yet, first run the
    already-scripted, already-proven MetamorphoSys extraction
    (tools/install_umls_release.sh) against a staged owner-supplied archive --
    closing the "second manual command" gap between "owner drops an archive"
    and "refresh runs" (issue #6 F9-R7 item 5). Never touches licensing/UTS
    authentication/source-location approval -- the archive must already be
    staged; this only automates the EXTRACTION step that used to require a
    separate manual invocation."""
    found = _find_release(arg)
    if found:
        return found
    archive = Path(os.environ["UMLS_ARCHIVE_PATH"]) if os.environ.get("UMLS_ARCHIVE_PATH") \
        else _DEFAULT_ARCHIVE
    if not archive.is_file():
        return None
    installer = ROOT / "tools" / "install_umls_release.sh"
    print(f"{archive}: staged UMLS archive found, no extracted RRF yet -- running "
          f"{installer.name} into {_INSTALL_SCRATCH} ...")
    subprocess.run(["sh", str(installer), str(archive), str(_INSTALL_SCRATCH)],
                   check=True)
    return _find_release(arg)


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


def _build_crosswalk(release: Path, conso: Path, versions: dict[str, str]) -> None:
    """Unchanged from before issue #6 F9-R7: the CPT/HCPCS<->SNOMED concept
    crosswalk. Degrades gracefully (prints + returns) independently of
    `_build_term_index` -- neither artifact's absence blocks the other."""
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
        print(f"{release}: no CPT/HCPT/HCPCS rows parsed from MRCONSO.RRF -- "
              f"skipping umls_cpt_snomed_crosswalk.")
        return
    if not snomed_codes_by_cui:
        print(f"{release}: no SNOMEDCT_US rows parsed from MRCONSO.RRF -- "
              f"skipping umls_cpt_snomed_crosswalk.")
        return

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
              f"concept -- skipping umls_cpt_snomed_crosswalk.")
        return

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


def _build_term_index(release: Path, conso: Path, versions: dict[str, str]) -> None:
    """data/codes/umls_term_index.json -- see module docstring. Independent of
    `_build_crosswalk`: needs no SNOMEDCT_US rows at all, only CPT/HCPT/HCPCS
    atoms whose code is CURRENT in this deployment's own authoritative
    registry (issue #6 F9-R7 item 2)."""
    current = _load_current_codes()
    if not current["cpt"] and not current["hcpcs"]:
        print(f"{release}: no current CPT/HCPCS registry loaded -- skipping "
              f"umls_term_index.")
        return

    cui_to_atoms: dict[str, list[dict]] = defaultdict(list)
    code_to_cuis: dict[str, set] = defaultdict(set)
    useful_cuis: set = set()
    with open(conso, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            f = line.rstrip("\n").split("|")
            if len(f) <= _SUPPRESS:
                continue
            if f[_LAT] != "ENG" or f[_SUPPRESS] != "N":
                continue
            sab = f[_SAB]
            system = _SAB_TO_SYSTEM.get(sab)
            if system is None:
                continue
            code, cui = f[_CODE].strip(), f[_CUI]
            if not code or not cui or code not in current[system]:
                continue
            term = f[_STR].strip()
            key = f"{system}:{code}"
            cui_to_atoms[cui].append({"sab": sab, "code": code, "term": term,
                                      "tty": f[_TTY]})
            code_to_cuis[key].add(cui)
            useful_cuis.add(cui)
    if not useful_cuis:
        print(f"{release}: no CPT/HCPT/HCPCS atom's code is current in the "
              f"authoritative registry -- skipping umls_term_index.")
        return

    # Second pass: every unsuppressed English atom's term, from ANY source
    # vocabulary, but ONLY for a CUI that already has a current CPT/HCPCS atom
    # (`useful_cuis` above) -- keeps the index proportional to the billable
    # concept space while still capturing synonym phrasing UMLS's OTHER
    # vocabularies (SNOMED, MeSH, ICD-10-CM, ...) contribute for that concept.
    term_to_cuis: dict[str, set] = defaultdict(set)
    with open(conso, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            f = line.rstrip("\n").split("|")
            if len(f) <= _SUPPRESS:
                continue
            if f[_LAT] != "ENG" or f[_SUPPRESS] != "N":
                continue
            cui = f[_CUI]
            if cui not in useful_cuis:
                continue
            term = _norm_term(f[_STR])
            if term:
                term_to_cuis[term].add(cui)

    out = DATA_DIR / "codes" / "umls_term_index.json"
    payload = {
        "source": "UMLS Metathesaurus (MRCONSO.RRF, term/CUI/atom recall index "
                  "for CPT/HCPT/HCPCS)",
        "release": release.name,
        "sab_versions": versions,
        "mrconso_sha256": _sha256(conso),
        "license": "UMLS Metathesaurus (NLM UMLS license, including its bundled AMA "
                   "CPT and SNOMED CT US Edition sub-licenses); confirm it applies to "
                   "your use before deploying",
        "provenance": ("MRCONSO.RRF, English/non-suppressed rows only. RECALL SEED "
                       "ONLY -- consulted by AuthoritativeSource.umls_candidates to "
                       "widen retrieval, never to select or eliminate a code; the "
                       "existing descriptor-entailment/typed-facet-uniqueness/"
                       "DOS-activity/CMS-validation path remains the sole selecting "
                       "authority. term_to_cuis is pruned to CUIs that resolve to at "
                       "least one CURRENT CPT/HCPCS atom via cui_to_atoms."),
        "generated": datetime.now(timezone.utc).isoformat(),
        "cui_count": len(useful_cuis),
        "term_count": len(term_to_cuis),
        "code_count": len(code_to_cuis),
        "term_to_cuis": {t: sorted(cuis) for t, cuis in sorted(term_to_cuis.items())},
        "cui_to_atoms": {c: atoms for c, atoms in sorted(cui_to_atoms.items())},
        "code_to_cuis": {k: sorted(cuis) for k, cuis in sorted(code_to_cuis.items())},
    }
    out.write_text(json.dumps(payload, indent=1))
    print(f"{release.name}: {len(useful_cuis)} current-code CUIs -> "
          f"{len(term_to_cuis)} terms, {len(code_to_cuis)} codes -> {out}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", help="path to a directory containing the installed "
                                      "UMLS RRF subset's MRCONSO.RRF")
    args = ap.parse_args()

    release = _ensure_installed(args.release)
    if not release:
        print("UMLS RRF subset not found (--release / $UMLS_RRF_DIR / data/sources/umls "
              f"/ ~/umls), and no staged archive to extract it from ($UMLS_ARCHIVE_PATH "
              f"/ {_DEFAULT_ARCHIVE}) -- skipping umls_crosswalk/umls_term_index build; "
              f"resolver degrades gracefully.")
        return 0
    conso = release / "MRCONSO.RRF"
    versions = _sab_versions(release)

    _build_crosswalk(release, conso, versions)
    _build_term_index(release, conso, versions)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
