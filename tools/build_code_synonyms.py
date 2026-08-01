#!/usr/bin/env python3
"""Build LLM clinical-synonym indexes for CPT, HCPCS, and ICD-10-CM.

Official descriptors are terse CMS/AMA terminology; notes use clinician
vocabulary and eponyms ('Haglund resection', 'pump bump', 'jones fracture').
ICD ships an authoritative Index (used already); CPT and HCPCS ship none, and
the ICD Index still misses eponyms. This tool has an LLM GENERATE the missing
synonym layer for any code system — grounded against each code's own descriptor
and disambiguated against its numeric-neighbour siblings — and writes it as a
PROVENANCE-TAGGED, non-authoritative data file the embedding folds in.

Safety: RETRIEVAL AID ONLY. A synonym only lifts a code into the coder's
candidate list; the coder still judges each candidate against the real
descriptor and note, and validation gates the claim. A wrong synonym costs a
rejected candidate, never a wrong bill. Net effect is measured by
tools/recall_benchmark.py.

Usage (in-container, needs API key):
  python tools/build_code_synonyms.py --system cpt      # all CPT
  python tools/build_code_synonyms.py --system hcpcs
  python tools/build_code_synonyms.py --system icd10
  ... --prefix 28 / --limit 500 / --codes A,B  to scope
"""
import argparse
import datetime as _dt
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import DATA_DIR
from app.core.logger import get_logger

logger = get_logger(__name__)

_BATCH = 40
_WORKERS = 8
_MAX_TERMS = 10
_CODE_RE = re.compile(r"\b\d{4,5}\b|\b[A-Z]\d{4}\b")

SYSTEMS = {
    "cpt":   {"attr": "cpt",   "label": "CPT",
              "out": "cpt_synonyms.json",
              "kind": "surgical or procedural service",
              "writes": "an operative or procedure note"},
    "hcpcs": {"attr": "hcpcs", "label": "HCPCS Level II",
              "out": "hcpcs_synonyms.json",
              "kind": "supply, drug, DME item, or service",
              "writes": "a clinic note or order"},
    "icd10": {"attr": "icd10", "label": "ICD-10-CM",
              "out": "icd10_synonyms.json",
              "kind": "diagnosis / clinical condition",
              "writes": "a clinical note"},
}


def _sys_prompt(cfg: dict) -> str:
    return (
        f"You are a medical-coding vocabulary expert building a RETRIEVAL "
        f"synonym index for {cfg['label']}. For each code you are given, list "
        f"the clinical names, eponyms, brand/common terms, and phrasings a "
        f"clinician writes in {cfg['writes']} that map SPECIFICALLY to THAT "
        f"{cfg['kind']} — and not to the neighbouring codes shown for context. "
        f"Rules: (1) ground every term in the code's own descriptor — never "
        f"invent something the descriptor does not describe; (2) when codes "
        f"are near-siblings, assign each term to the SINGLE best-matching code; "
        f"(3) omit terms already in the descriptor and omit generic words; "
        f"(4) if a code has no distinctive clinical synonym, return an empty "
        f"list for it. Return JSON ONLY: an object mapping each given code "
        f"(string) to a list of short term strings."
    )


def _load(cfg: dict) -> list[tuple[str, str]]:
    from app.rag.code_reference import CodeReferenceDB
    db = CodeReferenceDB()
    db.load_all()
    src = getattr(db, cfg["attr"])
    out = []
    for code, rec in src.items():
        if not isinstance(rec, dict):
            continue
        if cfg["attr"] == "icd10" and str(rec.get("status", "active")).lower() \
                not in ("active", ""):
            continue
        desc = (rec.get("long_description") or rec.get("description")
                or rec.get("short_description") or "").strip()
        if code and desc:
            out.append((str(code), desc))
    out.sort(key=lambda x: x[0])
    return out


def _clean(terms, descriptor: str) -> list[str]:
    desc_low = descriptor.lower()
    seen, out = set(), []
    for t in terms or []:
        t = str(t or "").strip()
        low = t.lower()
        if (not t or len(t) < 3 or len(t) > 60 or _CODE_RE.search(t)
                or low in desc_low or low in seen):
            continue
        seen.add(low)
        out.append(t)
    return out[:_MAX_TERMS]


def _gen_batch(batch, sysprompt) -> dict[str, list[str]]:
    from app.core.llm_client import chat_completion
    listing = "\n".join(f"{c}: {d}" for c, d in batch)
    user = (f"Codes (with descriptors, for mutual sibling context):\n{listing}"
            f"\n\nReturn the JSON synonym map for these codes.")
    try:
        text, _ = chat_completion(sysprompt, user, temperature=0.2,
                                  max_tokens=4000, json_mode=True)
        raw = json.loads(text)
    except Exception as exc:
        logger.warning(f"batch {batch[0][0]}..{batch[-1][0]} failed: {exc}")
        return {}
    desc_of = dict(batch)
    out = {}
    for code, terms in (raw.items() if isinstance(raw, dict) else []):
        code = str(code).strip()
        if code in desc_of:
            cleaned = _clean(terms if isinstance(terms, list) else [],
                             desc_of[code])
            if cleaned:
                out[code] = cleaned
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True, choices=list(SYSTEMS))
    ap.add_argument("--prefix")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--codes")
    ap.add_argument("--workers", type=int, default=_WORKERS,
                    help="parallel LLM calls (higher = faster, but watch API "
                         "rate limits; failed batches are skipped, re-runnable "
                         "with --resume)")
    ap.add_argument("--resume", action="store_true",
                    help="skip codes already present in the output file")
    args = ap.parse_args()
    cfg = SYSTEMS[args.system]
    out_file = DATA_DIR / "codes" / cfg["out"]
    sysprompt = _sys_prompt(cfg)

    # Existing synonyms are loaded up front so we can (a) --resume past them
    # and (b) write INCREMENTALLY without clobbering codes outside this scope.
    existing = {}
    if out_file.exists():
        try:
            existing = json.load(open(out_file)).get("terms", {})
        except Exception:
            existing = {}

    codes = _load(cfg)
    if args.prefix:
        codes = [c for c in codes if c[0].startswith(args.prefix)]
    if args.codes:
        want = {c.strip() for c in args.codes.split(",")}
        codes = [c for c in codes if c[0] in want]
    if args.limit:
        codes = codes[:args.limit]
    if args.resume:
        before = len(codes)
        codes = [c for c in codes if c[0] not in existing]
        logger.info(f"--resume: {before - len(codes)} codes already have "
                    f"synonyms, {len(codes)} remaining")
    if not codes:
        logger.info(f"no {cfg['label']} codes left to generate")
        return 0
    logger.info(f"Generating {cfg['label']} synonyms for {len(codes)} code(s) "
                f"in batches of {_BATCH} ({args.workers} workers)")

    batches = [codes[i:i + _BATCH] for i in range(0, len(codes), _BATCH)]
    scope = {c for c, _ in codes}
    # merged always = (existing OUTSIDE this scope) + (this run's terms), so a
    # partial file is valid and re-runnable at any moment.
    kept = {k: v for k, v in existing.items() if k not in scope}
    terms: dict[str, list[str]] = {}

    def _save():
        merged = {**kept, **terms}
        payload = {
            "provenance": "llm-generated, grounded against the code's own "
            "descriptor with sibling disambiguation; RETRIEVAL AID ONLY — NOT "
            "an authoritative source and never a coding-decision input",
            "code_system": cfg["label"],
            "generated": _dt.date.today().isoformat(),
            "count": len(merged),
            "terms": dict(sorted(merged.items())),
        }
        tmp = out_file.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=1))
        tmp.replace(out_file)   # atomic — a crash never leaves a half file
        return len(merged)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_gen_batch, b, sysprompt): b for b in batches}
        done = 0
        for fut in as_completed(futs):
            terms.update(fut.result())
            done += 1
            if done % 25 == 0:              # checkpoint to disk
                total = _save()
                logger.info(f"  {done}/{len(batches)} batches, {len(terms)} "
                            f"this run ({total} total) — checkpointed")

    merged_count = _save()
    logger.info(f"Wrote {merged_count} {cfg['label']} codes with synonyms -> "
                f"{out_file} ({len(terms)} refreshed this run)")
    for c in list(terms)[:3]:
        logger.info(f"  e.g. {c}: {terms[c]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
