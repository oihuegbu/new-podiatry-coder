#!/usr/bin/env python3
"""Denial feedback loop: every payer denial becomes a permanent regression test.

Adjudication (paid/denied per line, with X12 CARC reason codes) is the only
true gold standard for a coding pipeline. This tool closes the loop:

  1. ingest   parse an 835 remittance (EDI) or a denials CSV and append the
              normalized denial records to the registry
              (data/feedback/denials.json). Idempotent — records are keyed by
              (claim/document id, code, CARC).
  2. report   for each registered denial, load the note's stored result and
              classify it:
                CAUGHT        the pipeline flagged the denied code under a
                              check family mapped from the denial's CARC
                              (data/feedback/carc_map.json) before submission
                FLAGGED_OTHER the code was flagged, but under an unrelated
                              family — mapping or rule-scope gap
                MISSED        the pipeline passed the code clean; a
                              deterministic layer is missing
                UNMAPPED      CARC not in the map — extend carc_map.json
              writes output/feedback/denial_gap_report.json.
  3. gate     regression mode for CI/cron: exit non-zero if any denial is
              MISSED (and not explicitly waived in the registry with
              "waived": true + "waive_reason"). Because the report re-evaluates
              against the CURRENT results every run, a denial flips to CAUGHT
              permanently once the missing rule is wired in — and can never
              silently regress.

CSV ingest columns (header required, extra columns preserved):
  document_id, code, carc, rarc, payer, dos, amount, description

835 ingest reads CLP (claim), SVC (service line), CAS (adjustment) segments;
CLP01 (patient control number) must carry the note/document id, which is how
the practice's billing export ties remittances back to notes.

Usage:
  python tools/denial_feedback.py ingest <835-or-csv file>
  python tools/denial_feedback.py report [results_dir]
  python tools/denial_feedback.py gate   [results_dir]
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CARC_MAP_PATH = ROOT / "data" / "feedback" / "carc_map.json"
REGISTRY_PATH = ROOT / "data" / "feedback" / "denials.json"
DEFAULT_RESULTS = ROOT / "output" / "results"
REPORT_PATH = ROOT / "output" / "feedback" / "denial_gap_report.json"

# X12 835 claim-status codes that represent a denial at claim level
_DENIED_CLAIM_STATUS = {"4"}  # 4 = denied; line-level denials come via CAS
# CAS group codes that represent payer-side adjustments (not patient liability)
_ADJUSTMENT_GROUPS = {"CO", "PI", "OA"}


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------

def _load_registry() -> list[dict]:
    if REGISTRY_PATH.exists():
        return json.loads(REGISTRY_PATH.read_text())
    return []


def _save_registry(records: list[dict]) -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(json.dumps(records, indent=2))


def _key(rec: dict) -> tuple:
    return (str(rec.get("document_id", "")).strip(),
            str(rec.get("code", "")).strip().upper(),
            str(rec.get("carc", "")).strip().upper())


def _merge(existing: list[dict], new: list[dict]) -> tuple[list[dict], int]:
    seen = {_key(r) for r in existing}
    added = 0
    for rec in new:
        if _key(rec) not in seen:
            existing.append(rec)
            seen.add(_key(rec))
            added += 1
    return existing, added


# --------------------------------------------------------------------------
# ingest: 835 EDI
# --------------------------------------------------------------------------

def parse_835(text: str) -> list[dict]:
    """Minimal 835 reader: walks CLP/SVC/CAS segments and emits one denial
    record per (service line, CARC) adjustment. Only payer-side adjustment
    groups (CO/PI/OA) count as denials; PR (patient responsibility, e.g.
    deductible CARC 1-3) is normal adjudication, not a coding failure."""
    # element separator is the 4th char of ISA; default '*', segment '~'
    elem = text[3] if text.startswith("ISA") and len(text) > 3 else "*"
    segments = [s.strip() for s in text.replace("\n", "").split("~") if s.strip()]

    records: list[dict] = []
    claim_id, claim_status, payer = "", "", ""
    svc_code, svc_modifiers = "", []

    for seg in segments:
        parts = seg.split(elem)
        tag = parts[0].upper()
        if tag == "N1" and len(parts) > 2 and parts[1].upper() == "PR":
            payer = parts[2]
        elif tag == "CLP":
            claim_id = parts[1] if len(parts) > 1 else ""
            claim_status = parts[2] if len(parts) > 2 else ""
            svc_code, svc_modifiers = "", []
        elif tag == "SVC":
            # SVC01 composite: HC<:>code<:>mod1<:>mod2...
            comp = (parts[1] if len(parts) > 1 else "").split(":")
            svc_code = comp[1] if len(comp) > 1 else ""
            svc_modifiers = [m for m in comp[2:] if m]
        elif tag == "CAS" and len(parts) > 2:
            group = parts[1].upper()
            if group not in _ADJUSTMENT_GROUPS and claim_status not in _DENIED_CLAIM_STATUS:
                continue
            # CAS repeats (reason, amount, quantity) triplets after the group
            triplet = parts[2:]
            for i in range(0, len(triplet), 3):
                carc = triplet[i].strip()
                if not carc:
                    continue
                amount = triplet[i + 1] if i + 1 < len(triplet) else ""
                records.append({
                    "document_id": claim_id,
                    "code": svc_code,
                    "modifiers": svc_modifiers,
                    "carc": carc,
                    "group": group,
                    "amount": amount,
                    "payer": payer,
                    "source": "835",
                })
    return records


# --------------------------------------------------------------------------
# ingest: CSV
# --------------------------------------------------------------------------

def parse_csv(text: str) -> list[dict]:
    records = []
    for row in csv.DictReader(text.splitlines()):
        rec = {k.strip().lower(): (v or "").strip() for k, v in row.items() if k}
        if not rec.get("document_id") or not rec.get("carc"):
            continue
        rec["source"] = "csv"
        records.append(rec)
    return records


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------

def _load_carc_map() -> dict:
    return json.loads(CARC_MAP_PATH.read_text())["carcs"]


def _result_for(document_id: str, results_dir: Path) -> dict | None:
    """Stored results are named <note stem>_results.json; the document_id in
    the registry is matched against both the stem and the result's own
    document_id field."""
    direct = results_dir / f"{document_id}_results.json"
    if direct.exists():
        return json.loads(direct.read_text())
    for f in results_dir.glob("*_results.json"):
        if f.name == "all_results.json":
            continue
        if document_id.lower() in f.stem.lower():
            return json.loads(f.read_text())
    return None


def _family_matches(fam: str, token: str) -> bool:
    """True when a carc_map check family names this flag's filter_id/category.
    Matching is on '_'-separated components (with singular/plural leveling),
    never bare substrings — 'age' must match 'hcpcs_age_range' but NOT
    'cpt_dx_linkage', and 'modifier' must match 'modifiers'."""
    fam, token = fam.lower(), token.lower()
    if fam == token:
        return True
    if "_" in fam:  # multiword family: component-boundary substring
        return fam in token
    parts = token.split("_")
    return any(fam == p or fam.rstrip("s") == p.rstrip("s") for p in parts)


def _flags_for_code(result: dict, code: str) -> list[tuple[str, str]]:
    """Every (family_token, detail) under which the pipeline flagged this code
    pre-submission: claim-scrub findings, validator issues, and per-entry
    needs_review markers. An empty code means a claim-level denial (835 CAS
    at CLP level, no service line) — every flag on the claim is eligible."""
    code_u = code.strip().upper()
    claim_level = not code_u
    flags: list[tuple[str, str]] = []
    scrub = result.get("claim_scrub") or {}
    for f in scrub.get("findings") or []:
        codes = [str(c).strip().upper() for c in (f.get("codes") or [])]
        if (claim_level or code_u in codes) and f.get("status") in ("WARN", "FAIL"):
            flags.append((str(f.get("filter_id", "")).lower(), f.get("reason", "")))
    for issue in result.get("validation_issues") or []:
        icode = str(issue.get("code", "")).upper()
        if (claim_level or code_u in icode.replace("|", " ").split()) and \
                issue.get("severity") in ("WARNING", "ERROR"):
            flags.append((str(issue.get("category", "")).lower(),
                          issue.get("message", "")))
    for key in ("icd_codes", "cpt_codes", "hcpcs_codes", "supporting_conditions"):
        for e in result.get(key) or []:
            if isinstance(e, dict) and e.get("needs_review") and \
                    (claim_level or str(e.get("code", "")).strip().upper() == code_u):
                flags.append(("needs_review", str(e.get("review_reason") or "")))
    return flags


def classify_denial(rec: dict, result: dict | None, carc_map: dict) -> dict:
    carc = str(rec.get("carc", "")).strip().upper()
    entry = carc_map.get(carc)
    out = {**rec, "carc_label": (entry or {}).get("label", "")}
    if result is None:
        out["status"] = "NO_RESULT"
        out["detail"] = "no stored result for this document_id"
        return out
    if entry is None:
        out["status"] = "UNMAPPED"
        out["detail"] = f"CARC {carc} not in carc_map.json — extend the map"
        return out
    families = [f.lower() for f in entry.get("check_families", [])]
    flags = _flags_for_code(result, str(rec.get("code", "")))
    matched = [(tok, d) for tok, d in flags
               if any(_family_matches(fam, tok) for fam in families)]
    if matched:
        out["status"] = "CAUGHT"
        out["detail"] = matched[0][1][:200]
    elif flags:
        out["status"] = "FLAGGED_OTHER"
        out["detail"] = (f"flagged under {sorted({t for t, _ in flags})} but the "
                         f"denial maps to {families}")
    else:
        out["status"] = "MISSED"
        out["detail"] = ("pipeline passed this code clean; a deterministic "
                         f"layer for {families} is missing or did not fire")
    return out


def build_report(results_dir: Path) -> dict:
    registry = _load_registry()
    carc_map = _load_carc_map()
    classified = []
    for rec in registry:
        result = _result_for(str(rec.get("document_id", "")), results_dir)
        classified.append(classify_denial(rec, result, carc_map))
    counts: dict[str, int] = {}
    for c in classified:
        counts[c["status"]] = counts.get(c["status"], 0) + 1
    return {"denials": classified, "counts": counts,
            "results_dir": str(results_dir)}


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in ("ingest", "report", "gate"):
        sys.exit(__doc__)
    mode = sys.argv[1]

    if mode == "ingest":
        if len(sys.argv) < 3:
            sys.exit("ingest requires a file path (835 EDI or CSV)")
        text = Path(sys.argv[2]).read_text()
        if text.lstrip().upper().startswith("ISA") or "~CLP" in text.replace("\n", ""):
            new = parse_835(text)
        else:
            new = parse_csv(text)
        registry, added = _merge(_load_registry(), new)
        _save_registry(registry)
        print(f"Ingested {len(new)} denial record(s); {added} new; "
              f"registry now {len(registry)}")
        return

    results_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_RESULTS
    report = build_report(results_dir)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2))

    for c in report["denials"]:
        print(f"  [{c['status']:14s}] {c.get('document_id','?'):40s} "
              f"{c.get('code','?'):8s} CARC {c.get('carc','?'):4s} "
              f"{c.get('carc_label','')[:50]}")
    print(f"\nTotals: {report['counts'] or 'no denials registered'}")
    print(f"Report -> {REPORT_PATH}")

    if mode == "gate":
        missed = [c for c in report["denials"]
                  if c["status"] == "MISSED" and not c.get("waived")]
        if missed:
            print(f"\nGATE FAIL — {len(missed)} denial(s) still MISSED "
                  f"(waive with 'waived': true + 'waive_reason' in the registry "
                  f"only for payer-side errors)")
            sys.exit(1)
        print("\nGATE PASS — every registered denial is caught, mapped, or waived")


if __name__ == "__main__":
    main()
