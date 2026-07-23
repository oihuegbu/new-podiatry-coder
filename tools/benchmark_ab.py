"""Gold-standard benchmark: freeze verified outputs, then A/B any model or
process change against them — measurement instead of guesswork.

The 56 reviewed notes' verified results are the gold set. Any candidate run
(new model, new prompt, new pass structure) is scored per note on the fields
that decide accuracy and billability:

    ICD  : code set F1, primary-diagnosis match
    CPT  : code set F1, per-line modifier-set match, units match
    HCPCS: code set F1, modifier-set match
    disposition: CLEAN/REVIEW/REJECT agreement

Modes:
  freeze   copy the current results into the gold directory (one-time per
           accepted baseline; re-freeze only after a human-verified upgrade)
  score    score one results dir against gold
  compare  score BASELINE and CANDIDATE dirs against gold; the change is
           ACCEPTED only if the candidate is >= baseline on every aggregate
           metric and > on at least one (strict improvement).

Usage:
  python tools/benchmark_ab.py freeze  [results_dir] [gold_dir]
  python tools/benchmark_ab.py score   <results_dir> [gold_dir]
  python tools/benchmark_ab.py compare <baseline_dir> <candidate_dir> [gold_dir]
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

DEFAULT_RESULTS = Path("output/results")
DEFAULT_GOLD = Path("benchmark/gold")

METRICS = ("icd_f1", "primary_match", "cpt_f1", "cpt_modifiers",
           "cpt_units", "hcpcs_f1", "hcpcs_modifiers", "disposition")


def _entries(d: dict, key: str) -> dict[str, dict]:
    out = {}
    for e in d.get(key) or []:
        if isinstance(e, dict) and e.get("code"):
            out[str(e["code"]).strip().upper()] = e
    return out


def _f1(gold: set, cand: set) -> float:
    if not gold and not cand:
        return 1.0
    if not gold or not cand:
        return 0.0
    tp = len(gold & cand)
    prec, rec = tp / len(cand), tp / len(gold)
    return 2 * prec * rec / (prec + rec) if prec + rec else 0.0


def _line_agreement(gold: dict[str, dict], cand: dict[str, dict], field: str) -> float:
    """Among codes present in BOTH sets, the fraction whose field agrees.
    Modifier comparison is set-based; units numeric."""
    shared = set(gold) & set(cand)
    if not shared:
        return 1.0
    ok = 0
    for c in shared:
        g, k = gold[c], cand[c]
        if field == "modifiers":
            if set(map(str, g.get("modifiers") or [])) == set(map(str, k.get("modifiers") or [])):
                ok += 1
        elif field == "units":
            if (g.get("units") or 1) == (k.get("units") or 1):
                ok += 1
    return ok / len(shared)


def score_note(gold: dict, cand: dict) -> dict[str, float]:
    g_icd, c_icd = _entries(gold, "icd_codes"), _entries(cand, "icd_codes")
    g_cpt, c_cpt = _entries(gold, "cpt_codes"), _entries(cand, "cpt_codes")
    g_hc, c_hc = _entries(gold, "hcpcs_codes"), _entries(cand, "hcpcs_codes")

    def _primary(entries):
        for code, e in entries.items():
            if str(e.get("type", "")).lower() == "primary":
                return code
        return None

    return {
        "icd_f1": _f1(set(g_icd), set(c_icd)),
        "primary_match": float(_primary(g_icd) == _primary(c_icd)),
        "cpt_f1": _f1(set(g_cpt), set(c_cpt)),
        "cpt_modifiers": _line_agreement(g_cpt, c_cpt, "modifiers"),
        "cpt_units": _line_agreement(g_cpt, c_cpt, "units"),
        "hcpcs_f1": _f1(set(g_hc), set(c_hc)),
        "hcpcs_modifiers": _line_agreement(g_hc, c_hc, "modifiers"),
        "disposition": float((gold.get("final_disposition") or "")
                             == (cand.get("final_disposition") or "")),
    }


def score_dir(cand_dir: Path, gold_dir: Path, verbose: bool = True):
    gold_files = {f.name: f for f in gold_dir.glob("*_results.json")
                  if f.name != "all_results.json"}
    if not gold_files:
        sys.exit(f"No gold results in {gold_dir} — run 'freeze' first.")
    totals = {m: 0.0 for m in METRICS}
    n = 0
    missing = []
    for name, gf in sorted(gold_files.items()):
        cf = cand_dir / name
        if not cf.exists():
            missing.append(name)
            continue
        s = score_note(json.loads(gf.read_text()), json.loads(cf.read_text()))
        n += 1
        imperfect = {m: v for m, v in s.items() if v < 1.0}
        if verbose and imperfect:
            print(f"  {name}: " + ", ".join(f"{m}={v:.2f}" for m, v in imperfect.items()))
        for m in METRICS:
            totals[m] += s[m]
    if missing and verbose:
        print(f"  MISSING from candidate ({len(missing)}): {', '.join(missing[:6])}"
              + (" ..." if len(missing) > 6 else ""))
    agg = {m: (totals[m] / n if n else 0.0) for m in METRICS}
    agg["notes_scored"] = n
    agg["notes_missing"] = len(missing)
    return agg


def _print_agg(label: str, agg: dict):
    print(f"\n{label}  ({agg['notes_scored']} notes scored, "
          f"{agg['notes_missing']} missing)")
    for m in METRICS:
        print(f"  {m:16s} {agg[m]:.4f}")


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("freeze", "score", "compare"):
        sys.exit(__doc__)
    mode = sys.argv[1]

    if mode == "freeze":
        src = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_RESULTS
        dst = Path(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_GOLD
        dst.mkdir(parents=True, exist_ok=True)
        n = 0
        for f in sorted(src.glob("*_results.json")):
            if f.name == "all_results.json":
                continue
            shutil.copy2(f, dst / f.name)
            n += 1
        print(f"Froze {n} verified results: {src} -> {dst}")
        return

    if mode == "score":
        cand = Path(sys.argv[2])
        gold = Path(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_GOLD
        _print_agg(f"SCORE {cand}", score_dir(cand, gold))
        return

    # compare
    base_dir, cand_dir = Path(sys.argv[2]), Path(sys.argv[3])
    gold = Path(sys.argv[4]) if len(sys.argv) > 4 else DEFAULT_GOLD
    print("== baseline note-level misses ==")
    base = score_dir(base_dir, gold)
    print("\n== candidate note-level misses ==")
    cand = score_dir(cand_dir, gold)
    _print_agg(f"BASELINE {base_dir}", base)
    _print_agg(f"CANDIDATE {cand_dir}", cand)

    regressed = [m for m in METRICS if cand[m] < base[m] - 1e-9]
    improved = [m for m in METRICS if cand[m] > base[m] + 1e-9]
    print("\n==== VERDICT ====")
    if regressed:
        print(f"REJECT — candidate regresses on: {', '.join(regressed)}")
        sys.exit(1)
    if improved:
        print(f"ACCEPT — strict improvement on: {', '.join(improved)}; no regressions")
    else:
        print("NEUTRAL — no metric changed; keep the cheaper/simpler variant")


if __name__ == "__main__":
    main()
