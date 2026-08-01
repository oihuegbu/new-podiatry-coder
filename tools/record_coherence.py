"""Record coherence — the whole-record contradiction gate (zero LLM).

Why this exists: an outside reviewer reading a single saved result end to
end caught defects the pipeline's own review missed, because the
contradictions were BETWEEN fields of the record — an adjudication block
that says modifiers=[] next to a claim line carrying RT; three REVIEW run
dispositions next to a CLEAN final disposition. The in-pipeline reviewer
was shown a curated case file that excluded exactly those fields. Reading
a record for self-agreement is mechanical, so it belongs in code: this
module enumerates the cross-field pairs of a result that must agree and
returns one named violation per disagreement.

Enforcement points (all fail closed):
  - clinical_auditor._enforce_verdict — an upheld review cannot promote a
    self-contradicting record to CLEAN;
  - claims_registry.eligible_for_auto — a self-contradicting record is
    never auto-recorded as a verified claim;
  - the CLI sweep below — retroactively holds any saved CLEAN record that
    contradicts itself (report-only mode available for concurrent-safe
    inspection).

Every check compares the record's own fields to each other. No medical
knowledge, no code lists, no LLM.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger  # noqa: E402

_BILLING_ARRAYS = ("icd_codes", "cpt_codes", "hcpcs_codes")
COHERENCE_MARKER = "[record_coherence]"


def _codes(result: dict, arr: str) -> set[str]:
    return {str(e.get("code") or "").upper()
            for e in (result.get(arr) or []) if isinstance(e, dict)}


def _is_clean(result: dict) -> bool:
    return str(result.get("final_disposition", "")).upper() == "CLEAN"


def coherence_violations(result: dict,
                         require_audit_release: bool = True) -> list[str]:
    """Every way this record disagrees with itself, one message per
    violation. `require_audit_release=False` skips the released-by-review
    check for callers evaluating a record DURING the review's own
    promotion (the upheld block is being granted at that moment)."""
    v: list[str] = []
    if not isinstance(result, dict):
        return v
    clean = _is_clean(result)

    # 1. Disposition vs scrub verdict: CLEAN must be backed by a CLEAN
    # scrub. (REVIEW with a CLEAN scrub is a legitimate hold.)
    scrub = result.get("claim_scrub") or {}
    if clean and scrub and not (
            bool(scrub.get("clean"))
            or str(scrub.get("disposition", "")).upper() == "CLEAN"):
        v.append(f"final_disposition is CLEAN but the claim scrub's own "
                 f"disposition is {scrub.get('disposition')!r}")

    # 2. Disposition vs consistency: a non-unanimous note is never CLEAN.
    cons = result.get("consistency") or {}
    if clean and cons and int(cons.get("runs") or 0) >= 2 \
            and not cons.get("unanimous"):
        v.append("final_disposition is CLEAN but the consistency runs "
                 "disagree (unanimous=false)")

    # 3. Disposition vs routing: a note routed to human review stays there.
    if clean and result.get("review_routing") == "routed":
        v.append("final_disposition is CLEAN but review_routing says the "
                 "note was routed to human review")

    # 4. Disposition vs review release: under the universal CLEAN gate,
    # only an upheld clinical review releases a claim — a CLEAN record
    # must carry that upheld block. (Skipped when the audit stage is
    # disabled by configuration.)
    if require_audit_release and clean \
            and os.getenv("CLINICAL_AUDIT", "1") == "1":
        verdict = (result.get("clinical_audit") or {}).get("verdict")
        if verdict != "upheld":
            v.append(f"final_disposition is CLEAN but the clinical review "
                     f"verdict is {verdict!r} — nothing released the claim")

    # 5. Adjudication vs claim: every adjudicated decision must be
    # realized on the final claim (the survival invariant, re-checked
    # here so a record is judged coherent by ONE call).
    try:
        from tools.coder_adjudicator import survival_conflicts_of
        for c in survival_conflicts_of(result):
            v.append(f"adjudication decided {c.get('decision')} on "
                     f"{c.get('array')}/{c.get('code')} but the final "
                     f"claim shows {c.get('observed')}")
    except Exception as exc:  # never let the checker itself hide a record
        v.append(f"adjudication survival could not be verified ({exc})")

    # 6. Correction ledger vs claim state: a correction that says a code
    # was removed (added) must agree with the code's absence (presence).
    # The EFFECTIVE correction per code is the last fresh-pass one; a
    # prior-pass entry (carried_from_prior_pass) only counts when no
    # fresh correction touched the code — a later replay may legitimately
    # re-decide it. Only unambiguous presence verbs are judged; swaps and
    # attribute edits have their own checks above.
    _REMOVALS = {"removal", "derived_removal"}
    _ADDITIONS = {"auto_addition", "derived_addition"}
    on_claim = set().union(*(_codes(result, a) for a in _BILLING_ARRAYS))
    effective: dict[str, dict] = {}
    for m in (result.get("material_corrections") or []):
        if not isinstance(m, dict):
            continue
        code = str(m.get("code") or "").upper()
        if not code or m.get("action") not in (_REMOVALS | _ADDITIONS):
            continue
        cur = effective.get(code)
        if cur is not None and cur.get("carried_from_prior_pass") is None \
                and m.get("carried_from_prior_pass"):
            continue  # fresh correction already holds; older pass yields
        effective[code] = m
    for code, m in sorted(effective.items()):
        if m.get("action") in _REMOVALS and code in on_claim:
            v.append(f"a correction reports {code} removed from the claim "
                     f"but the code is still billed "
                     f"({str(m.get('message') or '')[:120]})")
        elif m.get("action") in _ADDITIONS and code not in on_claim:
            v.append(f"a correction reports {code} added to the claim but "
                     f"the code is not billed "
                     f"({str(m.get('message') or '')[:120]})")

    # 7. Linkage referential integrity: a procedure's linked diagnoses
    # must exist on the claim's diagnosis array.
    icd_on_claim = _codes(result, "icd_codes")
    for arr in ("cpt_codes", "hcpcs_codes"):
        for e in (result.get(arr) or []):
            if not isinstance(e, dict):
                continue
            dangling = [str(d).upper() for d in
                        (e.get("linked_diagnoses") or [])
                        if str(d).upper() not in icd_on_claim]
            if dangling:
                v.append(f"{arr}/{e.get('code')} links diagnoses "
                         f"{dangling} that are not on the claim's "
                         f"diagnosis list")

    # 8. Exactly one first-listed diagnosis on a billed claim.
    icds = [e for e in (result.get("icd_codes") or []) if isinstance(e, dict)]
    primaries = [str(e.get("code") or "").upper() for e in icds
                 if str(e.get("type") or "").lower() == "primary"]
    if icds and len(primaries) != 1:
        v.append(f"claim bills {len(icds)} diagnosis code(s) with "
                 f"{len(primaries)} marked primary "
                 f"({primaries or 'none'}) — exactly one first-listed "
                 f"diagnosis is required")

    # 9. Completeness invariant vs claim: the validator's completeness check
    # (CodingValidator._check_procedure_completeness) flags every documented
    # procedure that the final claim neither bills nor records as
    # integral/bundled/not-separately-billable. A record carrying that flag
    # describes documented surgical work missing from the claim — the exact
    # silent-drop failure (a primary procedure vanishing when its wrong
    # sibling code is struck and never substituted) that no code-by-code
    # scrubber filter can see, because the scrubber only reasons about codes
    # that ARE present. Enforced HERE, unconditionally, so it blocks the
    # clinical-audit CLEAN promotion (which evaluates the record while it is
    # still held at REVIEW) — the same reason checks 5–8 do not gate on
    # `clean`. Read off the already-emitted issue; not recomputed.
    for i in (result.get("validation_issues") or []):
        cat = i.get("category") if isinstance(i, dict) else getattr(
            i, "category", "")
        if cat == "documented_work_unaccounted":
            msg = (i.get("message") if isinstance(i, dict)
                   else getattr(i, "message", "")) or ""
            v.append(f"documented procedure not accounted for on the claim "
                     f"— documented surgical work may be uncoded "
                     f"({str(msg)[:140]})")
    return v


def enforce_coherence(result: dict,
                      require_audit_release: bool = True) -> list[str]:
    """Run the gate on a result and, on any violation, hold it at REVIEW
    with each contradiction named. Returns the violations (empty = the
    record agrees with itself; nothing was changed)."""
    violations = coherence_violations(
        result, require_audit_release=require_audit_release)
    if not violations:
        return []
    reasons = [r for r in (result.get("auto_coding_review_reasons") or [])
               if COHERENCE_MARKER not in str(r)]
    result["auto_coding_review_reasons"] = reasons + [
        f"{COHERENCE_MARKER} {viol}" for viol in violations]
    if _is_clean(result):
        result["final_disposition"] = "REVIEW"
        result["auto_coding_tier"] = "REVIEW"
        result["auto_coding_confidence"] = min(
            float(result.get("auto_coding_confidence") or 0.0), 0.84)
    return violations


def sweep(results_dir: Path, docs: list[str] | None = None,
          report_only: bool = False) -> dict:
    """Judge every saved result for self-agreement. In enforce mode a
    violating CLEAN record is rewritten held at REVIEW; report-only mode
    changes nothing (safe to run beside a live loop)."""
    stats = {"checked": 0, "incoherent": 0, "held": 0, "docs": {}}
    for f in sorted(results_dir.glob("*_results.json")):
        if f.name == "all_results.json":
            continue
        doc = f.stem.removesuffix("_results")
        if docs is not None and doc not in docs:
            continue
        try:
            result = json.loads(f.read_text())
        except Exception:
            continue
        if not isinstance(result, dict):
            continue
        stats["checked"] += 1
        violations = coherence_violations(result)
        if not violations:
            continue
        stats["incoherent"] += 1
        stats["docs"][doc] = violations
        if report_only:
            continue
        was_clean = _is_clean(result)
        enforce_coherence(result)
        f.write_text(json.dumps(result, indent=2, default=str))
        if was_clean:
            stats["held"] += 1
        logger.warning(f"Coherence {doc}: {len(violations)} "
                       f"contradiction(s)"
                       + (" — held at REVIEW" if was_clean else ""))
    return stats


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", default="output/results")
    p.add_argument("--docs", default="",
                   help="comma-separated document ids (default: all)")
    p.add_argument("--report-only", action="store_true",
                   help="list contradictions without rewriting any record")
    args = p.parse_args()
    docs = [s.strip() for s in args.docs.split(",") if s.strip()] or None
    stats = sweep(Path(args.results_dir), docs=docs,
                  report_only=args.report_only)
    print(json.dumps(stats, indent=2, default=str))


if __name__ == "__main__":
    main()
