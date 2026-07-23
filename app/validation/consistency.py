"""Self-consistency: run a note N times and flag every code that is not
unanimous across runs.

The LLM passes run at near-zero temperature but are not perfectly
deterministic (observed live: a MEDICAL_NECESSITY FAIL present in one run of
note 031 and absent in the next). Disagreement between independent runs is a
cheap, model-free signal that a code decision sits on a knife edge — exactly
the lines a human should look at. This module is pure comparison and
annotation: no medical codes, no thresholds beyond unanimity.

Identity compared per array:
    icd_codes / supporting_conditions : (code, type)
    cpt_codes / hcpcs_codes           : (code, modifier set, units)
    snomed_codes                      : (concept_id,)

A code present in every run but with differing attributes (modifiers, units,
primary/secondary type) counts as a disagreement just like a code missing
from some runs. Disposition disagreement (CLEAN/REVIEW/REJECT) is reported
separately.
"""

from __future__ import annotations

from typing import Any

_ARRAYS: dict[str, tuple[str, ...]] = {
    "icd_codes": ("type",),
    "supporting_conditions": ("type",),
    "cpt_codes": ("modifiers", "units"),
    "hcpcs_codes": ("modifiers", "units"),
    "snomed_codes": (),
}

# Arrays that never appear on a CMS-1500 — informational context only.
# SNOMED: observed live dominating run-to-run variance (21 of 25
# disagreements on the first measured note), routing EVERY note to REVIEW
# and drowning the billable signal. supporting_conditions: the validator
# itself treats them as advisory ("not validated as billable codes") and
# they hold non-billed context (Z-codes documenting med lists, M81.0
# alongside a fracture, etc.) — measured at 19 of 135 billing-gate
# disagreements across the first 28 consistency notes despite never
# reaching a claim line. Their disagreements are still recorded in the
# report, but only billing-relevant arrays gate unanimity/routing.
_ADVISORY_ARRAYS = {"snomed_codes", "supporting_conditions"}

# ICD-10-CM Chapter 20 (External causes of morbidity) spans V00-Y99 — a
# structural chapter boundary of the classification itself (same boundary
# mce.py uses). Per the ICD-10-CM Official Guidelines (I.C.20), "there is no
# national requirement for mandatory ICD-10-CM external cause code
# reporting" — these codes are supplementary context (activity, place,
# mechanism), never medical-necessity or payment drivers. A run-to-run
# PRESENCE flip on an optional supplementary code is therefore advisory:
# recorded, but not a reason to pull the note from auto-submission.
_EXTERNAL_CAUSE_FIRST_CHARS = ("V", "W", "X", "Y")


def _is_advisory(array: str, code: str) -> bool:
    if array in _ADVISORY_ARRAYS:
        return True
    return array == "icd_codes" and code[:1] in _EXTERNAL_CAUSE_FIRST_CHARS


def _norm_code(code: str) -> str:
    return "".join(ch for ch in str(code).upper() if ch.isalnum())


def _icd_flip_is_claim_inert(code: str, runs: list[dict], store) -> bool:
    """True when a run-to-run PRESENCE flip on a secondary ICD cannot change
    what the claim pays or whether it passes coverage — measured live as the
    second-largest disagreement class (marginal secondaries like I73.9 or
    Q84.5 present in 1 of 3 runs). Everything here is a claim/data fact:

      billing-gating (returns False) when, in ANY run, the code is
        - the primary diagnosis, or
        - pointed to by a service line (linked_diagnoses), or
        - named by an instructional note (useAdditionalCode/codeFirst/
          codeAlso) of another diagnosis on the claim — its presence is
          mandated by the Tabular List, so a flip is a real coding error, or
        - on a coverage policy's diagnosis list for any claim CPT — its
          presence can decide medical necessity.

    Anything else is documentation-context variance: recorded in the
    report, but not a reason to pull the note from auto-submission."""
    if store is None:
        return False
    target = _norm_code(code)

    other_icds: set[str] = set()
    claim_cpts: set[str] = set()
    for run in runs:
        for e in run.get("icd_codes") or []:
            if not isinstance(e, dict):
                continue
            c = _norm_code(e.get("code", ""))
            if c == target:
                if str(e.get("type", "")).strip().lower() == "primary":
                    return False
            elif c:
                other_icds.add(c)
        for array in ("cpt_codes", "hcpcs_codes"):
            for e in run.get(array) or []:
                if not isinstance(e, dict):
                    continue
                if any(_norm_code(d) == target
                       for d in (e.get("linked_diagnoses") or [])):
                    return False
                c = _norm_code(e.get("code", ""))
                if c:
                    claim_cpts.add(c)

    try:
        for other in other_icds:
            for groups_of in (store.use_additional_code_groups,
                              store.code_first_groups,
                              store.code_also_groups):
                for _carrier, refs in groups_of(other):
                    ref_norms = [_norm_code(ref) for ref, _note in refs]
                    if not any(r and target.startswith(r) for r in ref_norms):
                        continue
                    # The instruction names the target — but a note's refs
                    # are ALTERNATIVES (the validator's own satisfaction
                    # rule): when every run satisfies the group with some
                    # OTHER present code, the flipping code is a redundant
                    # alternative, not a mandated companion. Measured live
                    # (note 004): Z79.84 flipped 2/3 under an E11.x carrier
                    # whose use-additional group Z79.4 satisfied in every
                    # run — the Tabular instruction is met either way, so
                    # the flip cannot make the claim non-compliant.
                    for run in runs:
                        present = {
                            _norm_code(e.get("code", ""))
                            for e in run.get("icd_codes") or []
                            if isinstance(e, dict) and e.get("code")
                        } - {target}
                        if not any(
                                p.startswith(r) or r.startswith(p)
                                for r in ref_norms if r
                                for p in present):
                            return False
        for cpt in claim_cpts:
            for policy_id in store.coverage_policies_for_cpt(cpt):
                if not (store.coverage_policy_has_dx_rules(policy_id)
                        and store.coverage_icd_covered(policy_id, code)):
                    continue
                # The policy names the target — but coverage needs ONE
                # covered diagnosis, not all of them (the medical-necessity
                # agent's own satisfaction rule). When every run satisfies
                # this policy with some OTHER present code, the flipping
                # code is a redundant coverage alternative and cannot
                # change the necessity verdict. Measured live (note 003):
                # L60.2 flipped 1/3 on a nail-avulsion claim whose B35.1
                # covers the policy in every run. Mirrors the
                # instructional-group satisfaction arm above.
                for run in runs:
                    present = {
                        str(e.get("code", "")).strip().upper()
                        for e in run.get("icd_codes") or []
                        if isinstance(e, dict) and e.get("code")
                    } - {code.strip().upper()}
                    if not any(store.coverage_icd_covered(policy_id, p)
                               for p in present):
                        return False
    except Exception:
        return False  # any lookup failure stays conservative: billing-gating
    return True


def _merge_em_level_flips(disagreements: list[dict], runs: list[dict],
                          store) -> list[dict]:
    """Collapse paired E/M sibling presence flips (99213 in one run, 99214
    in another) into ONE kind='em_level' disagreement carrying the per-run
    code and MDM axis scores. Two separate 'presence' rows hid what actually
    happened — a single visit whose LEVEL sat on a knife edge — and the MDM
    axes are the evidence a reviewer (and the next deterministic layer)
    needs to see. Family membership comes from the codes' own AMA
    descriptors (store.em_family_prefix), never a code table. Only fires
    when every run carries exactly one member of the family; anything
    messier stays as plain presence flips."""
    if store is None:
        return disagreements
    flips = [d for d in disagreements
             if d["array"] == "cpt_codes" and d["kind"] == "presence"]
    by_family: dict[str, list[dict]] = {}
    for d in flips:
        try:
            prefix = store.em_family_prefix(d["code"])
        except Exception:
            prefix = None
        if prefix:
            by_family.setdefault(prefix, []).append(d)

    merged = list(disagreements)
    for prefix, members in by_family.items():
        if len(members) < 2:
            continue
        family_codes = {d["code"] for d in members}
        per_run = []
        clean = True
        for run in runs:
            present = []
            for e in run.get("cpt_codes") or []:
                if isinstance(e, dict) and _code_of(e, "cpt_codes") in family_codes:
                    mdm = e.get("mdm_details") or {}
                    present.append({
                        "code": _code_of(e, "cpt_codes"),
                        "mdm_level": mdm.get("mdm_level"),
                        "problems_score": mdm.get("problems_score"),
                        "data_score": mdm.get("data_score"),
                        "risk_score": mdm.get("risk_score"),
                    })
            if len(present) != 1:
                clean = False
                break
            per_run.append(present[0])
        if not clean:
            continue
        merged = [d for d in merged if d not in members]
        merged.append({
            "array": "cpt_codes",
            "code": "/".join(sorted(family_codes)),
            "kind": "em_level", "advisory": False,
            "codes": sorted(family_codes),
            "by_run": per_run, "runs": len(runs),
        })
    return merged


def _code_of(entry: dict, array: str) -> str:
    field = "concept_id" if array == "snomed_codes" else "code"
    return str(entry.get(field, "")).strip().upper()


def _attrs_of(entry: dict, array: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for f in _ARRAYS[array]:
        v = entry.get(f)
        if f == "modifiers":
            out[f] = sorted(str(m).upper() for m in (v or []))
        elif f == "units":
            # normalize numeric type so 1 vs 1.0 vs "1" never reads as a
            # disagreement between runs
            try:
                out[f] = int(float(v)) if v is not None else 1
            except (TypeError, ValueError):
                out[f] = v
        else:
            out[f] = (str(v).lower() if v is not None else None)
    return out


def compare_runs(runs: list[dict], store=None) -> dict:
    """Compare N result dicts; return a consistency report.

    unanimous=True only when every array's code set AND per-code attributes
    AND the final disposition agree across all runs.

    `store` (optional ComplianceDataStore) enables the data-driven
    refinements — E/M sibling-flip merging and claim-inert secondary-ICD
    classification. Without it the comparison is pure and every billing
    flip gates unanimity (strictly more conservative)."""
    n = len(runs)
    disagreements: list[dict] = []

    for array in _ARRAYS:
        presence: dict[str, list[int]] = {}
        attrs: dict[str, list[dict]] = {}
        for i, run in enumerate(runs):
            for e in run.get(array) or []:
                if not isinstance(e, dict):
                    continue
                code = _code_of(e, array)
                if not code:
                    continue
                if i not in presence.setdefault(code, []):
                    presence[code].append(i)
                    attrs.setdefault(code, []).append(_attrs_of(e, array))
        for code, in_runs in sorted(presence.items()):
            if len(in_runs) < n:
                advisory = _is_advisory(array, code)
                if (not advisory and array == "icd_codes"
                        and _icd_flip_is_claim_inert(code, runs, store)):
                    advisory = True
                disagreements.append({
                    "array": array, "code": code,
                    "kind": "presence", "advisory": advisory,
                    "present_in_runs": len(in_runs), "runs": n,
                })
            elif any(a != attrs[code][0] for a in attrs[code][1:]):
                differing = sorted({f for a in attrs[code]
                                    for f in a if a[f] != attrs[code][0][f]})
                disagreements.append({
                    "array": array, "code": code,
                    "kind": "attributes", "advisory": _is_advisory(array, code),
                    "fields": differing,
                    "values": attrs[code], "runs": n,
                })

    disagreements = _merge_em_level_flips(disagreements, runs, store)

    dispositions = [str(r.get("final_disposition") or "") for r in runs]
    tiers = [str(r.get("auto_coding_tier") or "") for r in runs]
    billing = [d for d in disagreements if not d["advisory"]]
    return {
        "runs": n,
        # unanimity (and therefore REVIEW routing) is judged on the arrays
        # that reach the claim form; advisory-array variance is reported only
        "unanimous": not billing and len(set(dispositions)) <= 1
                     and len(set(tiers)) <= 1,
        "disagreements": disagreements,
        "dispositions": dispositions,
        "tiers": tiers,
    }


def select_canonical(runs: list[dict]) -> int:
    """Index of the run that best agrees with the majority: for each BILLING
    (array, code) present in >= half the runs, a run scores +1 for carrying
    it; the highest-scoring run (first on ties) is canonical. This keeps the
    saved result the most 'typical' of the N, rather than arbitrarily the
    last. Advisory arrays are excluded from scoring — SNOMED lists are large
    (often 2-3x the billing codes) and would otherwise let informational
    agreement outvote the billing content the claim is actually made of."""
    n = len(runs)
    presence: dict[tuple[str, str], set[int]] = {}
    for array in _ARRAYS:
        if array in _ADVISORY_ARRAYS:
            continue
        for i, run in enumerate(runs):
            for e in run.get(array) or []:
                if isinstance(e, dict):
                    code = _code_of(e, array)
                    if code:
                        presence.setdefault((array, code), set()).add(i)
    majority = {k for k, v in presence.items() if len(v) * 2 >= n}
    scores = [sum(1 for k in majority if i in presence[k]) for i in range(n)]
    return max(range(n), key=lambda i: scores[i])


def annotate_result(result: dict, report: dict, route: bool = True) -> dict:
    """Fold the consistency verdict into the canonical result. Disagreeing
    codes get needs_review + review_reason; any disagreement (codes,
    attributes, or disposition) routes the note to REVIEW unless it is
    already REJECT. Pure function of (result, report) — deterministic.

    route=False defers the human-review verdict: the full consistency
    report is embedded (triage and actuation read it) but codes are not
    flagged and tier/disposition stay untouched. Used while an iteration
    loop is still working the note — routing to a human mid-loop would be
    premature, since the next accepted rule may converge it. The loop
    finalizes any remaining holdouts by re-calling this with route=True
    on the saved result (the embedded report makes that a pure replay)."""
    result.setdefault("consistency", report)
    if report["unanimous"]:
        result.pop("review_routing", None)
        return result
    if not route:
        result["review_routing"] = "deferred"
        return result
    result["review_routing"] = "routed"

    by_array: dict[str, dict[str, dict]] = {}
    for d in report["disagreements"]:
        if d.get("advisory"):
            continue  # reported in the embedded report; never gates routing
        if d["kind"] == "em_level":
            # one merged disagreement covers every sibling code — whichever
            # one the canonical run carries must be flagged
            for c in d.get("codes") or []:
                by_array.setdefault(d["array"], {})[c] = d
        else:
            by_array.setdefault(d["array"], {})[d["code"]] = d

    for array, flagged in by_array.items():
        for e in result.get(array) or []:
            if not isinstance(e, dict):
                continue
            code = _code_of(e, array)
            if code in flagged:
                d = flagged[code]
                if d["kind"] == "presence":
                    why = (f"self-consistency: present in only "
                           f"{d['present_in_runs']}/{d['runs']} independent runs")
                elif d["kind"] == "em_level":
                    picks = "/".join(r.get("code") or "?" for r in d.get("by_run") or [])
                    why = (f"self-consistency: E/M level flipped across "
                           f"{d['runs']} independent runs ({picks})")
                else:
                    why = (f"self-consistency: {', '.join(d['fields'])} differ "
                           f"across {d['runs']} independent runs")
                e["needs_review"] = True
                prior = e.get("review_reason")
                e["review_reason"] = f"{prior} | {why}" if prior else why

    reasons = list(result.get("auto_coding_review_reasons") or [])
    summary_bits = []
    billing = [d for d in report["disagreements"] if not d.get("advisory")]
    if billing:
        codes = sorted({d["code"] for d in billing})
        summary_bits.append(f"non-unanimous codes across {report['runs']} runs: "
                            + ", ".join(codes[:10]))
    if len(set(report["dispositions"])) > 1:
        summary_bits.append("disposition varied across runs: "
                            + "/".join(report["dispositions"]))
    reason = "Self-consistency check — " + "; ".join(summary_bits)
    reasons.append(reason)
    result["auto_coding_review_reasons"] = reasons

    if str(result.get("auto_coding_tier") or "").upper() != "REJECT":
        result["auto_coding_tier"] = "REVIEW"
    if str(result.get("final_disposition") or "").upper() == "CLEAN":
        result["final_disposition"] = "REVIEW"
    return result
