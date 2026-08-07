"""Stage 4 — Positive release gates (fail-CLOSED).

Every gate is a POSITIVE assertion about the claim. Release is earned only when
each gate returns PASS or a proven NOT_APPLICABLE. UNKNOWN, ERROR and BLOCKED
all stop autonomy — a check that could not run is never treated as success.
This is the opposite of "clean = no failure findings": here, silence is not
consent.

Every gate answers from the authoritative source (activity window, NCCI, MUE) or
from the note itself (evidence), never from a code list baked into this file.
"""
from __future__ import annotations

import re

from .data_access import AUTHORITY_UNAVAILABLE, CodeSource
from .models import (CodingResult, FactKind, GateResult, Outcome, RelationPredicate,
                     RelationState, ResolvedLine)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).lower()).strip()


def _worst(outcomes: list[Outcome]) -> Outcome:
    """Fail-closed precedence: any BLOCKED/ERROR dominates, then UNKNOWN, then
    NOT_APPLICABLE if that's all there was, else PASS."""
    for bad in (Outcome.ERROR, Outcome.BLOCKED, Outcome.UNKNOWN):
        if bad in outcomes:
            return bad
    if outcomes and all(o is Outcome.NOT_APPLICABLE for o in outcomes):
        return Outcome.NOT_APPLICABLE
    return Outcome.PASS


def dos_gate(result: CodingResult) -> GateResult:
    if not result.date_of_service:
        return GateResult("date_of_service", Outcome.BLOCKED,
                          "no date of service — every date-sensitive check is unverifiable",
                          "input contract")
    return GateResult("date_of_service", Outcome.PASS,
                      f"DOS = {result.date_of_service}", "input contract")


def evidence_gate(result: CodingResult, note_text: str) -> GateResult:
    note = _norm(note_text)
    outcomes: list[Outcome] = []
    misses: list[str] = []
    for ln in result.billable_lines:
        spans = ln.fact.evidence
        present = any(_norm(s.text) and _norm(s.text) in note for s in spans)
        outcomes.append(Outcome.PASS if present else Outcome.BLOCKED)
        if not present:
            misses.append(ln.chosen.code if ln.chosen else ln.fact.description)
    if not outcomes:
        return GateResult("verbatim_evidence", Outcome.NOT_APPLICABLE,
                          "no billable lines", "evidence")
    return GateResult("verbatim_evidence", _worst(outcomes),
                      "all lines have verbatim note support" if not misses
                      else f"unsupported: {misses}", "note text")


def code_active_gate(result: CodingResult, source: CodeSource) -> GateResult:
    outcomes, detail = [], []
    for ln in result.billable_lines:
        o = source.active_on(ln.chosen.code, ln.chosen.system, result.date_of_service)
        outcomes.append(o)
        if o is not Outcome.PASS:
            detail.append(f"{ln.chosen.code}:{o.value}")
    if not outcomes:
        return GateResult("code_active_on_dos", Outcome.NOT_APPLICABLE,
                          "no billable lines", "authoritative source")
    return GateResult("code_active_on_dos", _worst(outcomes),
                      "all codes active for DOS" if not detail else ", ".join(detail),
                      "authoritative source")


def ncci_gate(result: CodingResult, source: CodeSource) -> GateResult:
    lines = [ln for ln in result.billable_lines
             if ln.chosen and ln.chosen.system in ("cpt", "hcpcs")]
    outcomes, detail, retryable = [], [], False
    for i in range(len(lines)):
        for j in range(len(lines)):
            if i == j:
                continue
            ind = source.ncci_indicator(lines[i].chosen.code, lines[j].chosen.code,
                                        result.date_of_service)
            if ind == AUTHORITY_UNAVAILABLE:   # the check could not run -> not clean
                outcomes.append(Outcome.UNKNOWN)
                retryable = True               # operational failure -> SYSTEM_HOLD, not review
                detail.append(f"{lines[i].chosen.code}/{lines[j].chosen.code} "
                              f"NCCI check unavailable")
                continue
            if ind is None:
                continue                       # ran; this pair not in the edit table
            if ind == "0":                     # not separately reportable, no bypass
                outcomes.append(Outcome.BLOCKED)
                detail.append(f"{lines[i].chosen.code}/{lines[j].chosen.code} bundled (0)")
            elif ind == "1":                   # bypassable with a justified modifier
                pair = frozenset((lines[i].chosen.code, lines[j].chosen.code))
                if pair in result.bypassed_ncci:   # a distinct-service modifier was applied
                    outcomes.append(Outcome.PASS)
                else:
                    outcomes.append(Outcome.UNKNOWN)
                    detail.append(f"{lines[i].chosen.code}/{lines[j].chosen.code} needs modifier (1)")
    if len(lines) < 2:
        supp = getattr(result, "ncci_suppressed", [])
        if supp:
            pairs = "; ".join(f"{c} bundled into {p}" for c, p in supp)
            return GateResult("ncci_ptp", Outcome.PASS,
                              f"NCCI PTP applied in reconciliation ({pairs}); no unresolved "
                              f"conflicts among released line(s)", "NCCI PTP (data)")
        return GateResult("ncci_ptp", Outcome.NOT_APPLICABLE, "fewer than two procedures",
                          "NCCI PTP (data)")
    return GateResult("ncci_ptp", _worst(outcomes) if outcomes else Outcome.PASS,
                      "no unresolved PTP conflicts" if not detail else "; ".join(detail),
                      "NCCI PTP (data)", retryable=retryable)


def mue_gate(result: CodingResult, source: CodeSource) -> GateResult:
    outcomes, detail = [], []
    for ln in result.billable_lines:
        limit = source.mue_limit(ln.chosen.code, result.date_of_service)
        units = ln.units                        # descriptor-driven billing units
        if limit is None:
            continue                            # no MUE published for this code
        if units > limit:
            outcomes.append(Outcome.BLOCKED)
            detail.append(f"{ln.chosen.code}: {units} > MUE {limit}")
        else:
            outcomes.append(Outcome.PASS)
    if result.billable_lines and not source.mue_available():
        return GateResult("mue", Outcome.UNKNOWN,
                          "MUE table unavailable — unit limits cannot be asserted",
                          "MUE (data)", retryable=True)
    if not outcomes:
        return GateResult("mue", Outcome.NOT_APPLICABLE, "no MUE-constrained lines",
                          "MUE (data)")
    return GateResult("mue", _worst(outcomes),
                      "units within MUE" if not detail else "; ".join(detail), "MUE (data)")


# Necessity relation control floor: a model-authored REASON_FOR edge below this confidence
# does not, by itself, certify medical necessity (a 0.0-confidence edge must not). A control
# threshold, not a medical code.
_MIN_NECESSITY_RELATION_CONFIDENCE = 0.5


def medical_necessity_gate(result: CodingResult,
                           source: "CodeSource | None" = None) -> GateResult:
    """Every released procedure needs a diagnosis that JUSTIFIES IT, evaluated PER released
    service and against AUTHORITATIVE coverage policy:

    - the support relation (diagnosis --REASON_FOR--> service) must be asserted AND clear the
      necessity relation confidence floor -- a low/zero-confidence model edge cannot certify;
    - when the procedure is governed by an authoritative CMS coverage policy
      (source.qualifying_dx_for), at least one linked RELEASED diagnosis must qualify under
      it, else it is a coverage contradiction and HOLDs;
    - an unavailable required coverage evaluation is UNKNOWN/HOLD.

    (Codex F6-R3: an all-to-all count, and even a bare/zero-confidence edge, were insufficient.)"""
    procs = result.procedure_lines
    if not procs:
        return GateResult("medical_necessity", Outcome.NOT_APPLICABLE,
                          "no procedures to justify", "necessity")
    dx_lines = result.diagnosis_lines
    if not dx_lines:
        return GateResult("medical_necessity", Outcome.BLOCKED,
                          "performed procedure(s) with no documented diagnosis",
                          "necessity (structural)")
    released_dx = {ln.fact.fact_id: ln for ln in dx_lines
                   if ln.fact is not None and ln.fact.fact_id}
    reason_for = [r for r in (result.relations or [])
                  if r.predicate is RelationPredicate.REASON_FOR
                  and r.state is RelationState.ASSERTED
                  and float(getattr(r, "confidence", 0.0) or 0.0)
                  >= _MIN_NECESSITY_RELATION_CONFIDENCE]
    holds: list[str] = []
    for ln in procs:
        pid = ln.fact.fact_id if ln.fact is not None else None
        label = (ln.chosen.code if ln.chosen else None) or (
            ln.fact.description if ln.fact else "procedure")
        support_events = {r.subject_event_id for r in reason_for
                          if r.object_event_id == pid and r.subject_event_id in released_dx}
        if not support_events:
            holds.append(f"{label}: no confident REASON_FOR link to a released diagnosis")
            continue
        if source is not None and ln.chosen is not None:
            try:
                qualifying = source.qualifying_dx_for(ln.chosen.code, ln.chosen.system)
            except Exception:
                holds.append(f"{label}: coverage-policy evaluation unavailable")
                continue
            if qualifying is not None:                       # governed by a coverage policy
                want = {str(q).replace(".", "").upper() for q in qualifying}
                got = {released_dx[e].chosen.code.replace(".", "").upper()
                       for e in support_events if released_dx[e].chosen}
                if not (got & want):
                    holds.append(
                        f"{label}: linked diagnosis does not qualify under CMS coverage policy")
                    continue
    if holds:
        return GateResult("medical_necessity", Outcome.UNKNOWN, "; ".join(holds),
                          "necessity (structural + coverage)")
    return GateResult("medical_necessity", Outcome.PASS,
                      f"{len(procs)} procedure(s) each supported by a confident, "
                      "policy-qualifying released diagnosis", "necessity (structural + coverage)")


def icd_excludes_gate(result: CodingResult, source: CodeSource) -> GateResult:
    """ICD-10-CM Tabular Excludes1: two diagnoses in an Excludes1 relationship are
    'NOT CODED HERE' — mutually exclusive, not reportable together, UNLESS the two
    conditions are genuinely unrelated (an FY-guideline exception that is a human
    judgement). This is the diagnosis-axis analogue of the NCCI procedure gate, read
    from the authoritative instructional notes — never a baked-in pair list.

    Fail-closed: a detected Excludes1 co-occurrence returns UNKNOWN (stops autonomy,
    routes to review) rather than PASS or BLOCKED — the coder cannot deterministically
    assert the two conditions are unrelated, so it cannot certify the set clean, and
    it also must not silently drop a possibly-valid code."""
    dx = [ln.chosen for ln in result.billable_lines
          if ln.chosen and ln.chosen.system == "icd10"]
    if len(dx) < 2:
        return GateResult("icd_excludes1", Outcome.NOT_APPLICABLE,
                          "fewer than two diagnoses", "ICD-10-CM Tabular (data)")
    conflicts: list[str] = []
    seen: set[frozenset[str]] = set()
    for a in dx:
        refs = source.excludes1_refs(a.code, "icd10")
        if not refs:
            continue
        for b in dx:
            if a.code == b.code:
                continue
            bu = str(b.code).replace(".", "").upper()
            if any(bu.startswith(r) for r in refs):
                pair = frozenset((a.code, b.code))
                if pair not in seen:
                    seen.add(pair)
                    conflicts.append(f"{a.code}/{b.code}")
    if conflicts:
        return GateResult("icd_excludes1", Outcome.UNKNOWN,
                          "Excludes1 pair(s) — confirm the conditions are unrelated "
                          f"before reporting together: {'; '.join(conflicts)}",
                          "ICD-10-CM Tabular (data)")
    return GateResult("icd_excludes1", Outcome.PASS,
                      "no Excludes1 conflicts among diagnoses", "ICD-10-CM Tabular (data)")


def claim_ownership_gate(result: CodingResult) -> GateResult:
    """Tri-state claim ownership over structured actor and organization identities."""
    from .ownership import fact_ownership, classify_ownership
    blocked: list[str] = []
    unstated = 0
    for ln in result.billable_lines:
        if ln.fact.kind is FactKind.DIAGNOSIS:
            continue
        o = fact_ownership(ln.fact)
        st = classify_ownership(o.performer_id, o.billing_entity_id,
                                o.organization_id, o.performer_function)
        if st is Outcome.BLOCKED:
            blocked.append(ln.chosen.code if ln.chosen else ln.fact.description)
        elif st is Outcome.UNKNOWN:
            unstated += 1
    owned_lines = [ln for ln in result.billable_lines if ln.fact.kind is not FactKind.DIAGNOSIS]
    if not owned_lines:
        return GateResult("claim_ownership", Outcome.NOT_APPLICABLE, "no billable lines",
                          "ownership")
    if blocked:
        return GateResult("claim_ownership", Outcome.BLOCKED,
                          f"billed by an entity that did not perform the service: {blocked}",
                          "ownership")
    if unstated:
        return GateResult("claim_ownership", Outcome.UNKNOWN,
                          f"ownership unresolved for {unstated} billed line(s)",
                          "ownership", retryable=True)
    return GateResult("claim_ownership", Outcome.PASS,
                      "all billed services are owned by the billing entity",
                      "ownership")


def source_manifest_gate(result: CodingResult) -> GateResult:
    """Fail closed on a MISSING REQUIRED authoritative source. Degradation is loud:
    an absent required source (a code table / edit-policy file) BLOCKS release; absent
    OPTIONAL recall aids are recorded (NOT_APPLICABLE detail) but do not block."""
    try:
        from .capability import build_manifest
        man = build_manifest()
    except Exception as exc:
        return GateResult("source_manifest", Outcome.ERROR, f"manifest unavailable: {exc}",
                          "capability manifest", retryable=True)
    if man.get("missing_required"):
        return GateResult("source_manifest", Outcome.BLOCKED,
                          "required source(s) missing: " + ", ".join(man["missing_required"]),
                          "capability manifest")
    degraded = man.get("degraded_optional") or []
    detail = ("all required sources loaded"
              + (f"; optional absent: {', '.join(degraded)}" if degraded else ""))
    return GateResult("source_manifest", Outcome.PASS, detail, "capability manifest")


def run_gates(result: CodingResult, note_text: str, source: CodeSource) -> list[GateResult]:
    """All mandatory gates. Add a gate here (never a code list) as coverage grows."""
    try:
        return [
            source_manifest_gate(result),
            claim_ownership_gate(result),
            dos_gate(result),
            evidence_gate(result, note_text),
            code_active_gate(result, source),
            medical_necessity_gate(result, source),
            ncci_gate(result, source),
            mue_gate(result, source),
            icd_excludes_gate(result, source),
        ]
    except Exception as exc:  # a gate that crashes is ERROR, never a silent pass
        return [GateResult("gate_execution", Outcome.ERROR, str(exc), "runtime")]
