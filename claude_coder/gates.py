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

from .data_access import CodeSource
from .models import CodingResult, GateResult, Outcome, ResolvedLine


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
    outcomes, detail = [], []
    for i in range(len(lines)):
        for j in range(len(lines)):
            if i == j:
                continue
            ind = source.ncci_indicator(lines[i].chosen.code, lines[j].chosen.code,
                                        result.date_of_service)
            if ind is None:
                continue                       # this pair not in the edit table
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
        return GateResult("ncci_ptp", Outcome.NOT_APPLICABLE, "fewer than two procedures",
                          "NCCI PTP (data)")
    return GateResult("ncci_ptp", _worst(outcomes) if outcomes else Outcome.PASS,
                      "no unresolved PTP conflicts" if not detail else "; ".join(detail),
                      "NCCI PTP (data)")


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
    if not outcomes:
        return GateResult("mue", Outcome.NOT_APPLICABLE, "no MUE-constrained lines",
                          "MUE (data)")
    return GateResult("mue", _worst(outcomes),
                      "units within MUE" if not detail else "; ".join(detail), "MUE (data)")


def medical_necessity_gate(result: CodingResult) -> GateResult:
    """Every performed procedure needs a documented diagnosis to justify it.
    This is the structural floor of medical necessity; full LCD/NCD dx->procedure
    coverage linkage would query the policy data here (a documented gap), but a
    procedure with NO supporting diagnosis is never defensible and is blocked."""
    procs = result.procedure_lines
    if not procs:
        return GateResult("medical_necessity", Outcome.NOT_APPLICABLE,
                          "no procedures to justify", "necessity")
    if not result.diagnosis_lines:
        return GateResult("medical_necessity", Outcome.BLOCKED,
                          "performed procedure(s) with no documented diagnosis",
                          "necessity (structural)")
    return GateResult("medical_necessity", Outcome.PASS,
                      f"{len(procs)} procedure(s) supported by "
                      f"{len(result.diagnosis_lines)} diagnosis line(s)",
                      "necessity (structural)")


def run_gates(result: CodingResult, note_text: str, source: CodeSource) -> list[GateResult]:
    """All mandatory gates. Add a gate here (never a code list) as coverage grows."""
    try:
        return [
            dos_gate(result),
            evidence_gate(result, note_text),
            code_active_gate(result, source),
            medical_necessity_gate(result),
            ncci_gate(result, source),
            mue_gate(result, source),
        ]
    except Exception as exc:  # a gate that crashes is ERROR, never a silent pass
        return [GateResult("gate_execution", Outcome.ERROR, str(exc), "runtime")]
