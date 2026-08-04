"""The retrocalcaneal-exostectomy note as a FIXED coded state (from prior live
runs), so routing/verdict changes can be measured deterministically — codes held
constant, only the logic under test varies (isolates the change from LLM variance).
Synthetic-free: uses the note's real established codes only for measurement, never
in the coder. Import and call build_note_result()."""
from claude_coder.models import (ClinicalFact, CandidateCode, ResolvedLine, ResolutionMethod,
                                 FactKind, EvidenceSpan, CodingResult, GateResult, Outcome,
                                 Disposition, Verdict)

def _line(kind, desc, ev, code=None, system="cpt", method=ResolutionMethod.VERIFIED,
          conf=0.96, excluded=None, gap=None, disp=Disposition.PERFORMED, mods=None, units=1):
    f = ClinicalFact(kind=kind, description=desc, evidence=[EvidenceSpan(ev)],
                     disposition=disp, confidence=conf,
                     attributes={"laterality": "right"} if "right" in ev.lower() else {})
    chosen = CandidateCode(code, system, f"descriptor for {code}", 1.0) if code else None
    ln = ResolvedLine(fact=f, chosen=chosen, method=method if code else ResolutionMethod.ABSTAINED,
                      rationale=("authoritative descriptor entailed; independently confirmed" if code
                                 else ("PROVIDER QUERY — best code needs an element not stated" if gap
                                       else "no candidate's authoritative descriptor is fully entailed — escalate")))
    ln.excluded_reason = excluded; ln.documentation_gap = gap
    ln.modifiers = mods or []; ln.units = units
    return ln

def build_note_result():
    P, D = FactKind.PROCEDURE, FactKind.DIAGNOSIS
    lines = [
        _line(P, "Right retrocalcaneal exostectomy", "prominent right heel bone removed", "28118", mods=["RT"]),
        _line(FactKind.SUPPLY, "Suture anchors", "two suture anchors right heel", "C1713", system="hcpcs",
              method=ResolutionMethod.DETERMINISTIC),
        _line(D, "Haglund deformity of calcaneus", "Haglund prominence right heel", "M21.6X1", system="icd10"),
        _line(D, "Retrocalcaneal bursitis", "retrocalcaneal bursitis right", "M71.571", system="icd10"),
        _line(D, "Insertional Achilles enthesopathy", "insertional Achilles degeneration right", "M77.51", system="icd10"),
        _line(P, "Monitored anesthesia care with ankle block", "regional ankle block right",
              "01462", excluded="anesthesia-section service — not separately reportable by the operating provider"),
        # escalations
        _line(P, "Retrocalcaneal bursectomy", "inflamed retrocalcaneal bursa removed"),
        _line(P, "Achilles tendon debridement", "abnormal damaged tendon tissue removed"),
        _line(P, "Achilles tendon reattachment", "two suture anchors reattach right achilles",
              gap="does not establish this is a secondary (repeat/delayed) repair"),
        _line(FactKind.IMAGING, "Intraoperative fluoroscopy of right heel", "imaging confirmed resection right",
              gap="descriptor's time element and 'separate procedure' context not established"),
        # not billable
        _line(D, "Atrial fibrillation", "atrial fibrillation", disp=Disposition.HISTORICAL),
        _line(P, "Aspirin for DVT prophylaxis", "aspirin twice daily", disp=Disposition.ORDERED),
    ]
    gates = [
        GateResult("date_of_service", Outcome.PASS, "DOS = 2026-01-05", "input contract"),
        GateResult("verbatim_evidence", Outcome.PASS, "all lines supported", "note text"),
        GateResult("code_active_on_dos", Outcome.PASS, "all active", "authoritative source"),
        GateResult("medical_necessity", Outcome.PASS, "1 procedure <- 3 dx", "necessity (structural)"),
        GateResult("ncci_ptp", Outcome.PASS, "no conflicts", "NCCI PTP (data)"),
        GateResult("mue", Outcome.PASS, "within MUE", "MUE (data)"),
        GateResult("icd_excludes1", Outcome.UNKNOWN,
                   "Excludes1 pair(s) — confirm unrelated: M71.571/M77.51", "ICD-10-CM Tabular (data)"),
    ]
    return CodingResult(encounter_id="haglund", date_of_service="2026-01-05", lines=lines, gates=gates)
