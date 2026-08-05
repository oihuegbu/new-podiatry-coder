"""Documentation recommendations — turn escalations and gaps into ACTIONABLE guidance.

When the coder cannot confidently code something, the useful output is not just
"REVIEW" but WHY and WHAT WOULD FIX IT. This derives structured, provider-facing
recommendations from the coding result:

  • documentation_gap  — a code fits the service but its descriptor requires an
                         element the note does not state (a provider query);
  • unresolved_service — documentation too thin / ambiguous to select any code;
  • gate_<name>        — a release gate blocked (missing evidence, no supporting
                         diagnosis, missing DOS, inactive code).

Deterministic and agnostic: it reads fact kinds, resolution methods, and gate
outcomes — never a medical code. Each recommendation is
{issue, subject, detail, recommendation} so a UI / worklist can act on it.
"""
from __future__ import annotations

from .models import CodingResult, FactKind, Outcome

# Gate name -> what the provider/coder should do to clear it. Gate names are the
# tool's own identifiers, not codes.
_GATE_REMEDIATION = {
    "verbatim_evidence": "A billed service is not supported by verbatim note text — "
                         "ensure the note states each service exactly as coded.",
    "medical_necessity": "A performed procedure has no documented supporting "
                         "diagnosis — document the indication / diagnosis.",
    "date_of_service": "No date of service is present — add the DOS.",
    "code_active_on_dos": "A selected code is not valid/active for the date of "
                          "service — recode against the current code set.",
}


def build_recommendations(result: CodingResult) -> list[dict]:
    recs: list[dict] = []

    # 1. Documentation gaps flagged at resolution (a code fits, an element is
    #    undocumented) — a targeted provider query.
    for ln in result.lines:
        if ln.documentation_gap and not ln.resolved:
            recs.append({
                "issue": "documentation_gap",
                "subject": ln.fact.description,
                "fact_id": ln.fact.fact_id,
                "detail": ln.documentation_gap,
                "recommendation":
                    f"For '{ln.fact.description}', confirm and document: "
                    f"{ln.documentation_gap}. If performed, the specific code is then "
                    f"supported; if not, a less-specific code applies.",
            })

    # 2. Billable services/conditions that did not resolve to a released code. When
    #    the pipeline DID retrieve candidate code(s) but verification could not confirm
    #    one, the open question is a CODING decision (which code / separately reportable
    #    vs integral / policy) — a coder review, NOT a documentation query; surface the
    #    specific candidate code(s) so the coder starts from them, and flag a residual/
    #    'other'/NEC bucket when that is what was retrieved. Only when NOTHING was
    #    retrieved is the note genuinely too thin to code. DIAGNOSIS lines are included
    #    (they previously produced no recommendation at all).
    from .resolution import _RESIDUAL_MARKERS
    for ln in result.lines:
        if not (ln.fact.billable and not ln.resolved and not ln.documentation_gap):
            continue
        if ln.fact.kind is FactKind.EM:
            recs.append({"issue": "unresolved_service", "subject": ln.fact.description,
                         "fact_id": ln.fact.fact_id, "detail": ln.rationale,
                         "recommendation": (
                             "Document the medical decision-making elements (problems "
                             "addressed, data reviewed, risk) or the total visit time so "
                             "the E/M level can be determined.")})
            continue
        if ln.fact.kind not in (FactKind.PROCEDURE, FactKind.IMAGING, FactKind.SUPPLY,
                                FactKind.DRUG, FactKind.DIAGNOSIS):
            continue
        cands = [c for c in ln.alternatives if c.code]
        if cands:
            names = ", ".join(dict.fromkeys(
                f"{c.system.upper()} {c.code}" for c in cands[:4]))
            residual = next((c for c in cands
                             if any(m in c.descriptor.lower() for m in _RESIDUAL_MARKERS)),
                            None)
            extra = ("" if residual is None else
                     f" {residual.system.upper()} {residual.code} is a residual/'other'/"
                     f"NEC category — if used, confirm no more-specific code applies and "
                     f"evaluate any diagnosis-pair (tabular) or PTP interaction before "
                     f"release.")
            recs.append({
                "issue": "coder_review", "subject": ln.fact.description,
                "fact_id": ln.fact.fact_id, "detail": ln.rationale,
                "recommendation": (
                    f"Candidate code(s) {names} were identified for "
                    f"'{ln.fact.description}' but not confirmed by independent "
                    f"verification. This is a coding decision — code selection, or "
                    f"whether the service is separately reportable vs integral, or a "
                    f"payer-policy question — not necessarily a documentation gap. "
                    f"Coder to review and select/confirm.{extra}")})
        else:
            recs.append({
                "issue": "unresolved_service", "subject": ln.fact.description,
                "fact_id": ln.fact.fact_id, "detail": ln.rationale,
                "recommendation": (
                    f"No candidate code was retrieved for '{ln.fact.description}' — "
                    f"clarify the exact service/condition (site, laterality, technique, "
                    f"and any product/dose) so it can be coded.")})

    # 3. Gate-based remediation — what to fix to earn release.
    for g in result.gates:
        if g.outcome is Outcome.BLOCKED and g.name in _GATE_REMEDIATION:
            recs.append({"issue": f"gate_{g.name}", "subject": g.name, "fact_id": "",
                         "detail": g.detail,
                         "recommendation": _GATE_REMEDIATION[g.name]})

    return recs
