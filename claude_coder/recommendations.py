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

    # 2. Billable services that could not be resolved at all — the documentation was
    #    too thin or ambiguous to pick any code.
    for ln in result.lines:
        if not (ln.fact.billable and not ln.resolved and not ln.documentation_gap):
            continue
        if ln.fact.kind is FactKind.EM:
            rec = ("Document the medical decision-making elements (problems "
                   "addressed, data reviewed, risk) or the total visit time so the "
                   "E/M level can be determined.")
        elif ln.fact.kind in (FactKind.PROCEDURE, FactKind.IMAGING,
                              FactKind.SUPPLY, FactKind.DRUG):
            rec = (f"Documentation was insufficient to select a specific code for "
                   f"'{ln.fact.description}' — clarify the exact service (procedure, "
                   f"site, laterality, technique, and any product/dose).")
        else:
            continue
        recs.append({"issue": "unresolved_service", "subject": ln.fact.description,
                     "fact_id": ln.fact.fact_id, "detail": ln.rationale,
                     "recommendation": rec})

    # 3. Gate-based remediation — what to fix to earn release.
    for g in result.gates:
        if g.outcome is Outcome.BLOCKED and g.name in _GATE_REMEDIATION:
            recs.append({"issue": f"gate_{g.name}", "subject": g.name, "fact_id": "",
                         "detail": g.detail,
                         "recommendation": _GATE_REMEDIATION[g.name]})

    return recs
