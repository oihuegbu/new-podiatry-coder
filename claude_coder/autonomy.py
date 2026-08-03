"""The autonomy controller — calibrated abstention, not blanket automation.

Bounded autonomy: an encounter is released to billing with NO human only when
the evidence chain CLOSES — every mandatory gate clears, every billable fact
resolved to a code, and confidence clears the autonomy floor. Anything else
escalates to a human queue with a precise reason. This mirrors the pattern the
autonomous-coding leaders use (route to billing above a confidence threshold,
review below) and is the safety property that lets automation be trusted: the
system codes every note it CAN defend and steps back from the rest, instead of
coding everything and hoping.

The threshold is a policy dial, not a code fact — it lives here, configurable,
with a full audit trail for every decision.
"""
from __future__ import annotations

from .models import (
    CodingResult,
    Outcome,
    ResolutionMethod,
    ResolvedLine,
    Verdict,
)

# Autonomy floor. The category benchmark for hands-off release is ~95% coding
# confidence; below it, a human reviews. Tune per cohort/payer, never per code.
AUTONOMY_CONFIDENCE = 0.95
# Model tie-breaks are trusted less than a deterministic descriptor entailment.
_ARBITRATED_DISCOUNT = 0.9


def _line_confidence(line: ResolvedLine) -> float:
    if not line.resolved:
        return 0.0
    if line.method is ResolutionMethod.DETERMINISTIC:
        return line.fact.confidence
    if line.method is ResolutionMethod.ARBITRATED:
        return line.fact.confidence * _ARBITRATED_DISCOUNT
    return 0.0


def decide(result: CodingResult,
           floor: float = AUTONOMY_CONFIDENCE) -> Verdict:
    """Set and return the release verdict, recording the audit trail on the
    result. Fail-closed: unresolved conditions escalate or block, never auto."""
    reasons: list[str] = []

    # 1. Gates. A hard stop dominates everything.
    if any(g.outcome in (Outcome.BLOCKED, Outcome.ERROR) for g in result.gates):
        blocked = [g.name for g in result.gates
                   if g.outcome in (Outcome.BLOCKED, Outcome.ERROR)]
        result.notes.append(f"BLOCKED by gate(s): {blocked}")
        result.verdict = Verdict.BLOCKED
        return result.verdict

    if any(g.outcome is Outcome.UNKNOWN for g in result.gates):
        unknown = [g.name for g in result.gates if g.outcome is Outcome.UNKNOWN]
        reasons.append(f"gate(s) UNKNOWN (unverifiable): {unknown}")

    # 2. Completeness — every performed fact must be accounted for.
    abstained = [ln for ln in result.lines
                 if ln.fact.billable and not ln.resolved]
    if abstained:
        reasons.append(f"{len(abstained)} performed fact(s) unresolved — "
                       f"e.g. {abstained[0].fact.description!r} "
                       f"({abstained[0].rationale})")

    billable = result.billable_lines
    if not billable:
        reasons.append("no defensible billable line was produced")

    # 3. Confidence floor across every released line.
    low = [ln for ln in billable if _line_confidence(ln) < floor]
    if low:
        worst = min(billable, key=_line_confidence)
        reasons.append(f"{len(low)} line(s) below autonomy floor {floor:.2f} "
                       f"(min {_line_confidence(worst):.2f}, "
                       f"{worst.method.value})")

    if reasons:
        result.notes.extend(reasons)
        result.verdict = Verdict.REVIEW_REQUIRED
    else:
        result.notes.append(
            f"AUTO_READY — {len(billable)} line(s), all gates clear, "
            f"min confidence >= {floor:.2f}")
        result.verdict = Verdict.AUTO_READY
    return result.verdict
