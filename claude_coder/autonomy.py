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
    Destination,
    Outcome,
    ResolutionMethod,
    ResolvedLine,
    Verdict,
)

# Which destination wins when several apply: a hard stop first, then an operational
# retry, then genuine coding judgement, then a provider question, then a do-not-bill
# hold. A coder (REVIEW) only sees what truly needs a coder.
_PRECEDENCE = [Destination.BLOCKED, Destination.SYSTEM_HOLD, Destination.REVIEW,
               Destination.PROVIDER_QUERY, Destination.HOLD]

# Autonomy floor. The category benchmark for hands-off release is ~95% coding
# confidence; below it, a human reviews. Tune per cohort/payer, never per code.
AUTONOMY_CONFIDENCE = 0.95
# A single model tie-break is trusted less than a deterministic descriptor
# entailment or a cross-model-confirmed one.
_ARBITRATED_DISCOUNT = 0.9


def _line_confidence(line: ResolvedLine) -> float:
    if not line.resolved:
        return 0.0
    # DETERMINISTIC = authoritative index / structural descriptor entailment;
    # VERIFIED = propose-then-verify confirmed by an INDEPENDENT second model. Both
    # are high-trust groundings of the code itself, so both are gated only by how
    # well the underlying fact is documented (fact.confidence) — a cross-model-
    # confirmed line is not penalized to 0 just because an LLM was in the loop.
    if line.method in (ResolutionMethod.DETERMINISTIC, ResolutionMethod.VERIFIED):
        return line.fact.confidence
    if line.method is ResolutionMethod.ARBITRATED:      # single-model tie-break
        return line.fact.confidence * _ARBITRATED_DISCOUNT
    return 0.0


def decide(result: CodingResult,
           floor: float = AUTONOMY_CONFIDENCE) -> Verdict:
    """Set the release verdict AND route every open item to its real destination —
    an operational failure to SYSTEM_HOLD (retry), a documentation gap to
    PROVIDER_QUERY, a genuine coding judgement to REVIEW — instead of collapsing all
    of them into one human queue. Fail-closed: nothing auto-releases unless the chain
    closes. `verdict` stays AUTO_READY / REVIEW_REQUIRED / BLOCKED for compatibility;
    `destination` + `routing` carry the actionable breakdown."""
    routing: list[dict] = []

    def route(dest: Destination, subject: str, reason: str, blocking: bool = True) -> None:
        routing.append({"destination": dest.value, "subject": subject,
                        "reason": reason, "blocking": blocking})

    # 1. A hard gate stop dominates everything.
    hard = [g.name for g in result.gates
            if g.outcome in (Outcome.BLOCKED, Outcome.ERROR)]
    if hard:
        for name in hard:
            route(Destination.BLOCKED, name, "hard release gate failed")
        result.notes.append(f"BLOCKED by gate(s): {hard}")
        result.routing = routing
        result.destination = Destination.BLOCKED
        result.verdict = Verdict.BLOCKED
        return result.verdict

    # 2. Gates that could not be verified: an OPERATIONAL failure (authority
    #    unavailable) is a retry, not a coding problem; anything else is judgement.
    for g in result.gates:
        if g.outcome is Outcome.UNKNOWN:
            if g.retryable:
                route(Destination.SYSTEM_HOLD, g.name,
                      f"authority unavailable ({g.detail}) — retry, do not send to a coder")
            else:
                route(Destination.REVIEW, g.name,
                      f"unverifiable, needs coding/clinical judgement ({g.detail})")

    # 3. Every performed fact must be accounted for — but MATERIALLY. An unresolved
    #    SECONDARY diagnosis (necessity already met by another resolved diagnosis)
    #    does not change the billed claim's payment or necessity: clarify it via a
    #    provider query, but do NOT block the release of the defensible claim. A
    #    procedure/supply/etc. would ADD a billable line (material -> blocks); a code
    #    that FITS but needs an undocumented element is a material PROVIDER_QUERY.
    from .models import FactKind
    necessity_met = bool(result.diagnosis_lines)   # a resolved diagnosis already supports the claim
    for ln in result.lines:
        if ln.fact.billable and not ln.resolved:
            if ln.fact.kind is FactKind.DIAGNOSIS and necessity_met:
                route(Destination.PROVIDER_QUERY, ln.fact.description,
                      "secondary diagnosis could not be coded — non-material to the billed "
                      "claim (necessity met by another diagnosis); clarify to add specificity",
                      blocking=False)
            elif ln.documentation_gap:
                route(Destination.PROVIDER_QUERY, ln.fact.description, ln.documentation_gap)
            else:
                route(Destination.REVIEW, ln.fact.description, ln.rationale)

    billable = result.billable_lines
    if not billable:
        route(Destination.REVIEW, "claim", "no defensible billable line was produced")

    # 4. Confidence floor across every released line.
    low = [ln for ln in billable if _line_confidence(ln) < floor]
    if low:
        worst = min(billable, key=_line_confidence)
        route(Destination.REVIEW, "confidence",
              f"{len(low)} line(s) below autonomy floor {floor:.2f} "
              f"(min {_line_confidence(worst):.2f}, {worst.method.value})")

    result.routing = routing
    # Only MATERIAL (blocking) items gate release. Non-material clarifications go out
    # as provider queries in parallel while the defensible claim releases.
    blocking = [r for r in routing if r.get("blocking", True)]
    if not blocking:
        result.destination = Destination.AUTO_READY
        result.verdict = Verdict.AUTO_READY
        side = len(routing)
        result.notes.append(
            f"AUTO_READY — {len(billable)} line(s), all gates clear, no material block"
            + (f"; {side} non-material clarification(s) → PROVIDER_QUERY" if side else ""))
        for r in routing:
            result.notes.append(f"  [{r['destination']}] (non-blocking) "
                                f"{r['subject']}: {r['reason']}")
        return result.verdict

    present = {r["destination"] for r in blocking}
    result.destination = next(d for d in _PRECEDENCE if d.value in present)
    result.verdict = Verdict.REVIEW_REQUIRED
    from collections import Counter
    counts = Counter(r["destination"] for r in routing)
    result.notes.append("routing → " + ", ".join(f"{k}×{v}" for k, v in sorted(counts.items())))
    for r in routing:
        tag = "" if r.get("blocking", True) else " (non-blocking)"
        result.notes.append(f"  [{r['destination']}]{tag} {r['subject']}: {r['reason']}")
    return result.verdict
