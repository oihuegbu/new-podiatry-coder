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
    FactKind,
    Outcome,
    ResolutionMethod,
    ResolvedLine,
    Verdict,
)


def _necessity_authoritatively_met(result: CodingResult, source) -> bool:
    """Is the claim's medical necessity CONFIRMED by resolved diagnoses per
    authoritative coverage — so an unresolved diagnosis adds nothing the necessity
    requires (non-material)? True ONLY when every billed procedure/service is
    governed by a CMS LCD/Article AND a RESOLVED diagnosis on the claim is one of its
    qualifying (covered) indications. If the source is absent, there is no billed
    procedure, a billed procedure is governed by NO policy (necessity unconfirmable),
    or a governed procedure has no resolved qualifying diagnosis, this is False and an
    unresolved diagnosis BLOCKS (fail-closed). This replaces the earlier
    'another dx resolved -> secondary' PROXY with a data-driven necessity-linkage
    query — it cannot mistake a procedure's principal indication for a redundant one."""
    if source is None:
        return False
    procs = [ln for ln in result.billable_lines
             if ln.chosen and ln.chosen.system in ("cpt", "hcpcs")
             and ln.fact.kind is not FactKind.EM]
    if not procs:
        return False
    dx = {str(ln.chosen.code).replace(".", "").upper() for ln in result.diagnosis_lines}
    for p in procs:
        try:
            qd = source.qualifying_dx_for(p.chosen.code, p.chosen.system)
        except Exception:
            return False
        if not qd or not (dx & qd):      # ungoverned, or no resolved covered indication
            return False
    return True

# Which destination wins when several apply: a hard stop first, then an operational
# retry, then genuine coding judgement, then a provider question, then a do-not-bill
# hold. A coder (REVIEW) only sees what truly needs a coder.
_PRECEDENCE = [Destination.BLOCKED, Destination.SYSTEM_HOLD, Destination.REVIEW,
               Destination.PROVIDER_QUERY, Destination.HOLD]

# Autonomy floor. Retained as the ROUTING PRECEDENCE anchor and the calibration
# knob the metamorphic suite pins; it is NOT the release gate — release is gated on
# CLOSURE (grounded resolution + cleared deterministic gates), see decide() §4.
AUTONOMY_CONFIDENCE = 0.95
# A single model tie-break is trusted less than a deterministic descriptor
# entailment or a cross-model-confirmed one.
_ARBITRATED_DISCOUNT = 0.9
# The one place a self-reported number still gates: a fact whose own extraction
# confidence is this low is not "the code is uncertain" but "the NOTE barely
# documents the event" — the fact itself is shaky, so even a grounded code on it
# gets a human. This is a floor on DOCUMENTATION clarity, not on code choice.
SHAKY_EXTRACTION = 0.5


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
           floor: float = AUTONOMY_CONFIDENCE,
           source=None) -> Verdict:
    """Set the release verdict AND route every open item to its real destination —
    an operational failure to SYSTEM_HOLD (retry), a documentation gap to
    PROVIDER_QUERY, a genuine coding judgement to REVIEW — instead of collapsing all
    of them into one human queue. Fail-closed: nothing auto-releases unless the chain
    closes. `verdict` stays AUTO_READY / REVIEW_REQUIRED / BLOCKED for compatibility;
    `destination` + `routing` carry the actionable breakdown."""
    routing: list[dict] = []

    def route(dest: Destination, subject: str, reason: str, blocking: bool = True,
              fact_id: str = "") -> None:
        # fact_id is the STABLE join key back to this line's suggested-solution
        # recommendation (descriptions are not unique); the pipeline uses it to make
        # each routed item self-contained.
        routing.append({"destination": dest.value, "subject": subject, "reason": reason,
                        "blocking": blocking, "fact_id": fact_id})

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

    # 3. Every performed fact must be accounted for. MATERIALITY is decided from
    #    authoritative coverage, not a proxy: an unresolved DIAGNOSIS is non-material
    #    (non-blocking, clarify via provider query) ONLY when every billed procedure's
    #    medical necessity is already confirmed by a RESOLVED qualifying diagnosis per
    #    LCD/Article coverage — so the unresolved one adds nothing necessity requires.
    #    Otherwise (any ungoverned procedure, or necessity not yet met, or no coverage
    #    data) it BLOCKS — this is what keeps a procedure's PRINCIPAL indication (an
    #    unresolved principal diagnosis on an ungoverned procedure) from being
    #    released. A code that FITS but needs an undocumented element is a material
    #    PROVIDER_QUERY; anything else needs a coder.
    dx_non_material = _necessity_authoritatively_met(result, source)
    for ln in result.lines:
        if ln.fact.billable and not ln.resolved:
            if ln.fact.kind is FactKind.DIAGNOSIS and dx_non_material:
                route(Destination.PROVIDER_QUERY, ln.fact.description,
                      "diagnosis could not be coded — non-material: every billed procedure's "
                      "necessity is already met by a resolved qualifying diagnosis per "
                      "authoritative coverage; clarify to add specificity",
                      blocking=False, fact_id=ln.fact.fact_id)
            elif ln.documentation_gap:
                route(Destination.PROVIDER_QUERY, ln.fact.description, ln.documentation_gap,
                      fact_id=ln.fact.fact_id)
            else:
                route(Destination.REVIEW, ln.fact.description, ln.rationale,
                      fact_id=ln.fact.fact_id)

    billable = result.billable_lines
    if not billable:
        route(Destination.REVIEW, "claim", "no defensible billable line was produced")

    # 4. Release rests on CLOSURE, not on a self-reported confidence number. A line
    #    is autonomous when its code is GROUNDED — resolved by a deterministic
    #    authoritative match (DETERMINISTIC) or a code the documentation entails,
    #    confirmed by an INDEPENDENT second model (VERIFIED) — AND every applicable
    #    deterministic gate above cleared. A single-model tie-break (ARBITRATED) is
    #    NOT grounded: one model's pick among candidates, with no independent
    #    corroboration, is exactly the judgement call a coder owns. The LLM's
    #    self-reported/agreement confidence is deliberately NOT the gate (it is poorly
    #    calibrated); the only self-report still consulted is the SHAKY_EXTRACTION
    #    floor — a fact the note barely documents gets a human even when its code is
    #    grounded, because the uncertainty is in the DOCUMENTATION, not the code.
    for ln in billable:
        if ln.method is ResolutionMethod.ARBITRATED:
            route(Destination.REVIEW, ln.fact.description,
                  "code chosen by a single model among candidates (arbitrated), not "
                  "grounded by an authoritative match or an independently verified "
                  "entailment — needs a coder",
                  fact_id=ln.fact.fact_id)
        elif ln.fact.min_confidence < SHAKY_EXTRACTION:
            _wk = ln.fact.weakest_axis
            _axis = f", weakest axis '{_wk}'" if _wk else ""
            route(Destination.REVIEW, ln.fact.description,
                  f"the note barely documents this event (confidence "
                  f"{ln.fact.min_confidence:.2f} < {SHAKY_EXTRACTION:.2f}{_axis}) — "
                  f"clarify before billing",
                  fact_id=ln.fact.fact_id)

    result.routing = routing
    # Only MATERIAL (blocking) items gate release. Non-material clarifications go out
    # as provider queries in parallel while the defensible claim releases.
    blocking = [r for r in routing if r["blocking"]]
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
        tag = "" if r["blocking"] else " (non-blocking)"
        result.notes.append(f"  [{r['destination']}]{tag} {r['subject']}: {r['reason']}")
    return result.verdict
