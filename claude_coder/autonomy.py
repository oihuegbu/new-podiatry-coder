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
    """Is the claim's medical necessity ALREADY CONFIRMED for every billed procedure — so an
    unresolved diagnosis adds nothing the necessity requires (non-material)?

    This reads the necessity gate's OWN resolved binding (`result.necessity_support`), which
    records, per procedure, the claim-line diagnosis that justified it: an encounter-specific
    reconciled REASON_FOR linkage AND, where an authoritative CMS LCD/Article governs the
    service, that same diagnosis qualifying under the policy. It deliberately does not
    re-derive necessity from coverage membership: 'a covered diagnosis is somewhere on this
    claim' proves the pair CAN qualify, never that it justified THIS service — the same
    substitution the necessity gate itself no longer makes. (Codex F6-R3, adjacent instance.)

    False — and an unresolved diagnosis BLOCKS (fail-closed) — when the source is absent,
    there is no billed procedure, any billed procedure is governed by NO policy (necessity
    unconfirmable from published policy), or any billed procedure lacks a policy-qualifying
    linked diagnosis in the gate's binding.
    """
    if source is None:
        return False
    procs = [ln for ln in result.billable_lines
             if ln.chosen and ln.chosen.system in ("cpt", "hcpcs")
             and ln.fact.kind is not FactKind.EM]
    if not procs:
        return False
    bound = {b.get("procedure_event_id"): b for b in (result.necessity_support or [])}
    for p in procs:
        binding = bound.get(p.fact.fact_id if p.fact is not None else None)
        if binding is None or not binding.get("policy_governed"):
            return False                 # never justified here, or governed by no policy
        if not any(s.get("policy_qualifying") for s in (binding.get("supports") or [])):
            return False                 # linked, but not by a policy-qualifying diagnosis
    return True

# Which destination wins when several apply: a hard stop first, then an operational
# retry, then genuine coding judgement, then a provider question, then a do-not-bill
# hold. A coder (REVIEW) only sees what truly needs a coder.
# HOLD is the directive's NON_BILLABLE/EXCLUDED bucket and sits last on purpose: it is
# reached only when nothing else is open, so it can never mask a real block.
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
    # "INDEPENDENT" is a checked precondition, not a naming convention: `resolution`
    # mints VERIFIED only when the corroborating judgement came from a different declared
    # model provider (`verify.corroboration_origin`), so this branch cannot be reached by
    # one vendor agreeing with itself. Everything short of that lands on ARBITRATED below.
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
    #
    #    issue #6 F9-R8-A: when a gate NAMES which fact_ids its hold is actually
    #    about (`affected_fact_ids` -- e.g. `medical_necessity_gate` attributes a
    #    hold to the exact procedures lacking a resolved, qualifying diagnosis
    #    link, not the whole encounter), route it non-blocking here -- those
    #    named facts are excluded from THIS claim individually below, so the
    #    gate's own routing item must not ALSO hold every other, independently
    #    justified line hostage. A gate that does not name fact_ids (every gate
    #    before this round, and every encounter-wide authority/structural
    #    failure) is unchanged: empty `affected_fact_ids` still blocks everything.
    for g in result.gates:
        if g.outcome is Outcome.UNKNOWN:
            scoped = bool(g.affected_fact_ids)
            if g.retryable:
                route(Destination.SYSTEM_HOLD, g.name,
                      f"authority unavailable ({g.detail}) — retry, do not send to a coder",
                      blocking=not scoped)
            else:
                route(Destination.REVIEW, g.name,
                      f"unverifiable, needs coding/clinical judgement ({g.detail})",
                      blocking=not scoped)

    # issue #6 F9-R8-A: dependency-scoped partial release. An unresolved fact, or
    # a gate-named procedure, blocks ONLY the facts it can actually affect -- the
    # same clinical episode (via the graph's own edges, including REASON_FOR/
    # PART_OF relations the graph already models -- no new graph) or a procedure
    # a gate explicitly named -- never the whole encounter by default. Reuses
    # `ClinicalGraph.binding_for` exactly as the release certificate's own graph
    # binding already does.
    def _entangled(fact_id: str) -> set[str]:
        graph = getattr(result, "graph", None)
        binding_for = getattr(graph, "binding_for", None)
        if fact_id is None or not callable(binding_for):
            return set()
        return {e for e in binding_for([fact_id]).clinical_event_ids if e != fact_id}

    blocked_fact_ids: set[str] = set()
    for ln in result.lines:
        if ln.fact.billable and not ln.resolved and not ln.excluded_reason:
            blocked_fact_ids |= _entangled(ln.fact.fact_id)
    for g in result.gates:
        if g.outcome in (Outcome.UNKNOWN, Outcome.BLOCKED, Outcome.ERROR) and g.affected_fact_ids:
            for fid in g.affected_fact_ids:
                blocked_fact_ids.add(fid)
                blocked_fact_ids |= _entangled(fid)

    # issue #6 F9-R11-A/B: the one real, typed signal a caller (`pipeline.
    # _reconcile_claim_after_pruning`) needs to tell a DEPENDENCY exclusion
    # (this fact id) apart from a claim-set-mechanic-derived one (everything
    # else `excluded_reason` might name) -- set fresh every call, never
    # accumulated here; the caller is the one with a reason to accumulate.
    result.dependency_excluded_fact_ids = frozenset(blocked_fact_ids)

    # A currently-resolved, billable line entangled with an unresolved or
    # gate-held fact cannot be certified independently of it -- excluded from
    # THIS claim, visibly (never silently dropped: `excluded_reason` is exactly
    # the field `billable_lines`/the release certificate already key off).
    if blocked_fact_ids:
        for ln in result.lines:
            if (ln.resolved and ln.fact.billable and not ln.excluded_reason
                    and ln.fact.fact_id in blocked_fact_ids):
                ln.excluded_reason = (
                    f"excluded from this claim: entangled with an unresolved or "
                    f"gate-held fact sharing this line's clinical episode or "
                    f"necessity linkage, which could change this line's own "
                    f"billing correctness")

    # Computed AFTER the exclusion stamping above, not before: whether an
    # unresolved fact's own `blocking` flag (section 3) should fire depends on
    # whether its closure touches a fact that is STILL billable once every
    # OTHER exclusion this round decided has already been applied -- a fact
    # whose only entangled neighbor was itself JUST excluded (e.g. by a gate
    # naming it directly) must not ALSO keep blocking on that neighbor's
    # account; an isolated ambiguity with no real remaining impact must not
    # block the rest of the encounter, exactly like the existing
    # `dx_non_material` precedent this generalizes.
    remaining_billable_ids = {ln.fact.fact_id for ln in result.billable_lines
                              if ln.fact is not None}
    # A diagnosis-only billable set is not itself a submittable claim --
    # `claim_submitter.py` refuses a claim with no service lines -- so it must
    # not count as "something independently defensible to protect" for the
    # purpose of relaxing an UNRELATED diagnosis below (kills the exact mutant
    # `test_materiality_no_procedure_blocks` guards: no procedure anywhere ->
    # necessity can never be confirmed -> fail closed).
    has_remaining_procedure = any(ln.fact.kind is not FactKind.DIAGNOSIS
                                  for ln in result.billable_lines
                                  if ln.fact is not None)
    # Whether necessity was actually ASSESSED at all, for anything -- distinct
    # from `dx_non_material` (which asks whether it was assessed and CONFIRMED
    # for every procedure). When no `medical_necessity` gate ever ran (no
    # source was available to construct one), there is no signal at all about
    # which diagnoses matter, so a diagnosis cannot be safely treated as
    # unrelated merely because it has no graph edge to draw on.
    necessity_gate_ran = any(g.name == "medical_necessity" for g in result.gates)

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
        if ln.fact.billable and not ln.resolved and not ln.excluded_reason:
            if ln.fact.kind is FactKind.DIAGNOSIS and dx_non_material:
                route(Destination.PROVIDER_QUERY, ln.fact.description,
                      "diagnosis could not be coded — non-material: every billed procedure's "
                      "necessity is already met by a resolved qualifying diagnosis per "
                      "authoritative coverage; clarify to add specificity",
                      blocking=False, fact_id=ln.fact.fact_id)
            else:
                # issue #6 F9-R8-A, corrected by F9-R9-A (Codex's independent
                # re-review of 6ff2761): blocking only when this fact's own
                # closure touches a line that WAS billable -- an isolated
                # ambiguity with no documented relationship to anything else
                # must not hold an unrelated, independently defensible line
                # hostage. This now applies uniformly to EVERY fact kind,
                # diagnoses included: the unconditional `ln.fact.kind is
                # FactKind.DIAGNOSIS` clause this round removed is exactly
                # what F9-R9-A reproduced -- a diagnosis with NO relation to
                # anything (no REASON_FOR, no shared episode) still forced the
                # whole encounter away from AUTO_READY, even though its own
                # dependent procedure (if any) was already excluded via the
                # necessity gate's `affected_fact_ids` mechanism above, or it
                # had no dependent procedure at all. Dropping a code's
                # necessity justification is unaffected by this change --
                # `medical_necessity_gate` still requires the encounter-
                # specific REASON_FOR link (+ policy qualification where
                # governed) before a procedure ever reaches `billable_lines`/
                # `remaining_billable_ids` in the first place; this clause
                # only decides whether an UNRELATED open question is allowed
                # to block an otherwise-independent release.
                #
                # Three guards keep this from over-relaxing:
                # - requires something ELSE billable to protect in the first
                #   place. When NOTHING remains billable, `remaining_billable_
                #   ids` is empty and this item IS the whole encounter's
                #   material content -- it must stay blocking, so the "nothing
                #   open" catch-all a few lines below ("no defensible billable
                #   line was produced") never fires in its place with a
                #   vaguer, less specific reason.
                # - for a DIAGNOSIS specifically (never a procedure -- that
                #   question is its OWN entanglement check above, unaffected):
                #   a diagnosis-only billable set cannot submit on its own, so
                #   `has_remaining_procedure` must hold too, and necessity
                #   must have actually been ASSESSED (`necessity_gate_ran`) --
                #   no signal at all about which diagnoses matter is not
                #   license to assume this one is unrelated.
                _affects = (not remaining_billable_ids
                           or bool(_entangled(ln.fact.fact_id) & remaining_billable_ids)
                           or (ln.fact.kind is FactKind.DIAGNOSIS
                               and (not necessity_gate_ran or not has_remaining_procedure)))
                if ln.documentation_gap:
                    route(Destination.PROVIDER_QUERY, ln.fact.description,
                          ln.documentation_gap, blocking=_affects,
                          fact_id=ln.fact.fact_id)
                else:
                    route(Destination.REVIEW, ln.fact.description, ln.rationale,
                          blocking=_affects, fact_id=ln.fact.fact_id)

    # issue #6 item 7/F8-R3: a resolved code held for an unresolved (not
    # contradicted) administrative fact -- e.g. actor ownership -- is a real
    # code but not a submittable one. `billable_lines` already excludes it;
    # routed here, explicitly and blocking, so it can never silently vanish
    # (dropped from the claim) OR silently auto-release (the stamp existing but
    # nothing reading it, F8-R3's own finding). PROVIDER_QUERY because the open
    # question -- who performed this, under which billing entity -- is
    # administrative, not a clinical judgement a coder owns, and not an
    # operational/data failure a retry fixes.
    for ln in result.submission_held_lines:
        route(Destination.PROVIDER_QUERY, ln.fact.description,
              f"a defensible code ({ln.chosen.code}) was resolved, but submission "
              f"is held pending an unresolved administrative fact (claim ownership) "
              f"-- confirm the performing actor and billing entity before this "
              f"line is submitted",
              fact_id=ln.fact.fact_id)

    billable = result.billable_lines
    if not billable:
        # A claim with no billable line used to be treated as coding judgement by
        # default, which is the generic-review fallback the directive forbids (sections
        # 3 and 8). It is judgement only when nothing else already accounts for the
        # absence. Two earlier findings came through this one catch-all: an encounter
        # whose only open item was an unsettled fact axis (a provider query), and an
        # encounter whose every documented event had been excluded (NON_BILLABLE).
        _open = [r for r in routing if r["blocking"]]
        _all_disposed = bool(result.lines) and all(
            (ln.excluded_reason or not ln.fact.billable) for ln in result.lines)
        if _open:
            # Something is ALREADY open, and it is what explains the missing claim --
            # a provider question, a dependency to retry, a hard stop, or a line a
            # coder must settle. Adding "no defensible billable line was produced" on
            # top of it is a second, vaguer coder item for a cause already named, and
            # it made every SYSTEM_HOLD encounter with no lines look partly like coding
            # work. The precedence order is unaffected either way; the noise is not.
            pass
        elif _all_disposed:
            # NOTHING is left open and EVERY documented event was disposed of by an
            # EXPLICIT decision -- either the event is not a claimable occurrence at all
            # (not performed / not certain / not the patient's, so `fact.billable` is
            # False) or authoritative claim mechanics excluded it (integral, bundled,
            # not separately reportable, non-claim evidence). The documented event is
            # simply not claim-eligible, which is the directive's NON_BILLABLE/EXCLUDED
            # destination -- there is no judgement left for a coder to make.
            #
            # Until now `Destination.HOLD` appeared ONLY in `_PRECEDENCE` and was never
            # routed to by any path, so this entire class of encounter reached a coder
            # through the catch-all below. That is the fifth of the directive's five
            # named destinations, and it was unreachable.
            for ln in result.lines:
                route(Destination.HOLD, ln.fact.description,
                      ln.excluded_reason
                      or (f"not a claim-eligible event (disposition="
                          f"{ln.fact.disposition.value}, certain={ln.fact.certain}, "
                          f"experiencer={ln.fact.experiencer})"),
                      fact_id=ln.fact.fact_id)
        else:
            # Nothing is open and the events were NOT all disposed of -- in practice,
            # nothing at all was extracted from the document. Unexplained, so a human
            # looks at it; this is the one thing the catch-all was ever for.
            route(Destination.REVIEW, "claim",
                  "no defensible billable line was produced")

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
            # This floor is on DOCUMENTATION clarity, not on code choice (see the block
            # comment above). So when the WEAKEST AXIS is named, the open item already IS
            # a precise question about a specified claim field -- the directive's
            # AUTO_QUERY -- and not a judgement call a coder owns. With no axis recorded
            # the concern is diffuse and a coder remains the honest destination.
            route(Destination.PROVIDER_QUERY if _wk else Destination.REVIEW,
                  ln.fact.description,
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
