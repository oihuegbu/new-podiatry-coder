"""Phase-1 eligibility engine (SHADOW) — decide which documented events may become a
code-search candidate BEFORE any code retrieval, producing a code-free ClaimLineIntent.

This is the plane that fixes `billable == performed`: a performed event is not
automatically a claim line. Each billable fact runs a sequence of TRI-STATE gates
(PASS / BLOCK / UNRESOLVED — never a silent default-to-eligible), and only an event that
clears them becomes an ELIGIBLE intent. Integral / not-performed / supporting events stay
in the record as NON_CLAIM_EVIDENCE; material ambiguity becomes AUTO_HOLD.

Runs in shadow: it emits intents + a decision trace for audit/diffing; it does not yet
gate retrieval (that flip is Phase 1c, one gate at a time). Agnostic — reads fact kinds,
dispositions, actor ids, relationship assertions and evidence anchoring; no medical code.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .models import ClinicalFact, FactKind, Outcome, RelationPredicate, RelationState
from .ownership import classify_ownership, fact_ownership


class EligibilityState(str, Enum):
    ELIGIBLE_FOR_RETRIEVAL = "eligible_for_retrieval"
    NON_CLAIM_EVIDENCE = "non_claim_evidence"      # integral / not performed / supporting
    AUTO_HOLD = "auto_hold"                        # material ambiguity -> hold, do not bill


class ClaimComponent(str, Enum):
    SERVICE = "service"                            # a potential procedure/supply/drug line
    DIAGNOSIS_SUPPORT = "diagnosis_support"        # a diagnosis linked to a service, not a line


@dataclass
class EligibilityDecision:
    gate: str
    outcome: Outcome
    detail: str
    authority: str = ""


@dataclass
class ClaimLineIntent:
    """A code-free declaration that an event MAY legitimately become a claim line. This
    is the ONLY permissible input to billable-code retrieval (Phase 1c). It never carries
    a CPT/HCPCS/ICD code."""
    intent_id: str
    encounter_id: str
    component: ClaimComponent
    clinical_event_ids: list[str]
    fact_kind: str
    clinical_action: str
    attributes: dict[str, Any]
    date_of_service: str | None
    billing_entity_id: str | None
    source_span_ids: list[str]
    state: EligibilityState
    decisions: list[EligibilityDecision] = field(default_factory=list)
    service_episode_id: str | None = None
    mention_count: int = 1
    distinctness_facts: list[str] = field(default_factory=list)


def _intent_id(encounter_id: str, fact_id: str, action: str) -> str:
    raw = f"{encounter_id}|{fact_id}|{action}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


_SERVICE_KINDS = (FactKind.PROCEDURE, FactKind.IMAGING, FactKind.SUPPLY, FactKind.DRUG)


# --------------------------------------------------------------- tri-state gates
def _gate_evidence_required(fact: ClinicalFact) -> EligibilityDecision:
    """A billable event must rest on at least one VERIFIED (anchored) evidence span."""
    spans = fact.evidence or []
    if any(getattr(s, "anchored", False) for s in spans):
        return EligibilityDecision("evidence_required", Outcome.PASS,
                                   "anchored evidence present", "evidence integrity")
    if spans:
        return EligibilityDecision("evidence_required", Outcome.BLOCKED,
                                   "quoted evidence does not anchor to the source",
                                   "evidence integrity")
    return EligibilityDecision("evidence_required", Outcome.BLOCKED,
                               "no evidence span", "evidence integrity")


def _gate_occurrence(fact: ClinicalFact) -> EligibilityDecision:
    """Did the event actually occur for this billable encounter? Reuses the fact's own
    billability rule (disposition performed/administered/dispensed, certain, patient)."""
    if fact.billable:
        return EligibilityDecision("occurrence", Outcome.PASS,
                                   f"disposition={fact.disposition.value}", "occurrence")
    return EligibilityDecision("occurrence", Outcome.BLOCKED,
                               f"not a performed/claimable event "
                               f"(disposition={fact.disposition.value}, certain={fact.certain}, "
                               f"experiencer={fact.experiencer})", "occurrence")


def _gate_actor_ownership(fact: ClinicalFact) -> EligibilityDecision:
    o = fact_ownership(fact)
    st = classify_ownership(o.performer_id, o.billing_entity_id)
    if st is Outcome.BLOCKED:
        return EligibilityDecision("actor_ownership", Outcome.BLOCKED,
                                   "performed by a different actor than the billing entity",
                                   "claim ownership")
    if st is Outcome.UNKNOWN:
        return EligibilityDecision("actor_ownership", Outcome.UNKNOWN,
                                   "ownership unstated (assumed billing provider)",
                                   "claim ownership")
    return EligibilityDecision("actor_ownership", Outcome.PASS, "owned by billing entity",
                               "claim ownership")


def _relations_for(fact: ClinicalFact, relations: list) -> list:
    return [r for r in relations if r.subject_event_id == fact.fact_id]


def _gate_part_of_demotion(fact: ClinicalFact, relations: list) -> EligibilityDecision:
    """Demote a component ONLY when integrality is EXPLICITLY asserted (PART_OF, state
    ASSERTED) and NOT contradicted by a documented distinctness (SEPARATE_FROM asserted).
    A weak/UNCERTAIN relationship never demotes -- it defers to the conflict gate."""
    rels = _relations_for(fact, relations)
    part_of = [r for r in rels if r.predicate is RelationPredicate.PART_OF
               and r.state is RelationState.ASSERTED]
    if not part_of:
        return EligibilityDecision("part_of_demotion", Outcome.PASS,
                                   "no explicit integral-component relationship",
                                   "documented relationship")
    distinct = [r for r in rels if r.predicate is RelationPredicate.SEPARATE_FROM
                and r.state is RelationState.ASSERTED]
    if distinct:
        return EligibilityDecision("part_of_demotion", Outcome.PASS,
                                   "explicit PART_OF but documented distinctness present",
                                   "documented relationship")
    return EligibilityDecision("part_of_demotion", Outcome.BLOCKED,
                               "explicitly documented integral component of another event",
                               "documented relationship")


def _gate_conflict(fact: ClinicalFact, relations: list) -> EligibilityDecision:
    """A material relationship the passes could not agree on (UNCERTAIN) about whether
    this event is integral or distinct is a HOLD, not a guess."""
    material = {RelationPredicate.PART_OF, RelationPredicate.SEPARATE_FROM}
    uncertain = [r for r in _relations_for(fact, relations)
                 if r.predicate in material and r.state is RelationState.UNCERTAIN]
    if uncertain:
        preds = ", ".join(sorted({r.predicate.value for r in uncertain}))
        return EligibilityDecision("conflict", Outcome.UNKNOWN,
                                   f"unresolved relationship(s): {preds}",
                                   "relationship reconciliation")
    return EligibilityDecision("conflict", Outcome.PASS, "no unresolved material relationship",
                               "relationship reconciliation")


def _gate_documentation_minimum(fact: ClinicalFact) -> EligibilityDecision:
    """Enough structured signal to identify a coherent clinical service family: a
    clinical action (description) and a code system for its kind."""
    if str(fact.description or "").strip():
        return EligibilityDecision("documentation_minimum", Outcome.PASS,
                                   "clinical action documented", "documentation sufficiency")
    return EligibilityDecision("documentation_minimum", Outcome.UNKNOWN,
                               "no clinical action to search on", "documentation sufficiency")


def _classify(decisions: list[EligibilityDecision]) -> EligibilityState:
    """Precedence: NON_CLAIM (the event should never have been a line) before AUTO_HOLD
    (material ambiguity) before ELIGIBLE. A silent default-to-eligible is impossible —
    every state is reached only by an explicit gate outcome."""
    by = {d.gate: d.outcome for d in decisions}
    # NON_CLAIM: the event is definitively not an independent service line
    if by.get("occurrence") is Outcome.BLOCKED:
        return EligibilityState.NON_CLAIM_EVIDENCE
    if by.get("part_of_demotion") is Outcome.BLOCKED:
        return EligibilityState.NON_CLAIM_EVIDENCE
    # AUTO_HOLD: material ambiguity or a defensibility/ownership block
    if any(o is Outcome.BLOCKED for g, o in by.items()
           if g in ("evidence_required", "actor_ownership")):
        return EligibilityState.AUTO_HOLD
    if any(o is Outcome.UNKNOWN for g, o in by.items()
           if g in ("conflict", "documentation_minimum")):
        return EligibilityState.AUTO_HOLD
    return EligibilityState.ELIGIBLE_FOR_RETRIEVAL


@dataclass
class ServiceEpisode:
    """A clinically coherent session grouping related events (same encounter/DOS, linked
    by relationships). CONTEXT ONLY -- an episode never suppresses a claim line; it just
    scopes deduplication and distinctness analysis (SAME_EPISODE_AS != PART_OF !=
    non-reportable)."""
    episode_id: str
    encounter_id: str
    date_of_service: str | None
    event_ids: list[str]
    grouping_signals: list[str] = field(default_factory=list)


def _episode_id(encounter_id: str, dos_key: str) -> str:
    return hashlib.sha256(f"{encounter_id}|{dos_key}".encode("utf-8")).hexdigest()[:16]


def build_episodes(facts: list[ClinicalFact], relations: list, encounter_id: str,
                   dos: str | None):
    """Group events into service episodes. For a single encounter this is one episode per
    DOS (the operative session); relationships that link events reinforce it. Returns
    (episodes, fact_id -> episode_id). Grouping is non-suppressing."""
    groups: dict[str, list[str]] = {}
    for f in facts:
        groups.setdefault(dos or "unknown", []).append(f.fact_id)
    linked = any(r.predicate in (RelationPredicate.SAME_EPISODE_AS, RelationPredicate.PART_OF,
                                 RelationPredicate.USED_IN) and r.state is not RelationState.NEGATED
                 for r in (relations or []))
    episodes: list[ServiceEpisode] = []
    ep_map: dict[str, str] = {}
    for dos_key, ids in groups.items():
        eid = _episode_id(encounter_id, dos_key)
        signals = ["same_encounter_dos"] + (["linked_by_relationship"] if linked else [])
        episodes.append(ServiceEpisode(eid, encounter_id, dos, ids, signals))
        for i in ids:
            ep_map[i] = eid
    return episodes, ep_map


def _distinctness_facts(fact: ClinicalFact, relations: list) -> list[str]:
    """The DOCUMENTED facts that make a component potentially separate (a SEPARATE_FROM
    assertion, or a distinct site/session/objective attribute). Recorded, not acted on as
    a modifier -- Phase 1c/post-retrieval decides reportability."""
    out: list[str] = []
    for r in _relations_for(fact, relations):
        if r.predicate is RelationPredicate.SEPARATE_FROM and r.state is RelationState.ASSERTED:
            out.append(f"separate_from:{r.object_event_id}")
    a = fact.attributes or {}
    for axis in ("distinct_site", "distinct_session", "distinct_objective", "distinct_encounter"):
        if a.get(axis):
            out.append(f"{axis}:{a.get(axis)}")
    return out


def _service_key(intent: "ClaimLineIntent"):
    """Deterministic identity of 'the same performed service': kind + anatomy + laterality
    + the DISTINCTIVE action tokens. Used only to merge EXACT duplicates -- differently
    worded services keep separate intents (a merge can drop a line, so it is conservative)."""
    toks = tuple(sorted(t for t in re.split(r"[^a-z0-9]+", (intent.clinical_action or "").lower())
                        if len(t) > 3))
    a = intent.attributes or {}
    return (intent.fact_kind, str(a.get("anatomy", "")).lower(),
            str(a.get("laterality", "")).lower(), toks)


def _distinct_intents(a: "ClaimLineIntent", b: "ClaimLineIntent") -> bool:
    """Two same-key intents are DISTINCT services (must NOT merge) when documentation
    establishes distinctness: an explicit distinctness fact on either (a SEPARATE_FROM /
    distinct site/session), or a differing performer, approach, or site. A merge can drop a
    line, so ANY documented distinctness prohibits it. (Codex review F5.)"""
    if a.distinctness_facts or b.distinctness_facts:
        return True
    aa, ba = a.attributes or {}, b.attributes or {}
    for axis in ("performer_id", "performer", "approach", "distinct_site",
                 "distinct_session", "distinct_objective"):
        if str(aa.get(axis, "")).strip().lower() != str(ba.get(axis, "")).strip().lower():
            return True
    return False


def merge_duplicate_intents(intents: list["ClaimLineIntent"]) -> list["ClaimLineIntent"]:
    """same_episode_merge: multiple mentions of the SAME service (identical key, same
    episode + state, and NO documented distinctness) become ONE intent with accumulated
    evidence, distinctness facts, decisions, and mention count -- never multiple claim
    lines. An explicit SEPARATE_FROM / distinct site / differing performer PROHIBITS the
    merge, so an explicitly distinct service is never suppressed."""
    result: list = []
    for it in intents:
        target = None
        for m in result:
            if (m.service_episode_id == it.service_episode_id and m.state is it.state
                    and _service_key(m) == _service_key(it) and not _distinct_intents(m, it)):
                target = m
                break
        if target is None:
            result.append(it)
            continue
        target.clinical_event_ids = list(dict.fromkeys(
            list(target.clinical_event_ids) + list(it.clinical_event_ids)))
        target.source_span_ids = list(dict.fromkeys(
            list(target.source_span_ids) + list(it.source_span_ids)))
        target.distinctness_facts = list(dict.fromkeys(
            list(target.distinctness_facts) + list(it.distinctness_facts)))
        target.decisions = list(target.decisions) + list(it.decisions)
        target.mention_count += it.mention_count
    return result


def evaluate(facts: list[ClinicalFact], relations: list | None, encounter_id: str,
             date_of_service: str | None) -> list[ClaimLineIntent]:
    """Run the eligibility gates over every fact and produce a ClaimLineIntent for each.
    Service facts run the full service-line gate set; a diagnosis is a DIAGNOSIS_SUPPORT
    intent (eligible when billable) -- never a service line, never demoted by PART_OF."""
    relations = relations or []
    _episodes, _ep_map = build_episodes(facts, relations, encounter_id, date_of_service)
    intents: list[ClaimLineIntent] = []
    for f in facts:
        span_ids = [s.text_sha256 for s in (f.evidence or []) if getattr(s, "text_sha256", None)]
        billing_id = (f.attributes or {}).get("billing_entity_id")
        if f.kind is FactKind.DIAGNOSIS:
            decisions = [_gate_evidence_required(f), _gate_occurrence(f)]
            state = (EligibilityState.ELIGIBLE_FOR_RETRIEVAL
                     if all(d.outcome is Outcome.PASS for d in decisions)
                     else (EligibilityState.NON_CLAIM_EVIDENCE
                           if any(d.gate == "occurrence" and d.outcome is Outcome.BLOCKED
                                  for d in decisions)
                           else EligibilityState.AUTO_HOLD))
            component = ClaimComponent.DIAGNOSIS_SUPPORT
        elif f.kind in _SERVICE_KINDS:
            decisions = [_gate_evidence_required(f), _gate_occurrence(f),
                         _gate_actor_ownership(f), _gate_part_of_demotion(f, relations),
                         _gate_conflict(f, relations), _gate_documentation_minimum(f)]
            state = _classify(decisions)
            component = ClaimComponent.SERVICE
        else:
            continue                                   # E/M etc. handled elsewhere for now
        intents.append(ClaimLineIntent(
            intent_id=_intent_id(encounter_id, f.fact_id, f.description),
            encounter_id=encounter_id, component=component,
            clinical_event_ids=[f.fact_id] if f.fact_id else [],
            fact_kind=f.kind.value, clinical_action=f.description,
            attributes=dict(f.attributes or {}), date_of_service=date_of_service,
            billing_entity_id=billing_id, source_span_ids=span_ids,
            state=state, decisions=decisions,
            service_episode_id=_ep_map.get(f.fact_id),
            distinctness_facts=_distinctness_facts(f, relations)))
    return merge_duplicate_intents(intents)


def eligible_intents(intents: list[ClaimLineIntent]) -> list[ClaimLineIntent]:
    return [i for i in intents if i.state is EligibilityState.ELIGIBLE_FOR_RETRIEVAL]


def shadow_diff(facts: list[ClinicalFact], intents: list[ClaimLineIntent]) -> dict:
    """Compare the eligibility engine to TODAY'S behavior (performed==billable: every
    billable fact reaches retrieval). Categorizes each currently-retrieved fact by what
    the engine would do -- so divergence is measured on real notes BEFORE the engine is
    allowed to gate retrieval (Phase 1c). Purely observational.

      agree_eligible        both would send it to retrieval
      would_hold            engine holds it (unanchored evidence / contrary ownership /
                            unresolved relationship) that today bills
      would_suppress        engine marks it non-claim (not performed / explicitly integral)
                            that today bills
      eligible_not_billable engine would retrieve a fact today skips (should be empty)
    """
    billable_ids = {f.fact_id for f in facts if f.billable}
    state_by_fact: dict[str, str] = {}
    for it in intents:
        for eid in it.clinical_event_ids:
            state_by_fact[eid] = it.state.value
    agree, would_hold, would_suppress = [], [], []
    for f in facts:
        if f.fact_id not in billable_ids:
            continue
        st = state_by_fact.get(f.fact_id)
        if st == EligibilityState.ELIGIBLE_FOR_RETRIEVAL.value:
            agree.append(f.description)
        elif st == EligibilityState.AUTO_HOLD.value:
            would_hold.append(f.description)
        elif st == EligibilityState.NON_CLAIM_EVIDENCE.value:
            would_suppress.append(f.description)
    eligible_not_billable = [
        e for it in intents if it.state is EligibilityState.ELIGIBLE_FOR_RETRIEVAL
        for e in it.clinical_event_ids if e not in billable_ids]
    return {
        "billable_current": len(billable_ids),
        "agree_eligible": agree,
        "would_hold": would_hold,
        "would_suppress": would_suppress,
        "eligible_not_billable": eligible_not_billable,
        "divergent": bool(would_hold or would_suppress or eligible_not_billable),
    }


def summary(intents: list[ClaimLineIntent]) -> dict:
    """Shadow audit summary: counts by state + component, for diffing against the current
    performed==billable behavior."""
    out: dict[str, int] = {}
    for i in intents:
        out[i.state.value] = out.get(i.state.value, 0) + 1
    return {"total": len(intents), "by_state": out,
            "eligible": [i.clinical_action for i in eligible_intents(intents)]}
