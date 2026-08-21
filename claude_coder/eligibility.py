"""Enforced eligibility engine — decide which documented events may become a
code-search candidate BEFORE any code retrieval, producing a code-free ClaimLineIntent.

This is the plane that fixes `billable == performed`: a performed event is not
automatically a claim line. Each billable fact runs a sequence of TRI-STATE gates
(PASS / BLOCK / UNRESOLVED — never a silent default-to-eligible), and only an event that
clears them becomes an ELIGIBLE intent. Integral / not-performed / supporting events stay
in the record as NON_CLAIM_EVIDENCE; material ambiguity becomes AUTO_HOLD.

The pipeline treats ClaimLineIntent as a hard capability boundary: only an eligible
intent can construct a RetrievalRequest. Agnostic — reads fact kinds, dispositions,
actor ids, relationship assertions and evidence anchoring; no medical code.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .coreference import DISTINGUISHING_AXES as _DISTINCT_ATTR_AXES
from .models import (ClaimSubmissionStatus, ClinicalFact, FactKind, Outcome,
                     RelationPredicate, RelationState)
from .ownership import classify_ownership, fact_ownership

# `_DISTINCT_ATTR_AXES` keeps its historical name because the graph layer reads the
# eligibility engine's own conflict axes rather than restating them. It IS
# `coreference.DISTINGUISHING_AXES` -- one list, shared with claim assembly, so what
# makes two mentions two EVENTS and what makes a repeated code two OCCURRENCES can
# never drift apart (issue #6 F7-R3).


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
    # issue #6 item 7: a CODED state (ELIGIBLE_FOR_RETRIEVAL) and a SUBMITTABLE
    # claim are different questions. HELD only when `actor_ownership` was
    # specifically UNKNOWN (unresolved, never an affirmative contradiction -- that
    # still AUTO_HOLDs below exactly as before) and every other gate PASSED; see
    # `_classify`'s own comment for why this one axis is carved out narrowly rather
    # than repointing UNKNOWN generally.
    claim_submission_status: ClaimSubmissionStatus = ClaimSubmissionStatus.READY
    service_episode_id: str | None = None
    mention_count: int = 1
    distinctness_facts: list[str] = field(default_factory=list)
    # R7: canonical digest of the fact snapshot eligibility approved; re-checked before
    # retrieval so no retrieval/release-relevant field can be mutated afterward.
    fact_digest: str = ""


_SNAPSHOT_DIGEST_VERSION = "fsd-v3"   # v3: added governed_terms (issue #6 F7-R3-C4)


def fact_snapshot_digest(fact) -> str:
    """Canonical, versioned digest of EVERY retrieval/release-relevant field of a fact,
    captured at eligibility and re-verified before retrieval so nothing can change afterward:
    kind, clinical action, disposition, assertion certainty and experiencer, scalar and
    per-axis confidences (they set the post-retrieval autonomy floor), ALL attributes
    (measurements + actor ids live here), governed alternate wording (issue #6 F7-R3-C4:
    it changes what retrieval actually queries for, exactly like an attribute would), and
    ORDERED anchored evidence-span identity (order preserved so a reorder is detected).
    (Codex F6-R7.)"""
    evidence = [
        (str(getattr(sp, "span_id", "") or ""),
         str(getattr(sp, "text_sha256", "") or
             hashlib.sha256(str(getattr(sp, "text", "")).encode()).hexdigest()))
        for sp in (getattr(fact, "evidence", None) or [])]     # ORDER preserved
    governed_terms = {axis: sorted(alts) for axis, alts in
                      (getattr(fact, "governed_terms", None) or {}).items()}
    payload = {
        "v": _SNAPSHOT_DIGEST_VERSION,
        "kind": fact.kind.value,
        "description": fact.description,
        "disposition": getattr(getattr(fact, "disposition", None), "value", None),
        "certain": bool(getattr(fact, "certain", True)),
        "experiencer": getattr(fact, "experiencer", None),
        "confidence": getattr(fact, "confidence", None),
        "axis_confidence": dict(getattr(fact, "axis_confidence", None) or {}),
        "attributes": fact.attributes or {},
        "governed_terms": governed_terms,
        "evidence": evidence,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


@dataclass(frozen=True)
class RetrievalRequest:
    """The only accepted input to code retrieval.

    It binds one eligible intent to its source fact and validates that no caller can
    substitute a different event after eligibility was decided.
    """
    intent: ClaimLineIntent
    fact: ClinicalFact

    def __post_init__(self) -> None:
        if self.intent.state is not EligibilityState.ELIGIBLE_FOR_RETRIEVAL:
            raise ValueError("retrieval requires an ELIGIBLE ClaimLineIntent")
        if self.fact.fact_id not in self.intent.clinical_event_ids:
            raise ValueError("retrieval fact is not a member of the eligible intent")
        if self.fact.kind.value != self.intent.fact_kind:
            raise ValueError("retrieval fact kind differs from the eligible intent")
        if self.fact.description != self.intent.clinical_action:
            raise ValueError("retrieval action differs from the eligible intent")
        # R7: a retrieval-eligible intent MUST carry a non-empty canonical fact-snapshot
        # digest; a request built without one is rejected (no bypass via an empty digest).
        # The fact is then re-hashed and must match byte-for-byte -- any change to attributes,
        # measurements, actor ids, confidences, assertion/experiencer, disposition, or ordered
        # evidence identity rejects the request and yields zero retrieval. (Codex F6-R7.)
        if not self.intent.fact_digest:
            raise ValueError(
                "retrieval intent is missing its eligibility fact-snapshot digest")
        if fact_snapshot_digest(self.fact) != self.intent.fact_digest:
            raise ValueError(
                "retrieval fact mutated since eligibility (snapshot digest mismatch)")


def _intent_id(encounter_id: str, fact_id: str, action: str) -> str:
    raw = f"{encounter_id}|{fact_id}|{action}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


_SERVICE_KINDS = (FactKind.PROCEDURE, FactKind.IMAGING, FactKind.SUPPLY,
                  FactKind.DRUG, FactKind.EM)


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
    st = classify_ownership(o.performer_id, o.billing_entity_id,
                            o.organization_id, o.performer_function)
    if st is Outcome.BLOCKED:
        return EligibilityDecision("actor_ownership", Outcome.BLOCKED,
                                   "performed by a different actor than the billing entity",
                                   "claim ownership")
    if st is Outcome.UNKNOWN:
        return EligibilityDecision("actor_ownership", Outcome.UNKNOWN,
                                   "ownership identity/organization is not fully resolved",
                                   "claim ownership")
    return EligibilityDecision("actor_ownership", Outcome.PASS, "owned by billing entity",
                               "claim ownership")


_SYMMETRIC_PREDICATES = {RelationPredicate.SEPARATE_FROM,
                         RelationPredicate.SAME_EPISODE_AS}


def _relations_for(fact: ClinicalFact, relations: list) -> list:
    return [r for r in relations if r.subject_event_id == fact.fact_id
            or (r.predicate in _SYMMETRIC_PREDICATES
                and r.object_event_id == fact.fact_id)]


def _connects(r, left: str, right: str) -> bool:
    return {r.subject_event_id, r.object_event_id} == {left, right}


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
    distinct = [r for p in part_of for r in relations
                if r.predicate is RelationPredicate.SEPARATE_FROM
                and r.state is RelationState.ASSERTED
                and _connects(r, p.subject_event_id, p.object_event_id)]
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


def _gate_axis_consensus(fact: ClinicalFact) -> EligibilityDecision:
    """Two independent readings of the ORIGINAL DOCUMENT disagreed on a code-changing
    axis and the page itself could not settle it.

    That is a missing clinical FACT, not a coding judgement: the record does not state
    what the claim needs. UNKNOWN here holds the event before retrieval, and the
    pipeline converts THIS gate specifically into one precise provider question rather
    than a coder queue -- the product directive forbids routing an otherwise resolved
    claim to generic review because two models read differently.
    """
    conflicts = [str(c).strip() for c in (getattr(fact, "axis_conflicts", None) or [])
                 if str(c).strip()]
    if not conflicts:
        return EligibilityDecision("axis_consensus", Outcome.PASS,
                                   "no unsettled cross-reading axis",
                                   "source-settled fact axes")
    return EligibilityDecision("axis_consensus", Outcome.UNKNOWN, " ".join(conflicts),
                               "source-settled fact axes")


def _classify(decisions: list[EligibilityDecision]) -> EligibilityState:
    """Precedence: NON_CLAIM (the event should never have been a line) before AUTO_HOLD
    (material ambiguity) before ELIGIBLE. A silent default-to-eligible is impossible —
    every state is reached only by an explicit gate outcome.

    issue #6 item 7: `actor_ownership` UNKNOWN (unresolved -- e.g. an unstated
    performer/organization id) is deliberately EXCLUDED from the UNKNOWN-triggers-
    AUTO_HOLD set below, narrowly and only for this one gate. An affirmative
    ownership CONTRADICTION (`actor_ownership` BLOCKED -- performed by a
    documented, different actor than the billing entity) still AUTO_HOLDs exactly
    as before via the BLOCKED check above; only the "simply not yet resolved" case
    now reaches ELIGIBLE_FOR_RETRIEVAL. Every OTHER gate's UNKNOWN
    (`conflict`/`documentation_minimum`/`axis_consensus`) is untouched, so this
    cannot silently defeat a legitimate hold from any of them -- if actor_ownership
    AND one of those is UNKNOWN on the same event, the event still AUTO_HOLDs on
    the other gate's UNKNOWN. `evaluate()` separately stamps the resulting intent's
    `claim_submission_status` HELD for exactly this one case, so a line that now
    reaches retrieval on unresolved ownership is never indistinguishable from a
    fully clean one."""
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
           if g in ("conflict", "documentation_minimum", "axis_consensus")):
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
    """Deterministic identity of 'the same performed service': kind + anatomy +
    laterality + the action's DISTINCTIVE ROOT FORM.

    The action is compared through `coreference.action_form`, not through raw tokens:
    word order, inflection and documentation filler are how two writers spell one
    action, and treating them as two services is what let a re-described event become a
    second intent (issue #6 F7-R3). Merging on this key remains CONSERVATIVE -- a merge
    can drop a line, so it happens only where the two mentions genuinely resolve to one
    action form; anything less certain is settled after retrieval, against the
    authoritative code, by claim assembly.
    """
    from . import coreference as _coref
    toks = tuple(sorted(_coref.action_form(intent.clinical_action)))
    a = intent.attributes or {}
    return (intent.fact_kind, str(a.get("anatomy", "")).lower(),
            str(a.get("laterality", "")).lower(), toks)


def _known_known_conflict(a: "ClaimLineIntent", b: "ClaimLineIntent", source=None) -> bool:
    """Two intents conflict on a distinguishing attribute ONLY when BOTH carry a KNOWN
    value and they differ. Known-plus-missing does NOT conflict (it reconciles). (F5-R1.)

    The axis set is `coreference.DISTINGUISHING_AXES` -- the SAME one claim assembly
    reads when it decides whether a repeated code is a second occurrence. Two spellings
    of "what makes these two events different" is exactly how an event could be separate
    enough for two intents and identical enough to merge into extra units.

    `source` (issue #6 F7-R3-C4) is passed straight through, so this consults the same
    governed concept graph claim assembly does rather than raw string comparison alone.
    """
    from . import coreference as _coref
    return bool(_coref.known_known_differences(a.attributes, b.attributes, source=source))


def _reconcile_attrs(target: "ClaimLineIntent", src: "ClaimLineIntent") -> None:
    """Known-plus-missing reconciliation: fill a missing distinguishing attribute on the
    target from a known value on the source, without losing provenance."""
    for axis in _DISTINCT_ATTR_AXES:
        if not str(target.attributes.get(axis, "")).strip() and str(src.attributes.get(axis, "")).strip():
            target.attributes[axis] = src.attributes[axis]


def merge_duplicate_intents(intents: list["ClaimLineIntent"],
                            separate_pairs: set | None = None,
                            source=None) -> list["ClaimLineIntent"]:
    """same_episode_merge, PAIR-AWARE (Codex F5-R1): duplicate mentions (same key/episode/
    state) merge into ONE intent EXCEPT across a cannot-link -- an explicit SEPARATE_FROM
    (by event id) or a KNOWN-KNOWN attribute conflict. Cannot-links PROPAGATE one hop over
    coreference (a duplicate of an event inherits that event's separations), so an ambiguous
    duplicate adjacent to a SEPARATE_FROM is kept separate rather than merged into the wrong
    service. Known-plus-missing reconciles. No merged cluster contains both endpoints of a
    SEPARATE_FROM.

    `source` (issue #6 F7-R3-C4) is passed to the known-known conflict check."""
    n = len(intents)
    pairs = set(separate_pairs or set())

    def same_key(i, j):
        return (intents[i].service_episode_id == intents[j].service_episode_id
                and intents[i].state is intents[j].state
                and _service_key(intents[i]) == _service_key(intents[j]))

    def event_separated(i, j):
        for ea in intents[i].clinical_event_ids:
            for eb in intents[j].clinical_event_ids:
                if frozenset((ea, eb)) in pairs:
                    return True
        return False

    # initial intent-level cannot-link: known-known conflict OR an explicit event separation
    cannot = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if (_known_known_conflict(intents[i], intents[j], source)
                    or event_separated(i, j)):
                cannot[i][j] = cannot[j][i] = True

    # one-hop propagation over INITIAL coreference (same key, not initially cannot-linked):
    # a coref inherits its partner's separations. Symmetric -> order-independent; an
    # ambiguous duplicate between two mutually-separate services inherits BOTH -> isolated.
    base = [row[:] for row in cannot]
    ambiguous_coreference: set[int] = set()
    for i in range(n):
        for j in range(n):
            if i != j and same_key(i, j) and not base[i][j]:
                for k in range(n):
                    if base[i][k]:
                        cannot[j][k] = cannot[k][j] = True
                        ambiguous_coreference.add(j)

    # A third indistinguishable mention between explicitly separate endpoints cannot be
    # assigned to either service without inventing cardinality. It is evidence, not a
    # third retrieval line: hold that mention and preserve the two explicit endpoints.
    for idx in ambiguous_coreference:
        intent = intents[idx]
        intent.state = EligibilityState.AUTO_HOLD
        intent.decisions.append(EligibilityDecision(
            "coreference_assignment", Outcome.UNKNOWN,
            "duplicate mention cannot be assigned to either explicitly distinct event",
            "event identity reconciliation"))

    # union same-key, not-cannot-linked intents
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if same_key(i, j) and not cannot[i][j] and find(i) != find(j):
                parent[find(i)] = find(j)

    # build one merged intent per cluster, in first-seen order
    rep: dict = {}
    order: list = []
    for idx in range(n):
        r = find(idx)
        if r not in rep:
            rep[r] = intents[idx]
            order.append(r)
        else:
            t, src = rep[r], intents[idx]
            t.clinical_event_ids = list(dict.fromkeys(t.clinical_event_ids + src.clinical_event_ids))
            t.source_span_ids = list(dict.fromkeys(t.source_span_ids + src.source_span_ids))
            t.distinctness_facts = list(dict.fromkeys(t.distinctness_facts + src.distinctness_facts))
            t.decisions = list(t.decisions) + list(src.decisions)
            t.mention_count += src.mention_count
            _reconcile_attrs(t, src)

    # Codex F5-R1 re-review: an ambiguous duplicate -- a SINGLETON that had same-key
    # siblings but NO DIRECT cannot-link (no SEPARATE_FROM on its own events, no known-known
    # conflict) -- was isolated only by INHERITED separations and cannot be assigned to a
    # service. Keeping it ELIGIBLE would add an extra billable line (overbilling). Route it
    # to AUTO_HOLD so it can never create an additional retrieval/claim path; the two
    # explicit anchors stay eligible and separate. Never guess which service owns it.
    from collections import Counter as _Counter
    _size = _Counter(find(i) for i in range(n))
    for i in range(n):
        r = find(i)
        if _size[r] != 1:
            continue
        has_sib = any(j != i and same_key(i, j) for j in range(n))
        is_anchor = any(same_key(i, j) and base[i][j] for j in range(n))
        if has_sib and not is_anchor:
            amb = rep[r]
            amb.state = EligibilityState.AUTO_HOLD
            amb.decisions = list(amb.decisions) + [EligibilityDecision(
                "coreference", Outcome.UNKNOWN,
                "ambiguous duplicate mention: same-key to mutually-distinct services -- "
                "cannot assign; held so it cannot become an extra billable line",
                "coreference reconciliation")]
    return [rep[r] for r in order]


def _diagnosis_links(fact: ClinicalFact, relations: list) -> list[str]:
    """Service events this diagnosis justifies/supports (REASON_FOR, asserted).
    Makes the diagnosis-to-service linkage first-class (bound in the certificate via the
    decision) rather than a bare DIAGNOSIS_SUPPORT label. (Review gap: diagnoses remained
    independent retrieval candidates with no required service linkage.)"""
    out: list[str] = []
    for r in relations:
        if (r.subject_event_id == fact.fact_id
                and r.predicate is RelationPredicate.REASON_FOR
                and r.state is RelationState.ASSERTED):
            out.append(r.object_event_id)
    return list(dict.fromkeys(out))


def evaluate(facts: list[ClinicalFact], relations: list | None, encounter_id: str,
             date_of_service: str | None, source=None) -> list[ClaimLineIntent]:
    """Run the eligibility gates over every fact and produce a ClaimLineIntent for each.
    Service facts run the full service-line gate set; a diagnosis is a DIAGNOSIS_SUPPORT
    intent (eligible when billable) -- never a service line, never demoted by PART_OF.

    `source` (issue #6 F7-R3-C4) is passed to `merge_duplicate_intents`'s known-known
    conflict check."""
    relations = relations or []
    _episodes, _ep_map = build_episodes(facts, relations, encounter_id, date_of_service)
    intents: list[ClaimLineIntent] = []
    for f in facts:
        span_ids = [s.span_id for s in (f.evidence or []) if getattr(s, "span_id", None)]
        billing_id = (f.attributes or {}).get("billing_entity_id")
        submission_status = ClaimSubmissionStatus.READY
        if f.kind is FactKind.DIAGNOSIS:
            core = [_gate_evidence_required(f), _gate_occurrence(f),
                    _gate_axis_consensus(f)]
            state = (EligibilityState.ELIGIBLE_FOR_RETRIEVAL
                     if all(d.outcome is Outcome.PASS for d in core)
                     else (EligibilityState.NON_CLAIM_EVIDENCE
                           if any(d.gate == "occurrence" and d.outcome is Outcome.BLOCKED
                                  for d in core)
                           else EligibilityState.AUTO_HOLD))
            # Record the diagnosis-to-service linkage (REASON_FOR / SUPPORTS) as a
            # first-class decision bound in the certificate. RECORDED, NOT BLOCKING -- an
            # encounter diagnosis without an explicit documented link is still codeable (it
            # establishes necessity), and relations are LLM-extracted and often incomplete;
            # a downstream necessity gate can consult these links.
            _links = _diagnosis_links(f, relations)
            decisions = core + [EligibilityDecision(
                "diagnosis_linkage",
                Outcome.PASS if _links else Outcome.UNKNOWN,
                (f"supports service event(s): {', '.join(_links)}" if _links
                 else "no documented REASON_FOR link to a service (recorded)"),
                "diagnosis-to-service linkage")]
            component = ClaimComponent.DIAGNOSIS_SUPPORT
        elif f.kind in _SERVICE_KINDS:
            decisions = [_gate_evidence_required(f), _gate_occurrence(f),
                         _gate_actor_ownership(f), _gate_part_of_demotion(f, relations),
                         _gate_conflict(f, relations), _gate_documentation_minimum(f),
                         _gate_axis_consensus(f)]
            state = _classify(decisions)
            component = ClaimComponent.SERVICE
            # issue #6 item 7: only when actor_ownership's own UNKNOWN is why this
            # event reached ELIGIBLE_FOR_RETRIEVAL at all (see `_classify`'s
            # comment) -- a fact that is eligible with an unrelated PASS ownership
            # decision stays READY.
            if (state is EligibilityState.ELIGIBLE_FOR_RETRIEVAL
                    and any(d.gate == "actor_ownership" and d.outcome is Outcome.UNKNOWN
                            for d in decisions)):
                submission_status = ClaimSubmissionStatus.HELD
        else:
            continue
        intents.append(ClaimLineIntent(
            intent_id=_intent_id(encounter_id, f.fact_id, f.description),
            encounter_id=encounter_id, component=component,
            clinical_event_ids=[f.fact_id] if f.fact_id else [],
            fact_kind=f.kind.value, clinical_action=f.description,
            claim_submission_status=submission_status,
            attributes=dict(f.attributes or {}), date_of_service=date_of_service,
            billing_entity_id=billing_id, source_span_ids=span_ids,
            state=state, decisions=decisions,
            service_episode_id=_ep_map.get(f.fact_id),
            distinctness_facts=_distinctness_facts(f, relations),
            fact_digest=fact_snapshot_digest(f)))
    _separate_pairs = {frozenset((r.subject_event_id, r.object_event_id))
                       for r in relations
                       if r.predicate is RelationPredicate.SEPARATE_FROM
                       and r.state is RelationState.ASSERTED}
    return merge_duplicate_intents(intents, _separate_pairs, source)


#: Decisions RECORDED for the audit trail that never determine the eligibility state.
#: Defined by exclusion on purpose: every other gate -- including one added tomorrow --
#: counts as blocking, so the failure mode is over-escalation, never a silent downgrade.
NON_BLOCKING_GATES = frozenset({"diagnosis_linkage"})


#: WHO must act on an unresolved eligibility decision, in the product directive's
#: routing vocabulary (section 8). This exists because "a gate did not clear" says
#: nothing about whether the work is a coder's, the provider's, or the system's --
#: and a hold with no declared owner falls through to a generic coder queue, which
#: is exactly the fallback the directive forbids.
#:
#: The owner is keyed on (gate, outcome) because the same gate means different things
#: at different outcomes: ownership that CONTRADICTS the billing entity is an
#: integrity state; ownership that was never RESOLVED is context-resolver work.
#:
#: These are the directive's destination NAMES rather than models.Destination members
#: so this module keeps its one-way dependency on models (facts, not routing); the
#: pipeline translates them.
OWNER_PROVIDER_QUERY = "PROVIDER_QUERY"   # the record itself lacks a code-changing fact
OWNER_SYSTEM = "SYSTEM"                   # a dependency did not resolve -> retry
OWNER_INTEGRITY = "INTEGRITY"             # inconsistent/unverifiable -> BLOCKED
OWNER_NON_CLAIM = "NON_CLAIM"             # the event is simply not claim-eligible
OWNER_CODER = "CODER"                     # irreducible coding/clinical judgement

_HOLD_OWNERS: dict[tuple[str, Outcome], str] = {
    # The event is not an independent claim line at all.
    ("occurrence", Outcome.BLOCKED): OWNER_NON_CLAIM,
    ("part_of_demotion", Outcome.BLOCKED): OWNER_NON_CLAIM,
    # Nothing downstream can be verified against a fact with no anchored evidence, or
    # one billed by an entity the record says did not perform it.
    ("evidence_required", Outcome.BLOCKED): OWNER_INTEGRITY,
    ("actor_ownership", Outcome.BLOCKED): OWNER_INTEGRITY,
    # An event whose clinical action is empty is an unusable graph node -- neither a
    # coder nor the provider has anything to act on.
    ("documentation_minimum", Outcome.UNKNOWN): OWNER_INTEGRITY,
    # The performer/billing identity was never bound. That is the Encounter Context
    # Resolver's job, so it is a retry, never a coder queue.
    ("actor_ownership", Outcome.UNKNOWN): OWNER_SYSTEM,
    # Every remaining hold has the SAME shape: two readings of the original document
    # disagreed about a code-changing fact, or the document never states it, and the
    # page could not settle it. The answer exists only in the provider's head, so it
    # becomes one precise question -- the directive names model disagreement
    # explicitly as something that must NOT reach a coder.
    ("conflict", Outcome.UNKNOWN): OWNER_PROVIDER_QUERY,
    ("axis_consensus", Outcome.UNKNOWN): OWNER_PROVIDER_QUERY,
    ("coreference_assignment", Outcome.UNKNOWN): OWNER_PROVIDER_QUERY,
    ("coreference", Outcome.UNKNOWN): OWNER_PROVIDER_QUERY,
}


def hold_owner(decision: EligibilityDecision) -> str:
    """Who must act on this unresolved decision (see _HOLD_OWNERS).

    An UNDECLARED (gate, outcome) falls back to OWNER_CODER: over-escalation, the same
    failure direction NON_BLOCKING_GATES chooses. tests/test_validation_ladder.py pins
    the declaration so a new gate cannot reach production still relying on that
    fallback -- it is a safety net, not the routing design.
    """
    return _HOLD_OWNERS.get((decision.gate, decision.outcome), OWNER_CODER)


def declared_hold_gates() -> frozenset[str]:
    """Gate names carrying at least one declared owner -- the coverage guard's input."""
    return frozenset(g for g, _ in _HOLD_OWNERS)


def blocking_decisions(intent: ClaimLineIntent) -> list[EligibilityDecision]:
    """The decisions that actually produced this intent's state.

    A caller asking "why is this held?" must not be answered with decisions that were
    recorded but changed nothing -- that is how a single-cause hold gets mistaken for a
    multi-cause one and routed to the wrong destination.
    """
    return [d for d in intent.decisions
            if d.outcome is not Outcome.PASS and d.gate not in NON_BLOCKING_GATES]


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
