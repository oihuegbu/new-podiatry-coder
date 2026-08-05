"""Phase-1 eligibility engine: tri-state gates + code-free ClaimLineIntent.

The core safety property — a performed event is NOT automatically a claim line:
an explicitly integral component is demoted to NON_CLAIM_EVIDENCE BEFORE retrieval, a
non-performed event never becomes a line, and material ambiguity holds. Agnostic —
synthetic facts/relations, no medical code."""
from claude_coder import eligibility as el
from claude_coder.eligibility import EligibilityState, ClaimComponent
from claude_coder.models import (ClinicalFact, FactKind, Disposition, EvidenceSpan,
                                 Outcome, RelationAssertion, RelationPredicate, RelationState)


def _fact(kind=FactKind.PROCEDURE, desc="a performed service", anchored=True,
          disp=Disposition.PERFORMED, fid="F1", attrs=None):
    ev = [EvidenceSpan(text="quote", anchored=anchored, text_sha256="h1")]
    return ClinicalFact(kind, desc, attributes=attrs or {}, disposition=disp,
                        evidence=ev, fact_id=fid, confidence=0.9)


def _partof(subj="F1", obj="F2", state=RelationState.ASSERTED):
    return RelationAssertion(subject_event_id=subj, predicate=RelationPredicate.PART_OF,
                             object_event_id=obj, state=state)


# ------------------------------------------------------------------ gates
def test_evidence_gate_tristate():
    assert el._gate_evidence_required(_fact(anchored=True)).outcome is Outcome.PASS
    assert el._gate_evidence_required(_fact(anchored=False)).outcome is Outcome.BLOCKED
    bare = ClinicalFact(FactKind.PROCEDURE, "svc", evidence=[], fact_id="F1")
    assert el._gate_evidence_required(bare).outcome is Outcome.BLOCKED


def test_occurrence_gate():
    assert el._gate_occurrence(_fact(disp=Disposition.PERFORMED)).outcome is Outcome.PASS
    assert el._gate_occurrence(_fact(disp=Disposition.ORDERED)).outcome is Outcome.BLOCKED
    assert el._gate_occurrence(_fact(disp=Disposition.HISTORICAL)).outcome is Outcome.BLOCKED


def test_ownership_gate():
    assert el._gate_actor_ownership(_fact()).outcome is Outcome.UNKNOWN            # unstated
    same = _fact(attrs={"performer_id": "p1", "billing_entity_id": "p1"})
    assert el._gate_actor_ownership(same).outcome is Outcome.PASS
    other = _fact(attrs={"performer_id": "p2", "billing_entity_id": "p1"})
    assert el._gate_actor_ownership(other).outcome is Outcome.BLOCKED


def test_part_of_demotion_requires_explicit_integrality():
    f = _fact(fid="F1")
    assert el._gate_part_of_demotion(f, []).outcome is Outcome.PASS                 # no relation
    assert el._gate_part_of_demotion(f, [_partof()]).outcome is Outcome.BLOCKED      # explicit
    # weak/uncertain PART_OF does NOT demote (defers to the conflict gate)
    assert el._gate_part_of_demotion(f, [_partof(state=RelationState.UNCERTAIN)]).outcome \
        is Outcome.PASS
    # documented distinctness overrides the demotion
    sep = RelationAssertion("F1", RelationPredicate.SEPARATE_FROM, "F2",
                            state=RelationState.ASSERTED)
    assert el._gate_part_of_demotion(f, [_partof(), sep]).outcome is Outcome.PASS


def test_conflict_gate_holds_on_uncertain_material_relation():
    f = _fact(fid="F1")
    assert el._gate_conflict(f, [_partof(state=RelationState.UNCERTAIN)]).outcome is Outcome.UNKNOWN
    assert el._gate_conflict(f, [_partof()]).outcome is Outcome.PASS


# ------------------------------------------------------------------ classification
def _one(facts, relations=None):
    return el.evaluate(facts, relations or [], "enc", "2026-08-01")[0]


def test_clean_service_is_eligible():
    i = _one([_fact()])
    assert i.state is EligibilityState.ELIGIBLE_FOR_RETRIEVAL
    assert i.component is ClaimComponent.SERVICE
    assert i.clinical_action == "a performed service"
    # the intent carries NO code
    assert not hasattr(i, "code") and "code" not in i.attributes


def test_explicit_integral_component_demoted_before_retrieval():
    """The billable==performed fix: an event documented as part of another is NON_CLAIM,
    so retrieval never searches a code for it."""
    i = _one([_fact(fid="F1")], relations=[_partof("F1", "F2")])
    assert i.state is EligibilityState.NON_CLAIM_EVIDENCE


def test_non_performed_event_is_non_claim():
    assert _one([_fact(disp=Disposition.ORDERED)]).state is EligibilityState.NON_CLAIM_EVIDENCE


def test_unanchored_evidence_holds():
    assert _one([_fact(anchored=False)]).state is EligibilityState.AUTO_HOLD


def test_uncertain_relationship_holds():
    i = _one([_fact(fid="F1")], relations=[_partof("F1", "F2", RelationState.UNCERTAIN)])
    assert i.state is EligibilityState.AUTO_HOLD


def test_contrary_ownership_holds():
    i = _one([_fact(attrs={"performer_id": "p2", "billing_entity_id": "p1"})])
    assert i.state is EligibilityState.AUTO_HOLD


def test_diagnosis_is_support_not_service_line():
    i = _one([_fact(kind=FactKind.DIAGNOSIS, desc="a documented condition")])
    assert i.component is ClaimComponent.DIAGNOSIS_SUPPORT
    assert i.state is EligibilityState.ELIGIBLE_FOR_RETRIEVAL
    # a diagnosis is NOT demoted by a PART_OF relationship (not a service line)
    j = el.evaluate([_fact(kind=FactKind.DIAGNOSIS, desc="dx", fid="D1")],
                    [_partof("D1", "F2")], "enc", "2026-08-01")[0]
    assert j.state is EligibilityState.ELIGIBLE_FOR_RETRIEVAL


def test_summary_counts_by_state():
    intents = el.evaluate(
        [_fact(fid="F1"), _fact(fid="F2", disp=Disposition.ORDERED),
         _fact(fid="F3", anchored=False)], [], "enc", "2026-08-01")
    s = el.summary(intents)
    assert s["total"] == 3
    assert s["by_state"][EligibilityState.ELIGIBLE_FOR_RETRIEVAL.value] == 1
    assert s["by_state"][EligibilityState.NON_CLAIM_EVIDENCE.value] == 1
    assert s["by_state"][EligibilityState.AUTO_HOLD.value] == 1


# ------------------------------------------------------------------ Phase 1b
from claude_coder.eligibility import ServiceEpisode  # noqa: E402


def test_episode_groups_same_dos_and_is_attached():
    intents = el.evaluate([_fact(fid="F1"), _fact(fid="F2", desc="another service")],
                          [], "enc", "2026-08-01")
    eids = {i.service_episode_id for i in intents}
    assert len(eids) == 1 and None not in eids                 # one episode, attached


def test_episode_grouping_does_not_suppress_a_line():
    """SAME_EPISODE_AS groups for context but must NOT demote (grouping != integrality)."""
    same_ep = RelationAssertion("F1", RelationPredicate.SAME_EPISODE_AS, "F2",
                                state=RelationState.ASSERTED)
    intents = el.evaluate([_fact(fid="F1"), _fact(fid="F2", desc="another service")],
                          [same_ep], "enc", "2026-08-01")
    assert all(i.state is EligibilityState.ELIGIBLE_FOR_RETRIEVAL for i in intents)
    assert len(intents) == 2                                    # both survive, not merged


def test_identical_service_mentions_merge_to_one_intent():
    """same_episode_merge: the same service mentioned twice -> ONE intent, mention_count 2,
    unioned evidence -- never two claim lines."""
    intents = el.evaluate([_fact(fid="F1"), _fact(fid="F2")], [], "enc", "2026-08-01")
    assert len(intents) == 1
    m = intents[0]
    assert m.mention_count == 2
    assert set(m.clinical_event_ids) == {"F1", "F2"}


def test_distinct_services_do_not_merge():
    a = _fact(fid="F1", attrs={"anatomy": "calcaneus", "laterality": "right"})
    b = _fact(fid="F2", attrs={"anatomy": "calcaneus", "laterality": "left"})   # diff laterality
    intents = el.evaluate([a, b], [], "enc", "2026-08-01")
    assert len(intents) == 2                                    # different key -> not merged


def test_documented_distinctness_recorded_and_overrides_demotion():
    sep = RelationAssertion("F1", RelationPredicate.SEPARATE_FROM, "F2",
                            state=RelationState.ASSERTED)
    partof = RelationAssertion("F1", RelationPredicate.PART_OF, "F2",
                               state=RelationState.ASSERTED)
    i = el.evaluate([_fact(fid="F1")], [partof, sep], "enc", "2026-08-01")[0]
    assert i.state is EligibilityState.ELIGIBLE_FOR_RETRIEVAL   # distinctness overrides part_of
    assert any("separate_from" in d for d in i.distinctness_facts)


def test_build_episodes_signals():
    eps, ep_map = el.build_episodes([_fact(fid="F1")], [], "enc", "2026-08-01")
    assert len(eps) == 1 and "same_encounter_dos" in eps[0].grouping_signals
    assert ep_map["F1"] == eps[0].episode_id


# ------------------------------------------------------------------ Phase 1c shadow-diff
def test_shadow_diff_agrees_when_no_relations():
    facts = [_fact(fid="F1"), _fact(fid="F2", desc="another service")]
    d = el.shadow_diff(facts, el.evaluate(facts, [], "enc", "2026-08-01"))
    assert d["divergent"] is False
    assert len(d["agree_eligible"]) == 2 and not d["would_hold"] and not d["would_suppress"]


def test_shadow_diff_flags_hold_on_unanchored_evidence():
    facts = [_fact(fid="F1", anchored=False)]        # billable today, engine holds it
    d = el.shadow_diff(facts, el.evaluate(facts, [], "enc", "2026-08-01"))
    assert d["divergent"] is True and len(d["would_hold"]) == 1 and not d["would_suppress"]


def test_shadow_diff_flags_suppress_on_explicit_integral():
    facts = [_fact(fid="F1")]
    rels = [_partof("F1", "F2")]                     # explicitly integral -> engine suppresses
    d = el.shadow_diff(facts, el.evaluate(facts, rels, "enc", "2026-08-01"))
    assert d["divergent"] is True and len(d["would_suppress"]) == 1


# ---- Codex review F5: distinctness-aware dedup ----
def test_separate_from_prohibits_merge_of_identical_events():
    """Two otherwise-identical events with an explicit SEPARATE_FROM must NOT merge -- a
    merge would suppress a documented distinct service."""
    sep = RelationAssertion("F1", RelationPredicate.SEPARATE_FROM, "F2",
                            state=RelationState.ASSERTED)
    intents = el.evaluate([_fact(fid="F1"), _fact(fid="F2")], [sep], "enc", "2026-08-01")
    assert len(intents) == 2


def test_differing_performer_prohibits_merge():
    a = _fact(fid="F1", attrs={"performer_id": "p1"})
    b = _fact(fid="F2", attrs={"performer_id": "p2"})
    assert len(el.evaluate([a, b], [], "enc", "2026-08-01")) == 2


def test_true_duplicate_merge_unions_decisions_and_mentions():
    single = el.evaluate([_fact(fid="F1")], [], "enc", "2026-08-01")[0]
    both = el.evaluate([_fact(fid="F1"), _fact(fid="F2")], [], "enc", "2026-08-01")[0]
    assert both.mention_count == 2
    assert len(both.decisions) == 2 * len(single.decisions)          # both mentions retained


# ---- Codex review F5-R1: pair-aware, propagating cannot-link + known-plus-missing --------
def _in_one_intent(intents, a, b):
    return any(a in i.clinical_event_ids and b in i.clinical_event_ids for i in intents)


def test_attribute_distinguished_third_event_isolated_duplicates_merge():
    """Common case: F1/F2 share a performer, F3 has a DIFFERENT known performer. F3 is
    isolated (known-known conflict); F1/F2 merge into one intent."""
    fs = [_fact(fid="F1", attrs={"performer_id": "p1"}),
          _fact(fid="F2", attrs={"performer_id": "p1"}),
          _fact(fid="F3", attrs={"performer_id": "p3"})]
    intents = el.evaluate(fs, [], "enc", "2026-08-01")
    assert _in_one_intent(intents, "F1", "F2")                 # duplicates merged
    assert not _in_one_intent(intents, "F1", "F3")
    assert not _in_one_intent(intents, "F2", "F3")             # distinct service not collapsed


def test_separate_from_never_collapses_a_distinct_service():
    """Codex F5-R1 reproduction: F1/F2/F3 same key, F1 SEPARATE_FROM F3. No cluster may
    contain both endpoints of the SEPARATE_FROM, and the ambiguous duplicate F2 must NOT be
    merged into the service explicitly distinct from F1."""
    sep = RelationAssertion("F1", RelationPredicate.SEPARATE_FROM, "F3",
                            state=RelationState.ASSERTED)
    fs = [_fact(fid="F1"), _fact(fid="F2"), _fact(fid="F3")]
    intents = el.evaluate(fs, [sep], "enc", "2026-08-01")
    assert not _in_one_intent(intents, "F1", "F3")             # SEPARATE_FROM endpoints apart
    assert not _in_one_intent(intents, "F2", "F3")             # duplicate not attached to F3


def test_known_plus_missing_reconciles_not_splits():
    """A known performer on one duplicate and a MISSING performer on the other must
    reconcile into one intent (not two), filling the missing value."""
    fs = [_fact(fid="F1", attrs={"performer_id": "p1"}), _fact(fid="F2", attrs={})]
    intents = el.evaluate(fs, [], "enc", "2026-08-01")
    assert len(intents) == 1 and intents[0].mention_count == 2
    assert intents[0].attributes.get("performer_id") == "p1"   # reconciled, provenance kept
