"""Codex F6-R1 / F6-R2: extraction is fail-closed and actor identity comes only from context.

F6-R1 — invalid JSON, a malformed fact/relation, a blank/duplicate fact id must raise a
typed ExtractionSchemaError (the pipeline turns it into a retryable SYSTEM_HOLD with zero
retrieval), never silently drop a claim-affecting assertion.

F6-R2 — a model-supplied performer/organization id is trusted only when present in the
structured encounter context; an invented id (or no context at all) leaves ownership
unresolved so the service holds before retrieval.
"""
import json
import pytest
from claude_coder.extraction import extract_note, ExtractionSchemaError


def _stub(payload):
    return lambda system, user: (payload if isinstance(payload, str) else json.dumps(payload))


def _fact(**over):
    f = {"fact_id": "F1", "kind": "procedure", "description": "svc",
         "attributes": {}, "disposition": "performed_today", "certainty": "confirmed",
         "evidence": ["svc performed"], "confidence": 0.99,
         "axis_confidence": {"occurrence": 0.99, "action": 0.99, "evidence": 0.99,
                             "temporal": 0.99, "performer": 0.99, "relationship": 0.99}}
    f.update(over)
    return f


# ---------------------------------------------------------------- F6-R1 fail-closed schema
def test_invalid_json_raises_typed_error():
    with pytest.raises(ExtractionSchemaError):
        extract_note("note", _stub("this is not json at all"))


def test_malformed_fact_object_raises():
    with pytest.raises(ExtractionSchemaError):
        extract_note("note", _stub({"facts": ["not-an-object"]}))


def test_unrecognized_kind_raises_not_silently_dropped():
    with pytest.raises(ExtractionSchemaError):
        extract_note("note", _stub({"facts": [_fact(kind="not-a-kind")]}))


def test_blank_description_raises():
    with pytest.raises(ExtractionSchemaError):
        extract_note("note", _stub({"facts": [_fact(description="   ")]}))


def test_duplicate_fact_id_raises():
    with pytest.raises(ExtractionSchemaError):
        extract_note("note", _stub({"facts": [_fact(fact_id="X"), _fact(fact_id="X")]}))


def test_malformed_relation_raises_not_silently_dropped():
    payload = {"facts": [_fact(fact_id="F1"), _fact(fact_id="F2")],
               "relations": [{"predicate": "misspelled_part_of",
                              "subject_event_id": "F1", "object_event_id": "F2"}]}
    with pytest.raises(ExtractionSchemaError):
        extract_note("note", _stub(payload))


# ---------------------------------------------------------------- F6-R2 actor from context
def _person_ctx():
    # actor-1 is a person, context-designated performer, affiliated to org-1 (an organization)
    return {"billing_entity_id": "org-1", "participants": [
        {"id": "actor-1", "type": "person", "roles": ["performer"],
         "function": "operating surgeon", "affiliations": ["org-1"]},
        {"id": "org-1", "type": "organization"}]}


def _attrs(payload_attrs, ctx):
    res = extract_note("note", _stub({"facts": [_fact(attributes=payload_attrs)]}),
                       billing_context=ctx)
    return res.facts[0].attributes


def test_invented_actor_id_is_discarded():
    a = _attrs({"performer_id": "invented", "billing_entity_id": "x"}, _person_ctx())
    assert "performer_id" not in a and a["billing_entity_id"] == "org-1"


def test_authorized_person_performer_is_kept_with_context_function():
    a = _attrs({"performer_id": "actor-1", "performer_function": "janitor"}, _person_ctx())
    assert a["performer_id"] == "actor-1"
    assert a["performer_function"] == "operating surgeon"     # function from context, not model


def test_organization_id_cannot_be_used_as_performer():
    # the model relabels the billing organization as the performer -> rejected (type=org)
    a = _attrs({"performer_id": "org-1"}, _person_ctx())
    assert "performer_id" not in a


def test_unaffiliated_or_invented_organization_is_dropped():
    ctx = {"billing_entity_id": "org-1", "participants": [
        {"id": "actor-1", "type": "person", "roles": ["performer"], "affiliations": ["org-1"]},
        {"id": "org-1", "type": "organization"}, {"id": "org-2", "type": "organization"}]}
    a = _attrs({"performer_id": "actor-1", "organization_id": "org-2"}, ctx)  # not affiliated
    assert a["performer_id"] == "actor-1" and "organization_id" not in a


def test_invented_function_is_discarded_when_context_gives_none():
    ctx = {"billing_entity_id": "actor-1", "participants": [
        {"id": "actor-1", "type": "person", "roles": ["performer"]}]}
    a = _attrs({"performer_id": "actor-1", "performer_function": "chief surgeon"}, ctx)
    assert a["performer_id"] == "actor-1" and "performer_function" not in a


def test_missing_context_leaves_actor_unresolved():
    a = _attrs({"performer_id": "actor-1", "billing_entity_id": "actor-1"}, None)
    assert "performer_id" not in a


def test_organization_as_performer_reaches_retrieval_with_held_submission():
    # end-to-end through eligibility: an org-as-performer fact strips to unresolved
    # (never a contradiction -- nothing IS asserted once the invented id is discarded),
    # so per issue #6 item 7 it now reaches retrieval, but the resulting line is held.
    from claude_coder.eligibility import evaluate, EligibilityState
    from claude_coder.models import ClaimSubmissionStatus, Disposition, EvidenceSpan, FactKind
    res = extract_note("note", _stub({"facts": [_fact(
        attributes={"performer_id": "org-1"})]}), billing_context=_person_ctx())
    f = res.facts[0]
    f.disposition = Disposition.PERFORMED
    f.evidence = [EvidenceSpan("svc performed", anchored=True, text_sha256="h", span_id="s")]
    intents = evaluate([f], [], "enc", "2026-08-01")
    assert all(i.state is EligibilityState.ELIGIBLE_FOR_RETRIEVAL for i in intents)
    assert all(i.claim_submission_status is ClaimSubmissionStatus.HELD for i in intents)


# ------------------------------------------------- F6-R1 round 3: confidence is never coerced
# A JSON boolean passes Python's `isinstance(x, (int, float))` because bool subclasses int, so
# `"confidence": true` used to become 1.0 -- MAXIMUM confidence manufactured out of malformed
# output. Numeric strings, NaN/Infinity (which json.loads accepts by default) and out-of-range
# numbers were likewise coerced. Every one of them must raise instead.
# `None` (absent) is the ONE legal non-number and is covered separately below.
_BAD_CONFIDENCES = [True, False, "0.9", "high", float("nan"), float("inf"), float("-inf"),
                    -0.1, 1.5, [], {}]


@pytest.mark.parametrize("bad", _BAD_CONFIDENCES)
def test_malformed_scalar_confidence_raises(bad):
    with pytest.raises(ExtractionSchemaError):
        extract_note("note", _stub({"facts": [_fact(confidence=bad)]}))


@pytest.mark.parametrize("bad", _BAD_CONFIDENCES)
def test_malformed_axis_confidence_raises(bad):
    axes = dict(_fact()["axis_confidence"])
    axes["occurrence"] = bad
    with pytest.raises(ExtractionSchemaError):
        extract_note("note", _stub({"facts": [_fact(axis_confidence=axes)]}))


def test_malformed_unrequired_axis_confidence_also_raises():
    # an axis the kind does not require is still schema output: malformed is malformed
    axes = dict(_fact()["axis_confidence"])
    axes["assertion"] = True                            # not required for a procedure
    with pytest.raises(ExtractionSchemaError):
        extract_note("note", _stub({"facts": [_fact(axis_confidence=axes)]}))


@pytest.mark.parametrize("kind,axes", [
    ("supply", ("occurrence", "action", "evidence", "temporal", "performer", "relationship")),
    ("drug", ("occurrence", "action", "evidence", "temporal", "performer", "relationship")),
    ("diagnosis", ("occurrence", "action", "evidence", "temporal", "assertion", "experiencer")),
    ("imaging", ("occurrence", "action", "evidence", "temporal", "performer", "relationship")),
    ("evaluation_management",
     ("occurrence", "action", "evidence", "temporal", "performer", "relationship")),
])
def test_confidence_validation_covers_every_fact_kind(kind, axes):
    """No fact kind bypasses the confidence schema -- SUPPLY/DRUG included."""
    good = {a: 0.9 for a in axes}
    bad = dict(good, **{axes[0]: True})
    extract_note("note", _stub({"facts": [_fact(kind=kind, axis_confidence=good)]}))  # valid
    with pytest.raises(ExtractionSchemaError):
        extract_note("note", _stub({"facts": [_fact(kind=kind, confidence=True,
                                                    axis_confidence=good)]}))
    with pytest.raises(ExtractionSchemaError):
        extract_note("note", _stub({"facts": [_fact(kind=kind, axis_confidence=bad)]}))


@pytest.mark.parametrize("bad", _BAD_CONFIDENCES)
def test_malformed_relation_confidence_raises(bad):
    payload = {"facts": [_fact(fact_id="F1"), _fact(fact_id="F2")],
               "relations": [{"predicate": "reason_for", "subject_event_id": "F1",
                              "object_event_id": "F2", "state": "asserted",
                              "evidence_fact_ids": ["F1"], "confidence": bad}]}
    with pytest.raises(ExtractionSchemaError):
        extract_note("note", _stub(payload))


def test_absent_confidence_is_zero_not_maximum():
    """The one permitted non-number: an omitted confidence is 0.0 (fail-closed), never 1.0."""
    f = _fact()
    f.pop("confidence")
    res = extract_note("note", _stub({"facts": [f]}))
    assert res.facts[0].confidence == 0.0
    missing_axes = _fact()
    missing_axes["axis_confidence"] = {}
    res2 = extract_note("note", _stub({"facts": [missing_axes]}))
    assert set(res2.facts[0].axis_confidence.values()) == {0.0}


@pytest.mark.parametrize("bad", [True, False, 7, 1.5, ["nested"], {"no": "text"},
                                 {"text": True}, {"text": 7}, {"text": "  "}, None])
def test_malformed_evidence_element_raises(bad):
    """Adjacent instance of the same coercion class: a non-string quote must NOT be
    stringified into a pseudo-span (which would then fail ANCHORING for the wrong reason
    instead of failing the schema loudly)."""
    with pytest.raises(ExtractionSchemaError):
        extract_note("note", _stub({"facts": [_fact(evidence=[bad])]}))


def test_boundary_confidences_are_accepted():
    for value in (0, 1, 0.0, 1.0, 0.5):
        res = extract_note("note", _stub({"facts": [_fact(confidence=value)]}))
        assert res.facts[0].confidence == float(value)


# ------------------------------- F6-R2 round 3: explicit performer designation, strict schema
def test_roleless_person_is_not_an_authorized_performer():
    """The removed `or not prec["roles"]` wildcard: a KNOWN but non-designated person must
    never be elevated into the billing performer by the model's selection alone."""
    ctx = {"billing_entity_id": "actor-1", "participants": [
        {"id": "actor-1", "type": "person"}]}                    # no roles at all
    a = _attrs({"performer_id": "actor-1"}, ctx)
    assert "performer_id" not in a

    ctx_empty = {"billing_entity_id": "actor-1", "participants": [
        {"id": "actor-1", "type": "person", "roles": []}]}       # explicitly empty roles
    assert "performer_id" not in _attrs({"performer_id": "actor-1"}, ctx_empty)


def test_non_performer_role_is_not_an_authorized_performer():
    ctx = {"billing_entity_id": "org-1", "participants": [
        {"id": "actor-1", "type": "person", "roles": ["scribe", "supervisor"]},
        {"id": "org-1", "type": "organization"}]}
    assert "performer_id" not in _attrs({"performer_id": "actor-1"}, ctx)


def test_roleless_self_billing_person_reaches_retrieval_with_held_submission():
    """extraction -> eligibility -> ownership: the roleless person self-bills, so performer ==
    billing entity would have PASSED ownership. With no performer resolved, ownership is
    UNKNOWN -- unresolved, never a contradiction -- so per issue #6 item 7 the line now
    reaches eligibility for retrieval, but with submission held rather than ready."""
    from claude_coder.eligibility import evaluate, EligibilityState
    from claude_coder.models import ClaimSubmissionStatus, Disposition, EvidenceSpan, Outcome
    from claude_coder.ownership import classify_ownership, fact_ownership
    ctx = {"billing_entity_id": "actor-1", "participants": [
        {"id": "actor-1", "type": "person"}]}
    res = extract_note("note", _stub({"facts": [_fact(
        attributes={"performer_id": "actor-1"})]}), billing_context=ctx)
    f = res.facts[0]
    f.disposition = Disposition.PERFORMED
    f.evidence = [EvidenceSpan("svc performed", anchored=True, text_sha256="h", span_id="s")]
    own = fact_ownership(f)
    assert own.performer_id is None
    assert classify_ownership(own.performer_id, own.billing_entity_id,
                              own.organization_id, own.performer_function) is Outcome.UNKNOWN
    intents = evaluate([f], [], "enc", "2026-08-01")
    assert all(i.state is EligibilityState.ELIGIBLE_FOR_RETRIEVAL for i in intents)
    assert all(i.claim_submission_status is ClaimSubmissionStatus.HELD for i in intents)


@pytest.mark.parametrize("roles", [
    {"performer": True},                      # a MAPPING was iterated by key -> fake role
    "performer",                              # a bare string was iterated character-by-char
    [""], ["  "], [True], [1], [{"role": "performer"}], [None],
])
def test_malformed_role_container_is_rejected(roles):
    ctx = {"billing_entity_id": "actor-1", "participants": [
        {"id": "actor-1", "type": "person", "roles": roles}]}
    with pytest.raises(ExtractionSchemaError):
        extract_note("note", _stub({"facts": [_fact()]}), billing_context=ctx)


@pytest.mark.parametrize("affiliations", [{"org-1": True}, "org-1", [None], [3]])
def test_malformed_affiliation_container_is_rejected(affiliations):
    ctx = {"billing_entity_id": "org-1", "participants": [
        {"id": "actor-1", "type": "person", "roles": ["performer"],
         "affiliations": affiliations},
        {"id": "org-1", "type": "organization"}]}
    with pytest.raises(ExtractionSchemaError):
        extract_note("note", _stub({"facts": [_fact()]}), billing_context=ctx)


def test_duplicate_participant_id_is_rejected_not_last_write_wins():
    """Last-write-wins silently resolved a conflicting identity; a repeated id is now fatal."""
    ctx = {"billing_entity_id": "org-1", "participants": [
        {"id": "actor-1", "type": "person", "roles": ["scribe"]},
        {"id": "actor-1", "type": "person", "roles": ["performer"]},   # conflicting duplicate
        {"id": "org-1", "type": "organization"}]}
    with pytest.raises(ExtractionSchemaError):
        extract_note("note", _stub({"facts": [_fact(
            attributes={"performer_id": "actor-1"})]}), billing_context=ctx)


def test_duplicate_participant_id_with_type_conflict_is_rejected():
    ctx = {"billing_entity_id": "org-1", "participants": [
        {"id": "x", "type": "organization"},
        {"id": "x", "type": "person", "roles": ["performer"]}]}
    with pytest.raises(ExtractionSchemaError):
        extract_note("note", _stub({"facts": [_fact(
            attributes={"performer_id": "x"})]}), billing_context=ctx)


@pytest.mark.parametrize("participants", [
    "actor-1", {"id": "actor-1"}, ["actor-1"], [{"type": "person"}],
    [{"id": "", "type": "person"}], [{"id": "actor-1"}],
    [{"id": "actor-1", "type": "robot"}], [{"id": "actor-1", "type": ""}],
    [{"id": "actor-1", "type": "person", "function": 7}],
    [{"id": "actor-1", "type": "person", "function": "  "}],
])
def test_malformed_participant_shapes_are_rejected(participants):
    ctx = {"billing_entity_id": "org-1", "participants": participants}
    with pytest.raises(ExtractionSchemaError):
        extract_note("note", _stub({"facts": [_fact()]}), billing_context=ctx)


@pytest.mark.parametrize("ctx", [
    "not-a-context", [], {"participants": {}, "billing_entity_id": "b"},
    {"billing_entity_id": ""}, {"billing_entity_id": 7},
])
def test_malformed_billing_context_is_rejected(ctx):
    with pytest.raises(ExtractionSchemaError):
        extract_note("note", _stub({"facts": [_fact()]}), billing_context=ctx)


def test_malformed_context_fails_before_the_extraction_call():
    """A malformed roster can never produce trustworthy ownership -- it must fail closed
    BEFORE the model is asked anything."""
    calls = []

    def _llm(system, user):
        calls.append(user)
        return json.dumps({"facts": []})

    with pytest.raises(ExtractionSchemaError):
        extract_note("note", _llm, billing_context={"participants": [{"id": "a"}]})
    assert calls == []


# --------------------------------------------------------- F9-R5 attribute_evidence
def test_malformed_attribute_evidence_shape_raises():
    with pytest.raises(ExtractionSchemaError):
        extract_note("note", _stub({"facts": [
            _fact(attribute_evidence={"laterality": "not-a-list"})]}))


def test_empty_attribute_evidence_entry_raises():
    with pytest.raises(ExtractionSchemaError):
        extract_note("note", _stub({"facts": [
            _fact(attribute_evidence={"laterality": [{"text": "  ", "scope": "local"}]})]}))


def test_a_local_scope_entry_needs_no_parent_and_is_kept():
    result = extract_note("note", _stub({"facts": [_fact(
        attributes={"laterality": "right"},
        attribute_evidence={"laterality": [
            {"text": "performed on the right side", "scope": "local"}]})]}))
    entries = result.facts[0].attribute_evidence["laterality"]
    assert len(entries) == 1
    assert entries[0].scope == "local"
    assert entries[0].span.text == "performed on the right side"
    assert entries[0].source_relation_id == ""


# --------------------------------------------- F9-R6-R2 (5th) assertion_state
def test_assertion_state_asserted_parses_onto_the_entry():
    from claude_coder.models import RelationState
    result = extract_note("note", _stub({"facts": [_fact(
        attributes={"laterality": "right"},
        attribute_evidence={"laterality": [
            {"text": "performed on the right side", "scope": "local",
             "assertion_state": "asserted"}]})]}))
    entry = result.facts[0].attribute_evidence["laterality"][0]
    assert entry.assertion_state is RelationState.ASSERTED


def test_assertion_state_negated_parses_onto_the_entry():
    from claude_coder.models import RelationState
    result = extract_note("note", _stub({"facts": [_fact(
        attributes={"laterality": "right"},
        attribute_evidence={"laterality": [
            {"text": "right side was ultimately ruled out", "scope": "local",
             "assertion_state": "negated"}]})]}))
    entry = result.facts[0].attribute_evidence["laterality"][0]
    assert entry.assertion_state is RelationState.NEGATED


def test_missing_assertion_state_defaults_to_uncertain_never_asserted():
    """Fail-closed: an omitted judgement must never be read as a positive one --
    the same convention `_relation`'s own `state` field already uses."""
    from claude_coder.models import RelationState
    result = extract_note("note", _stub({"facts": [_fact(
        attributes={"laterality": "right"},
        attribute_evidence={"laterality": [
            {"text": "performed on the right side", "scope": "local"}]})]}))
    entry = result.facts[0].attribute_evidence["laterality"][0]
    assert entry.assertion_state is RelationState.UNCERTAIN


def test_invalid_assertion_state_raises_never_silently_coerced():
    with pytest.raises(ExtractionSchemaError):
        extract_note("note", _stub({"facts": [_fact(
            attribute_evidence={"laterality": [
                {"text": "performed on the right side", "scope": "local",
                 "assertion_state": "maybe-ish"}]})]}))


def test_an_inherited_entry_resolves_only_against_a_real_relation():
    """Codex F9-R5/F9-R5-A: an "inherited" claim is kept as a CANDIDATE only when a
    real, correctly-directed part_of relation actually connects this fact to the
    named parent -- naming a parent alone is never enough. `scope_validated` is not
    set here (extraction time, before reconciliation) -- see
    `provenance.validate_attribute_evidence` for the authoritative check."""
    payload = {
        "facts": [
            _fact(fact_id="F1", description="parent step",
                 evidence=["The parent step was performed on the right side"]),
            _fact(fact_id="F2", description="component step",
                 attributes={"laterality": "right"},
                 attribute_evidence={"laterality": [
                     {"text": "performed on the right side", "scope": "inherited",
                      "parent_fact_id": "F1"}]}),
        ],
        "relations": [{"subject_event_id": "F2", "predicate": "part_of",
                       "object_event_id": "F1", "state": "asserted"}],
    }
    result = extract_note("note", _stub(payload))
    component = next(f for f in result.facts if f.fact_id == "F2")
    entries = component.attribute_evidence["laterality"]
    assert len(entries) == 1
    assert entries[0].scope == "inherited"
    assert entries[0].span.text == "performed on the right side"
    rel = next(r for r in result.relations if r.subject_event_id == "F2")
    assert entries[0].source_relation_id == rel.relation_id


def test_an_inherited_entrys_assertion_state_survives_the_second_pass():
    """issue #6 F9-R6-R2, fifth re-review: assertion_state must thread through the
    SAME second-pass relation-resolution rebuild an inherited entry's scope/
    parent/source_relation_id already go through -- not silently reset to the
    dataclass default when the entry is re-constructed there."""
    from claude_coder.models import RelationState
    payload = {
        "facts": [
            _fact(fact_id="F1", description="parent step",
                 evidence=["The parent step was performed on the right side"]),
            _fact(fact_id="F2", description="component step",
                 attributes={"laterality": "right"},
                 attribute_evidence={"laterality": [
                     {"text": "performed on the right side", "scope": "inherited",
                      "parent_fact_id": "F1", "assertion_state": "negated"}]}),
        ],
        "relations": [{"subject_event_id": "F2", "predicate": "part_of",
                       "object_event_id": "F1", "state": "asserted"}],
    }
    result = extract_note("note", _stub(payload))
    component = next(f for f in result.facts if f.fact_id == "F2")
    entry = component.attribute_evidence["laterality"][0]
    assert entry.assertion_state is RelationState.NEGATED


# --------------------------------------------- F9-R6-R2 (6th) value binding
def test_value_parses_onto_the_entry():
    result = extract_note("note", _stub({"facts": [_fact(
        attributes={"laterality": "right"},
        attribute_evidence={"laterality": [
            {"text": "performed on the right side", "scope": "local",
             "assertion_state": "asserted", "value": "right"}]})]}))
    entry = result.facts[0].attribute_evidence["laterality"][0]
    assert entry.value == "right"


def test_missing_value_defaults_to_empty_string_never_the_axis_value():
    """Fail-closed: an entry that never names which value it proves must not be
    silently bound to whatever attributes[axis] happens to hold -- that is
    exactly the un-bound state graph_consensus.claim_authorized_value treats as
    proof of nothing."""
    result = extract_note("note", _stub({"facts": [_fact(
        attributes={"laterality": "right"},
        attribute_evidence={"laterality": [
            {"text": "performed on the right side", "scope": "local",
             "assertion_state": "asserted"}]})]}))
    entry = result.facts[0].attribute_evidence["laterality"][0]
    assert entry.value == ""


def test_a_boolean_value_raises_never_silently_coerced():
    with pytest.raises(ExtractionSchemaError):
        extract_note("note", _stub({"facts": [_fact(
            attribute_evidence={"laterality": [
                {"text": "performed on the right side", "scope": "local",
                 "value": True}]})]}))


def test_a_negated_entrys_value_is_the_value_it_negates():
    """A quote proving the note RULES OUT a value still names that ruled-out
    value in "value", paired with assertion_state=negated -- the negated
    entry's value is what it negates, not whatever the fact ultimately settled
    on for that axis."""
    from claude_coder.models import RelationState
    result = extract_note("note", _stub({"facts": [_fact(
        attributes={"laterality": "left"},
        attribute_evidence={"laterality": [
            {"text": "right involvement was ultimately ruled out", "scope": "local",
             "assertion_state": "negated", "value": "right"},
            {"text": "left side was addressed", "scope": "local",
             "assertion_state": "asserted", "value": "left"},
        ]})]}))
    entries = result.facts[0].attribute_evidence["laterality"]
    negated = next(e for e in entries if e.assertion_state is RelationState.NEGATED)
    asserted = next(e for e in entries if e.assertion_state is RelationState.ASSERTED)
    assert negated.value == "right"
    assert asserted.value == "left"


def test_an_inherited_entrys_value_survives_the_second_pass():
    payload = {
        "facts": [
            _fact(fact_id="F1", description="parent step",
                 evidence=["The parent step was performed on the right side"]),
            _fact(fact_id="F2", description="component step",
                 attributes={"laterality": "right"},
                 attribute_evidence={"laterality": [
                     {"text": "performed on the right side", "scope": "inherited",
                      "parent_fact_id": "F1", "assertion_state": "asserted",
                      "value": "right"}]}),
        ],
        "relations": [{"subject_event_id": "F2", "predicate": "part_of",
                       "object_event_id": "F1", "state": "asserted"}],
    }
    result = extract_note("note", _stub(payload))
    component = next(f for f in result.facts if f.fact_id == "F2")
    entry = component.attribute_evidence["laterality"][0]
    assert entry.value == "right"


def test_an_inherited_entry_with_no_matching_relation_is_dropped_not_kept_unproven():
    """The claimed parent exists as a fact, but NO part_of relation was actually
    emitted connecting the two -- the entry must be dropped entirely, never kept as
    an unproven inheritance claim."""
    payload = {
        "facts": [
            _fact(fact_id="F1", description="parent step"),
            _fact(fact_id="F2", description="component step",
                 attributes={"laterality": "right"},
                 attribute_evidence={"laterality": [
                     {"text": "performed on the right side", "scope": "inherited",
                      "parent_fact_id": "F1"}]}),
        ],
        "relations": [],
    }
    result = extract_note("note", _stub(payload))
    component = next(f for f in result.facts if f.fact_id == "F2")
    assert component.attribute_evidence == {}


def test_same_episode_as_is_never_a_candidate_relation_for_inheritance():
    """Issue #6 F9-R5-A: same_episode_as does not imply the same laterality/anatomy/
    product/count/approach -- only part_of may even become a CANDIDATE relation for
    an inherited attribute, regardless of direction."""
    payload = {
        "facts": [
            _fact(fact_id="F1", description="parent step"),
            _fact(fact_id="F2", description="component step",
                 attributes={"laterality": "right"},
                 attribute_evidence={"laterality": [
                     {"text": "performed on the right side", "scope": "inherited",
                      "parent_fact_id": "F1"}]}),
        ],
        "relations": [{"subject_event_id": "F2", "predicate": "same_episode_as",
                       "object_event_id": "F1", "state": "asserted"}],
    }
    result = extract_note("note", _stub(payload))
    component = next(f for f in result.facts if f.fact_id == "F2")
    assert component.attribute_evidence == {}


def test_a_reversed_part_of_direction_is_never_a_candidate_relation():
    """Issue #6 F9-R5-A: the relation exists, but runs the WRONG way (the named
    parent is asserted part_of the component instead of the reverse) -- must never
    even become a candidate, let alone validate later."""
    payload = {
        "facts": [
            _fact(fact_id="F1", description="parent step"),
            _fact(fact_id="F2", description="component step",
                 attributes={"laterality": "right"},
                 attribute_evidence={"laterality": [
                     {"text": "performed on the right side", "scope": "inherited",
                      "parent_fact_id": "F1"}]}),
        ],
        "relations": [{"subject_event_id": "F1", "predicate": "part_of",
                       "object_event_id": "F2", "state": "asserted"}],
    }
    result = extract_note("note", _stub(payload))
    component = next(f for f in result.facts if f.fact_id == "F2")
    assert component.attribute_evidence == {}


def test_an_inherited_entry_naming_an_unknown_parent_is_dropped():
    """An unknown parent_fact_id is not itself a schema error (a claimed parent that
    is not any fact's real id simply can never resolve against a relation) -- the
    entry is dropped, not kept and not a crash."""
    payload = {
        "facts": [_fact(fact_id="F1", description="component step",
                        attributes={"laterality": "right"},
                        attribute_evidence={"laterality": [
                            {"text": "right side", "scope": "inherited",
                             "parent_fact_id": "GHOST"}]})],
        "relations": [],
    }
    result = extract_note("note", _stub(payload))
    assert result.facts[0].attribute_evidence == {}


def test_inherited_scope_without_a_parent_fact_id_is_malformed():
    with pytest.raises(ExtractionSchemaError):
        extract_note("note", _stub({"facts": [_fact(
            attribute_evidence={"laterality": [
                {"text": "right side", "scope": "inherited"}]})]}))
