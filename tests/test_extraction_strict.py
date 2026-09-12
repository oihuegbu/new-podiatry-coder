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


# ------------------------------------------------------------ F9-R11-E malformed-draw retry
# A malformed-shape response is a single bad draw from the model, not a deterministic
# property of the note or the prompt (the same premise the vision extraction call's own
# retry loop already acts on) -- so extract_note must retry a schema failure instead of
# turning one bad draw into an encounter-wide SYSTEM_HOLD. Verified live: an operative-note
# run's extraction batch returned a fact whose 'attribute_evidence' was not an object,
# holding the whole encounter at pre_retrieval_integrity with zero diagnosis/service lines.
def _sequenced_llm(*payloads):
    """One entry per call, in order -- proves retry behavior, not just single-shot shape."""
    calls = iter(payloads)
    call_count = []

    def _llm(system, user):
        call_count.append(1)
        payload = next(calls)
        return payload if isinstance(payload, str) else json.dumps(payload)

    _llm.call_count = call_count
    return _llm


def test_a_malformed_draw_is_retried_and_recovers_on_a_later_attempt():
    good = {"facts": [_fact()]}
    llm = _sequenced_llm("not json at all", good)
    result = extract_note("note", llm)
    assert len(result.facts) == 1 and result.facts[0].fact_id == "F1"
    assert len(llm.call_count) == 2


def test_a_persistently_malformed_draw_still_raises_after_bounded_retries():
    llm = _sequenced_llm("bad 1", "bad 2", "bad 3")
    with pytest.raises(ExtractionSchemaError):
        extract_note("note", llm)
    assert len(llm.call_count) == 3        # bounded -- never retried a 4th time


# ------------------------------------------------- F9-R13-C release-gate root cause 1
# issue #6, Codex's independent re-review: a schema-VALID response whose "attributes"
# and "attribute_evidence" do not actually agree with each other used to reach
# resolution as a silently unresolved axis instead of retrying -- the prompt already
# documents this contract, `_parse_extraction_response` just never enforced it.
def test_an_emitted_attribute_without_matching_evidence_retries_extraction():
    """An axis in "attributes" with no same-name, same-value "attribute_evidence"
    entry (here: the entry's own value is the WRONG side) is a malformed draw, not a
    genuinely undocumented axis -- it must retry exactly like any other schema
    defect, and recover once a later draw actually binds the claimed value."""
    bad = {"facts": [_fact(
        attributes={"laterality": "right"},
        attribute_evidence={"laterality": [
            {"text": "performed on the right side", "scope": "local",
             "value": "left"}]})]}
    good = {"facts": [_fact(
        attributes={"laterality": "right"},
        attribute_evidence={"laterality": [
            {"text": "performed on the right side", "scope": "local",
             "value": "right"}]})]}
    llm = _sequenced_llm(bad, good)
    result = extract_note("note", llm)
    assert len(llm.call_count) == 2
    assert result.facts[0].attributes["laterality"] == "right"
    assert result.facts[0].attribute_evidence["laterality"][0].value == "right"


def test_orphan_attribute_evidence_retries_extraction():
    """An "attribute_evidence" entry naming an axis absent from "attributes" is
    equally a malformed contract -- the model claims to have evidence for
    something it never actually emitted a value for."""
    bad = {"facts": [_fact(
        attributes={},
        attribute_evidence={"laterality": [
            {"text": "performed on the right side", "scope": "local",
             "value": "right"}]})]}
    good = {"facts": [_fact(attributes={})]}
    llm = _sequenced_llm(bad, good)
    result = extract_note("note", llm)
    assert len(llm.call_count) == 2
    assert result.facts[0].attribute_evidence == {}


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
            {"text": "performed on the right side", "scope": "local",
             "value": "right"}]})]}))
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
             "assertion_state": "asserted", "value": "right"}]})]}))
    entry = result.facts[0].attribute_evidence["laterality"][0]
    assert entry.assertion_state is RelationState.ASSERTED


def test_assertion_state_negated_parses_onto_the_entry():
    from claude_coder.models import RelationState
    result = extract_note("note", _stub({"facts": [_fact(
        attributes={"laterality": "right"},
        attribute_evidence={"laterality": [
            {"text": "right side was ultimately ruled out", "scope": "local",
             "assertion_state": "negated", "value": "right"}]})]}))
    entry = result.facts[0].attribute_evidence["laterality"][0]
    assert entry.assertion_state is RelationState.NEGATED


def test_missing_assertion_state_defaults_to_uncertain_never_asserted():
    """Fail-closed: an omitted judgement must never be read as a positive one --
    the same convention `_relation`'s own `state` field already uses."""
    from claude_coder.models import RelationState
    result = extract_note("note", _stub({"facts": [_fact(
        attributes={"laterality": "right"},
        attribute_evidence={"laterality": [
            {"text": "performed on the right side", "scope": "local",
             "value": "right"}]})]}))
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
                      "parent_fact_id": "F1", "value": "right"}]}),
        ],
        "relations": [{"subject_event_id": "F2", "predicate": "part_of",
                       "object_event_id": "F1", "state": "asserted",
                       "evidence_fact_ids": ["F1", "F2"]}],
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
                      "parent_fact_id": "F1", "assertion_state": "negated",
                      "value": "right"}]}),
        ],
        "relations": [{"subject_event_id": "F2", "predicate": "part_of",
                       "object_event_id": "F1", "state": "asserted",
                       "evidence_fact_ids": ["F1", "F2"]}],
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
        attributes={"laterality": ""},
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
                       "object_event_id": "F1", "state": "asserted",
                       "evidence_fact_ids": ["F1", "F2"]}],
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
                      "parent_fact_id": "F1", "value": "right"}]}),
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
                      "parent_fact_id": "F1", "value": "right"}]}),
        ],
        "relations": [{"subject_event_id": "F2", "predicate": "same_episode_as",
                       "object_event_id": "F1", "state": "asserted",
                       "evidence_fact_ids": ["F1", "F2"]}],
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
                      "parent_fact_id": "F1", "value": "right"}]}),
        ],
        "relations": [{"subject_event_id": "F1", "predicate": "part_of",
                       "object_event_id": "F2", "state": "asserted",
                       "evidence_fact_ids": ["F1", "F2"]}],
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
                             "parent_fact_id": "GHOST", "value": "right"}]})],
        "relations": [],
    }
    result = extract_note("note", _stub(payload))
    assert result.facts[0].attribute_evidence == {}


def test_inherited_scope_without_a_parent_fact_id_is_malformed():
    with pytest.raises(ExtractionSchemaError):
        extract_note("note", _stub({"facts": [_fact(
            attribute_evidence={"laterality": [
                {"text": "right side", "scope": "inherited"}]})]}))


# ------------------------------------------------- F9-R11-F strict wire extraction contract
# `attributes`/`attribute_evidence`/`axis_confidence` are dynamic-key maps that neither
# provider's structured-output grammar can express (`additionalProperties` must be
# explicitly `false` on every object -- confirmed live: a bare `{"type": "object"}` 400s,
# and closing it with no declared properties is accepted but then the grammar can ONLY ever
# produce `{}`, discarding whatever the model was asked to put there). The wire contract
# below represents each dynamic map as a closed array of named entries instead;
# `wire_to_legacy_extraction_json` converts it back to the exact legacy dict shape every
# OTHER test in this file exercises directly via `_stub`/`_fact` -- this section tests ONLY
# the wire<->legacy boundary and the two production callers, never the domain model.
from claude_coder.extraction import (        # noqa: E402
    EXTRACTION_WIRE_SCHEMA, wire_to_legacy_extraction_json, _default_llm,
    default_second_extract_llm,
)
import app.core.llm_client as _llm_client
import app.core.config as _llm_config


def _wire_fact(**over):
    f = {"fact_id": "F1", "kind": "procedure", "description": "svc",
         "attributes": {"strings": [], "numbers": [], "booleans": []},
         "attribute_evidence": [], "disposition": "performed_today",
         "negated": False, "certainty": "confirmed", "experiencer": "patient",
         "evidence": ["svc performed"], "confidence": 0.99,
         "axis_confidence": [{"name": a, "confidence": 0.99} for a in
                             ("occurrence", "action", "evidence", "temporal",
                              "performer", "relationship")]}
    f.update(over)
    return f


def _wire_payload(facts, relations=None):
    return {"schema_version": "extraction-wire-v1", "facts": facts,
           "relations": relations or []}


def test_both_extraction_callers_request_the_strict_wire_schema():
    """Primary AND second-reading extraction must both be grammar-constrained -- an
    unwired caller is exactly the production gap F9-R11-F reported (only
    json_mode=True, no json_schema, on either)."""
    seen = {}
    orig = _llm_client.chat_completion
    _llm_client.chat_completion = lambda s, u, **kw: (
        seen.update(kw), (json.dumps(_wire_payload([])), {}))[1]
    try:
        _default_llm("sys", "user")
        assert seen.get("json_schema") == EXTRACTION_WIRE_SCHEMA

        seen.clear()
        orig_provider, orig_openai_model = (
            _llm_config.LLM_PROVIDER, _llm_config.OPENAI_MODEL)
        _llm_config.LLM_PROVIDER = "claude"
        _llm_config.OPENAI_MODEL = "gpt-x"
        try:
            default_second_extract_llm("sys", "user")
        finally:
            _llm_config.LLM_PROVIDER, _llm_config.OPENAI_MODEL = (
                orig_provider, orig_openai_model)
        assert seen.get("json_schema") == EXTRACTION_WIRE_SCHEMA
    finally:
        _llm_client.chat_completion = orig


def test_wire_adapter_flattens_typed_attribute_arrays():
    wire = {"strings": [{"name": "laterality", "value": "right"}],
           "numbers": [{"name": "depth_mm", "value": 3}],
           "booleans": [{"name": "new_patient", "value": True}]}
    legacy = wire_to_legacy_extraction_json(
        _wire_payload([_wire_fact(attributes=wire)]))["facts"][0]
    assert legacy["attributes"] == {
        "laterality": "right", "depth_mm": 3, "new_patient": True}


def test_wire_adapter_rejects_duplicate_name_across_type_arrays():
    wire = {"strings": [{"name": "count", "value": "two"}],
           "numbers": [{"name": "count", "value": 2}], "booleans": []}
    with pytest.raises(ExtractionSchemaError):
        wire_to_legacy_extraction_json(_wire_payload([_wire_fact(attributes=wire)]))


def test_wire_adapter_rejects_duplicate_name_within_one_type_array():
    wire = {"strings": [{"name": "anatomy", "value": "heel"},
                        {"name": "anatomy", "value": "ankle"}],
           "numbers": [], "booleans": []}
    with pytest.raises(ExtractionSchemaError):
        wire_to_legacy_extraction_json(_wire_payload([_wire_fact(attributes=wire)]))


def test_wire_adapter_rejects_duplicate_axis_confidence_name():
    axes = [{"name": "occurrence", "confidence": 0.9},
           {"name": "occurrence", "confidence": 0.5}]
    with pytest.raises(ExtractionSchemaError):
        wire_to_legacy_extraction_json(_wire_payload([_wire_fact(axis_confidence=axes)]))


def test_wire_adapter_rejects_blank_attribute_evidence_name():
    ev = [{"name": "  ", "text": "quote", "scope": "local", "parent_fact_id": "",
          "assertion_state": "asserted", "value": "right"}]
    with pytest.raises(ExtractionSchemaError):
        wire_to_legacy_extraction_json(_wire_payload([_wire_fact(attribute_evidence=ev)]))


def test_wire_adapter_groups_attribute_evidence_by_name_not_rejected_as_duplicate():
    """Unlike attributes/axis_confidence, several attribute_evidence entries naming the
    SAME axis are expected (several quotes bearing on one value) -- must group, never
    reject as a duplicate."""
    ev = [{"name": "laterality", "text": "right heel", "scope": "local",
          "parent_fact_id": "", "assertion_state": "asserted", "value": "right"},
         {"name": "laterality", "text": "right-sided procedure", "scope": "local",
          "parent_fact_id": "", "assertion_state": "asserted", "value": "right"}]
    legacy = wire_to_legacy_extraction_json(
        _wire_payload([_wire_fact(attribute_evidence=ev)]))["facts"][0]
    assert len(legacy["attribute_evidence"]["laterality"]) == 2


def test_wire_inherited_evidence_survives_the_existing_relation_validation_path():
    """End-to-end proof through the UNCHANGED domain model: a wire response with an
    inherited-scope attribute_evidence entry, backed by a real part_of relation, must
    resolve through extract_note exactly like the legacy shape already does (see
    test_an_inherited_value_resolves_against_its_part_of_relation above)."""
    parent = _wire_fact(fact_id="F1", attributes={
        "strings": [{"name": "laterality", "value": "right"}],
        "numbers": [], "booleans": []},
        attribute_evidence=[{"name": "laterality", "text": "right heel",
                             "scope": "local", "parent_fact_id": "",
                             "assertion_state": "asserted", "value": "right"}])
    child = _wire_fact(fact_id="F2", description="component",
                       attributes={"strings": [{"name": "laterality", "value": "right"}],
                                  "numbers": [], "booleans": []},
                       attribute_evidence=[{"name": "laterality",
                                            "text": "the component step",
                                            "scope": "inherited",
                                            "parent_fact_id": "F1",
                                            "assertion_state": "asserted",
                                            "value": "right"}])
    relations = [{"subject_event_id": "F2", "predicate": "part_of",
                 "object_event_id": "F1", "state": "asserted",
                 "evidence_fact_ids": ["F1", "F2"], "confidence": 0.9}]
    legacy = wire_to_legacy_extraction_json(_wire_payload([parent, child], relations))
    result = extract_note("note", _stub(legacy))
    child_fact = next(f for f in result.facts if f.fact_id == "F2")
    assert child_fact.attribute_evidence["laterality"][0].scope == "inherited"
    assert child_fact.attribute_evidence["laterality"][0].parent_fact_id == "F1"


def test_second_reading_and_primary_produce_identical_legacy_shape_from_equivalent_wire():
    """Parity: both callers share the SAME adapter, so an equivalent wire response
    converts to an identical legacy fact regardless of which caller produced it."""
    wire = _wire_payload([_wire_fact(attributes={
        "strings": [{"name": "laterality", "value": "right"}],
        "numbers": [], "booleans": []})])
    orig = _llm_client.chat_completion
    _llm_client.chat_completion = lambda s, u, **kw: (json.dumps(wire), {})
    try:
        primary_out = _default_llm("sys", "user")
        orig_provider, orig_openai_model = (
            _llm_config.LLM_PROVIDER, _llm_config.OPENAI_MODEL)
        _llm_config.LLM_PROVIDER = "claude"
        _llm_config.OPENAI_MODEL = "gpt-x"
        try:
            second_out = default_second_extract_llm("sys", "user")
        finally:
            _llm_config.LLM_PROVIDER, _llm_config.OPENAI_MODEL = (
                orig_provider, orig_openai_model)
    finally:
        _llm_client.chat_completion = orig
    assert json.loads(primary_out) == json.loads(second_out)


def test_wire_level_malformation_retries_through_extract_note_and_recovers():
    """A wire-adapter rejection (duplicate name across arrays -- something the
    provider's grammar itself cannot forbid) must retry through extract_note's
    bounded loop exactly like a parser-level rejection, not escape it (the `llm(...)`
    call is now INSIDE the try/except for exactly this reason)."""
    bad = _wire_payload([_wire_fact(attributes={
        "strings": [{"name": "count", "value": "two"}],
        "numbers": [{"name": "count", "value": 2}], "booleans": []})])
    good = _wire_payload([_wire_fact()])
    calls = []

    def _llm(system, user):
        calls.append(1)
        wire = bad if len(calls) < 2 else good
        return json.dumps(wire_to_legacy_extraction_json(wire))

    result = extract_note("note", _llm)
    assert len(result.facts) == 1 and len(calls) == 2


def test_wire_level_malformation_persists_still_bounds_and_raises():
    """Never an empty silent success -- persistent wire malformation raises after
    exactly _EXTRACTION_MAX_ATTEMPTS, matching F9-R11-E's own bound."""
    bad = _wire_payload([_wire_fact(attributes={
        "strings": [{"name": "count", "value": "two"}],
        "numbers": [{"name": "count", "value": 2}], "booleans": []})])
    calls = []

    def _llm(system, user):
        calls.append(1)
        return json.dumps(wire_to_legacy_extraction_json(bad))

    with pytest.raises(ExtractionSchemaError):
        extract_note("note", _llm)
    assert len(calls) == 3


# ------------------------------------------------ F9-R11-G relation/fact_id identity
def test_relation_with_unknown_subject_is_rejected_inside_extract_note():
    """Codex F9-R11-G: an identity defect must raise from extract_note's OWN retry
    loop, not only much later in provenance.validate_relations (a RelationIntegrityError
    that loop never sees)."""
    payload = {"facts": [_fact(fact_id="F1")],
              "relations": [{"subject_event_id": "GHOST", "predicate": "part_of",
                             "object_event_id": "F1", "state": "asserted",
                             "evidence_fact_ids": ["F1"], "confidence": 0.9}]}
    with pytest.raises(ExtractionSchemaError):
        extract_note("note", _stub(payload))


def test_relation_with_unknown_object_is_rejected_inside_extract_note():
    payload = {"facts": [_fact(fact_id="F1")],
              "relations": [{"subject_event_id": "F1", "predicate": "part_of",
                             "object_event_id": "GHOST", "state": "asserted",
                             "evidence_fact_ids": ["F1"], "confidence": 0.9}]}
    with pytest.raises(ExtractionSchemaError):
        extract_note("note", _stub(payload))


def test_self_referential_relation_is_rejected():
    payload = {"facts": [_fact(fact_id="F1")],
              "relations": [{"subject_event_id": "F1", "predicate": "part_of",
                             "object_event_id": "F1", "state": "asserted",
                             "evidence_fact_ids": ["F1"], "confidence": 0.9}]}
    with pytest.raises(ExtractionSchemaError):
        extract_note("note", _stub(payload))


def test_empty_evidence_fact_ids_is_rejected():
    payload = {"facts": [_fact(fact_id="F1"), _fact(fact_id="F2")],
              "relations": [{"subject_event_id": "F1", "predicate": "part_of",
                             "object_event_id": "F2", "state": "asserted",
                             "evidence_fact_ids": [], "confidence": 0.9}]}
    with pytest.raises(ExtractionSchemaError):
        extract_note("note", _stub(payload))


def test_evidence_fact_ids_naming_an_unknown_id_is_rejected():
    payload = {"facts": [_fact(fact_id="F1"), _fact(fact_id="F2")],
              "relations": [{"subject_event_id": "F1", "predicate": "part_of",
                             "object_event_id": "F2", "state": "asserted",
                             "evidence_fact_ids": ["F1", "GHOST"], "confidence": 0.9}]}
    with pytest.raises(ExtractionSchemaError):
        extract_note("note", _stub(payload))


def test_relation_to_a_negated_fact_is_rejected_before_pipeline_provenance():
    """Codex F9-R11-G item 3: a relation naming a fact the model itself marked
    negated is an invalid retained graph -- caught here, never a silently-dropped
    edge and never left to fail only downstream in provenance.validate_relations."""
    payload = {"facts": [_fact(fact_id="F1"), _fact(fact_id="F2", negated=True)],
              "relations": [{"subject_event_id": "F1", "predicate": "part_of",
                             "object_event_id": "F2", "state": "asserted",
                             "evidence_fact_ids": ["F1"], "confidence": 0.9}]}
    with pytest.raises(ExtractionSchemaError):
        extract_note("note", _stub(payload))


def test_relation_to_a_ruled_out_fact_is_rejected_before_pipeline_provenance():
    payload = {"facts": [_fact(fact_id="F1"),
                        _fact(fact_id="F2", certainty="ruled_out")],
              "relations": [{"subject_event_id": "F1", "predicate": "part_of",
                             "object_event_id": "F2", "state": "asserted",
                             "evidence_fact_ids": ["F1"], "confidence": 0.9}]}
    with pytest.raises(ExtractionSchemaError):
        extract_note("note", _stub(payload))


def test_missing_fact_id_is_rejected_never_autogenerated():
    """Never a fallback f"F{i+1}" for a missing fact_id -- that fallback made this
    exact check unreachable and could desynchronize model relation references from
    retained event identity (Codex F9-R11-G adjacent defect)."""
    bad = _fact()
    bad.pop("fact_id")
    with pytest.raises(ExtractionSchemaError):
        extract_note("note", _stub({"facts": [bad]}))


def test_blank_fact_id_is_rejected_never_autogenerated():
    with pytest.raises(ExtractionSchemaError):
        extract_note("note", _stub({"facts": [_fact(fact_id="   ")]}))


def test_a_dangling_relation_retries_with_categorical_feedback_and_recovers():
    """Codex F9-R11-G items 5/6: a relation-identity defect retries through the SAME
    bounded loop as a malformed-shape response. The second attempt's prompt carries
    fixed, CATEGORICAL guidance -- never the raw exception text (which could quote
    the dangling id back) and never a repeat of the note -- and a valid second draw
    succeeds."""
    from claude_coder import extraction as _ext
    calls = []
    good_payload = {"facts": [_fact(fact_id="F1")], "relations": []}
    bad_payload = {"facts": [_fact(fact_id="F1")],
                   "relations": [{"subject_event_id": "F1", "predicate": "part_of",
                                  "object_event_id": "GHOST", "state": "asserted",
                                  "evidence_fact_ids": ["F1"], "confidence": 0.9}]}

    def flaky(system, user):
        calls.append(user)
        return json.dumps(bad_payload if len(calls) == 1 else good_payload)

    result = extract_note("note", flaky)
    assert len(calls) == 2
    assert result.facts[0].fact_id == "F1"
    first, second = json.loads(calls[0]), json.loads(calls[1])
    assert "validation_feedback" not in first
    assert second.get("validation_feedback") == _ext._RETRY_VALIDATION_FEEDBACK
    assert "GHOST" not in second["validation_feedback"]
    assert second["note"] == first["note"]                 # note is not repeated/altered


def test_three_dangling_relation_responses_stay_bounded_and_fail_closed():
    bad_payload = {"facts": [_fact(fact_id="F1")],
                   "relations": [{"subject_event_id": "F1", "predicate": "part_of",
                                  "object_event_id": "GHOST", "state": "asserted",
                                  "evidence_fact_ids": ["F1"], "confidence": 0.9}]}
    calls = []

    def always_bad(system, user):
        calls.append(1)
        return json.dumps(bad_payload)

    with pytest.raises(ExtractionSchemaError):
        extract_note("note", always_bad)
    assert len(calls) == 3


def test_a_valid_relation_graph_is_unchanged_by_the_new_identity_checks():
    """Sanity: a normal, well-formed graph is unaffected by F9-R11-G -- not a
    regression in disguise."""
    payload = {
        "facts": [_fact(fact_id="F1", description="parent step"),
                 _fact(fact_id="F2", description="component step")],
        "relations": [{"subject_event_id": "F2", "predicate": "part_of",
                       "object_event_id": "F1", "state": "asserted",
                       "evidence_fact_ids": ["F1", "F2"], "confidence": 0.9}],
    }
    result = extract_note("note", _stub(payload))
    assert len(result.facts) == 2
    assert len(result.relations) == 1
    rel = result.relations[0]
    assert rel.subject_event_id == "F2" and rel.object_event_id == "F1"
