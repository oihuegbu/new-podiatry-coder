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


def test_organization_as_performer_holds_through_eligibility():
    # end-to-end through eligibility: an org-as-performer fact must NOT become eligible
    from claude_coder.eligibility import evaluate, EligibilityState
    from claude_coder.models import Disposition, EvidenceSpan, FactKind
    res = extract_note("note", _stub({"facts": [_fact(
        attributes={"performer_id": "org-1"})]}), billing_context=_person_ctx())
    f = res.facts[0]
    f.disposition = Disposition.PERFORMED
    f.evidence = [EvidenceSpan("svc performed", anchored=True, text_sha256="h", span_id="s")]
    intents = evaluate([f], [], "enc", "2026-08-01")
    assert all(i.state is not EligibilityState.ELIGIBLE_FOR_RETRIEVAL for i in intents)
