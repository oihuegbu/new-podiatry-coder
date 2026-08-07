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
def test_invented_actor_id_is_discarded():
    # context authorizes only real-actor; the model invents another -> stripped
    payload = {"facts": [_fact(attributes={"performer_id": "invented",
                                            "billing_entity_id": "invented"})]}
    ctx = {"billing_entity_id": "real-actor", "performer_id": "real-actor"}
    res = extract_note("note", _stub(payload), billing_context=ctx)
    attrs = res.facts[0].attributes
    assert "performer_id" not in attrs                       # invented id discarded
    assert attrs["billing_entity_id"] == "real-actor"        # entity always from context


def test_authorized_actor_id_is_kept():
    payload = {"facts": [_fact(attributes={"performer_id": "real-actor"})]}
    ctx = {"billing_entity_id": "real-actor", "performer_id": "real-actor"}
    res = extract_note("note", _stub(payload), billing_context=ctx)
    assert res.facts[0].attributes["performer_id"] == "real-actor"


def test_missing_context_leaves_actor_unresolved():
    payload = {"facts": [_fact(attributes={"performer_id": "actor-1",
                                           "billing_entity_id": "actor-1"})]}
    res = extract_note("note", _stub(payload))               # no billing_context
    assert "performer_id" not in res.facts[0].attributes     # cannot verify -> dropped
