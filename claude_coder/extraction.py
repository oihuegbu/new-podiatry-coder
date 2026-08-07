"""Stage 1 — Clinical Language Understanding (fact extraction).

The model reads the note and emits STRUCTURED CLINICAL FACTS with verbatim
evidence — and nothing else. It is never asked for, and must never output, a
medical code. This is the deliberate inversion: the LLM does the genuinely
LLM-shaped job (understanding messy prose, negation, laterality, whether a thing
was performed vs merely discussed), and the deterministic layer downstream does
the code assignment from authoritative data.

Because the prompt carries no codes, it cannot go stale when the code sets
change, and the hardcoding guard has nothing to catch here.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any
from typing import Callable

from .models import (ClinicalFact, Disposition, EvidenceSpan, FactKind,
                     RelationAssertion, RelationPredicate, RelationState)

# A callable (system_prompt, user_prompt) -> JSON string. Injectable for tests.
LLMFn = Callable[[str, str], str]

_SYSTEM = """You are a clinical language understanding engine for medical coding.
Read the clinical note and extract every DISTINCT billable clinical event as a
structured fact. You describe WHAT HAPPENED in plain clinical language — you do
NOT assign or output any billing codes of any kind.

For each fact return an object with:
  - "fact_id": a unique local id such as F1; relations use these exact ids
  - "kind": one of procedure | diagnosis | supply | drug | imaging |
            evaluation_management
  - "description": a precise clinical phrase for the event (no codes)
  - "attributes": the axes that determine specificity, when documented —
        anatomy, laterality (left/right/bilateral), count/quantity, depth,
        area/size, product/material, drug + dose + wasted amount, approach,
        contrast, technical_vs_professional. Omit what the note does not state;
        never infer laterality, count, or site that is not written.
        For performed services also capture actor participation using ONLY ids supplied
        in encounter_context: performer_id, performer_function, organization_id, and
        billing_entity_id. Never invent an id or equate a person with an organization.
        For an evaluation_management fact, also give the medical-decision-making
        elements when documented: "problems", "data", "risk" each as one of
        straightforward | low | moderate | high, plus "new_patient" (true/false),
        "setting" (office | emergency | inpatient | observation | nursing | home —
        from the place of service / note header, default office for a clinic),
        "total_time_minutes" if the note records visit time, and
        "separately_identifiable" (true only if the note documents E/M work
        significant and separate from any procedure done the same day).
  - "disposition": performed_today | ordered | planned | discussed |
        historical | unclear  — ONLY performed_today / dispensed work is billable.
        For a PROCEDURE/supply/drug this is whether it was actually done today.
        For a DIAGNOSIS, use performed_today for a CURRENT/active condition
        addressed at this encounter (this is the default for anything in the
        assessment/impression); use historical ONLY when the note frames it as
        past — "history of", "resolved", "status post", or listed under past
        medical history.
  - "negated": true if the note denies/rules out this finding, else false
  - "certainty": confirmed | suspected | ruled_out — a probable/possible/likely/
        working/rule-out/differential condition is "suspected" and, per outpatient
        coding rules, must NOT be coded as if confirmed; "confirmed" for a
        definitively documented condition/finding; "ruled_out" for one the note
        excludes. Default confirmed only when the note states the condition plainly.
  - "experiencer": patient | family | other — whose condition/finding this is; a
        family-history or other-person mention is NOT the patient's coded condition.
  - "evidence": a list of VERBATIM quotes copied exactly from the note that
        support this fact (never paraphrased)
  - "confidence": 0.0-1.0, your certainty this event is documented as stated
  - "axis_confidence": confidence for each required extraction axis. Always emit
        occurrence, action, evidence, temporal; for diagnoses also assertion and
        experiencer; for services also performer and relationship. Missing/unclear is
        0.0, never omitted or averaged away.

Also return "relations": documented edges between facts. Each has
subject_event_id, predicate (part_of | used_in | reason_for | same_episode_as |
separate_from), object_event_id, state (asserted | negated | uncertain),
evidence_fact_ids (facts whose verbatim evidence supports the edge), and confidence.
Do not infer integrality or distinctness from clinical convention; emit only what the
note documents. PART_OF/USED_IN/REASON_FOR are directional; SEPARATE_FROM and
SAME_EPISODE_AS are symmetric.

Rules: quote evidence verbatim; separate a planned/ordered service from a
performed one; capture negation; do not merge distinct events; do not invent
facts the note does not support. For a DIAGNOSIS, the "description" must be the
concise clinical name of ONE condition — when a note phrase lists several
conditions together, emit a SEPARATE diagnosis fact for each, and keep severity
prose, counts, and functional-limitation wording OUT of the description (put
them in attributes or omit). Return JSON only:
{"facts": [ ... ], "relations": [ ... ]}."""


@dataclass
class ExtractionResult:
    facts: list[ClinicalFact] = field(default_factory=list)
    relations: list[RelationAssertion] = field(default_factory=list)
    schema_version: str = "clinical-graph-v1"


_REQUIRED_AXES: dict[FactKind, tuple[str, ...]] = {
    FactKind.DIAGNOSIS: ("occurrence", "action", "evidence", "temporal",
                         "assertion", "experiencer"),
    FactKind.PROCEDURE: ("occurrence", "action", "evidence", "temporal",
                         "performer", "relationship"),
    FactKind.SUPPLY: ("occurrence", "action", "evidence", "temporal",
                      "performer", "relationship"),
    FactKind.DRUG: ("occurrence", "action", "evidence", "temporal",
                    "performer", "relationship"),
    FactKind.IMAGING: ("occurrence", "action", "evidence", "temporal",
                       "performer", "relationship"),
    FactKind.EM: ("occurrence", "action", "evidence", "temporal",
                  "performer", "relationship"),
}


def _coerce_kind(value: str) -> FactKind | None:
    try:
        return FactKind(str(value).strip().lower())
    except ValueError:
        return None


def _coerce_disposition(value) -> Disposition:
    # Fail-closed: a missing (None) or unrecognized disposition is UNCLEAR, never
    # assumed performed. Only an explicit, valid disposition is trusted.
    try:
        return Disposition(str(value).strip().lower())
    except (ValueError, AttributeError):
        return Disposition.UNCLEAR


def _extract_json(text: str) -> dict:
    text = text.strip()
    if not text.startswith("{"):
        m = re.search(r"\{.*\}", text, re.DOTALL)
        text = m.group(0) if m else "{}"
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def _default_llm(system: str, user: str) -> str:
    from app.core.llm_client import chat_completion
    out, _ = chat_completion(system, user, temperature=0.0, json_mode=True)
    return out


def _confidence(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _evidence_span(value: Any) -> EvidenceSpan | None:
    if isinstance(value, dict):
        text = str(value.get("text", ""))
        if not text.strip():
            return None
        start = value.get("start")
        try:
            start = int(start) if start is not None else None
        except (TypeError, ValueError):
            start = None
        page = value.get("page")
        try:
            page = int(page) if page is not None else None
        except (TypeError, ValueError):
            page = None
        return EvidenceSpan(text=text, section=value.get("section"), start=start, page=page)
    text = str(value)
    return EvidenceSpan(text=text) if text.strip() else None


def _relation(value: Any) -> RelationAssertion | None:
    if not isinstance(value, dict):
        return None
    try:
        pred = RelationPredicate(str(value.get("predicate", "")).strip().lower())
        state = RelationState(str(value.get("state", "uncertain")).strip().lower())
    except ValueError:
        return None
    subject = str(value.get("subject_event_id", "")).strip()
    obj = str(value.get("object_event_id", "")).strip()
    if not subject or not obj:
        return None
    efi = value.get("evidence_fact_ids")
    if efi is not None and not isinstance(efi, list):
        return None                                  # malformed -> extract_note raises (R1)
    refs = [f"event:{str(x).strip()}" for x in (efi or []) if str(x).strip()]
    return RelationAssertion(subject, pred, obj, state=state, evidence_span_ids=refs,
                             extraction_source="clinical-graph-v1",
                             confidence=_confidence(value.get("confidence")))


class ExtractionSchemaError(ValueError):
    """The extractor returned output that is not a valid claim graph: invalid JSON, a
    malformed fact/relation object, a blank or duplicate fact id. Raised so the pipeline
    fails closed to a retryable SYSTEM_HOLD with ZERO retrieval, instead of silently
    discarding a claim-affecting assertion (a dropped PART_OF leaves an integral component
    billable; an unparseable graph must not read as 'no findings'). (Codex F6-R1.)"""


def _strict_extract_json(text: str) -> dict:
    """Parse the extractor's JSON, failing closed on anything unparseable."""
    text = (text or "").strip()
    if not text.startswith("{"):
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            raise ExtractionSchemaError("extractor output contains no JSON object")
        text = m.group(0)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ExtractionSchemaError(f"extractor output is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ExtractionSchemaError("extractor output is not a JSON object")
    return data


def _participant_index(billing_context: dict[str, Any] | None) -> dict[str, dict]:
    """Typed participant graph from the STRUCTURED encounter context: id -> record carrying
    type (person/organization), roles, function, and organization affiliations. Actor
    identity is resolved against THIS graph -- the model may SELECT a participant but cannot
    invent identity, its type, its affiliation, or its function. (Codex F6-R2.)"""
    idx: dict[str, dict] = {}
    for p in (billing_context or {}).get("participants") or []:
        if not isinstance(p, dict):
            continue
        pid = str(p.get("id", "")).strip()
        if not pid:
            continue
        idx[pid] = {
            "type": str(p.get("type", "")).strip().lower(),
            "roles": {str(r).strip().lower() for r in (p.get("roles") or [])},
            "function": (str(p.get("function")).strip() if p.get("function") else None),
            "affiliations": {str(a).strip() for a in (p.get("affiliations") or [])
                             if str(a).strip()},
        }
    return idx


def extract_note(note_text: str, llm: LLMFn | None = None,
                 billing_context: dict[str, Any] | None = None) -> ExtractionResult:
    llm = llm or _default_llm
    user = json.dumps({"encounter_context": billing_context or {}, "note": note_text},
                      sort_keys=True)
    raw = _strict_extract_json(llm(_SYSTEM, user))
    participants = _participant_index(billing_context)
    seen_ids: set[str] = set()
    # R1: strict top-level schema -- 'facts' must be a present array (missing/null/wrong-type
    # is a malformed graph, NOT an empty note); 'relations' must be an array when present.
    facts_in = raw.get("facts")
    if not isinstance(facts_in, list):
        raise ExtractionSchemaError("extractor output is missing a 'facts' array")
    relations_in = raw.get("relations")
    if relations_in is None:
        relations_in = []
    if not isinstance(relations_in, list):
        raise ExtractionSchemaError("'relations' must be an array when present")
    facts: list[ClinicalFact] = []
    for i, item in enumerate(facts_in):
        if not isinstance(item, dict):
            raise ExtractionSchemaError(f"fact #{i} is not a JSON object")
        kind = _coerce_kind(item.get("kind", ""))
        desc = str(item.get("description", "")).strip()
        if kind is None:
            raise ExtractionSchemaError(
                f"fact #{i} has an unrecognized kind: {item.get('kind')!r}")
        if not desc:
            raise ExtractionSchemaError(f"fact #{i} has no description")
        # A negated finding, or one the note RULES OUT, is documentation of ABSENCE
        # — never billed. An OMITTED certainty defaults to confirmed (a plainly
        # documented condition, per the prompt); an explicit value is taken as-is.
        raw_cert = item.get("certainty")
        certainty = str(raw_cert).strip().lower() if raw_cert is not None else "confirmed"
        if item.get("negated") is True or certainty == "ruled_out":
            continue
        # Fail-closed on both assertion axes: a condition is coded as present ONLY when
        # it is explicitly CONFIRMED — suspected/probable/possible, or any unrecognized
        # certainty, is not coded as confirmed; and it is the PATIENT's condition only
        # when the experiencer is explicitly the patient — family/other, or any
        # unrecognized experiencer, is not the patient's coded condition.
        certain = certainty == "confirmed"
        experiencer = str(item.get("experiencer", "patient")).strip().lower() or "patient"
        # R1: typed nested shapes. `evidence` must be a LIST of non-empty quotes/spans -- a
        # bare string must never be iterated character-by-character into fake spans;
        # `confidence` must be numeric; `axis_confidence` must be an object.
        ev_in = item.get("evidence")
        ev_in = [] if ev_in is None else ev_in
        if not isinstance(ev_in, list):
            raise ExtractionSchemaError(f"fact #{i} 'evidence' must be a list of quotes/spans")
        spans = []
        for q in ev_in:
            sp = _evidence_span(q)
            if sp is None:
                raise ExtractionSchemaError(
                    f"fact #{i} has an empty/malformed evidence element")
            spans.append(sp)
        raw_conf = item.get("confidence")
        if raw_conf is not None and not isinstance(raw_conf, (int, float)):
            raise ExtractionSchemaError(f"fact #{i} 'confidence' must be numeric")
        scalar = _confidence(raw_conf)
        supplied_axes = item.get("axis_confidence")
        if supplied_axes is not None and not isinstance(supplied_axes, dict):
            raise ExtractionSchemaError(f"fact #{i} 'axis_confidence' must be an object")
        supplied_axes = supplied_axes or {}
        axes = {axis: _confidence(supplied_axes.get(axis)) for axis in _REQUIRED_AXES[kind]}
        attrs_in = item.get("attributes")
        if attrs_in is not None and not isinstance(attrs_in, dict):
            raise ExtractionSchemaError(f"fact #{i} 'attributes' must be an object")
        attributes = dict(attrs_in or {})
        # R2: actor identity is resolved EXCLUSIVELY from the structured encounter context.
        # A model-supplied performer/organization id absent from the authoritative roster is
        # invented/unauthorized and is discarded (ownership then resolves to UNKNOWN and
        # HOLDs before retrieval); an unproven function drops with its performer. The billing
        # entity is always the context's. (Codex F6-R2.)
        # Resolve actor identity ONLY from the typed participant graph. The model may
        # SELECT a participant id, but its type, affiliation, and function come from the
        # context -- never the model. A performer id that is unknown, is an ORGANIZATION
        # (not a person), or is not context-designated as a performer is rejected; an
        # organization id is kept only when the context AFFILIATES this performer to it; a
        # model-authored function is discarded and replaced only by the context's function.
        # Anything unresolved leaves ownership UNKNOWN so the service HOLDS before retrieval.
        perf = str(attributes.get("performer_id", "")).strip()
        prec = participants.get(perf)
        attributes.pop("performer_function", None)         # never trust a model-authored function
        if perf and prec and prec["type"] == "person" and (
                "performer" in prec["roles"] or not prec["roles"]):
            attributes["performer_id"] = perf
            if prec["function"]:
                attributes["performer_function"] = prec["function"]
            org = str(attributes.get("organization_id", "")).strip()
            if (org and org in prec["affiliations"]
                    and participants.get(org, {}).get("type") == "organization"):
                attributes["organization_id"] = org
            else:
                attributes.pop("organization_id", None)
        else:
            attributes.pop("performer_id", None)
            attributes.pop("organization_id", None)
        if billing_context and billing_context.get("billing_entity_id"):
            attributes["billing_entity_id"] = billing_context["billing_entity_id"]
        fid = str(item.get("fact_id") or f"F{i+1}").strip()
        if not fid:
            raise ExtractionSchemaError(f"fact #{i} has a blank fact_id")
        if fid in seen_ids:
            raise ExtractionSchemaError(f"duplicate fact_id: {fid}")
        seen_ids.add(fid)
        facts.append(ClinicalFact(
            kind=kind, description=desc, attributes=attributes,
            disposition=_coerce_disposition(item.get("disposition")),
            certain=certain, experiencer=experiencer, evidence=spans,
            confidence=scalar, axis_confidence=axes, fact_id=fid,
        ))
    relations: list[RelationAssertion] = []
    for j, x in enumerate(relations_in):
        rel = _relation(x)
        if rel is None:
            raise ExtractionSchemaError(
                f"relation #{j} is malformed and cannot be safely dropped: {x!r}")
        relations.append(rel)
    return ExtractionResult(facts=facts, relations=relations)


def extract_facts(note_text: str, llm: LLMFn | None = None) -> list[ClinicalFact]:
    """Backward-compatible fact-only view for non-pipeline callers and tests."""
    return extract_note(note_text, llm).facts
