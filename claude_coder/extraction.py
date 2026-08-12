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

import hashlib
import json
import math
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

Whenever you emit a DIRECTIONAL relation, give each of its two endpoint facts an
ADDITIONAL evidence quote: the shortest verbatim phrase inside the linking sentence
that names that endpoint. The direction is checked from where those two phrases sit in
the note and from the wording between them, so two endpoints supported only by one
identical long quote cannot be verified and the relation will be treated as unproven.

Rules: quote evidence verbatim; separate a planned/ordered service from a
performed one; capture negation; do not merge distinct events; do not invent
facts the note does not support. For a DIAGNOSIS, the "description" must be the
concise clinical name of ONE condition — when a note phrase lists several
conditions together, emit a SEPARATE diagnosis fact for each, and keep severity
prose, counts, and functional-limitation wording OUT of the description (put
them in attributes or omit). Return JSON only:
{"facts": [ ... ], "relations": [ ... ]}."""


_SCHEMA_VERSION = "clinical-graph-v1"


@dataclass(frozen=True)
class ExtractionOrigin:
    """Identity of ONE extraction call — the unit of assertion independence.

    Everything a single response emits shares this origin, so an extraction model that
    repeats the same edge inside one response cannot make it look like two sources agreed.
    Two origins differ only when something that could actually make them independent
    differs: the run, the provider/profile that answered, the prompt, or the response
    schema. This is derived from the call metadata the pipeline already records
    (`_model_profile_identity`) — it is not a parallel id scheme. (Codex F6-R3.)

    What a count of distinct origins is FOR: the audit trail and confidence display. It is
    not evidence about the record, so it cannot ground a claim-affecting relation — see
    `provenance.MULTIPLY_ASSERTED`, which is a separate axis from
    `provenance.GROUNDED_RECONCILIATION_STATUSES` precisely so that agreement between runs,
    same-provider or cross-provider, can never be read as documentation.
    """
    run_id: str
    provider: str = ""
    profile: str = ""
    prompt_sha256: str = ""
    schema_version: str = _SCHEMA_VERSION

    @property
    def origin_id(self) -> str:
        raw = json.dumps({"run_id": self.run_id, "provider": self.provider,
                          "profile": self.profile, "prompt_sha256": self.prompt_sha256,
                          "schema_version": self.schema_version},
                         sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def as_record(self) -> dict[str, str]:
        """Auditable, credential-free record of this origin."""
        return {"origin_id": self.origin_id, "run_id": self.run_id,
                "provider": self.provider, "profile": self.profile,
                "prompt_sha256": self.prompt_sha256,
                "schema_version": self.schema_version}


def _profile_identity(model_profile: Any) -> tuple[str, str]:
    """(provider, canonical profile identity) from the recorded call metadata. Never
    includes a credential value — the pipeline's profile dict carries provider/model/
    callable identity only."""
    if not isinstance(model_profile, dict):
        return "", ""
    provider = str(model_profile.get("provider", "") or "")
    canonical = json.dumps({str(k): (None if v is None else str(v))
                            for k, v in model_profile.items()},
                           sort_keys=True, separators=(",", ":"))
    return provider, canonical


def call_origin(note_text: str, raw_response: str, *, run_id: str | None = None,
                model_profile: Any = None,
                schema_version: str = _SCHEMA_VERSION) -> ExtractionOrigin:
    """The origin identity for one extraction call.

    `run_id` is supplied by a caller that genuinely runs more than one pass (each pass is
    its own run, so two passes of the SAME provider are two origins and the edge is recorded
    as multiply-asserted — which is an observation about the model, not about the note, and
    releases nothing on its own). When it is
    not supplied the run is identified by its own content — document, prompt, profile and
    the exact response — so a single pass is reproducible (certificates stay stable) and a
    response cannot be counted twice by being replayed into the same graph."""
    prompt_sha = hashlib.sha256(_SYSTEM.encode("utf-8")).hexdigest()
    provider, profile = _profile_identity(model_profile)
    rid = str(run_id).strip() if run_id is not None and str(run_id).strip() else ""
    if not rid:
        seed = "|".join((hashlib.sha256((note_text or "").encode("utf-8")).hexdigest(),
                         prompt_sha, profile, str(schema_version),
                         hashlib.sha256((raw_response or "").encode("utf-8")).hexdigest()))
        rid = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]
    return ExtractionOrigin(run_id=rid, provider=provider, profile=profile,
                            prompt_sha256=prompt_sha, schema_version=str(schema_version))


@dataclass
class ExtractionResult:
    facts: list[ClinicalFact] = field(default_factory=list)
    relations: list[RelationAssertion] = field(default_factory=list)
    schema_version: str = _SCHEMA_VERSION
    # WHICH call produced this graph. Every relation above carries this origin's id, so a
    # downstream corroboration count is a count of DISTINCT origins, never of repetitions.
    origin: ExtractionOrigin | None = None


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


class ExtractionSchemaError(ValueError):
    """The extractor returned output that is not a valid claim graph: invalid JSON, a
    malformed fact/relation object, a blank or duplicate fact id, a malformed confidence, or
    a malformed encounter/billing context. Raised so the pipeline fails closed to a retryable
    SYSTEM_HOLD with ZERO retrieval, instead of silently discarding a claim-affecting
    assertion (a dropped PART_OF leaves an integral component billable; an unparseable graph
    must not read as 'no findings') or COERCING malformed output into a trusted value.
    (Codex F6-R1/F6-R2.)"""


def _confidence(value: Any, where: str, *, missing_ok: bool = True) -> float:
    """A confidence is a FINITE JSON number in [0.0, 1.0] — nothing else.

    Malformed output is REJECTED, never coerced: `true`/`false` (JSON booleans, which pass
    Python's ``isinstance(x, (int, float))`` because ``bool`` subclasses ``int``), numeric
    strings, ``NaN``/``Infinity`` (which Python's json accepts by default), and out-of-range
    numbers all raise. Silently coercing them turns malformed model output into a TRUSTED —
    frequently MAXIMUM — confidence that then drives eligibility, relation and autonomy
    thresholds; the required behaviour is a typed, retryable extraction hold.

    Out-of-range numbers REJECT rather than clamp: a value outside [0,1] is not a confidence
    the schema can interpret, and clamping 42 -> 1.0 is the same silent-maximum defect.
    An ABSENT (null/omitted) confidence is the one permitted non-number and means 0.0 —
    fail-closed, since zero confidence cannot clear any control floor. (Codex F6-R1.)
    """
    if value is None:
        if missing_ok:
            return 0.0
        raise ExtractionSchemaError(f"{where} is required and must be a number in [0.0, 1.0]")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ExtractionSchemaError(
            f"{where} must be a JSON number in [0.0, 1.0], got {type(value).__name__} "
            f"{value!r}")
    num = float(value)
    if not math.isfinite(num):
        raise ExtractionSchemaError(f"{where} must be a finite number, got {value!r}")
    if not 0.0 <= num <= 1.0:
        raise ExtractionSchemaError(f"{where} must be within [0.0, 1.0], got {value!r}")
    return num


def _evidence_span(value: Any) -> EvidenceSpan | None:
    """An evidence span, or None when the element is malformed (the caller raises).

    A quote must be a JSON string (or an object with a string `text`). Anything else --
    a boolean, a number, a nested list -- is NOT stringified into a pseudo-quote: the same
    coercion class as the confidence defect, and a fabricated "True" span would go on to
    fail anchoring for the wrong reason instead of failing the schema loudly.
    """
    if isinstance(value, dict):
        raw_text = value.get("text", "")
        if isinstance(raw_text, bool) or not isinstance(raw_text, str):
            return None
        text = raw_text
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
    if isinstance(value, bool) or not isinstance(value, str):
        return None                      # never stringify a non-quote into a pseudo-quote
    return EvidenceSpan(text=value) if value.strip() else None


def _relation(value: Any, index: int) -> RelationAssertion | None:
    """A relation, or None when its identity/shape is malformed (the caller raises).

    Confidence is validated with the SAME strict rule as fact confidence and raises
    directly, so a boolean/string/NaN relation confidence can never be coerced into a
    trusted edge weight that the necessity control floor then reads. (Codex F6-R1.)"""
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
    conf = _confidence(value.get("confidence"), f"relation #{index} 'confidence'")
    return RelationAssertion(subject, pred, obj, state=state, evidence_span_ids=refs,
                             extraction_source="clinical-graph-v1", confidence=conf)


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


# Participant kinds the encounter context may declare. This is an identity vocabulary for
# WHO takes part in an encounter (a natural person vs a legal entity) -- not a medical code
# set, and nothing here is code-family shaped.
_PARTICIPANT_TYPES = ("person", "organization")

# The role designation that authorizes a person to be billed as having performed a service.
# Only an explicit, context-issued designation counts -- there is no "no roles means anything"
# wildcard, because an unauthorized-but-known person is exactly the actor-authorization defect
# this graph exists to prevent. (Codex F6-R2.)
_PERFORMER_ROLE = "performer"


def _string_list(value: Any, where: str) -> list[str]:
    """A strictly typed list of non-blank strings, or a typed error.

    A mapping such as ``{"performer": true}`` is NOT silently iterated by key (which would
    manufacture a valid ``performer`` role out of malformed input), and a bare string is not
    iterated character-by-character. Absent/null is an empty list; anything else raises.
    (Codex F6-R2.)"""
    if value is None:
        return []
    if not isinstance(value, list):
        raise ExtractionSchemaError(
            f"{where} must be a JSON array of strings, got {type(value).__name__}")
    out: list[str] = []
    for i, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, str) or not item.strip():
            raise ExtractionSchemaError(
                f"{where}[{i}] must be a non-blank string, got {item!r}")
        out.append(item.strip())
    return out


def _participant_index(billing_context: dict[str, Any] | None) -> dict[str, dict]:
    """Typed participant graph from the STRUCTURED encounter context: id -> record carrying
    type (person/organization), roles, function, and organization affiliations. Actor
    identity is resolved against THIS graph -- the model may SELECT a participant but cannot
    invent identity, its type, its affiliation, or its function. (Codex F6-R2.)

    The WHOLE context schema is validated strictly and fails closed, because every downstream
    ownership decision is only as trustworthy as this roster:
      - `participants` must be an array of objects, each with a non-blank string `id`;
      - `type` must be an explicitly declared participant kind;
      - `roles`/`affiliations` must be arrays of non-blank strings (a mapping or a bare
        string is malformed input, never iterated into roles);
      - `function` must be a non-blank string when present;
      - a REPEATED participant id is rejected outright rather than resolved last-write-wins,
        since a duplicate/conflicting identity makes ownership unknowable.
    """
    if billing_context is None:
        return {}
    if not isinstance(billing_context, dict):
        raise ExtractionSchemaError("billing_context must be a JSON object")
    raw_participants = billing_context.get("participants")
    if raw_participants is None:
        raw_participants = []
    if not isinstance(raw_participants, list):
        raise ExtractionSchemaError("billing_context 'participants' must be an array")
    entity = billing_context.get("billing_entity_id")
    if entity is not None and (isinstance(entity, bool) or not isinstance(entity, str)
                               or not entity.strip()):
        raise ExtractionSchemaError(
            "billing_context 'billing_entity_id' must be a non-blank string when present")
    idx: dict[str, dict] = {}
    for i, p in enumerate(raw_participants):
        if not isinstance(p, dict):
            raise ExtractionSchemaError(f"billing_context participant #{i} is not an object")
        pid = p.get("id")
        if isinstance(pid, bool) or not isinstance(pid, str) or not pid.strip():
            raise ExtractionSchemaError(
                f"billing_context participant #{i} has no non-blank string 'id'")
        pid = pid.strip()
        if pid in idx:
            raise ExtractionSchemaError(
                f"billing_context declares participant id {pid!r} more than once; a duplicate "
                f"or conflicting identity makes claim ownership unknowable")
        ptype = p.get("type")
        if isinstance(ptype, bool) or not isinstance(ptype, str) \
                or ptype.strip().lower() not in _PARTICIPANT_TYPES:
            raise ExtractionSchemaError(
                f"billing_context participant {pid!r} must declare type one of "
                f"{list(_PARTICIPANT_TYPES)}, got {ptype!r}")
        function = p.get("function")
        if function is not None and (isinstance(function, bool) or not isinstance(function, str)
                                     or not function.strip()):
            raise ExtractionSchemaError(
                f"billing_context participant {pid!r} 'function' must be a non-blank string "
                f"when present")
        idx[pid] = {
            "type": ptype.strip().lower(),
            "roles": {r.lower() for r in
                      _string_list(p.get("roles"), f"participant {pid!r} 'roles'")},
            "function": function.strip() if function else None,
            "affiliations": set(
                _string_list(p.get("affiliations"), f"participant {pid!r} 'affiliations'")),
        }
    return idx


def extract_note(note_text: str, llm: LLMFn | None = None,
                 billing_context: dict[str, Any] | None = None, *,
                 run_id: str | None = None,
                 model_profile: dict[str, Any] | None = None) -> ExtractionResult:
    llm = llm or _default_llm
    # Validate the authoritative encounter context BEFORE spending an extraction call: a
    # malformed roster can never produce trustworthy ownership, so it fails closed up front.
    participants = _participant_index(billing_context)
    user = json.dumps({"encounter_context": billing_context or {}, "note": note_text},
                      sort_keys=True)
    raw_response = llm(_SYSTEM, user)
    raw = _strict_extract_json(raw_response)
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
        # Every confidence -- scalar and per-axis -- is validated as a finite JSON number.
        # This applies to EVERY fact kind (procedure, diagnosis, supply, drug, imaging, E/M):
        # `_REQUIRED_AXES` covers all of them, and EVERY supplied axis value is checked, not
        # only the required axes, so an unused-but-malformed axis cannot ride along either.
        scalar = _confidence(item.get("confidence"), f"fact #{i} 'confidence'")
        supplied_axes = item.get("axis_confidence")
        if supplied_axes is not None and not isinstance(supplied_axes, dict):
            raise ExtractionSchemaError(f"fact #{i} 'axis_confidence' must be an object")
        supplied_axes = supplied_axes or {}
        for axis_name, axis_value in supplied_axes.items():
            _confidence(axis_value, f"fact #{i} axis_confidence[{axis_name!r}]")
        axes = {axis: _confidence(supplied_axes.get(axis),
                                  f"fact #{i} axis_confidence[{axis!r}]")
                for axis in _REQUIRED_AXES[kind]}
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
        # The performer designation must be EXPLICIT: a context person carrying no roles (or
        # only non-performer roles such as a scribe/supervisor/referrer) is NOT authorized to
        # be billed as the performer. The former "or not prec['roles']" wildcard let the model
        # elevate any known person into the billing performer. (Codex F6-R2.)
        perf = str(attributes.get("performer_id", "")).strip()
        prec = participants.get(perf)
        attributes.pop("performer_function", None)         # never trust a model-authored function
        if perf and prec and prec["type"] == "person" and _PERFORMER_ROLE in prec["roles"]:
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
            attributes["billing_entity_id"] = str(
                billing_context["billing_entity_id"]).strip()
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
    # ONE origin for this whole response: every edge below is stamped with it, so repeating
    # an edge inside this response accumulates raw `support` but NOT independent support.
    origin = call_origin(note_text, raw_response, run_id=run_id, model_profile=model_profile)
    relations: list[RelationAssertion] = []
    for j, x in enumerate(relations_in):
        rel = _relation(x, j)
        if rel is None:
            raise ExtractionSchemaError(
                f"relation #{j} is malformed and cannot be safely dropped: {x!r}")
        rel.assertion_origins = [origin.origin_id]
        relations.append(rel)
    return ExtractionResult(facts=facts, relations=relations, origin=origin)


def extract_facts(note_text: str, llm: LLMFn | None = None) -> list[ClinicalFact]:
    """Backward-compatible fact-only view for non-pipeline callers and tests."""
    return extract_note(note_text, llm).facts
