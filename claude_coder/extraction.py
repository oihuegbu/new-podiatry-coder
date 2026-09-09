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

from .models import (AttributeEvidence, ClinicalFact, Disposition, EvidenceSpan,
                     FactKind, RelationAssertion, RelationPredicate, RelationState)

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
        Emit each attribute as ONE entry in exactly one of three arrays, chosen
        by the value's own natural type — never coerce a value into the wrong
        array to make it fit, and never emit the same axis name in more than
        one array (or twice in the same array):
          "strings": {"name": <axis>, "value": <string>} — anatomy, laterality,
              product/material, drug name, approach, contrast,
              technical_vs_professional, performer_id, performer_function,
              organization_id, billing_entity_id, problems, data, risk,
              setting, and any other text-valued axis.
          "numbers": {"name": <axis>, "value": <number>} — depth, area/size,
              count/quantity, dose, wasted amount, total_time_minutes.
          "booleans": {"name": <axis>, "value": true|false} — new_patient,
              separately_identifiable.
  - "attribute_evidence": a FLAT array — one entry per verbatim quote proving
        one attribute value, not grouped by axis. Each entry is {"name": <the
        exact axis name from "attributes" this quote is about>, "text":
        <verbatim quote>, "scope": "local"|"inherited", "parent_fact_id": <id,
        or "" when scope is "local">, "assertion_state":
        "asserted"|"negated"|"uncertain", "value": <the exact value this quote
        is about, always written as a string even when the attribute itself is
        numeric or boolean — e.g. "3" for a numeric depth of 3, "true" for a
        boolean new_patient>} — every axis name you emit anywhere in
        "attributes" (any of the three arrays) needs at least one entry here
        naming it; an axis may have several entries when several quotes bear
        on it — the SAME per-endpoint quoting discipline you already use for
        directional relations below, applied to attributes instead. "value"
        MUST equal, verbatim, one of the values you wrote for this axis in
        "attributes" (e.g. if "attributes" says laterality: "right", an entry
        proving that says "value": "right") —
        never a description of the quote, never a different value than the one it
        is cited for; if you also have evidence bearing on a DIFFERENT candidate
        value for the same axis (for example the note first says "left" then
        corrects to "right"), that is a SEPARATE entry with its own "value". A
        quote proving a value the note ultimately RULES OUT still names that
        ruled-out value in "value", paired with "assertion_state": "negated" — a
        negated entry's "value" is what it negates, not the value you finally
        settled on. "assertion_state" is whether THIS quote itself asserts,
        negates, or leaves uncertain the "value" it names —
        judge it from the FULL sentence you are quoting from, however far the negation
        cue sits from the value word ("no left-sided involvement" and "left-sided
        involvement was considered but ultimately ruled out" are BOTH "negated" for
        "left", even though the second puts the negation many words later); this is
        independent of the fact's own overall "disposition"/"certainty"/"negated" —
        one specific attribute mention can be hedged or ruled out even when the fact
        as a whole is confirmed, or vice versa for a value drawn from a different
        sentence. Default "asserted" only when the quote plainly states the value with
        no hedge or negation anywhere in its own sentence; use "uncertain" for hedged/
        possible phrasing, "negated" for anything the sentence rules out or denies.
        "scope" is "local" when the quote sits in this fact's
        own sentence; it is "inherited" only when the value is instead stated once
        in a heading or a linked parent event and this fact's own sentence does not
        repeat it — in that case "parent_fact_id" must name that parent fact, and you
        must ALSO emit a part_of relation with THIS fact as subject_event_id and the
        parent as object_event_id (this fact IS part_of the parent — never the
        reverse, and never same_episode_as, which does not establish that two events
        share the same laterality/anatomy/product/count/approach or any other
        code-changing attribute; use "" for "parent_fact_id" on "local" scope). Never
        mark a value "inherited" without a real, correctly-directed part_of relation
        behind it, and never fabricate a quote a value is not literally present in.
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
  - "axis_confidence": a FLAT array of {"name": <axis>, "confidence": 0.0-1.0}
        entries, one per required extraction axis. Always emit occurrence, action,
        evidence, temporal; for diagnoses also assertion and experiencer; for
        services also performer and relationship. Missing/unclear is 0.0, never
        omitted or averaged away.

Also return "relations": documented edges between facts. Each has
subject_event_id, predicate (part_of | used_in | reason_for | same_episode_as |
separate_from | uses_device | guides | repairs | removes), object_event_id,
state (asserted | negated | uncertain), evidence_fact_ids (facts whose verbatim
evidence supports the edge), and confidence.
Do not infer integrality or distinctness from clinical convention; emit only what the
note documents. PART_OF/USED_IN/REASON_FOR/USES_DEVICE/GUIDES/REPAIRS/REMOVES are
directional; SEPARATE_FROM and SAME_EPISODE_AS are symmetric. USES_DEVICE names a
device/material/instrument the subject event employed; GUIDES names imaging or another
event that steered the subject event; REPAIRS/REMOVES name what the subject event did
to the object event's anatomical target when the note states it as a distinct,
nameable action (e.g. "the tendon was reattached" -> REPAIRS the tendon finding; "the
bursa was excised" -> REMOVES the bursa finding) -- these narrow the SAME action-verb
vocabulary already used in "description", not a new inference the note does not make.

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
{"schema_version": "extraction-wire-v1", "facts": [ ... ], "relations": [ ... ]}."""


#: v2 (issue #6 F9-R11-F): the model now emits attributes/attribute_evidence/
#: axis_confidence as the closed EXTRACTION_WIRE_SCHEMA below rather than
#: advisory json_mode-only dicts. Bumped so an origin/audit record can never
#: mix a pre-wire-schema generation with a strict-wire one under one identity.
_SCHEMA_VERSION = "clinical-graph-v2"


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


def profile_identity(model_profile: Any) -> tuple[str, str]:
    """(provider, canonical profile identity) from the recorded call metadata. Never
    includes a credential value — the pipeline's profile dict carries provider/model/
    callable identity only.

    This is the ONE primitive that answers "whose model answered this call?". It is public
    because independence is not an extraction-only question: every place that decides
    whether two assertions came from distinct origins — the relation graph here, and the
    code-corroboration check in `verify.corroboration_origin` — must read the provider the
    same way, from the same declared metadata, rather than growing its own notion of
    identity that can drift out of agreement with this one."""
    if not isinstance(model_profile, dict):
        return "", ""
    provider = str(model_profile.get("provider", "") or "")
    canonical = json.dumps({str(k): (None if v is None else str(v))
                            for k, v in model_profile.items()},
                           sort_keys=True, separators=(",", ":"))
    return provider, canonical


_profile_identity = profile_identity          # historical private name, kept for callers


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
    provider, profile = profile_identity(model_profile)
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


#: issue #6 F9-R11-F. `attributes`/`attribute_evidence`/`axis_confidence` are
#: genuinely dynamic-key maps -- the model names its own attribute vocabulary
#: per note (anatomy, laterality, depth, performer_id, ...), which was never a
#: fixed enumerable set. Neither Anthropic's nor OpenAI's structured-output
#: grammar can express an open-key object: `additionalProperties` must be
#: explicitly `false` on every object node (confirmed live -- a bare
#: `{"type": "object"}` 400s; adding `additionalProperties: false` with no
#: declared `properties` is accepted, but then the grammar can ONLY ever
#: produce `{}`, silently discarding whatever the model was asked to put
#: there). So the WIRE contract below represents every dynamic map as a
#: closed, generic array of named entries instead, and `wire_to_legacy_*`
#: converts it back to the plain dict shape `_parse_extraction_response`
#: already validates and consumes -- the provider schema is a wire contract,
#: not the domain model; `ClinicalFact`/`AttributeEvidence` and every
#: downstream consumer are unchanged.
_SCHEMA_VERSION_WIRE = "extraction-wire-v1"


def _closed(properties: dict) -> dict:
    """A closed JSON-Schema object: every property enumerated and required,
    `additionalProperties: false` -- the discipline `app/coding/schemas.py`
    documents as empirically required by both providers' structured-output
    grammar (this module's own docstring records where that was verified)."""
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


_NAMED_STRING = _closed({"name": {"type": "string"}, "value": {"type": "string"}})
_NAMED_NUMBER = _closed({"name": {"type": "string"}, "value": {"type": "number"}})
_NAMED_BOOLEAN = _closed({"name": {"type": "string"}, "value": {"type": "boolean"}})

#: Each dynamic attribute is one named entry in exactly one of three typed
#: arrays, chosen by the value's own type (never coerced) -- see the prompt.
_WIRE_ATTRIBUTES = _closed({
    "strings": {"type": "array", "items": _NAMED_STRING},
    "numbers": {"type": "array", "items": _NAMED_NUMBER},
    "booleans": {"type": "array", "items": _NAMED_BOOLEAN},
})

_WIRE_AXIS_CONFIDENCE = {
    "type": "array",
    "items": _closed({"name": {"type": "string"}, "confidence": {"type": "number"}}),
}

#: Flat, not grouped by axis on the wire -- `wire_to_legacy_*` groups by
#: "name" into the existing `dict[str, list[AttributeEvidence]]` shape.
_WIRE_ATTRIBUTE_EVIDENCE = {
    "type": "array",
    "items": _closed({
        "name": {"type": "string"},
        "text": {"type": "string"},
        "scope": {"type": "string", "enum": ["local", "inherited"]},
        "parent_fact_id": {"type": "string"},
        "assertion_state": {"type": "string",
                            "enum": ["asserted", "negated", "uncertain"]},
        "value": {"type": "string"},
    }),
}

_WIRE_FACT = _closed({
    "fact_id": {"type": "string"},
    "kind": {"type": "string", "enum": [k.value for k in FactKind]},
    "description": {"type": "string"},
    "attributes": _WIRE_ATTRIBUTES,
    "attribute_evidence": _WIRE_ATTRIBUTE_EVIDENCE,
    "disposition": {"type": "string", "enum": [d.value for d in Disposition]},
    "negated": {"type": "boolean"},
    "certainty": {"type": "string",
                 "enum": ["confirmed", "suspected", "ruled_out"]},
    "experiencer": {"type": "string", "enum": ["patient", "family", "other"]},
    "evidence": {"type": "array", "items": {"type": "string"}},
    "confidence": {"type": "number"},
    "axis_confidence": _WIRE_AXIS_CONFIDENCE,
})

#: Only the predicates the prompt actually asks the extractor to emit
#: (PERFORMED_BY/ON_BEHALF_OF are established elsewhere in the pipeline, never
#: by this call) -- a schema no wider than what's asked for.
_WIRE_RELATION_PREDICATES = ["part_of", "used_in", "reason_for", "same_episode_as",
                            "separate_from", "uses_device", "guides", "repairs",
                            "removes"]

_WIRE_RELATION = _closed({
    "subject_event_id": {"type": "string"},
    "predicate": {"type": "string", "enum": _WIRE_RELATION_PREDICATES},
    "object_event_id": {"type": "string"},
    "state": {"type": "string", "enum": ["asserted", "negated", "uncertain"]},
    "evidence_fact_ids": {"type": "array", "items": {"type": "string"}},
    "confidence": {"type": "number"},
})

EXTRACTION_WIRE_SCHEMA = _closed({
    "schema_version": {"type": "string", "enum": [_SCHEMA_VERSION_WIRE]},
    "facts": {"type": "array", "items": _WIRE_FACT},
    "relations": {"type": "array", "items": _WIRE_RELATION},
})


def _wire_name(raw: Any, where: str) -> str:
    """A non-blank string name, or a typed error -- the ONE thing the provider's
    grammar cannot itself enforce across independent array entries (a closed
    schema types EACH entry correctly but has no cross-array uniqueness
    constraint, so a model could legally repeat one axis name in two arrays)."""
    if isinstance(raw, bool) or not isinstance(raw, str) or not raw.strip():
        raise ExtractionSchemaError(f"{where} 'name' must be a non-blank string")
    return raw.strip()


def _wire_attributes_to_legacy(wire: dict, where: str) -> dict[str, Any]:
    """The three typed arrays flattened to the legacy `{name: value}` dict,
    rejecting a name repeated within or across arrays -- a duplicate/conflicting
    axis makes the attribute value unknowable, the same discipline
    `_participant_index` already applies to a duplicated participant id."""
    flat: dict[str, Any] = {}
    for group in ("strings", "numbers", "booleans"):
        entries = wire.get(group)
        if not isinstance(entries, list):
            raise ExtractionSchemaError(f"{where} {group!r} must be an array")
        for entry in entries:
            if not isinstance(entry, dict):
                raise ExtractionSchemaError(f"{where} {group!r} entry is not an object")
            name = _wire_name(entry.get("name"), f"{where} {group!r} entry")
            if name in flat:
                raise ExtractionSchemaError(
                    f"{where} declares attribute {name!r} more than once")
            flat[name] = entry.get("value")
    return flat


def _wire_axis_confidence_to_legacy(wire: list, where: str) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for entry in wire:
        if not isinstance(entry, dict):
            raise ExtractionSchemaError(f"{where} entry is not an object")
        name = _wire_name(entry.get("name"), f"{where} entry")
        if name in flat:
            raise ExtractionSchemaError(f"{where} declares axis {name!r} more than once")
        flat[name] = entry.get("confidence")
    return flat


def _wire_attribute_evidence_to_legacy(wire: list, where: str) -> dict[str, list]:
    """The flat wire array grouped by "name" into the legacy
    `{name: [entry, ...]}` shape `_parse_extraction_response` already walks --
    unlike attributes/axis_confidence, several entries per name are expected
    (several quotes may bear on one axis), so this groups rather than rejects
    a repeat."""
    grouped: dict[str, list] = {}
    for entry in wire:
        if not isinstance(entry, dict):
            raise ExtractionSchemaError(f"{where} entry is not an object")
        name = _wire_name(entry.get("name"), f"{where} entry")
        grouped.setdefault(name, []).append({
            "text": entry.get("text"),
            "scope": entry.get("scope"),
            "parent_fact_id": entry.get("parent_fact_id"),
            "assertion_state": entry.get("assertion_state"),
            "value": entry.get("value"),
        })
    return grouped


def _wire_fact_to_legacy(wire_fact: Any, index: int) -> dict:
    if not isinstance(wire_fact, dict):
        raise ExtractionSchemaError(f"wire fact #{index} is not an object")
    where = f"wire fact #{index}"
    legacy = dict(wire_fact)  # fact_id/kind/description/disposition/negated/
                              # certainty/experiencer/evidence/confidence are
                              # IDENTICAL shape on the wire and in the legacy
                              # contract -- only the three dynamic maps convert.
    legacy["attributes"] = _wire_attributes_to_legacy(
        wire_fact.get("attributes") or {}, f"{where} 'attributes'")
    legacy["axis_confidence"] = _wire_axis_confidence_to_legacy(
        wire_fact.get("axis_confidence") or [], f"{where} 'axis_confidence'")
    legacy["attribute_evidence"] = _wire_attribute_evidence_to_legacy(
        wire_fact.get("attribute_evidence") or [], f"{where} 'attribute_evidence'")
    return legacy


def wire_to_legacy_extraction_json(raw: dict) -> dict:
    """Convert one EXTRACTION_WIRE_SCHEMA-shaped, already-parsed response into
    the legacy `{"facts": [...], "relations": [...]}` shape
    `_parse_extraction_response` validates and consumes -- the ONE place the
    strict wire contract meets the existing domain model. Everything
    downstream of this function is unchanged from before F9-R11-F."""
    version = raw.get("schema_version") if isinstance(raw, dict) else None
    if version != _SCHEMA_VERSION_WIRE:
        raise ExtractionSchemaError(
            f"unrecognized extraction wire schema_version: {version!r}")
    facts_in = raw.get("facts")
    if not isinstance(facts_in, list):
        raise ExtractionSchemaError("wire output is missing a 'facts' array")
    relations_in = raw.get("relations")
    if relations_in is not None and not isinstance(relations_in, list):
        raise ExtractionSchemaError("wire 'relations' must be an array when present")
    return {
        "facts": [_wire_fact_to_legacy(f, i) for i, f in enumerate(facts_in)],
        "relations": relations_in or [],
    }


def _default_llm(system: str, user: str) -> str:
    from app.core.llm_client import chat_completion
    out, _ = chat_completion(system, user, temperature=0.0, json_mode=True,
                             json_schema=EXTRACTION_WIRE_SCHEMA)
    legacy = wire_to_legacy_extraction_json(_strict_extract_json(out))
    return json.dumps(legacy)


#: Which provider gives the INDEPENDENT second reading, given the primary one. Two
#: calls into one vendor share training data, tokeniser and failure modes, so the second
#: reading is pinned to the other declared provider whenever one is configured.
_SECOND_READING_PROVIDER = {"claude": "openai", "openai": "claude"}


class SecondReadingUnavailable(RuntimeError):
    """No independent provider is configured for the second reading of the note.

    Raised rather than silently falling back to the SAME provider (which would make the
    audit record claim an independence the run never had) or to no second reading at all
    (which would silently drop a control). The pipeline turns it into a retryable system
    hold with zero retrieval.
    """


def default_second_extract_llm(system: str, user: str) -> str:
    """The second, independent reading of the note.

    Same prompt and same schema as the primary reading on purpose: the comparison
    downstream is between two readings of the DOCUMENT, so anything else that differed
    would confound it. The provider is the one the primary extraction is NOT using.
    """
    from app.core import config
    from app.core.llm_client import chat_completion
    primary = str(getattr(config, "LLM_PROVIDER", "") or "").strip().lower()
    provider = _SECOND_READING_PROVIDER.get(primary)
    if provider is None:
        raise SecondReadingUnavailable(
            f"no independent second-reading provider is configured for primary "
            f"extraction provider {primary!r}")
    model = (config.OPENAI_MODEL if provider == "openai" else config.CLAUDE_MODEL)
    out, _ = chat_completion(system, user, model=model, provider=provider,
                             temperature=0.0, json_mode=True, use_batch=False,
                             json_schema=EXTRACTION_WIRE_SCHEMA)
    legacy = wire_to_legacy_extraction_json(_strict_extract_json(out))
    return json.dumps(legacy)


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


def _attribute_evidence_entry(raw: Any) -> tuple[str, str, str, RelationState, str] | None:
    """(text, scope, parent_fact_id, assertion_state, value) for one raw
    `attribute_evidence` entry, or None when malformed (the caller raises -- same
    discipline as `_evidence_span`). Never stringifies a non-quote into a
    pseudo-quote, matching `_evidence_span`; "local" scope needs no parent,
    "inherited" scope requires one (resolved against the relations graph by the
    caller, once relations are parsed -- issue #6 F9-R5).

    "assertion_state" parses with the EXACT same fail-closed convention `_relation`
    already uses for its own `state` field (issue #6 F9-R6-R2, fifth re-review): an
    invalid value fails the whole entry (never silently coerced), a missing one
    defaults to UNCERTAIN, never ASSERTED -- an omitted judgement must never be read
    as a positive one.

    "value" (issue #6 F9-R6-R2, sixth re-review) binds this entry to the SPECIFIC
    attribute value it proves -- a malformed (non-string/boolean) value fails the
    whole entry, same discipline as "text", but a genuinely OMITTED value defaults
    to "" (unbound) rather than failing the entry: an older-style extraction that
    hasn't caught up to this field still gets a real, if unbound, evidence record
    (`graph_consensus.claim_authorized_value` treats unbound evidence as if it
    never named a value, falling back to its own weaker whole-fact-text check --
    never silently treated as proof of whatever `attributes[axis]` happens to
    hold)."""
    if not isinstance(raw, dict):
        return None
    raw_text = raw.get("text", "")
    if isinstance(raw_text, bool) or not isinstance(raw_text, str) or not raw_text.strip():
        return None
    scope = str(raw.get("scope", "local") or "local").strip().lower()
    if scope not in ("local", "inherited"):
        return None
    parent = str(raw.get("parent_fact_id", "") or "").strip()
    if scope == "inherited" and not parent:
        return None
    try:
        assertion_state = RelationState(
            str(raw.get("assertion_state", "uncertain")).strip().lower())
    except ValueError:
        return None
    raw_value = raw.get("value", "")
    if isinstance(raw_value, bool) or not isinstance(raw_value, str):
        return None
    bound_value = raw_value.strip()
    return raw_text, scope, parent, assertion_state, bound_value


def _relation(value: Any, index: int, retained_fact_ids: set[str],
             filtered_reason: dict[str, str] | None = None) -> RelationAssertion | None:
    """A relation, or None when its SHAPE is malformed (the caller raises a generic
    "cannot be safely dropped" error). An IDENTITY defect -- an endpoint or evidence
    reference that does not name a fact THIS SAME response actually retained -- raises
    ExtractionSchemaError directly with a specific message, so it retries through
    `extract_note`'s bounded loop exactly like a malformed-shape response, instead of
    surfacing only much later in `provenance.validate_relations` (a RelationIntegrityError
    that loop never sees, which held the whole encounter on the very first bad draw
    instead of retrying it). `provenance.validate_relations` remains the downstream
    defense-in-depth check for anything that reaches it anyway -- unchanged, not
    weakened. `retained_fact_ids` deliberately excludes a fact the model itself marked
    negated/ruled_out: a relation to a fact that was never billable is an invalid
    retained graph, not an edge to silently drop. (Codex F9-R11-G.)

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
    filtered_reason = filtered_reason or {}
    for role, event_id in (("subject_event_id", subject), ("object_event_id", obj)):
        if event_id not in retained_fact_ids:
            reason = filtered_reason.get(event_id)
            detail = (f"names fact {event_id!r}, which this response marked {reason} and "
                     f"did not retain" if reason else
                     f"names {event_id!r}, which is not a fact_id this response emitted")
            raise ExtractionSchemaError(f"relation #{index} {role!r} {detail}")
    if subject == obj:
        raise ExtractionSchemaError(f"relation #{index} is self-referential: {subject!r}")
    efi = value.get("evidence_fact_ids")
    if efi is not None and not isinstance(efi, list):
        return None                                  # malformed -> extract_note raises (R1)
    raw_efi = [str(x).strip() for x in (efi or []) if str(x).strip()]
    if not raw_efi:
        raise ExtractionSchemaError(
            f"relation #{index} 'evidence_fact_ids' must be a non-empty list of "
            f"retained fact_ids")
    unretained = [x for x in raw_efi if x not in retained_fact_ids]
    if unretained:
        raise ExtractionSchemaError(
            f"relation #{index} 'evidence_fact_ids' names {unretained!r}, which is not "
            f"a fact_id this response retained")
    refs = [f"event:{x}" for x in raw_efi]
    conf = _confidence(value.get("confidence"), f"relation #{index} 'confidence'")
    return RelationAssertion(subject, pred, obj, state=state, evidence_span_ids=refs,
                             extraction_source=_SCHEMA_VERSION, confidence=conf)


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


#: A malformed-shape response (invalid JSON, a fact missing a required field, an
#: attribute_evidence entry that isn't the object the prompt specifies, ...) is a
#: single bad draw from the model, not a deterministic property of the note -- the
#: SAME class of failure the vision extraction call already retries against (see
#: app/ingestion/pdf_parser.py's per-attempt JSON/shape validation loop, added after
#: a malformed-but-unretried response reached the pipeline live). Retrying here closes
#: the one place that class of failure could still turn a single bad draw into a
#: whole encounter held at pre_retrieval_integrity instead of a normal coded result.
_EXTRACTION_MAX_ATTEMPTS = 3


#: Fixed, categorical retry guidance -- NEVER the raw exception text (which can quote
#: model-supplied ids/values back into the prompt) and NEVER the note text again (already
#: present in `base_payload["note"]`). Regenerates the FULL response from scratch; the
#: prior malformed response is discarded, never patched in place. (Codex F9-R11-G item 5.)
_RETRY_VALIDATION_FEEDBACK = (
    "Your previous response was rejected for a structural error and discarded. "
    "Regenerate the FULL response from scratch -- do not try to patch the prior one. "
    "Two rules the prior response violated somewhere: (1) every fact's \"fact_id\" "
    "must be an explicit, non-blank string you choose yourself, unique across the "
    "whole response -- never leave it blank or omit it. (2) every relation's "
    "subject_event_id, object_event_id, and every entry in evidence_fact_ids must "
    "exactly name the fact_id of a fact YOU ALSO EMIT in this same response as a "
    "RETAINED fact -- a fact you mark \"negated\": true, or \"certainty\": "
    "\"ruled_out\", is not retained and must never be a relation endpoint or "
    "evidence reference."
)


def extract_note(note_text: str, llm: LLMFn | None = None,
                 billing_context: dict[str, Any] | None = None, *,
                 run_id: str | None = None,
                 model_profile: dict[str, Any] | None = None) -> ExtractionResult:
    llm = llm or _default_llm
    # Validate the authoritative encounter context BEFORE spending an extraction call: a
    # malformed roster can never produce trustworthy ownership, so it fails closed up front.
    participants = _participant_index(billing_context)
    base_payload = {"encounter_context": billing_context or {}, "note": note_text}
    user = json.dumps(base_payload, sort_keys=True)
    for attempt in range(1, _EXTRACTION_MAX_ATTEMPTS + 1):
        try:
            # `llm(...)` is INSIDE the try (issue #6 F9-R11-F): a production llm
            # callable now does its own wire->legacy conversion (see
            # `wire_to_legacy_extraction_json`) and can raise ExtractionSchemaError
            # itself -- e.g. a duplicate attribute name across wire arrays, which
            # the provider's grammar cannot itself forbid. That must retry exactly
            # like a malformed-shape response already parsed by
            # `_parse_extraction_response` below, not escape the retry loop.
            raw_response = llm(_SYSTEM, user)
            return _parse_extraction_response(
                raw_response, participants, billing_context, note_text,
                run_id=run_id, model_profile=model_profile)
        except ExtractionSchemaError as exc:
            if attempt == _EXTRACTION_MAX_ATTEMPTS:
                raise
            from app.core.logger import get_logger
            get_logger(__name__).warning(
                f"  Extraction attempt {attempt}: {exc} — retrying")
            user = json.dumps({**base_payload,
                              "validation_feedback": _RETRY_VALIDATION_FEEDBACK},
                             sort_keys=True)
    raise AssertionError("unreachable")  # loop always returns or raises above


def _parse_extraction_response(
    raw_response: str, participants: dict[str, Any],
    billing_context: dict[str, Any] | None, note_text: str, *,
    run_id: str | None, model_profile: dict[str, Any] | None,
) -> ExtractionResult:
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
    # The set every relation endpoint/evidence reference is checked against below --
    # every raw fact_id the model declared, MINUS the ones it also marked negated/
    # ruled_out. `filtered_reason` records WHY a raw id was excluded, purely so a
    # relation naming it gets a specific "this fact was negated/ruled_out" message
    # instead of an indistinguishable "unknown fact" one. (Codex F9-R11-G.)
    retained_fact_ids: set[str] = set()
    filtered_reason: dict[str, str] = {}
    # (fact_id, attr_name, text, parent_fact_id, assertion_state) for every
    # "inherited"-scope attribute_evidence entry -- resolved against the relations
    # graph in the second pass below, once every relation has been parsed (issue #6
    # F9-R5; assertion_state added issue #6 F9-R6-R2, fifth re-review).
    pending_inherited: list[tuple[str, str, str, str, RelationState, str]] = []
    for i, item in enumerate(facts_in):
        if not isinstance(item, dict):
            raise ExtractionSchemaError(f"fact #{i} is not a JSON object")
        # fact_id identity is validated and registered for EVERY raw fact -- BEFORE
        # negated/ruled_out filtering below -- because a relation can reference this
        # exact id regardless of whether the fact it names goes on to be billable.
        # NEVER a fallback f"F{i+1}": that silently manufactured an id for a missing/
        # blank one, made the very next line's blank-id check unreachable, and could
        # collide with (or diverge from) an id the model itself used elsewhere in the
        # SAME response -- e.g. in a relation or an attribute_evidence entry. A
        # missing/blank fact_id is always the model's error to retry, never something
        # this parser papers over. (Codex F9-R11-G item 1/adjacent defect.)
        raw_fid = item.get("fact_id")
        if isinstance(raw_fid, bool) or not isinstance(raw_fid, str) or not raw_fid.strip():
            raise ExtractionSchemaError(f"fact #{i} has a missing or blank fact_id")
        fid = raw_fid.strip()
        if fid in seen_ids:
            raise ExtractionSchemaError(f"duplicate fact_id: {fid}")
        seen_ids.add(fid)
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
            # `fid` is already registered in `seen_ids` above (so a duplicate is still
            # caught) but deliberately kept OUT of `retained_fact_ids`: a relation
            # naming this id -- e.g. a stale part_of/separate_from written before the
            # model decided to negate/rule out the fact -- names an invalid retained
            # graph, not an edge that can be silently dropped (Codex F9-R11-G item 3).
            filtered_reason[fid] = "negated" if item.get("negated") is True else "ruled_out"
            continue
        retained_fact_ids.add(fid)
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
        # R1, same discipline as `evidence`/`attributes`: typed nested shape or a loud
        # schema error, never a silent drop of a malformed entry (issue #6 F9-R5).
        # "local"-scope entries need nothing else and are built here; "inherited"-scope
        # entries are queued in `pending_inherited` and resolved against the relations
        # graph in the second pass below, once every fact_id and relation exists to
        # validate against -- an "inherited" claim with no matching relation is never
        # constructed at all (fail-closed, not silently degraded to "local").
        attr_ev_in = item.get("attribute_evidence")
        if attr_ev_in is not None and not isinstance(attr_ev_in, dict):
            raise ExtractionSchemaError(f"fact #{i} 'attribute_evidence' must be an object")
        attribute_evidence: dict[str, list[AttributeEvidence]] = {}
        for attr_name, entries in (attr_ev_in or {}).items():
            if not isinstance(entries, list):
                raise ExtractionSchemaError(
                    f"fact #{i} attribute_evidence[{attr_name!r}] must be a list")
            for entry in entries:
                parsed = _attribute_evidence_entry(entry)
                if parsed is None:
                    raise ExtractionSchemaError(
                        f"fact #{i} attribute_evidence[{attr_name!r}] has an "
                        f"empty/malformed entry")
                text, scope, parent, assertion_state, bound_value = parsed
                if scope == "local":
                    attribute_evidence.setdefault(str(attr_name), []).append(
                        AttributeEvidence(span=EvidenceSpan(text=text), scope="local",
                                         assertion_state=assertion_state,
                                         value=bound_value))
                else:
                    pending_inherited.append(
                        (fid, str(attr_name), text, parent, assertion_state, bound_value))
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
        facts.append(ClinicalFact(
            kind=kind, description=desc, attributes=attributes,
            disposition=_coerce_disposition(item.get("disposition")),
            certain=certain, experiencer=experiencer, evidence=spans,
            confidence=scalar, axis_confidence=axes, fact_id=fid,
            attribute_evidence={k: tuple(v) for k, v in attribute_evidence.items()},
        ))
    # ONE origin for this whole response: every edge below is stamped with it, so repeating
    # an edge inside this response accumulates raw `support` but NOT independent support.
    origin = call_origin(note_text, raw_response, run_id=run_id, model_profile=model_profile)
    relations: list[RelationAssertion] = []
    for j, x in enumerate(relations_in):
        rel = _relation(x, j, retained_fact_ids, filtered_reason)
        if rel is None:
            raise ExtractionSchemaError(
                f"relation #{j} is malformed and cannot be safely dropped: {x!r}")
        rel.assertion_origins = [origin.origin_id]
        relations.append(rel)
    # Second pass (issue #6 F9-R5, hardened per F9-R5-A): identify a CANDIDATE relation
    # for every "inherited"-scope attribute_evidence entry against the NOW-COMPLETE
    # (but not yet reconciled) relations graph. Only `PART_OF`, in the EXACT required
    # direction (this fact IS PART_OF the named parent -- `SAME_EPISODE_AS` never
    # qualifies: same episode does not imply the same laterality/anatomy/product/
    # count/approach, and an unordered/reversed match would let a value flow the wrong
    # way). This is a CANDIDATE only -- state, and whether the document actually
    # grounds the relation, are not yet knowable here (relations have not been
    # reconciled), so `scope_validated` stays False; `provenance.validate_attribute_
    # evidence` is the sole authority that may set it True, after reconciliation.
    # An entry whose claimed parent names no matching PART_OF relation at all is
    # dropped entirely here, never silently kept as unproven provenance.
    if pending_inherited:
        by_fid = {f.fact_id: f for f in facts}
        for fid, attr_name, text, parent, assertion_state, bound_value in pending_inherited:
            fact = by_fid.get(fid)
            if fact is None or parent not in by_fid:
                continue
            match = next((r for r in relations
                         if r.predicate is RelationPredicate.PART_OF
                         and r.subject_event_id == fid and r.object_event_id == parent),
                        None)
            if match is None:
                continue
            entry = AttributeEvidence(span=EvidenceSpan(text=text), scope="inherited",
                                      parent_fact_id=parent,
                                      source_relation_id=match.relation_id,
                                      assertion_state=assertion_state,
                                      value=bound_value)
            fact.attribute_evidence = {
                **fact.attribute_evidence,
                attr_name: fact.attribute_evidence.get(attr_name, ()) + (entry,),
            }
    return ExtractionResult(facts=facts, relations=relations, origin=origin)


def extract_facts(note_text: str, llm: LLMFn | None = None) -> list[ClinicalFact]:
    """Backward-compatible fact-only view for non-pipeline callers and tests."""
    return extract_note(note_text, llm).facts
