"""Strict JSON Schemas for the four coding passes' structured outputs.

Sent through the provider's structured-output API (Anthropic
output_config.format / OpenAI json_schema response_format), these schemas
eliminate the malformed-JSON error class observed live — bare strings inside
code arrays (note 053), string entries in corrections_made (note 022), and
fields silently dropped by the verify pass — at generation time instead of
repairing them downstream (_normalize_code_arrays/_normalize_corrections stay
as backstops).

Design rules, each earned from a live 400 from the API's grammar compiler:
  * additionalProperties must be false on every object, so each entry schema
    enumerates the fields that pass's prompt asks for — a strict grammar that
    omitted a requested field would silently FORBID the model from emitting
    it.
  * EVERY property is required (the API limits optional parameters to 24 per
    schema; an all-optional union schema 400'd at 54). All-required also
    means the verify pass can never silently drop a field: it must emit the
    key.
  * Union/nullable parameters are limited to 16 per schema (400'd at 69), so
    plain types are used everywhere the model can always emit a value ("" /
    [] / 0 when inapplicable), and anyOf-null is reserved for the few fields
    downstream logic genuinely distinguishes null on (laterality,
    review_reason, mdm_details, to_code).
  * No medical codes appear anywhere here — the schemas constrain SHAPE only.
"""

from __future__ import annotations

_STR = {"type": "string"}
_NUM = {"type": "number"}
# units must be a whole number: the CodingResult models declare units: int,
# and a grammar-legal 1.5 would fail Pydantic coercion downstream
_INT = {"type": "integer"}
_BOOL = {"type": "boolean"}
_STR_ARR = {"type": "array", "items": {"type": "string"}}
# nullable — budget these (see module docstring)
_NSTR = {"anyOf": [{"type": "string"}, {"type": "null"}]}


def _obj(props: dict) -> dict:
    """Closed object with every property required (see module docstring)."""
    return {
        "type": "object",
        "properties": props,
        "required": list(props),
        "additionalProperties": False,
    }


_MODIFIER_REASONING = {
    "type": "array",
    "items": _obj({"modifier": _STR, "status": _STR, "reason": _STR}),
}

# Always-required object with plain fields: non-E/M lines emit zeros/"" —
# downstream only reads mdm_details for E/M codes, so this costs nothing and
# avoids a union.
# mdm_level is a closed enum, not free text: observed live (note 009), a run
# emitted the whole 2-of-3 derivation sentence as the level ("high (problems)
# / moderate (data, risk) → overall MDM moderate...") — every downstream
# floor/ceiling/consistency layer gates on claimed-level == recomputed-level
# and silently skipped the entry, letting a 99214/99215 flip through the
# consistency gate. The grammar now forces the bare level; the derivation
# belongs in the *_rationale fields. "" is the non-E/M-line value.
_MDM_DETAILS = _obj({
    "problems_score": _NUM,
    "data_score": _NUM,
    "risk_score": _NUM,
    "mdm_level": {"type": "string",
                  "enum": ["", "straightforward", "low", "moderate", "high"]},
    "problems_rationale": _STR,
    "data_rationale": _STR,
    "risk_rationale": _STR,
})

# Diagnosis entry: pass 1's icd10_codes/supporting_conditions fields.
# Unions: laterality, review_reason (2 per instance).
_ICD_ENTRY = _obj({
    "code": _STR,
    "description": _STR,
    "type": _STR,
    "confidence": _NUM,
    "rationale": _STR,
    "supporting_text": _STR,
    "laterality": _NSTR,
    "source_section": _STR,
    "billable_tier": _STR,
    "needs_review": _BOOL,
    "review_reason": _NSTR,
})

# Procedure/supply entry: pass 2 (CPT) and pass 3 (HCPCS) fields.
# Unions: laterality, review_reason (2 per instance).
_PROC_ENTRY = _obj({
    "code": _STR,
    "description": _STR,
    "confidence": _NUM,
    "modifiers": _STR_ARR,
    "modifier_reasoning": _MODIFIER_REASONING,
    "source": _STR,
    "mdm_details": _MDM_DETAILS,
    "procedure_status": _STR,
    "laterality": _NSTR,
    "linked_diagnoses": _STR_ARR,
    "units": _INT,
    "evidence_spans": _STR_ARR,
    "rationale": _STR,
    "supporting_text": _STR,
    "needs_review": _BOOL,
    "review_reason": _NSTR,
})

_ICD_ARR = {"type": "array", "items": _ICD_ENTRY}
_PROC_ARR = {"type": "array", "items": _PROC_ENTRY}

_SNOMED_ENTRY = _obj({
    "concept_id": _STR,
    "description": _STR,
    "entity_text": _STR,
    "category": _STR,
    "confidence": _NUM,
})

# Unions: to_code (1) — null vs value is load-bearing for CHANGED enforcement.
_CORRECTION_ENTRY = _obj({
    "type": _STR,
    "code": _STR,
    "to_code": _NSTR,
    "reason": _STR,
    "evidence": _STR,
})

# Union budget: 2 (icd) x2 arrays + 1 top-level = 5
ICD_PASS_SCHEMA = _obj({
    "icd10_codes": _ICD_ARR,
    "supporting_conditions": _ICD_ARR,
    "sequencing_reasoning": _STR,
})

# Union budget: 2 (proc) = 2
CPT_PASS_SCHEMA = _obj({
    "cpt_codes": _PROC_ARR,
    "em_level_reasoning": _STR,
})

# Union budget: 2 (proc) = 2
HCPCS_PASS_SCHEMA = _obj({
    "hcpcs_codes": _PROC_ARR,
    "snomed_codes": {"type": "array", "items": _SNOMED_ENTRY},
})

# The verify pass re-emits every array, so full entry objects four times
# blew the API's compiled-grammar size cap ("grammar too large" 400) — even
# with $defs/$ref dedup. The structural fix uses what the pipeline already
# guarantees: _inherit_dropped_fields backfills ANY field verify omits from
# the pre-verification entries, so the verify schema only needs the fields
# verification can intentionally CHANGE (code identity, sequencing type,
# modifiers, dx links, units, review routing). Everything else (confidence,
# rationale, evidence spans, mdm_details, ...) is restored deterministically
# from passes 1-3 — which is also the safer contract: the audit pass cannot
# accidentally rewrite evidence it didn't act on.
# Descriptions are excluded on purpose even beyond grammar cost:
# _enforce_real_descriptions overwrites every description from the reference
# DB deterministically, so letting verify emit them only re-opens the
# fabricated-descriptor channel it exists to close.
_VERIFY_ICD_ENTRY = _obj({
    "code": _STR,
    "type": _STR,
    "needs_review": _BOOL,
    "review_reason": _NSTR,
})

_VERIFY_PROC_ENTRY = _obj({
    "code": _STR,
    "modifiers": _STR_ARR,
    "modifier_reasoning": _MODIFIER_REASONING,
    "linked_diagnoses": _STR_ARR,
    "units": _INT,
    "needs_review": _BOOL,
    "review_reason": _NSTR,
})

_VERIFY_SNOMED_ENTRY = _obj({
    "concept_id": _STR,
    "description": _STR,
})

# Union budget: 2 (icd review_reason x2) + 2 (proc x2) + 1 (to_code) = 5
VERIFY_PASS_SCHEMA = _obj({
    "corrections_made": {"type": "array", "items": _CORRECTION_ENTRY},
    "icd10_codes": {"type": "array", "items": _VERIFY_ICD_ENTRY},
    "supporting_conditions": {"type": "array", "items": _VERIFY_ICD_ENTRY},
    "cpt_codes": {"type": "array", "items": _VERIFY_PROC_ENTRY},
    "hcpcs_codes": {"type": "array", "items": _VERIFY_PROC_ENTRY},
    "snomed_codes": {"type": "array", "items": _VERIFY_SNOMED_ENTRY},
    "em_level_reasoning": _STR,
    "audit_notes": _STR,
    "auto_coding_review_reasons": _STR_ARR,
    "auto_coding_summary": _STR,
})
