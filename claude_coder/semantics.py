"""Compiled semantic records — one generic, versioned identity record per authoritative
code, assembled from data already loaded by `AuthoritativeSource` (issue #6,
compiled-semantic-layer plan item 1).

WHY THIS EXISTS
----------------
Retrieval used to search a fact's whole code system broadly and only narrow candidates
AFTER retrieval, in `tiebreak.narrow` — descriptor-derived, but running too late to keep
an irrelevant code family (a different anatomy, a different action entirely) out of the
pool it ever had to rank against. This module compiles, for one code, the same kind of
descriptor-grammar structure `tiebreak`/`ontology` already derive — but as a standing,
typed record any caller can check compatibility against BEFORE retrieval runs, not only
after a tie.

NOTHING HERE IS COMPUTED FRESH. Every field is read from data `AuthoritativeSource`
already loads for its own existing purposes (`ontology.parse_descriptor`'s grammar
parser, `semantic_class`/`component_relationships` added alongside this module,
`global_period`/`bilat_indicator`'s existing CMS PFS fields) or is honestly left
empty/unknown when no authoritative field answers it — never filled with a fixed
clinical vocabulary or code list to make the record LOOK complete.

Agnostic: no medical code or clinical term is written here; every value is read through
`source`.
"""
from __future__ import annotations

from typing import Any

from . import ontology as _ontology

def _laterality_behavior(code: str, system: str, descriptor: str, source: Any) -> str:
    """Prefer the CMS PFS bilateral-surgery indicator (a real authoritative field,
    CPT/HCPCS only) over the descriptor's own laterality wording — the PFS indicator
    says whether modifier 50 actually applies, which the descriptor's use of the word
    "bilateral" does not by itself guarantee."""
    if system in ("cpt", "hcpcs"):
        bilat = getattr(source, "bilat_indicator", None)
        if callable(bilat):
            try:
                indicator = bilat(code)
            except Exception:
                indicator = None
            if indicator:
                return f"pfs_bilateral_indicator:{indicator}"
    words = _ontology.parse_descriptor(descriptor).laterality
    return "/".join(sorted(words)) if words else ""


def _unit_behavior(descriptor: str) -> str:
    feats = _ontology.parse_descriptor(descriptor)
    if feats.cardinality:
        return feats.cardinality
    if feats.interval and feats.interval.bounded():
        return "range"
    return "single"


def _required_attributes(descriptor: str) -> list[str]:
    """Attributes the descriptor's own grammar structurally requires to pick THIS
    code over a sibling — currently only a measurement interval (issue #6 item 1:
    honestly minimal; not every distinguishing attribute has a mechanical detector
    yet, so nothing is asserted here that isn't actually derivable)."""
    feats = _ontology.parse_descriptor(descriptor)
    return ["measurement"] if feats.interval and feats.interval.bounded() else []


def _component_relationships_list(code: str, system: str, source: Any) -> list[dict]:
    getter = getattr(source, "component_relationships", None)
    if not callable(getter):
        return []
    try:
        rels = getter(code, system) or {}
    except Exception:
        return []
    return [{"type": rel_type, "target_code_refs": sorted(refs)}
            for rel_type, refs in sorted(rels.items()) if refs]


def _effective_period(record: dict) -> dict[str, str]:
    start = record.get("effective_from") or record.get("effective_date") or ""
    end = record.get("effective_to") or ""
    return {"from": str(start), "to": str(end)}


#: issue #6 item 1: ICD-10-CM codes are, by the structure of that code SYSTEM itself
#: (not a clinical judgement about any particular code), always diagnoses. Analogous
#: to how `fact.system` already names cpt/hcpcs/icd10 as a structural identifier
#: throughout this package.
_SYSTEM_SEMANTIC_CLASS = {"icd10": "diagnosis"}


def compiled_record(code: str, system: str, source: Any,
                    source_identity: dict[str, Any] | None = None) -> dict[str, Any]:
    """The generic semantic record for one code (issue #6 item 1's schema):

        code, system, semantic_class, action_concepts, anatomy_concepts, approach,
        temporal_role, required_attributes, optional_attributes, excluded_attributes,
        component_relationships, laterality_behavior, unit_behavior, source_identity,
        effective_period

    Returns a record with every field present but many legitimately empty/None when
    no authoritative field answers them (`temporal_role`, `optional_attributes`,
    `excluded_attributes`, and CPT/HCPCS `component_relationships` have no wired
    source yet) -- absence is the honest answer, never a guess. Returns None when the
    code itself is not in the authoritative reference (nothing to compile a record
    FROM), which the caller must treat the same as any other authoritative-lookup
    miss, not as "this code has no semantics".

    A NOTE FOR THE NEXT CONSUMER (item 4, semantic eligibility-before-retrieval):
    this function deliberately never raises -- every source call is caught and
    degraded to an honest empty/None field, so one missing table (say
    `coding_semantics.json`) never takes down `component_relationships` and
    `anatomy_concepts` along with it. That means `semantic_class: None` here is
    AMBIGUOUS between two different situations a fail-closed eligibility gate must
    NOT treat the same way: "this code's descriptor/global-period genuinely
    matched no configured rule" (`AuthoritativeSource.semantic_class` returning
    `None` cleanly) vs. "the classifier config itself could not be read"
    (`SemanticClassUnavailable`, which `AuthoritativeSource.semantic_class` itself
    is documented to raise rather than degrade, precisely so a caller CAN choose to
    hold on it). A caller that needs that distinction should call
    `source.semantic_class(code, system)` directly rather than reading it off this
    record.
    """
    lookup = getattr(source, "lookup", None)
    record = lookup(code, system) if callable(lookup) else None
    if not isinstance(record, dict):
        return None
    descriptor = str(record.get("long_description") or record.get("description")
                     or record.get("short_description") or "")

    if system == "icd10":
        feats = _ontology.parse_descriptor(descriptor)
        action_concepts: list[str] = []
        anatomy_concepts = sorted(feats.core_tokens)
        semantic_class = _SYSTEM_SEMANTIC_CLASS["icd10"]
        laterality_behavior = "/".join(sorted(feats.laterality)) if feats.laterality else ""
        unit_behavior = ""
        required_attributes: list[str] = []
        component_relationships = _component_relationships_list(code, system, source)
    else:
        feats = _ontology.parse_descriptor(descriptor)
        action_concepts = sorted(feats.action_tokens)
        anatomy_concepts = sorted(feats.anatomy_tokens)
        classifier = getattr(source, "semantic_class", None)
        try:
            semantic_class = classifier(code, system) if callable(classifier) else None
        except Exception:
            semantic_class = None
        laterality_behavior = _laterality_behavior(code, system, descriptor, source)
        unit_behavior = _unit_behavior(descriptor)
        required_attributes = _required_attributes(descriptor)
        # NCCI PTP edits are the real CPT/HCPCS analogue of a component relationship,
        # but they are inherently PAIRWISE (code A vs. code B), not a per-code
        # enumerable list the way ICD-10's Excludes/codeFirst/useAdditionalCode notes
        # are -- checking them belongs at eligibility/arbitration time against a
        # specific other candidate (where `ncci_indicator`/`ncci_edit` already run),
        # not baked into one code's own static record. Honestly empty here.
        component_relationships = []

    # CALLER-SUPPLIED, deliberately not computed here: `source.data_fingerprint()`
    # rebuilds the WHOLE capability manifest fresh on every call (every declared
    # file's identity, counts, checksums) -- it is not cached on `source`, so
    # calling it once per code would mean once per CANDIDATE, in what is meant to
    # become a per-encounter eligibility-narrowing hot path. The caller computes
    # it once per encounter (the same value the certificate's own `AuthorityBinding`
    # already carries) and passes it down; an omitted identity is honestly empty,
    # never silently recomputed at a cost this function's caller did not ask for.
    if source_identity is None:
        source_identity = {}

    return {
        "code": code,
        "system": system,
        "semantic_class": semantic_class,
        "action_concepts": action_concepts,
        "anatomy_concepts": anatomy_concepts,
        # No provenance-bound structured approach field is currently available.
        # Descriptor word scans are not a governed substitute; absence stays honest.
        "approach": [],
        "temporal_role": [],
        "required_attributes": required_attributes,
        "optional_attributes": [],
        "excluded_attributes": [],
        "component_relationships": component_relationships,
        "laterality_behavior": laterality_behavior,
        "unit_behavior": unit_behavior,
        "source_identity": source_identity,
        "effective_period": _effective_period(record),
    }
