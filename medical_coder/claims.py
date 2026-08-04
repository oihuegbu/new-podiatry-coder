from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import (
    Candidate,
    ClaimContext,
    ClaimLine,
    DecisionState,
    EvidenceFact,
    EvidenceGraph,
    FactStatus,
    GateResult,
    GateStatus,
    RoutingDecision,
)
from .terminology import Artifact, TerminologySnapshot


@dataclass(frozen=True)
class SystemRoles:
    clinical_terminology_systems: frozenset[str]
    procedure_systems: frozenset[str]
    diagnosis_systems: frozenset[str]
    modifier_systems: frozenset[str]
    place_of_service_system: str
    procedure_fact_types: frozenset[str]
    diagnosis_fact_types: frozenset[str]
    modifier_fact_types: frozenset[str]
    quantity_attribute_fields: tuple[str, ...]
    constraint_exception_relations: dict[str, tuple[str, ...]]
    sequencing_directive_fields: frozenset[str]
    default_single_service_units: float

    @classmethod
    def load(cls, path: Path) -> "SystemRoles":
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("schema_version") != 1:
            raise ValueError("unsupported system-role configuration")
        return cls(
            clinical_terminology_systems=frozenset(value["clinical_terminology_systems"]),
            procedure_systems=frozenset(value["procedure_systems"]),
            diagnosis_systems=frozenset(value["diagnosis_systems"]),
            modifier_systems=frozenset(value["modifier_systems"]),
            place_of_service_system=value["place_of_service_system"],
            procedure_fact_types=frozenset(item.casefold() for item in value["procedure_fact_types"]),
            diagnosis_fact_types=frozenset(item.casefold() for item in value["diagnosis_fact_types"]),
            modifier_fact_types=frozenset(item.casefold() for item in value["modifier_fact_types"]),
            quantity_attribute_fields=tuple(value["quantity_attribute_fields"]),
            constraint_exception_relations={
                relation: tuple(term.casefold() for term in terms)
                for relation, terms in value["constraint_exception_relations"].items()
            },
            sequencing_directive_fields=frozenset(value["sequencing_directive_fields"]),
            default_single_service_units=float(value["default_single_service_units"]),
        )


@dataclass(frozen=True)
class ClaimPlan:
    candidates: tuple[Candidate, ...]
    units: dict[str, float]
    gates: dict[str, tuple[GateResult, ...]]
    claim_lines: tuple[ClaimLine, ...]
    routing: RoutingDecision
    ptp_exception_artifact_ids: frozenset[str] = frozenset()


class ClaimPlanner:
    def __init__(self, snapshot: TerminologySnapshot, roles: SystemRoles) -> None:
        self.snapshot = snapshot
        self.roles = roles

    @staticmethod
    def _source_ref(artifact: Artifact) -> str:
        return f"{artifact.source_hash}:{artifact.source_locator}"

    def _candidate(self, fact: EvidenceFact) -> Candidate | None:
        normalization = fact.normalization
        if not normalization or not normalization.target_artifact_id or normalization.alternatives:
            return None
        artifact = self.snapshot.artifact(normalization.target_artifact_id)
        if not artifact:
            return None
        return Candidate(artifact.artifact_id, artifact.system, artifact.version, artifact.display, normalization.confidence, (fact.fact_id,))

    def _facts_by_candidate(self, evidence: EvidenceGraph) -> tuple[dict[str, list[EvidenceFact]], list[EvidenceFact]]:
        grouped: dict[str, list[EvidenceFact]] = {}
        unresolved: list[EvidenceFact] = []
        reportable_types = self.roles.procedure_fact_types | self.roles.diagnosis_fact_types | self.roles.modifier_fact_types
        for fact in evidence.facts:
            if fact.fact_type.casefold() not in reportable_types:
                continue
            candidate = self._candidate(fact)
            if candidate:
                grouped.setdefault(candidate.artifact_id, []).append(fact)
            elif fact.status in {FactStatus.PERFORMED, FactStatus.PRESENT}:
                unresolved.append(fact)
        return grouped, unresolved

    def _units(
        self,
        facts: list[EvidenceFact],
        evidence: EvidenceGraph,
        procedure_fact_ids: set[str],
    ) -> float:
        explicit: list[float] = []
        quantity_fact_ids: set[str] = set()
        for relationship in evidence.relationships:
            if str(relationship.get("relation")) != "quantity_of":
                continue
            source = str(relationship.get("source_fact_id"))
            target = str(relationship.get("target_fact_id"))
            if target in procedure_fact_ids:
                quantity_fact_ids.add(source)
            if source in procedure_fact_ids:
                quantity_fact_ids.add(target)
        unit_facts = facts + [fact for fact in evidence.facts if fact.fact_id in quantity_fact_ids]
        for fact in unit_facts:
            values = [fact.attributes.get(field) for field in self.roles.quantity_attribute_fields]
            for raw in (value for value in values if value not in (None, "")):
                try:
                    value = float(raw)
                except (TypeError, ValueError):
                    raise ValueError("documented units are not numeric")
                if value <= 0:
                    raise ValueError("documented units must be positive")
                explicit.append(value)
        if explicit and len(set(explicit)) != 1:
            raise ValueError("documented units conflict")
        if explicit:
            return explicit[0]
        return self.roles.default_single_service_units

    def _sequencing_gate(self, artifact: Artifact) -> GateResult:
        rows = self.snapshot.semantics(artifact.artifact_id, {"instructional_notes"})
        refs = tuple(f"{row['source_hash']}:{row['source_locator']}" for row in rows)
        directives = []
        for row in rows:
            directives.extend(key for key in row["payload"] if key in self.roles.sequencing_directive_fields and row["payload"].get(key))
        if directives:
            return GateResult(
                "sequencing", GateStatus.FAIL,
                "applicable tabular sequencing directives require an explicit compiled inter-code relationship: " + ", ".join(sorted(set(directives))),
                refs,
            )
        return GateResult("sequencing", GateStatus.PASS, "no unresolved tabular sequencing directive applies", refs or (self._source_ref(artifact),))

    def plan(self, evidence: EvidenceGraph, context: ClaimContext) -> ClaimPlan:
        grouped, unresolved = self._facts_by_candidate(evidence)
        if unresolved:
            ambiguous = [fact for fact in unresolved if fact.normalization and fact.normalization.alternatives]
            route = "PROVIDER_QUERY" if ambiguous else "SOURCE_DATA_REQUIRED"
            reason = "a code-changing documented fact is not uniquely resolved by the authoritative snapshot"
            query = "Please clarify the documented term, anatomy, laterality, or service detail needed to resolve the ambiguous phrase." if ambiguous else None
            return ClaimPlan((), {}, {}, (), RoutingDecision(route, reason, query, tuple(fact.fact_id for fact in unresolved)))

        candidates: list[Candidate] = []
        artifacts: dict[str, Artifact] = {}
        facts_by_id = {fact.fact_id: fact for fact in evidence.facts}
        for artifact_id, facts in grouped.items():
            artifact = self.snapshot.artifact(artifact_id)
            if not artifact:
                continue
            role_types = self.roles.procedure_fact_types if artifact.system in self.roles.procedure_systems else self.roles.diagnosis_fact_types if artifact.system in self.roles.diagnosis_systems else self.roles.modifier_fact_types
            supporting = tuple(sorted(fact.fact_id for fact in facts if fact.fact_type.casefold() in role_types))
            if not supporting:
                continue
            artifacts[artifact_id] = artifact
            if artifact.system not in self.roles.modifier_systems:
                candidates.append(Candidate(artifact_id, artifact.system, artifact.version, artifact.display, 1.0, supporting))

        procedures = [artifact for artifact in artifacts.values() if artifact.system in self.roles.procedure_systems]
        diagnoses = [artifact for artifact in artifacts.values() if artifact.system in self.roles.diagnosis_systems]
        modifiers = [artifact for artifact in artifacts.values() if artifact.system in self.roles.modifier_systems]
        if not procedures or not diagnoses:
            missing_role = "performed service" if not procedures else "documented diagnosis"
            return ClaimPlan(
                tuple(candidates), {}, {}, (),
                RoutingDecision(
                    "PROVIDER_QUERY",
                    f"no uniquely resolved {missing_role} is available for claim construction",
                    f"Please document or clarify the {missing_role} needed for this claim.",
                ),
            )
        units: dict[str, float] = {}
        gates: dict[str, tuple[GateResult, ...]] = {}
        artifact_for_fact = {
            fact.fact_id: fact.normalization.target_artifact_id
            for fact in evidence.facts if fact.normalization and fact.normalization.target_artifact_id
        }
        related_diagnoses: dict[str, set[str]] = {artifact.artifact_id: set() for artifact in procedures}
        related_modifiers: dict[str, set[str]] = {artifact.artifact_id: set() for artifact in procedures}
        diagnosis_ids = {item.artifact_id for item in diagnoses}
        modifier_ids = {item.artifact_id for item in modifiers}
        integral_fact_ids: set[str] = set()
        exception_relations: dict[str, set[str]] = {artifact.artifact_id: set() for artifact in procedures}
        for relationship in evidence.relationships:
            source_fact = str(relationship.get("source_fact_id"))
            target_fact = str(relationship.get("target_fact_id"))
            relation = str(relationship.get("relation"))
            source_artifact = artifact_for_fact.get(source_fact)
            target_artifact = artifact_for_fact.get(target_fact)
            if relation in {"component_of", "integral_to"}:
                integral_fact_ids.add(source_fact)
            if relation in self.roles.constraint_exception_relations:
                if source_artifact in exception_relations:
                    exception_relations[source_artifact].add(relation)
                if target_artifact in exception_relations:
                    exception_relations[target_artifact].add(relation)
            if relation == "diagnosis_supports_service":
                if source_artifact in diagnosis_ids and target_artifact in related_diagnoses:
                    related_diagnoses[target_artifact].add(source_artifact)
                if target_artifact in diagnosis_ids and source_artifact in related_diagnoses:
                    related_diagnoses[source_artifact].add(target_artifact)
            if relation in {"laterality_of", "modifier_applies_to"}:
                if source_artifact in modifier_ids and target_artifact in related_modifiers:
                    related_modifiers[target_artifact].add(source_artifact)
                if target_artifact in modifier_ids and source_artifact in related_modifiers:
                    related_modifiers[source_artifact].add(target_artifact)
        modifier_assignments = {
            modifier_id: {procedure_id for procedure_id, assigned in related_modifiers.items() if modifier_id in assigned}
            for modifier_id in modifier_ids
        }
        ambiguous_modifiers = {modifier_id for modifier_id, assigned in modifier_assignments.items() if len(assigned) != 1}
        if ambiguous_modifiers:
            affected = tuple(
                sorted(
                    fact.fact_id for modifier_id in ambiguous_modifiers
                    for fact in grouped.get(modifier_id, [])
                )
            )
            return ClaimPlan(
                tuple(candidates), {}, {}, (),
                RoutingDecision(
                    "PROVIDER_QUERY",
                    "documented modifier evidence is not uniquely related to one reported service",
                    "Please clarify which documented service the modifier evidence applies to.",
                    affected,
                ),
            )
        ptp_exception_artifact_ids: set[str] = set()
        for procedure_id, relations in exception_relations.items():
            for modifier_id in related_modifiers[procedure_id]:
                descriptor_tokens = {
                    token.casefold().strip(".,;:()[]{}")
                    for token in artifacts[modifier_id].display.split()
                }
                if any(
                    descriptor_tokens.intersection(self.roles.constraint_exception_relations[relation])
                    for relation in relations
                ):
                    ptp_exception_artifact_ids.add(procedure_id)
                    break
        place = self.snapshot.artifact_by_code(self.roles.place_of_service_system, context.place_of_service, context.date_of_service)
        context_gate = GateResult(
            "claim_context", GateStatus.PASS if place else GateStatus.FAIL,
            "place of service is active in the authoritative context set" if place else "place of service is not active in the authoritative context set",
            (self._source_ref(place),) if place else (),
        )
        for candidate in candidates:
            artifact = artifacts[candidate.artifact_id]
            supporting_facts = [facts_by_id[fact_id] for fact_id in candidate.supporting_fact_ids]
            documentation = GateResult(
                "documentation", GateStatus.PASS if all(fact.source_spans for fact in supporting_facts) else GateStatus.FAIL,
                "every claim-driving fact has immutable source spans",
                tuple(f"{span.document_id}:{span.start_offset}-{span.end_offset}" for fact in supporting_facts for span in fact.source_spans),
            )
            reportable_status = FactStatus.PERFORMED if artifact.system in self.roles.procedure_systems else FactStatus.PRESENT
            reportability = GateResult(
                "reportability",
                GateStatus.PASS if any(fact.status == reportable_status for fact in supporting_facts) and not any(fact.fact_id in integral_fact_ids for fact in supporting_facts) else GateStatus.FAIL,
                "fact status and relationship are independently reportable" if any(fact.status == reportable_status for fact in supporting_facts) and not any(fact.fact_id in integral_fact_ids for fact in supporting_facts) else "fact is documented but not independently reportable in this claim role",
                (self._source_ref(artifact),),
            )
            if artifact.system in self.roles.procedure_systems:
                units[candidate.artifact_id] = self._units(
                    supporting_facts, evidence, set(candidate.supporting_fact_ids)
                )
                linked_diagnoses = [item for item in diagnoses if item.artifact_id in related_diagnoses[artifact.artifact_id]]
                coverage = self.snapshot.coverage(
                    artifact, linked_diagnoses, context.date_of_service, context.jurisdiction,
                    context.payer_identifier, context.payer_type, context.contract_profile,
                )
                coverage_gate = GateResult("coverage", GateStatus.PASS if coverage.status == "pass" else GateStatus.FAIL, coverage.reason, coverage.source_refs)
                necessity = GateResult(
                    "medical_necessity",
                    GateStatus.PASS if coverage.status == "pass" and bool(linked_diagnoses) else GateStatus.FAIL,
                    "documented diagnosis is explicitly related to the service and linked by applicable policy" if coverage.status == "pass" and linked_diagnoses else "medical necessity is not established by an evidence relationship plus compiled policy",
                    coverage.source_refs,
                )
                unit_gate = GateResult("units", GateStatus.PASS, "units are derived from documented quantity or configured single-service mechanics", (self._source_ref(artifact),))
                modifier_gate = GateResult(
                    "modifiers", GateStatus.PASS,
                    "modifier evidence is absent or uniquely related to its service",
                    tuple(self._source_ref(item) for item in modifiers if item.artifact_id in related_modifiers[artifact.artifact_id]),
                )
            else:
                coverage_gate = GateResult("coverage", GateStatus.PASS, "coverage is evaluated on procedure-diagnosis claim relationships", (self._source_ref(artifact),))
                necessity = GateResult("medical_necessity", GateStatus.PASS, "diagnosis establishes claim evidence and is linked at the procedure gate", (self._source_ref(artifact),))
                unit_gate = GateResult("units", GateStatus.PASS, "diagnosis classification does not carry service units", (self._source_ref(artifact),))
                modifier_gate = GateResult("modifiers", GateStatus.PASS, "no modifier applies to this classification role", (self._source_ref(artifact),))
            gates[candidate.artifact_id] = (
                documentation, reportability, coverage_gate, necessity, unit_gate, modifier_gate, self._sequencing_gate(artifact), context_gate
            )

        lines = tuple(
            ClaimLine(
                artifact.artifact_id,
                tuple(sorted(related_diagnoses[artifact.artifact_id])),
                units.get(artifact.artifact_id, self.roles.default_single_service_units),
                tuple(sorted(related_modifiers[artifact.artifact_id])),
                DecisionState.SUPPORTED_REPORTABLE,
                next(candidate.supporting_fact_ids for candidate in candidates if candidate.artifact_id == artifact.artifact_id),
            )
            for artifact in sorted(procedures, key=lambda item: item.artifact_id)
        )
        route = RoutingDecision("AUTONOMOUS", "all code-changing facts resolved from the authoritative snapshot")
        return ClaimPlan(
            tuple(candidates), units, gates, lines, route,
            frozenset(ptp_exception_artifact_ids),
        )
