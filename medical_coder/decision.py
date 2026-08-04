from __future__ import annotations

from dataclasses import asdict
from typing import Iterable

from .models import (
    Candidate,
    CandidateDecision,
    ClaimContext,
    DecisionState,
    EvidenceGraph,
    FactStatus,
    GateResult,
    GateStatus,
)
from .predicates import UnsupportedPredicate, entails
from .terminology import Artifact, TerminologySnapshot


MANDATORY_GATES = (
    "evidence",
    "documentation",
    "autonomy_scope",
    "source_identity",
    "effective_date",
    "classification",
    "descriptor_entailment",
    "specificity",
    "constraints",
    "units",
    "modifiers",
    "sequencing",
    "reportability",
    "medical_necessity",
    "claim_context",
    "coverage",
    "provenance",
)


class DecisionEngine:
    def __init__(self, snapshot: TerminologySnapshot) -> None:
        self.snapshot = snapshot

    def evaluate(
        self,
        evidence: EvidenceGraph,
        context: ClaimContext,
        candidates: Iterable[Candidate],
        *,
        required_capabilities: set[str],
        units: dict[str, float] | None = None,
        coverage_evidence: dict[str, tuple[str, ...]] | None = None,
        scope_eligible: bool = True,
        scope_reason: str = "eligible for configured autonomy scope",
        supplemental_gates: dict[str, tuple[GateResult, ...]] | None = None,
        ptp_exception_artifact_ids: frozenset[str] = frozenset(),
    ) -> tuple[CandidateDecision, ...]:
        candidate_list = tuple(candidates)
        unit_map = units or {}
        artifacts = {
            candidate.artifact_id: self.snapshot.artifact(candidate.artifact_id)
            for candidate in candidate_list
        }
        selected_codes = {
            artifact.code
            for artifact_id, artifact in artifacts.items()
            if artifact is not None and artifact_id in unit_map
        }
        available = {
            name
            for name, enabled in self.snapshot.manifest.get("capabilities", {}).items()
            if enabled is True
        }
        missing_capabilities = sorted(required_capabilities - available)
        ptp_exception_codes = {
            exception_artifact.code
            for artifact_id in ptp_exception_artifact_ids
            if (exception_artifact := self.snapshot.artifact(artifact_id)) is not None
        }
        decisions = []
        for candidate in candidate_list:
            artifact = artifacts[candidate.artifact_id]
            decisions.append(
                self._evaluate_one(
                    evidence,
                    context,
                    candidate,
                    artifact,
                    selected_codes,
                    missing_capabilities,
                    unit_map,
                    coverage_evidence or {},
                    scope_eligible,
                    scope_reason,
                    supplemental_gates or {},
                    ptp_exception_codes,
                )
            )
        return tuple(decisions)

    def _evaluate_one(
        self,
        evidence: EvidenceGraph,
        context: ClaimContext,
        candidate: Candidate,
        artifact: Artifact | None,
        selected_codes: set[str],
        missing_capabilities: list[str],
        units: dict[str, float],
        coverage_evidence: dict[str, tuple[str, ...]],
        scope_eligible: bool,
        scope_reason: str,
        supplemental_gates: dict[str, tuple[GateResult, ...]],
        ptp_exception_codes: set[str],
    ) -> CandidateDecision:
        facts = [fact for fact in evidence.facts if fact.fact_id in candidate.supporting_fact_ids]
        gates: list[GateResult] = []
        gates.append(
            self._gate(
                "autonomy_scope",
                scope_eligible,
                scope_reason,
                scope_reason,
            )
        )
        performed = [fact for fact in facts if fact.status in {FactStatus.PERFORMED, FactStatus.PRESENT}]
        gates.append(
            self._gate(
                "evidence",
                bool(performed and all(fact.source_spans for fact in performed)),
                "performed facts have traceable source spans",
                "no performed source-backed fact supports this candidate",
            )
        )
        gates.append(
            self._gate(
                "source_identity",
                artifact is not None,
                "candidate exists in the immutable snapshot",
                "candidate is outside the immutable snapshot",
            )
        )
        if artifact is None:
            return self._finish(candidate, gates, DecisionState.SOURCE_UNAVAILABLE)

        gates.append(
            self._gate(
                "effective_date",
                self.snapshot.active_on(artifact, context.date_of_service),
                "candidate is active for the date of service",
                "candidate is inactive for the date of service",
                (self._source_ref(artifact),),
            )
        )
        gates.append(
            self._gate(
                "classification",
                artifact.billable and artifact.status == "active",
                "source classifies candidate as active and billable",
                "source does not classify candidate as active and billable",
                (self._source_ref(artifact),),
            )
        )

        entailed = False
        entailment_reason = "authoritative descriptor requirements are unavailable"
        if artifact.requirements:
            try:
                entailed = entails(performed, artifact.requirements)
                entailment_reason = (
                    "all compiled descriptor requirements are established"
                    if entailed
                    else "one or more compiled descriptor requirements are not established"
                )
            except UnsupportedPredicate:
                entailment_reason = "source requirement uses an unsupported predicate"
        else:
            material_tokens = self.snapshot.material_descriptor_tokens(artifact.artifact_id)
            confirming_facts = [
                fact for fact in performed
                if fact.normalization
                and fact.normalization.target_artifact_id == artifact.artifact_id
                and not fact.normalization.alternatives
            ]
            evidence_tokens = {
                token
                for fact in confirming_facts
                for token in self.snapshot.normalized_phrase(
                    " ".join((fact.text, *map(str, fact.attributes.values())))
                ).split()
            }
            exact_descriptor = any(
                "exact_authoritative_descriptor" in fact.normalization.resolution_factors
                for fact in confirming_facts if fact.normalization
            )
            derived_confirmation = any(
                "all_source_derived_material_descriptor_tokens_entailed" in fact.normalization.resolution_factors
                for fact in confirming_facts if fact.normalization
            )
            entailed = bool(confirming_facts) and (
                exact_descriptor
                or derived_confirmation
                or bool(material_tokens and material_tokens.issubset(evidence_tokens))
            )
            if entailed:
                entailment_reason = "licensed descriptor is independently confirmed by source-grounded evidence"
            elif confirming_facts:
                entailment_reason = "index or synonym retrieval is not independent descriptor confirmation"
        gates.append(
            GateResult(
                "descriptor_entailment",
                GateStatus.PASS if entailed else GateStatus.FAIL,
                entailment_reason,
                (self._source_ref(artifact),),
            )
        )
        specificity = entailed and any(
            fact.normalization
            and fact.normalization.target_artifact_id == artifact.artifact_id
            and fact.normalization.source == f"snapshot:{self.snapshot.snapshot_id}"
            and not fact.normalization.alternatives
            for fact in performed
        )
        gates.append(
            self._gate(
                "specificity",
                specificity,
                "source-validated identity resolves uniquely",
                "hierarchy or uniquely resolved identity is unavailable",
                (self._source_ref(artifact),),
            )
        )

        constraints_ok, constraint_reason, constraint_refs = self._constraints(
            artifact,
            selected_codes,
            context,
            units.get(candidate.artifact_id),
            ptp_exception_codes,
        )
        if missing_capabilities:
            constraints_ok = False
            constraint_reason = "required source capabilities unavailable: " + ", ".join(missing_capabilities)
        gates.append(
            self._gate(
                "constraints",
                constraints_ok,
                constraint_reason,
                constraint_reason,
                constraint_refs,
            )
        )
        missing_context = context.missing_fields()
        gates.append(
            self._gate(
                "claim_context",
                not missing_context,
                "claim context is complete",
                "missing claim context: " + ", ".join(missing_context),
            )
        )
        supplied = {gate.name: gate for gate in supplemental_gates.get(candidate.artifact_id, ())}
        if "coverage" in supplied:
            gates.append(supplied.pop("coverage"))
            coverage_refs = gates[-1].source_refs if gates[-1].status == GateStatus.PASS else ()
        else:
            coverage_refs = coverage_evidence.get(candidate.artifact_id, ())
            gates.append(
                self._gate(
                    "coverage",
                    bool(coverage_refs),
                    "applicable coverage policy was evaluated",
                    "applicable coverage policy was not deterministically evaluated",
                    coverage_refs,
                )
            )
        gates.extend(supplied.values())
        gates.append(
            self._gate(
                "provenance",
                bool(artifact.source_hash and self.snapshot.snapshot_id),
                "source and snapshot hashes are present",
                "source or snapshot provenance is incomplete",
                (self._source_ref(artifact),),
            )
        )
        if not scope_eligible:
            state = DecisionState.POLICY_REVIEW
        elif not self.snapshot.active_on(artifact, context.date_of_service):
            state = DecisionState.CANDIDATE_INACTIVE
        elif missing_context:
            state = DecisionState.CLAIM_CONTEXT_REQUIRED
        elif missing_capabilities:
            state = DecisionState.SOURCE_UNAVAILABLE
        elif not entailed:
            state = DecisionState.CANDIDATE_NOT_ENTAILED
        elif not constraints_ok:
            state = DecisionState.SOURCE_CONFLICT
        elif any(gate.status == GateStatus.FAIL for gate in gates if gate.name in MANDATORY_GATES):
            state = DecisionState.SOURCE_UNAVAILABLE
        else:
            state = DecisionState.SUPPORTED_REPORTABLE
        return self._finish(candidate, gates, state)

    def _constraints(
        self,
        artifact: Artifact,
        selected_codes: set[str],
        context: ClaimContext,
        units: float | None,
        ptp_exception_codes: set[str],
    ) -> tuple[bool, str, tuple[str, ...]]:
        if units is None:
            return True, "candidate is not a service-line constraint target", ()
        rows = self.snapshot.constraints(
            ("cannot_coexist", "ptp_edit", "unit_limit", "add_on_relationship"),
            selected_codes,
            context.date_of_service,
        )
        refs: list[str] = []
        for row in rows:
            refs.append(f"{row['source_hash']}:{row['source_locator']}")
            if row["kind"] == "cannot_coexist":
                if row["left_code"] in selected_codes and row["right_code"] in selected_codes:
                    return False, "authoritative incompatibility applies", tuple(refs)
            if row["kind"] == "ptp_edit":
                if row["left_code"] in selected_codes and row["right_code"] in selected_codes:
                    indicator = str(row["properties"].get("modifier_indicator", "")).strip()
                    if indicator == "9":
                        continue
                    if indicator == "1":
                        if row["right_code"] in ptp_exception_codes:
                            continue
                        return False, "pair edit requires independently established modifier-exception evidence", tuple(refs)
                    return False, "pair edit prohibits the reported combination", tuple(refs)
            if row["kind"] == "unit_limit" and row["target_code"] == artifact.code:
                if units is None:
                    return False, "units are required for an applicable unit limit", tuple(refs)
                if row["numeric_value"] is not None and units > row["numeric_value"]:
                    return False, "reported units exceed the authoritative limit", tuple(refs)
            if row["kind"] == "add_on_relationship" and row["right_code"] == artifact.code:
                if row["left_code"] not in selected_codes:
                    return False, "required primary relationship is not satisfied", tuple(refs)
        return True, "all available deterministic constraints pass", tuple(refs)

    @staticmethod
    def _source_ref(artifact: Artifact) -> str:
        return f"{artifact.source_hash}:{artifact.source_locator}"

    @staticmethod
    def _gate(
        name: str,
        passed: bool,
        pass_reason: str,
        fail_reason: str,
        source_refs: tuple[str, ...] = (),
    ) -> GateResult:
        return GateResult(name, GateStatus.PASS if passed else GateStatus.FAIL, pass_reason if passed else fail_reason, source_refs)

    @staticmethod
    def _finish(
        candidate: Candidate,
        gates: list[GateResult],
        state: DecisionState,
    ) -> CandidateDecision:
        present = {gate.name for gate in gates}
        for name in MANDATORY_GATES:
            if name not in present:
                gates.append(GateResult(name, GateStatus.FAIL, "evaluation stopped before this gate"))
        release = state == DecisionState.SUPPORTED_REPORTABLE and all(
            gate.status == GateStatus.PASS for gate in gates if gate.name in MANDATORY_GATES
        )
        return CandidateDecision(candidate.artifact_id, state, tuple(gates), release)

