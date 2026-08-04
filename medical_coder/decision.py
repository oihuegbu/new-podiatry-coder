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
    "autonomy_scope",
    "source_identity",
    "effective_date",
    "classification",
    "descriptor_entailment",
    "specificity",
    "constraints",
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
    ) -> tuple[CandidateDecision, ...]:
        candidate_list = tuple(candidates)
        artifacts = {
            candidate.artifact_id: self.snapshot.artifact(candidate.artifact_id)
            for candidate in candidate_list
        }
        selected_codes = {artifact.code for artifact in artifacts.values() if artifact is not None}
        available = {
            name
            for name, enabled in self.snapshot.manifest.get("capabilities", {}).items()
            if enabled is True
        }
        missing_capabilities = sorted(required_capabilities - available)
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
                    units or {},
                    coverage_evidence or {},
                    scope_eligible,
                    scope_reason,
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
        performed = [fact for fact in facts if fact.status == FactStatus.PERFORMED]
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
            normalized_ids = {
                fact.normalization.concept_id
                for fact in performed
                if fact.normalization
                and fact.normalization.source == f"snapshot:{self.snapshot.snapshot_id}"
                and not fact.normalization.alternatives
            }
            identity = artifact.concept_id or artifact.artifact_id
            entailed = identity in normalized_ids
            if entailed:
                entailment_reason = "unique exact terminology identity is source-validated"
        gates.append(
            GateResult(
                "descriptor_entailment",
                GateStatus.PASS if entailed else GateStatus.FAIL,
                entailment_reason,
                (self._source_ref(artifact),),
            )
        )
        specificity = entailed and (
            bool(artifact.requirements)
            or any(
                fact.normalization
                and fact.normalization.source == f"snapshot:{self.snapshot.snapshot_id}"
                and not fact.normalization.alternatives
                for fact in performed
            )
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
        elif not coverage_refs:
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
    ) -> tuple[bool, str, tuple[str, ...]]:
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

