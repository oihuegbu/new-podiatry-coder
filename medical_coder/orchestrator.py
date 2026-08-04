from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from .audit import AuditStore
from .certificate import build_certificate, stable_hash
from .claims import ClaimPlanner, SystemRoles
from .decision import DecisionEngine
from .documents import DocumentIngestor
from .extraction import normalized_consensus
from .models import CodingResult, ClaimContext, RoutingDecision
from .providers import IndependentExtractor, load_profiles
from .scope import AutonomyScope
from .serde import coding_result_from_dict
from .terminology import TerminologySnapshot


class CodingWorkflow:
    def __init__(
        self,
        snapshot_directory: Path,
        provider_config: Path,
        roles_config: Path,
        scope_config: Path,
        audit_database: Path,
        *,
        ocr_command: tuple[str, ...] | None = None,
    ) -> None:
        self.snapshot = TerminologySnapshot(snapshot_directory)
        self.roles = SystemRoles.load(roles_config)
        self.scope = AutonomyScope(scope_config)
        self.extractor = IndependentExtractor(load_profiles(provider_config))
        self.ingestor = DocumentIngestor(ocr_command)
        self.audit = AuditStore(audit_database)

    def _clinical_normalize(self, graph, service_date):
        clinical_types = self.roles.procedure_fact_types | self.roles.diagnosis_fact_types
        facts = tuple(
            self.snapshot.normalize_fact(
                replace(fact, normalization=None),
                service_date,
                set(self.roles.clinical_terminology_systems),
            )
            if fact.fact_type.casefold() in clinical_types else fact
            for fact in graph.facts
        )
        return replace(graph, facts=facts)

    def _billing_normalize(self, evidence, service_date):
        facts = []
        for fact in evidence.facts:
            fact_type = fact.fact_type.casefold()
            if fact_type in self.roles.procedure_fact_types:
                systems = set(self.roles.procedure_systems)
            elif fact_type in self.roles.diagnosis_fact_types:
                systems = set(self.roles.diagnosis_systems)
            elif fact_type in self.roles.modifier_fact_types:
                systems = set(self.roles.modifier_systems)
            else:
                facts.append(fact)
                continue
            facts.append(
                self.snapshot.normalize_fact(
                    replace(fact, normalization=None), service_date, systems
                )
            )
        return replace(evidence, facts=tuple(facts))

    def code(self, encounter_id: str, document_path: Path, context: ClaimContext) -> CodingResult:
        document = self.ingestor.ingest(document_path, document_id=encounter_id)
        context_hash = stable_hash(context)
        self.audit.begin(encounter_id, document.source_sha256, self.snapshot.snapshot_id, context_hash)
        existing_result = self.audit.result(encounter_id)
        if existing_result:
            self.audit.verify_chain(encounter_id)
            return coding_result_from_dict(existing_result)
        self.audit.append(encounter_id, "DOCUMENT_INGESTED", {
            "source_hash": document.source_sha256,
            "text_hash": document.text_sha256,
            "page_count": len(document.pages),
            "extractor": document.extractor,
        })
        scope = self.scope.evaluate(context)
        self.audit.append(encounter_id, "SCOPE_EVALUATED", asdict(scope))
        graphs = self.extractor.extract_independent(document)
        self.audit.append(encounter_id, "INDEPENDENT_EXTRACTION_COMPLETE", {
            "provider_count": len(graphs),
            "fact_counts": [len(graph.facts) for graph in graphs],
            "graph_hashes": [stable_hash(graph) for graph in graphs],
        })
        clinical_graphs = [
            self._clinical_normalize(graph, context.date_of_service) for graph in graphs
        ]
        self.audit.append(encounter_id, "CLINICAL_TERMINOLOGY_NORMALIZED", {
            "system_scope": sorted(self.roles.clinical_terminology_systems),
            "providers": [
                {
                    "graph_hash": stable_hash(graph),
                    "normalizations": [
                        {
                            "fact_id": fact.fact_id,
                            "raw_phrase": fact.normalization.raw_phrase,
                            "concept_id": fact.normalization.concept_id,
                            "confidence": fact.normalization.confidence,
                            "alternatives": fact.normalization.alternatives,
                            "candidate_expansions": fact.normalization.candidate_expansions,
                            "resolution_factors": fact.normalization.resolution_factors,
                        }
                        for fact in graph.facts if fact.normalization
                    ],
                }
                for graph in clinical_graphs
            ],
        })
        outcome = normalized_consensus(clinical_graphs)
        if not outcome.evidence:
            empty = clinical_graphs[0]
            certificate = build_certificate(encounter_id, self.snapshot.snapshot_id, empty, context, ())
            result = CodingResult(
                encounter_id, "EXTRACTION_CONFLICT", (),
                RoutingDecision("AUTOMATED_RETRY", "independent providers disagree on a code-changing fact", None, ()),
                certificate, outcome.conflicts,
            )
            self.audit.finish(encounter_id, result.state, asdict(result))
            self.audit.verify_chain(encounter_id)
            return result
        evidence = self._billing_normalize(outcome.evidence, context.date_of_service)
        self.audit.append(encounter_id, "EVIDENCE_CONSENSUS", {
            "evidence_hash": stable_hash(evidence),
            "fact_count": len(evidence.facts),
            "normalization_sources": sorted({fact.normalization.source for fact in evidence.facts if fact.normalization}),
        })
        plan = ClaimPlanner(self.snapshot, self.roles).plan(evidence, context)
        if plan.routing.route != "AUTONOMOUS":
            certificate = build_certificate(encounter_id, self.snapshot.snapshot_id, evidence, context, ())
            result = CodingResult(encounter_id, plan.routing.route, (), plan.routing, certificate)
            self.audit.finish(encounter_id, result.state, asdict(result))
            self.audit.verify_chain(encounter_id)
            return result
        required_capabilities = set(scope.required_capabilities)
        decisions = DecisionEngine(self.snapshot).evaluate(
            evidence,
            context,
            plan.candidates,
            required_capabilities=required_capabilities,
            units=plan.units,
            scope_eligible=scope.eligible,
            scope_reason=scope.reason,
            supplemental_gates=plan.gates,
            ptp_exception_artifact_ids=plan.ptp_exception_artifact_ids,
        )
        certificate = build_certificate(encounter_id, self.snapshot.snapshot_id, evidence, context, decisions)
        released = {decision.artifact_id for decision in decisions if decision.autonomous_release}
        lines = tuple(
            line for line in plan.claim_lines
            if line.artifact_id in released and all(diagnosis in released for diagnosis in line.diagnosis_artifact_ids)
        )
        all_claim_candidates = {line.artifact_id for line in plan.claim_lines} | {
            diagnosis for line in plan.claim_lines for diagnosis in line.diagnosis_artifact_ids
        }
        state = "AUTONOMOUS_RELEASE" if lines and all_claim_candidates <= released else "POLICY_REVIEW"
        if state == "AUTONOMOUS_RELEASE":
            routing = RoutingDecision("AUTONOMOUS", "all mandatory deterministic gates passed")
        else:
            failed = {
                gate.name
                for decision in decisions for gate in decision.gates
                if gate.status.value == "fail"
            }
            source_gates = {"source_identity", "effective_date", "constraints", "coverage", "provenance", "claim_context"}
            if failed.intersection(source_gates):
                routing = RoutingDecision(
                    "SOURCE_DATA_REQUIRED",
                    "one or more authoritative policy, context, or source gates did not pass",
                )
            else:
                routing = RoutingDecision(
                    "PROVIDER_QUERY",
                    "the note lacks code-changing evidence required by a deterministic gate",
                    "Please clarify the documented service, diagnosis, anatomy, units, or modifier evidence identified by the failed gates.",
                )
        result = CodingResult(encounter_id, state, lines, routing, certificate)
        self.audit.finish(encounter_id, result.state, asdict(result))
        self.audit.verify_chain(encounter_id)
        return result
