from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from datetime import date
from pathlib import Path

from medical_coder.compiler import SourceCompiler, SourceIntegrityError
from medical_coder.decision import DecisionEngine, MANDATORY_GATES
from medical_coder.extraction import ExtractionContractError, validate_extraction_payload
from medical_coder.models import (
    Candidate,
    ClaimContext,
    DecisionState,
    EvidenceFact,
    EvidenceGraph,
    FactStatus,
    SourceSpan,
)
from medical_coder.terminology import TerminologySnapshot


class KernelTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        source = {
            "metadata": {"version": "release-a"},
            "items": [
                {
                    "identifier": "alpha-item",
                    "label": "documented action target",
                    "starts": "2025-01-01",
                    "ends": None,
                    "state": "active",
                    "concept": "concept-alpha",
                },
                {
                    "identifier": "retired-item",
                    "label": "historical action target",
                    "starts": "2020-01-01",
                    "ends": "2020-12-31",
                    "state": "active",
                    "concept": "concept-retired",
                },
            ],
        }
        (self.root / "source").mkdir()
        (self.root / "source" / "items.json").write_text(json.dumps(source), encoding="utf-8")
        pack = {
            "pack_id": "test-pack",
            "schema_version": 1,
            "capabilities": {
                "closed_world_identity": True,
                "temporal_validity": True,
            },
            "artifacts": [
                {
                    "path": "source/items.json",
                    "authority": "test-authority",
                    "system": "test-system",
                    "container": "items",
                    "version_path": "metadata.version",
                    "code_field": "identifier",
                    "display_fields": ["label"],
                    "effective_start_field": "starts",
                    "effective_end_field": "ends",
                    "status_field": "state",
                    "concept_field": "concept",
                    "billable_default": True,
                }
            ],
        }
        (self.root / "pack.json").write_text(json.dumps(pack), encoding="utf-8")
        self.snapshot_directory = SourceCompiler(self.root).compile(
            self.root / "pack.json",
            self.root / "snapshots",
        )
        self.snapshot = TerminologySnapshot(self.snapshot_directory)
        span = SourceSpan("document-a", 0, 24, "documented action target")
        self.fact = EvidenceFact(
            fact_id="fact-a",
            fact_type="performed_action",
            text="documented action target",
            status=FactStatus.PERFORMED,
            source_spans=(span,),
        )
        self.evidence = EvidenceGraph(
            document_id="document-a",
            document_hash="document-hash",
            facts=(self.fact,),
        )
        self.context = ClaimContext(
            date_of_service=date(2026, 1, 1),
            payer_identifier="payer",
            payer_type="type",
            claim_type="claim",
            billing_entity_role="billing",
            performing_entity_role="performing",
            place_of_service="place",
            jurisdiction="jurisdiction",
            contract_profile="contract",
            authorization_status="not-required",
            organizational_profile="medium-private-practice",
            facility_type="office",
            practice_profile="surgical",
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_snapshot_is_content_addressed_and_idempotent(self) -> None:
        again = SourceCompiler(self.root).compile(self.root / "pack.json", self.root / "snapshots")
        self.assertEqual(again, self.snapshot_directory)
        manifest = json.loads((again / "manifest.json").read_text())
        self.assertEqual(manifest["snapshot_id"], again.name)
        self.assertEqual(manifest["capabilities"]["closed_world_identity"], True)

    def test_search_is_closed_world_and_temporal(self) -> None:
        candidates = self.snapshot.search(
            self.evidence,
            self.context.date_of_service,
            {"test-system"},
        )
        self.assertEqual(len(candidates), 1)
        self.assertIn("alpha-item", candidates[0].artifact_id)
        self.assertIsNone(self.snapshot.artifact("outside/snapshot/identifier"))

    def test_exact_source_normalization_can_entail(self) -> None:
        normalization = self.snapshot.normalize_exact(
            self.fact,
            self.context.date_of_service,
            {"test-system"},
        )
        self.assertIsNotNone(normalization)
        evidence = replace(self.evidence, facts=(replace(self.fact, normalization=normalization),))
        candidate = self.snapshot.search(evidence, self.context.date_of_service, {"test-system"})[0]
        decisions = DecisionEngine(self.snapshot).evaluate(
            evidence,
            self.context,
            (candidate,),
            required_capabilities={"closed_world_identity", "temporal_validity"},
            coverage_evidence={candidate.artifact_id: ("policy-hash:section",)},
        )
        self.assertEqual(decisions[0].state, DecisionState.SUPPORTED_REPORTABLE)
        self.assertTrue(decisions[0].autonomous_release)
        self.assertEqual({gate.name for gate in decisions[0].gates}, set(MANDATORY_GATES))

    def test_missing_required_source_capability_blocks_release(self) -> None:
        candidate = self.snapshot.search(
            self.evidence,
            self.context.date_of_service,
            {"test-system"},
        )[0]
        decision = DecisionEngine(self.snapshot).evaluate(
            self.evidence,
            self.context,
            (candidate,),
            required_capabilities={"unavailable-policy"},
        )[0]
        self.assertEqual(decision.state, DecisionState.SOURCE_UNAVAILABLE)
        self.assertFalse(decision.autonomous_release)

    def test_extractor_cannot_emit_identifier_fields(self) -> None:
        with self.assertRaises(ExtractionContractError):
            validate_extraction_payload(
                {
                    "document_id": "document-a",
                    "document_hash": "hash",
                    "facts": [{"candidate_codes": ["forbidden"]}],
                    "relationships": [],
                }
            )

    def test_source_span_is_verifiable(self) -> None:
        self.fact.source_spans[0].verify_against("documented action target")
        with self.assertRaises(ValueError):
            self.fact.source_spans[0].verify_against("different text")


if __name__ == "__main__":
    unittest.main()

