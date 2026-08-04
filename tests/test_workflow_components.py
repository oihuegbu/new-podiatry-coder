from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from datetime import date
from pathlib import Path

from medical_coder.audit import AuditIntegrityError, AuditStore
from medical_coder.compiler import SourceCompiler, normalize_date
from medical_coder.decision import DecisionEngine
from medical_coder.documents import DocumentIngestor, load_ingested_document
from medical_coder.extraction import normalized_consensus
from medical_coder.models import ClaimContext, EvidenceFact, EvidenceGraph, FactStatus, Normalization, SourceSpan
from medical_coder.providers import ProviderError, load_profiles
from medical_coder.terminology import TerminologySnapshot


class WorkflowComponentTestCase(unittest.TestCase):
    def test_exact_terminology_phrase_cannot_be_crowded_out_by_common_token(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data").mkdir()
            artifacts = [
                {"identifier": f"GENERIC-{index}", "display": "Deformity"}
                for index in range(600)
            ] + [{"identifier": "TARGET", "display": "Specific clinical concept"}]
            terms = {
                **{f"GENERIC-{index}": ["Deformity"] for index in range(600)},
                "TARGET": ["Haglund's deformity"],
            }
            (root / "data" / "artifacts.json").write_text(json.dumps(artifacts))
            (root / "data" / "terms.json").write_text(json.dumps({"terms": terms}))
            pack = {
                "pack_id": "exact-phrase", "schema_version": 3, "capabilities": {},
                "artifacts": [{
                    "path": "data/artifacts.json", "authority": "terminology-authority",
                    "system": "clinical-system", "version": "one", "code_field": "identifier",
                    "display_fields": ["display"], "billable_default": False,
                }],
                "lookup_maps": [{
                    "path": "data/terms.json", "authority": "terminology-authority",
                    "system": "clinical-system", "container": "terms",
                    "term_kind": "official_description",
                }],
            }
            (root / "pack.json").write_text(json.dumps(pack))
            snapshot = TerminologySnapshot(SourceCompiler(root).compile(root / "pack.json", root / "snapshots"))
            phrase = "Haglund's deformity"
            span = SourceSpan("doc", 0, len(phrase), phrase)
            fact = EvidenceFact("fact", "condition", phrase, FactStatus.PRESENT, (span,))
            normalized = snapshot.normalize_fact(fact, date(2026, 1, 1), {"clinical-system"})
            self.assertEqual(snapshot.artifact(normalized.normalization.target_artifact_id).code, "TARGET")
            self.assertEqual(normalized.normalization.candidate_expansions, ("Specific clinical concept",))

    def test_cpt_descriptor_retrieval_operates_without_cpt_link_terms(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data").mkdir()
            (root / "data" / "artifacts.json").write_text(json.dumps([
                {"code": "PROC-A", "long": "uncommon alpha operative service"},
            ]))
            (root / "data" / "empty_index.json").write_text(json.dumps({"terms": {}}))
            pack = {
                "pack_id": "descriptor-only", "schema_version": 3, "capabilities": {},
                "artifacts": [{
                    "path": "data/artifacts.json", "authority": "licensed-authority", "system": "CPT",
                    "version": "one", "code_field": "code", "display_fields": ["long"],
                    "search_fields": ["long"], "billable_default": True,
                }],
                "lookup_maps": [{
                    "path": "data/empty_index.json", "authority": "optional-index", "system": "CPT",
                    "container": "terms", "term_kind": "licensed_alphabetic_index",
                }],
            }
            (root / "pack.json").write_text(json.dumps(pack))
            snapshot = TerminologySnapshot(SourceCompiler(root).compile(root / "pack.json", root / "snapshots"))
            span = SourceSpan("doc", 0, 32, "uncommon alpha operative service")
            fact = EvidenceFact("fact", "procedure", span.verbatim_text, FactStatus.PERFORMED, (span,))
            normalized = snapshot.normalize_fact(fact, date(2026, 1, 1), {"CPT"})
            self.assertIsNotNone(normalized.normalization)
            self.assertFalse(normalized.normalization.alternatives)
            self.assertIn("exact_authoritative_descriptor", normalized.normalization.resolution_factors)

    def test_optional_index_term_is_candidate_not_descriptor_confirmation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data").mkdir()
            (root / "data" / "artifacts.json").write_text(json.dumps([
                {"code": "PROC-A", "long": "uncommon alpha operative service"},
            ]))
            (root / "data" / "index.json").write_text(json.dumps({"terms": {"PROC-A": ["named operation"]}}))
            pack = {
                "pack_id": "index-candidate", "schema_version": 3, "capabilities": {},
                "artifacts": [{
                    "path": "data/artifacts.json", "authority": "licensed-authority", "system": "CPT",
                    "version": "one", "code_field": "code", "display_fields": ["long"],
                    "billable_default": True,
                }],
                "lookup_maps": [{
                    "path": "data/index.json", "authority": "optional-index", "system": "CPT",
                    "container": "terms", "term_kind": "licensed_alphabetic_index",
                }],
            }
            (root / "pack.json").write_text(json.dumps(pack))
            snapshot = TerminologySnapshot(SourceCompiler(root).compile(root / "pack.json", root / "snapshots"))
            span = SourceSpan("doc", 0, 15, "named operation")
            fact = EvidenceFact("fact", "procedure", span.verbatim_text, FactStatus.PERFORMED, (span,))
            normalized = snapshot.normalize_fact(fact, date(2026, 1, 1), {"CPT"})
            self.assertIsNotNone(normalized.normalization)
            self.assertFalse(normalized.normalization.alternatives)
            self.assertNotIn(
                "all_source_derived_material_descriptor_tokens_entailed",
                normalized.normalization.resolution_factors,
            )

    def test_fuzzy_descriptor_vocabulary_is_recall_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data").mkdir()
            (root / "data" / "artifacts.json").write_text(json.dumps([
                {"code": "PROC-A", "long": "operative uncommon service"},
                {"code": "PROC-B", "long": "unrelated distinct service"},
            ]))
            pack = {
                "pack_id": "lexical-expansion", "schema_version": 3, "capabilities": {},
                "semantic_derivations": {
                    "descriptor_token_requirements": {
                        "enabled": True, "maximum_document_frequency_ratio": 0.5, "minimum_token_length": 2,
                    },
                    "source_derived_candidate_recall": {
                        "enabled": True, "minimum_token_length": 6,
                        "minimum_common_substring_length": 6,
                        "minimum_common_substring_coverage": 0.75,
                        "minimum_sequence_similarity": 0.75,
                        "minimum_unique_coverage_margin": 0.1, "maximum_hops": 2,
                    },
                },
                "artifacts": [{
                    "path": "data/artifacts.json", "authority": "licensed-authority", "system": "CPT",
                    "version": "one", "code_field": "code", "display_fields": ["long"],
                    "billable_default": True,
                }],
            }
            (root / "pack.json").write_text(json.dumps(pack))
            snapshot = TerminologySnapshot(SourceCompiler(root).compile(root / "pack.json", root / "snapshots"))
            phrase = "preoperative uncommon service"
            span = SourceSpan("doc", 0, len(phrase), phrase)
            fact = EvidenceFact("fact", "procedure", phrase, FactStatus.PERFORMED, (span,))
            candidates = snapshot.search(
                EvidenceGraph("doc", "hash", (fact,)), date(2026, 1, 1), {"CPT"}
            )
            self.assertTrue(candidates)
            normalized = snapshot.normalize_fact(fact, date(2026, 1, 1), {"CPT"})
            self.assertIsNone(normalized.normalization)

    def test_document_offsets_round_trip_across_pages(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "note.txt"
            source.write_text("documented service", encoding="utf-8")
            document = DocumentIngestor().ingest(source, "encounter")
            target = document.write(root / "document.json")
            loaded = load_ingested_document(target)
            self.assertEqual(loaded.text, "documented service")
            self.assertEqual(loaded.page_for_offset(0), 1)

    def test_explicit_source_date_grammar(self) -> None:
        self.assertEqual(normalize_date("07/01/2026"), "2026-07-01")

    def test_provider_profiles_must_be_independent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "providers.json"
            profile = {
                "name": "same", "kind": "openai", "model": "model", "endpoint": "https://authority.example",
                "api_key_env": "KEY",
            }
            path.write_text(json.dumps({"profiles": [profile, {**profile, "name": "other"}]}), encoding="utf-8")
            with self.assertRaises(ProviderError):
                load_profiles(path)

    def test_normalized_consensus_accepts_same_identity_with_overlapping_spans(self) -> None:
        span_left = SourceSpan("doc", 0, 8, "raw term")
        span_right = SourceSpan("doc", 0, 8, "raw term")
        normalization = Normalization("concept", "system", "snapshot:test", 1.0, target_artifact_id="artifact")
        left = EvidenceFact("left", "diagnosis", "raw term", FactStatus.PRESENT, (span_left,), normalization=normalization)
        right = EvidenceFact("right", "condition", "raw term", FactStatus.PRESENT, (span_right,), normalization=normalization)
        outcome = normalized_consensus([
            EvidenceGraph("doc", "hash", (left,)), EvidenceGraph("doc", "hash", (right,))
        ])
        self.assertIsNotNone(outcome.evidence)
        self.assertFalse(outcome.conflicts)

    def test_normalized_consensus_rejects_unmatched_present_diagnosis(self) -> None:
        span = SourceSpan("doc", 0, 8, "raw term")
        fact = EvidenceFact("left", "diagnosis", "raw term", FactStatus.PRESENT, (span,))
        outcome = normalized_consensus([
            EvidenceGraph("doc", "hash", (fact,)), EvidenceGraph("doc", "hash", ())
        ])
        self.assertIsNone(outcome.evidence)
        self.assertTrue(outcome.conflicts)

    def test_audit_chain_detects_input_reuse(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = AuditStore(Path(directory) / "audit.sqlite")
            store.begin("encounter", "document", "snapshot", "context")
            store.append("encounter", "STEP", {"value": 1})
            store.verify_chain("encounter")
            with self.assertRaises(AuditIntegrityError):
                store.begin("encounter", "different", "snapshot", "context")

    def test_audit_completion_is_atomic_and_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = AuditStore(Path(directory) / "audit.sqlite")
            store.begin("encounter", "document", "snapshot", "context")
            result = {"state": "complete"}
            store.finish("encounter", "COMPLETE", result)
            store.finish("encounter", "COMPLETE", result)
            self.assertEqual(store.result("encounter"), result)
            store.verify_chain("encounter")
            with self.assertRaises(AuditIntegrityError):
                store.finish("encounter", "COMPLETE", {"state": "changed"})

    def test_coverage_preserves_procedure_scope_and_group_composition(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data").mkdir()
            artifacts = [
                {"code": "PROC-A", "display": "performed service alpha"},
                {"code": "PROC-B", "display": "performed service beta"},
                {"code": "DX-PRIMARY", "display": "present condition primary"},
                {"code": "DX-SECONDARY", "display": "present condition secondary"},
            ]
            coverage = {
                "articles": [{
                    "id": "POLICY-A", "title": "policy", "procedures": ["PROC-A", "PROC-B"],
                    "states": ["STATE-A"], "covered": [], "noncovered": [],
                    "groups": [
                        {"group": 1, "role": "primary_eligible", "scope": ["PROC-A"], "codes": ["DX-PRIMARY"]},
                        {"group": 2, "role": "required_secondary", "scope": ["PROC-A"], "codes": ["DX-SECONDARY"]},
                    ],
                }]
            }
            jurisdictions = {"rows": [{"jurisdiction": "J-A", "aliases": ["STATE-A"]}]}
            (root / "data" / "artifacts.json").write_text(json.dumps(artifacts))
            (root / "data" / "coverage.json").write_text(json.dumps(coverage))
            (root / "data" / "jurisdictions.json").write_text(json.dumps(jurisdictions))
            pack = {
                "pack_id": "coverage-test", "schema_version": 3, "capabilities": {},
                "artifacts": [{
                    "path": "data/artifacts.json", "authority": "test", "system": "test",
                    "version": "one", "code_field": "code", "display_fields": ["display"],
                    "billable_default": True,
                }],
                "coverage_sources": [{
                    "path": "data/coverage.json", "authority": "test", "container": "articles",
                    "payer_types": ["payer-type-a"],
                    "policy_id_field": "id", "title_field": "title", "procedure_fields": ["procedures"],
                    "diagnosis_fields": {"covered": ["covered"], "noncovered": ["noncovered"]},
                    "jurisdiction_fields": ["states"],
                    "diagnosis_group_fields": [{
                        "field": "groups", "codes_field": "codes", "procedure_scope_field": "scope",
                        "group_id_field": "group", "role_field": "role",
                        "allowed_roles": ["unspecified", "primary_eligible", "required_secondary"],
                        "disposition": "covered",
                    }],
                }],
                "jurisdiction_sources": [{
                    "path": "data/jurisdictions.json", "authority": "test", "container": "rows",
                    "canonical_field": "jurisdiction", "alias_fields": ["aliases"],
                }],
            }
            (root / "pack.json").write_text(json.dumps(pack))
            snapshot = TerminologySnapshot(SourceCompiler(root).compile(root / "pack.json", root / "snapshots"))
            service_a = snapshot.artifact_by_code("test", "PROC-A", date(2026, 1, 1))
            service_b = snapshot.artifact_by_code("test", "PROC-B", date(2026, 1, 1))
            primary = snapshot.artifact_by_code("test", "DX-PRIMARY", date(2026, 1, 1))
            secondary = snapshot.artifact_by_code("test", "DX-SECONDARY", date(2026, 1, 1))
            payer = ("payer-a", "payer-type-a", "contract-a")
            self.assertEqual(snapshot.coverage(service_a, [primary], date(2026, 1, 1), "STATE-A", *payer).status, "fail")
            self.assertEqual(snapshot.coverage(service_a, [primary, secondary], date(2026, 1, 1), "STATE-A", *payer).status, "pass")
            self.assertEqual(snapshot.coverage(service_b, [primary, secondary], date(2026, 1, 1), "STATE-A", *payer).status, "fail")
            other_payer = ("payer-a", "payer-type-b", "contract-a")
            self.assertEqual(
                snapshot.coverage(service_a, [primary, secondary], date(2026, 1, 1), "STATE-A", *other_payer).status,
                "not_determinable",
            )


    def test_pair_edit_exception_requires_source_backed_exception_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data").mkdir()
            (root / "data" / "artifacts.json").write_text(json.dumps([
                {"code": "PROC-A", "display": "service alpha"},
                {"code": "PROC-B", "display": "service beta"},
            ]))
            (root / "data" / "constraints.json").write_text(json.dumps([
                {"left": "PROC-A", "right": "PROC-B", "modifier_indicator": "1"},
            ]))
            pack = {
                "pack_id": "constraint-test", "schema_version": 3, "capabilities": {},
                "artifacts": [{
                    "path": "data/artifacts.json", "authority": "test", "system": "test",
                    "version": "one", "code_field": "code", "display_fields": ["display"],
                    "billable_default": True,
                }],
                "constraints": [{
                    "path": "data/constraints.json", "authority": "test", "kind": "ptp_edit",
                    "left_field": "left", "right_field": "right",
                    "properties": ["modifier_indicator"],
                }],
            }
            (root / "pack.json").write_text(json.dumps(pack))
            snapshot = TerminologySnapshot(SourceCompiler(root).compile(root / "pack.json", root / "snapshots"))
            service = snapshot.artifact_by_code("test", "PROC-A", date(2026, 1, 1))
            context = ClaimContext(
                date(2026, 1, 1), "payer", "type", "professional", "billing", "performing",
                "place", "jurisdiction", "contract", "not-required",
                "medium-private-practice", "office", "surgical",
            )
            engine = DecisionEngine(snapshot)
            blocked = engine._constraints(service, {"PROC-A", "PROC-B"}, context, 1.0, set())
            allowed = engine._constraints(service, {"PROC-A", "PROC-B"}, context, 1.0, {"PROC-B"})
            self.assertFalse(blocked[0])
            self.assertTrue(allowed[0])


if __name__ == "__main__":
    unittest.main()
