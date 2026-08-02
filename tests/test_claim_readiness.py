"""Deterministic tests for immutable, fail-closed release certificates."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest import mock

from app.release.claim_readiness import (build_readiness_certificate,
                                         verify_readiness_certificate)
from app.release.scope_registry import sign_scope


class ClaimReadinessTest(unittest.TestCase):
    def setUp(self):
        self.note = "Documented diagnosis and performed service with side."
        self.scope_key = "test-only-scope-key-with-32-bytes-minimum"
        scope = {
            "id": "test-scope", "approved": True,
            "approved_by": "safety-reviewer",
            "effective_from": "2020-01-01", "effective_to": "2099-12-31",
            "dimensions": {
                "payer_kinds": ["*"], "payer_ids": ["*"],
                "plans": ["*"], "provider_specialties": ["*"],
                "rendering_npis": ["*"], "billing_npis": ["*"],
                "places_of_service": ["*"], "jurisdictions": ["*"],
                "note_categories": ["*"], "claim_families": ["*"],
            },
        }
        scope["signature"] = sign_scope(scope, self.scope_key)
        self.tmp = tempfile.TemporaryDirectory()
        self.scope_path = Path(self.tmp.name) / "scopes.json"
        self.scope_path.write_text(json.dumps({"scopes": [scope]}))
        self.env = mock.patch.dict(os.environ, {
            "AUTONOMOUS_SCOPE_REGISTRY": str(self.scope_path),
            "AUTONOMOUS_SCOPE_SIGNING_KEY": self.scope_key,
            "CLAIM_READINESS_SIGNING_KEY": "test-certificate-key-with-32-bytes-minimum",
        })
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def result(self):
        text_hash = "sha256:" + hashlib.sha256(self.note.encode()).hexdigest()
        result = {
            "document_id": "claim-a", "success": True,
            "final_disposition": "CLEAN",
            "patient_metadata": {
                "insurance": "UnitedHealthcare Choice Plus",
                "date_of_service": "2026-01-05",
                "provider_specialty": "podiatry", "place_of_service": "office",
                "insurance_plan": "Choice Plus", "member_id": "member-1",
                "provider_npi": "1888888882", "billing_npi": "1999999984",
                "state": "FL",
            },
            "rag_context": {"note_full_text": self.note,
                            "vision_context": {
                                "note_category": "established_visit"}},
            "consistency": {"runs": 3, "unanimous": True},
            "claim_scrub": {"clean": True, "disposition": "CLEAN"},
            "icd_codes": [{
                "code": "DX", "type": "primary",
                "evidence_spans": ["Documented diagnosis"],
                "source_record_ids": ["icd10_codes:DX"],
                "source_effective_from": "2020-01-01",
                "source_effective_to": "2099-12-31",
                "source_temporal_authority": True,
            }],
            "cpt_codes": [{
                "code": "SERVICE", "units": 1,
                "modifiers": [], "linked_diagnoses": ["DX"],
                "evidence_spans": ["performed service"],
                "source_record_ids": ["cpt_codes:SERVICE"],
                "source_effective_from": "2020-01-01",
                "source_effective_to": "2099-12-31",
                "source_temporal_authority": True,
            }],
            "hcpcs_codes": [], "material_corrections": [],
            "note_integrity": {
                "complete": True, "page_count": 3,
                "extracted_page_count": 3,
                "source_pdf_sha256": "sha256:" + "a" * 64,
                "extracted_text_sha256": text_hash,
                "page_coverage": [
                    {"page_number": n, "status": "extracted",
                     "text_sha256": "sha256:" + str(n) * 64}
                    for n in range(1, 4)
                ],
            },
        }
        from app.compliance.agents import build_default_agents
        required = [a.filter_id for a in build_default_agents(None)]
        result["claim_scrub"]["expected_filter_count"] = len(required)
        result["claim_scrub"]["filter_results"] = [
            {"filter_id": filter_id, "status": "PASS"}
            for filter_id in required]
        from app.release.source_manifest import build_source_manifest
        result["authoritative_source_manifest"] = build_source_manifest()
        result["candidate_claim"] = {
            "icd_codes": deepcopy(result["icd_codes"]),
            "cpt_codes": deepcopy(result["cpt_codes"]), "hcpcs_codes": [],
        }
        from tools.clinical_auditor import corrections_fingerprint
        result["clinical_audit"] = {
            "verdict": "upheld",
            "fingerprint": corrections_fingerprint(result),
        }
        return result

    def test_complete_claim_is_auto_ready_and_verifiable(self):
        result = self.result()
        cert = build_readiness_certificate(result)
        self.assertEqual(cert.disposition.value, "AUTO_READY")
        result["claim_readiness_certificate"] = cert.model_dump(mode="json")
        self.assertEqual(verify_readiness_certificate(result), (True, ""))

    def test_any_claim_change_invalidates_certificate(self):
        result = self.result()
        cert = build_readiness_certificate(result).model_dump(mode="json")
        result["cpt_codes"][0]["units"] = 2
        ok, why = verify_readiness_certificate(result, cert)
        self.assertFalse(ok)
        self.assertIn("claim changed", why)

    def test_source_document_change_invalidates_certificate(self):
        result = self.result()
        cert = build_readiness_certificate(result).model_dump(mode="json")
        result["note_integrity"]["source_pdf_sha256"] = "sha256:different"
        ok, why = verify_readiness_certificate(result, cert)
        self.assertFalse(ok)
        self.assertIn("source document changed", why)

    def test_submission_configuration_change_invalidates_context(self):
        from app.release.claim_readiness import encounter_context_fingerprint
        result = self.result()
        with mock.patch("tools.claim_submitter.load_practice_config",
                        return_value={"billing_provider": {"npi": "1"},
                                      "fee_schedule": {"version": "one"}}):
            first = encounter_context_fingerprint(result)
        with mock.patch("tools.claim_submitter.load_practice_config",
                        return_value={"billing_provider": {"npi": "1"},
                                      "fee_schedule": {"version": "two"}}):
            second = encounter_context_fingerprint(result)
        self.assertNotEqual(first, second)

    def test_artifact_refresh_is_idempotent_when_inputs_are_unchanged(self):
        from app.release.claim_readiness import refresh_release_artifacts
        result = self.result()
        first = refresh_release_artifacts(result).model_dump(mode="json")
        second = refresh_release_artifacts(result).model_dump(mode="json")
        self.assertEqual(first, second)

    def test_tampered_source_manifest_blocks(self):
        result = self.result()
        result["authoritative_source_manifest"]["records"][0]["sha256"] = \
            "sha256:changed-without-rehash"
        cert = build_readiness_certificate(result)
        self.assertEqual(cert.disposition.value, "BLOCKED")

    def test_missing_filter_execution_trail_blocks(self):
        result = self.result()
        result["claim_scrub"].pop("filter_results")
        cert = build_readiness_certificate(result)
        self.assertEqual(cert.disposition.value, "BLOCKED")

    def test_filter_fail_cannot_be_hidden_by_clean_summary(self):
        result = self.result()
        result["claim_scrub"]["filter_results"][0]["status"] = "FAIL"
        cert = build_readiness_certificate(result)
        self.assertEqual(cert.disposition.value, "BLOCKED")

    def test_incomplete_document_blocks(self):
        result = self.result()
        result["note_integrity"]["extracted_page_count"] = 2
        cert = build_readiness_certificate(result)
        self.assertEqual(cert.disposition.value, "BLOCKED")

    def test_unresolved_mutation_blocks(self):
        result = self.result()
        result["cpt_codes"][0]["units"] = 2
        cert = build_readiness_certificate(result)
        self.assertEqual(cert.disposition.value, "BLOCKED")

    def test_unstructured_ledger_entry_blocks_even_without_a_diff(self):
        result = self.result()
        result["mutation_ledger"] = ["not a provenance record"]
        cert = build_readiness_certificate(result)
        self.assertEqual(cert.disposition.value, "BLOCKED")

    def test_temporal_window_without_authority_blocks(self):
        result = self.result()
        result["cpt_codes"][0]["source_temporal_authority"] = False
        cert = build_readiness_certificate(result)
        self.assertEqual(cert.disposition.value, "BLOCKED")

    def test_provenance_enrichment_is_not_a_claim_mutation(self):
        from app.release.mutation_ledger import claim_diff
        result = self.result()
        candidate = deepcopy(result["candidate_claim"])
        candidate["cpt_codes"][0].pop("source_record_ids")
        candidate["cpt_codes"][0].pop("source_effective_from")
        candidate["cpt_codes"][0].pop("source_effective_to")
        self.assertEqual(claim_diff(candidate, result), [])

    def test_unsigned_scope_requires_review(self):
        payload = json.loads(self.scope_path.read_text())
        payload["scopes"][0]["signature"] = "invalid"
        self.scope_path.write_text(json.dumps(payload))
        cert = build_readiness_certificate(self.result())
        self.assertEqual(cert.disposition.value, "REVIEW_REQUIRED")


if __name__ == "__main__":
    unittest.main()
