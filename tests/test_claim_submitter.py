"""Claim submission (tools/claim_submitter.py): dynamic envelope resolution,
fail-closed gates, 837P payload construction, idempotent submission ledger,
and adapter degradation when the clearinghouse is unconfigured."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools import claim_submitter as cs  # noqa: E402
from tools import claims_registry as reg  # noqa: E402


def _practice_config(**overrides) -> dict:
    cfg = {
        "billing_provider": {
            "organization_name": "Test Podiatry PLLC",
            "npi": "1999999984",
            "tax_id": "123456789",
            "taxonomy_code": "213E00000X",
            "address": {"address1": "1 Test Way", "city": "Testville",
                        "state": "FL", "postal_code": "33101"},
            "contact_name": "Billing", "phone": "5555550100",
        },
        "rendering_providers": {
            "providers": [
                {"match": ["yvonne baptiste"], "first_name": "Yvonne",
                 "last_name": "Baptiste", "npi": "1888888884",
                 "taxonomy_code": "213E00000X"},
            ],
            "trust_note_npi": True,
            "default": None,
        },
        "submitter": {"organization_name": "Test Podiatry PLLC",
                      "contact_name": "Billing", "phone": "5555550100"},
        "fee_schedule": {
            "charges": {"28118": 850.0, "A4570": 45.0, "99213": 120.0},
            "missing_code_policy": "block",
        },
        "claim_defaults": {
            "claim_frequency_code": "1",
            "signature_indicator": "Y",
            "plan_participation_code": "A",
            "release_information_code": "Y",
            "benefits_assignment_certification_indicator": "Y",
            "claim_filing_code": {"by_kind": {"medicare_ffs": "MB",
                                              "commercial": "CI"},
                                  "default": "CI"},
            "payer_overrides": {},
        },
        "submission_policy": {
            "verification_tiers": ["auto", "adjudicated", "human"],
            "require_clean_disposition": True,
        },
    }
    cfg.update(overrides)
    return cfg


def _result(**overrides) -> dict:
    r = {
        "document_id": "note_x",
        "success": True,
        "final_disposition": "CLEAN",
        "patient_metadata": {
            "patient_name": "Alistair Castellanos",
            "date_of_birth": "09/02/1982",
            "date_of_service": "January 5, 2026",
            "provider": "Dr. Yvonne Baptiste, DPM",
            "npi": "4777752013",
            "mrn": "359907",
            "insurance": ("UnitedHealthcare Choice Plus, Member/Policy ID "
                          "70736715633, Group Number GRP93623"),
            "place_of_service": "11",
            "gender": "F",
        },
        "icd_codes": [
            {"code": "M71.571", "type": "primary", "description": "Bursitis"},
            {"code": "M77.31", "type": "secondary",
             "description": "Calcaneal spur, right foot"},
        ],
        "cpt_codes": [
            {"code": "28118", "description": "Ostectomy, calcaneus",
             "modifiers": ["RT"], "units": 1},
        ],
        "hcpcs_codes": [
            {"code": "A4570", "description": "Splint", "units": 1},
        ],
        "consistency": {"runs": 3, "unanimous": True,
                        "input_consistent": True},
        "claim_readiness_certificate": {"test_certificate": True},
    }
    r.update(overrides)
    return r


def _reg_event(result: dict, verification="auto") -> dict:
    return reg.make_finalized_event(result["document_id"], result,
                                    verification=verification,
                                    verified_by="test", source="test.json")


class BuildClaimTest(unittest.TestCase):
    def setUp(self):
        self.cfg = _practice_config()
        self.result = _result()
        self.event = _reg_event(self.result)

    def test_full_payload_builds(self):
        payload, blocks = cs.build_claim("note_x", self.event, self.result,
                                         self.cfg)
        self.assertEqual(blocks, [])
        self.assertIsNotNone(payload)
        ci = payload["claimInformation"]
        # charges come from the fee schedule, never invented
        self.assertEqual(ci["claimChargeAmount"], "895.00")
        self.assertEqual(len(ci["serviceLines"]), 2)
        line = ci["serviceLines"][0]["professionalService"]
        self.assertEqual(line["procedureCode"], "28118")
        self.assertEqual(line["lineItemChargeAmount"], "850.00")
        self.assertEqual(line["procedureModifiers"], ["RT"])
        # primary diagnosis first, ABK type, undotted
        self.assertEqual(ci["healthCareCodeInformation"][0],
                         {"diagnosisTypeCode": "ABK",
                          "diagnosisCode": "M71571"})
        self.assertEqual(ci["healthCareCodeInformation"][1]
                         ["diagnosisTypeCode"], "ABF")
        # envelope comes from config
        self.assertEqual(payload["billing"]["npi"], "1999999984")
        self.assertEqual(payload["billing"]["employerId"], "123456789")
        # rendering provider resolved from the roster by note provider name
        self.assertEqual(payload["rendering"]["npi"], "1888888884")
        # subscriber from the note's own metadata
        self.assertEqual(payload["subscriber"]["memberId"], "70736715633")
        self.assertEqual(payload["subscriber"]["dateOfBirth"], "19820902")
        self.assertEqual(ci["serviceLines"][0]["serviceDate"], "20260105")

    def test_missing_fee_schedule_entry_blocks(self):
        del self.cfg["fee_schedule"]["charges"]["A4570"]
        payload, blocks = cs.build_claim("note_x", self.event, self.result,
                                         self.cfg)
        self.assertIsNone(payload)
        self.assertTrue(any("A4570" in b for b in blocks))

    def test_unknown_payer_blocks(self):
        self.result["patient_metadata"]["insurance"] = \
            "Totally Unknown Insurance Co, Member/Policy ID X1"
        payload, blocks = cs.build_claim("note_x", self.event, self.result,
                                         self.cfg)
        self.assertIsNone(payload)
        self.assertTrue(any("stedi_trading_partner_id" in b for b in blocks))

    def test_missing_member_id_blocks(self):
        self.result["patient_metadata"]["insurance"] = "Medicare Part B"
        payload, blocks = cs.build_claim("note_x", self.event, self.result,
                                         self.cfg)
        self.assertIsNone(payload)
        self.assertTrue(any("Member/Policy ID" in b for b in blocks))

    def test_conflicting_structured_member_id_blocks(self):
        self.result["patient_metadata"]["member_id"] = "DIFFERENT"
        payload, blocks = cs.build_claim("note_x", self.event, self.result,
                                         self.cfg)
        self.assertIsNone(payload)
        self.assertTrue(any("IDs disagree" in block for block in blocks))

    def test_unparseable_dob_blocks(self):
        self.result["patient_metadata"]["date_of_birth"] = "sometime in 1982"
        payload, blocks = cs.build_claim("note_x", self.event, self.result,
                                         self.cfg)
        self.assertIsNone(payload)
        self.assertTrue(any("DOB" in b for b in blocks))

    def test_invalid_place_of_service_blocks(self):
        self.result["patient_metadata"]["place_of_service"] = "office"
        payload, blocks = cs.build_claim("note_x", self.event, self.result,
                                         self.cfg)
        self.assertIsNone(payload)
        self.assertTrue(any("authoritative CMS POS" in block
                            for block in blocks))

    def test_invalid_gender_blocks(self):
        self.result["patient_metadata"]["gender"] = "not recorded"
        payload, blocks = cs.build_claim("note_x", self.event, self.result,
                                         self.cfg)
        self.assertIsNone(payload)
        self.assertTrue(any("gender/sex" in block for block in blocks))

    def test_rendering_falls_back_to_note_npi(self):
        self.result["patient_metadata"]["provider"] = "Dr. Someone Else, DPM"
        payload, blocks = cs.build_claim("note_x", self.event, self.result,
                                         self.cfg)
        self.assertEqual(blocks, [])
        self.assertEqual(payload["rendering"]["npi"], "4777752013")

    def test_no_provider_resolution_blocks(self):
        self.cfg["rendering_providers"]["trust_note_npi"] = False
        self.result["patient_metadata"]["provider"] = "Dr. Someone Else"
        payload, blocks = cs.build_claim("note_x", self.event, self.result,
                                         self.cfg)
        self.assertIsNone(payload)
        self.assertTrue(any("rendering provider" in b for b in blocks))

    def test_claim_filing_code_follows_payer_kind(self):
        self.result["patient_metadata"]["insurance"] = \
            "Medicare Part B, Member/Policy ID 1EG4TE5MK73"
        payload, blocks = cs.build_claim("note_x", self.event, self.result,
                                         self.cfg)
        self.assertEqual(blocks, [])
        self.assertEqual(payload["claimInformation"]["claimFilingCode"],
                         "MB")

    def test_missing_claim_indicator_blocks_instead_of_defaulting(self):
        self.cfg["claim_defaults"].pop("signature_indicator")
        payload, blocks = cs.build_claim("note_x", self.event, self.result,
                                         self.cfg)
        self.assertIsNone(payload)
        self.assertIn("claim_defaults.signature_indicator missing", blocks)

    def test_unresolved_claim_filing_code_blocks(self):
        self.cfg["claim_defaults"]["claim_filing_code"] = {"by_kind": {}}
        payload, blocks = cs.build_claim("note_x", self.event, self.result,
                                         self.cfg)
        self.assertIsNone(payload)
        self.assertTrue(any("claim filing code" in block for block in blocks))

    def test_units_multiply_charge(self):
        self.result["cpt_codes"][0]["units"] = 2
        event = _reg_event(self.result)
        payload, blocks = cs.build_claim("note_x", event, self.result,
                                         self.cfg)
        self.assertEqual(blocks, [])
        line = payload["claimInformation"]["serviceLines"][0]
        self.assertEqual(line["professionalService"]["serviceUnitCount"],
                         "2")
        self.assertEqual(line["professionalService"]
                         ["lineItemChargeAmount"], "1700.00")

    def test_provider_name_strips_honorific_and_credentials(self):
        self.assertEqual(cs._split_provider_name("Dr. Sandra Kim, DPM"),
                         ("Sandra", "Kim"))
        self.assertEqual(cs._split_provider_name("Yvonne Baptiste"),
                         ("Yvonne", "Baptiste"))
        self.assertIsNone(cs._split_provider_name("Dr. Kim, DPM"))

    def test_control_number_is_stable_and_valid(self):
        a = cs._control_number("note_x")
        b = cs._control_number("note_x")
        c = cs._control_number("note_y")
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertRegex(a, r"^\d{9}$")


class ConfigValidationTest(unittest.TestCase):
    def test_valid_config_passes(self):
        self.assertEqual(cs.validate_config(_practice_config()), [])

    def test_missing_tin_reported(self):
        cfg = _practice_config()
        cfg["billing_provider"]["tax_id"] = ""
        self.assertTrue(any("tax_id" in p for p in cs.validate_config(cfg)))

    def test_bad_npi_reported(self):
        cfg = _practice_config()
        cfg["billing_provider"]["npi"] = "12345"
        self.assertTrue(any("check digit" in p for p in cs.validate_config(cfg)))

    def test_config_hot_reload(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "practice_config.json"
            path.write_text(json.dumps(_practice_config()))
            with mock.patch.dict("os.environ",
                                 {"PRACTICE_CONFIG_PATH": str(path)}):
                cfg1 = cs.load_practice_config()
                self.assertEqual(cfg1["billing_provider"]["tax_id"],
                                 "123456789")
                updated = _practice_config()
                updated["billing_provider"]["tax_id"] = "987654321"
                path.write_text(json.dumps(updated))
                import os
                os.utime(path, (path.stat().st_atime,
                                path.stat().st_mtime + 2))
                cfg2 = cs.load_practice_config()
                self.assertEqual(cfg2["billing_provider"]["tax_id"],
                                 "987654321")


class _FakeAdapter:
    def __init__(self, submitted=True, errors=None, reference="REF-1"):
        self.submitted_payloads = []
        self._submitted = submitted
        self._errors = errors or []
        self._reference = reference

    def is_configured(self):
        return True

    def submit_claim(self, payload):
        from app.compliance.adapters.stedi import SubmissionResult
        self.submitted_payloads.append(payload)
        return SubmissionResult(configured=True, submitted=self._submitted,
                                claim_reference=self._reference,
                                errors=self._errors)


class SubmitAllTest(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        base = Path(self.td.name)
        self.results_dir = base / "results"
        self.results_dir.mkdir()
        self.registry = base / "claims_registry.jsonl"
        self.ledger = base / "submissions.jsonl"
        self.cfg_path = base / "practice_config.json"
        self.cfg_path.write_text(json.dumps(_practice_config()))

        self.result = _result()
        (self.results_dir / "note_x_results.json").write_text(
            json.dumps(self.result))

        self.patches = [
            mock.patch.object(cs, "LEDGER_PATH", self.ledger),
            mock.patch.object(cs, "REGISTRY_PATH", self.registry),
            mock.patch.dict("os.environ",
                            {"PRACTICE_CONFIG_PATH": str(self.cfg_path)}),
            mock.patch(
                "app.release.claim_readiness.verify_readiness_certificate",
                return_value=(True, "")),
        ]
        for p in self.patches:
            p.start()
        # Registry binding includes the active submission configuration, so
        # create the event only after the test configuration is active.
        reg.append_events([_reg_event(self.result)], self.registry)

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.td.cleanup()

    def test_clean_verified_claim_submits_once(self):
        adapter = _FakeAdapter()
        stats = cs.submit_all(self.results_dir, adapter=adapter)
        self.assertEqual(stats["submitted"], 1)
        self.assertEqual(len(adapter.submitted_payloads), 1)
        # second run: idempotent, no re-transmission
        stats2 = cs.submit_all(self.results_dir, adapter=adapter)
        self.assertEqual(stats2["submitted"], 0)
        self.assertEqual(stats2["already_submitted"], 1)
        self.assertEqual(len(adapter.submitted_payloads), 1)

    def test_changed_claim_after_submission_blocks(self):
        adapter = _FakeAdapter()
        cs.submit_all(self.results_dir, adapter=adapter)
        changed = _result()
        changed["cpt_codes"][0]["modifiers"] = ["LT"]
        reg.append_events([_reg_event(changed)], self.registry)
        stats = cs.submit_all(self.results_dir, adapter=adapter)
        self.assertEqual(stats["blocked"], 1)
        self.assertIn("replacement", stats["docs"]["note_x"])
        self.assertEqual(len(adapter.submitted_payloads), 1)

    def test_non_clean_disposition_blocked(self):
        dirty = _result(document_id="note_dirty",
                        final_disposition="REVIEW")
        (self.results_dir / "note_dirty_results.json").write_text(
            json.dumps(dirty))
        reg.append_events([_reg_event(dirty)], self.registry)
        adapter = _FakeAdapter()
        stats = cs.submit_all(self.results_dir, docs=["note_dirty"],
                              adapter=adapter)
        self.assertEqual(stats["blocked"], 1)
        self.assertIn("CLEAN", stats["docs"]["note_dirty"])
        self.assertEqual(adapter.submitted_payloads, [])

    def test_disallowed_tier_blocked(self):
        cfg = _practice_config()
        cfg["submission_policy"]["verification_tiers"] = ["human"]
        self.cfg_path.write_text(json.dumps(cfg))
        import os
        os.utime(self.cfg_path, (self.cfg_path.stat().st_atime,
                                 self.cfg_path.stat().st_mtime + 2))
        adapter = _FakeAdapter()
        stats = cs.submit_all(self.results_dir, adapter=adapter)
        self.assertEqual(stats["blocked"], 1)
        self.assertIn("tier", stats["docs"]["note_x"])

    def test_clearinghouse_rejection_recorded_not_marked_submitted(self):
        adapter = _FakeAdapter(submitted=False, errors=["bad subscriber"],
                               reference=None)
        stats = cs.submit_all(self.results_dir, adapter=adapter)
        self.assertEqual(stats["submitted"], 0)
        self.assertEqual(stats["blocked"], 1)
        # a rejection never poisons idempotency — retry is allowed
        adapter2 = _FakeAdapter()
        stats2 = cs.submit_all(self.results_dir, adapter=adapter2)
        self.assertEqual(stats2["submitted"], 1)

    def test_dry_run_writes_payload_transmits_nothing(self):
        with mock.patch.object(cs, "DRYRUN_DIR",
                               Path(self.td.name) / "dryrun"):
            stats = cs.submit_all(self.results_dir, dry_run=True)
            self.assertEqual(stats["submitted"], 1)
            files = list((Path(self.td.name) / "dryrun").glob("*.json"))
            self.assertEqual(len(files), 1)
            payload = json.loads(files[0].read_text())
            self.assertEqual(payload["claimInformation"]
                             ["claimChargeAmount"], "895.00")
        # dry run leaves no 'submitted' ledger event -> still submittable
        self.assertEqual(cs.submitted_keys(cs.load_ledger(self.ledger)), {})

    def test_repeated_block_not_reappended_to_ledger(self):
        # remove the fee entry so the claim blocks
        cfg = _practice_config()
        del cfg["fee_schedule"]["charges"]["A4570"]
        self.cfg_path.write_text(json.dumps(cfg))
        import os
        os.utime(self.cfg_path, (self.cfg_path.stat().st_atime,
                                 self.cfg_path.stat().st_mtime + 2))
        adapter = _FakeAdapter()
        cs.submit_all(self.results_dir, adapter=adapter)
        cs.submit_all(self.results_dir, adapter=adapter)
        cs.submit_all(self.results_dir, adapter=adapter)
        blocked = [e for e in cs.load_ledger(self.ledger)
                   if e["event"] == "blocked"]
        self.assertEqual(len(blocked), 1)

    def test_invalid_config_blocks_everything(self):
        cfg = _practice_config()
        cfg["billing_provider"]["npi"] = ""
        self.cfg_path.write_text(json.dumps(cfg))
        import os
        os.utime(self.cfg_path, (self.cfg_path.stat().st_atime,
                                 self.cfg_path.stat().st_mtime + 2))
        adapter = _FakeAdapter()
        stats = cs.submit_all(self.results_dir, adapter=adapter)
        self.assertTrue(stats["config_problems"])
        self.assertEqual(stats["submitted"], 0)
        self.assertEqual(adapter.submitted_payloads, [])


class AdapterDegradationTest(unittest.TestCase):
    def test_unconfigured_adapter_refuses(self):
        from app.compliance.adapters.stedi import StediAdapter
        with mock.patch.dict("os.environ", {"STEDI_API_KEY": ""}):
            a = StediAdapter(api_key="")
            res = a.submit_claim({"any": "payload"})
        self.assertFalse(res.configured)
        self.assertFalse(res.submitted)
        self.assertTrue(res.errors)


class DxPointerTest(unittest.TestCase):
    def test_pipeline_pointers_honored(self):
        e = {"dx_pointers": [2, 1, 2, 99]}
        self.assertEqual(cs._dx_pointers(e, 3), [2, 1])

    def test_default_points_at_documented_dx_capped_at_four(self):
        self.assertEqual(cs._dx_pointers({}, 6), [1, 2, 3, 4])
        self.assertEqual(cs._dx_pointers({}, 1), [1])

    def test_linked_diagnoses_translate_to_positions(self):
        dx = ["M77.31", "M76.61", "M71.571"]
        e = {"linked_diagnoses": ["M76.61"]}
        self.assertEqual(cs._dx_pointers(e, 3, dx), [2])
        e = {"linked_diagnoses": ["M77.31", "M71.571"]}
        self.assertEqual(cs._dx_pointers(e, 3, dx), [1, 3])

    def test_linked_diagnoses_off_claim_fall_back(self):
        dx = ["M77.31"]
        e = {"linked_diagnoses": ["L60.0"]}
        self.assertEqual(cs._dx_pointers(e, 1, dx), [1])

    def test_numeric_pointers_outrank_linked(self):
        dx = ["M77.31", "M76.61"]
        e = {"dx_pointers": [2], "linked_diagnoses": ["M77.31"]}
        self.assertEqual(cs._dx_pointers(e, 2, dx), [2])


if __name__ == "__main__":
    unittest.main()
