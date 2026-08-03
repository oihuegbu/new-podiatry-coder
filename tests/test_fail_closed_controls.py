"""Regression tests for claim-release controls that must never fail open."""
from __future__ import annotations

import unittest
import sqlite3
import json
from datetime import date, timedelta
from unittest.mock import patch

from app.compliance.agents.base import ComplianceAgent
from app.compliance.agents.ncci_ptp import NCCIPTPAgent
from app.compliance.agents.specificity import SpecificityAgent
from app.compliance.engine import ClaimScrubber
from app.compliance.datastore.store import ComplianceDataStore, cpt_edition_window
from app.compliance.models import (
    Claim, ClaimLine, DenialRisk, Disposition, Finding, ScrubResult, Status,
)
from app.rag.code_reference import CodeReferenceDB
from app.validation.consistency import compare_runs


class _CrashAgent(ComplianceAgent):
    filter_id = "CRASH_TEST"
    filter_name = "Crash test"

    def check(self, claim):
        raise RuntimeError("simulated filter failure")


class _CodeStore:
    def code_exists(self, system, code, dos=None):
        return True

    def code_active_any_date(self, system, code):
        return True

    def children_exist(self, system, code):
        return False


class _NCCIStore:
    def __init__(self, available=True, edit=None):
        self.available = available
        self.edit = edit

    def ncci_data_available(self, dos=None):
        return self.available

    def ncci_pair(self, code1, code2, dos=None):
        return self.edit

    def anatomic_modifiers(self):
        return set()

    def is_em_code(self, code, dos=None):
        return False

    def modifier_codes_for_role(self, role):
        return set()


class TestFailClosedRelease(unittest.TestCase):
    def test_zero_consistency_runs_is_not_unanimous(self):
        report = compare_runs([])
        self.assertFalse(report["unanimous"])
        self.assertFalse(report["input_consistent"])

    def test_structural_code_classes_come_from_edition_bound_authority(self):
        store = ComplianceDataStore()
        store.build_or_load()
        self.assertTrue(store.is_external_cause("W01.0XXA"))
        self.assertFalse(store.is_external_cause("S93.401A"))
        self.assertTrue(store.is_injury_or_poisoning("S93.401A"))
        self.assertTrue(store.exclude_from_assessment_completion("R26.2"))
        self.assertTrue(store.is_performance_measure_tracking(
            "4269F", date(2026, 6, 1)))
        self.assertFalse(store.is_performance_measure_tracking(
            "4269F", date(2025, 6, 1)))
        self.assertEqual(store.icd_with_extension(
            "S93.401D", "sequela", date(2026, 6, 1)), "S93.401S")

    def test_cpt_category_membership_matches_the_licensed_edition(self):
        from app.compliance.datastore.store import CPT_CATEGORIES_FILE
        from app.core.config import CPT_FILE
        categories = json.loads(CPT_CATEGORIES_FILE.read_text())
        cpt = json.loads(CPT_FILE.read_text())
        expected = {
            row["code"] for row in cpt["codes"]
            if len(row["code"]) == 5 and row["code"][:-1].isdigit()
            and row["code"].endswith("F")
        }
        self.assertEqual(
            set(categories["categories"]["performance_measure_tracking"]),
            expected,
        )

    def test_cpt_authority_is_limited_to_the_licensed_edition(self):
        edition = {"metadata": {"year": "2026"}}
        self.assertEqual(
            cpt_edition_window(edition, {"effective_date": "20240101"}),
            ("2026-01-01", "2026-12-31", True),
        )
        self.assertEqual(
            cpt_edition_window(edition, {"effective_date": "20260701"}),
            ("2026-07-01", "2026-12-31", True),
        )
        self.assertEqual(
            cpt_edition_window({}, {"effective_date": "20260701"})[2], False)

    def test_identical_codes_do_not_mask_extraction_drift(self):
        base = {
            "icd_codes": [], "supporting_conditions": [], "cpt_codes": [],
            "hcpcs_codes": [], "snomed_codes": [],
            "final_disposition": "CLEAN", "auto_coding_tier": "AUTO",
            "note_integrity": {
                "complete": True, "page_count": 2, "extracted_page_count": 2,
                "source_pdf_sha256": "sha256:source",
                "extracted_text_sha256": "sha256:first",
            },
            "patient_metadata": {"date_of_service": "2026-02-01"},
        }
        changed = {**base, "note_integrity": {
            **base["note_integrity"], "extracted_text_sha256": "sha256:second"}}
        report = compare_runs([base, changed])
        self.assertFalse(report["unanimous"])
        self.assertFalse(report["input_consistent"])
        self.assertEqual(report["input_disagreements"][0]["field"],
                         "extracted_text_sha256")

    def test_identical_codes_do_not_mask_procedure_extraction_drift(self):
        base = {
            "icd_codes": [], "supporting_conditions": [], "cpt_codes": [],
            "hcpcs_codes": [], "snomed_codes": [],
            "final_disposition": "CLEAN", "auto_coding_tier": "AUTO",
            "rag_context": {"vision_context": {
                "procedures_performed_today": ["documented procedure"]}},
        }
        changed = {**base, "rag_context": {"vision_context": {
            "procedures_performed_today": []}}}
        report = compare_runs([base, changed])
        self.assertFalse(report["unanimous"])
        self.assertIn("procedures", {
            row["field"] for row in report["input_disagreements"]})

    def test_unknown_and_execution_error_are_blocking(self):
        for status in (Status.UNKNOWN, Status.ERROR):
            with self.subTest(status=status):
                result = ScrubResult(findings=[Finding(
                    filter_id="TEST",
                    status=status,
                    denial_risk=DenialRisk.HIGH,
                )]).finalize(filter_count=1)
                self.assertFalse(result.clean)
                self.assertEqual(result.disposition, Disposition.REVIEW)

    def test_agent_exception_routes_claim_to_review(self):
        result = ClaimScrubber(_CodeStore(), [_CrashAgent(_CodeStore())]).scrub({
            "document_id": "crash-control",
            "patient_metadata": {"date_of_service": "2026-02-01"},
        })
        self.assertFalse(result.clean)
        self.assertEqual(result.disposition, Disposition.REVIEW)
        self.assertEqual(result.blocking_findings[0].status, Status.ERROR)
        self.assertEqual(result.filter_results[0]["status"], Status.ERROR.value)
        self.assertEqual(result.filter_results[0]["findings"], 1)

    def test_missing_and_future_dos_block_specificity(self):
        agent = SpecificityAgent(_CodeStore())
        missing = agent.check(Claim())
        future = agent.check(Claim(date_of_service=date.today() + timedelta(days=1)))
        self.assertEqual(missing[0].status, Status.UNKNOWN)
        self.assertTrue(missing[0].is_blocking)
        self.assertEqual(future[0].status, Status.FAIL)


class TestNCCITemporalControls(unittest.TestCase):
    def _claim(self):
        return Claim(
            date_of_service=date(2026, 2, 1),
            lines=[
                ClaimLine(code="PROCEDURE", code_system="CPT"),
                ClaimLine(code="SUPPLY", code_system="HCPCS"),
            ],
        )

    def test_missing_release_is_unknown_not_no_edit(self):
        findings = NCCIPTPAgent(_NCCIStore(available=False)).check(self._claim())
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].status, Status.UNKNOWN)
        self.assertEqual(findings[0].clause, "data_availability")

    def test_hcpcs_pairs_are_evaluated(self):
        edit = {"col1": "PROCEDURE", "col2": "SUPPLY", "modifier_indicator": "0"}
        findings = NCCIPTPAgent(_NCCIStore(edit=edit)).check(self._claim())
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].status, Status.FAIL)
        self.assertEqual(findings[0].suggested_fix, {"remove_code": "SUPPLY"})

    def test_store_does_not_fall_back_to_a_nearby_release(self):
        store = ComplianceDataStore.__new__(ComplianceDataStore)
        store._conn = sqlite3.connect(":memory:")
        store._conn.row_factory = sqlite3.Row
        store.conn.execute(
            "CREATE TABLE ncci_ptp (col1 TEXT, col2 TEXT, modifier_indicator TEXT, "
            "effective_from TEXT, effective_to TEXT)"
        )
        store.conn.execute(
            "INSERT INTO ncci_ptp VALUES (?, ?, ?, ?, ?)",
            ("PROCEDURE", "SUPPLY", "0", "2025-01-01", "2025-12-31"),
        )
        try:
            release_dos = date(2025, 2, 1)
            self.assertTrue(store.ncci_data_available(release_dos))
            self.assertIsNotNone(
                store.ncci_pair("PROCEDURE", "SUPPLY", release_dos))
            unsupported_dos = date(2026, 2, 1)
            self.assertFalse(store.ncci_data_available(unsupported_dos))
            self.assertIsNone(
                store.ncci_pair("PROCEDURE", "SUPPLY", unsupported_dos))
        finally:
            store.conn.close()

    def test_release_window_queries_are_cached_per_database_instance(self):
        store = ComplianceDataStore.__new__(ComplianceDataStore)
        store._conn = sqlite3.connect(":memory:")
        store._conn.row_factory = sqlite3.Row
        store.conn.execute(
            "CREATE TABLE ncci_ptp (effective_from TEXT)"
        )
        store.conn.execute("INSERT INTO ncci_ptp VALUES (?)", ("2026-07-01",))
        traced = []
        store.conn.set_trace_callback(traced.append)
        try:
            self.assertTrue(store.ncci_data_available(date(2026, 7, 1)))
            self.assertTrue(store.ncci_data_available(date(2026, 8, 1)))
            self.assertEqual(
                sum("MAX(effective_from)" in statement for statement in traced), 1
            )
        finally:
            store.conn.close()

        class _ReferenceConnection:
            def __init__(self):
                self.executions = 0

            def execute(self, _sql):
                self.executions += 1
                return self

            def fetchone(self):
                return ("2026-07-01",)

            def close(self):
                pass

        connection = _ReferenceConnection()
        with patch("app.rag.code_reference.sqlite3.connect", return_value=connection):
            reference = CodeReferenceDB()
            self.assertTrue(reference.ncci_data_available(date(2026, 7, 1)))
            self.assertTrue(reference.ncci_data_available(date(2026, 8, 1)))
        self.assertEqual(connection.executions, 1)


if __name__ == "__main__":
    unittest.main()
