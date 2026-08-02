"""Regression tests for claim-release controls that must never fail open."""
from __future__ import annotations

import unittest
import sqlite3
from datetime import date, timedelta

from app.compliance.agents.base import ComplianceAgent
from app.compliance.agents.ncci_ptp import NCCIPTPAgent
from app.compliance.agents.specificity import SpecificityAgent
from app.compliance.engine import ClaimScrubber
from app.compliance.datastore.store import ComplianceDataStore
from app.compliance.models import (
    Claim, ClaimLine, DenialRisk, Disposition, Finding, ScrubResult, Status,
)


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


class TestFailClosedRelease(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
