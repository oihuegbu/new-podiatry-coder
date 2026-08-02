"""Regression tests for the autonomous-release fail-closed boundaries."""

from datetime import date

from app.coding.code_assigner import _gate_verify_additions, _hard_db_gate
from app.compliance.agents.benefits import BenefitsAgent
from app.compliance.agents.mue_mai import MUEAgent
from app.compliance.agents.prior_auth import PriorAuthAgent
from app.compliance.models import Claim, ClaimLine, Status


class _DB:
    def validate_icd10(self, code):
        return {"code": code}

    def validate_cpt(self, code):
        return {"code": code}

    def validate_hcpcs(self, code):
        return {"code": code}


def test_valid_but_unretrieved_code_is_removed():
    rows = [{"code": "VALID_BUT_NOT_OFFERED"}]
    assert _hard_db_gate(rows, "cpt", _DB(), {"OFFERED"}) == []


def test_icd_candidate_identity_normalizes_decimal_presentation():
    rows = [{"code": "A00.1"}]
    assert _hard_db_gate(rows, "icd10", _DB(), {"A001"}) == rows


def test_missing_reference_database_removes_every_code():
    assert _hard_db_gate([{"code": "OFFERED"}], "cpt", None,
                         {"OFFERED"}) == []


def test_verification_cannot_introduce_unretrieved_code():
    final = {"icd10_codes": [], "supporting_conditions": [],
             "cpt_codes": [{"code": "NOT_OFFERED"}], "hcpcs_codes": []}
    combined = {key: [] for key in final}
    gated = _gate_verify_additions(
        final, combined, _DB(), allowed_codes={
            "icd10": set(), "cpt": {"OFFERED"}, "hcpcs": set()})
    assert gated["cpt_codes"] == []


def test_mue_release_gap_is_unknown():
    class Store:
        @staticmethod
        def mue_data_available(_dos):
            return False

    claim = Claim(date_of_service=date(2026, 1, 5),
                  lines=[ClaimLine(code="TEST")])
    findings = MUEAgent(Store()).check(claim)
    assert findings[0].status is Status.UNKNOWN


def test_missing_eligibility_identity_is_unknown():
    findings = BenefitsAgent(None).check(Claim())
    assert findings[0].status is Status.UNKNOWN


def test_unresolved_payer_prior_auth_is_unknown():
    findings = PriorAuthAgent(None).check(Claim())
    assert findings[0].status is Status.UNKNOWN
