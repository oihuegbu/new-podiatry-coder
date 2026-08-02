"""Regression tests for the autonomous-release fail-closed boundaries."""

import json
import re
from datetime import date

from app.coding.code_assigner import (
    CPT_SYSTEM_PROMPT,
    HCPCS_SNOMED_SYSTEM_PROMPT,
    ICD_SYSTEM_PROMPT,
    VERIFICATION_SYSTEM_PROMPT,
    _build_mdm_reference_block,
    _expand_allowed_icd_family_candidates,
    _gate_verify_additions,
    _hard_db_gate,
    _resolve_patient_status,
)
from app.compliance.agents.benefits import BenefitsAgent
from app.compliance.agents.mue_mai import MUEAgent
from app.compliance.agents.prior_auth import PriorAuthAgent
from app.compliance.adapters.stedi import StediAdapter
from app.compliance.datastore.store import ComplianceDataStore, _is_valid_date
from app.compliance.models import Claim, ClaimLine, Status
from app.pipeline import MedicalCodingPipeline


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


def test_candidate_merge_preserves_each_entity_top_result_before_second_rank():
    pipeline = object.__new__(MedicalCodingPipeline)
    entity = {
        "diagnosis:first": {"candidates": {"icd10": [
            {"code": "FIRST_TOP", "similarity_score": .99},
            {"code": "FIRST_SECOND", "similarity_score": .98},
        ]}},
        "diagnosis:second": {"candidates": {"icd10": [
            {"code": "SECOND_TOP", "similarity_score": .40},
        ]}},
    }
    merged = pipeline._merge_candidates(entity, {})
    assert [row["code"] for row in merged["icd10"]] == [
        "FIRST_TOP", "SECOND_TOP", "FIRST_SECOND"]


def test_authoritative_icd_family_member_can_survive_verification_gate():
    class FamilyDB(_DB):
        def icd10_siblings(self, prefix):
            assert prefix == "Z88"
            return [("Z885", "first"), ("Z886", "second")]

    allowed = _expand_allowed_icd_family_candidates({"Z885"}, FamilyDB())
    assert allowed == {"Z885", "Z886"}

    # The low-level gate accepts the exact expanded authoritative set used by
    # assign_codes; this regression pins the formerly contradictory boundary.
    final = {"icd10_codes": [{"code": "Z88.6"}],
             "supporting_conditions": [], "cpt_codes": [], "hcpcs_codes": []}
    combined = {key: [] for key in final}
    gated = _gate_verify_additions(
        final, combined, FamilyDB(),
        allowed_codes={"icd10": allowed,
                       "cpt": set(), "hcpcs": set()})
    assert gated["icd10_codes"] == [{"code": "Z88.6"}]


def test_patient_status_is_tri_state_and_conflicts_do_not_default():
    assert _resolve_patient_status({"note_type": "new patient"}, {}) == "new patient"
    assert _resolve_patient_status({}, {}) is None
    assert _resolve_patient_status(
        {"note_type": "new patient"},
        {"note_category": "established patient"}) is None


def test_runtime_prompts_contain_no_fixed_medical_code_examples():
    medical_code = re.compile(
        r"\b(?:[A-TV-Z][0-9]{2}(?:\.[0-9A-Z]{1,4})?|[0-9]{5})\b")
    for prompt in (ICD_SYSTEM_PROMPT, CPT_SYSTEM_PROMPT,
                   HCPCS_SNOMED_SYSTEM_PROMPT, VERIFICATION_SYSTEM_PROMPT):
        assert not medical_code.search(prompt)


def test_missing_dos_never_loads_current_mdm_grid():
    class Store:
        @staticmethod
        def mdm_grid(_dos):
            raise AssertionError("missing DOS must not default to today's grid")

    block = _build_mdm_reference_block(Store(), None)
    assert "No effective grid covers the DOS" in block


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


def test_complete_prior_auth_corpus_still_requires_claim_dos(tmp_path):
    store = ComplianceDataStore(tmp_path / "pa.db")
    store.conn.execute(
        "CREATE TABLE prior_auth_policy ("
        "payer TEXT, plan TEXT, complete INTEGER, effective_from TEXT, "
        "effective_to TEXT, source TEXT, status TEXT)"
    )
    store.conn.execute(
        "INSERT INTO prior_auth_policy VALUES "
        "('payer','','1','2026-01-01','2026-12-31','source','active')"
    )
    status = store.prior_auth_policy_status("payer", dos=None)
    assert status == {"available": False, "reason": "dos_unresolved"}


def test_invalid_policy_date_never_becomes_temporal_authority():
    assert _is_valid_date("2026-02-28")
    assert _is_valid_date("02/28/2026")
    assert not _is_valid_date("2026-02-30")
    assert not _is_valid_date("unknown")


def test_stedi_request_is_dos_and_procedure_specific(monkeypatch):
    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        @staticmethod
        def read():
            return json.dumps({
                "benefitsInformation": [{"code": "1", "name": "Active Coverage"}]
            }).encode()

    def fake_urlopen(request, timeout):
        captured["body"] = json.loads(request.data.decode())
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    result = StediAdapter(api_key="test", timeout=7).check_eligibility(
        payer_id="PAYER", member_id="MEMBER", first_name="A",
        last_name="B", date_of_birth="19800101", npi="1234567893",
        date_of_service=date(2026, 7, 20), procedure_code="11721",
        product_or_service_id_qualifier="CJ",
    )
    assert captured["body"]["encounter"] == {
        "dateOfService": "20260720",
        "procedureCode": "11721",
        "productOrServiceIDQualifier": "CJ",
    }
    assert captured["body"]["provider"]["npi"] == "1234567893"
    assert captured["timeout"] == 7
    assert result.active is True
    assert result.service_coverage_confirmed is True


def test_ambiguous_eligibility_benefit_is_unknown_not_inactive(monkeypatch):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        @staticmethod
        def read():
            return json.dumps({
                "benefitsInformation": [{"code": "B", "name": "Co-payment"}]
            }).encode()

    monkeypatch.setattr("urllib.request.urlopen", lambda *_args, **_kw: Response())
    result = StediAdapter(api_key="test").check_eligibility(
        payer_id="PAYER", member_id="MEMBER", first_name="A",
        last_name="B", date_of_birth="19800101", npi="1234567893",
        date_of_service=date(2026, 7, 20), procedure_code="11721",
        product_or_service_id_qualifier="CJ",
    )
    assert result.active is None
    assert result.service_coverage_confirmed is None
