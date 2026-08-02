"""Regression tests for autonomous-coding architecture controls."""

from __future__ import annotations

import json
import io
import sqlite3
import zipfile
from datetime import date, datetime, timezone
from unittest.mock import patch

import pytest

from app.clinical_facts import build_clinical_fact_report
from app.compliance.datastore.store import (
    ComplianceDataStore, _pfs_published_effective_date,
    _published_effective_date,
)
from app.compliance.refresh import parsers as refresh_parsers
from app.compliance.refresh import preflight as refresh_preflight
from app.compliance.refresh import runner as refresh_runner
from app.core import config
from app.core.model_profiles import (
    CodingExecutionProfile, autonomous_execution_errors, configured_profiles,
    profiles_for_runs,
)
from app.models.schemas import ClinicalEntity
from app.rag.retriever import CandidateRetriever
from app.release.claim_readiness import _source_contract_errors
from app.release.scope_bootstrap import ScopeBootstrapError, _scope_payload
from app.validation.consistency import compare_runs
from tools.build_terminology_pack import _pairs, materialize_pack


def _profile(provider: str, model: str) -> dict:
    return {"profile_id": f"{provider}-profile", "provider": provider,
            "model": model, "independence_domain": provider}


def test_credentials_alone_do_not_authorize_a_second_phi_processor():
    with patch.dict("os.environ", {
            "AUTHORIZED_MODEL_PROVIDERS": "claude,openai",
            "CODING_EXECUTION_PROFILES": ""}), patch.object(
                config, "LLM_PROVIDER", "claude"), patch.object(
                config, "ANTHROPIC_API_KEY", "live-secret"), patch.object(
                config, "OPENAI_API_KEY", "another-live-secret"):
        profiles = configured_profiles()
    assert [profile.provider for profile in profiles] == ["claude"]


def test_explicit_authorized_profiles_are_scheduled_across_domains():
    raw = [_profile("claude", "first-model"),
           _profile("openai", "second-model")]
    with patch.dict("os.environ", {
            "AUTHORIZED_MODEL_PROVIDERS": "claude,openai",
            "CODING_EXECUTION_PROFILES": json.dumps(raw)}), patch.object(
                config, "ANTHROPIC_API_KEY", "live-secret"), patch.object(
                config, "OPENAI_API_KEY", "another-live-secret"):
        scheduled = profiles_for_runs(3)
    assert [profile.provider for profile in scheduled] == [
        "claude", "openai", "claude"]


def test_enabled_autonomy_preflights_diversity_and_auditor_separation():
    scheduled = [CodingExecutionProfile(
        "first", "claude", "first-model", "claude"),
        CodingExecutionProfile(
            "second", "openai", "second-model", "openai")]
    with patch.object(config, "MIN_INDEPENDENT_MODEL_DOMAINS", 2), \
            patch.object(config, "LLM_PROVIDER", "claude"), \
            patch.object(config, "CLAUDE_VERIFY_MODEL", "verify-model"), \
            patch.dict("os.environ", {"CLINICAL_AUDIT": "1",
                                      "CLINICAL_AUDIT_PASSES": "2",
                                      "CLINICAL_AUDITOR_MODEL": "audit-model",
                                      "CODER_ADJUDICATOR_MODEL": "judge-one",
                                      "CODER_ADJUDICATOR_ALT_MODEL": "judge-two"}):
        assert autonomous_execution_errors(scheduled, 2) == []
        with patch.dict("os.environ", {"CLINICAL_AUDITOR_MODEL": "verify-model"}):
            assert any("auditor" in error for error in
                       autonomous_execution_errors(scheduled, 2))
        assert any("run count" in error for error in
                   autonomous_execution_errors(scheduled[:1], 1))


def test_consistency_requires_valid_profiles_from_multiple_domains():
    shared = {"icd_codes": [], "cpt_codes": [], "hcpcs_codes": [],
              "supporting_conditions": [], "snomed_codes": [],
              "final_disposition": "CLEAN", "auto_coding_tier": "AUTO"}
    diverse = [dict(shared, model_execution=_profile("claude", "first")),
               dict(shared, model_execution=_profile("openai", "second"))]
    assert compare_runs(diverse)["model_independence"]["satisfied"] is True

    same_domain = [dict(shared, model_execution=_profile("claude", "first")),
                   dict(shared, model_execution=_profile("claude", "second"))]
    assert compare_runs(same_domain)["model_independence"]["satisfied"] is False


def test_clinical_facts_bind_exact_evidence_and_retrieval_uses_only_verified():
    note = "Documented finding. Procedure performed today."
    entity = ClinicalEntity(
        text="Documented finding", category="finding",
        clinical_term="documented finding", source_section="ASSESSMENT",
        source_span={"verified": True, "document_start": 0,
                     "document_end": len("Documented finding")},
    )
    report = build_clinical_fact_report(
        entities=[entity], sections={"full_text": note},
        procedures=["Procedure performed today"], imaging=[],
        supplies=["Absent extracted item"], prior_surgery={})
    assert report["status"] == "REVIEW_REQUIRED"
    assert report["facts"][1]["evidence_verified"] is True
    assert report["facts"][2]["evidence_verified"] is False

    class Store:
        def __init__(self):
            self.queries = []

        def search(self, query, system, top_k=None):
            self.queries.append((query, system))
            return [{"code": "candidate"}]

    store = Store()
    CandidateRetriever(store).retrieve_for_clinical_facts(report)
    assert store.queries == [("Procedure performed today", "cpt")]


def test_scope_bootstrap_requires_every_explicit_dimension_and_billing_identity():
    dimensions = {
        name: ["*"] for name in (
            "payer_kinds", "payer_ids", "plans", "provider_specialties",
            "rendering_npis", "places_of_service", "jurisdictions",
            "note_categories", "claim_families")}
    dimensions["billing_npis"] = ["1999999984"]
    practice = {
        "billing_provider": {"npi": "1999999984"},
        "autonomy": {
            "enabled": True, "approved_by": "named-approver",
            "approval_reference": "approved-policy-record",
            "effective_from": date.today().isoformat(),
            "effective_to": date.today().isoformat(),
            "dimensions": dimensions,
        },
    }
    assert _scope_payload(practice)["dimensions"] == dimensions
    del dimensions["plans"]
    with pytest.raises(ScopeBootstrapError, match="plans"):
        _scope_payload(practice)


def test_terminology_derivation_accepts_only_explicit_matching_initialisms():
    args = {"ignored": {"of", "the"}, "min_words": 2, "max_words": 8,
            "min_alias": 2, "max_alias": 8}
    assert list(_pairs("Long Testing Phrase (LTP)", **args)) == [
        ("LTP", "Long Testing Phrase")]
    assert list(_pairs("Long Testing Phrase (ZZ)", **args)) == []


def test_terminology_materialization_repairs_tampered_generated_pack(tmp_path):
    expected = {"schema_version": 1, "input_fingerprint": "sha256:expected",
                "entries": []}
    output = tmp_path / "derived.json"
    output.write_text(json.dumps({**expected, "entries": [{"tampered": True}]}))
    with patch("tools.build_terminology_pack.build_pack",
               return_value=expected):
        result = materialize_pack(output)
    assert result["changed"] is True
    assert json.loads(output.read_text()) == expected


def test_mue_uses_published_release_identity_from_source_filename():
    assert _published_effective_date({
        "effective_date": "2026-03-31",
        "source_file": "PractitionerServices_Eff_04-01-2026.csv",
    }) == "2026-04-01"


def test_pfs_release_identity_and_lookup_are_date_bounded():
    assert _pfs_published_effective_date({
        "version": "RVU26C", "source": "PPRRVU2026_Jul_nonQPP.csv",
    }) == "2026-07-01"
    store = ComplianceDataStore()
    store._conn = sqlite3.connect(":memory:")
    store._conn.row_factory = sqlite3.Row
    store.conn.execute(
        "CREATE TABLE global_period (code TEXT, glob_days TEXT, "
        "effective_from TEXT, effective_to TEXT)")
    store.conn.execute(
        "INSERT INTO global_period VALUES (?,?,?,?)",
        ("CANDIDATE", "value", "2026-07-01", "9999-12-31"))
    assert store.global_period("candidate", date(2026, 6, 30)) is None
    assert store.global_period("candidate", date(2026, 7, 1)) == "value"
    store.close()


def test_medicare_claim_requires_current_unit_and_coverage_authorities():
    result = {
        "patient_metadata": {"insurance": "Medicare"},
        "cpt_codes": [{"code": "candidate"}], "hcpcs_codes": [],
    }
    records = {
        source_id: {"release_effective_from": "2026-01-01",
                    "release_effective_to": "2026-12-31"}
        for source_id in (
            "icd10_codes", "cpt_codes", "mue_limits", "pfs_indicators")
    }
    records["mcd_coverage_cache"] = {
        "fetched_at": datetime.now(timezone.utc).isoformat()}
    assert _source_contract_errors(
        result, records, date(2026, 6, 1)) == []
    records["mue_limits"]["release_effective_to"] = "2026-03-31"
    assert any("mue_limits" in error for error in _source_contract_errors(
        result, records, date(2026, 6, 1)))


def test_refresh_preflight_fails_closed_and_releases_database_connection():
    from app.compliance.refresh import preflight

    class Store:
        def __init__(self):
            self.closed = False

        def build_or_load(self):
            return None

        def close(self):
            self.closed = True

    store = Store()
    with patch.object(preflight, "ComplianceDataStore", return_value=store), \
            patch.object(preflight, "stale_refreshable_sources",
                         side_effect=[["mue"], ["mue"]]), \
            patch.object(preflight, "refresh_source",
                         return_value={"source": "mue", "ok": False,
                                       "error": "refresh failed"}):
        with pytest.raises(RuntimeError, match="refresh failed"):
            preflight.refresh_stale_sources(require_current=True)
    assert store.closed is True


def _hcpcs_row(record_id: str, code: str, description: str,
               *, short: str = "", coverage: str = "",
               betos: str = "", add: str = "", effective: str = "",
               termination: str = "", action: str = "") -> str:
    chars = [" "] * 293
    if record_id in {"3", "4"}:
        chars[0:5] = code.ljust(5)[:5]
    else:
        chars[3:5] = code.ljust(2)[:2]
    chars[5:10] = list("00100")
    chars[10] = record_id
    chars[11:91] = list(description.ljust(80)[:80])
    chars[91:119] = list(short.ljust(28)[:28])
    chars[229] = coverage[:1] or " "
    chars[256:259] = list(betos.ljust(3)[:3])
    chars[268:276] = list(add.ljust(8)[:8])
    chars[276:284] = list(effective.ljust(8)[:8])
    chars[284:292] = list(termination.ljust(8)[:8])
    chars[292] = action[:1] or " "
    return "".join(chars).rstrip()


def _hcpcs_fixture() -> str:
    return "\n".join([
        _hcpcs_row("3", "A0001", "First authoritative description", short="Short",
                   coverage="C", betos="D1A", add="20260101",
                   effective="20260701", action="N"),
        _hcpcs_row("4", "A0001", "continued from CMS"),
        _hcpcs_row("7", "AA", "Modifier description", short="Modifier",
                   add="20260101", effective="20260701", action="N"),
    ])


def test_hcpcs_fixed_width_parser_preserves_both_record_families_and_provenance():
    records = refresh_parsers.parse_hcpcs_fixed_width(
        _hcpcs_fixture(), source_file="HCPC2026_JUL_ANWEB.txt",
        source_url="https://www.cms.gov/authoritative.zip")
    assert [row["code"] for row in records] == ["A0001", "AA"]
    assert records[0]["long_description"].endswith("continued from CMS")
    assert records[0]["effective_from"] == "2026-07-01"
    assert records[0]["metadata"] == {
        "source_file": "HCPC2026_JUL_ANWEB.txt",
        "source_url": "https://www.cms.gov/authoritative.zip",
        "record_type": "procedure",
    }


def test_hcpcs_refresh_atomically_replaces_source_and_rebuilds_store(tmp_path):
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("HCPC2026_JUL_ANWEB.txt", _hcpcs_fixture())
    installed = tmp_path / "hcpcs_codes.json"
    installed.write_text(json.dumps([
        {"code": "B0002", "metadata": {"record_type": "procedure"}},
        {"code": "BB", "metadata": {"record_type": "modifier"}},
    ]))

    class Store:
        rebuilt = False

        def build_or_load(self):
            self.rebuilt = True

    store = Store()
    with patch.object(refresh_runner, "HCPCS_FILE", installed):
        result = refresh_runner.refresh_source(
            store, "hcpcs", effective_from="2026-07-01",
            local_bytes=archive.getvalue())
    assert result["ok"] is True and result["ingested_records"] == 2
    assert store.rebuilt is True
    assert {row["code"] for row in json.loads(installed.read_text())} == {
        "A0001", "AA"}
    assert not installed.with_suffix(".json.tmp").exists()


def test_hcpcs_freshness_is_bound_to_one_cms_quarter_file(tmp_path):
    path = tmp_path / "hcpcs.json"
    path.write_text(json.dumps([
        {"code": "A0001", "metadata": {
            "source_file": "HCPC2026_JUL_ANWEB_06172026.txt"}},
        {"code": "AA", "metadata": {
            "source_file": "HCPC2026_JUL_ANWEB_06172026.txt"}},
    ]))
    with patch.object(refresh_preflight, "HCPCS_FILE", path):
        assert refresh_preflight._hcpcs_release() == date(2026, 7, 1)
        payload = json.loads(path.read_text())
        payload[1]["metadata"]["source_file"] = "different-quarter.txt"
        path.write_text(json.dumps(payload))
        assert refresh_preflight._hcpcs_release() is None


def test_pos_refresh_preserves_payment_designation_and_rejects_unknown_code():
    rows, columns = refresh_parsers.parse_pos(
        "<table><tr><td>11</td><td>Current CMS name</td>"
        "<td>Description</td></tr></table>", "2026-01-01")
    assert columns == ["code", "name", "facility"]
    assert rows == [("11", "Current CMS name", None)]
    store = ComplianceDataStore()
    store._conn = sqlite3.connect(":memory:")
    store._conn.row_factory = sqlite3.Row
    store.conn.execute(
        "CREATE TABLE pos (code TEXT PRIMARY KEY, name TEXT, facility TEXT)")
    store.conn.execute(
        "INSERT INTO pos VALUES (?,?,?)", ("11", "Old name", "F"))
    assert refresh_runner._replace_pos_reference(
        store, [("11", "Current CMS name", None)]) == 1
    row = store.conn.execute(
        "SELECT name, facility FROM pos WHERE code=?", ("11",)).fetchone()
    assert tuple(row) == ("Current CMS name", "F")
    with pytest.raises(ValueError, match="without an authoritative"):
        refresh_runner._replace_pos_reference(
            store, [("11", "Current CMS name", None),
                    ("12", "New CMS setting", None)])
    assert store.conn.execute(
        "SELECT COUNT(*) FROM pos").fetchone()[0] == 1
    store.close()
