"""Architecture invariants for the evidence-first autonomous coder."""

import json
from dataclasses import replace

import pytest

from app.autonomous.evidence import EvidencePacket, EvidencePacketError
from app.clinical_facts import build_clinical_fact_report
from app.terminology import TerminologyNormalizer
from app.validation.consistency import compare_runs
from tools.clinical_auditor import _full_record_view


def _packet() -> EvidencePacket:
    return EvidencePacket.create(
        document_id="note",
        source_document_sha256="sha256:" + "a" * 64,
        payload={"sections": {"full_text": "source note"},
                 "rag_candidates": {"icd10": [], "cpt": [], "hcpcs": []}},
    )


def test_evidence_packet_is_deeply_immutable_and_returns_fresh_copies():
    packet = _packet()
    first = packet.payload()
    first["sections"]["full_text"] = "mutated"
    assert packet.payload()["sections"]["full_text"] == "source note"


def test_evidence_packet_detects_boundary_tampering():
    packet = replace(_packet(), canonical_payload='{"changed":true}')
    with pytest.raises(EvidencePacketError, match="fingerprint mismatch"):
        packet.payload()


def test_evidence_artifact_round_trips_and_rejects_tampering(tmp_path):
    packet = _packet()
    path = packet.persist(tmp_path)
    loaded = EvidencePacket.load(path)
    assert loaded == packet
    body = json.loads(path.read_text())
    body["payload"]["sections"]["full_text"] = "changed"
    path.write_text(json.dumps(body))
    with pytest.raises(EvidencePacketError, match="fingerprint mismatch"):
        EvidencePacket.load(path)


def test_consistency_proves_every_coder_received_identical_evidence():
    manifest = _packet().manifest()
    base = {
        "evidence_packet": manifest,
        "icd_codes": [], "cpt_codes": [], "hcpcs_codes": [],
        "supporting_conditions": [], "snomed_codes": [],
        "final_disposition": "REVIEW", "auto_coding_tier": "REVIEW",
        "model_execution": {"provider": "claude", "model": "one",
                            "independence_domain": "claude",
                            "models_used": ["one"]},
    }
    other = {**base, "model_execution": {
        "provider": "openai", "model": "two",
        "independence_domain": "openai", "models_used": ["two"]}}
    report = compare_runs([base, other])
    assert report["input_consistent"] is True
    assert report["input_complete"] is False
    assert report["unanimous"] is False
    changed = {**other, "evidence_packet": {
        **manifest, "evidence_fingerprint": "sha256:" + "b" * 64}}
    report = compare_runs([base, changed])
    assert report["input_consistent"] is False
    assert any(row["field"] == "evidence_fingerprint"
               for row in report["input_disagreements"])


def test_heading_words_are_not_misclassified_as_abbreviations():
    note = ("PATIENT EDUCATION: Discussed recovery.\n"
            "PLAN: Aspirin BID for VTE prophylaxis while NWB.")
    _, report = TerminologyNormalizer().normalize_entities(
        [], {"plan": note, "full_text": note})
    raw = {row["raw_text"] for row in report["note_occurrences"]}
    assert "PATIENT" not in raw
    assert "EDUCATION" not in raw
    assert "PLAN" not in raw
    resolutions = {row["raw_text"]: row for row in report["note_occurrences"]}
    assert resolutions["BID"]["status"] == "accepted"
    assert resolutions["VTE"]["status"] == "accepted"


def test_performed_event_uses_verified_source_quote_not_summary_text():
    note = "The tendon was carefully debrided and securely reattached."
    report = build_clinical_fact_report(
        entities=[], sections={"full_text": note},
        procedures=["tendon reconstruction"], imaging=[], supplies=[],
        prior_surgery={}, event_evidence=[{
            "kind": "procedure", "label": "tendon reconstruction",
            "evidence_quote": note, "source_span_verified": True,
        }])
    fact = next(row for row in report["facts"]
                if row["kind"] == "performed_procedure")
    assert fact["evidence_span"] == note
    assert fact["evidence_verified"] is True
    assert report["status"] == "PASS"


def test_audit_view_excludes_non_coding_identifiers_but_keeps_defense():
    result = {
        "document_id": "note",
        "patient_metadata": {
            "patient_name": "private", "member_id": "private-member",
            "date_of_service": "2026-01-01", "insurance": "payer",
        },
        "evidence_packet": _packet().manifest(),
        "clinical_facts": {"status": "PASS", "facts": []},
        "icd_codes": [], "cpt_codes": [], "hcpcs_codes": [],
        "candidate_claim": {}, "claim_scrub": {"clean": True},
    }
    evidence = {
        "note_category": "encounter",
        "rag_candidates": {"procedure": [{
            "code": "candidate-code", "description": "candidate description",
            "score": 0.9}]},
        "performed_event_evidence": [{"label": "documented service"}],
        "physician_documented_codes": [],
    }
    view = _full_record_view(result, evidence)
    assert "patient_name" not in view["encounter_context"]
    assert "member_id" not in view["encounter_context"]
    assert view["encounter_context"]["date_of_service"] == "2026-01-01"
    assert view["evidence_packet"] == result["evidence_packet"]
    assert view["retained_evidence"]["retrieved_candidates"]["procedure"][0][
        "description"] == "candidate description"
