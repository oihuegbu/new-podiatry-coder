"""Calibration dataset (`tools/calibration_dataset.py`) — validation-ladder rung 8.

Rewritten as pytest cases when the exporter was migrated off the retired
`app.pipeline` result shape. The previous version of this file was a standalone
`main()` script with no `test_` functions, so pytest collected nothing from it and CI
never ran it — which is how the exporter kept reading a shape the deployed entrypoint
had stopped writing without anything failing.

The property that matters most here is negative: an artifact this tool cannot
calibrate must be REFUSED BY NAME, never counted as "skipped". A measurement pipeline
that reports a clean run over an empty dataset is worse than one that reports nothing.
"""

from __future__ import annotations

import json

import pytest

from app.contracts.claim_bundle import (
    AuthorityBinding,
    BundleOrigin,
    ClaimBundle,
    DiagnosisLine,
    EncounterIdentity,
    LineMethod,
    ReleaseDestination,
    ReleaseStatus,
    ServiceLine,
    SourceDocument,
)
from app.release.outcome_ledger import OutcomeLedger, Rung
from tools.calibration_dataset import (
    CalibrationInputError,
    build_row,
    calibrate,
    decision_class,
    export,
    extract_features,
)


def _bundle(*, encounter="E1", destination=ReleaseDestination.AUTO_READY,
            routing=(), modifiers=("LT",), units=2) -> ClaimBundle:
    from app.contracts.claim_bundle import AuditSurface

    return ClaimBundle(
        produced_by=BundleOrigin.CLAUDE_CODER,
        encounter=EncounterIdentity(
            encounter_id=encounter, document_id=encounter,
            date_of_service="2026-01-05",
            source_document=SourceDocument(filename=f"{encounter}.pdf",
                                           document_version=f"doc-{encounter}")),
        diagnoses=(DiagnosisLine(sequence=1, system="icd10", code="DX001",
                                 descriptor="a documented condition",
                                 method=LineMethod.DETERMINISTIC),),
        service_lines=(ServiceLine(sequence=1, system="cpt", code="SVC001",
                                   descriptor="a documented service",
                                   units=units, modifiers=tuple(modifiers),
                                   diagnosis_pointers=(1,),
                                   method=LineMethod.DETERMINISTIC),),
        authority=AuthorityBinding(data_fingerprint="data-v1",
                                   database_snapshot_digest="db-v1",
                                   index_build_id="index-v1",
                                   model_profiles={"extraction": "profile-a"}),
        audit=AuditSurface(routing=tuple(routing)),
        release=ReleaseStatus(destination=destination))


def _payload(bundle: ClaimBundle) -> dict:
    return json.loads(bundle.model_dump_json())


# --------------------------------------------------------------------------
# features and decision class
# --------------------------------------------------------------------------

def test_features_come_from_the_canonical_contract():
    features = extract_features(_bundle())
    assert features["n_diagnoses"] == 1
    assert features["n_service_lines"] == 1
    assert features["n_modifiers"] == 1
    assert features["total_units"] == 2
    assert features["method_mix"] == {"deterministic": 2}
    assert features["context_is_resolved"] is False      # no context provider here


def test_the_routers_verdict_is_the_CLASS_and_is_never_also_a_feature():
    """Leakage guard. A model trained on a feature that restates the routing decision
    just relearns the router instead of predicting whether the router was right."""
    bundle = _bundle(destination=ReleaseDestination.AUTO_QUERY)
    assert decision_class(bundle) == "AUTO_QUERY"
    features = extract_features(bundle)
    for leak in ("destination", "decision_class", "needs_review", "releasable"):
        assert leak not in features


def test_the_weakest_axis_is_taken_from_the_routing_decision_that_used_it():
    """Rung 8 calibrates by weakest axis, so the axis recorded must be the SAME one
    the routing decision named -- not a second, parallel derivation."""
    routed = _bundle(routing=({"destination": "PROVIDER_QUERY", "subject": "svc",
                               "reason": "the note barely documents this event "
                                         "(confidence 0.40 < 0.50, weakest axis "
                                         "'laterality') - clarify before billing",
                               "blocking": True, "fact_id": "f1"},))
    assert extract_features(routed)["weakest_axis"] == "laterality"
    assert extract_features(_bundle())["weakest_axis"] == ""


# --------------------------------------------------------------------------
# the refusal that used to be a silent skip
# --------------------------------------------------------------------------

def test_a_legacy_artifact_is_refused_by_name_not_silently_skipped():
    """The defect this migration fixed.

    The exporter read `cpt_codes`/`icd_codes`/`final_disposition` and treated anything
    without `success` as "skipped", so against a ClaimBundle artifact it reported a
    clean run over zero rows. Silence is the failure; the refusal must name the file.
    """
    legacy = {"document_id": "n1", "success": True, "final_disposition": "CLEAN",
              "cpt_codes": [{"code": "99213"}], "icd_codes": [], "hcpcs_codes": []}
    with pytest.raises(CalibrationInputError) as excinfo:
        build_row("n1", legacy, {}, [])
    assert "not a ClaimBundle" in str(excinfo.value)


def test_export_reports_refusals_and_does_not_call_them_skips(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    (results / "good_results.json").write_text(
        json.dumps(_payload(_bundle(encounter="good"))))
    (results / "legacy_results.json").write_text(
        json.dumps({"document_id": "legacy", "success": True, "cpt_codes": []}))

    stats = export(results, tmp_path / "out.jsonl")
    assert stats["total"] == 1 and stats["new"] == 1
    assert stats["refused"] == 1
    assert stats["refusals"] and "legacy" in stats["refusals"][0]


# --------------------------------------------------------------------------
# export identity and idempotence
# --------------------------------------------------------------------------

def test_each_row_carries_the_exact_claim_data_and_model_identity(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    (results / "good_results.json").write_text(
        json.dumps(_payload(_bundle(encounter="good"))))
    out = tmp_path / "out.jsonl"
    export(results, out)

    row = json.loads(out.read_text().splitlines()[0])
    for field in ("encounter_id", "document_version", "claim_fingerprint",
                  "data_fingerprint", "database_snapshot_digest"):
        assert row["identity"][field], field
    assert row["decision_class"] == "AUTO_READY"


def test_export_is_idempotent_and_updates_a_changed_claim_in_place(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    path = results / "good_results.json"
    path.write_text(json.dumps(_payload(_bundle(encounter="good"))))
    out = tmp_path / "out.jsonl"

    assert export(results, out)["new"] == 1
    assert export(results, out)["unchanged"] == 1
    path.write_text(json.dumps(_payload(_bundle(encounter="good", units=5))))
    stats = export(results, out)
    assert stats["updated"] == 1 and stats["total"] == 1


# --------------------------------------------------------------------------
# rung 8 -- calibration by decision class and weakest axis
# --------------------------------------------------------------------------

def test_calibration_is_per_decision_class_never_one_global_number():
    rows = [
        {"decision_class": "AUTO_READY", "features": {"weakest_axis": ""},
         "labels": {"human_corrected": False, "denied": None}},
        {"decision_class": "AUTO_READY", "features": {"weakest_axis": ""},
         "labels": {"human_corrected": True, "denied": None}},
        {"decision_class": "AUTO_QUERY", "features": {"weakest_axis": "laterality"},
         "labels": {"human_corrected": None, "denied": None}},
    ]
    report = calibrate(rows)
    assert set(report["by_decision_class"]) == {"AUTO_READY", "AUTO_QUERY"}
    assert report["by_decision_class"]["AUTO_READY"]["human_correction_rate"] == 0.5
    assert report["by_weakest_axis"]["laterality"]["claims"] == 1


def test_an_unlabelled_bucket_reports_unknown_not_a_perfect_score():
    """`None` is not zero. A class with no human verdict and no payer outcome has an
    UNKNOWN correction rate, and reporting it as 0% would read as flawless."""
    rows = [{"decision_class": "AUTO_QUERY", "features": {"weakest_axis": ""},
             "labels": {"human_corrected": None, "denied": None}}]
    bucket = calibrate(rows)["by_decision_class"]["AUTO_QUERY"]
    assert bucket["claims"] == 1
    assert bucket["human_verdicts"] == 0
    assert bucket["human_correction_rate"] is None
    assert bucket["denial_rate"] is None


def test_a_payer_outcome_labels_only_the_exact_claim_it_names(tmp_path):
    """The rung-7 join, end to end: an outcome recorded against one exact
    (claim, data, model) identity must not label the same encounter re-run against a
    different data snapshot."""
    ledger = OutcomeLedger(tmp_path / "ledger")
    bundle = _bundle(encounter="good")
    ledger.record(Rung.OUTCOME_FEEDBACK, bundle, {"denied": True, "carcs": ["CO-16"]},
                  observed_at="2026-02-01", source="835")
    rows = ledger.observations(Rung.OUTCOME_FEEDBACK)

    labelled = build_row("good", _payload(bundle), {}, rows)
    assert labelled["labels"]["denied"] is True
    assert labelled["labels"]["carcs"] == ["CO-16"]

    other = _bundle(encounter="good")
    other = other.model_copy(update={
        "authority": AuthorityBinding(data_fingerprint="data-v2",
                                      database_snapshot_digest="db-v2")})
    unlabelled = build_row("good", _payload(other), {}, rows)
    assert unlabelled["labels"]["denied"] is None, (
        "an outcome must not attach to a claim answered from a different snapshot")
