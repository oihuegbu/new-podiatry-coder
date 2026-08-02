"""Architectural tests for governed clinical shorthand normalization."""

from __future__ import annotations

import json

import pytest
from unittest.mock import patch

from app.models.schemas import ClinicalEntity
from app.rag.retriever import CandidateRetriever
from app.pipeline import MedicalCodingPipeline
from app.release.claim_readiness import _terminology_control
from app.terminology import TerminologyConfigError, TerminologyNormalizer
from app.validation.consistency import compare_runs
from app.ner.biomed_ner import enrich_entities


def _entity(text, term, section="PE", category="finding", laterality=None):
    return ClinicalEntity(
        text=text,
        category=category,
        clinical_term=term,
        source_section=section,
        laterality=laterality,
    )


def _note_sections():
    sections = {
        "pmh_medications_allergies": (
            "Peripheral arterial disease, Osteoarthritis, History of DVT"),
        "physical_examination": (
            "Right foot: No pain with fifth MTPJ ROM. DP & PT pulses "
            "diminished 1/4 b/l; ABI 0.72; monophasic waveforms by Doppler."),
        "assessment_diagnoses": (
            "Tailor's bunion (bunionette), right foot"),
        "plan": (
            "Heel WB in a surgical shoe for 4 weeks. RTC in 1 week; "
            "sutures out at 14d. Radiographs at 8wk and athletic shoes at 6wks."),
    }
    sections["full_text"] = "\n".join(sections.values())
    return sections


def test_attached_note_shorthand_is_context_resolved_with_exact_spans():
    normalizer = TerminologyNormalizer()
    sections = _note_sections()
    entities, report = normalizer.normalize_entities([
        _entity("History of DVT", "history of deep vein thrombosis",
                section="PMH", category="diagnosis"),
        _entity("Heel WB in a surgical shoe", "protected heel weight bearing",
                section="PLAN"),
    ], sections)

    assert report["status"] == "PASS"
    assert report["unresolved_billing_relevant"] == []
    assert entities[0].normalized_text == "History of deep vein thrombosis"
    assert entities[1].normalized_text == "Heel weight bearing in a surgical shoe"
    for entity in entities:
        span = entity.source_span
        assert span["verified"] is True
        assert sections["full_text"][span["document_start"]:
                                     span["document_end"]] == entity.text

    occurrences = {row["raw_text"]: row for row in report["note_occurrences"]}
    assert occurrences["MTPJ"]["expansion"] == "metatarsophalangeal joint"
    assert occurrences["ROM"]["expansion"] == "range of motion"
    assert occurrences["MTPJ"]["negated"] is True
    assert occurrences["ROM"]["negated"] is True
    assert occurrences["ABI"]["expansion"] == "ankle-brachial index"
    abi_alternatives = occurrences["ABI"]["alternatives"]
    assert {row["expansion"] for row in abi_alternatives} == {
        "ankle-brachial index", "acquired brain injury"}
    rejected = next(row for row in abi_alternatives
                    if row["expansion"] == "acquired brain injury")
    assert rejected["eligible"] is False
    assert rejected["rejection_reasons"]
    assert occurrences["PT"]["expansion"] == "posterior tibial"
    assert occurrences["b/l"]["expansion"] == "bilateral"
    assert occurrences["RTC"]["coding_impact"] is False


def test_context_selects_same_abbreviation_without_guessing():
    normalizer = TerminologyNormalizer()
    pulse = {"physical_examination": "Foot DP and PT pulses palpable.",
             "full_text": "Foot DP and PT pulses palpable."}
    therapy = {"plan": "Refer to PT for gait strengthening.",
               "full_text": "Refer to PT for gait strengthening."}
    _, pulse_report = normalizer.normalize_entities([], pulse)
    _, therapy_report = normalizer.normalize_entities([], therapy)
    pulse_pt = next(row for row in pulse_report["note_occurrences"]
                    if row["raw_text"] == "PT")
    therapy_pt = next(row for row in therapy_report["note_occurrences"]
                      if row["raw_text"] == "PT")
    assert pulse_pt["expansion"] == "posterior tibial"
    assert therapy_pt["expansion"] == "physical therapy"


def test_structured_laterality_corroborates_or_rejects_expansion():
    normalizer = TerminologyNormalizer()
    sections = {"physical_examination": "RLE pain",
                "full_text": "RLE pain"}
    accepted, _ = normalizer.normalize_entities([
        _entity("RLE pain", "lower extremity pain", laterality="RIGHT")
    ], sections)
    assert accepted[0].normalized_text == "right lower extremity pain"
    assert accepted[0].normalization_status == "accepted"

    rejected, report = normalizer.normalize_entities([
        _entity("RLE pain", "lower extremity pain", laterality="LEFT")
    ], sections)
    assert rejected[0].normalization_status == "unresolved"
    assert report["status"] == "REVIEW_REQUIRED"


def test_ambiguous_or_unknown_affirmed_term_requires_review_but_negated_does_not():
    normalizer = TerminologyNormalizer()
    ambiguous = {"assessment_diagnoses": "PT abnormality",
                 "full_text": "PT abnormality"}
    _, report = normalizer.normalize_entities([], ambiguous)
    assert report["status"] == "REVIEW_REQUIRED"
    assert report["unresolved_billing_relevant"][0]["raw_text"] == "PT"

    unknown = {"assessment_diagnoses": "XYZ deformity",
               "full_text": "XYZ deformity"}
    _, report = normalizer.normalize_entities([], unknown)
    assert report["status"] == "REVIEW_REQUIRED"
    assert report["unresolved_billing_relevant"][0]["raw_text"] == "XYZ"

    negated = {"physical_examination": "No XYZ deformity.",
               "full_text": "No XYZ deformity."}
    _, report = normalizer.normalize_entities([], negated)
    assert report["status"] == "PASS"
    occurrence = report["note_occurrences"][0]
    assert occurrence["status"] == "unresolved"
    assert occurrence["negated"] is True


class _SearchStore:
    def __init__(self):
        self.queries = []

    def search(self, query, system, top_k=None):
        self.queries.append((query, system))
        return [{"code": query, "similarity_score": 1.0}]


class _RankedSearchStore:
    def search(self, query, system, top_k=None):
        # Deliberately incomparable scores: preserving query-local order must
        # still place each query's first result before any second result.
        return [
            {"code": f"{query}-first", "similarity_score": 0.1},
            {"code": f"{query}-second", "similarity_score": 0.99},
        ]


def test_retrieval_receives_model_raw_and_governed_expanded_forms():
    normalizer = TerminologyNormalizer()
    sections = {"pmh_medications_allergies": "History of DVT",
                "full_text": "History of DVT"}
    entities, _ = normalizer.normalize_entities([
        _entity("History of DVT", "venous thromboembolism history",
                section="PMH", category="diagnosis")], sections)
    store = _SearchStore()
    CandidateRetriever(store).retrieve_for_entity(entities[0])
    queries = {query for query, system in store.queries if system == "icd10"}
    assert "venous thromboembolism history" in queries
    assert "History of DVT" in queries
    assert "History of deep vein thrombosis" in queries


def test_multi_query_fusion_preserves_each_query_local_rank():
    entity = _entity("raw phrase", "normalized phrase", category="diagnosis")
    entity.retrieval_terms = ["normalized phrase", "raw phrase"]
    results = CandidateRetriever(_RankedSearchStore()).retrieve_for_entity(
        entity, top_k=4)["icd10"]
    assert [row["code"] for row in results] == [
        "normalized phrase-first", "raw phrase-first",
        "normalized phrase-second", "raw phrase-second",
    ]


def test_pipeline_merge_does_not_resort_incomparable_query_scores():
    pipeline = MedicalCodingPipeline()
    entity_candidates = {"entity": {"candidates": {"icd10": [
        {"code": "raw-first", "similarity_score": 0.1},
        {"code": "expanded-first", "similarity_score": 0.99},
    ]}}}
    merged = pipeline._merge_candidates(
        entity_candidates, {"icd10": [], "cpt": [], "hcpcs": []})
    assert [row["code"] for row in merged["icd10"]] == [
        "raw-first", "expanded-first"]


def test_consistency_compares_normalized_entity_fingerprint():
    normalizer = TerminologyNormalizer()
    sections = {"assessment_diagnoses": "PT abnormality",
                "full_text": "PT abnormality"}
    _, first = normalizer.normalize_entities([], sections)
    changed = json.loads(json.dumps(first))
    changed["entity_fingerprint"] = "sha256:" + "b" * 64
    base = {
        "icd_codes": [], "supporting_conditions": [], "cpt_codes": [],
        "hcpcs_codes": [], "snomed_codes": [],
        "final_disposition": "REVIEW", "auto_coding_tier": "REVIEW",
        "terminology_normalization": first,
    }
    other = {**base, "terminology_normalization": changed}
    report = compare_runs([base, other])
    assert report["unanimous"] is False
    assert any(row["field"] == "terminology_entity_fingerprint"
               for row in report["input_disagreements"])


def test_release_control_accepts_resolved_terms_and_routes_unresolved_terms():
    normalizer = TerminologyNormalizer()
    sections = {"pmh_medications_allergies": "History of DVT",
                "full_text": "History of DVT"}
    entities, report = normalizer.normalize_entities([
        _entity("History of DVT", "history of deep vein thrombosis",
                section="PMH", category="diagnosis")], sections)
    result = {"ner_entities": [entity.model_dump() for entity in entities],
              "terminology_normalization": report}
    assert _terminology_control(result).outcome.value == "PASS"

    _, unresolved = normalizer.normalize_entities(
        [], {"assessment_diagnoses": "XYZ deformity",
             "full_text": "XYZ deformity"})
    result = {"ner_entities": [], "terminology_normalization": unresolved}
    assert _terminology_control(result).outcome.value == "REVIEW_REQUIRED"


def test_registry_validation_fails_closed(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"schema_version": 1, "version": "x"}))
    with pytest.raises(TerminologyConfigError):
        TerminologyNormalizer(path)


def test_unavailable_biomedical_confirmation_is_not_reported_as_certainty():
    entities = [{
        "text": "History of DVT", "clinical_term": "deep vein thrombosis",
        "ner_source": "llm", "ner_confidence": 1.0,
    }]
    with patch("app.ner.biomed_ner.get_confirmed_spans", return_value={}):
        enriched = enrich_entities(entities, "History of DVT")
    assert enriched[0]["ner_source"] == "llm_only"
    assert enriched[0]["ner_confidence"] == 0.7
