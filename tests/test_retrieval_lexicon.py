import hashlib
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path

from app.core import config
from app.rag.retrieval_lexicon import (
    RetrievalLexiconRegistry, current_retrieval_lexicon_registry, mapping_key,
)
from tools.recall_benchmark import compare_reports, load_corpus, score_rows
from tools import corroborate_retrieval_lexicon as corroborator


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _catalog(tmp_path: Path, *, pack: dict, source: dict,
             status="active", provenance="government_primary",
             mapping_attestations=None, complete=None):
    source_path = tmp_path / "source.json"
    pack_path = tmp_path / "pack.json"
    _write(source_path, source)
    payload = {"count": len(pack), "terms": pack}
    pack_format = ("governed_v1" if provenance.startswith("generated")
                   and status == "active" else
                   "legacy_generated" if provenance.startswith("generated")
                   else "legacy_primary")
    if pack_format == "governed_v1":
        candidate_path = tmp_path / "candidate.json"
        candidate_payload = {
            "schema_version": 1, "pack_id": "generic-pack-candidate",
            "code_system": "generic", "authority_role": "retrieval_only",
            "provenance_kind": "generated",
            "code_source_sha256": _sha(source_path), "count": len(pack),
            "terms": pack,
        }
        _write(candidate_path, candidate_payload)
        payload.update({"schema_version": 1, "pack_id": "generic-pack",
                        "code_system": "generic",
                        "authority_role": "retrieval_only",
                        "provenance_kind": provenance,
                        "code_source_sha256": _sha(source_path),
                        "source_candidate_sha256": _sha(candidate_path)})
    if mapping_attestations is not None:
        payload["mapping_attestations"] = mapping_attestations
    if complete is not None:
        mapping_count = sum(map(len, pack.values()))
        payload.update({"complete": complete,
                        "candidate_mapping_count": mapping_count,
                        "evaluated_mapping_count": mapping_count,
                        "accepted_mapping_count": mapping_count,
                        "rejected_mapping_count": 0,
                        "rejected_mapping_fingerprints": []})
    _write(pack_path, payload)
    catalog = {
        "schema_version": 1, "version": "test",
        "policy": {"minimum_term_characters": 2,
                   "maximum_term_characters": 80,
                   "maximum_term_code_fanout": 1,
                   "minimum_generated_independence_domains": 2},
        "packs": [{"id": "generic-pack", "status": status,
                   "code_system": "generic", "authority_role": "retrieval_only",
                   "provenance_kind": provenance,
                   "pack_format": pack_format,
                   "path": "pack.json", "pack_sha256": _sha(pack_path),
                   "code_source": "source.json",
                   "code_source_sha256": _sha(source_path),
                   "attestations": []}],
    }
    if pack_format == "governed_v1":
        catalog["packs"][0].update({
            "candidate_source": "candidate.json",
            "candidate_source_sha256": _sha(candidate_path),
        })
    catalog_path = tmp_path / "catalog.json"
    _write(catalog_path, catalog)
    return catalog_path, pack_path, source_path


def test_active_primary_pack_filters_ambiguous_terms(monkeypatch, tmp_path):
    source = [{"code": "ENTITY_ONE", "description": "first descriptor"},
              {"code": "ENTITY_TWO", "description": "second descriptor"}]
    pack = {"ENTITY_ONE": ["unique first", "shared phrase"],
            "ENTITY_TWO": ["unique second", "shared phrase"]}
    catalog, _pack, _source = _catalog(tmp_path, pack=pack, source=source)
    monkeypatch.setattr(config, "BASE_DIR", tmp_path)
    registry = RetrievalLexiconRegistry(catalog)
    assert registry.report["errors"] == []
    assert registry.synonyms_for("generic") == {
        "ENTITYONE": ["unique first"],
        "ENTITYTWO": ["unique second"],
    }
    active = registry.report["active_packs"][0]
    assert active["rejected_fanout_count"] == 2
    assert active["authority_role"] == "retrieval_only"


def test_active_packs_apply_fanout_across_catalog(monkeypatch, tmp_path):
    source = [{"code": "ENTITY_ONE", "description": "first descriptor"},
              {"code": "ENTITY_TWO", "description": "second descriptor"}]
    catalog_path, _pack, _source = _catalog(
        tmp_path, pack={"ENTITY_ONE": ["shared phrase"]}, source=source)
    second_path = tmp_path / "pack-two.json"
    _write(second_path, {"count": 1,
                         "terms": {"ENTITY_TWO": ["shared phrase"]}})
    catalog = json.loads(catalog_path.read_text())
    second = dict(catalog["packs"][0])
    second.update({"id": "generic-pack-two", "path": "pack-two.json",
                   "pack_sha256": _sha(second_path)})
    catalog["packs"].append(second)
    _write(catalog_path, catalog)
    monkeypatch.setattr(config, "BASE_DIR", tmp_path)
    registry = RetrievalLexiconRegistry(catalog_path)
    assert registry.report["errors"] == []
    assert registry.synonyms_for("generic") == {}
    assert sum(row["rejected_cross_pack_fanout_count"]
               for row in registry.report["active_packs"]) == 2


def test_generated_candidate_is_quarantined(monkeypatch, tmp_path):
    source = [{"code": "ENTITY_ONE", "description": "descriptor"}]
    catalog, _pack, _source = _catalog(
        tmp_path, pack={"ENTITY_ONE": ["candidate phrase"]}, source=source,
        status="candidate", provenance="generated")
    monkeypatch.setattr(config, "BASE_DIR", tmp_path)
    registry = RetrievalLexiconRegistry(catalog)
    assert registry.report["errors"] == []
    assert registry.synonyms_for("generic") == {}
    row = registry.report["quarantined_packs"][0]
    assert row["accepted_term_count"] == 0
    assert "independent" in " ".join(row["quarantine_reasons"])


def test_generated_active_pack_requires_mapping_level_domains(
        monkeypatch, tmp_path):
    source = [{"code": "ENTITY_ONE", "description": "descriptor"}]
    term = "corroborated phrase"
    attestations = {mapping_key("ENTITY_ONE", term): [
        {"profile_id": "one", "provider": "provider_one",
         "model": "model-one", "independence_domain": "provider_one",
         "decision": "approved"},
        {"profile_id": "two", "provider": "provider_two",
         "model": "model-two", "independence_domain": "provider_two",
         "decision": "approved"},
    ]}
    catalog, _pack, _source = _catalog(
        tmp_path, pack={"ENTITY_ONE": [term]}, source=source,
        status="active", provenance="generated_corroborated",
        mapping_attestations=attestations, complete=True)
    monkeypatch.setattr(config, "BASE_DIR", tmp_path)
    registry = RetrievalLexiconRegistry(catalog)
    assert registry.report["errors"] == []
    assert registry.synonyms_for("generic") == {"ENTITYONE": [term]}
    assert registry.report["active_packs"][0][
        "independence_domains"] == ["provider_one", "provider_two"]


def test_generated_active_pack_with_one_domain_is_quarantined(
        monkeypatch, tmp_path):
    source = [{"code": "ENTITY_ONE", "description": "descriptor"}]
    term = "single-source phrase"
    attestations = {mapping_key("ENTITY_ONE", term): [
        {"profile_id": "one", "provider": "provider_one",
         "model": "model-one", "independence_domain": "provider_one",
         "decision": "approved"},
    ]}
    catalog, _pack, _source = _catalog(
        tmp_path, pack={"ENTITY_ONE": [term]}, source=source,
        status="active", provenance="generated_corroborated",
        mapping_attestations=attestations, complete=True)
    monkeypatch.setattr(config, "BASE_DIR", tmp_path)
    registry = RetrievalLexiconRegistry(catalog)
    assert registry.synonyms_for("generic") == {}
    assert registry.report["quarantined_packs"]


def test_generated_accounting_rejects_duplicate_rejection_evidence(
        monkeypatch, tmp_path):
    source = [{"code": "ENTITY_ONE", "description": "descriptor"}]
    term = "corroborated phrase"
    attestations = {mapping_key("ENTITY_ONE", term): [
        {"profile_id": "one", "provider": "provider_one",
         "model": "model-one", "independence_domain": "provider_one",
         "decision": "approved"},
        {"profile_id": "two", "provider": "provider_two",
         "model": "model-two", "independence_domain": "provider_two",
         "decision": "approved"},
    ]}
    catalog, pack_path, _source = _catalog(
        tmp_path, pack={"ENTITY_ONE": [term]}, source=source,
        status="active", provenance="generated_corroborated",
        mapping_attestations=attestations, complete=True)
    payload = json.loads(pack_path.read_text())
    duplicate = "sha256:" + "0" * 64
    payload.update({"candidate_mapping_count": 3,
                    "evaluated_mapping_count": 3,
                    "rejected_mapping_count": 2,
                    "rejected_mapping_fingerprints": [duplicate, duplicate]})
    _write(pack_path, payload)
    catalog_payload = json.loads(catalog.read_text())
    catalog_payload["packs"][0]["pack_sha256"] = _sha(pack_path)
    _write(catalog, catalog_payload)
    monkeypatch.setattr(config, "BASE_DIR", tmp_path)
    registry = RetrievalLexiconRegistry(catalog)
    assert registry.synonyms_for("generic") == {}
    reasons = registry.report["quarantined_packs"][0]["quarantine_reasons"]
    assert "accounting" in " ".join(reasons)


def test_source_tampering_fails_registry(monkeypatch, tmp_path):
    source = [{"code": "ENTITY_ONE", "description": "descriptor"}]
    catalog, _pack, source_path = _catalog(
        tmp_path, pack={"ENTITY_ONE": ["phrase"]}, source=source)
    source_path.write_text("[]")
    monkeypatch.setattr(config, "BASE_DIR", tmp_path)
    registry = RetrievalLexiconRegistry(catalog)
    assert registry.synonyms_for("generic") == {}
    assert "source bytes changed" in " ".join(registry.report["errors"])


def test_registry_cache_detects_metadata_preserving_pack_edit(
        monkeypatch, tmp_path):
    source = [{"code": "ENTITY_ONE", "description": "descriptor"}]
    catalog, pack_path, _source = _catalog(
        tmp_path, pack={"ENTITY_ONE": ["phrase one"]}, source=source)
    monkeypatch.setattr(config, "BASE_DIR", tmp_path)
    first = current_retrieval_lexicon_registry(catalog)
    assert first.report["errors"] == []
    stat = pack_path.stat()
    original = pack_path.read_text()
    changed = original.replace("phrase one", "phrase two")
    assert len(changed) == len(original)
    pack_path.write_text(changed)
    os.utime(pack_path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    second = current_retrieval_lexicon_registry(catalog)
    assert second is not first
    assert "pack bytes" in " ".join(second.report["errors"])


def test_benchmark_corpus_is_data_bound_and_comparison_detects_burden(tmp_path):
    path = tmp_path / "benchmark.json"
    _write(path, {"schema_version": 1, "version": "test",
                  "purpose": "retrieval_evaluation_only",
                  "authority_role": "non_production_benchmark",
                  "probes": [{"query": "documented phrase",
                              "expected_code": "EXPECTED_ID",
                              "code_system": "icd10"}]})
    corpus = load_corpus(path)
    assert corpus["corpus_fingerprint"].startswith("sha256:")
    probe = "sha256:probe"
    baseline_rows = [{"probe_fingerprint": probe, "code_system": "icd10",
                      "rank": 1, "false_candidates_ahead": 0}]
    candidate_rows = [{"probe_fingerprint": probe, "code_system": "icd10",
                       "rank": 2, "false_candidates_ahead": 1}]
    assert score_rows(candidate_rows)["overall"][
        "mean_false_candidates_ahead"] == 1
    comparison = compare_reports(
        {"corpus_fingerprint": corpus["corpus_fingerprint"],
         "rows": baseline_rows},
        {"corpus_fingerprint": corpus["corpus_fingerprint"],
         "rows": candidate_rows})
    assert comparison["rank_regressions"] == 1
    assert comparison["additional_false_candidates_ahead"] == 1
    assert comparison["passes_no_regression_gate"] is False
    try:
        compare_reports(
            {"corpus_fingerprint": corpus["corpus_fingerprint"],
             "rows": baseline_rows},
            {"corpus_fingerprint": corpus["corpus_fingerprint"], "rows": []})
    except ValueError as exc:
        assert "identical probe sets" in str(exc)
    else:
        raise AssertionError("missing benchmark probes must fail comparison")


def test_corroborator_keeps_only_cross_domain_unanimity(monkeypatch, tmp_path):
    @dataclass(frozen=True)
    class Profile:
        profile_id: str
        provider: str
        model: str
        independence_domain: str

        def model_dump(self):
            return asdict(self)

    source = tmp_path / "source.json"
    candidate = tmp_path / "candidate.json"
    _write(source, [{"code": "ENTITY_ONE", "description": "descriptor"}])
    terms = {"ENTITY_ONE": ["accepted phrase", "disputed phrase"]}
    _write(candidate, {
        "authority_role": "retrieval_only", "provenance_kind": "generated",
        "pack_id": "candidate", "code_system": "generic",
        "code_source_sha256": _sha(source), "terms": terms,
    })
    profiles = [
        Profile("one", "one", "model-one", "one"),
        Profile("two", "two", "model-two", "two"),
    ]

    def fake_judge(profile, mappings):
        return {
            row["mapping_id"]: {
                "decision": ("rejected" if profile.profile_id == "two"
                             and row["term"] == "disputed phrase" else "approved"),
                "reason": "test",
            }
            for row in mappings
        }

    monkeypatch.setattr(corroborator, "_judge_batch", fake_judge)
    result = corroborator.corroborate(
        candidate_path=candidate, code_source=source,
        profiles=profiles, batch_size=10)
    assert result["complete"] is True
    assert result["terms"] == {"ENTITYONE": ["accepted phrase"]}
    assert result["accepted_mapping_count"] == 1
    assert result["rejected_mapping_count"] == 1
    attestations = result["mapping_attestations"][
        mapping_key("ENTITY_ONE", "accepted phrase")]
    assert {row["independence_domain"] for row in attestations} == {"one", "two"}
