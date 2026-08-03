"""Regression tests for generation-safe consistency-run persistence."""

import json

import pytest

from app.validation.run_store import (
    atomic_write_json, inherit_run_metadata, load_runs, persist_runs,
    prune_obsolete_runs,
)


def _commit(results_dir, document_id, runs):
    references = persist_runs(results_dir, document_id, runs)
    main = {
        "document_id": document_id,
        "consistency": {"runs": len(runs), "run_files": references},
    }
    atomic_write_json(results_dir / f"{document_id}_results.json", main)
    prune_obsolete_runs(results_dir, document_id, references)
    return main


def test_new_manifest_cannot_consume_stale_third_run(tmp_path):
    document_id = "note with spaces"
    _commit(tmp_path, document_id, [
        {"generation": "old", "run": 1},
        {"generation": "old", "run": 2},
        {"generation": "old", "run": 3},
    ])
    legacy = tmp_path / "consistency_runs" / f"{document_id}_run3.json"
    legacy.write_text(json.dumps({"generation": "legacy", "run": 3}))

    main = _commit(tmp_path, document_id, [
        {"generation": "new", "run": 1},
        {"generation": "new", "run": 2},
    ])

    assert load_runs(document_id, tmp_path, main) == [
        {"generation": "new", "run": 1},
        {"generation": "new", "run": 2},
    ]
    assert sorted(path.name for path in (tmp_path / "consistency_runs").iterdir()) \
        == sorted(main["consistency"]["run_files"])


def test_uncommitted_generation_does_not_change_previous_manifest(tmp_path):
    main = _commit(tmp_path, "note", [{"generation": "committed"}])
    persist_runs(tmp_path, "note", [{"generation": "orphan"}])
    assert load_runs("note", tmp_path) == [{"generation": "committed"}]
    assert load_runs("note", tmp_path, main) == [{"generation": "committed"}]


def test_manifest_rejects_mixed_or_unsafe_references(tmp_path):
    first = persist_runs(tmp_path, "note", [{"run": 1}, {"run": 2}])
    second = persist_runs(tmp_path, "note", [{"run": 1}, {"run": 2}])
    mixed = {"consistency": {"runs": 2,
                             "run_files": [first[0], second[1]]}}
    with pytest.raises(ValueError, match="one ordered generation"):
        load_runs("note", tmp_path, mixed)

    unsafe = {"consistency": {"runs": 1, "run_files": ["../run.json"]}}
    with pytest.raises(ValueError, match="invalid consistency run reference"):
        load_runs("note", tmp_path, unsafe)


def test_recomputed_report_inherits_manifest_and_strategy():
    previous = {
        "run_files": ["note__0123456789abcdef0123456789abcdef_run1.json"],
        "execution_strategy": {"mode": "adaptive"},
        "disagreements": [{"old": True}],
    }
    rebuilt = inherit_run_metadata({"runs": 1, "disagreements": []}, previous)
    assert rebuilt["run_files"] == previous["run_files"]
    assert rebuilt["execution_strategy"] == {"mode": "adaptive"}
    assert rebuilt["disagreements"] == []
