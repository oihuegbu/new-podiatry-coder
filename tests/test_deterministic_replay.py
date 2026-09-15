"""Sealed deterministic run-contract regression tests.

These tests deliberately use synthetic identifiers only.  They prove the
entrypoint's repeatability boundary, not any medical-code scenario.
"""

import run
from claude_coder import pipeline
from types import SimpleNamespace


class _Source:
    def data_fingerprint(self):
        return {"fingerprint_sha256": "sha256:" + "a" * 64}


def _contract(monkeypatch):
    monkeypatch.setattr(run, "decision_source_tree_digest",
                        lambda: "sha256:" + "b" * 64)
    return run.note_run_contract(
        run.run_contract(_Source(), {"billing": "synthetic"}, ""),
        "sha256:synthetic-document")


def test_same_inputs_produce_the_same_sealed_run_contract(monkeypatch):
    contract = _contract(monkeypatch)
    repeated = _contract(monkeypatch)

    assert run._valid_run_contract(contract)
    assert pipeline._valid_run_contract(contract)
    assert contract == repeated


def test_any_contract_change_is_visible_in_the_new_attempt(monkeypatch):
    contract = _contract(monkeypatch)
    changed = dict(contract)
    changed["controls"] = {**contract["controls"], "graph_consensus": not contract[
        "controls"]["graph_consensus"]}
    changed = run._seal_run_contract(changed)

    assert changed != contract
    assert run._valid_run_contract(changed)


def test_unsealed_contract_is_not_a_valid_attempt_identity(monkeypatch):
    contract = _contract(monkeypatch)
    unsealed = dict(contract)
    unsealed["contract_sha256"] = "sha256:" + "0" * 64

    assert run._valid_run_contract(unsealed) is False
    assert pipeline._valid_run_contract(unsealed) is False


def test_held_bundle_path_still_binds_the_sealed_contract(monkeypatch):
    contract = _contract(monkeypatch)
    result = SimpleNamespace(certificate=None)

    binding = run.authority_binding(
        result, _Source(), execution_contract=contract)

    assert binding.model_profiles["execution_contract"] == contract
