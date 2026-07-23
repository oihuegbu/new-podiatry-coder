"""Tests for the audit-convergence loop — the automation that turns a
grounded clinical-review dispute into a verified adjudicated claim, then
into deterministic structure, then replays the note until the review has
nothing left to dispute.

Covers:
  1. Finding/verdict -> disputed-item translation (mechanizable vs
     residual, advisory findings never mutate billing, field allowlists
     per finding kind, merged fields on repeat findings).
  2. Donor materialization — an 'include' target is rebuilt only from the
     data (supporting_conditions, reference descriptors), never invented.
  3. Single-primary enforcement after an adjudicated type promotion.
  4. The loop driver — converges when disputes clear, stalls (and leaves
     disputes with a human) when an iteration produces no adjudications,
     no accepted rules, and no claim changes.

Everything runs against stubs — no live reference data, no network.

Run:  PYTHONPATH=. .venv/bin/python -m pytest tests/test_audit_convergence.py -q
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools.coder_adjudicator import (_adjudicated_code_targets,
                                     _audit_disputed_items,
                                     _enforce_single_primary,
                                     _fresh_review_contradicts, _item_key,
                                     _materialize_donor, _sig_row,
                                     _split_disagreement_keys)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def _result_with_audit(items=None, findings=None, mats=None):
    return {
        "success": True,
        "document_id": "doc1",
        "icd_codes": [{"code": "AAA.1", "type": "primary",
                       "description": "fixture dx one"},
                      {"code": "BBB.2", "type": "secondary",
                       "description": "fixture dx two"}],
        "cpt_codes": [{"code": "11111", "description": "fixture proc",
                       "modifiers": [], "units": 1}],
        "hcpcs_codes": [],
        "supporting_conditions": [
            {"code": "CCC.3", "description": "fixture demoted dx",
             "review_reason": "demoted by fixture layer"}],
        "material_corrections": mats or [],
        "clinical_audit": {
            "verdict": "disputed",
            "fingerprint": "f" * 16,
            "items": items or [],
            "claim_findings": findings or [],
        },
    }


# ---------------------------------------------------------------------------
# 1. finding/verdict -> disputed items
# ---------------------------------------------------------------------------

class DisputedItemsTest(unittest.TestCase):
    def test_overturned_removal_becomes_presence_item(self):
        main = _result_with_audit(
            mats=[{"category": "x", "code": "22222", "action": "removal",
                   "interpretive": True, "message": "removed 22222"}],
            items=[{"index": 0, "verdict": "overturn",
                    "authority": "fixture authority",
                    "note_evidence": "fixture quote"}])
        items, residual = _audit_disputed_items(main)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["kind"], "presence")
        self.assertEqual(items[0]["code"], "22222")
        self.assertEqual(items[0]["array"], "cpt_codes")
        self.assertEqual(items[0]["allegation"]["source"],
                         "clinical_review/correction_verdict")
        self.assertEqual(residual, [])

    def test_upheld_corrections_are_not_disputed(self):
        main = _result_with_audit(
            mats=[{"category": "x", "code": "22222", "action": "removal",
                   "interpretive": True, "message": "removed 22222"}],
            items=[{"index": 0, "verdict": "uphold",
                    "authority": "a", "note_evidence": "e"}])
        items, _ = _audit_disputed_items(main)
        self.assertEqual(items, [])

    def test_missing_code_finding_becomes_presence_item(self):
        main = _result_with_audit(findings=[{
            "kind": "missing_code", "array": "icd_codes", "code": "CCC.3",
            "materiality": "billing_material",
            "finding": "documented dx missing from the claim",
            "authority": "fixture", "note_evidence": "quote"}])
        items, residual = _audit_disputed_items(main)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["kind"], "presence")
        self.assertEqual(items[0]["array"], "icd_codes")
        self.assertEqual(residual, [])

    def test_field_allowlists_by_kind(self):
        main = _result_with_audit(findings=[
            {"kind": "primary_designation", "array": "icd_codes",
             "code": "BBB.2", "materiality": "billing_material",
             "finding": "wrong primary", "authority": "a",
             "note_evidence": "e"},
            {"kind": "modifier", "array": "cpt_codes", "code": "11111",
             "materiality": "uncertain", "finding": "modifier issue",
             "authority": "a", "note_evidence": "e"},
        ])
        items, _ = _audit_disputed_items(main)
        by_code = {i["code"]: i for i in items}
        self.assertEqual(by_code["BBB.2"]["fields"], ["type"])
        self.assertEqual(by_code["11111"]["fields"], ["modifiers"])
        for i in items:
            self.assertEqual(i["kind"], "attributes")

    def test_advisory_materiality_findings_never_become_items(self):
        # a plain advisory-materiality finding (style/nicety) mutates
        # nothing and raises no residual — it only feeds rule growth
        main = _result_with_audit(findings=[{
            "kind": "other", "array": "claim", "code": "",
            "materiality": "advisory", "finding": "style nicety",
            "authority": "a", "note_evidence": "e"}])
        items, residual = _audit_disputed_items(main)
        self.assertEqual(items, [])
        self.assertEqual(residual, [])

    def test_advisory_defect_without_code_is_residual(self):
        # an advisory_defect names the advisory's CODE or it cannot be
        # resolved to a machine identity — residual, never a guess
        main = _result_with_audit(findings=[{
            "kind": "advisory_defect", "array": "claim", "code": "",
            "materiality": "advisory", "finding": "wrong advisory",
            "authority": "a", "note_evidence": "e"}])
        items, residual = _audit_disputed_items(main)
        self.assertEqual(items, [])
        self.assertEqual(len(residual), 1)
        self.assertIn("advisory_defect", residual[0])

    def test_claim_level_material_finding_is_residual(self):
        main = _result_with_audit(findings=[{
            "kind": "coverage", "array": "claim", "code": "",
            "materiality": "billing_material",
            "finding": "coverage pathway broken at claim level",
            "authority": "a", "note_evidence": "e"}])
        items, residual = _audit_disputed_items(main)
        self.assertEqual(items, [])
        self.assertEqual(len(residual), 1)
        self.assertIn("coverage", residual[0])

    def test_repeat_findings_on_same_code_merge_fields(self):
        main = _result_with_audit(findings=[
            {"kind": "modifier", "array": "cpt_codes", "code": "11111",
             "materiality": "billing_material", "finding": "mod",
             "authority": "a", "note_evidence": "e"},
            {"kind": "units", "array": "cpt_codes", "code": "11111",
             "materiality": "billing_material", "finding": "units",
             "authority": "a", "note_evidence": "e"},
        ])
        items, _ = _audit_disputed_items(main)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["fields"], ["modifiers", "units"])

    def test_non_interpretive_corrections_are_skipped(self):
        main = _result_with_audit(
            mats=[{"category": "x", "code": "22222", "action": "removal",
                   "interpretive": False, "message": "data-grounded"}],
            items=[{"index": 0, "verdict": "overturn",
                    "authority": "a", "note_evidence": "e"}])
        items, _ = _audit_disputed_items(main)
        self.assertEqual(items, [])


class SplitDisagreementKeysTest(unittest.TestCase):
    """Jurisdiction on consistency holdouts: the codes the runs disagree
    about belong to the unanimity machinery; review findings about codes
    every run agrees on are adjudicable (a wrong deterministic decision
    is unanimous by construction — consistency can never decide it, and
    the old blanket skip stranded such findings with nobody ruling)."""

    def test_billing_disagreements_claim_their_codes(self):
        keys = _split_disagreement_keys({"disagreements": [
            {"array": "cpt_codes", "code": "29515", "kind": "presence",
             "advisory": False},
            {"array": "icd_codes", "code": "M77.31", "kind": "attributes",
             "advisory": False},
        ]})
        self.assertEqual(keys, {("cpt_codes", "29515"),
                                ("icd_codes", "M77.31")})

    def test_advisory_variance_claims_nothing(self):
        keys = _split_disagreement_keys({"disagreements": [
            {"array": "snomed_codes", "code": "12345", "kind": "presence",
             "advisory": True}]})
        self.assertEqual(keys, set())

    def test_em_level_entry_claims_every_sibling(self):
        keys = _split_disagreement_keys({"disagreements": [
            {"array": "cpt_codes", "code": "99213/99214",
             "kind": "em_level", "advisory": False,
             "codes": ["99213", "99214"]}]})
        self.assertEqual(keys, {("cpt_codes", "99213"),
                                ("cpt_codes", "99214")})

    def test_disjoint_review_findings_survive_the_partition(self):
        # the exact routine_00001 shape: findings on codes all runs agree
        # on (A4570 coverage, missing 27654) must stay adjudicable while
        # the split codes (29515) defer to the consistency machinery
        main = _result_with_audit(findings=[
            {"kind": "coverage", "array": "hcpcs_codes", "code": "A4570",
             "materiality": "billing_material", "finding": "MUE 0",
             "authority": "a", "note_evidence": "e"},
            {"kind": "wrong_code", "array": "cpt_codes", "code": "29515",
             "materiality": "billing_material", "finding": "bundled",
             "authority": "a", "note_evidence": "e"},
        ])
        disputed, _ = _audit_disputed_items(main)
        split = _split_disagreement_keys({"disagreements": [
            {"array": "cpt_codes", "code": "29515", "kind": "presence",
             "advisory": False}]})
        in_scope = [d for d in disputed
                    if (d["array"], d["code"]) not in split]
        deferred = [d for d in disputed
                    if (d["array"], d["code"]) in split]
        self.assertEqual([d["code"] for d in in_scope], ["A4570"])
        self.assertEqual([d["code"] for d in deferred], ["29515"])


class PerCodeTargetExtractionTest(unittest.TestCase):
    """Per-code verified targets: an adjudicated verdict rendered as the
    exact billing rows it mandates, read from the runs after mechanical
    application and before replay — the verdict itself, never whatever a
    replay layer left of it."""

    def _presence(self, code="27654", array="cpt_codes"):
        return {"array": array, "code": code, "kind": "presence",
                "fields": []}

    def test_include_verdict_freezes_the_applied_row(self):
        d = self._presence()
        decisions = {_item_key(d): ("include",)}
        entry = {"code": "27654", "modifiers": ["RT"], "units": 1,
                 "description": "secondary Achilles repair"}
        aligned = [{"cpt_codes": [dict(entry)]},
                   {"cpt_codes": [dict(entry)]}]
        targets = _adjudicated_code_targets([d], decisions, aligned)
        self.assertEqual(targets, [
            {"array": "cpt_codes", "code": "27654",
             "row": {"code": "27654", "modifiers": ["RT"], "units": "1"}}])

    def test_exclude_verdict_freezes_absence(self):
        d = self._presence(code="A4570", array="hcpcs_codes")
        decisions = {_item_key(d): ("exclude",)}
        aligned = [{"hcpcs_codes": []}, {"hcpcs_codes": []}]
        targets = _adjudicated_code_targets([d], decisions, aligned)
        self.assertEqual(targets, [
            {"array": "hcpcs_codes", "code": "A4570", "row": None}])

    def test_divergent_rows_across_runs_yield_no_target(self):
        # the verdict did not realize identically -> nothing verified
        d = self._presence()
        decisions = {_item_key(d): ("include",)}
        aligned = [
            {"cpt_codes": [{"code": "27654", "modifiers": ["RT"],
                            "units": 1}]},
            {"cpt_codes": [{"code": "27654", "modifiers": ["LT"],
                            "units": 1}]}]
        self.assertEqual(
            _adjudicated_code_targets([d], decisions, aligned), [])

    def test_icd_rows_carry_the_diagnosis_type(self):
        row = _sig_row("icd_codes", {"code": "m76.61", "type": "Primary",
                                     "modifiers": [], "units": None})
        self.assertEqual(row, {"code": "M76.61", "modifiers": [],
                               "units": "", "type": "primary"})

    def test_em_level_select_freezes_winner_and_absent_siblings(self):
        d = {"array": "cpt_codes", "kind": "em_level",
             "codes": ["99213", "99214"], "fields": []}
        decisions = {_item_key(d): ("select", "99214")}
        aligned = [{"cpt_codes": [{"code": "99214", "modifiers": [],
                                   "units": 1}]}]
        targets = _adjudicated_code_targets([d], decisions, aligned)
        by_code = {t["code"]: t["row"] for t in targets}
        self.assertEqual(by_code["99214"]["code"], "99214")
        self.assertIsNone(by_code["99213"])


class FreshReviewConcurrenceTest(unittest.TestCase):
    """Only verdicts the fresh post-adjudication review does not side
    AGAINST become per-code targets — a reviewer-vs-adjudicator
    disagreement is a human case, never verified truth."""

    def _setup(self, decision, finding_kind, code="27654",
               array="cpt_codes", billed=False, materiality
               ="billing_material"):
        d = {"array": array, "code": code, "kind": "presence",
             "fields": []}
        decisions = {_item_key(d): (decision,)}
        block = {"claim_findings": [
            {"kind": finding_kind, "array": array, "code": code,
             "materiality": materiality, "finding": "x"}], "items": []}
        payload = {a: [] for a in ("icd_codes", "cpt_codes",
                                   "hcpcs_codes")}
        if billed:
            payload[array] = [{"code": code}]
        return block, [d], decisions, payload

    def test_missing_code_finding_agrees_with_include(self):
        # the layer stripped the line; the reviewer reporting it missing
        # is AGREEING with the adjudicator's include verdict
        block, disputed, decisions, payload = self._setup(
            "include", "missing_code")
        self.assertEqual(_fresh_review_contradicts(
            block, disputed, decisions, payload), set())

    def test_wrong_code_finding_contests_an_include(self):
        block, disputed, decisions, payload = self._setup(
            "include", "wrong_code", billed=True)
        self.assertEqual(_fresh_review_contradicts(
            block, disputed, decisions, payload),
            {("cpt_codes", "27654")})

    def test_coverage_finding_agrees_with_exclude(self):
        block, disputed, decisions, payload = self._setup(
            "exclude", "coverage", code="A4570", array="hcpcs_codes",
            billed=True)
        self.assertEqual(_fresh_review_contradicts(
            block, disputed, decisions, payload), set())

    def test_missing_code_finding_contests_an_exclude(self):
        block, disputed, decisions, payload = self._setup(
            "exclude", "missing_code", code="A4570", array="hcpcs_codes")
        self.assertEqual(_fresh_review_contradicts(
            block, disputed, decisions, payload),
            {("hcpcs_codes", "A4570")})

    def test_advisory_findings_never_contest(self):
        block, disputed, decisions, payload = self._setup(
            "include", "wrong_code", billed=True, materiality="advisory")
        self.assertEqual(_fresh_review_contradicts(
            block, disputed, decisions, payload), set())

    def test_upheld_removal_of_an_included_absent_code_contests(self):
        # the fresh review ENDORSED the correction that removed the code
        # the adjudicator ruled present, and the code is indeed gone —
        # the reviewer sides with the layer, so no target
        d = {"array": "cpt_codes", "code": "27654", "kind": "presence",
             "fields": []}
        decisions = {_item_key(d): ("include",)}
        payload = {
            "icd_codes": [], "cpt_codes": [], "hcpcs_codes": [],
            "material_corrections": [
                {"category": "undocumented_procedure_indication",
                 "code": "27654", "action": "auto_correction",
                 "interpretive": True, "message": "removed"}],
        }
        block = {"claim_findings": [],
                 "items": [{"index": 0, "verdict": "uphold",
                            "authority": "a", "note_evidence": "e"}]}
        self.assertEqual(_fresh_review_contradicts(
            block, [d], decisions, payload), {("cpt_codes", "27654")})


class RecordAdjudicatedCodesTest(unittest.TestCase):
    """The registry's per-code target events: append-only, idempotent,
    always outranked by a full-claim human/adjudicated record."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "registry.jsonl"

    def tearDown(self):
        self.tmp.cleanup()

    def _record(self, targets=None):
        from tools.claims_registry import record_adjudicated_codes
        return record_adjudicated_codes(
            "doc1",
            targets or [{"array": "cpt_codes", "code": "27654",
                         "row": {"code": "27654", "modifiers": ["RT"],
                                 "units": "1"}}],
            "doc1_results.json", by="coder-llm/test",
            registry_path=self.path)

    def test_records_and_reads_back(self):
        from tools.claims_registry import verified_code_targets
        self.assertIsNotNone(self._record())
        targets = verified_code_targets(registry_path=self.path)
        self.assertEqual(
            targets["doc1"][("cpt_codes", "27654")],
            {"code": "27654", "modifiers": ["RT"], "units": "1"})

    def test_identical_targets_are_idempotent(self):
        self.assertIsNotNone(self._record())
        self.assertIsNone(self._record())
        from tools.claims_registry import load_events
        self.assertEqual(len(load_events(self.path)), 1)

    def test_changed_targets_append_and_latest_wins(self):
        from tools.claims_registry import verified_code_targets
        self._record()
        self._record([{"array": "cpt_codes", "code": "27654",
                       "row": None}])
        targets = verified_code_targets(registry_path=self.path)
        self.assertIsNone(targets["doc1"][("cpt_codes", "27654")])

    def test_full_claim_record_supersedes_per_code_targets(self):
        from tools.claims_registry import (record_adjudicated,
                                           verified_code_targets)
        self._record()
        result = {"icd_codes": [], "cpt_codes": [], "hcpcs_codes": [],
                  "final_disposition": "CLEAN", "consistency": {}}
        record_adjudicated("doc1", result, "doc1_results.json",
                           by="coder-llm/test", registry_path=self.path)
        self.assertNotIn(
            "doc1", verified_code_targets(registry_path=self.path))

    def test_human_record_blocks_new_per_code_targets(self):
        from tools.claims_registry import (append_events,
                                           make_finalized_event)
        result = {"icd_codes": [], "cpt_codes": [], "hcpcs_codes": [],
                  "final_disposition": "CLEAN"}
        append_events([make_finalized_event(
            "doc1", result, verification="human", verified_by="coder",
            source="s")], self.path)
        self.assertIsNone(self._record())


# ---------------------------------------------------------------------------
# 2. donor materialization
# ---------------------------------------------------------------------------

class _StubRep:
    class _DB:
        def validate_icd10(self, code):
            if str(code).upper().replace(".", "") == "DDD4":
                return {"description": "fixture descriptor dx"}
            return None

        def validate_cpt(self, code):
            return None

        def validate_hcpcs(self, code):
            return None

    class _Store:
        def icd10_tabular_description(self, cat):
            return ""

        def use_additional_code_groups(self, code):
            return []

        def mue(self, code):
            return None

        def mdm_requirements(self, code):
            return None

    def __init__(self):
        self.db = self._DB()
        self.store = self._Store()


class DonorTest(unittest.TestCase):
    def test_demoted_dx_materializes_from_supporting_conditions(self):
        main = _result_with_audit()
        items = [{"array": "icd_codes", "code": "CCC.3",
                  "kind": "presence"}]
        donor = _materialize_donor(_StubRep(), main, [], items)
        self.assertEqual(len(donor["icd_codes"]), 1)
        ent = donor["icd_codes"][0]
        self.assertEqual(ent["code"], "CCC.3")
        self.assertEqual(ent["description"], "fixture demoted dx")
        self.assertEqual(ent["type"], "secondary")

    def test_reference_descriptor_materializes_unknown_dx(self):
        main = _result_with_audit()
        items = [{"array": "icd_codes", "code": "DDD.4",
                  "kind": "presence"}]
        donor = _materialize_donor(_StubRep(), main, [], items)
        self.assertEqual(donor["icd_codes"][0]["description"],
                         "fixture descriptor dx")

    def test_codes_without_identity_stay_unmaterializable(self):
        main = _result_with_audit()
        items = [{"array": "cpt_codes", "code": "99998",
                  "kind": "presence"}]
        donor = _materialize_donor(_StubRep(), main, [], items)
        self.assertEqual(donor["cpt_codes"], [])

    def test_codes_already_billed_are_not_duplicated(self):
        main = _result_with_audit()
        items = [{"array": "icd_codes", "code": "AAA.1",
                  "kind": "presence"}]
        donor = _materialize_donor(_StubRep(), main, [], items)
        self.assertEqual(donor["icd_codes"], [])


# ---------------------------------------------------------------------------
# 3. single-primary enforcement
# ---------------------------------------------------------------------------

class SinglePrimaryTest(unittest.TestCase):
    def test_promotion_demotes_the_other_primary_and_leads_the_array(self):
        run = {"icd_codes": [
            {"code": "AAA.1", "type": "primary"},
            {"code": "BBB.2", "type": "secondary"},
        ]}
        d = {"array": "icd_codes", "code": "BBB.2", "kind": "attributes",
             "fields": ["type"]}
        decisions = {_item_key(d): ("set", (("type", '"primary"'),))}
        # simulate the applied decision, then the invariant
        run["icd_codes"][1]["type"] = "primary"
        _enforce_single_primary(run, decisions, [d])
        self.assertEqual(run["icd_codes"][0]["code"], "BBB.2")
        self.assertEqual(run["icd_codes"][0]["type"], "primary")
        self.assertEqual(run["icd_codes"][1]["type"], "secondary")

    def test_no_promotion_means_no_change(self):
        run = {"icd_codes": [
            {"code": "AAA.1", "type": "primary"},
            {"code": "BBB.2", "type": "secondary"},
        ]}
        d = {"array": "icd_codes", "code": "BBB.2", "kind": "attributes",
             "fields": ["modifiers"]}
        decisions = {_item_key(d): ("set", (("modifiers", '["XX"]'),))}
        _enforce_single_primary(run, decisions, [d])
        self.assertEqual(run["icd_codes"][0]["type"], "primary")
        self.assertEqual(run["icd_codes"][0]["code"], "AAA.1")


# ---------------------------------------------------------------------------
# 4. the loop driver
# ---------------------------------------------------------------------------

class ConvergeTest(unittest.TestCase):
    def _write(self, d: Path, doc: str, verdict: str):
        payload = _result_with_audit()
        payload["document_id"] = doc
        payload["clinical_audit"]["verdict"] = verdict
        (d / f"{doc}_results.json").write_text(
            json.dumps(payload, default=str))

    def _run(self, results_dir: Path, adjudicate_side_effect):
        from tools import audit_convergence_loop as acl
        audits = {"calls": 0}

        def fake_audit_batch(results_dir_, docs=None):
            audits["calls"] += 1
            return {"audited": 0, "upheld": 0, "disputed": 0,
                    "skipped": 0, "docs": {}}

        with mock.patch("tools.clinical_auditor.audit_batch",
                        side_effect=fake_audit_batch), \
             mock.patch("tools.coder_adjudicator.adjudicate_audit",
                        side_effect=adjudicate_side_effect), \
             mock.patch("tools.auto_actuate.actuate",
                        return_value={"actuated": 0}), \
             mock.patch("tools.auto_actuate.Replayer") as rep_cls, \
             mock.patch.object(acl, "replay_scope", return_value=0), \
             mock.patch("tools.flip_triage.scan",
                        return_value={"new_classes": 0}), \
             mock.patch("tools.claims_registry.ingest",
                        return_value={"recorded": 0, "unchanged": 0,
                                      "human_protected": 0, "skipped": 0}):
            rep_cls.return_value = object()
            return acl.converge(results_dir, max_iterations=3)

    def test_converges_when_adjudication_clears_the_dispute(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            self._write(d, "doc1", "disputed")

            def adjudicate(results_dir, docs=None, rep=None, **kw):
                self._write(d, "doc1", "upheld")
                return {"adjudicated": 1, "still_disputed": 0}

            summary = self._run(d, adjudicate)
        self.assertEqual(summary["status"], "converged")
        self.assertEqual(summary["final_disputed"], [])
        self.assertEqual(summary["iterations"][0]["adjudicated"], 1)

    def test_stalls_and_leaves_disputes_for_a_human(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            self._write(d, "doc1", "disputed")

            def adjudicate(results_dir, docs=None, rep=None, **kw):
                return {"adjudicated": 0, "still_disputed": 1}

            summary = self._run(d, adjudicate)
        self.assertEqual(summary["status"], "stalled")
        self.assertEqual(summary["final_disputed"], ["doc1"])
        # exactly one working iteration: no progress -> stop immediately
        self.assertEqual(len(summary["iterations"]), 1)

    def test_no_disputes_is_an_immediate_convergence(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            self._write(d, "doc1", "upheld")
            summary = self._run(
                d, lambda *a, **k: {"adjudicated": 0, "still_disputed": 0})
        self.assertEqual(summary["status"], "converged")
        self.assertEqual(summary["iterations"], [{"iteration": 1,
                                                  "disputed": 0}])


# ---------------------------------------------------------------------------
# 5. adjudication survival — no layer silently outvotes the adjudicator
# ---------------------------------------------------------------------------

from tools.coder_adjudicator import (_adjudication_conflicts,  # noqa: E402
                                     _apply_override_hold, recheck_survival)


class SurvivalInvariantTest(unittest.TestCase):
    """The routine_00008 defect, as a permanent regression test: the
    adjudicator set modifiers=[] on a line, a replay layer re-added the
    modifier, and the claim shipped CLEAN. The invariant must catch every
    shape of override deterministically."""

    def _claim(self, mods=("XX",)):
        return {"cpt_codes": [{"code": "11111", "modifiers": list(mods),
                               "units": 1}],
                "icd_codes": [], "hcpcs_codes": []}

    def _attr_case(self):
        d = {"array": "cpt_codes", "code": "11111", "kind": "attributes",
             "fields": ["modifiers"]}
        decisions = {_item_key(d): ("set", (("modifiers", "[]"),))}
        return d, decisions

    def test_overridden_attribute_is_a_conflict(self):
        d, decisions = self._attr_case()
        conflicts = _adjudication_conflicts(self._claim(("XX",)),
                                            decisions, [d])
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0]["code"], "11111")
        self.assertIn("modifiers", conflicts[0]["decision"])

    def test_realized_attribute_is_clean(self):
        d, decisions = self._attr_case()
        conflicts = _adjudication_conflicts(self._claim(()),
                                            decisions, [d])
        self.assertEqual(conflicts, [])

    def test_presence_exclude_still_billed_is_a_conflict(self):
        d = {"array": "cpt_codes", "code": "11111", "kind": "presence"}
        decisions = {_item_key(d): ("exclude",)}
        conflicts = _adjudication_conflicts(self._claim(), decisions, [d])
        self.assertEqual(len(conflicts), 1)
        self.assertIn("still on the final claim", conflicts[0]["observed"])

    def test_presence_include_dropped_is_a_conflict(self):
        d = {"array": "icd_codes", "code": "AAA.1", "kind": "presence"}
        decisions = {_item_key(d): ("include",)}
        conflicts = _adjudication_conflicts(self._claim(), decisions, [d])
        self.assertEqual(len(conflicts), 1)
        self.assertIn("absent", conflicts[0]["observed"])

    def test_hold_blocks_clean_and_names_the_conflict(self):
        payload = {"final_disposition": "CLEAN", "auto_coding_tier": "AUTO",
                   "auto_coding_confidence": 0.9, "adjudication": {}}
        _apply_override_hold(payload, [{
            "array": "cpt_codes", "code": "11111", "kind": "attributes",
            "decision": "modifiers = []", "observed": "modifiers = ['XX']",
            "authority": "fixture"}])
        self.assertEqual(payload["final_disposition"], "REVIEW")
        self.assertTrue(payload["adjudication"]["overridden_by_replay"])
        self.assertTrue(any("[adjudication/overridden]" in r for r in
                            payload["auto_coding_review_reasons"]))

    def test_registry_gate_fails_closed_on_override(self):
        from tools.claims_registry import eligible_for_auto
        r = {"success": True,
             "consistency": {"runs": 3, "unanimous": True},
             "final_disposition": "CLEAN",
             "adjudication": {"overridden_by_replay": [{"code": "11111"}]},
             "clinical_audit": {"verdict": "upheld", "fingerprint": "x"}}
        ok, why = eligible_for_auto(r)
        self.assertFalse(ok)
        self.assertIn("overrode", why)

    def test_upheld_review_cannot_promote_an_overridden_claim(self):
        from tools.clinical_auditor import _enforce_verdict
        result = {"final_disposition": "CLEAN", "auto_coding_tier": "AUTO",
                  "auto_coding_confidence": 0.9,
                  "claim_scrub": {"clean": True, "disposition": "CLEAN"},
                  "adjudication": {"overridden_by_replay": [
                      {"array": "cpt_codes", "code": "11111",
                       "decision": "modifiers = []",
                       "observed": "modifiers = ['XX']"}]}}
        _enforce_verdict(result, {"verdict": "upheld"}, [], {}, "")
        self.assertEqual(result["final_disposition"], "REVIEW")
        self.assertTrue(any("[adjudication/overridden]" in r for r in
                            result["auto_coding_review_reasons"]))

    def test_recheck_survival_holds_and_quarantines(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            payload = {
                "document_id": "docx", "success": True,
                "final_disposition": "CLEAN", "auto_coding_tier": "AUTO",
                "auto_coding_confidence": 0.9,
                "icd_codes": [], "hcpcs_codes": [],
                "cpt_codes": [{"code": "11111", "modifiers": ["XX"],
                               "units": 1}],
                "adjudication": {"items": [
                    {"array": "cpt_codes", "code": "11111",
                     "kind": "attributes", "decision": "set",
                     "fields": {"modifiers": []},
                     "authority": "fixture authority",
                     "note_evidence": "fixture"}]},
            }
            (d / "docx_results.json").write_text(json.dumps(payload))
            reg = d / "claims_registry.jsonl"
            reg.write_text(json.dumps({
                "event": "finalized", "document_id": "docx",
                "verification": "adjudicated", "claim": {}}) + "\n")
            stats = recheck_survival(d, registry_path=reg)
            self.assertEqual(stats["conflicted"], 1)
            self.assertEqual(stats["quarantined_records"], 1)
            saved = json.loads((d / "docx_results.json").read_text())
            self.assertEqual(saved["final_disposition"], "REVIEW")
            self.assertEqual(reg.read_text().strip(), "")
            backups = list(d.glob("*.bak_survival_*"))
            self.assertEqual(len(backups), 1)

    def test_recheck_ignores_realized_adjudications(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            payload = {
                "document_id": "docy", "success": True,
                "final_disposition": "CLEAN",
                "icd_codes": [], "hcpcs_codes": [],
                "cpt_codes": [{"code": "11111", "modifiers": [],
                               "units": 1}],
                "adjudication": {"items": [
                    {"array": "cpt_codes", "code": "11111",
                     "kind": "attributes", "decision": "set",
                     "fields": {"modifiers": []},
                     "authority": "a", "note_evidence": "e"}]},
            }
            (d / "docy_results.json").write_text(json.dumps(payload))
            reg = d / "claims_registry.jsonl"
            reg.write_text("")
            stats = recheck_survival(d, registry_path=reg)
            self.assertEqual(stats["conflicted"], 0)
            saved = json.loads((d / "docy_results.json").read_text())
            self.assertEqual(saved["final_disposition"], "CLEAN")


# ---------------------------------------------------------------------------
# 6. correction history survives replays
# ---------------------------------------------------------------------------

class CorrectionCarryTest(unittest.TestCase):
    """The other routine_00008 defect: an E/M removed in the ORIGINAL pass
    (with a medical-necessity flag) vanished from the adjudicated record
    because the replay rebuilt from post-validation arrays. Prior-pass
    corrections must be carried forward, deduplicated and tagged."""

    def _rebuild(self, run, report):
        from tools.replay_reconcile import _rebuild_run

        class _Scrub:
            clean = False

            class disposition:
                value = "REVIEW"
            summary = "fixture"

            def model_dump(self, mode=None):
                return {"clean": False, "disposition": "REVIEW"}

            @property
            def blocking_findings(self):
                return []

        class _Scrubber:
            def scrub(self, payload):
                return _Scrub()

        return _rebuild_run(run, {}, report, _Scrubber(), "note text")

    def test_prior_corrections_carry_forward_tagged(self):
        prior = {"category": "em", "code": "99999",
                 "action": "removal", "interpretive": True,
                 "message": "fixture E/M removed in the original pass"}
        run = {"icd_codes": [], "cpt_codes": [], "hcpcs_codes": [],
               "material_corrections": [prior]}
        out = self._rebuild(run, {"material_corrections": []})
        mats = out["material_corrections"]
        self.assertEqual(len(mats), 1)
        self.assertEqual(mats[0]["code"], "99999")
        self.assertTrue(mats[0]["carried_from_prior_pass"])

    def test_fresh_duplicates_are_not_doubled(self):
        m = {"category": "em", "code": "99999", "action": "removal",
             "interpretive": True, "message": "same correction"}
        run = {"icd_codes": [], "cpt_codes": [], "hcpcs_codes": [],
               "material_corrections": [dict(m)]}
        out = self._rebuild(run, {"material_corrections": [dict(m)]})
        self.assertEqual(len(out["material_corrections"]), 1)
        self.assertNotIn("carried_from_prior_pass",
                         out["material_corrections"][0])


# ---------------------------------------------------------------------------
# 7. autonomous convergence cycles (unanimity loop helpers)
# ---------------------------------------------------------------------------

class ConvergenceCycleHelpersTest(unittest.TestCase):
    """The outer convergence-cycle loop keys on two measurements: the
    CLEAN count (the loop's real goal — unanimity alone is not done) and
    the structure signature (whether finalization minted structure worth
    a fresh generative pass). The signature must see an AMENDMENT — old
    rule disabled, replacement appended — which leaves the enabled-rule
    COUNT unchanged and used to read as 'no progress'."""

    def setUp(self):
        import tools.unanimity_loop as ul
        self.ul = ul
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.rules = root / "validator_rules.json"
        self.results = root / "results"
        self.results.mkdir()
        self._patch = mock.patch.multiple(
            ul, RULES_PATH=self.rules, RESULTS_DIR=self.results,
            AUTO_TEMPLATES_DIR=root / "auto_templates",
            GRADUATED_DIR=root / "graduated")
        self._patch.start()
        self.addCleanup(self._patch.stop)

    def _write_pack(self, rules):
        self.rules.write_text(json.dumps({"rules": rules}))

    def test_amendment_changes_signature_at_equal_count(self):
        self._write_pack([{"id": "rule-a", "enabled": True},
                          {"id": "rule-b", "enabled": True}])
        before = self.ul._structure_sig()
        # amendment: rule-a disabled, rule-a-r2 appended (count unchanged)
        self._write_pack([{"id": "rule-a", "enabled": False},
                          {"id": "rule-a-r2", "enabled": True},
                          {"id": "rule-b", "enabled": True}])
        after = self.ul._structure_sig()
        self.assertNotEqual(before, after)
        self.assertEqual(len(before[0]), len(after[0]))

    def test_disable_only_changes_signature(self):
        self._write_pack([{"id": "rule-a", "enabled": True}])
        before = self.ul._structure_sig()
        self._write_pack([{"id": "rule-a", "enabled": False}])
        self.assertNotEqual(before, self.ul._structure_sig())

    def test_new_template_changes_signature(self):
        self._write_pack([])
        before = self.ul._structure_sig()
        tdir = Path(self.tmp.name) / "auto_templates"
        tdir.mkdir()
        (tdir / "new_mechanic.py").write_text("TEMPLATE_NAME='x'")
        self.assertNotEqual(before, self.ul._structure_sig())

    def test_cleanliness_counts_dispositions_not_unanimity(self):
        (self.results / "doc_a_results.json").write_text(json.dumps(
            {"final_disposition": "CLEAN",
             "consistency": {"unanimous": True}}))
        # unanimous but still REVIEW (e.g. audit-disputed): NOT done
        (self.results / "doc_b_results.json").write_text(json.dumps(
            {"final_disposition": "REVIEW",
             "consistency": {"unanimous": True}}))
        clean, total, non_clean = self.ul._cleanliness(["doc_a", "doc_b"])
        self.assertEqual((clean, total), (1, 2))
        self.assertEqual(non_clean, ["doc_b"])

    def test_cleanliness_missing_file_is_not_clean(self):
        clean, total, non_clean = self.ul._cleanliness(["ghost"])
        self.assertEqual((clean, total), (0, 1))
        self.assertEqual(non_clean, ["ghost"])

    def _write_doc(self, doc, splits=0, advisory=0, disputed=False,
                   material=0, unanimous=False, disposition="REVIEW"):
        dis = ([{"advisory": False}] * splits
               + [{"advisory": True}] * advisory)
        ca = {}
        if disputed:
            ca = {"verdict": "disputed",
                  "claim_findings":
                      [{"materiality": "billing_material"}] * material
                      + [{"materiality": "advisory"}]}
        (self.results / f"{doc}_results.json").write_text(json.dumps(
            {"final_disposition": disposition,
             "consistency": {"unanimous": unanimous,
                             "disagreements": dis},
             "clinical_audit": ca}))

    def _flat_targets(self):
        """Pin the registry-target components to zero so the finding/split
        assertions below stay hermetic (the real _target_progress reads
        the live claims registry)."""
        p = mock.patch.object(self.ul, "_target_progress",
                              return_value=(0, 0))
        p.start()
        self.addCleanup(p.stop)

    def test_progress_vector_sees_split_and_finding_shrinkage(self):
        # A SINGLE note can never gain CLEAN until fully done — patience
        # keyed on CLEAN alone would cap a lone claim at --patience
        # cycles, the fixed budget this loop exists to remove. Splits
        # 4->2 and material findings 3->1 must both read as progress.
        self._flat_targets()
        self._write_doc("doc_a", splits=4, disputed=True, material=3)
        before = self.ul._progress_vector(["doc_a"])
        self._write_doc("doc_a", splits=2, disputed=True, material=1)
        after = self.ul._progress_vector(["doc_a"])
        self.assertTrue(any(a > b for a, b in zip(after, before)))
        self.assertEqual(-after[4], 2)   # billing splits
        self.assertEqual(-after[5], 1)   # material findings

    def test_progress_vector_ignores_advisory_splits(self):
        self._flat_targets()
        self._write_doc("doc_a", splits=1, advisory=30)
        self.assertEqual(-self.ul._progress_vector(["doc_a"])[4], 1)

    def test_progress_vector_disputed_without_findings_counts_one(self):
        # fail closed: an unparseable disputed review is still a blocker
        self._flat_targets()
        self._write_doc("doc_a", disputed=True, material=0)
        self.assertEqual(-self.ul._progress_vector(["doc_a"])[5], 1)

    def test_progress_vector_upheld_flat_note_is_zero_blockers(self):
        self._flat_targets()
        self._write_doc("doc_a", unanimous=True, disposition="CLEAN")
        self.assertEqual(self.ul._progress_vector(["doc_a"]),
                         (1, 1, 0, 0, 0, 0))

    def test_progress_vector_credits_target_components(self):
        # The finding count alone INVERTS on a claim getting better
        # (routine_00003 cycle 2: satisfying an adjudicated target gave
        # the reviewer a richer claim and MORE findings). Satisfied and
        # recorded targets are monotone credits that must outvote it.
        self._write_doc("doc_a", disputed=True, material=1)
        with mock.patch.object(self.ul, "_target_progress",
                               return_value=(0, 1)):
            before = self.ul._progress_vector(["doc_a"])
        # next cycle: MORE findings (2 > 1), but the recorded target is
        # now satisfied and a fresh one was recorded
        self._write_doc("doc_a", disputed=True, material=2)
        with mock.patch.object(self.ul, "_target_progress",
                               return_value=(1, 2)):
            after = self.ul._progress_vector(["doc_a"])
        self.assertTrue(any(a > b for a, b in zip(after, before)))
        self.assertEqual(after[2], 1)   # targets satisfied
        self.assertEqual(after[3], 2)   # targets recorded

    def test_progress_vector_survives_target_progress_crash(self):
        # fail open on measurement, not on the loop: a registry read
        # crash must not take the whole vector down
        self._write_doc("doc_a", unanimous=True, disposition="CLEAN")
        with mock.patch.object(self.ul, "_target_progress",
                               side_effect=RuntimeError("boom")):
            self.assertEqual(self.ul._progress_vector(["doc_a"]),
                             (1, 1, 0, 0, 0, 0))


class TargetProgressTest(unittest.TestCase):
    """_target_progress measures the scope's verified realignment targets:
    recorded = registry ground truth exists; satisfied = the current saved
    record already realizes it (whole-claim signature match, per-code row
    present/absent as verified, observable emission matching the verdict)."""

    def setUp(self):
        import tools.unanimity_loop as ul
        import tools.auto_actuate as aa
        self.ul, self.aa = ul, aa
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.results = Path(self.tmp.name) / "results"
        self.results.mkdir()
        p = mock.patch.object(ul, "RESULTS_DIR", self.results)
        p.start()
        self.addCleanup(p.stop)

    def _write(self, doc, icd=None, cpt=None, scrub=None):
        rec = {"icd_codes": icd or [], "cpt_codes": cpt or [],
               "hcpcs_codes": []}
        if scrub is not None:
            rec["claim_scrub"] = {"findings": scrub}
        (self.results / f"{doc}_results.json").write_text(json.dumps(rec))

    def _targets(self, registry=None, code=None, obs=None):
        return mock.patch.multiple(
            self.aa,
            _registry_verified_claims=mock.Mock(
                return_value=registry or {}),
            _per_code_targets=mock.Mock(return_value=code or {}),
            _advisory_targets=mock.Mock(return_value=obs or {}))

    def test_no_targets_measures_zero(self):
        self._write("doc_a", icd=[{"code": "L60.8", "type": "primary"}])
        with self._targets():
            self.assertEqual(self.ul._target_progress(["doc_a"]), (0, 0))

    def test_per_code_presence_target(self):
        row = ("L60.8", (), "", "primary")
        self._write("doc_a", icd=[{"code": "L60.8", "type": "primary"}])
        self._write("doc_b", icd=[])   # target NOT yet satisfied
        code_t = {"doc_a": {("icd_codes", "L60.8"): row},
                  "doc_b": {("icd_codes", "L60.8"): row}}
        with self._targets(code=code_t):
            self.assertEqual(self.ul._target_progress(["doc_a", "doc_b"]),
                             (1, 2))

    def test_per_code_absence_target(self):
        # row None = verified ABSENT: satisfied only when the code is gone
        self._write("doc_a", icd=[{"code": "Z74.9", "type": "secondary"}])
        code_t = {"doc_a": {("icd_codes", "Z74.9"): None}}
        with self._targets(code=code_t):
            self.assertEqual(self.ul._target_progress(["doc_a"]), (0, 1))
        self._write("doc_a", icd=[])
        with self._targets(code=code_t):
            self.assertEqual(self.ul._target_progress(["doc_a"]), (1, 1))

    def test_whole_claim_registry_target(self):
        self._write("doc_a", icd=[{"code": "L60.8", "type": "primary"}])
        goal = self.aa.Replayer.signature(
            [{"code": "L60.8", "type": "primary"}], [], [])
        with self._targets(registry={"doc_a": goal}):
            self.assertEqual(self.ul._target_progress(["doc_a"]), (1, 1))
        miss = self.aa.Replayer.signature(
            [{"code": "L60.3", "type": "primary"}], [], [])
        with self._targets(registry={"doc_a": miss}):
            self.assertEqual(self.ul._target_progress(["doc_a"]), (0, 1))

    def test_observable_emission_target(self):
        # verified must-NOT-fire: satisfied once the advisory stops firing
        key = ("advisory_emission", "MEDICAL_NECESSITY|11720")
        self._write("doc_a", cpt=[{"code": "11720"}], scrub=[
            {"filter_id": "MEDICAL_NECESSITY", "status": "WARN",
             "codes": ["11720"], "reason": "x"}])
        with self._targets(obs={"doc_a": {key: False}}):
            self.assertEqual(self.ul._target_progress(["doc_a"]), (0, 1))
        self._write("doc_a", cpt=[{"code": "11720"}], scrub=[])
        with self._targets(obs={"doc_a": {key: False}}):
            self.assertEqual(self.ul._target_progress(["doc_a"]), (1, 1))

    def test_missing_record_counts_recorded_not_satisfied(self):
        code_t = {"ghost": {("icd_codes", "L60.8"):
                            ("L60.8", (), "", "primary")}}
        with self._targets(code=code_t):
            self.assertEqual(self.ul._target_progress(["ghost"]), (0, 1))


class AnchoredUnworkedClassesTest(unittest.TestCase):
    """The stall-grace guard: audit-dispute classes that hold a verified
    target but were never actuated (targets land AFTER the cycle's
    actuation pass) must be visible to the outer loop so a STALL verdict
    doesn't strand actionable ground truth."""

    def setUp(self):
        import tools.unanimity_loop as ul
        self.ul = ul

    def _queue(self, classes):
        import tools.flip_triage as ft
        return mock.patch.object(ft, "load_queue", return_value=classes)

    def test_filters_kind_status_and_anchor(self):
        import tools.auto_actuate as aa
        classes = [
            {"class_key": "a", "kind": "audit_dispute", "status": "open"},
            {"class_key": "b", "kind": "audit_dispute",
             "status": "awaiting_verification"},
            {"class_key": "c", "kind": "audit_dispute",
             "status": "resolved"},          # already done
            {"class_key": "d", "kind": "consistency", "status": "open"},
        ]
        with self._queue(classes), \
                mock.patch.object(aa, "_audit_class_anchored",
                                  side_effect=lambda c:
                                  c["class_key"] != "a"):
            self.assertEqual(self.ul._anchored_unworked_classes(), ["b"])

    def test_fails_closed_to_empty_on_crash(self):
        import tools.flip_triage as ft
        with mock.patch.object(ft, "load_queue",
                               side_effect=RuntimeError("boom")):
            self.assertEqual(self.ul._anchored_unworked_classes(), [])


# ---------------------------------------------------------------------------
# 6. advisory-shaped audit disputes — the emission-state pathway
# ---------------------------------------------------------------------------

def _result_with_advisory_scrub(findings=None, scrub=None):
    """A disputed record whose review disputes a scrubber ADVISORY: the
    claim is correct as billed and the dispute's machine identity lives in
    claim_scrub.findings, not in any billed line."""
    main = _result_with_audit(findings=findings)
    main["claim_scrub"] = {"findings": scrub if scrub is not None else [
        {"filter_id": "MEDICAL_NECESSITY", "status": "WARN",
         "codes": ["11111"], "reason": "coverage pathway advisory"},
        {"filter_id": "DOCUMENTATION", "status": "PASS",
         "codes": ["11111"], "reason": "fine"},
    ]}
    return main


class AdvisoryDisputeItemsTest(unittest.TestCase):
    """advisory_defect findings resolve — through the advisory_emission
    OBSERVABLE — to observable-kind disputed items by matching the ONE
    live WARN scrub finding on the code; ambiguity is a human case, never
    a guess, and the verdict mutates no claim array."""

    _FND = {"kind": "advisory_defect", "array": "cpt_codes",
            "code": "11111", "materiality": "advisory",
            "finding": "the advisory demands pathway A; the authority "
                       "recognizes pathway B, which the note documents",
            "authority": "LCD L12345", "note_evidence": "quote"}

    def test_unique_warn_finding_becomes_observable_item(self):
        main = _result_with_advisory_scrub(findings=[dict(self._FND)])
        items, residual = _audit_disputed_items(main)
        self.assertEqual(residual, [])
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["kind"], "observable")
        self.assertEqual(items[0]["code"], "11111")
        self.assertEqual(items[0]["observable"], "advisory_emission")
        self.assertEqual(items[0]["key"], "MEDICAL_NECESSITY|11111")
        self.assertEqual(items[0]["allegation"]["kind"], "advisory_defect")

    def test_no_live_warn_finding_is_residual(self):
        main = _result_with_advisory_scrub(findings=[dict(self._FND)],
                                           scrub=[])
        items, residual = _audit_disputed_items(main)
        self.assertEqual(items, [])
        self.assertEqual(len(residual), 1)
        self.assertIn("not identifiable", residual[0])

    def test_multiple_warn_findings_on_code_are_ambiguous(self):
        main = _result_with_advisory_scrub(
            findings=[dict(self._FND)],
            scrub=[{"filter_id": "MEDICAL_NECESSITY", "status": "WARN",
                    "codes": ["11111"], "reason": "advisory one"},
                   {"filter_id": "LCD_COVERAGE", "status": "WARN",
                    "codes": ["11111"], "reason": "advisory two"}])
        items, residual = _audit_disputed_items(main)
        self.assertEqual(items, [])
        self.assertEqual(len(residual), 1)
        self.assertIn("ambiguous", residual[0])

    def test_norm_decision_accepts_only_suppress_or_stand(self):
        from tools.coder_adjudicator import _norm_decision
        for kind in ("observable", "advisory"):  # advisory = legacy items
            base = {"kind": kind, "array": "cpt_codes", "code": "11111"}
            self.assertEqual(
                _norm_decision(dict(base, decision="suppress")),
                ("suppress",))
            self.assertEqual(_norm_decision(dict(base, decision="stand")),
                             ("stand",))
            self.assertIsNone(
                _norm_decision(dict(base, decision="exclude")))
            self.assertIsNone(
                _norm_decision(dict(base, decision="abstain")))

    def test_observable_verdict_mutates_no_claim_array(self):
        from tools.coder_adjudicator import _apply_to_run
        d = {"array": "cpt_codes", "code": "11111", "kind": "observable",
             "observable": "advisory_emission",
             "key": "MEDICAL_NECESSITY|11111"}
        decisions = {_item_key(d): ("suppress",)}
        run = {"cpt_codes": [{"code": "11111", "modifiers": ["RT"],
                              "units": 1}]}
        out = _apply_to_run(run, decisions, [d], [run])
        self.assertEqual(out["cpt_codes"], run["cpt_codes"])

    def test_observable_verdict_never_freezes_a_code_target(self):
        d = {"array": "cpt_codes", "code": "11111", "kind": "observable",
             "observable": "advisory_emission",
             "key": "MEDICAL_NECESSITY|11111"}
        decisions = {_item_key(d): ("suppress",)}
        aligned = [{"cpt_codes": [{"code": "11111", "modifiers": [],
                                   "units": 1}]}]
        self.assertEqual(
            _adjudicated_code_targets([d], decisions, aligned), [])

    def test_record_observable_targets_records_both_directions(self):
        from tools.coder_adjudicator import _record_observable_targets
        d_sup = {"array": "cpt_codes", "code": "11111",
                 "kind": "observable", "observable": "advisory_emission",
                 "key": "MEDICAL_NECESSITY|11111"}
        d_std = {"array": "cpt_codes", "code": "22222",
                 "kind": "observable", "observable": "advisory_emission",
                 "key": "LCD_COVERAGE|22222"}
        decisions = {_item_key(d_sup): ("suppress",),
                     _item_key(d_std): ("stand",)}
        payload = {"adjudication": {
            "model": "test", "at": "now", "passes": 2,
            "items": [{"array": "cpt_codes", "code": "11111",
                       "kind": "observable", "decision": "suppress",
                       "authority": "LCD L12345"}]}}
        with mock.patch("tools.claims_registry."
                        "record_adjudicated_observables") as rec:
            _record_observable_targets("doc1", "src", payload,
                                       [d_sup, d_std], decisions)
        targets = rec.call_args[0][1]
        by_key = {t["key"]: t for t in targets}
        self.assertFalse(by_key["MEDICAL_NECESSITY|11111"]["emit"])
        self.assertTrue(by_key["LCD_COVERAGE|22222"]["emit"])
        self.assertEqual(by_key["MEDICAL_NECESSITY|11111"]["authority"],
                         "LCD L12345")

    def test_record_observable_targets_accepts_legacy_advisory_items(self):
        # items stored under the pre-generalization shape (kind
        # "advisory", filter_id, no key) must still record — synthesized
        # from filter_id|CODE
        from tools.coder_adjudicator import _record_observable_targets
        d = {"array": "cpt_codes", "code": "11111", "kind": "advisory",
             "filter_id": "MEDICAL_NECESSITY"}
        decisions = {_item_key(d): ("suppress",)}
        payload = {"adjudication": {"model": "test", "at": "now",
                                    "passes": 2, "items": []}}
        with mock.patch("tools.claims_registry."
                        "record_adjudicated_observables") as rec:
            _record_observable_targets("doc1", "src", payload, [d],
                                       decisions)
        t = rec.call_args[0][1][0]
        self.assertEqual(t["observable"], "advisory_emission")
        self.assertEqual(t["key"], "MEDICAL_NECESSITY|11111")
        self.assertFalse(t["emit"])


class RecordAdjudicatedAdvisoriesTest(unittest.TestCase):
    """The registry's advisory-emission target events: append-only,
    idempotent, latest wins — and NEVER superseded by full-claim records
    (those verify claim lines, not advisory emissions)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "registry.jsonl"

    def tearDown(self):
        self.tmp.cleanup()

    def _record(self, targets=None):
        from tools.claims_registry import record_adjudicated_advisories
        return record_adjudicated_advisories(
            "doc1",
            targets or [{"filter_id": "MEDICAL_NECESSITY",
                         "code": "11111", "emit": False,
                         "authority": "LCD L12345"}],
            "doc1_results.json", by="coder-llm/test",
            registry_path=self.path)

    def test_records_and_reads_back(self):
        from tools.claims_registry import verified_advisory_targets
        self.assertIsNotNone(self._record())
        targets = verified_advisory_targets(registry_path=self.path)
        self.assertEqual(
            targets["doc1"][("MEDICAL_NECESSITY", "11111")], False)

    def test_identical_targets_are_idempotent(self):
        from tools.claims_registry import load_events
        self.assertIsNotNone(self._record())
        self.assertIsNone(self._record())
        self.assertEqual(len(load_events(self.path)), 1)

    def test_changed_targets_append_and_latest_wins(self):
        from tools.claims_registry import verified_advisory_targets
        self._record()
        self._record([{"filter_id": "MEDICAL_NECESSITY", "code": "11111",
                       "emit": True, "authority": "revised"}])
        targets = verified_advisory_targets(registry_path=self.path)
        self.assertEqual(
            targets["doc1"][("MEDICAL_NECESSITY", "11111")], True)

    def test_full_claim_record_does_not_supersede(self):
        from tools.claims_registry import (record_adjudicated,
                                           verified_advisory_targets)
        self._record()
        result = {"icd_codes": [], "cpt_codes": [], "hcpcs_codes": [],
                  "final_disposition": "CLEAN", "consistency": {}}
        record_adjudicated("doc1", result, "doc1_results.json",
                           by="coder-llm/test", registry_path=self.path)
        self.assertIn(
            "doc1", verified_advisory_targets(registry_path=self.path))

    def test_malformed_targets_are_dropped(self):
        self.assertIsNone(self._record(
            [{"filter_id": "", "code": "11111", "emit": False},
             {"filter_id": "X", "code": "", "emit": False},
             {"filter_id": "X", "code": "11111", "emit": "yes"}]))


class AdvisoryEmissionGateHelpersTest(unittest.TestCase):
    """The measurement vocabulary the emission-aware replay gate uses."""

    def test_class_advisory_goals_scope_to_class_codes(self):
        from tools.auto_actuate import _class_advisory_goals
        targets = {"doc1": {("advisory_emission",
                             "MEDICAL_NECESSITY|11111"): False,
                            ("advisory_emission",
                             "LCD_COVERAGE|99999"): True},
                   "doc2": {("advisory_emission",
                             "MEDICAL_NECESSITY|11111"): False}}
        goals = _class_advisory_goals(targets, {"doc1"}, {"11111"})
        self.assertEqual(goals, {"doc1": {("advisory_emission",
                                           "MEDICAL_NECESSITY|11111"):
                                          False}})

    def test_class_advisory_goals_empty_when_nothing_covers(self):
        from tools.auto_actuate import _class_advisory_goals
        targets = {"doc1": {("advisory_emission",
                             "MEDICAL_NECESSITY|99999"): False}}
        self.assertEqual(
            _class_advisory_goals(targets, {"doc1"}, {"11111"}), {})

    def test_advisory_signature_reads_only_warn_findings(self):
        from tools.observables import _advisory_signature
        result = {"claim_scrub": {"findings": [
            {"filter_id": "MEDICAL_NECESSITY", "status": "WARN",
             "codes": ["11111", "22222"]},
            {"filter_id": "DOCUMENTATION", "status": "PASS",
             "codes": ["11111"]},
            {"filter_id": "MUE_MAI", "status": "FAIL",
             "codes": ["33333"]},
        ]}}
        self.assertEqual(_advisory_signature(result), {
            "MEDICAL_NECESSITY|11111", "MEDICAL_NECESSITY|22222"})

    def test_advisory_signature_tolerates_enum_statuses(self):
        # producers dump mode="json" now, but a record assembled by a
        # future producer that forgets must still measure correctly —
        # str(Status.WARN) is "Status.WARN", which silently measured
        # every in-memory replay as advisory-free (live, routine_00003)
        from app.compliance.models import Status
        from tools.observables import _advisory_signature
        result = {"claim_scrub": {"findings": [
            {"filter_id": "MEDICAL_NECESSITY", "status": Status.WARN,
             "codes": ["11111"]},
            {"filter_id": "MUE_MAI", "status": Status.FAIL,
             "codes": ["33333"]},
        ]}}
        self.assertEqual(_advisory_signature(result),
                         {"MEDICAL_NECESSITY|11111"})

    def test_observable_sigs_expose_every_observable(self):
        from tools.audit_convergence_loop import _observable_sigs
        result = {"claim_scrub": {"findings": [
            {"filter_id": "MEDICAL_NECESSITY", "status": "WARN",
             "codes": ["11111"]}]}}
        sigs = _observable_sigs(result)
        self.assertIn("advisory_emission", sigs)
        self.assertEqual(sigs["advisory_emission"],
                         frozenset({"MEDICAL_NECESSITY|11111"}))

    def test_suppression_rewrites_warn_to_pass_with_audit_trail(self):
        from app.compliance.engine import _apply_advisory_suppressions
        from app.compliance.models import Finding, Status
        warn = Finding(filter_id="MEDICAL_NECESSITY", status=Status.WARN,
                       codes=["11111"], reason="pathway A required")
        fail = Finding(filter_id="MEDICAL_NECESSITY", status=Status.FAIL,
                       codes=["11111"], reason="hard gate")
        sup = [{"filter_id": "MEDICAL_NECESSITY", "code": "11111",
                "rule_id": "fixture-rule", "authority": "LCD L12345",
                "note": "pathway B documented"}]
        out = _apply_advisory_suppressions([warn, fail], sup,
                                           "MEDICAL_NECESSITY")
        by_status = {f.status for f in out}
        self.assertIn(Status.FAIL, by_status)   # FAIL untouchable
        self.assertNotIn(Status.WARN, by_status)
        passed = [f for f in out if f.status == Status.PASS][0]
        self.assertIn("fixture-rule", passed.reason)
        self.assertIn("pathway A required", passed.reason)
        self.assertEqual(passed.source_rule, "LCD L12345")

    def test_suppression_for_other_filter_or_code_is_inert(self):
        from app.compliance.engine import _apply_advisory_suppressions
        from app.compliance.models import Finding, Status
        warn = Finding(filter_id="MEDICAL_NECESSITY", status=Status.WARN,
                       codes=["11111"], reason="advisory")
        out = _apply_advisory_suppressions(
            [warn], [{"filter_id": "LCD_COVERAGE", "code": "11111"},
                     {"filter_id": "MEDICAL_NECESSITY", "code": "22222"}],
            "MEDICAL_NECESSITY")
        self.assertEqual(out, [warn])


class RebuiltRecordShapeTest(unittest.TestCase):
    """The record contract every measurement reads is the SAVED-FILE
    shape: string statuses, JSON scalars. Measured live (routine_00003):
    _rebuild_run stored `scrub.model_dump()` with Status/DenialRisk enum
    members, so `str(Status.WARN)` ("Status.WARN") never equaled "WARN"
    and the advisory_emission observable measured NOTHING on in-memory
    replays — baseline_resolves closed a freshly-verified must-not-fire
    class as 'already resolved', gate_replay could never credit an
    emission realignment, and replay_scope saw a perpetual emission diff
    against the saved file (an audit/adjudication loop every iteration).
    These tests lock the invariant with the REAL ScrubResult model."""

    def _rebuild(self, findings):
        from app.compliance.models import ScrubResult
        from tools.replay_reconcile import _rebuild_run
        scrub = ScrubResult(document_id="doc1", findings=findings)

        class _Scrubber:
            def scrub(self, payload):
                return scrub

        run = {"icd_codes": [], "cpt_codes": [], "hcpcs_codes": []}
        return _rebuild_run(run, {}, {}, _Scrubber(), "note text")

    def test_rebuilt_scrub_statuses_are_plain_strings(self):
        from app.compliance.models import Finding, Status
        out = self._rebuild([Finding(filter_id="MEDICAL_NECESSITY",
                                     status=Status.WARN, codes=["11111"],
                                     reason="advisory")])
        statuses = [f["status"] for f in out["claim_scrub"]["findings"]]
        self.assertEqual(statuses, ["WARN"])
        self.assertIsInstance(statuses[0], str)
        self.assertNotIsInstance(statuses[0], type(Status.WARN))

    def test_emission_measured_identically_in_memory_and_after_save(self):
        import json as _json

        from app.compliance.models import Finding, Status
        from tools.observables import record_signatures
        out = self._rebuild([Finding(filter_id="MEDICAL_NECESSITY",
                                     status=Status.WARN, codes=["11111"],
                                     reason="advisory")])
        in_memory = record_signatures(out)
        self.assertEqual(in_memory["advisory_emission"],
                         frozenset({"MEDICAL_NECESSITY|11111"}))
        # the round trip through a saved file must measure the same —
        # this equality is exactly replay_scope's rewrite criterion
        saved = _json.loads(_json.dumps(out, default=str))
        self.assertEqual(record_signatures(saved), in_memory)


class RepairFeedbackMissTest(unittest.TestCase):
    """A 'does not land' rejection carries per-document miss diagnostics,
    not a violation block — the repair brief must surface them, or the
    designer's second attempt is a blind retry (observed live on
    routine_00003's advisory-suppression synthesis)."""

    def test_advisory_emission_miss_reaches_the_brief(self):
        from tools.auto_actuate import _repair_feedback
        detail = {"documents": {"doc1": {
            "advisory_emission_miss": {
                "targets": {"advisory_emission|MEDICAL_NECESSITY|11111":
                            "suppress"},
                "candidate_emission_per_run": [
                    {"advisory_emission|MEDICAL_NECESSITY|11111": True}],
            }}}}
        brief = _repair_feedback("candidate does not land", detail)
        self.assertIn("EXACT MISS", brief)
        self.assertIn("MEDICAL_NECESSITY|11111", brief)
        self.assertIn("suppress_scrub_advisory", brief)

    def test_violation_block_still_outranks_misses(self):
        from tools.auto_actuate import _repair_feedback
        detail = {"violation": {"document_id": "doc1", "kind": "registry",
                                "replay_vs_target_diffs": []},
                  "documents": {"doc1": {"advisory_emission_miss": {}}}}
        brief = _repair_feedback("alters a verified claim", detail)
        self.assertIn("EXACT VIOLATION", brief)

    def test_bare_reason_when_no_diagnostics(self):
        from tools.auto_actuate import _repair_feedback
        self.assertEqual(_repair_feedback("declined", {"documents": {}}),
                         "declined")


class TemplateNameClampTest(unittest.TestCase):
    """Hint names come from LLM free text; the module static gate caps
    TEMPLATE_NAME at 41 chars. A 45-char hint name burned a whole design
    attempt live (routine_00003) — the clamp reconciles the two at the
    synthesis entrance."""

    def test_short_names_pass_through(self):
        from tools.auto_actuate import _clamp_template_name
        self.assertEqual(_clamp_template_name("laterality_arbitration"),
                         "laterality_arbitration")

    def test_long_names_lose_trailing_segments_not_midword(self):
        from tools.auto_actuate import _clamp_template_name
        out = _clamp_template_name(
            "scrub_advisory_documented_pathway_suppression")  # 45 chars
        self.assertLessEqual(len(out), 41)
        self.assertEqual(out, "scrub_advisory_documented_pathway")
        import re as _re
        self.assertTrue(_re.fullmatch(r"[a-z][a-z0-9_]{2,40}", out))

    def test_garbage_falls_back_to_generic_stem(self):
        from tools.auto_actuate import _clamp_template_name
        self.assertEqual(_clamp_template_name("123!!"),
                         "synthesized_mechanic")
        self.assertEqual(_clamp_template_name(""), "synthesized_mechanic")

    def test_single_overlong_segment_hard_truncates(self):
        from tools.auto_actuate import _clamp_template_name
        out = _clamp_template_name("a" * 60)
        self.assertEqual(out, "a" * 41)


class AuditClassAnchorTest(unittest.TestCase):
    """An audit-dispute class whose verified targets vanished (registry
    wipe, voided verdict) must park at awaiting_verification instead of
    consuming proposal/synthesis budget — with no target in the registry,
    gate_replay's 'land on the verified target' criterion is unsatisfiable
    by construction (measured live on routine_00003's stale Z79.01
    class)."""

    def setUp(self):
        import tools.auto_actuate as aa
        self.aa = aa
        self.cls = {"kind": "audit_dispute", "code": "Z79.01",
                    "class_key": "audit_dispute|icd_codes|Z79.01",
                    "documents": [{"document_id": "doc1",
                                   "disagreement": {"codes": ["Z79.01"]}}]}
        self._orig = (aa._registry_verified_claims, aa._per_code_targets,
                      aa._advisory_targets)

    def tearDown(self):
        (self.aa._registry_verified_claims, self.aa._per_code_targets,
         self.aa._advisory_targets) = self._orig

    def _stub(self, registry=None, code_targets=None, advisory=None):
        self.aa._registry_verified_claims = lambda: registry or {}
        self.aa._per_code_targets = lambda: code_targets or {}
        self.aa._advisory_targets = lambda: advisory or {}

    def test_unanchored_when_registry_is_empty(self):
        self._stub()
        self.assertFalse(self.aa._audit_class_anchored(self.cls))

    def test_whole_claim_record_anchors(self):
        self._stub(registry={"doc1": ("sig",)})
        self.assertTrue(self.aa._audit_class_anchored(self.cls))

    def test_per_code_target_anchors_only_when_covering_class_codes(self):
        self._stub(code_targets={"doc1": {("icd_codes", "Z79.01"): None}})
        self.assertTrue(self.aa._audit_class_anchored(self.cls))
        self._stub(code_targets={"doc1": {("icd_codes", "L60.3"): None}})
        self.assertFalse(self.aa._audit_class_anchored(self.cls))

    def test_advisory_emission_target_anchors(self):
        self._stub(advisory={"doc1": {("advisory_emission",
                                       "unjustified_zcode|Z79.01"): False}})
        self.assertTrue(self.aa._audit_class_anchored(self.cls))

    def test_targets_on_other_documents_do_not_anchor(self):
        self._stub(registry={"other_doc": ("sig",)},
                   advisory={"other_doc": {("advisory_emission",
                                            "X|Z79.01"): False}})
        self.assertFalse(self.aa._audit_class_anchored(self.cls))


if __name__ == "__main__":
    unittest.main()
