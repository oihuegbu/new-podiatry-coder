"""Tests for the automated actuation pipeline: flip triage clustering,
proposal acceptance gates, and the rule engine's auto-rule dispatch.

The replay gate itself needs the full reference DB + compliance store, so it
is exercised in-container (auto_actuate --dry-run); here we prove the pure
logic: clustering is idempotent and evidence-preserving, the no-hardcode and
structural gates reject what they must, and a defective auto rule can never
crash validation.
"""

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools import flip_triage
from tools.auto_actuate import gate_structural, gate_no_code_literals


def _result(doc, disagreements, runs=3, note_text="Debridement of ulcer."):
    return {
        "document_id": doc,
        "consistency": {"runs": runs, "unanimous": not disagreements,
                        "disagreements": disagreements},
        "rag_context": {"note_full_text": note_text},
    }


def _run_payload(cpt_codes):
    return {"cpt_codes": cpt_codes, "icd_codes": [], "hcpcs_codes": []}


class FlipTriageTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.results = Path(self.tmp.name) / "results"
        (self.results / "consistency_runs").mkdir(parents=True)
        self.queue = Path(self.tmp.name) / "flip_queue.jsonl"

    def tearDown(self):
        self.tmp.cleanup()

    def _write(self, doc, result, runs=None):
        (self.results / f"{doc}_results.json").write_text(json.dumps(result))
        for i, r in enumerate(runs or [], 1):
            (self.results / "consistency_runs" /
             f"{doc}_run{i}.json").write_text(json.dumps(r))

    def test_clusters_same_code_across_documents(self):
        flip = {"array": "hcpcs_codes", "code": "L9999", "kind": "presence",
                "advisory": False, "present_in_runs": 1, "runs": 3}
        self._write("noteA", _result("noteA", [flip]))
        self._write("noteB", _result("noteB", [flip]))
        stats = flip_triage.scan(self.results, self.queue)
        self.assertEqual(stats["total_classes"], 1)
        cls = flip_triage.load_queue(self.queue)[0]
        self.assertEqual(cls["class_key"], "presence|hcpcs_codes|L9999")
        self.assertEqual(
            [d["document_id"] for d in cls["documents"]], ["noteA", "noteB"])
        self.assertEqual(cls["status"], "open")

    def test_advisory_and_unanimous_are_ignored(self):
        advisory = {"array": "snomed_codes", "code": "123", "advisory": True,
                    "kind": "presence", "present_in_runs": 1, "runs": 3}
        self._write("noteA", _result("noteA", [advisory]))
        self._write("noteB", _result("noteB", []))
        stats = flip_triage.scan(self.results, self.queue)
        self.assertEqual(stats["total_classes"], 0)

    def test_scan_is_idempotent_and_preserves_status(self):
        flip = {"array": "cpt_codes", "code": "11111", "kind": "presence",
                "advisory": False, "present_in_runs": 2, "runs": 3}
        self._write("noteA", _result("noteA", [flip]))
        flip_triage.scan(self.results, self.queue)
        flip_triage.set_status("presence|cpt_codes|11111", "actuated",
                               {"rule_id": "x"}, self.queue)
        flip_triage.scan(self.results, self.queue)  # rescan same evidence
        q = flip_triage.load_queue(self.queue)
        self.assertEqual(len(q), 1)
        self.assertEqual(q[0]["status"], "actuated")

    def test_per_run_evidence_and_note_sentences(self):
        flip = {"array": "cpt_codes", "code": "11111", "kind": "presence",
                "advisory": False, "present_in_runs": 1, "runs": 2}
        runs = [
            _run_payload([{"code": "11111", "modifiers": ["LT"],
                           "description": "Debridement of skin"}]),
            _run_payload([]),
        ]
        self._write("noteA", _result(
            "noteA", [flip],
            note_text="Sharp debridement performed today.\nStable."), runs)
        flip_triage.scan(self.results, self.queue)
        d = flip_triage.load_queue(self.queue)[0]["documents"][0]
        self.assertEqual(d["per_run_entry"][0]["modifiers"], ["LT"])
        self.assertIsNone(d["per_run_entry"][1])
        self.assertTrue(any("debridement" in s.lower()
                            for s in d["note_sentences"]))

    def test_actuated_class_reopens_when_flip_recurs_on_new_document(self):
        flip = {"array": "cpt_codes", "code": "11111", "kind": "presence",
                "advisory": False, "present_in_runs": 1, "runs": 3}
        self._write("noteA", _result("noteA", [flip]))
        flip_triage.scan(self.results, self.queue)
        flip_triage.set_status("presence|cpt_codes|11111", "actuated",
                               {"rule_id": "r1"}, self.queue)
        # the same flip now appears on a document the rule never saw
        self._write("noteB", _result("noteB", [flip]))
        stats = flip_triage.scan(self.results, self.queue)
        cls = flip_triage.load_queue(self.queue)[0]
        self.assertEqual(cls["status"], "open")
        self.assertIn("new document", cls["reopened"]["reason"])
        self.assertEqual(stats.get("reopened_classes"), 1)

    def test_actuated_class_reopens_when_flip_persists_after_rerun(self):
        flip = {"array": "cpt_codes", "code": "11111", "kind": "presence",
                "advisory": False, "present_in_runs": 1, "runs": 3}
        self._write("noteA", _result("noteA", [flip]))
        flip_triage.scan(self.results, self.queue)
        flip_triage.set_status("presence|cpt_codes|11111", "actuated",
                               {"rule_id": "r1"}, self.queue)
        # same document REPROCESSED after actuation, flip still there
        r = _result("noteA", [flip])
        r["timestamp"] = "9999-12-31T00:00:00"
        self._write("noteA", r)
        flip_triage.scan(self.results, self.queue)
        cls = flip_triage.load_queue(self.queue)[0]
        self.assertEqual(cls["status"], "open")
        self.assertIn("persisted", cls["reopened"]["reason"])

    def test_actuated_class_stays_closed_on_stale_rescan(self):
        flip = {"array": "cpt_codes", "code": "11111", "kind": "presence",
                "advisory": False, "present_in_runs": 1, "runs": 3}
        r = _result("noteA", [flip])
        r["timestamp"] = "2020-01-01T00:00:00"  # older than the actuation
        self._write("noteA", r)
        flip_triage.scan(self.results, self.queue)
        flip_triage.set_status("presence|cpt_codes|11111", "actuated",
                               {"rule_id": "r1"}, self.queue)
        flip_triage.scan(self.results, self.queue)  # same stale evidence
        cls = flip_triage.load_queue(self.queue)[0]
        self.assertEqual(cls["status"], "actuated")

    def test_em_level_flips_cluster_as_one_class(self):
        flip = {"array": "cpt_codes", "code": "99213/99214",
                "codes": ["99213", "99214"], "kind": "em_level",
                "advisory": False, "runs": 3,
                "by_run": [{"code": "99213"}, {"code": "99214"},
                           {"code": "99214"}]}
        self._write("noteA", _result("noteA", [flip]))
        cls = None
        flip_triage.scan(self.results, self.queue)
        cls = flip_triage.load_queue(self.queue)[0]
        self.assertEqual(cls["class_key"], "em_level|cpt_codes|EM")


class GateTest(unittest.TestCase):
    def _rule(self, **over):
        rule = {
            "id": "test-context-gate", "template": "context_gate",
            "enabled": True, "authority": "CMS NCCI Policy Manual Ch.1",
            "applies_to": {"array": "cpt_codes",
                           "descriptor_prefix": "debridement"},
            "mention_terms": {"source": "descriptor_head_stem"},
            "contexts": [{"label": "planned", "regex": r"\bplan(?:ned)?\b"}],
            "action": {"severity": "WARNING", "category": "context",
                       "message": "{code} suppressed ({contexts})",
                       "recommendation": "review", "denial_risk": "HIGH"},
        }
        rule.update(over)
        return rule

    def test_valid_rule_passes_both_gates(self):
        r = self._rule()
        self.assertEqual(gate_structural(r), "")
        self.assertEqual(gate_no_code_literals(r), "")

    def test_unknown_template_rejected(self):
        self.assertIn("unknown template",
                      gate_structural(self._rule(template="magic")))

    def test_bad_array_rejected(self):
        self.assertIn("array", gate_structural(
            self._rule(applies_to={"array": "snomed_codes"})))

    def test_missing_authority_rejected(self):
        self.assertIn("authority", gate_structural(self._rule(authority="")))

    def test_cpt_literal_rejected_wherever_it_hides(self):
        r = self._rule(contexts=[{"label": "x", "regex": r"\b11720\b"}])
        self.assertIn("11720", gate_no_code_literals(r))

    def test_icd_and_hcpcs_literals_rejected(self):
        self.assertIn("L3260", gate_no_code_literals(
            self._rule(applies_to={"array": "hcpcs_codes",
                                   "code_regex": "L3260"})))
        self.assertIn("E11.621", gate_no_code_literals(
            self._rule(tiers={"E11.621": 1})))

    def test_authority_prose_may_cite_chapters(self):
        # 5-digit strings in the authority citation are prose, not selectors
        r = self._rule(authority="IOM Pub 100-04 section 10012")
        self.assertEqual(gate_no_code_literals(r), "")

    def test_icd_category_prefix_regex_allowed_full_code_rejected(self):
        # '^M77\.' selects a Tabular category structurally — allowed; the
        # descriptor grammar does the real selection. '^M77\.41$' IS the
        # code, escaped dot or not — rejected.
        ok = self._rule(template="documented_diagnosis_completion",
                        applies_to={"array": "icd_codes"},
                        family={"code_regex": r"^M77\."})
        self.assertEqual(gate_no_code_literals(ok), "")
        bad = self._rule(template="documented_diagnosis_completion",
                         applies_to={"array": "icd_codes"},
                         family={"code_regex": r"^M77\.41$"})
        self.assertIn("M77.41", gate_no_code_literals(bad))

    def test_template_array_coherence(self):
        self.assertIn("icd_codes only", gate_structural(
            self._rule(template="documented_diagnosis_completion",
                       applies_to={"array": "cpt_codes"})))
        self.assertIn("CPT/HCPCS-only", gate_structural(
            self._rule(template="documented_service_completion",
                       applies_to={"array": "icd_codes"})))

    def test_display_prose_may_mention_codes_but_selectors_may_not(self):
        # message/recommendation are rendered for humans, never matched
        r = self._rule(action={
            "severity": "WARNING", "category": "x",
            "message": "{code} conflicts with 11720 same-session",
            "recommendation": "Rebill 11720 with modifier 59 if distinct",
            "denial_risk": "HIGH"})
        self.assertEqual(gate_no_code_literals(r), "")
        # the same literal in a selecting field still rejects
        r2 = self._rule(mention_terms={"source": "descriptor_head_stem",
                                       "extra": "11720"})
        self.assertIn("11720", gate_no_code_literals(r2))


class PackAuditTest(unittest.TestCase):
    """audit_pack is the post-deployment bug check: it must flag exactly
    the defects that would make the live pack unsafe."""

    def setUp(self):
        import tools.auto_actuate as aa
        from app.validation import auto_templates as at
        self.aa, self.at = aa, at
        self.tmp = TemporaryDirectory()
        self._old = aa.RULES_PATH
        aa.RULES_PATH = Path(self.tmp.name) / "pack.json"
        self._old_proposals = aa.PROPOSALS_DIR
        aa.PROPOSALS_DIR = Path(self.tmp.name) / "proposals"
        self._old_dir = at.AUTO_TEMPLATES_DIR
        at.AUTO_TEMPLATES_DIR = Path(self.tmp.name) / "auto_templates"

    def tearDown(self):
        self.aa.RULES_PATH = self._old
        self.aa.PROPOSALS_DIR = self._old_proposals
        self.at.AUTO_TEMPLATES_DIR = self._old_dir
        self.tmp.cleanup()

    def _write(self, rules):
        self.aa.RULES_PATH.write_text(
            json.dumps({"version": "t", "rules": rules}))

    def test_clean_pack_passes(self):
        self._write([{"id": "a", "template": "context_gate",
                      "auto_generated": True, "authority": "x",
                      "applies_to": {"array": "cpt_codes"}}])
        self.assertEqual(self.aa.audit_pack(), [])

    def test_duplicate_ids_and_bad_template_flagged(self):
        self._write([
            {"id": "a", "template": "context_gate", "auto_generated": True},
            {"id": "a", "template": "no_such", "auto_generated": True},
        ])
        problems = " ".join(self.aa.audit_pack())
        self.assertIn("duplicate rule id", problems)
        self.assertIn("unknown template", problems)

    def test_code_literal_in_auto_rule_flagged(self):
        self._write([{"id": "a", "template": "context_gate",
                      "auto_generated": True,
                      "contexts": [{"label": "x", "regex": "11720"}]}])
        self.assertIn("11720", " ".join(self.aa.audit_pack()))

    def test_disabled_rollback_corpse_is_exempt(self):
        # a rolled-back synthesized rule stays in the pack disabled, with
        # its template file removed — auditing that corpse rolled back a
        # healthy later acceptance in production; disabled rules never
        # execute and must never fail the audit
        self._write([{"id": "dead", "template": "removed_template",
                      "auto_generated": True, "enabled": False,
                      "contexts": [{"label": "x", "regex": "11720"}]}])
        self.assertEqual(self.aa.audit_pack(), [])

    def test_provenance_audit_trail_is_exempt(self):
        # provenance records flip-class keys and replay detail BY DESIGN —
        # it selects nothing. The first live pack audit flagged it and
        # rolled back a healthy synthesized template; never again.
        self._write([{"id": "a", "template": "context_gate",
                      "auto_generated": True,
                      "provenance": {
                          "flip_class": "attributes|cpt_codes|28110",
                          "replay": {"documents": {"note": {"changed": 1}}},
                      }}])
        self.assertEqual(self.aa.audit_pack(), [])

    def test_tampered_template_file_flagged_by_pack_audit(self):
        # a template edited AFTER deployment (outside every gate) must be
        # caught the next time the pack changes — hardcoded codes never
        # ride shotgun in executor code
        self._write([])
        self.at.AUTO_TEMPLATES_DIR.mkdir()
        (self.at.AUTO_TEMPLATES_DIR / "t.py").write_text(
            _GOOD_TEMPLATE + "\nSPECIAL = 'always add 11720'\n")
        problems = " ".join(self.aa.audit_pack())
        self.assertIn("template t.py", problems)
        self.assertIn("11720", problems)

    def test_disable_rule_creates_inert_retirement_proposal(self):
        self._write([{"id": "a", "template": "context_gate",
                      "auto_generated": True, "enabled": True}])
        self.aa._disable_rule("a")
        pack = json.loads(self.aa.RULES_PATH.read_text())
        self.assertTrue(pack["rules"][0]["enabled"])
        drafts = list(self.aa.PROPOSALS_DIR.glob("*.json"))
        self.assertEqual(len(drafts), 1)
        proposal = json.loads(drafts[0].read_text())
        self.assertEqual(proposal["status"], "draft")
        self.assertEqual(proposal["rule"]["target_rule_id"], "a")


class PerCodeTargetGateTest(unittest.TestCase):
    """Scoped realignment: per-code verified targets from partial
    adjudications judge exactly the class's covered codes — each covered
    code's replay row must equal its target row (None = absent), and
    targets covering none of the class codes yield no judgment."""

    def setUp(self):
        from tools.auto_actuate import (Replayer, _lands_on_code_targets,
                                        _per_code_targets)
        self.sig = Replayer.signature
        self.lands = _lands_on_code_targets
        self.convert = _per_code_targets

    def _sig_with(self, cpt):
        return self.sig([], cpt, [])

    def test_landing_on_present_target(self):
        targets = {("cpt_codes", "27654"): ("27654", ("RT",), "1")}
        good = self._sig_with([{"code": "27654", "modifiers": ["RT"],
                                "units": 1}])
        self.assertTrue(self.lands([good, good], targets, {"27654"}))

    def test_wrong_row_fails(self):
        targets = {("cpt_codes", "27654"): ("27654", ("RT",), "1")}
        bad = self._sig_with([{"code": "27654", "modifiers": [],
                               "units": 1}])
        self.assertFalse(self.lands([bad], targets, {"27654"}))

    def test_missing_line_fails_a_present_target(self):
        targets = {("cpt_codes", "27654"): ("27654", ("RT",), "1")}
        self.assertFalse(self.lands([self._sig_with([])],
                                    targets, {"27654"}))

    def test_absent_target_requires_absence(self):
        targets = {("hcpcs_codes", "A4570"): None}
        gone = self.sig([], [], [])
        still = self.sig([], [], [{"code": "A4570", "units": 1}])
        self.assertTrue(self.lands([gone], targets, {"A4570"}))
        self.assertFalse(self.lands([still], targets, {"A4570"}))

    def test_targets_covering_no_class_code_yield_no_judgment(self):
        targets = {("cpt_codes", "27654"): ("27654", ("RT",), "1")}
        self.assertIsNone(self.lands([self._sig_with([])],
                                     targets, {"29515"}))

    def test_projection_pre_applies_verified_rows(self):
        # an include verdict may exist in NO stored run (donor-
        # materialized) — the trial input must carry it anyway, and an
        # absent-target row must be stripped
        from tools.auto_actuate import _project_code_targets

        class _Rep:  # descriptor lookup only
            pass

        payloads = [{"cpt_codes": [{"code": "29515", "modifiers": ["RT"],
                                    "units": 1}],
                     "hcpcs_codes": [{"code": "A4570", "units": 1}],
                     "icd_codes": []}]
        targets = {("cpt_codes", "27654"): ("27654", ("RT",), "1"),
                   ("hcpcs_codes", "A4570"): None}
        with mock.patch("tools.auto_actuate._authoritative_evidence",
                        return_value=[{"descriptor": "secondary repair"}]):
            out = _project_code_targets(payloads, targets,
                                        {"27654", "A4570"}, _Rep())
        self.assertIsNot(out, payloads)
        cpt = {e["code"]: e for e in out[0]["cpt_codes"]}
        self.assertIn("27654", cpt)
        self.assertEqual(cpt["27654"]["modifiers"], ["RT"])
        self.assertEqual(cpt["27654"]["description"], "secondary repair")
        self.assertIn("29515", cpt)  # untouched line survives
        self.assertEqual(out[0]["hcpcs_codes"], [])  # absent target strips
        # source payloads never mutated
        self.assertEqual(len(payloads[0]["cpt_codes"]), 1)

    def test_projection_without_coverage_is_identity(self):
        from tools.auto_actuate import _project_code_targets
        payloads = [{"cpt_codes": []}]
        out = _project_code_targets(
            payloads, {("cpt_codes", "27654"): None}, {"99999"}, None)
        self.assertIs(out, payloads)

    def test_registry_rows_convert_to_signature_rows(self):
        with mock.patch("tools.claims_registry.verified_code_targets",
                        return_value={"doc1": {
                            ("cpt_codes", "27654"):
                                {"code": "27654", "modifiers": ["RT"],
                                 "units": "1"},
                            ("icd_codes", "M76.61"):
                                {"code": "M76.61", "modifiers": [],
                                 "units": "", "type": "primary"},
                            ("hcpcs_codes", "A4570"): None,
                        }}):
            out = self.convert()
        self.assertEqual(out["doc1"][("cpt_codes", "27654")],
                         ("27654", ("RT",), "1"))
        self.assertEqual(out["doc1"][("icd_codes", "M76.61")],
                         ("M76.61", (), "", "primary"))
        self.assertIsNone(out["doc1"][("hcpcs_codes", "A4570")])


class AuditDisputeOpensOnCodeTargetTest(unittest.TestCase):
    """An audit-dispute class whose (array, code) is covered by a per-code
    verified target opens for actuation — the scoped verification a
    partial adjudication records is enough; the note no longer needs the
    full-claim registry record it may never get."""

    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.results = Path(self.tmp.name) / "results"
        (self.results / "consistency_runs").mkdir(parents=True)
        self.queue = Path(self.tmp.name) / "flip_queue.jsonl"

    def tearDown(self):
        self.tmp.cleanup()

    def _disputed_result(self, doc):
        r = _result(doc, [])
        r["clinical_audit"] = {
            "verdict": "disputed",
            "claim_findings": [
                {"kind": "missing_code", "array": "cpt_codes",
                 "code": "27654", "materiality": "billing_material",
                 "finding": "omitted", "authority": "a",
                 "note_evidence": "e"}],
            "items": [],
        }
        (self.results / f"{doc}_results.json").write_text(json.dumps(r))

    def _scan(self, verified=frozenset(), targets=None):
        with mock.patch.object(flip_triage, "_verified_docs",
                               return_value=set(verified)), \
             mock.patch.object(flip_triage, "_verified_code_targets",
                               return_value=targets or {}):
            flip_triage.scan(self.results, self.queue)
        return {c["class_key"]: c
                for c in flip_triage.load_queue(self.queue)}

    def test_stays_awaiting_without_any_verification(self):
        self._disputed_result("noteA")
        classes = self._scan()
        cls = classes["audit_dispute|cpt_codes|27654"]
        self.assertEqual(cls["status"], "awaiting_verification")

    def test_opens_when_a_per_code_target_covers_the_class(self):
        self._disputed_result("noteA")
        classes = self._scan(targets={"noteA": {
            ("cpt_codes", "27654"): {"code": "27654",
                                     "modifiers": ["RT"], "units": "1"}}})
        cls = classes["audit_dispute|cpt_codes|27654"]
        self.assertEqual(cls["status"], "open")

    def test_target_on_a_different_code_does_not_open_it(self):
        self._disputed_result("noteA")
        classes = self._scan(targets={"noteA": {
            ("hcpcs_codes", "A4570"): None}})
        cls = classes["audit_dispute|cpt_codes|27654"]
        self.assertEqual(cls["status"], "awaiting_verification")


class ImplicatedRulesTest(unittest.TestCase):
    """Amendment candidates: exactly the ENABLED auto-generated pack rules
    whose action.category matches a material correction recorded on the
    class's own codes — the deployed rules that acted on the disputed
    content, nothing else."""

    def setUp(self):
        import tools.auto_actuate as aa
        self.aa = aa
        self.tmp = TemporaryDirectory()
        self._old = aa.RULES_PATH
        aa.RULES_PATH = Path(self.tmp.name) / "pack.json"
        self._old_proposals = aa.PROPOSALS_DIR
        aa.PROPOSALS_DIR = Path(self.tmp.name) / "proposals"
        self.results = Path(self.tmp.name) / "results"
        (self.results / "consistency_runs").mkdir(parents=True)

    def tearDown(self):
        self.aa.RULES_PATH = self._old
        self.aa.PROPOSALS_DIR = self._old_proposals
        self.tmp.cleanup()

    def test_selects_enabled_matching_rules_only(self):
        self.aa.RULES_PATH.write_text(json.dumps({"rules": [
            {"id": "match-live", "auto_generated": True, "enabled": True,
             "action": {"category": "undocumented_procedure_indication"}},
            {"id": "match-dead", "auto_generated": True, "enabled": False,
             "action": {"category": "undocumented_procedure_indication"}},
            {"id": "other-cat", "auto_generated": True, "enabled": True,
             "action": {"category": "drug_dose_undocumented"}},
            {"id": "hand-written", "enabled": True,
             "action": {"category": "undocumented_procedure_indication"}},
        ]}))
        (self.results / "noteA_results.json").write_text(json.dumps({
            "document_id": "noteA",
            "material_corrections": [
                {"category": "undocumented_procedure_indication",
                 "code": "27654", "action": "auto_correction",
                 "interpretive": True, "message": "removed"}],
        }))
        cls = {"kind": "audit_dispute", "array": "cpt_codes",
               "code": "27654",
               "documents": [{"document_id": "noteA",
                              "disagreement": {"codes": ["27654"]}}]}
        rules = self.aa._implicated_rules(cls, self.results)
        self.assertEqual([r["id"] for r in rules], ["match-live"])

    def test_no_correction_on_class_codes_means_no_candidates(self):
        self.aa.RULES_PATH.write_text(json.dumps({"rules": [
            {"id": "match-live", "auto_generated": True, "enabled": True,
             "action": {"category": "undocumented_procedure_indication"}},
        ]}))
        (self.results / "noteA_results.json").write_text(json.dumps({
            "document_id": "noteA",
            "material_corrections": [
                {"category": "undocumented_procedure_indication",
                 "code": "99999", "action": "auto_correction",
                 "interpretive": True, "message": "removed"}],
        }))
        cls = {"kind": "audit_dispute", "array": "cpt_codes",
               "code": "27654",
               "documents": [{"document_id": "noteA",
                              "disagreement": {"codes": ["27654"]}}]}
        self.assertEqual(self.aa._implicated_rules(cls, self.results), [])


class AmendDisableHelpersTest(unittest.TestCase):
    """Supersession bookkeeping: a disabled rule names why and by what;
    a rollback re-enable restores it cleanly."""

    def setUp(self):
        import tools.auto_actuate as aa
        self.aa = aa
        self.tmp = TemporaryDirectory()
        self._old = aa.RULES_PATH
        aa.RULES_PATH = Path(self.tmp.name) / "pack.json"
        self._old_proposals = aa.PROPOSALS_DIR
        aa.PROPOSALS_DIR = Path(self.tmp.name) / "proposals"
        aa.RULES_PATH.write_text(json.dumps({"rules": [
            {"id": "old-rule", "auto_generated": True, "enabled": True,
             "template": "context_gate"}]}))

    def tearDown(self):
        self.aa.RULES_PATH = self._old
        self.aa.PROPOSALS_DIR = self._old_proposals
        self.tmp.cleanup()

    def test_disable_proposes_reason_and_successor_without_live_change(self):
        self.aa._disable_rule("old-rule",
                              reason="superseded by amendment",
                              superseded_by="old-rule-r2")
        r = json.loads(self.aa.RULES_PATH.read_text())["rules"][0]
        self.assertTrue(r["enabled"])
        draft = json.loads(next(self.aa.PROPOSALS_DIR.glob("*.json")).read_text())
        self.assertEqual(draft["rule"]["reason"], "superseded by amendment")
        self.assertEqual(draft["rule"]["superseded_by"], "old-rule-r2")

    def test_reenable_restores_the_rule(self):
        self.aa._disable_rule("old-rule", reason="x",
                              superseded_by="y")
        self.aa._reenable_rule("old-rule")
        r = json.loads(self.aa.RULES_PATH.read_text())["rules"][0]
        self.assertTrue(r["enabled"])
        self.assertNotIn("disabled_reason", r.get("provenance", {}))
        self.assertNotIn("superseded_by", r.get("provenance", {}))


class RegistryRealignmentTest(unittest.TestCase):
    """Directional registry protection: a candidate whose replays all land
    byte-identical on the verified claim is agreement with ground truth;
    anything else that moves a verified replay stays rejected."""

    def setUp(self):
        from tools.auto_actuate import Replayer, _realigns
        self.sig, self._realigns = Replayer.signature, _realigns
        # a registry claim as stored (slimmed billable fields)
        self.goal = self.sig(
            [{"code": "L97.821", "type": "secondary",
              "description": "ulcer"},
             {"code": "I87.2", "type": "primary"}],
            [{"code": "97597", "modifiers": ["LT"], "units": 1}],
            [],
        )

    def _run_sig(self, icd_types=("secondary", "primary")):
        # the same claim as a replayed run's arrays (extra non-billing
        # fields must not matter to the signature)
        return self.sig(
            [{"code": "L97.821", "type": icd_types[0],
              "description": "ulcer", "rationale": "x"},
             {"code": "I87.2", "type": icd_types[1], "needs_review": True}],
            [{"code": "97597", "modifiers": ["LT"], "units": 1,
              "dx_pointers": [1]}],
            [],
        )

    def test_replay_landing_on_verified_claim_realigns(self):
        self.assertTrue(self._realigns(
            [self._run_sig(), self._run_sig()], self.goal))

    def test_any_other_movement_still_rejected(self):
        # one run flips the primary/secondary typing -> NOT a realignment
        swapped = self._run_sig(icd_types=("primary", "secondary"))
        self.assertFalse(self._realigns(
            [self._run_sig(), swapped], self.goal))
        self.assertFalse(self._realigns([], self.goal))


class AutoRuleDispatchTest(unittest.TestCase):
    """A defective auto rule degrades to a skipped rule, never a crash, and
    dispatch honors the enabled/auto_generated flags."""

    def _engine(self, rules):
        import app.validation.rule_engine as re_mod
        from app.validation.validator import CodingValidator

        class _V(CodingValidator):
            """Validator stand-in: real language helpers (tokenizer,
            negation scrub, clinical note view — the engine's evidence
            surface), no reference data, silent issue reporter."""

            def __init__(self):
                self.db = None
                self.store = None
                self.issues = []
                self._non_billable_codes_to_suppress = set()
                self._bundled_codes_to_suppress = set()

            def _add(self, *a, **k):
                pass

        self.tmp = TemporaryDirectory()
        pack = Path(self.tmp.name) / "pack.json"
        pack.write_text(json.dumps({"version": "test", "rules": rules}))
        self._old = re_mod.RULES_FILE
        re_mod.RULES_FILE = pack
        re_mod.load_rule_pack.cache_clear()
        self.re_mod = re_mod
        return re_mod.RuleEngine(_V())

    def tearDown(self):
        if getattr(self, "re_mod", None):
            self.re_mod.RULES_FILE = self._old
            self.re_mod.load_rule_pack.cache_clear()
            self.tmp.cleanup()

    def test_malformed_auto_rule_is_swallowed(self):
        eng = self._engine([{
            "id": "broken", "template": "context_gate", "enabled": True,
            "auto_generated": True,
            # missing every field context_gate needs -> raises inside,
            # must be caught by the dispatch guard
        }])
        eng.run_auto_rules([], [{"code": "97597"}], [], {}, "note text", "")

    def test_documented_service_completion_adds_and_suppresses(self):
        rule = {
            "id": "unna-boot-completion", "enabled": True,
            "auto_generated": True,
            "template": "documented_service_completion",
            "authority": "test",
            "applies_to": {"array": "hcpcs_codes"},
            "family": {"descriptor_requires_any": ["unna boot"]},
            "evidence": {"service_stems": ["unna boot"],
                         "exclusions": [{"label": "planned",
                                         "regex": r"\bwill apply\b"}],
                         "min_descriptor_tokens": 1,
                         "scrub_negation": False},
            "action": {"severity": "WARNING", "category": "svc",
                       "message_added": "{code} added ({desc})",
                       "message_undocumented": "{code} undocumented",
                       "rationale_added": "documented",
                       "review_reason_added": "verify {code}",
                       "recommendation": "review", "denial_risk": "MEDIUM"},
        }
        eng = self._engine([rule])

        class _DB:
            hcpcs = {"A6456": {"description": "Unna boot application zinc"},
                     "A9999": {"description": "Misc supply"}}
        eng.v.db = _DB()
        eng.v._tokens = lambda s: set(
            w for w in s.lower().split() if w.isalpha())
        eng.v._stem = lambda t: t
        eng.v._DESC_STOPWORDS = set()

        # documented -> the matching family member is added
        hcpcs = []
        eng.documented_service_completion(
            "unna-boot-completion", hcpcs,
            "Unna boot applied with zinc wrap today.")
        self.assertEqual([e["code"] for e in hcpcs], ["A6456"])

        # only excluded-context mention -> billed member suppressed
        eng.v._non_billable_codes_to_suppress = set()
        hcpcs = [{"code": "A6456"}]
        eng.documented_service_completion(
            "unna-boot-completion", hcpcs,
            "We will apply an unna boot at the next visit.")
        self.assertIn("A6456", eng.v._non_billable_codes_to_suppress)

    def _dx_rule(self, **over):
        rule = {
            "id": "metatarsalgia-completion", "enabled": True,
            "auto_generated": True,
            "template": "documented_diagnosis_completion",
            "authority": "ICD-10-CM Official Guidelines I.B.4",
            "applies_to": {"array": "icd_codes"},
            "family": {"descriptor_requires_any": ["metatarsalgia"]},
            "evidence": {"condition_stems": ["metatarsalgia"],
                         "exclusions": [{"label": "history",
                                         "regex": r"\bhistory of\b"}],
                         "min_descriptor_tokens": 1,
                         "scrub_negation": False},
            "action": {"severity": "WARNING", "category": "dx",
                       "message_added": "{code} added ({desc})",
                       "message_undocumented": "{code} undocumented",
                       "rationale_added": "documented",
                       "review_reason_added": "verify {code}",
                       "review_reason_demoted": "undocumented {code}",
                       "recommendation": "review", "denial_risk": "MEDIUM"},
        }
        rule.update(over)
        return rule

    def _dx_engine(self, rule):
        eng = self._engine([rule])

        class _DB:
            icd10 = {"M7740": {"description":
                               "Metatarsalgia, unspecified foot"},
                     "M2540": {"description":
                               "Effusion, unspecified joint"}}
        eng.v.db = _DB()
        eng.v._tokens = lambda s: set(
            w.strip(",") for w in s.lower().split() if w.strip(",").isalpha())
        eng.v._stem = lambda t: t
        eng.v._DESC_STOPWORDS = {"unspecified"}
        return eng

    def test_documented_diagnosis_completion_adds_secondary(self):
        eng = self._dx_engine(self._dx_rule())
        icd = [{"code": "M21.6X1", "type": "primary"}]
        eng.documented_diagnosis_completion(
            "metatarsalgia-completion", icd, {},
            "Metatarsalgia of the left foot with forefoot overload.")
        added = icd[-1]
        self.assertEqual(added["code"], "M77.40")  # re-dotted from the DB key
        self.assertEqual(added["type"], "secondary")  # primary exists
        self.assertTrue(added["needs_review"])

    def test_documented_diagnosis_completion_demotes_undocumented(self):
        eng = self._dx_engine(self._dx_rule())
        icd = [{"code": "M21.6X1", "type": "primary"},
               {"code": "M77.40", "type": "secondary"}]
        coding_result = {}
        eng.documented_diagnosis_completion(
            "metatarsalgia-completion", icd, coding_result,
            "History of metatarsalgia. Today: bunion evaluation.")
        self.assertEqual([e["code"] for e in icd], ["M21.6X1"])
        self.assertEqual(
            [e["code"] for e in coding_result["supporting_conditions"]],
            ["M77.40"])

    def test_documented_diagnosis_completion_never_demotes_primary(self):
        eng = self._dx_engine(self._dx_rule())
        icd = [{"code": "M77.40", "type": "primary"}]
        eng.documented_diagnosis_completion(
            "metatarsalgia-completion", icd, {},
            "History of metatarsalgia. Today: bunion evaluation.")
        self.assertEqual([e["code"] for e in icd], ["M77.40"])

    def test_unknown_template_and_flags_respected(self):
        calls = []
        eng = self._engine([
            {"id": "u", "template": "no_such_template", "enabled": True,
             "auto_generated": True},
            {"id": "off", "template": "context_gate", "enabled": False,
             "auto_generated": True},
            {"id": "manual", "template": "context_gate", "enabled": True},
        ])
        eng.context_gate = lambda rid, *a, **k: calls.append(rid)
        eng.run_auto_rules([], [], [], {}, "note", "")
        self.assertEqual(calls, [])  # disabled + manual rules never dispatch


_GOOD_TEMPLATE = '''\
TEMPLATE_NAME = "sorted_marker"
SCHEMA_DOC = """Test mechanic. Rule fields: action.category (str)."""


def execute(engine, rule, icd, cpt, hcpcs, coding_result,
            note_full_text, note_assessment_text):
    act = rule.get("action") or {}
    for e in sorted(cpt, key=lambda x: str(x.get("code") or "")):
        engine.v._add("INFO", e.get("code", ""),
                      act.get("category", "auto"), "marker", "none",
                      clause="test_marker")
'''


class AutoTemplateSafetyTest(unittest.TestCase):
    """The static gate for LLM-authored template code: everything outside
    the tiny approved surface is a rejection, and a clean module passes."""

    def setUp(self):
        from app.validation import auto_templates as at
        self.at = at

    def _problems(self, src):
        return " | ".join(self.at.validate_template_source(src))

    def test_clean_template_passes(self):
        self.assertEqual(self.at.validate_template_source(_GOOD_TEMPLATE),
                         [])

    def test_forbidden_imports_and_names(self):
        base = ('TEMPLATE_NAME = "t_x"\nSCHEMA_DOC = "d"\n'
                "def execute(engine, rule, icd, cpt, hcpcs, coding_result,"
                " note_full_text, note_assessment_text):\n")
        self.assertIn("import 'os' not allowed",
                      self._problems("import os\n" + base + "    pass\n"))
        self.assertIn("forbidden name 'open'",
                      self._problems(base + "    open('/etc/passwd')\n"))
        self.assertIn("forbidden name 'eval'",
                      self._problems(base + "    eval('1')\n"))
        self.assertIn("dunder attribute",
                      self._problems(base + "    rule.__class__\n"))
        self.assertIn("while loops are forbidden",
                      self._problems(base + "    while True:\n        "
                                            "pass\n"))
        self.assertIn("class definitions are forbidden",
                      self._problems(base + "    pass\nclass X:\n"
                                            "    pass\n"))

    def test_recursion_rejected(self):
        src = ('TEMPLATE_NAME = "t_rec"\nSCHEMA_DOC = "d"\n'
               "def helper(n):\n    return helper(n)\n"
               "def execute(engine, rule, icd, cpt, hcpcs, coding_result,"
               " note_full_text, note_assessment_text):\n"
               "    helper(1)\n")
        self.assertIn("recursion cycle", self._problems(src))

    def test_code_literal_in_source_rejected(self):
        src = _GOOD_TEMPLATE + "\nSPECIAL = 'code 11720 is bundled'\n"
        self.assertIn("literal medical code", self._problems(src))
        # escaped-dot regex form of an ICD literal is caught too
        src = _GOOD_TEMPLATE.replace(
            "marker", "pattern M77\\\\.41 here")
        self.assertIn("literal medical code", self._problems(src))

    def test_code_literal_assembled_from_fragments_rejected(self):
        # a code the raw-source scan can't see as one token must still be
        # caught: explicit concatenation, implicit adjacency, f-strings
        for expr in ("'117' + '20'", "'117' '20'", "f'11720'",
                     "'M77' + '.41'"):
            src = _GOOD_TEMPLATE + f"\nSPECIAL = {expr}\n"
            self.assertIn("literal medical code", self._problems(src),
                          msg=f"evaded via {expr}")
        # dynamic f-string holes break adjacency — no false positive
        src = _GOOD_TEMPLATE + "\nOK = f'117{len([])}20'\n"
        self.assertEqual(self.at.validate_template_source(src), [])

    def test_contract_violations_rejected(self):
        self.assertIn("missing top-level TEMPLATE_NAME",
                      self._problems("SCHEMA_DOC = 'd'\n"
                                     "def execute(engine, rule, icd, cpt, "
                                     "hcpcs, coding_result, note_full_text,"
                                     " note_assessment_text):\n    pass\n"))
        self.assertIn("signature must be exactly",
                      self._problems('TEMPLATE_NAME = "t_sig"\n'
                                     "SCHEMA_DOC = 'd'\n"
                                     "def execute(engine, rule):\n"
                                     "    pass\n"))


class AutoTemplateLoadDispatchTest(unittest.TestCase):
    """A gated template module loads, joins the vocabulary, and executes
    through the generic dispatch; a defective file degrades to skipped."""

    def setUp(self):
        from app.validation import auto_templates as at
        self.at = at
        self.tmp = TemporaryDirectory()
        self._old_dir = at.AUTO_TEMPLATES_DIR
        at.AUTO_TEMPLATES_DIR = Path(self.tmp.name) / "auto_templates"
        at.AUTO_TEMPLATES_DIR.mkdir()
        at._cache.clear()

    def tearDown(self):
        self.at.AUTO_TEMPLATES_DIR = self._old_dir
        self.at._cache.clear()
        if getattr(self, "re_mod", None):
            self.re_mod.RULES_FILE = self._old_rules
            self.re_mod.load_rule_pack.cache_clear()
            self.tmp2.cleanup()
        self.tmp.cleanup()

    def _engine(self, rules):
        import app.validation.rule_engine as re_mod

        class _V:
            _non_billable_codes_to_suppress = set()
            db = None

            def __init__(self):
                self.added = []

            def _add(self, *a, **k):
                self.added.append(a)

        self.tmp2 = TemporaryDirectory()
        pack = Path(self.tmp2.name) / "pack.json"
        pack.write_text(json.dumps({"version": "test", "rules": rules}))
        self._old_rules = re_mod.RULES_FILE
        re_mod.RULES_FILE = pack
        re_mod.load_rule_pack.cache_clear()
        self.re_mod = re_mod
        return re_mod.RuleEngine(_V())

    def test_loader_accepts_good_and_skips_bad(self):
        (self.at.AUTO_TEMPLATES_DIR / "good.py").write_text(_GOOD_TEMPLATE)
        (self.at.AUTO_TEMPLATES_DIR / "bad.py").write_text(
            "import os\nTEMPLATE_NAME = 't_bad'\nSCHEMA_DOC = 'd'\n")
        loaded = self.at.load_auto_templates()
        self.assertEqual(sorted(loaded), ["sorted_marker"])
        self.assertIn("Test mechanic", loaded["sorted_marker"]["schema_doc"])

    def test_dispatch_executes_installed_template(self):
        (self.at.AUTO_TEMPLATES_DIR / "m.py").write_text(_GOOD_TEMPLATE)
        eng = self._engine([{
            "id": "m-rule", "template": "sorted_marker", "enabled": True,
            "auto_generated": True, "authority": "test",
            "applies_to": {"array": "cpt_codes"},
            "action": {"category": "test_cat"}}])
        eng.run_auto_rules([], [{"code": "99213"}], [], {}, "note", "")
        self.assertEqual(len(eng.v.added), 1)
        self.assertEqual(eng.v.added[0][2], "test_cat")

    def test_vocabulary_and_prompt_grow_with_installed_template(self):
        import tools.auto_actuate as aa
        self.assertNotIn("sorted_marker", aa.all_templates())
        (self.at.AUTO_TEMPLATES_DIR / "m.py").write_text(_GOOD_TEMPLATE)
        self.assertIn("sorted_marker", aa.all_templates())
        prompt = aa._system_prompt()
        self.assertIn("sorted_marker", prompt)
        self.assertIn("Test mechanic", prompt)
        # constraints stay AFTER the injected schema docs
        self.assertLess(prompt.index("sorted_marker"),
                        prompt.index("HARD CONSTRAINTS"))

    def test_template_name_of(self):
        self.assertEqual(self.at.template_name_of(_GOOD_TEMPLATE),
                         "sorted_marker")
        self.assertEqual(self.at.template_name_of("not python ::"), "")

    def test_permitted_import_re_works_in_sandbox(self):
        src = (
            "import re\n"
            'TEMPLATE_NAME = "regex_marker"\n'
            'SCHEMA_DOC = "d"\n'
            "def execute(engine, rule, icd, cpt, hcpcs, coding_result,\n"
            "            note_full_text, note_assessment_text):\n"
            "    if re.search(r'ulcer', note_full_text or ''):\n"
            "        engine.v._add('INFO', 'X', 'rx', 'm', 'r')\n")
        self.assertEqual(self.at.validate_template_source(src), [])
        (self.at.AUTO_TEMPLATES_DIR / "r.py").write_text(src)
        loaded = self.at.load_auto_templates()
        self.assertIn("regex_marker", loaded)

        class _V:
            added = []

            def _add(self, *a, **k):
                self.added.append(a)

        class _Eng:
            v = _V()
        loaded["regex_marker"]["execute"](
            _Eng(), {}, [], [], [], {}, "plantar ulcer noted", "")
        self.assertEqual(len(_V.added), 1)


class SightedSynthesisTest(unittest.TestCase):
    """The ground-truth-as-input upgrades: verified claims ride in
    dossiers as design targets, replay rejections carry row-level diffs,
    and the repair brief spells both out."""

    def setUp(self):
        import tools.auto_actuate as aa
        self.aa = aa
        self.sig = aa.Replayer.signature

    def _mk(self, cpt_mods):
        return self.sig(
            [{"code": "I87.2", "type": "primary"}],
            [{"code": "17110", "modifiers": cpt_mods}],
            [{"code": "J7354"}])

    def test_sig_view_spells_out_the_claim(self):
        view = self.aa._sig_view(self._mk(["RT"]))
        self.assertEqual(view["cpt_codes"],
                         [{"code": "17110", "modifiers": ["RT"],
                           "units": ""}])
        self.assertEqual(view["icd_codes"][0]["type"], "primary")
        self.assertEqual(view["hcpcs_codes"][0]["code"], "J7354")

    def test_sig_diff_isolates_the_divergent_rows(self):
        goal = self._mk(["RT"])
        produced = self._mk(["59", "RT"])
        diff = self.aa._sig_diff([produced], goal)
        added = diff["replay_vs_target_diffs"][0][
            "rows_your_replay_added_or_changed"]
        lost = diff["replay_vs_target_diffs"][0][
            "rows_the_target_requires_but_replay_lost"]
        self.assertEqual(added, [["17110", ["59", "RT"], ""]])
        self.assertEqual(lost, [["17110", ["RT"], ""]])
        # untouched rows never appear in the diff
        flat = json.dumps(diff["replay_vs_target_diffs"])
        self.assertNotIn("J7354", flat)
        self.assertNotIn("I87.2", flat)
        # the full target claim is included for realignment
        self.assertEqual(diff["target_claim"]["cpt_codes"][0]["modifiers"],
                         ["RT"])

    def test_sig_diff_empty_when_replays_land_on_target(self):
        goal = self._mk(["RT"])
        diff = self.aa._sig_diff([goal, goal], goal)
        self.assertEqual(diff["replay_vs_target_diffs"], [])

    def test_repair_feedback_composes_reason_and_diff(self):
        detail = {"violation": dict(
            self.aa._sig_diff([self._mk(["59", "RT"])], self._mk(["RT"])),
            document_id="doc_x", kind="registry")}
        fb = self.aa._repair_feedback("alters a registry-VERIFIED claim",
                                      detail)
        self.assertIn("alters a registry-VERIFIED claim", fb)
        self.assertIn("doc_x", fb)
        self.assertIn("registry-VERIFIED claim", fb)
        self.assertIn("17110", fb)
        # without a structured violation the reason passes through bare
        self.assertEqual(self.aa._repair_feedback("inert", {}), "inert")
        self.assertEqual(self.aa._repair_feedback("inert", None), "inert")

    def test_dossier_carries_verified_claim_as_design_target(self):
        aa = self.aa
        cls = {"class_key": "attributes|cpt_codes|17110",
               "kind": "attributes", "array": "cpt_codes",
               "code": "17110",
               "documents": [{"document_id": "doc_verified",
                              "disagreement": {"codes": ["17110"]},
                              "per_run_entry": []}]}

        class _Boom:
            def __getattr__(self, k):
                raise RuntimeError("no lookups in this test")

        class _Rep:
            db = _Boom()
            store = _Boom()

        old = aa._registry_verified_claims
        aa._registry_verified_claims = lambda: {
            "doc_verified": self._mk(["RT"])}
        try:
            with TemporaryDirectory() as tmp:
                d = aa.build_dossier(cls, _Rep(), Path(tmp))
        finally:
            aa._registry_verified_claims = old
        doc = d["documents"][0]
        target = doc["registry_verified_claim"]
        self.assertEqual(target["cpt_codes"][0]["modifiers"], ["RT"])
        self.assertIn("ground truth", target["constraint"])
        # the dossier stays JSON-serializable (it goes into prompts)
        json.dumps(d)

    def test_prompts_carry_the_precision_contract(self):
        aa = self.aa
        self.assertIn("SINGLE-AXIS MUTATION", aa._DESIGN_SYSTEM_PROMPT)
        self.assertIn("CODE-CLASS FACTS FROM REFERENCE DATA ONLY",
                      aa._DESIGN_SYSTEM_PROMPT)
        self.assertIn("registry_verified_claim", aa._DESIGN_SYSTEM_PROMPT)
        self.assertIn("ncci_pair", aa._DESIGN_SYSTEM_PROMPT)
        self.assertIn("REGISTRY GROUND TRUTH", aa._SYSTEM_PROMPT)
        self.assertIn("registry_verified_claim", aa._SYSTEM_PROMPT)

    def test_design_prompt_documents_laterality_authority(self):
        # An anatomic-modifier mechanic must be designable against the
        # authoritative CMS bilateral-surgery indicator and the
        # modifier-name-derived side lookup — not section/prefix guesses.
        self.assertIn("bilat_surg", self.aa._DESIGN_SYSTEM_PROMPT)
        self.assertIn("modifier_laterality", self.aa._DESIGN_SYSTEM_PROMPT)

    def test_proposer_prompt_forbids_invented_template_names(self):
        self.assertIn("NEVER return decision=\"rule\" citing a template",
                      self.aa._SYSTEM_PROMPT)


class UnknownTemplateHintTest(unittest.TestCase):
    """A rule proposal citing a NONEXISTENT template is a missing-template
    hint in disguise: _unknown_template_hint converts it so the class
    enters template synthesis instead of dying in the human queue. The
    live failure this guards: a proposal citing
    'laterality_modifier_arbitration' escalated as a plain 'unknown
    template' under protocol 10 and never reached the designer."""

    def setUp(self):
        import tools.auto_actuate as aa
        self.aa = aa

    def test_nonexistent_wellformed_name_converts(self):
        rule = {"id": "toe-mod-completion", "template": "anatomic_modifier_completion",
                "applies_to": {"array": "cpt_codes"}, "authority": "CMS PFS"}
        hint = self.aa._unknown_template_hint(rule, "runs flap on RT")
        self.assertIsNotNone(hint)
        self.assertEqual(hint["name"], "anatomic_modifier_completion")
        self.assertIn("runs flap on RT", hint["mechanism"])
        self.assertIs(hint["attempted_rule"], rule)
        # hints ride into prompts and the flip queue — must serialize
        json.dumps(hint)

    def test_existing_template_yields_no_hint(self):
        # 'unknown template' was not the real failure; nothing to design
        self.assertIsNone(self.aa._unknown_template_hint(
            {"template": "context_gate"}, "r"))

    def test_malformed_names_yield_no_hint(self):
        for bad in ("", "Kebab-Case-Name", "x", "1starts_with_digit",
                    "a" * 60):
            self.assertIsNone(
                self.aa._unknown_template_hint({"template": bad}),
                msg=f"expected no hint for template name {bad!r}")

    def test_missing_rationale_still_produces_mechanism(self):
        hint = self.aa._unknown_template_hint(
            {"template": "some_future_mechanic"})
        self.assertTrue(hint["mechanism"].strip())


class GraduationTest(unittest.TestCase):
    """tools/graduate_templates: a synthesized template graduates into
    the app tree exactly when every proven-in-production criterion holds,
    and the promotion is transactional."""

    def setUp(self):
        import tools.graduate_templates as gt
        from app.validation import auto_templates as at
        from tools import flip_triage as ft
        self.gt, self.at, self.ft = gt, at, ft
        self.tmp = TemporaryDirectory()
        base = Path(self.tmp.name)
        self._saved = (gt.RULES_PATH, gt.GRADUATED_DIR, gt.PROPOSALS_DIR, gt.MIN_DAYS,
                       gt.MIN_DOCS, at.AUTO_TEMPLATES_DIR, ft.QUEUE_PATH)
        gt.RULES_PATH = base / "pack.json"
        gt.GRADUATED_DIR = base / "graduated"
        gt.GRADUATED_DIR.mkdir()
        gt.PROPOSALS_DIR = base / "proposals"
        at.AUTO_TEMPLATES_DIR = base / "auto_templates"
        at.AUTO_TEMPLATES_DIR.mkdir()
        at._cache.clear()
        ft.QUEUE_PATH = base / "flip_queue.jsonl"
        self.results = base / "results"
        self.results.mkdir()
        gt.MIN_DAYS, gt.MIN_DOCS = 14.0, 3

    def tearDown(self):
        (self.gt.RULES_PATH, self.gt.GRADUATED_DIR, self.gt.PROPOSALS_DIR, self.gt.MIN_DAYS,
         self.gt.MIN_DOCS, self.at.AUTO_TEMPLATES_DIR,
         self.ft.QUEUE_PATH) = self._saved
        self.at._cache.clear()
        self.tmp.cleanup()

    def _install(self, actuated_days_ago=30.0, enabled=True,
                 rule_id="g-rule", extra_rules=()):
        (self.at.AUTO_TEMPLATES_DIR / "m.py").write_text(_GOOD_TEMPLATE)
        from datetime import datetime, timedelta, timezone
        when = (datetime.now(timezone.utc)
                - timedelta(days=actuated_days_ago)).isoformat()
        rules = [{"id": rule_id, "template": "sorted_marker",
                  "auto_generated": True, "enabled": enabled,
                  "provenance": {"actuated_at": when}}] + list(extra_rules)
        self.gt.RULES_PATH.write_text(
            json.dumps({"version": "t", "rules": rules}))

    def _fresh_results(self, n):
        for i in range(n):
            (self.results / f"NOTE_{i:03d}_results.json").write_text("{}")

    def test_ineligible_young_and_unexposed(self):
        self._install(actuated_days_ago=1.0)
        rep = self.gt.eligibility(
            "sorted_marker", self.at.AUTO_TEMPLATES_DIR / "m.py",
            self.results)
        self.assertFalse(rep["eligible"])
        self.assertFalse(rep["checks"]["age"]["ok"])
        self.assertFalse(rep["checks"]["exposure"]["ok"])
        # rules/held/static all fine — only maturity blocks
        self.assertTrue(rep["checks"]["rules"]["ok"])
        self.assertTrue(rep["checks"]["held"]["ok"])
        self.assertTrue(rep["checks"]["static"]["ok"])

    def test_disabled_rule_is_disproof(self):
        self._install(extra_rules=[{
            "id": "g-rolled-back", "template": "sorted_marker",
            "auto_generated": True, "enabled": False}])
        self._fresh_results(5)
        rep = self.gt.eligibility(
            "sorted_marker", self.at.AUTO_TEMPLATES_DIR / "m.py",
            self.results)
        self.assertFalse(rep["eligible"])
        self.assertIn("g-rolled-back", rep["checks"]["rules"]["disabled"])

    def test_reopened_flip_class_is_disproof(self):
        self._install()
        self._fresh_results(5)
        self.ft.QUEUE_PATH.write_text(json.dumps({
            "class_key": "k1", "status": "open",
            "actuation": {"rule_id": "g-rule"},
            "reopened": {"reason": "flip recurred"}}) + "\n")
        rep = self.gt.eligibility(
            "sorted_marker", self.at.AUTO_TEMPLATES_DIR / "m.py",
            self.results)
        self.assertFalse(rep["eligible"])
        self.assertEqual(rep["checks"]["held"]["reopened_classes"], ["k1"])

    def test_graduation_creates_inert_proposal(self):
        self._install()
        self._fresh_results(5)
        summary = self.gt.graduate(self.results)
        self.assertEqual([p["template"] for p in summary["promoted"]],
                         ["sorted_marker"])
        self.assertFalse((self.gt.GRADUATED_DIR / "sorted_marker.py").exists())
        proposals = list(self.gt.PROPOSALS_DIR.glob("graduate-*.json"))
        self.assertEqual(len(proposals), 1)
        proposal = json.loads(proposals[0].read_text())
        self.assertEqual(proposal["status"], "draft")
        self.assertEqual(proposal["proposal_type"], "graduate_template")
        # sandbox copy and live rules are untouched
        self.assertTrue((self.at.AUTO_TEMPLATES_DIR / "m.py").exists())
        pack = json.loads(self.gt.RULES_PATH.read_text())
        self.assertTrue(pack["rules"][0]["enabled"])

    def test_promotion_collision_rolls_back(self):
        self._install()
        self._fresh_results(5)
        (self.gt.GRADUATED_DIR / "sorted_marker.py").write_text("# taken\n")
        summary = self.gt.graduate(self.results)
        self.assertEqual(summary["promoted"], [])
        self.assertEqual(len(summary["failed"]), 1)
        # sandbox copy stays authoritative
        self.assertTrue((self.at.AUTO_TEMPLATES_DIR / "m.py").exists())

    def test_proposal_fingerprints_exact_sandbox_source(self):
        src = (
            'TEMPLATE_NAME = "sorted_marker"\n'
            'SCHEMA_DOC = "d"\n'
            "def execute(engine, rule, icd, cpt, hcpcs, coding_result,\n"
            "            note_full_text, note_assessment_text):\n"
            "    if re.search(r'ulcer', note_full_text or ''):\n"
            "        engine.v._add('INFO', 'X', 'rx', 'm', 'r',\n"
            "                      clause='test_marker')\n")
        self._install()
        (self.at.AUTO_TEMPLATES_DIR / "m.py").write_text(src)
        self.at._cache.clear()
        self._fresh_results(5)
        summary = self.gt.graduate(self.results)
        self.assertEqual(len(summary["promoted"]), 1, summary)
        proposal = json.loads(next(self.gt.PROPOSALS_DIR.glob(
            "graduate-*.json")).read_text())
        import hashlib
        expected = "sha256:" + hashlib.sha256(src.encode()).hexdigest()
        self.assertEqual(proposal["source_sha256"], expected)

    def test_dry_run_promotes_nothing(self):
        self._install()
        self._fresh_results(5)
        summary = self.gt.graduate(self.results, dry_run=True)
        self.assertEqual(summary["promoted"],
                         [{"template": "sorted_marker", "dry_run": True}])
        self.assertFalse(
            (self.gt.GRADUATED_DIR / "sorted_marker.py").exists())
        self.assertTrue((self.at.AUTO_TEMPLATES_DIR / "m.py").exists())


class GraduatedDispatchTest(unittest.TestCase):
    """The engine dispatches graduated templates ahead of sandboxed ones,
    and the actuation vocabulary keeps graduated names."""

    def setUp(self):
        from app.validation import auto_templates as at
        from app.validation import graduated as gr
        self.at, self.gr = at, gr
        self.tmp = TemporaryDirectory()
        self._old_dir = at.AUTO_TEMPLATES_DIR
        at.AUTO_TEMPLATES_DIR = Path(self.tmp.name) / "auto_templates"
        at.AUTO_TEMPLATES_DIR.mkdir()
        at._cache.clear()
        self._old_grad = dict(gr.GRADUATED)

    def tearDown(self):
        self.at.AUTO_TEMPLATES_DIR = self._old_dir
        self.at._cache.clear()
        self.gr.GRADUATED.clear()
        self.gr.GRADUATED.update(self._old_grad)
        if getattr(self, "re_mod", None):
            self.re_mod.RULES_FILE = self._old_rules
            self.re_mod.load_rule_pack.cache_clear()
            self.tmp2.cleanup()
        self.tmp.cleanup()

    def _inject_graduated(self, name, marker):
        def execute(engine, rule, icd, cpt, hcpcs, coding_result,
                    note_full_text, note_assessment_text):
            engine.v._add("INFO", "X", marker, "m", "r")
        self.gr.GRADUATED[name] = {"name": name, "schema_doc": "Grad doc.",
                                   "execute": execute, "path": ""}

    def _engine(self, rules):
        import app.validation.rule_engine as re_mod

        class _V:
            _non_billable_codes_to_suppress = set()
            db = None

            def __init__(self):
                self.added = []

            def _add(self, *a, **k):
                self.added.append(a)

        self.tmp2 = TemporaryDirectory()
        pack = Path(self.tmp2.name) / "pack.json"
        pack.write_text(json.dumps({"version": "test", "rules": rules}))
        self._old_rules = re_mod.RULES_FILE
        re_mod.RULES_FILE = pack
        re_mod.load_rule_pack.cache_clear()
        self.re_mod = re_mod
        return re_mod.RuleEngine(_V())

    def test_dispatch_prefers_graduated_over_sandboxed(self):
        # same name in both places: graduated wins (engine trust order)
        (self.at.AUTO_TEMPLATES_DIR / "m.py").write_text(_GOOD_TEMPLATE)
        self._inject_graduated("sorted_marker", "graduated_won")
        eng = self._engine([{
            "id": "g1", "template": "sorted_marker", "enabled": True,
            "auto_generated": True, "authority": "test",
            "applies_to": {"array": "cpt_codes"},
            "action": {"category": "test_cat"}}])
        eng.run_auto_rules([], [{"code": "99213"}], [], {}, "note", "")
        self.assertEqual([a[2] for a in eng.v.added], ["graduated_won"])

    def test_vocabulary_and_prompt_include_graduated(self):
        import tools.auto_actuate as aa
        self._inject_graduated("grad_only_marker", "x")
        self.assertIn("grad_only_marker", aa.all_templates())
        prompt = aa._system_prompt()
        self.assertIn("grad_only_marker", prompt)
        self.assertIn("Grad doc.", prompt)


class ReplayCompletenessWiringTest(unittest.TestCase):
    """The replay path realizes the FINAL shipped claim, so it must feed the
    completeness invariant the documented procedures — otherwise
    _check_procedure_completeness no-ops on exactly the claim that ships and
    the coherence gate reads zero completeness flags (the integration gap a
    live run exposed: per-run validation flagged the drop, but the replayed
    record did not carry it). This guards the pass-through so it cannot
    silently regress. Mock-based: no reference DB / compliance store load."""

    def _replay(self, payload):
        from tools.auto_actuate import Replayer
        rep = Replayer.__new__(Replayer)  # skip the heavy DB/store __init__
        captured = {}

        class _FakeValidator:
            issues: list = []

            def validate(self, coding_result, **kw):
                captured.update(kw)
                return {"validation_issues": []}

        rep._fresh_validator = lambda: _FakeValidator()
        rep.replay_arrays(payload, "note text")
        return captured

    def test_replay_forwards_persisted_procedures(self):
        procs = ["Retrocalcaneal exostectomy (Haglund resection), right heel"]
        captured = self._replay({
            "cpt_codes": [], "hcpcs_codes": [], "icd_codes": [],
            "patient_metadata": {}, "procedures_performed_today": procs})
        self.assertEqual(captured.get("procedures_performed"), procs)

    def test_replay_of_legacy_result_passes_none(self):
        # A pre-fix stored result has no procedures_performed_today — the
        # replayer must degrade to None (completeness no-op), never crash.
        captured = self._replay({
            "cpt_codes": [], "hcpcs_codes": [], "icd_codes": [],
            "patient_metadata": {}})
        self.assertIsNone(captured.get("procedures_performed"))


if __name__ == "__main__":
    unittest.main()
