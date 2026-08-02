"""Tests for rule-pack consolidation — the growth loop's maintenance
counterpart.

Covers:
  1. Pack/corpus hashing (the scan cache key).
  2. The leave-one-out exercise scan (load-bearing detection, caching).
  3. Dormancy tagging (metadata only — enabled flags never change).
  4. Merge candidate grouping and the merge gates (template discipline,
     structural/code-literal reuse, corpus-equivalence rejection).
  5. apply_merge bookkeeping (originals disabled + superseded_by,
     merged rule provenance).
  6. The consolidate driver: acceptance, post-write rollback, and the
     declined-merge ledger (a declined family is not re-asked until the
     pack changes).

All replay is stubbed — no reference DB, no LLM.

Run:  PYTHONPATH=. python -m pytest tests/test_pack_consolidation.py -q
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import tools.auto_actuate as aa
import tools.pack_consolidation as pcons


def _rule(rid, template="context_gate", enabled=True, auto=True, **kw):
    r = {"id": rid, "template": template, "enabled": enabled,
         "auto_generated": auto, "message": "m", "authority": "a"}
    r.update(kw)
    return r


class _PackDir:
    """Temp pack file + state file, with auto_actuate.RULES_PATH and
    pack_consolidation.STATE_PATH pointed at them."""

    def __init__(self, rules):
        self.rules = rules

    def __enter__(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        self.pack_path = d / "validator_rules.json"
        self.pack_path.write_text(json.dumps(
            {"version": "test", "rules": self.rules}))
        self.state_path = d / "rule_exercise.json"
        self.proposals_dir = d / "proposals"
        self.patches = [
            mock.patch.object(aa, "RULES_PATH", self.pack_path),
            mock.patch.object(pcons, "STATE_PATH", self.state_path),
            mock.patch.object(pcons, "PROPOSALS_DIR", self.proposals_dir),
        ]
        for p in self.patches:
            p.start()
        return self

    def __exit__(self, *exc):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()
        return False

    def pack(self):
        return json.loads(self.pack_path.read_text())


# ---------------------------------------------------------------------------
# 1. hashing
# ---------------------------------------------------------------------------

class HashTest(unittest.TestCase):
    def test_pack_hash_tracks_enabled_rules_only(self):
        p1 = {"rules": [_rule("r1"), _rule("r2", enabled=False)]}
        p2 = {"rules": [_rule("r1")]}
        self.assertEqual(pcons._pack_hash(p1), pcons._pack_hash(p2))
        p3 = {"rules": [_rule("r1"), _rule("r2")]}
        self.assertNotEqual(pcons._pack_hash(p1), pcons._pack_hash(p3))

    def test_corpus_hash_tracks_docs_and_run_counts(self):
        c1 = [("docA", [{}, {}], "n")]
        c2 = [("docA", [{}, {}, {}], "n")]
        self.assertNotEqual(pcons._corpus_hash(c1), pcons._corpus_hash(c2))

    def test_corpus_hash_tracks_run_content_not_just_counts(self):
        # a fresh generative cycle overwrites runs with the SAME count —
        # the scan cache must miss, or dormancy evidence goes stale
        c1 = [("docA", [{"cpt_codes": [{"code": "X"}]}], "n")]
        c2 = [("docA", [{"cpt_codes": [{"code": "Y"}]}], "n")]
        self.assertNotEqual(pcons._corpus_hash(c1), pcons._corpus_hash(c2))

    def test_pack_hash_ignores_dormancy_metadata(self):
        # tag_dormancy writes metadata derived FROM the scan — it must
        # not invalidate the scan it came from
        p1 = {"rules": [_rule("r1")]}
        p2 = {"rules": [dict(_rule("r1"), dormant_on_corpus=True,
                             dormant_since="2026-01-01")]}
        self.assertEqual(pcons._pack_hash(p1), pcons._pack_hash(p2))


# ---------------------------------------------------------------------------
# 2. exercise scan
# ---------------------------------------------------------------------------

class ExerciseScanTest(unittest.TestCase):
    def _scan(self, fingerprint_seq):
        """Run exercise_scan over a 2-auto-rule pack with _corpus and
        _fingerprints stubbed; fingerprint_seq drives baseline + one
        leave-one-out replay per rule."""
        with _PackDir([_rule("r1"), _rule("r2"),
                       _rule("hand", auto=False)]) as pd, \
                mock.patch.object(pcons, "_corpus",
                                  return_value=[("docA", [{}], "note")]), \
                mock.patch.object(pcons, "_fingerprints",
                                  side_effect=fingerprint_seq) as fps, \
                mock.patch.object(aa, "_advisory_scrubber",
                                  return_value=None):
            scan = pcons.exercise_scan(Path("unused"), rep=object())
            return scan, fps, pd

    def test_load_bearing_detection(self):
        scan, _, _ = self._scan([
            {"docA": ["base"]},      # baseline
            {"docA": ["changed"]},   # r1 disabled -> differs
            {"docA": ["base"]},      # r2 disabled -> identical
        ])
        self.assertEqual(scan["rules"]["r1"]["load_bearing_on"], ["docA"])
        self.assertEqual(scan["rules"]["r2"]["load_bearing_on"], [])
        self.assertNotIn("hand", scan["rules"])

    def test_scan_is_cached_by_pack_and_corpus_hash(self):
        with _PackDir([_rule("r1")]) as pd, \
                mock.patch.object(pcons, "_corpus",
                                  return_value=[("docA", [{}], "note")]), \
                mock.patch.object(pcons, "_fingerprints",
                                  return_value={"docA": ["x"]}) as fps, \
                mock.patch.object(aa, "_advisory_scrubber",
                                  return_value=None):
            pcons.exercise_scan(Path("unused"), rep=object())
            n = fps.call_count
            scan2 = pcons.exercise_scan(Path("unused"), rep=object())
            self.assertEqual(fps.call_count, n)  # no replays on hit
            self.assertIn("r1", scan2["rules"])


# ---------------------------------------------------------------------------
# 3. dormancy tags
# ---------------------------------------------------------------------------

class DormancyTest(unittest.TestCase):
    def test_tag_and_clear_without_touching_enabled(self):
        with _PackDir([_rule("r1"), _rule("r2")]) as pd:
            out = pcons.tag_dormancy({"rules": {
                "r1": {"load_bearing_on": []},
                "r2": {"load_bearing_on": ["docA"]}}})
            self.assertEqual(out["tagged"], ["r1"])
            rules = {r["id"]: r for r in pd.pack()["rules"]}
            self.assertTrue(rules["r1"]["enabled"])
            self.assertNotIn("dormant_on_corpus", rules["r1"])
            state = json.loads(pd.state_path.read_text())
            self.assertIn("r1", state["dormancy"])
            # a later scan finds r1 load-bearing -> tag clears
            out = pcons.tag_dormancy({"rules": {
                "r1": {"load_bearing_on": ["docB"]}}})
            self.assertEqual(out["cleared"], ["r1"])
            rules = {r["id"]: r for r in pd.pack()["rules"]}
            self.assertNotIn("dormant_on_corpus", rules["r1"])
            self.assertNotIn("dormant_since", rules["r1"])
            state = json.loads(pd.state_path.read_text())
            self.assertNotIn("r1", state["dormancy"])


# ---------------------------------------------------------------------------
# 4. merge candidates + gates
# ---------------------------------------------------------------------------

class MergeGateTest(unittest.TestCase):
    def test_candidates_group_by_template_min_two(self):
        pack = {"rules": [
            _rule("a1", "context_gate"), _rule("a2", "context_gate"),
            _rule("b1", "companion_completion"),
            _rule("dis", "context_gate", enabled=False),
            _rule("hand", "context_gate", auto=False)]}
        fams = pcons.merge_candidates(pack, {})
        self.assertEqual(len(fams), 1)
        self.assertEqual([r["id"] for r in fams[0]], ["a1", "a2"])

    def _gate(self, merged, fingerprint_after, pack=None):
        family = [_rule("a1"), _rule("a2")]
        pack = pack or {"rules": family + [_rule("hand", auto=False)]}
        baseline = {"docA": ["base"]}
        with _PackDir(pack["rules"]), \
                mock.patch.object(aa, "gate_structural",
                                  return_value=""), \
                mock.patch.object(aa, "gate_no_code_literals",
                                  return_value=""), \
                mock.patch.object(pcons, "_fingerprints",
                                  return_value=fingerprint_after):
            return pcons.gate_merge(merged, family, pack, baseline,
                                    object(), None,
                                    [("docA", [{}], "note")])

    def test_template_mismatch_rejected(self):
        why = self._gate(_rule("m1", template="companion_completion"),
                         {"docA": ["base"]})
        self.assertIn("template", why)

    def test_id_collision_rejected(self):
        why = self._gate(_rule("hand"), {"docA": ["base"]})
        self.assertIn("collides", why)

    def test_nonidentical_corpus_replay_rejected(self):
        why = self._gate(_rule("m1"), {"docA": ["DIFFERENT"]})
        self.assertIn("not byte-identical", why)

    def test_identical_replay_accepted(self):
        self.assertEqual(self._gate(_rule("m1"), {"docA": ["base"]}), "")

    def test_structural_gate_is_consulted(self):
        family = [_rule("a1"), _rule("a2")]
        with _PackDir(family), \
                mock.patch.object(aa, "gate_structural",
                                  return_value="bad rule id"):
            why = pcons.gate_merge(_rule("m1"), family,
                                   {"rules": family}, {}, object(),
                                   None, [])
        self.assertIn("structural", why)


# ---------------------------------------------------------------------------
# 5. apply_merge bookkeeping
# ---------------------------------------------------------------------------

class ApplyMergeTest(unittest.TestCase):
    def test_merge_is_inert_proposal_and_live_pack_is_unchanged(self):
        with _PackDir([_rule("a1"), _rule("a2"),
                       _rule("hand", auto=False)]) as pd:
            path = pcons.apply_merge(_rule("merged-1"),
                                     [_rule("a1"), _rule("a2")])
            rules = {r["id"]: r for r in pd.pack()["rules"]}
            for rid in ("a1", "a2"):
                self.assertTrue(rules[rid]["enabled"])
            self.assertNotIn("merged-1", rules)
            self.assertTrue(rules["hand"]["enabled"])
            proposal = json.loads(path.read_text())
            self.assertEqual(proposal["status"], "draft")
            self.assertFalse(proposal["rule"]["enabled"])
            self.assertEqual(proposal["rule"]["provenance"]
                             ["consolidated_from"], ["a1", "a2"])


# ---------------------------------------------------------------------------
# 6. consolidate driver
# ---------------------------------------------------------------------------

class ConsolidateDriverTest(unittest.TestCase):
    def _drive(self, proposal, fingerprint_seq, rules=None,
               and_then=None):
        """Run consolidate with everything replay/LLM-shaped stubbed.
        Returns (summary, pack dict, state dict) captured INSIDE the
        temp-dir context; `and_then(pd)` runs extra in-context steps."""
        rules = rules or [_rule("a1"), _rule("a2")]
        scan = {"pack_hash": "x", "corpus_hash": "y",
                "rules": {r["id"]: {"load_bearing_on": ["docA"]}
                          for r in rules}}
        with _PackDir(rules) as pd, \
                mock.patch.object(pcons, "exercise_scan",
                                  return_value=scan), \
                mock.patch.object(pcons, "tag_dormancy",
                                  return_value={"tagged": [],
                                                "cleared": []}), \
                mock.patch.object(pcons, "_corpus",
                                  return_value=[("docA", [{}], "n")]), \
                mock.patch.object(pcons, "_fingerprints",
                                  side_effect=fingerprint_seq), \
                mock.patch.object(pcons, "propose_merge",
                                  return_value=proposal), \
                mock.patch.object(aa, "_advisory_scrubber",
                                  return_value=None), \
                mock.patch.object(aa, "gate_structural",
                                  return_value=""), \
                mock.patch.object(aa, "gate_no_code_literals",
                                  return_value=""):
            summary = pcons.consolidate(Path("unused"), rep=object())
            pack = pd.pack()
            state = (json.loads(pd.state_path.read_text())
                     if pd.state_path.exists() else {})
            if and_then is not None:
                and_then(pd)
            return summary, pack, state

    def test_accepted_merge_creates_proposal_only(self):
        summary, pack, _ = self._drive(
            {"decision": "merge", "rule": _rule("merged-1")},
            [{"docA": ["base"]},   # baseline
             {"docA": ["base"]}])  # gate_merge replay
        self.assertEqual(summary["merges"][0]["status"], "draft")
        self.assertEqual(summary["merges"][0]["merged_id"], "merged-1")
        rules = {r["id"]: r for r in pack["rules"]}
        self.assertNotIn("merged-1", rules)
        self.assertTrue(rules["a1"]["enabled"])

    def test_no_post_write_live_pack_mutation_occurs(self):
        with mock.patch.object(aa, "_disable_rule") as dis, \
                mock.patch.object(aa, "_reenable_rule") as ren:
            summary, _, _ = self._drive(
                {"decision": "merge", "rule": _rule("merged-1")},
                [{"docA": ["base"]},        # baseline
                 {"docA": ["base"]}])       # proposal gate
        self.assertEqual(len(summary["merges"]), 1)
        dis.assert_not_called()
        ren.assert_not_called()

    def test_proposer_decline_is_ledgered_and_not_reasked(self):
        def second_run(pd):
            # same pack -> family skipped entirely on the next run
            with mock.patch.object(pcons, "propose_merge") as pm:
                pcons.consolidate(Path("unused"), rep=object())
                pm.assert_not_called()

        summary, _, state = self._drive(
            {"decision": "decline", "why": "different policies"},
            # baseline for run 1; run 2 filters the family out before
            # any replay, so no further fingerprints are consumed
            [{"docA": ["base"]}, {"docA": ["base"]}],
            and_then=second_run)
        self.assertEqual(summary["merges"], [])
        self.assertEqual(summary["declined"][0]["why"],
                         "different policies")
        self.assertEqual(len(state["declined_merges"]), 1)

    def test_nonidentical_gate_rejection_is_ledgered(self):
        summary, _, state = self._drive(
            {"decision": "merge", "rule": _rule("merged-1")},
            [{"docA": ["base"]},        # baseline
             {"docA": ["DIFFERENT"]}])  # gate_merge replay differs
        self.assertEqual(summary["merges"], [])
        self.assertIn("not byte-identical", summary["rejected"][0]["why"])
        self.assertEqual(len(state["declined_merges"]), 1)


if __name__ == "__main__":
    unittest.main()
