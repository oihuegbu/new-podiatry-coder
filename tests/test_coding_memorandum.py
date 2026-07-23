"""Tests for the coding memorandum — the pack's proven corrections fed
back upstream to the generative coder.

Covers:
  1. Entry compilation: only enabled auto rules with an actuation
     rationale qualify; scan evidence narrows to load-bearing rules;
     no scan evidence means include-everything (teach unproven truths
     rather than nothing).
  2. The prompt block shape and the CODING_MEMORANDUM=0 toggle.
  3. Automatic recompilation when the pack file changes (no
     regeneration step to forget).
  4. Failure posture: a missing/corrupt pack yields an empty block,
     never an exception into the coding pass.

Run:  PYTHONPATH=. python -m pytest tests/test_coding_memorandum.py -q
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import app.coding.memorandum as memo


def _rule(rid, enabled=True, auto=True, rationale="always add RT when "
          "the descriptor is unilateral and the note documents the side",
          docs=("docA",)):
    r = {"id": rid, "template": "context_gate", "enabled": enabled,
         "auto_generated": auto, "authority": "CPT Appendix A"}
    if rationale is not None:
        r["provenance"] = {"rationale": rationale,
                           "documents": list(docs)}
    return r


class _MemoDir:
    def __init__(self, rules, scan=None):
        self.rules, self.scan = rules, scan

    def __enter__(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        self.rules_path = d / "validator_rules.json"
        self.rules_path.write_text(json.dumps(
            {"version": "t", "rules": self.rules}))
        self.exercise_path = d / "rule_exercise.json"
        if self.scan is not None:
            self.exercise_path.write_text(json.dumps(
                {"scan": {"rules": self.scan}}))
        self.patches = [
            mock.patch.object(memo, "RULES_PATH", self.rules_path),
            mock.patch.object(memo, "EXERCISE_PATH", self.exercise_path),
        ]
        for p in self.patches:
            p.start()
        memo._cache["key"] = None
        memo._cache["block"] = ""
        return self

    def __exit__(self, *exc):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()
        memo._cache["key"] = None
        memo._cache["block"] = ""
        return False


class EntryCompilationTest(unittest.TestCase):
    def test_only_enabled_auto_rules_with_rationale(self):
        with _MemoDir([_rule("keep"),
                       _rule("disabled", enabled=False),
                       _rule("hand", auto=False),
                       _rule("no-rationale", rationale=None)]):
            block = memo.memorandum_block()
        self.assertIn("always add RT", block)
        self.assertEqual(block.count("always add RT"), 1)

    def test_scan_evidence_drops_proven_inert_rules(self):
        with _MemoDir([_rule("bearing"), _rule("inert")],
                      scan={"bearing": {"load_bearing_on": ["docA"]},
                            "inert": {"load_bearing_on": []}}):
            pack = json.loads(memo.RULES_PATH.read_text())
            entries = memo._entries(pack)
        self.assertEqual([e["rule_id"] for e in entries], ["bearing"])

    def test_rules_the_scan_has_not_seen_stay_included(self):
        # a rule minted AFTER the last scan is unknown, not inert — a
        # stale scan must never silence the pack's freshest corrections
        with _MemoDir([_rule("scanned"), _rule("fresh")],
                      scan={"scanned": {"load_bearing_on": ["docA"]}}):
            pack = json.loads(memo.RULES_PATH.read_text())
            entries = memo._entries(pack)
        self.assertEqual({e["rule_id"] for e in entries},
                         {"scanned", "fresh"})

    def test_no_scan_evidence_includes_everything(self):
        with _MemoDir([_rule("r1"), _rule("r2")]):
            pack = json.loads(memo.RULES_PATH.read_text())
            entries = memo._entries(pack)
        self.assertEqual(len(entries), 2)

    def test_authority_rides_along(self):
        with _MemoDir([_rule("r1")]):
            block = memo.memorandum_block()
        self.assertIn("[CPT Appendix A]", block)


class ToggleAndBlockTest(unittest.TestCase):
    def test_toggle_off_yields_empty_block(self):
        with _MemoDir([_rule("r1")]), \
                mock.patch.dict(os.environ, {"CODING_MEMORANDUM": "0"}):
            self.assertEqual(memo.memorandum_block(), "")

    def test_empty_pack_yields_empty_block(self):
        with _MemoDir([_rule("hand", auto=False)]):
            self.assertEqual(memo.memorandum_block(), "")

    def test_block_carries_the_header(self):
        with _MemoDir([_rule("r1")]):
            block = memo.memorandum_block()
        self.assertTrue(block.startswith("## CODING MEMORANDUM"))


class RecompilationTest(unittest.TestCase):
    def test_pack_change_recompiles_without_a_regeneration_step(self):
        with _MemoDir([_rule("r1")]) as md:
            b1 = memo.memorandum_block()
            self.assertIn("always add RT", b1)
            md.rules_path.write_text(json.dumps({"version": "t", "rules": [
                _rule("r2", rationale="never report A-codes with a "
                                      "global-period procedure")]}))
            b2 = memo.memorandum_block()
        self.assertIn("never report A-codes", b2)
        self.assertNotIn("always add RT", b2)

    def test_unchanged_pack_serves_the_cache(self):
        with _MemoDir([_rule("r1")]):
            b1 = memo.memorandum_block()
            with mock.patch.object(memo, "_entries") as ent:
                b2 = memo.memorandum_block()
                ent.assert_not_called()
        self.assertEqual(b1, b2)


class FailurePostureTest(unittest.TestCase):
    def test_missing_pack_is_empty_block(self):
        with _MemoDir([_rule("r1")]) as md:
            md.rules_path.unlink()
            self.assertEqual(memo.memorandum_block(), "")

    def test_corrupt_pack_is_empty_block_not_exception(self):
        with _MemoDir([_rule("r1")]) as md:
            md.rules_path.write_text("{not json")
            self.assertEqual(memo.memorandum_block(), "")


if __name__ == "__main__":
    unittest.main()
