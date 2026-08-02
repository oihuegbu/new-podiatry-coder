"""Auto-authored rules remain inert drafts until governed promotion."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools import auto_actuate


class GovernedProposalTest(unittest.TestCase):
    def test_accept_rule_never_mutates_active_pack(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            pack = root / "validator_rules.json"
            pack.write_text(json.dumps({"version": "test", "rules": []}))
            proposals = root / "proposals"
            with mock.patch.object(auto_actuate, "RULES_PATH", pack), \
                    mock.patch.object(auto_actuate, "PROPOSALS_DIR", proposals):
                auto_actuate.accept_rule(
                    {"id": "candidate", "template": "context_gate"},
                    {"class_key": "test-class",
                     "documents": [{"document_id": "doc"}]},
                    "grounded rationale", {"documents": {}})
            self.assertEqual(json.loads(pack.read_text())["rules"], [])
            drafts = list(proposals.glob("*.json"))
            self.assertEqual(len(drafts), 1)
            proposal = json.loads(drafts[0].read_text())
            self.assertEqual(proposal["status"], "draft")
            self.assertFalse(proposal["rule"]["enabled"])
            self.assertIn("independent_human_review",
                          proposal["required_lifecycle"])

    def test_template_replay_never_writes_live_executable_directory(self):
        from app.validation import auto_templates
        source = '''TEMPLATE_NAME = "candidate_template"
SCHEMA_DOC = "test"
def execute(engine, rule, icd, cpt, hcpcs, coding_result, note_full_text, note_assessment_text):
    return None
'''
        with tempfile.TemporaryDirectory() as td:
            live = Path(td) / "live"
            live.mkdir()
            old = auto_templates.AUTO_TEMPLATES_DIR
            auto_templates.AUTO_TEMPLATES_DIR = live

            def replay(*args, **kwargs):
                self.assertNotEqual(auto_templates.AUTO_TEMPLATES_DIR, live)
                self.assertFalse((live / "candidate_template.py").exists())
                return "", {"documents": {}}

            try:
                with mock.patch.object(auto_actuate, "_self_authored",
                                       return_value={}), \
                        mock.patch.object(auto_actuate, "gate_structural",
                                          return_value=""), \
                        mock.patch.object(auto_actuate,
                                          "gate_no_code_literals",
                                          return_value=""), \
                        mock.patch.object(auto_actuate, "gate_replay",
                                          side_effect=replay):
                    reason, _detail, returned = \
                        auto_actuate._gate_template_pair(
                            source, {"template": "candidate_template"},
                            {}, [], object(), Path(td), (), {})
            finally:
                auto_templates.AUTO_TEMPLATES_DIR = old
            self.assertEqual(reason, "")
            self.assertEqual(returned, source)
            self.assertEqual(list(live.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
