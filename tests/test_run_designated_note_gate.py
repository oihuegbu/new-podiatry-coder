"""tools/run_designated_note_gate.py: the release-gate acceptance run must
refuse to start without a versioned, non-PHI context fixture -- it must never
spend model calls on a note predetermined to hold on every line (issue #6,
Codex's independent re-review, root cause 1)."""
import importlib.util
import json
import pathlib
import unittest
from unittest import mock


def _mod():
    root = pathlib.Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        "run_designated_note_gate", root / "tools" / "run_designated_note_gate.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class ReleaseGateContextRequirementTest(unittest.TestCase):

    def test_refuses_when_encounter_context_is_absent(self):
        m = _mod()
        with mock.patch.object(m, "BILLING_CONTEXT", pathlib.Path("/nonexistent/billing.json")), \
                mock.patch.object(m, "ENCOUNTER_CONTEXT",
                                  pathlib.Path("/nonexistent/encounter.json")), \
                mock.patch.object(m.subprocess, "call") as fake_call:
            rc = m.main([])
        self.assertEqual(rc, 2)
        fake_call.assert_not_called()

    def test_refuses_a_malformed_or_unversioned_fixture(self):
        m = _mod()
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            billing = pathlib.Path(td) / "billing.json"
            billing.write_text("{}")
            bad_encounter = pathlib.Path(td) / "encounter.json"
            bad_encounter.write_text(json.dumps({"schema": "encounter_context/2"}))
            with mock.patch.object(m, "BILLING_CONTEXT", billing), \
                    mock.patch.object(m, "ENCOUNTER_CONTEXT", bad_encounter), \
                    mock.patch.object(m.subprocess, "call") as fake_call:
                rc = m.main([])
        self.assertEqual(rc, 2, "a fixture declaring no version must be refused")
        fake_call.assert_not_called()

    def test_valid_fixture_invokes_the_real_run_py_entrypoint_with_both_contexts(self):
        m = _mod()
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            billing = pathlib.Path(td) / "billing.json"
            billing.write_text('{"billing_entity_id": "x", "participants": []}')
            encounter = pathlib.Path(td) / "encounter.json"
            encounter.write_text(json.dumps({
                "schema": "encounter_context/2",
                "version": "test-edition-0001",
                "patients": {}, "payers": {}, "providers": {},
                "billing_entities": {}, "facilities": {},
                "coverages": [], "affiliations": [], "authorizations": [],
                "encounters": {}}))
            with mock.patch.object(m, "BILLING_CONTEXT", billing), \
                    mock.patch.object(m, "ENCOUNTER_CONTEXT", encounter), \
                    mock.patch.object(m.subprocess, "call", return_value=0) as fake_call:
                rc = m.main([])
        self.assertEqual(rc, 0)
        (args,), _kwargs = fake_call.call_args
        self.assertIn("--note", args)
        self.assertIn(m.DESIGNATED_NOTE, args)
        self.assertIn("--billing-context", args)
        self.assertIn(str(billing), args)
        self.assertIn("--encounter-context", args)
        self.assertIn(str(encounter), args)


if __name__ == "__main__":
    unittest.main()
