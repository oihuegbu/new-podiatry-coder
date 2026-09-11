"""tools/run_designated_note_gate.py: the release-gate acceptance run must
refuse to start without a versioned, non-PHI context fixture -- it must never
spend model calls on a note predetermined to hold on every line (issue #6,
Codex's independent re-review, root cause 1).

Round 2 (Codex's independent re-review, P1): the original version validated
only that the fixture FILES EXIST, so genuinely malformed content of either
kind could still start the paid note run -- its own regression test happened
to supply a trivial `{}` billing context and never independently proved
billing validation triggers anything (the encounter side alone was invalid).
Validation now goes through the SAME loaders the real pipeline uses
(`run.load_billing_context` + `claude_coder.extraction._participant_index`;
`app.contracts.encounter_context.build_provider(...).preflight()`), and each
side is now tested with the OTHER side held valid, so a pass can only be
attributed to the side actually under test.
"""
import importlib.util
import json
import pathlib
import unittest
from unittest import mock

_VALID_ENCOUNTER = {
    "schema": "encounter_context/2",
    "version": "test-edition-0001",
    "patients": {}, "payers": {}, "providers": {},
    "billing_entities": {}, "facilities": {},
    "coverages": [], "affiliations": [], "authorizations": [],
    "encounters": {"Right_Retrocalcaneal_Exostectomy_Operative_Note": {
        "patient_id": "", "coverage_id": "", "rendering_provider_npi": "",
        "facility_id": "", "authorization_id": "", "date_of_service": ""}},
}
_VALID_BILLING = {"billing_entity_id": "x", "participants": []}


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

    def test_refuses_an_unversioned_encounter_context_even_with_valid_billing(self):
        m = _mod()
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            billing = pathlib.Path(td) / "billing.json"
            billing.write_text(json.dumps(_VALID_BILLING))
            bad_encounter = pathlib.Path(td) / "encounter.json"
            bad_encounter.write_text(json.dumps({"schema": "encounter_context/2"}))
            with mock.patch.object(m, "BILLING_CONTEXT", billing), \
                    mock.patch.object(m, "ENCOUNTER_CONTEXT", bad_encounter), \
                    mock.patch.object(m.subprocess, "call") as fake_call:
                rc = m.main([])
        self.assertEqual(rc, 2, "a fixture declaring no version must be refused")
        fake_call.assert_not_called()

    def test_refuses_an_encounter_context_missing_the_designated_encounter(self):
        """A schema-valid, versioned fixture that simply never mentions THIS
        gate's designated note cannot resolve context for it -- refused before
        any model call, not discovered only after the note fails downstream."""
        m = _mod()
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            billing = pathlib.Path(td) / "billing.json"
            billing.write_text(json.dumps(_VALID_BILLING))
            encounter = pathlib.Path(td) / "encounter.json"
            no_designated_note = dict(_VALID_ENCOUNTER, encounters={})
            encounter.write_text(json.dumps(no_designated_note))
            with mock.patch.object(m, "BILLING_CONTEXT", billing), \
                    mock.patch.object(m, "ENCOUNTER_CONTEXT", encounter), \
                    mock.patch.object(m.subprocess, "call") as fake_call:
                rc = m.main([])
        self.assertEqual(rc, 2)
        fake_call.assert_not_called()

    def test_refuses_a_malformed_billing_context_even_with_valid_encounter_context(self):
        """issue #6, Codex's independent re-review, root cause 1 (P1): the
        SAME encounter fixture that validates cleanly in the passing test
        below, but billing context has a participant with no 'id' -- must
        be refused by the REAL `_participant_index` validation, never by
        merely checking the file exists."""
        m = _mod()
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            billing = pathlib.Path(td) / "billing.json"
            billing.write_text(json.dumps(
                {"billing_entity_id": "x", "participants": [{"type": "person"}]}))
            encounter = pathlib.Path(td) / "encounter.json"
            encounter.write_text(json.dumps(_VALID_ENCOUNTER))
            with mock.patch.object(m, "BILLING_CONTEXT", billing), \
                    mock.patch.object(m, "ENCOUNTER_CONTEXT", encounter), \
                    mock.patch.object(m.subprocess, "call") as fake_call:
                rc = m.main([])
        self.assertEqual(rc, 2,
                         "a billing participant with no 'id' must be refused")
        fake_call.assert_not_called()

    def test_an_import_failure_during_billing_validation_is_never_relabeled_as_invalid_billing(
            self):
        """issue #6, Codex's independent re-review (round 3): a bare `except
        Exception` used to wrap BOTH the `from run import ...` / `from
        claude_coder.extraction import ...` statements AND the actual validation
        calls, so an import-time failure with nothing to do with whether the
        billing content is valid got printed as "billing context is invalid" and
        exit 2 -- indistinguishable, to anything checking only the exit code,
        from a genuine validation failure. That let a test assert refusal on
        malformed billing content while validation never actually ran (Codex's
        own environment hit exactly this: importing the production loader also
        imports an unavailable PDF dependency, and the "invalid billing"-shaped
        tests above passed on that import failure instead of `_participant_index`
        ever running). Simulated here with an otherwise fully VALID billing
        fixture and `run` failing to import -- this must surface as the import
        failure itself, never a "billing context is invalid" refusal.
        """
        m = _mod()
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            billing = pathlib.Path(td) / "billing.json"
            billing.write_text(json.dumps(_VALID_BILLING))
            encounter = pathlib.Path(td) / "encounter.json"
            encounter.write_text(json.dumps(_VALID_ENCOUNTER))
            with mock.patch.object(m, "BILLING_CONTEXT", billing), \
                    mock.patch.object(m, "ENCOUNTER_CONTEXT", encounter), \
                    mock.patch.dict("sys.modules", {"run": None}):
                with self.assertRaises(ImportError):
                    m.main([])

    def test_valid_fixture_invokes_the_real_run_py_entrypoint_with_both_contexts(self):
        m = _mod()
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            billing = pathlib.Path(td) / "billing.json"
            billing.write_text(json.dumps(_VALID_BILLING))
            encounter = pathlib.Path(td) / "encounter.json"
            encounter.write_text(json.dumps(_VALID_ENCOUNTER))
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
