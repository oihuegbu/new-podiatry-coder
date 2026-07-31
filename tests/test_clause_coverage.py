"""The clause-tagging migration's guards, run as part of the suite.

check_no_hardcoding.py and check_rule_coverage.py are standalone commands
documented in the README — which means they gate only when someone
remembers to type them. This migration will run for weeks across 141 sites
and four trees; a guard that depends on recall is not a ratchet. So the
counter is exercised here too, and the admission gate's behavior is pinned
where it can regress silently: in the direction of the null-clause
fallback.
"""
from __future__ import annotations

import unittest

from app.validation.auto_templates import validate_template_clause_tagging
from tests import check_clause_coverage as ccc


class TestClauseRatchet(unittest.TestCase):
    def test_untagged_site_count_matches_the_ratchet(self):
        total, _ = ccc.scan()
        self.assertEqual(
            total, ccc.BASELINE_UNTAGGED,
            f"Untagged _add(...) sites = {total}, ratchet = "
            f"{ccc.BASELINE_UNTAGGED}. If you tagged sites, lower "
            f"BASELINE_UNTAGGED to {total}. If you added an untagged "
            f"emission site, pass clause= instead of raising the ratchet.")

    def test_guard_exits_zero_at_baseline(self):
        self.assertEqual(ccc.main(), 0)

    def test_every_scanned_tree_is_declared(self):
        """A tree that can reach _add but is not scanned is a hole the
        counter reads as progress."""
        names = {d.name for _, d, _ in ccc.SCAN_TREES}
        self.assertIn("auto_templates", names)
        self.assertIn("graduated", names)


_TPL = '''
TEMPLATE_NAME = "clause_gate_fixture"
SCHEMA_DOC = "fixture template for the clause-tagging admission gate"
def execute(engine, rule, icd, cpt, hcpcs, coding_result,
            note_full_text, note_assessment_text):
    v = engine.v
{body}
'''


def _problems(body: str) -> list[str]:
    return validate_template_clause_tagging(_TPL.format(body=body))


class TestAdmissionGate(unittest.TestCase):
    def test_untagged_add_is_rejected(self):
        p = _problems('    v._add("WARNING", "x", "cat", "m", "r")')
        self.assertTrue(p)
        self.assertIn("clause", p[0])

    def test_tagged_add_is_accepted(self):
        self.assertEqual(
            _problems('    v._add("WARNING", "x", "cat", "m", "r",\n'
                      '           clause="coverage_composition")'), [])

    def test_empty_clause_is_rejected(self):
        """An empty clause is reachable by unscoped suppression, i.e. it is
        untagged wearing a tag."""
        self.assertTrue(_problems('    v._add("W", "x", "c", "m", "r", '
                                  'clause="")'))

    def test_kwargs_spread_is_rejected_not_assumed_tagged(self):
        self.assertTrue(_problems('    v._add(**payload)'))

    def test_free_form_clause_is_rejected(self):
        """Clauses are matched exactly against Finding.clause, so prose
        would fail to match while reading as tagged."""
        self.assertTrue(_problems('    v._add("W", "x", "c", "m", "r", '
                                  'clause="Coverage Composition!")'))

    def test_gate_is_not_wired_into_the_load_time_gate(self):
        """The separation is the design: folding clause tagging into
        validate_template_source would skip all 14 installed templates on
        the next load and disable every rule referencing them."""
        from app.validation import auto_templates as at
        src = _TPL.format(
            body='    v._add("WARNING", "x", "cat", "m", "r")')
        self.assertEqual(at.validate_template_source(src), [])
        self.assertTrue(at.validate_template_clause_tagging(src))

    def test_installed_templates_still_load_untagged(self):
        """The 20 untagged sites in the sandbox must keep working — the
        migration may not take live rules dark."""
        from app.validation.auto_templates import load_auto_templates
        self.assertTrue(load_auto_templates())


class TestSuppressionRoundTrip(unittest.TestCase):
    """The inversion is a ROUND TRIP, so testing either half alone proves
    nothing about it: the validator emits the directive, it rides the
    report as `scrub_advisory_suppressions` (validator.py, the
    "scrub_advisory_suppressions" report key), and the engine matches it at
    engine._apply_advisory_suppressions.

    This needs no note text, no pipeline run, and no decision about the
    fixture capture boundary — `_apply_advisory_suppressions(findings,
    suppressions, filter_id)` is a pure function over a list of Findings.
    The FIXTURE CORPUS needs the wide boundary; this regression does not,
    and conflating the two is what made this look blocked.
    """

    def setUp(self):
        from app.compliance.engine import _apply_advisory_suppressions
        from app.compliance.models import DenialRisk, Finding, Status
        from app.validation.validator import CodingValidator
        self.apply = _apply_advisory_suppressions
        self.Finding, self.Status, self.DenialRisk = (
            Finding, Status, DenialRisk)
        self.v = CodingValidator.__new__(CodingValidator)
        self.v._scrub_advisory_suppressions = []

    def _finding(self, clause):
        return self.Finding(
            filter_id="MEDICAL_NECESSITY", status=self.Status.WARN,
            codes=["Z79.01"], reason="advisory", clause=clause,
            denial_risk=self.DenialRisk.LOW)

    def _directive(self, clause):
        """Emit through the validator's own API, so the test breaks if the
        directive's shape drifts from what the engine expects."""
        self.v._scrub_advisory_suppressions = []
        self.v.suppress_scrub_advisory(
            "MEDICAL_NECESSITY", "Z79.01", rule_id="rt-rule",
            authority="ICD-10-CM I.C.21.c.3", clause=clause,
            validator_categories=["unjustified_zcode"])
        return self.v._scrub_advisory_suppressions

    def _survives(self, directive_clause, finding_clause):
        out = self.apply([self._finding(finding_clause)],
                         self._directive(directive_clause),
                         "MEDICAL_NECESSITY")
        return [f for f in out if f.status == self.Status.WARN]

    def test_named_clause_is_retired(self):
        self.assertFalse(self._survives("class_findings_modifier",
                                        "class_findings_modifier"))

    def test_sibling_clause_survives(self):
        """routine_00003: a suppression grounded on the Q-modifier pathway
        must not flip the composition gate to PASS."""
        self.assertTrue(self._survives("class_findings_modifier",
                                       "coverage_composition"))

    def test_engine_is_both_null_exact(self):
        """The engine's null semantics, pinned so the validator's
        transitional ramp is never 'fixed' by loosening this half."""
        self.assertFalse(self._survives("", ""))
        self.assertTrue(self._survives("", "coverage_composition"))
        self.assertTrue(self._survives("class_findings_modifier", ""))

    def test_fail_is_never_config_suppressible(self):
        f = self._finding("class_findings_modifier")
        f.status = self.Status.FAIL
        out = self.apply([f], self._directive("class_findings_modifier"),
                         "MEDICAL_NECESSITY")
        self.assertEqual([x.status for x in out], [self.Status.FAIL])

    def test_the_two_halves_disagree_only_on_the_documented_axis(self):
        """The ONLY intended divergence is the untagged-issue ramp:
        engine = both-null-exact, validator = unscoped-directive-matches-
        any. Everything else must agree, or the round trip can ship an
        advisory suppressed at one end and active at the other — the F8
        bug this whole mechanism exists to prevent.
        """
        from app.models.schemas import ValidationIssue

        def validator_suppresses(d_clause, i_clause):
            v = self.v
            v.issues = [ValidationIssue(
                severity="WARNING", code="Z79.01",
                category="unjustified_zcode", clause=i_clause)]
            v._advisory_suppression_corrections = []
            self._directive(d_clause)
            v._apply_validator_advisory_suppressions()
            return not v.issues

        for d_clause, i_clause in (("class_findings_modifier",
                                    "class_findings_modifier"),
                                   ("class_findings_modifier",
                                    "coverage_composition"),
                                   ("", "")):
            with self.subTest(directive=d_clause, carried=i_clause):
                engine_kills = not self._survives(d_clause, i_clause)
                self.assertEqual(engine_kills,
                                 validator_suppresses(d_clause, i_clause))

        # ...and the one documented divergence, asserted explicitly so it
        # is a decision on the record rather than an accident.
        self.assertTrue(self._survives("", "coverage_composition"),
                        "engine: unscoped directive must NOT reach a "
                        "clause-carrying finding")
        self.assertTrue(validator_suppresses("", "coverage_composition"),
                        "validator: unscoped directive still reaches a "
                        "tagged issue (the migration ramp)")


if __name__ == "__main__":
    unittest.main()
