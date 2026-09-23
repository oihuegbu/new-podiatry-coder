"""Governed coding conventions (`claude_coder.conventions`) -- product-owner
decision 2026-09-22, option 2: config with a cited authority may authorize ONE
value on ONE descriptor axis the record is silent on. Synthetic vocabulary
throughout; the real pack is checked only structurally (never for content)."""
import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from claude_coder import conventions, resolution, tiebreak, verify
from claude_coder import requirement as req
from claude_coder.models import CandidateCode, ClinicalFact, EvidenceSpan, FactKind
from app.contracts.source_evidence import (ReconciliationStatus, SourceReconciliation,
                                           SpanReconciliation)


def _pack(tmp, conventions_list):
    path = Path(tmp) / "pack.json"
    path.write_text(json.dumps({"version": "test", "conventions": conventions_list}))
    conventions.load_pack.cache_clear()
    return str(path)


SYNTHETIC = [{
    "id": "structure-alpha-reattachment-is-secondary",
    "enabled": True,
    "axis": "sequence_qualifier",
    "value": "secondary",
    "applies_when": {"fact_kinds": ["procedure"],
                     "candidate_descriptor_regex": r"\bstructure alpha\b",
                     "evidence_regex": r"\breattach(?:ed|ment)?\b",
                     "absent_regex": r"\bsevered\b"},
    "authority": "synthetic authority citation",
    "verification_status": "test",
}]


def _fact(text, kind=FactKind.PROCEDURE, anchored=True):
    return ClinicalFact(kind=kind, description="assembly of structure alpha",
                        evidence=[EvidenceSpan(text=text, anchored=anchored, span_id="s1")],
                        confidence=0.9, fact_id="F1")


class PackStructureGuard(unittest.TestCase):
    """The REAL pack: every entry cites an authority and names an axis/value;
    the pack never names a medical code (the same rule validator_rules.json is
    held to)."""

    def test_real_pack_is_well_formed_and_code_free(self):
        conventions.load_pack.cache_clear()
        text = conventions.PACK_PATH.read_text()
        self.assertNotRegex(text, r"\b\d{5}\b", "a five-digit code literal")
        self.assertNotRegex(text, r"\b[A-Z]\d{2}\.\d", "an ICD-10-style code literal")
        self.assertNotRegex(text, r"\b[A-Z]\d{4}\b", "a HCPCS-style code literal")
        for rule in conventions.load_pack():
            for key in ("id", "axis", "value", "authority", "verification_status"):
                self.assertTrue(str(rule.get(key) or "").strip(), (rule.get("id"), key))
            self.assertTrue((rule.get("applies_when") or {}).get("evidence_regex"), rule.get("id"))


class AuthorizedValueTest(unittest.TestCase):

    def test_matches_only_within_scope(self):
        with tempfile.TemporaryDirectory() as tmp:
            pack = _pack(tmp, SYNTHETIC)
            fact = _fact("Structure alpha was reattached with two anchors.")
            match = conventions.authorized_value(
                fact, "sequence_qualifier", None,
                candidate_descriptor="Repair, secondary, structure alpha", pack_path=pack)
            self.assertIsNotNone(match)
            self.assertEqual(match.value, "secondary")
            self.assertEqual(match.convention_id, "structure-alpha-reattachment-is-secondary")
            self.assertIn("synthetic authority citation", match.describe())
            # wrong axis / wrong candidate family / absent-term present / other kind / no evidence
            self.assertIsNone(conventions.authorized_value(
                fact, "laterality", None, candidate_descriptor="structure alpha", pack_path=pack))
            self.assertIsNone(conventions.authorized_value(
                fact, "sequence_qualifier", None, candidate_descriptor="Repair, structure beta",
                pack_path=pack))
            self.assertIsNone(conventions.authorized_value(
                _fact("Structure alpha was severed and reattached."), "sequence_qualifier", None,
                candidate_descriptor="structure alpha", pack_path=pack))
            self.assertIsNone(conventions.authorized_value(
                _fact("Structure alpha was reattached.", kind=FactKind.DIAGNOSIS),
                "sequence_qualifier", None, candidate_descriptor="structure alpha", pack_path=pack))
            self.assertIsNone(conventions.authorized_value(
                _fact("Structure alpha was reattached.", anchored=False), "sequence_qualifier",
                None, candidate_descriptor="structure alpha", pack_path=pack))
        conventions.load_pack.cache_clear()

    def test_absent_pack_is_a_noop(self):
        conventions.load_pack.cache_clear()
        self.assertIsNone(conventions.authorized_value(
            _fact("Structure alpha was reattached."), "sequence_qualifier", None,
            candidate_descriptor="structure alpha", pack_path="/nonexistent/pack.json"))
        conventions.load_pack.cache_clear()


class ConventionResolvesTheWinnersOwnAxisTest(unittest.TestCase):
    """The designated-note shape (F4): the record documents a reattachment, the
    only fitting descriptor requires 'secondary', and the record never says it.
    Without a convention the line holds; with the governed convention it
    releases, naming the convention in its rationale and audit record."""

    WINNER = CandidateCode(code="CAND_SEC", system="cpt", score=0.6, source="retrieval",
                           descriptor="Repair, secondary, structure alpha, with or without graft")
    LOSER = CandidateCode(code="CAND_PRI", system="cpt", score=0.4, source="retrieval",
                          descriptor="Repair, primary, severed structure alpha")

    def _line(self, pack_path):
        fact = _fact("Structure alpha was reattached to the site with two anchors.")
        candidates = [self.WINNER, self.LOSER]
        requirements = req.compile_requirements(candidates)
        judgement = verify.Judgement(
            chosen=self.WINNER, entailed=(self.WINNER.code,),
            eliminated={self.LOSER.code: "no severed structure alpha is documented"},
            declared=True)
        reconciliation = SourceReconciliation(spans=(
            SpanReconciliation(span_id="s1", status=ReconciliationStatus.AGREED),))
        with patch.object(conventions, "PACK_PATH", Path(pack_path)):
            conventions.load_pack.cache_clear()
            try:
                return resolution._entailed_line(
                    fact, self.WINNER, candidates, "single-evaluator entailment",
                    requirements=requirements, judgements=[judgement],
                    reconciliation=reconciliation, coverage=None)
            finally:
                conventions.load_pack.cache_clear()

    def test_holds_without_a_convention(self):
        line = self._line("/nonexistent/pack.json")
        self.assertIsNone(line.chosen)
        self.assertIn("sequence_qualifier", line.documentation_gap or "")

    def test_releases_with_the_governed_convention_named(self):
        with tempfile.TemporaryDirectory() as tmp:
            line = self._line(_pack(tmp, SYNTHETIC))
        self.assertEqual(line.chosen.code if line.chosen else None, "CAND_SEC")
        self.assertEqual(line.method, resolution.ResolutionMethod.VERIFIED)
        self.assertIn("structure-alpha-reattachment-is-secondary", line.rationale)
        self.assertIn("synthetic authority citation", line.rationale)
        self.assertIn("convention_authorized", line.tie_record or {})

    def test_a_convention_never_asserts_a_value_the_descriptor_does_not_state(self):
        """The convention says 'secondary'; a candidate whose descriptor states
        'primary' is not confirmed by it."""
        with tempfile.TemporaryDirectory() as tmp:
            pack = _pack(tmp, SYNTHETIC)
            fact = _fact("Structure alpha was reattached to the site.")
            candidates = [self.WINNER, self.LOSER]
            requirements = req.compile_requirements(candidates)
            judgement = verify.Judgement(chosen=self.LOSER, entailed=(self.LOSER.code,),
                                         eliminated={}, declared=True)
            with patch.object(conventions, "PACK_PATH", Path(pack)):
                conventions.load_pack.cache_clear()
                confirmed, _ = resolution._chosen_own_requirements_confirmed(
                    fact, self.LOSER, requirements, [judgement], None, None)
                conventions.load_pack.cache_clear()
        self.assertFalse(confirmed)


class ConventionSettlesATieTest(unittest.TestCase):

    def test_narrow_settles_a_silent_axis_by_convention(self):
        win = CandidateCode(code="CAND_SEC", system="cpt", score=0.6,
                            descriptor="Repair, secondary, structure alpha")
        lose = CandidateCode(code="CAND_PRI", system="cpt", score=0.4,
                             descriptor="Repair, primary, structure alpha")
        fact = _fact("Structure alpha was reattached to the site.")
        with tempfile.TemporaryDirectory() as tmp:
            pack = _pack(tmp, SYNTHETIC)
            with patch.object(conventions, "PACK_PATH", Path(pack)):
                conventions.load_pack.cache_clear()
                tie = tiebreak.narrow(fact, [win, lose], None)
                conventions.load_pack.cache_clear()
        self.assertEqual(tie.winner.code if tie.winner else None, "CAND_SEC")
        self.assertIn("governed coding convention", tie.detail)

    def test_a_documented_value_is_never_overridden(self):
        win = CandidateCode(code="CAND_SEC", system="cpt", score=0.6,
                            descriptor="Repair, secondary, structure alpha")
        lose = CandidateCode(code="CAND_PRI", system="cpt", score=0.4,
                             descriptor="Repair, primary, structure alpha")
        fact = _fact("A primary repair: structure alpha was reattached to the site.")
        with tempfile.TemporaryDirectory() as tmp:
            pack = _pack(tmp, SYNTHETIC)
            with patch.object(conventions, "PACK_PATH", Path(pack)):
                conventions.load_pack.cache_clear()
                tie = tiebreak.narrow(fact, [win, lose], None)
                conventions.load_pack.cache_clear()
        self.assertEqual(tie.winner.code if tie.winner else None, "CAND_PRI")
        self.assertNotIn("governed coding convention", tie.detail)
