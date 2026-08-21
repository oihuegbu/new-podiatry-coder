"""Evidence preservation + run identity (issue #6 items 8/9).

`service_intents` (composition's grouping) and `candidate_eligibility` (semantic
eligibility's per-candidate audit) must survive into the bundle's audit surface
regardless of whether the encounter released -- and `AuthorityBinding` carries a
build-time application identity when one was baked into the image. Agnostic --
synthetic codes + stub LLMs, no API key."""
import os
import unittest

from app.contracts.claim_bundle import (AuthorityBinding, EncounterContext,
                                        SourceDocument, bundle_from_coding_result)
from claude_coder import semantic_eligibility as semelig
from claude_coder.data_access import MockSource
from claude_coder.models import CandidateCode, ClinicalFact, FactKind
from claude_coder.pipeline import code_encounter
from claude_coder.provenance import NullAuditRepository
from tests import shortlist_verdict as _sv


class EligibilityReportReasons(unittest.TestCase):
    def test_eligible_candidate_has_no_reason(self):
        source = MockSource(records={("Z1", "cpt"): {"long_description": "a service"}})
        fact = ClinicalFact(FactKind.PROCEDURE, "did a thing")
        cand = CandidateCode("Z1", "cpt", "a service", 0.5)
        report = semelig.eligibility_report([fact], [cand], source, None)
        self.assertEqual(report, [{"code": "Z1", "system": "cpt",
                                   "eligible": True, "reason": None}])

    def test_ineligible_candidate_carries_a_reason(self):
        source = MockSource(records={("Z1", "cpt"): {"long_description": "a service",
                                                      "active": False}})
        fact = ClinicalFact(FactKind.PROCEDURE, "did a thing")
        cand = CandidateCode("Z1", "cpt", "a service", 0.5)
        report = semelig.eligibility_report([fact], [cand], source, "2026-01-01")
        self.assertFalse(report[0]["eligible"])
        self.assertIn("active", report[0]["reason"])

    def test_reason_matches_eligible_partition_decision(self):
        """`eligible()`/`eligible_partition()` and `eligibility_report()` must never
        disagree -- both read the same `_ineligibility_reason`. Two candidates
        (not one) so `eligible_partition`'s own never-empty-from-non-empty
        safety net (issue #6 item 4) never masks the comparison: with a second,
        genuinely eligible candidate present, the ineligible one is truly excluded
        from `kept`, not kept only because excluding it would empty the pool."""
        source = MockSource(records={("Z1", "cpt"): {"long_description": "a service",
                                                      "active": False},
                                     ("Z2", "cpt"): {"long_description": "a service",
                                                     "active": True}})
        fact = ClinicalFact(FactKind.PROCEDURE, "did a thing")
        blocked = CandidateCode("Z1", "cpt", "a service", 0.5)
        active = CandidateCode("Z2", "cpt", "a service", 0.4)
        kept = semelig.eligible_partition([fact], [blocked, active], source, "2026-01-01")
        report = semelig.eligibility_report([fact], [blocked, active], source, "2026-01-01")
        by_code = {r["code"]: r["eligible"] for r in report}
        self.assertEqual(blocked in kept, by_code["Z1"])
        self.assertEqual(active in kept, by_code["Z2"])
        self.assertNotIn(blocked, kept)
        self.assertIn(active, kept)


_FACTS_TWO_SERVICES = (
    '{"facts":['
    '{"kind":"procedure","description":"first thing",'
    '"attributes":{"performer_id":"actor-1","billing_entity_id":"actor-1"},'
    '"disposition":"performed_today","negated":false,'
    '"evidence":["first thing performed"],"confidence":0.99,"fact_id":"F1"},'
    '{"kind":"procedure","description":"second thing",'
    '"attributes":{"performer_id":"actor-1","billing_entity_id":"actor-1"},'
    '"disposition":"ordered","negated":false,'
    '"evidence":["second thing was ordered"],"confidence":0.99,"fact_id":"F2"}'
    ']}')
_NOTE = "first thing performed today. second thing was ordered."
_sel = _sv.judge(pick=1, reason="x")


def _src():
    return MockSource(
        records={("PROC_A", "cpt"): {"active": True}, ("PROC_B", "cpt"): {"active": True}},
        retrieval={("*", "cpt"): [CandidateCode("PROC_A", "cpt", "First thing, each", 0.9)]})


class ServiceIntentsAndCandidateEligibilitySurviveIntoTheBundle(unittest.TestCase):
    def test_service_intents_present_even_when_one_line_is_blocked(self):
        """F2 is documented `disposition: ordered`, never performed -- NON_CLAIM_
        EVIDENCE, diverted before retrieval entirely. F1 resolves cleanly.
        `service_intents` is computed over ALL facts regardless, so it is
        non-empty here whether or not every event billed."""
        r = code_encounter("e", _NOTE, "2026-03-14", source=_src(),
                           extract_llm=lambda s, u: _FACTS_TWO_SERVICES, verify_llm=_sel,
                           corroborate_llm=_sel, audit_repository=NullAuditRepository())
        self.assertTrue(r.service_intents)
        ids_covered = {eid for si in r.service_intents for eid in si["component_event_ids"]}
        self.assertIn("F1", ids_covered)
        self.assertIn("F2", ids_covered)

    def test_candidate_eligibility_present_on_a_line_that_reached_retrieval(self):
        r = code_encounter("e", _NOTE, "2026-03-14", source=_src(),
                           extract_llm=lambda s, u: _FACTS_TWO_SERVICES, verify_llm=_sel,
                           corroborate_llm=_sel, audit_repository=NullAuditRepository())
        resolved = [ln for ln in r.lines if ln.fact.fact_id == "F1"]
        self.assertTrue(resolved)
        self.assertIsNotNone(resolved[0].candidate_eligibility)
        self.assertTrue(resolved[0].candidate_eligibility)

    def test_candidate_eligibility_absent_on_a_line_diverted_before_retrieval(self):
        """F2 never reaches `resolve()` at all -- honestly None, a different thing
        from 'ran and excluded nothing', never guessed as an empty list."""
        r = code_encounter("e", _NOTE, "2026-03-14", source=_src(),
                           extract_llm=lambda s, u: _FACTS_TWO_SERVICES, verify_llm=_sel,
                           corroborate_llm=_sel, audit_repository=NullAuditRepository())
        diverted = [ln for ln in r.lines if ln.fact.fact_id == "F2"]
        self.assertTrue(diverted)
        self.assertIsNone(diverted[0].candidate_eligibility)

    def test_bundle_audit_surfaces_both_fields(self):
        r = code_encounter("e", _NOTE, "2026-03-14", source=_src(),
                           extract_llm=lambda s, u: _FACTS_TWO_SERVICES, verify_llm=_sel,
                           corroborate_llm=_sel, audit_repository=NullAuditRepository())
        bundle = bundle_from_coding_result(
            r,
            source_document=SourceDocument(),
            context=EncounterContext(),
            authority=AuthorityBinding())
        self.assertTrue(bundle.audit.service_intents)
        self.assertTrue(bundle.audit.candidate_eligibility)
        f1_record = next(c for c in bundle.audit.candidate_eligibility
                         if c["fact_id"] == "F1")
        self.assertTrue(f1_record["candidates"])


class ApplicationCommitShaFromEnvironment(unittest.TestCase):
    def test_authority_binding_reads_env_when_present(self):
        import run as entrypoint
        old_sha = os.environ.get("APPLICATION_COMMIT_SHA")
        old_digest = os.environ.get("IMAGE_DIGEST")
        try:
            os.environ["APPLICATION_COMMIT_SHA"] = "deadbeef"
            os.environ["IMAGE_DIGEST"] = "sha256:beefdead"
            source = MockSource()
            result = type("R", (), {"certificate": None})()
            binding = entrypoint.authority_binding(result, source)
            self.assertEqual(binding.application_commit_sha, "deadbeef")
            self.assertEqual(binding.image_digest, "sha256:beefdead")
        finally:
            for key, old in (("APPLICATION_COMMIT_SHA", old_sha),
                             ("IMAGE_DIGEST", old_digest)):
                if old is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = old

    def test_authority_binding_defaults_to_empty(self):
        import run as entrypoint
        old_sha = os.environ.pop("APPLICATION_COMMIT_SHA", None)
        old_digest = os.environ.pop("IMAGE_DIGEST", None)
        try:
            source = MockSource()
            result = type("R", (), {"certificate": None})()
            binding = entrypoint.authority_binding(result, source)
            self.assertEqual(binding.application_commit_sha, "")
            self.assertEqual(binding.image_digest, "")
        finally:
            if old_sha is not None:
                os.environ["APPLICATION_COMMIT_SHA"] = old_sha
            if old_digest is not None:
                os.environ["IMAGE_DIGEST"] = old_digest


if __name__ == "__main__":
    unittest.main()
