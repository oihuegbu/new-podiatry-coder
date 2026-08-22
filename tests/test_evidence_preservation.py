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
        # Codex F8-R5: a well-formed 40-hex SHA / sha256:<64-hex> digest -- not
        # "deadbeef", which `AuthorityBinding.problems()` now correctly rejects
        # as not shaped like a real commit SHA at all.
        import run as entrypoint
        sha = "a" * 40
        digest = "sha256:" + "b" * 64
        old_sha = os.environ.get("APPLICATION_COMMIT_SHA")
        old_digest = os.environ.get("IMAGE_DIGEST")
        try:
            os.environ["APPLICATION_COMMIT_SHA"] = sha
            os.environ["IMAGE_DIGEST"] = digest
            source = MockSource()
            result = type("R", (), {"certificate": None})()
            binding = entrypoint.authority_binding(result, source)
            self.assertEqual(binding.application_commit_sha, sha)
            self.assertEqual(binding.image_digest, digest)
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


class ApplicationIdentityFormatValidation(unittest.TestCase):
    """Codex F8-R5: a SUPPLIED application identity must be well-formed; absence
    is not (yet) a release blocker (see AuthorityBinding's own docstring on why),
    but an arbitrary string self-attesting as one is always rejected."""

    def test_well_formed_commit_sha_and_digest_have_no_problems(self):
        # Other AuthorityBinding fields (data_fingerprint etc.) have their own,
        # unrelated problems() checks -- irrelevant here, so only assert that
        # NEITHER new check fires for a well-formed value.
        binding = AuthorityBinding(application_commit_sha="a" * 40,
                                   image_digest="sha256:" + "b" * 64)
        problems = binding.problems()
        self.assertFalse(any("application_commit_sha" in p for p in problems))
        self.assertFalse(any("image_digest" in p for p in problems))

    def test_absent_identity_is_not_a_problem(self):
        binding = AuthorityBinding()
        problems = binding.problems()
        self.assertFalse(any("application_commit_sha" in p for p in problems))
        self.assertFalse(any("image_digest" in p for p in problems))

    def test_malformed_commit_sha_is_a_problem(self):
        binding = AuthorityBinding(application_commit_sha="deadbeef")
        problems = binding.problems()
        self.assertTrue(any("application_commit_sha" in p for p in problems))

    def test_malformed_image_digest_is_a_problem(self):
        binding = AuthorityBinding(image_digest="sha256:beefdead")
        problems = binding.problems()
        self.assertTrue(any("image_digest" in p for p in problems))

    def test_malformed_identity_blocks_release_via_authority_problems(self):
        """The consumer-side re-derivation (`release_blockers`) already extends
        `self.authority.problems()` -- confirming the new format check actually
        reaches release, not just the unit-level `problems()` call."""
        bundle = bundle_from_coding_result(
            type("R", (), {"lines": [], "encounter_id": "e", "date_of_service": None,
                           "certificate": None})(),
            source_document=SourceDocument(), context=EncounterContext(),
            authority=AuthorityBinding(application_commit_sha="deadbeef"))
        self.assertTrue(any("application_commit_sha" in b
                            for b in bundle.release_blockers()))


class DeterministicIndexPathCarriesEligibilityAudit(unittest.TestCase):
    """Codex F8-R2 (P2): a deterministic authoritative-index hit (`_take()`'s clean,
    no-verification-needed path in `resolution.resolve`) must still carry a
    `candidate_eligibility` audit record -- it skips the RECALL pool's eligibility
    FILTER by design (these are exact term->code hits, not a semantic guess), but
    skipping the filter is not license to skip the audit trail too."""

    def test_icd_index_deterministic_hit_carries_candidate_eligibility(self):
        from claude_coder.resolution import resolve
        from tests.test_measurement import _request

        source = MockSource(
            records={("DX1", "icd10"): {"long_description": "a documented condition",
                                        "active": True}},
            index={"documented condition": {"DX1"}})
        fact = ClinicalFact(FactKind.DIAGNOSIS, "documented condition", fact_id="fx")
        line = resolve(_request(fact), source)
        self.assertEqual(line.chosen.code if line.chosen else None, "DX1")
        self.assertIsNotNone(line.candidate_eligibility,
                             "a deterministic index hit must still carry an "
                             "eligibility audit record, not an unexplained None")
        self.assertTrue(line.candidate_eligibility)
        self.assertEqual(line.candidate_eligibility[0]["code"], "DX1")
        self.assertTrue(line.candidate_eligibility[0]["eligible"])


class AdvisoryProcedureSynonymRecallExpansion(unittest.TestCase):
    """Codex F8-R2 (escalated, product-owner-narrowed acceptance): the advisory
    (LLM-generated, round-trip-validated) procedure-synonym index widens RECALL
    queries only -- it never settles identity, excludes a candidate, or authorizes
    release. `MockSource.retrieve` here is keyed ONLY by the expanded phrase, not
    the fact's own raw description or a wildcard, so the candidate is findable
    ONLY if the expansion query actually ran."""

    def test_a_unique_advisory_match_widens_recall_and_is_recorded(self):
        from claude_coder.resolution import resolve
        from tests.test_measurement import _request

        source = MockSource(
            records={("PROC_X", "cpt"): {"long_description": "Excision, lesion",
                                         "active": True}},
            retrieval={("removal of skin lesion", "cpt"):
                      [CandidateCode("PROC_X", "cpt", "Excision, lesion", 0.9)]},
            procedure_concept_lookup={"excision of lesion": {
                "term": "excision of lesion", "candidates": ["PROC_X"],
                "method": "retrieval_consistency_validated", "unique": True,
                "expansions": ["removal of skin lesion"], "source_identity": {"v": 1}}})
        fact = ClinicalFact(FactKind.PROCEDURE, "excision of lesion", fact_id="fx")
        line = resolve(_request(fact), source)
        self.assertEqual(line.chosen.code if line.chosen else None, "PROC_X",
                         "the advisory expansion query must actually widen recall")
        self.assertIsNotNone(line.advisory_terminology)
        entry = line.advisory_terminology[0]
        self.assertEqual(entry["term"], "excision of lesion")
        self.assertEqual(entry["method"], "retrieval_consistency_validated")
        self.assertIn("removal of skin lesion", entry["expansions"])

    def test_no_advisory_match_leaves_the_field_honestly_none(self):
        from claude_coder.resolution import resolve
        from tests.test_measurement import _request

        source = MockSource(
            records={("PROC_X", "cpt"): {"long_description": "Excision, lesion",
                                         "active": True}},
            retrieval={("*", "cpt"): [CandidateCode("PROC_X", "cpt",
                                                     "Excision, lesion", 0.9)]})
        fact = ClinicalFact(FactKind.PROCEDURE, "excision of lesion", fact_id="fx")
        line = resolve(_request(fact), source)
        self.assertIsNone(line.advisory_terminology)

    def test_advisory_expansion_never_runs_for_a_diagnosis_fact(self):
        """ICD-10 has no advisory index -- `_advisory_procedure_expansions` is
        gated on system (cpt/hcpcs), never called for a diagnosis at all."""
        from claude_coder.resolution import resolve
        from tests.test_measurement import _request

        source = MockSource(
            records={("DX1", "icd10"): {"long_description": "a documented condition",
                                        "active": True}},
            retrieval={("*", "icd10"): [CandidateCode("DX1", "icd10",
                                                       "a documented condition", 0.9)]},
            concept_lookup={"a documented condition": {
                "term": "a documented condition", "candidates": ["DX1"],
                "method": "retrieval_consistency_validated", "unique": True,
                "expansions": ["should never be used"], "source_identity": {}}})
        fact = ClinicalFact(FactKind.DIAGNOSIS, "a documented condition", fact_id="fx")
        line = resolve(_request(fact), source)
        self.assertIsNone(line.advisory_terminology)


if __name__ == "__main__":
    unittest.main()
