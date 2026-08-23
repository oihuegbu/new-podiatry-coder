"""Directive section 4 — high-recall candidate generation vs HARD code verification.

Two properties are under test, and they are the two the directive states as rules
rather than as goals:

  * FUZZY/LEXICAL/SEMANTIC SIMILARITY MAY ONLY WIDEN THE CANDIDATE POOL. Retrieval
    rank, a score margin, and descriptor/notes token overlap can put a code in front
    of the resolver; none of them may select it. The tests below pin that by giving
    one candidate an overwhelming retrieval lead and asserting it is still not billed.

  * THE TIE POLICY runs in its stated order — eliminate, select only a unique
    satisfier, re-inspect ONLY the discriminating axes against the ORIGINAL DOCUMENT,
    release what the page uniquely entails, else ONE targeted provider query. The
    forbidden outcome is the generic coder queue, reached because candidates tied or
    two models disagreed.

Everything here is synthetic: the identifiers are not codes and the descriptors name
no real condition, product or region. The distinguishing axes are ordinary English
words in made-up descriptors, which is the point — the mechanism reads descriptor
grammar, never a term list.
"""
import unittest

from claude_coder import tiebreak
from claude_coder.data_access import MockSource
from claude_coder.models import (CandidateCode, ClinicalFact, Disposition, EvidenceSpan,
                                 FactKind, ResolutionMethod)
from claude_coder.resolution import resolve
from tests import shortlist_verdict as _sv


# --------------------------------------------------------------------------- helpers
def _spans(*texts):
    return [EvidenceSpan(text=t, start=0, end=len(t), anchored=True,
                         span_id=f"span-{i}") for i, t in enumerate(texts)]


def _fact(description, *evidence, attributes=None, kind=FactKind.PROCEDURE):
    return ClinicalFact(kind=kind, description=description,
                        attributes=dict(attributes or {}),
                        disposition=Disposition.PERFORMED, evidence=list(_spans(*evidence)),
                        confidence=0.99, fact_id="F1")


def _request(fact):
    from claude_coder.eligibility import (ClaimComponent, ClaimLineIntent,
                                          EligibilityState, RetrievalRequest,
                                          fact_snapshot_digest)
    intent = ClaimLineIntent(
        intent_id="test-F1", encounter_id="test", component=ClaimComponent.SERVICE,
        clinical_event_ids=[fact.fact_id], fact_kind=fact.kind.value,
        clinical_action=fact.description, attributes=dict(fact.attributes),
        date_of_service=None, billing_entity_id=None, source_span_ids=[],
        state=EligibilityState.ELIGIBLE_FOR_RETRIEVAL,
        fact_digest=fact_snapshot_digest(fact))
    return RetrievalRequest(intent, fact)


def _cand(code, descriptor, score):
    return CandidateCode(code=code, system="cpt", descriptor=descriptor, score=score,
                         source="retrieval")


def _source(*candidates):
    return MockSource(
        records={(c.code, "cpt"): {"active": True, "long_description": c.descriptor}
                for c in candidates},
        retrieval={("*", "cpt"): list(candidates)})


def _agreed(*span_ids):
    """A reconciliation in which the ORIGINAL PAGE confirmed these quotations."""
    from app.contracts.source_evidence import (ReconciliationStatus, SourceReconciliation,
                                               SpanReconciliation)
    return SourceReconciliation(spans=tuple(
        SpanReconciliation(span_id=s, status=ReconciliationStatus.AGREED, pages=(1,))
        for s in span_ids))


def _disagreed(*span_ids):
    from app.contracts.source_evidence import (ReconciliationStatus, SourceReconciliation,
                                               SpanReconciliation)
    return SourceReconciliation(spans=tuple(
        SpanReconciliation(span_id=s, status=ReconciliationStatus.DISAGREED, pages=(1,))
        for s in span_ids))


# Two candidates that both satisfy every documented axis and differ on ONE descriptor
# word. Equal retrieval scores, so nothing but the document can separate them.
POWERED = _cand("CAND_POWERED", "assembly service, powered technique", 0.90)
MANUAL = _cand("CAND_MANUAL", "assembly service, manual technique", 0.90)


# ------------------------------------------------------- the axis derivation itself
class DiscriminatingAxesTest(unittest.TestCase):

    def test_only_the_difference_between_descriptors_becomes_an_axis(self):
        (probe,) = tiebreak.discriminating_axes([POWERED, MANUAL])
        self.assertEqual(probe.axis, tiebreak.AXIS_DESCRIPTOR_TERM)
        # "assembly"/"service"/"technique" are shared, so they say nothing about which
        # candidate the record means and are not axes.
        self.assertEqual(probe.terms_by_code["CAND_POWERED"], ("powered",))
        self.assertEqual(probe.terms_by_code["CAND_MANUAL"], ("manual",))

    def test_laterality_is_reported_as_its_own_named_axis(self):
        left = _cand("CAND_L", "assembly service, left structure", 0.9)
        right = _cand("CAND_R", "assembly service, right structure", 0.9)
        axes = {p.axis: p for p in tiebreak.discriminating_axes([left, right])}
        self.assertIn(tiebreak.AXIS_LATERALITY, axes)
        self.assertEqual(axes[tiebreak.AXIS_LATERALITY].terms_by_code["CAND_L"], ("left",))
        # laterality words are not double-counted as generic descriptor terms
        self.assertNotIn(tiebreak.AXIS_DESCRIPTOR_TERM, axes)

    def test_identical_descriptors_have_no_discriminating_axis(self):
        a = _cand("CAND_A", "assembly service", 0.9)
        b = _cand("CAND_B", "assembly service", 0.9)
        self.assertEqual(tiebreak.discriminating_axes([a, b]), ())

    def test_classification_grammar_is_never_an_axis(self):
        """A record states a condition, never the residual bucket its code lives in, so
        'other'/'unspecified'/'not elsewhere classified' can never be proven from a page
        and must not become a question anyone could answer."""
        a = _cand("CAND_A", "assembly disorder, unspecified", 0.9)
        b = _cand("CAND_B", "other specified assembly disorder", 0.9)
        for probe in tiebreak.discriminating_axes([a, b]):
            for terms in probe.terms_by_code.values():
                self.assertNotIn("other", terms)
                self.assertNotIn("unspecified", terms)


# ------------------------------------------------- steps 3 and 4: narrow and release
class TieNarrowsAgainstTheOriginalDocumentTest(unittest.TestCase):

    def test_genuine_tie_narrows_on_the_discriminating_axis_and_releases(self):
        """Two candidates satisfy every documented axis. The ORIGINAL PAGE confirms a
        quotation that states one candidate's distinguishing word and not the other's,
        so exactly one becomes uniquely entailed and is released — deterministically,
        with the page named in the audit trail."""
        fact = _fact("assembly service",
                     "assembly service completed using the powered technique")
        line = resolve(_request(fact), _source(POWERED, MANUAL),
                       reconciliation=_agreed("span-0"))
        self.assertTrue(line.resolved, line.rationale)
        self.assertEqual(line.chosen.code, "CAND_POWERED")
        self.assertIs(line.method, ResolutionMethod.DETERMINISTIC)
        self.assertIn("tie narrowed against the original document", line.rationale)
        self.assertIn("original_page_reconciliation", line.rationale)
        self.assertIsNone(line.documentation_gap)

    def test_narrowing_reads_the_page_not_the_retrieval_order(self):
        """The SAME pool, with the retrieval scores reversed so the other candidate now
        leads by a wide margin. The document still decides, so the released code does
        not move — which is the whole point of separating recall from verification."""
        fact = _fact("assembly service",
                     "assembly service completed using the powered technique")
        leader = _cand("CAND_MANUAL", "assembly service, manual technique", 0.99)
        trailer = _cand("CAND_POWERED", "assembly service, powered technique", 0.61)
        line = resolve(_request(fact), _source(leader, trailer),
                       reconciliation=_agreed("span-0"))
        self.assertTrue(line.resolved, line.rationale)
        self.assertEqual(line.chosen.code, "CAND_POWERED")

    def test_unconfirmed_quotations_cannot_narrow_a_tie(self):
        """The page CONTRADICTS the quotation this event rests on. That is a
        source-integrity stop owned by the source-evidence control, not a documentation
        gap: no code is released, and no provider is asked to document a misreading."""
        fact = _fact("assembly service",
                     "assembly service completed using the powered technique")
        outcome = tiebreak.narrow(fact, [POWERED, MANUAL], _disagreed("span-0"))
        self.assertIsNone(outcome.winner)
        self.assertTrue(outcome.source_integrity)
        self.assertEqual(outcome.provider_question, "")

    def test_an_axis_words_cannot_settle_holds_the_line(self):
        """A bounded measurement interval is satisfied by a typed, unit-converted
        comparison, never by a word appearing on a page. Such an axis is reported so the
        query can name it, and — because release requires EVERY axis to be settled — it
        always holds, even when another axis is documented."""
        small = _cand("CAND_SMALL", "assembly service, powered, up to 4 cm", 0.9)
        large = _cand("CAND_LARGE", "assembly service, manual, more than 4 cm", 0.9)
        fact = _fact("assembly service", "assembly service by powered means")
        outcome = tiebreak.narrow(fact, [small, large], _agreed("span-0"))
        self.assertIsNone(outcome.winner)
        self.assertIn(tiebreak.AXIS_MEASUREMENT, outcome.unsettled)
        self.assertTrue(outcome.provider_question)


# ------------------------------------- step 5: a targeted query, never a coder queue
class TieThatStaysTiedTest(unittest.TestCase):

    def _tied_line(self):
        fact = _fact("assembly service", "assembly service was completed today")
        return fact, resolve(_request(fact), _source(POWERED, MANUAL),
                             reconciliation=_agreed("span-0"))

    def test_undocumented_distinguishing_fact_holds_for_a_coder_not_a_provider(self):
        """The record genuinely does not state the distinguishing fact, but that
        fact is only an untyped leftover descriptor token (issue #6 F9-R2-B, third
        pass: token cardinality does not establish clinical meaning) -- never a
        provider question, regardless of how clean the pair looks. The line still
        holds, with both candidates visible as alternatives; a CODER decides."""
        _fact_obj, line = self._tied_line()
        self.assertFalse(line.resolved)
        self.assertIs(line.method, ResolutionMethod.ABSTAINED)
        self.assertIsNone(line.documentation_gap, line.rationale)
        # both candidates stay visible as the rejected alternatives
        self.assertEqual({c.code for c in line.alternatives},
                         {"CAND_POWERED", "CAND_MANUAL"})

    def test_a_tie_routes_to_the_coder_queue_not_a_provider_question(self):
        """The mirror of the directive's forbidden outcome: an untyped tie is
        correctly a CODER decision (real candidates, no governed distinguishing
        fact genuinely absent), never manufactured into a provider question."""
        from claude_coder.autonomy import Destination
        from claude_coder.models import CodingResult
        from claude_coder.recommendations import build_recommendations
        _fact_obj, line = self._tied_line()
        result = CodingResult(encounter_id="enc", date_of_service="2026-03-14",
                              lines=[line])
        issues = {r["issue"] for r in build_recommendations(result)}
        self.assertIn("coder_review", issues)
        self.assertNotIn("documentation_gap", issues)

    def test_both_candidates_documented_is_still_a_tie(self):
        """When the record states BOTH distinguishing words, the document singles out
        nobody. Fail-closed: no winner -- and, issue #6 F9-R2-B, no provider
        question either: BOTH words are already documented, so nothing is missing
        for a provider to add. The record contradicting itself is a coder's
        candidate-mapping judgement, not a documentation gap."""
        fact = _fact("assembly service",
                     "assembly service used a powered step and a manual step")
        outcome = tiebreak.narrow(fact, [POWERED, MANUAL], _agreed("span-0"))
        self.assertIsNone(outcome.winner)
        self.assertEqual(outcome.provider_question, "")
        self.assertIn(tiebreak.AXIS_DESCRIPTOR_TERM, outcome.documented)

    def test_descriptors_that_differ_on_nothing_documentable_hold_without_a_query(self):
        """Identical descriptors leave no fact a provider could be asked for, so the
        line holds and says so rather than inventing an unanswerable question."""
        a = _cand("CAND_A", "assembly service", 0.9)
        b = _cand("CAND_B", "assembly service", 0.9)
        outcome = tiebreak.narrow(_fact("assembly service", "assembly service done"),
                                  [a, b], _agreed("span-0"))
        self.assertIsNone(outcome.winner)
        self.assertEqual(outcome.provider_question, "")
        self.assertIn("no differing axis", outcome.detail)


# -------------------------------------- issue #6 F9-R2: isolated contrast promotion
class IsolatedContrastPromotion(unittest.TestCase):
    """A provider question is permitted only for ONE isolated, clinically meaningful,
    unambiguous distinction -- never an open-ended bag of leftover descriptor tokens,
    and never a word two candidates share (which would not map each candidate to a
    distinct, unambiguous answer)."""

    def test_a_multiword_bag_never_reaches_the_provider(self):
        """One candidate's descriptor differs from the other's by SEVERAL words, not
        one -- that is retrieval noise, not a single fact a provider could confirm,
        and must never surface as a provider question."""
        wordy = _cand("CAND_WORDY",
                      "assembly service, powered heavy duty extended reach technique",
                      0.9)
        terse = _cand("CAND_TERSE", "assembly service, manual technique", 0.9)
        fact = _fact("assembly service", "assembly service was completed today")
        outcome = tiebreak.narrow(fact, [wordy, terse], _agreed("span-0"))
        self.assertIsNone(outcome.winner)
        self.assertEqual(outcome.provider_question, "",
                         "a multi-word descriptor bag must never become a provider "
                         "question")

    def test_a_multiword_bag_routes_to_review_not_the_provider(self):
        """The same shape, end to end: with nothing left that is a legitimate
        provider question, the line falls to the coder queue -- never PROVIDER_QUERY,
        and never silently unrouted either."""
        from claude_coder.autonomy import Destination, decide
        from claude_coder.models import CodingResult

        wordy = _cand("CAND_WORDY",
                      "assembly service, powered heavy duty extended reach technique",
                      0.9)
        terse = _cand("CAND_TERSE", "assembly service, manual technique", 0.9)
        fact = _fact("assembly service", "assembly service was completed today")
        line = resolve(_request(fact), _source(wordy, terse),
                       reconciliation=_agreed("span-0"))
        self.assertFalse(line.resolved, line.rationale)
        self.assertFalse(line.documentation_gap, line.documentation_gap)
        result = CodingResult(encounter_id="enc", date_of_service="2026-03-14",
                              lines=[line])
        verdict = decide(result)
        kinds = {r["destination"] for r in result.routing}
        self.assertIn(Destination.REVIEW.value, kinds, result.routing)
        self.assertNotIn(Destination.PROVIDER_QUERY.value, kinds, result.routing)

    def test_an_arbitrary_single_leftover_word_never_reaches_the_provider(self):
        """Codex F9-R2-B, third pass: token CARDINALITY does not establish clinical
        meaning. A clean, one-word-per-candidate, all-distinct AXIS_DESCRIPTOR_TERM
        contrast -- meaningless placeholder words, not a real qualifier -- must
        never reach the provider, exactly as a multi-word bag must not. Only a
        GOVERNED, typed axis (laterality, measurement) may ever be named."""
        alpha = _cand("CAND_ALPHA", "assembly service, alpha variant", 0.9)
        beta = _cand("CAND_BETA", "assembly service, beta variant", 0.9)
        fact = _fact("assembly service", "assembly service was completed today")
        outcome = tiebreak.narrow(fact, [alpha, beta], _agreed("span-0"))
        self.assertIsNone(outcome.winner)
        self.assertEqual(outcome.provider_question, "",
                         "an arbitrary single leftover word is not a governed "
                         "typed qualifier and must never become a provider question")

    def test_ordinary_category_words_that_look_like_an_approach_never_select_a_code(self):
        """Codex F9-R2-B, fourth pass: an earlier APPROACH axis treated
        `semantics._APPROACH_WORDS` as if it were a governed field -- it is a
        fixed six-word Python tuple with no versioned identity or semantic-role
        parse, and scanning for those words anywhere in a descriptor cannot tell
        their genuine clinical role from ordinary category wording. Reverted
        entirely: "open"/"percutaneous" now fall into the same untyped
        AXIS_DESCRIPTOR_TERM bucket as any other leftover word, and never
        select a candidate or reach a provider on their own -- confirmed
        end-to-end, not just at the tiebreak layer."""
        open_cand = _cand("CAND_OPEN", "assembly service, open technique", 0.9)
        percutaneous = _cand("CAND_PERCUTANEOUS",
                             "assembly service, percutaneous technique", 0.9)
        fact = _fact("assembly service", "assembly service was completed today")
        line = resolve(_request(fact), _source(open_cand, percutaneous),
                       reconciliation=_agreed("span-0"))
        self.assertFalse(line.resolved, line.rationale)
        self.assertIsNone(line.documentation_gap, line.rationale)
        self.assertIsNotNone(line.tie_record)

    def test_a_value_documented_by_two_candidates_never_reaches_the_provider(self):
        """Codex F9-R2-B, reproduced exactly: the record already states 'right', but
        TWO candidates both carry 'right' as their term (a third states 'left') --
        the value is genuinely documented, so asking the provider to document it
        again is asking for a fact already in the note. The real gap (which of the
        two 'right' candidates applies) is a coder's mapping question."""
        right_a = _cand("CAND_RIGHT_A", "assembly service, right structure, variant one",
                        0.9)
        right_b = _cand("CAND_RIGHT_B", "assembly service, right structure, variant two",
                        0.9)
        left = _cand("CAND_LEFT", "assembly service, left structure", 0.9)
        fact = _fact("assembly service", "assembly service performed on the right side")
        outcome = tiebreak.narrow(fact, [right_a, right_b, left], _agreed("span-0"))
        self.assertIsNone(outcome.winner)
        self.assertEqual(outcome.provider_question, "",
                         "laterality is already documented -- must never be re-asked")
        self.assertIn(tiebreak.AXIS_LATERALITY, outcome.documented)


# ------------------------------------------- similarity widens, it never verifies
class SimilarityMayOnlyWidenThePoolTest(unittest.TestCase):

    def test_a_large_retrieval_lead_does_not_confirm_a_code(self):
        """The forbidden shortcut, stated as a test: one candidate leads the other by a
        margin that used to close the decision outright. Nothing GOVERNED in the
        record distinguishes them, so nothing is billed -- held for a coder, not
        auto-released on the retrieval margin."""
        leader = _cand("CAND_POWERED", "assembly service, powered technique", 0.98)
        trailer = _cand("CAND_MANUAL", "assembly service, manual technique", 0.72)
        fact = _fact("assembly service", "assembly service was completed today")
        line = resolve(_request(fact), _source(leader, trailer),
                       reconciliation=_agreed("span-0"))
        self.assertFalse(line.resolved, line.rationale)
        self.assertIsNone(line.documentation_gap)
        self.assertIsNotNone(line.tie_record)

    def test_lexical_overlap_with_the_note_does_not_confirm_a_code(self):
        """Token overlap between a descriptor and the note is a RANK signal only. Here
        one descriptor repeats the note's own wording and still cannot be selected,
        because the word it shares is one BOTH candidates state."""
        wordy = _cand("CAND_WORDY", "assembly service of the structure, powered technique",
                      0.90)
        terse = _cand("CAND_TERSE", "assembly service of the structure, manual technique",
                      0.90)
        fact = _fact("assembly service of the structure",
                     "assembly service of the structure was completed today")
        line = resolve(_request(fact), _source(wordy, terse),
                       reconciliation=_agreed("span-0"))
        self.assertFalse(line.resolved, line.rationale)

    def test_widening_still_works_a_recalled_candidate_can_be_released(self):
        """The other half of the rule: retrieval must still SUPPLY candidates. A lone
        recalled candidate that contradicts nothing is released, so tightening selection
        did not turn the resolver off."""
        only = _cand("CAND_ONLY", "assembly service", 0.90)
        fact = _fact("assembly service", "assembly service was completed today")
        line = resolve(_request(fact), _source(only), reconciliation=_agreed("span-0"))
        self.assertTrue(line.resolved, line.rationale)
        self.assertEqual(line.chosen.code, "CAND_ONLY")

    def test_a_documented_axis_still_selects_uniquely(self):
        """Step 2 is untouched: a candidate that POSITIVELY satisfies a documented axis
        the others do not is still selected automatically, with no page re-inspection
        needed. That is a documented-axis decision, not a similarity one."""
        sided = _cand("CAND_RIGHT", "assembly service, right structure", 0.90)
        plain = _cand("CAND_PLAIN", "assembly service, unspecified structure", 0.90)
        fact = _fact("assembly service", "assembly service on the right structure",
                     attributes={"laterality": "right"})
        line = resolve(_request(fact), _source(sided, plain),
                       reconciliation=_agreed("span-0"))
        self.assertTrue(line.resolved, line.rationale)
        self.assertEqual(line.chosen.code, "CAND_RIGHT")
        self.assertNotIn("tie narrowed", line.rationale)


# ------------------------------------------------- the whole encounter, in order
NO_PICK = '{"choice": 0, "confidence": 0.0, "reason": "declined"}'


class TiePolicyEndToEndTest(unittest.TestCase):
    """Trace one real tie through `code_encounter`, to prove the steps run in the
    directive's ORDER and that no later stage quietly re-decides a tie a model was not
    allowed to decide."""

    NOTE = ("Procedure: assembly service was completed today. "
            "Assessment: condition alpha.")

    FACTS = ('{"facts": [{"kind": "supply", "description": "assembly service",'
             ' "attributes": {"performer_id": "actor-1", "billing_entity_id": "actor-1"},'
             ' "disposition": "performed_today", "negated": false,'
             ' "evidence": ["assembly service was completed today"], "confidence": 0.99,'
             ' "axis_confidence": {"occurrence": 0.99, "action": 0.99, "evidence": 0.99,'
             ' "temporal": 0.99, "performer": 0.99, "relationship": 0.99}}]}')

    def _run(self, arbitrate, audit_repository=None):
        from claude_coder.pipeline import code_encounter
        from claude_coder.provenance import NullAuditRepository
        supply_a = CandidateCode("CAND_POWERED", "hcpcs",
                                 "assembly item, powered technique", 0.90, "retrieval")
        supply_b = CandidateCode("CAND_MANUAL", "hcpcs",
                                 "assembly item, manual technique", 0.90, "retrieval")
        source = MockSource(
            records={("CAND_POWERED", "hcpcs"): {"active": True},
                     ("CAND_MANUAL", "hcpcs"): {"active": True}},
            retrieval={("*", "hcpcs"): [supply_a, supply_b]})
        return code_encounter(
            "enc-tie", self.NOTE, "2026-03-14", source=source,
            extract_llm=lambda s, u: self.FACTS, arbitrate_llm=arbitrate,
            audit_repository=(audit_repository or NullAuditRepository()),
            billing_context={"billing_entity_id": "actor-1",
                             "participants": [{"id": "actor-1", "type": "person",
                                               "roles": ["performer"]}]})

    def test_a_tie_is_never_handed_to_a_bounded_model_pick(self):
        """A supply does not go through propose-then-verify, so before this change a tie
        fell straight through to arbitration — a single model picking the billed code.
        The tie policy now answers first, and because it produced a targeted question the
        arbitrator is never consulted at all: the stub records whether it was called, and
        it must not have been."""
        called = []

        def arbitrate(system, user):
            called.append(user)
            return '{"choice":1,"confidence":0.99,"reason":"picked"}'

        result = self._run(arbitrate)
        (line,) = [ln for ln in result.lines if ln.fact.kind is FactKind.SUPPLY]
        self.assertFalse(line.resolved, line.rationale)
        self.assertIsNotNone(line.tie_record, line.rationale)
        self.assertEqual(called, [], "arbitration decided a tie the document must own")

    def test_the_tie_decision_is_written_to_the_audit_trail(self):
        """A tie is a claim-affecting decision, so "why not the other candidate?" has to
        be answerable from the audit chain, not only from a rationale string."""
        from claude_coder.provenance import AuditRepository

        class Recording(AuditRepository):
            def __init__(self):
                self.records = []

            def append(self, encounter_id, kind, record):
                self.records.append((kind, record))
                return "h%d" % len(self.records)

        repo = Recording()
        result = self._run(lambda s, u: NO_PICK, audit_repository=repo)
        ties = [r for kind, r in repo.records if kind == "code_tie_resolution"]
        self.assertEqual(len(ties), 1, [k for k, _ in repo.records])
        (tie,) = ties
        self.assertFalse(tie["released"])
        self.assertEqual(tie["fact_id"], "F1")
        self.assertEqual([a["axis"] for a in tie["axes"]], ["descriptor_term"])
        self.assertEqual(sorted(tie["unsettled_axes"]), ["descriptor_term"])
        # issue #6 F9-R2-B, third pass: an untyped leftover-token axis is never a
        # provider question, regardless of shape -- this tie is a coder decision.
        self.assertEqual(tie["provider_question"], "")
        self.assertFalse(tie["source_integrity"])
        self.assertEqual(result.destination.value, "REVIEW")

    def test_a_failed_tie_audit_write_holds_the_encounter_instead_of_crashing(self):
        """The failure path of the audit record itself. Durable audit is enforced, so a
        store that cannot accept the tie record must produce the typed, RETRYABLE system
        hold the router knows how to dispatch — not an exception escaping the pipeline,
        and not a claim quietly released with no record of why the alternatives lost."""
        from claude_coder.models import Outcome, Verdict
        from claude_coder.provenance import AuditRepository

        class Failing(AuditRepository):
            def append(self, encounter_id, kind, record):
                if kind == "code_tie_resolution":
                    raise RuntimeError("audit store unavailable")
                return "h"

        result = self._run(lambda s, u: NO_PICK, audit_repository=Failing())
        self.assertNotEqual(result.verdict, Verdict.AUTO_READY)
        (gate,) = result.gates
        self.assertTrue(gate.name.startswith("code_tie_audit_persistence:"), gate.name)
        self.assertIs(gate.outcome, Outcome.UNKNOWN)
        self.assertTrue(gate.retryable)
        self.assertEqual(result.lines, [])

    def test_the_encounter_routes_an_untyped_tie_to_the_coder_not_the_provider(self):
        """Issue #6 F9-R2-B, third pass: an untyped leftover-token tie (powered vs.
        manual, no governed axis distinguishes them) is a real, decided tie -- but
        never a provider question. It routes to the coder queue."""
        result = self._run(lambda s, u: NO_PICK)
        from claude_coder.autonomy import Destination
        routed = [r for r in result.routing if r.get("fact_id")]
        self.assertTrue(routed, result.routing)
        self.assertEqual({r["destination"] for r in routed},
                         {Destination.REVIEW.value}, result.routing)
        issues = {r["issue"] for r in result.recommendations}
        self.assertIn("coder_review", issues)
        self.assertNotIn("documentation_gap", issues)




# =============================================================================
# Codex F8-R1 — MODEL AGREEMENT ON ONE CANDIDATE IS NOT CODE-SELECTION UNIQUENESS
# =============================================================================
# The reviewer's reproduction: a documented fact states BOTH "component one" and
# "component two"; candidate A's authoritative descriptor requires component one and
# candidate B's requires component two, so the documentation independently entails BOTH.
# The selector picked A, the independent corroborator confirmed A, and A auto-released —
# even though B was equally entailed and nothing ever eliminated it.
#
# The rule these tests pin: a code auto-releases only when EXACTLY ONE shortlisted
# candidate is still entailed and every other one carries a NAMED elimination. Several
# still entailed is a TIE, and a tie is settled by the ORIGINAL DOCUMENT or asked about —
# the same `tiebreak` machinery the deterministic path uses, never a second mechanism.

SYN_A = _cand("SYN_A", "assembly service including component one", 0.90)
SYN_B = _cand("SYN_B", "assembly service including component two", 0.90)


def _judge(entails, prefers=None, declare=True):
    """A judging model under the shortlist contract.

    `entails(descriptor)` decides which options this model finds entailed; `prefers` picks
    the one it would code among them. Every other option gets a NAMED elimination, which is
    what the contract asks for. `declare=False` reproduces a model that answers with a bare
    pick and says nothing about the rest — the answer that CANNOT establish uniqueness, and
    must therefore never release on its own.
    """
    return _sv.judge(entails=entails, prefer=prefers, declare=declare,
                     reason="shortlist verdict")


def _pinned(fn, provider):
    """Declare the provider identity, so an agreement can be credited as INDEPENDENT and
    the released line carries the autonomy-eligible VERIFIED method."""
    from claude_coder import verify as _verify
    return _verify.declare_model_profile(fn, provider=provider)


_ONE = "component one"
_TWO = "component two"


class SelectionUniquenessTest(unittest.TestCase):

    BOTH_DOCUMENTED = ("assembly service performed, including component one and "
                       "component two")
    ONE_DOCUMENTED = "assembly service performed, including component one"

    def _resolve(self, evidence, primary, second):
        fact = _fact("assembly service", evidence)
        return resolve(_request(fact), _source(SYN_A, SYN_B),
                       llm=_pinned(primary, "provider-a"),
                       corroborate=_pinned(second, "provider-b"),
                       reconciliation=_agreed("span-0"))

    # ---- the reviewer's exact reproduction ------------------------------------------
    def test_two_entailed_candidates_do_not_release_even_when_both_models_agree(self):
        """Codex F8-R1, reproduced end to end through `resolve`: both models judge BOTH
        descriptors entailed and both would code SYN_A. Before this change that released
        SYN_A on agreement alone. It must now hold, with the OTHER candidate named."""
        both = lambda d: True                                   # noqa: E731
        line = self._resolve(self.BOTH_DOCUMENTED,
                             _judge(both, prefers=lambda d: _ONE in d),
                             _judge(both, prefers=lambda d: _ONE in d))
        self.assertIsNone(line.chosen, line.rationale)
        self.assertIs(line.method, ResolutionMethod.ABSTAINED)
        self.assertEqual(line.tie_record["still_entailed"], ["SYN_A", "SYN_B"])
        self.assertEqual(line.tie_record["selected"], "SYN_A")
        # BOTH_DOCUMENTED states BOTH distinguishing words ("component one" AND
        # "component two") -- issue #6 F9-R2-B: an axis the record already documents
        # must never become a provider question ("please document" a fact that is
        # already there). The record contradicting itself between two candidates is a
        # coder's candidate-mapping judgement, not a provider documentation gap.
        self.assertIsNone(line.documentation_gap, line.rationale)

    def test_the_reviewers_reproduction_holds_the_whole_encounter(self):
        """The same reproduction through the real entrypoint, because the finding is
        about what gets BILLED. Nothing may reach the claim, and the encounter must be
        held -- not auto-released. Routed to the CODER (issue #6 F9-R2-B), not the
        provider: BOTH_DOCUMENTED states both distinguishing words, so nothing is
        genuinely undocumented -- the record contradicting itself between two
        candidates is a candidate-mapping judgement, not a documentation gap."""
        from claude_coder.autonomy import Destination
        from claude_coder.pipeline import code_encounter
        from claude_coder.provenance import NullAuditRepository
        note = ("Procedure: " + self.BOTH_DOCUMENTED + ". Assessment: condition alpha.")
        facts = ('{"facts": [{"kind": "procedure", "description": "assembly service",'
                 ' "attributes": {"performer_id": "actor-1",'
                 ' "billing_entity_id": "actor-1"},'
                 ' "disposition": "performed_today", "negated": false,'
                 ' "evidence": ["' + self.BOTH_DOCUMENTED + '"], "confidence": 0.99,'
                 ' "axis_confidence": {"occurrence": 0.99, "action": 0.99,'
                 ' "evidence": 0.99, "temporal": 0.99, "performer": 0.99,'
                 ' "relationship": 0.99}}]}')
        both = lambda d: True                                   # noqa: E731
        result = code_encounter(
            "enc-f8r1", note, "2026-03-14", source=_source(SYN_A, SYN_B),
            extract_llm=lambda s, u: facts,
            arbitrate_llm=lambda s, u: NO_PICK,
            verify_llm=_pinned(_judge(both, prefers=lambda d: _ONE in d), "provider-a"),
            corroborate_llm=_pinned(_judge(both, prefers=lambda d: _ONE in d),
                                    "provider-b"),
            audit_repository=NullAuditRepository(),
            billing_context={"billing_entity_id": "actor-1",
                             "participants": [{"id": "actor-1", "type": "person",
                                               "roles": ["performer"]}]})
        self.assertEqual([ln.chosen.code for ln in result.billable_lines], [])
        (line,) = [ln for ln in result.lines if ln.fact.kind is FactKind.PROCEDURE]
        self.assertIsNone(line.chosen, line.rationale)
        self.assertIsNone(line.documentation_gap, line.rationale)
        self.assertEqual(result.destination, Destination.REVIEW)
        issues = {r["issue"] for r in result.recommendations}
        self.assertIn("coder_review", issues)
        self.assertNotIn("documentation_gap", issues)

    # ---- the common path must not regress -------------------------------------------
    def test_a_single_entailed_candidate_still_auto_releases(self):
        """The overwhelmingly common case: only one shortlisted descriptor survives, both
        models say so, and the other is eliminated with a NAMED reason. That still
        releases automatically — tightening uniqueness must not turn the coder off."""
        only_one = lambda d: _ONE in d                          # noqa: E731
        line = self._resolve(self.ONE_DOCUMENTED,
                             _judge(only_one), _judge(only_one))
        self.assertTrue(line.resolved, line.rationale)
        self.assertEqual(line.chosen.code, "SYN_A")
        self.assertIs(line.method, ResolutionMethod.VERIFIED)
        self.assertIsNone(line.documentation_gap)
        # the release states WHY the alternative is gone, in the audit record
        self.assertEqual(line.tie_record["still_entailed"], ["SYN_A"])
        self.assertTrue(line.tie_record["eliminated"]["SYN_B"])
        self.assertEqual(line.tie_record["released_code"], "SYN_A")

    def test_a_tie_the_page_settles_releases_through_the_tie_policy(self):
        """Both models are loose and call BOTH descriptors entailed, but the ORIGINAL
        DOCUMENT states only one candidate's distinguishing word. The tie policy's step 3
        narrowing resolves it, so the line still releases — via the document, not via the
        agreement, and the audit record shows which one settled it."""
        both = lambda d: True                                   # noqa: E731
        line = self._resolve(self.ONE_DOCUMENTED,
                             _judge(both, prefers=lambda d: _ONE in d),
                             _judge(both, prefers=lambda d: _ONE in d))
        self.assertTrue(line.resolved, line.rationale)
        self.assertEqual(line.chosen.code, "SYN_A")
        self.assertIs(line.method, ResolutionMethod.VERIFIED)
        self.assertIn("narrowed against the original document", line.rationale)
        self.assertEqual(line.tie_record["winner"], "SYN_A")
        self.assertEqual(line.tie_record["still_entailed"], ["SYN_A", "SYN_B"])

    # ---- what "eliminated" has to mean ----------------------------------------------
    def test_a_bare_pick_cannot_establish_uniqueness_but_a_named_elimination_can(self):
        """The same shortlist, judged two ways. When both models NAME why the
        alternative is out AND the original document itself documents only the winner's
        distinguishing term (so the elimination is independently GROUNDED, not merely
        asserted), the line releases. When they answer with a bare pick and say nothing
        about the alternative, silence is not an elimination and the line holds — the
        fail-closed default, with no separate code path."""
        named = _judge(lambda d: _ONE in d)
        released = self._resolve(self.ONE_DOCUMENTED, named, _judge(lambda d: _ONE in d))
        self.assertTrue(released.resolved, released.rationale)
        self.assertEqual(released.chosen.code, "SYN_A")

        bare = _judge(lambda d: True, prefers=lambda d: _ONE in d, declare=False)
        held = self._resolve(self.BOTH_DOCUMENTED, bare,
                             _judge(lambda d: True, prefers=lambda d: _ONE in d,
                                    declare=False))
        self.assertIsNone(held.chosen, held.rationale)
        self.assertEqual(held.tie_record["still_entailed"], ["SYN_A", "SYN_B"])
        self.assertFalse(
            held.tie_record["judgements"][0]["declared_shortlist_verdict"])

    # ---- Codex F8-R1, round-9 re-review: a NAMED reason is not itself grounds ---------
    def test_a_false_named_elimination_the_document_contradicts_does_not_release(self):
        """The reviewer's exact round-9 counterexample: the documentation states BOTH
        components (so SYN_B is genuinely, independently entailed), but two SEPARATE
        judging models both falsely name SYN_B as eliminated. Before this fix, any
        non-empty reason string from every model was enough to drop a candidate from the
        standing set — 'model agreement as proof, only in a richer JSON shape.' The
        elimination must now be independently confirmed against the original document
        (the same `tiebreak.narrow` proof the tie policy already uses) before it can
        remove a candidate, and the document here confirms the OPPOSITE of what both
        models claimed, so SYN_B must stay standing."""
        only_one = lambda d: _ONE in d                          # noqa: E731
        line = self._resolve(self.BOTH_DOCUMENTED, _judge(only_one), _judge(only_one))
        self.assertIsNone(line.chosen, line.rationale)
        self.assertEqual(line.tie_record["still_entailed"], ["SYN_A", "SYN_B"])
        # BOTH_DOCUMENTED states both distinguishing words -- issue #6 F9-R2-B: an
        # axis the record already documents is never a provider question.
        self.assertIsNone(line.documentation_gap, line.rationale)
        self.assertNotIn("SYN_B", line.tie_record["eliminated"])

    def test_a_named_elimination_from_a_disagreed_span_does_not_release(self):
        """Codex F8-R1, exact-SHA re-review: the raw-evidence-text fallback used to
        trigger for ANY reason `tiebreak.narrow` gave up, including a reconciliation
        that explicitly DISAGREED with this fact's only anchored quotation -- laundering
        rejected source evidence into document-confirmed grounds for an elimination,
        the same unsafe direction as the original defect one layer deeper. A
        reconciliation that was actually consulted and did not confirm the span must
        refuse the elimination outright, never fall back to that same rejected text."""
        only_one = lambda d: _ONE in d                          # noqa: E731
        fact = _fact("assembly service", self.ONE_DOCUMENTED)
        line = resolve(_request(fact), _source(SYN_A, SYN_B),
                       llm=_pinned(_judge(only_one), "provider-a"),
                       corroborate=_pinned(_judge(only_one), "provider-b"),
                       reconciliation=_disagreed("span-0"))
        self.assertIsNone(line.chosen, line.rationale)
        self.assertEqual(line.tie_record["still_entailed"], ["SYN_A", "SYN_B"])
        self.assertNotIn("SYN_B", line.tie_record["eliminated"])

    def test_a_named_elimination_from_an_unlocatable_quotation_does_not_release(self):
        """Codex F8-R1, exact-SHA re-review, second pass: the previous fix refused the
        fallback only when this fact's evidence happened to be ANCHORED and a supplied
        reconciliation rejected it. An UNANCHORED, un-locatable quotation -- one that
        never even resolved to a span reconciliation could check -- fell through to
        the same unsafe raw-text trust whenever a reconciliation object existed at all.
        A quotation that cannot be located is not a weaker case than one that was
        checked and disagreed with; both must refuse, and the refusal must depend only
        on whether a reconciliation channel was supplied for this call -- never on
        whether this particular fact's own evidence happened to anchor."""
        only_one = lambda d: _ONE in d                          # noqa: E731
        fact = ClinicalFact(
            kind=FactKind.PROCEDURE, description="assembly service", attributes={},
            disposition=Disposition.PERFORMED,
            evidence=[EvidenceSpan(text=self.ONE_DOCUMENTED, start=0,
                                   end=len(self.ONE_DOCUMENTED), anchored=False,
                                   span_id="")],
            confidence=0.99, fact_id="F1")
        line = resolve(_request(fact), _source(SYN_A, SYN_B),
                       llm=_pinned(_judge(only_one), "provider-a"),
                       corroborate=_pinned(_judge(only_one), "provider-b"),
                       reconciliation=_agreed("span-0"))
        self.assertIsNone(line.chosen, line.rationale)
        self.assertEqual(line.tie_record["still_entailed"], ["SYN_A", "SYN_B"])
        self.assertNotIn("SYN_B", line.tie_record["eliminated"])

    def test_the_corroborator_evaluates_the_shortlist_not_only_the_pick(self):
        """The corroborator's own view of the OTHER candidates has to count. Here it
        agrees with the pick — the only thing it used to be asked — while independently
        finding the alternative entailed too. That disagreement leaves the alternative
        STANDING, so the line holds instead of releasing on the agreement."""
        line = self._resolve(self.BOTH_DOCUMENTED,
                             _judge(lambda d: _ONE in d),
                             _judge(lambda d: True, prefers=lambda d: _ONE in d))
        self.assertIsNone(line.chosen, line.rationale)
        self.assertEqual(line.tie_record["still_entailed"], ["SYN_A", "SYN_B"])
        # BOTH_DOCUMENTED states both distinguishing words -- issue #6 F9-R2-B: an
        # axis the record already documents is never a provider question.
        self.assertIsNone(line.documentation_gap)


if __name__ == "__main__":
    unittest.main()
