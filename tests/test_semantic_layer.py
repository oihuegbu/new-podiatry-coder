"""Regression tests for the compiled semantic layer (issue #6, compiled-semantic-
layer plan item 1/3): `AuthoritativeSource.component_relationships`/`semantic_class`/
axis-aware `concept_lookup`, `ontology.parse_descriptor`'s action/anatomy split, and
`semantics.compiled_record`'s assembly.

Deterministic unit coverage uses `MockSource`/pre-populated caches (SYNTHETIC
identifiers only, per this suite's own convention). One smoke test exercises the
real `AuthoritativeSource` against whatever authoritative data is actually loaded,
selecting its subject DYNAMICALLY by semantic property rather than asserting a
literal code value, per the same test philosophy this plan's own spec states.
"""
import unittest

from claude_coder.data_access import AuthoritativeSource, MockSource
from claude_coder import ontology, semantics


class OntologyActionAnatomySplit(unittest.TestCase):
    """`parse_descriptor`'s new `action_tokens`/`anatomy_tokens` fields."""

    def test_a_short_action_target_descriptor_splits_cleanly(self):
        feats = ontology.parse_descriptor("Widgetectomy, sprocket")
        self.assertEqual(feats.action_tokens, {"widgetectomy"})
        self.assertEqual(feats.anatomy_tokens, {"sprocket"})

    def test_a_multi_clause_descriptor_still_splits_on_the_first_punctuation(self):
        feats = ontology.parse_descriptor("Repair, secondary, widget, with or without graft")
        self.assertEqual(feats.action_tokens, {"repair"})
        self.assertEqual(feats.anatomy_tokens, {"secondary", "widget", "graft"})

    def test_a_descriptor_with_no_punctuation_is_honestly_unsplit(self):
        feats = ontology.parse_descriptor("Unlisted widget procedure")
        self.assertEqual(feats.action_tokens, set())
        self.assertEqual(feats.anatomy_tokens, set())
        # ...but core_tokens (the pre-existing, unchanged field) still carries them.
        self.assertIn("widget", feats.core_tokens)

    def test_a_long_paragraph_style_descriptor_is_honestly_unsplit(self):
        """The E/M-style regression: real punctuation exists, but the descriptor is
        long enough that the "Action, Target" convention almost certainly does not
        hold, so guessing would manufacture noise rather than real anatomy tokens."""
        long_descriptor = (
            "Office or other outpatient visit for the evaluation and management of "
            "an established patient, which requires a medically appropriate history "
            "and examination and a low level of decision making, twenty minutes")
        self.assertGreater(len(long_descriptor.split()), ontology._SPLITTABLE_MAX_WORDS)
        feats = ontology.parse_descriptor(long_descriptor)
        self.assertEqual(feats.action_tokens, set())
        self.assertEqual(feats.anatomy_tokens, set())

    def test_a_short_descriptor_at_the_word_count_boundary_still_splits(self):
        short = "Action, target"
        self.assertLessEqual(len(short.split()), ontology._SPLITTABLE_MAX_WORDS)
        feats = ontology.parse_descriptor(short)
        self.assertEqual(feats.action_tokens, {"action"})
        self.assertEqual(feats.anatomy_tokens, {"target"})


class ComponentRelationshipsAccessor(unittest.TestCase):
    """`AuthoritativeSource.component_relationships` / `MockSource`'s mirror."""

    def test_mock_source_gathers_from_code_and_ancestor_category(self):
        source = MockSource(component_relationships={
            "X10": {"useAdditionalCode": {"Y99"}},   # category-level note
            "X102": {"codeAlso": {"Z01"}},            # specific-code note
        })
        rel = source.component_relationships("X10.2", "icd10")
        self.assertEqual(rel.get("useAdditionalCode"), {"Y99"},
                         "a category-level note must apply to its child code")
        self.assertEqual(rel.get("codeAlso"), {"Z01"})

    def test_non_icd10_system_returns_nothing(self):
        source = MockSource(component_relationships={"X10": {"excludes1": {"Y99"}}})
        self.assertEqual(source.component_relationships("X10", "cpt"), {})

    def test_excludes1_is_folded_into_the_same_dict_as_the_other_relationships(self):
        source = MockSource(excludes1={"X10": {"Y99"}},
                            component_relationships={"X10": {"codeAlso": {"Z01"}}})
        rel = source.component_relationships("X10.2", "icd10")
        self.assertEqual(rel, {"excludes1": {"Y99"}, "codeAlso": {"Z01"}})

    def test_a_code_with_no_relationships_at_all_returns_empty(self):
        source = MockSource()
        self.assertEqual(source.component_relationships("X99", "icd10"), {})


class UmlsCrosswalkAccessor(unittest.TestCase):
    """`AuthoritativeSource.umls_crosswalk_entry` / `MockSource`'s mirror (issue #6,
    UMLS 2026AA ingestion). Recall/audit-only accessor -- these tests only prove it
    reads and degrades correctly, never that anything consumes it for selection
    (nothing does)."""

    def test_mock_source_returns_the_configured_entry(self):
        source = MockSource(umls_crosswalk={
            "99213": {"cuis": ["C0000001"], "matched_snomed_concept_ids": ["12345"]},
        })
        self.assertEqual(source.umls_crosswalk_entry("99213", "cpt"),
                         {"cuis": ["C0000001"], "matched_snomed_concept_ids": ["12345"]})

    def test_mock_source_unconfigured_code_returns_empty_not_an_error(self):
        source = MockSource(umls_crosswalk={"99213": {"cuis": ["C0000001"]}})
        self.assertEqual(source.umls_crosswalk_entry("99999", "cpt"), {})

    def test_mock_source_non_cpt_hcpcs_system_returns_empty(self):
        source = MockSource(umls_crosswalk={"A00.0": {"cuis": ["C0000001"]}})
        self.assertEqual(source.umls_crosswalk_entry("A00.0", "icd10"), {})

    def test_mock_source_hcpcs_system_is_recognized(self):
        source = MockSource(umls_crosswalk={
            "J1234": {"cuis": ["C0000002"], "matched_snomed_concept_ids": ["67890"]},
        })
        self.assertEqual(source.umls_crosswalk_entry("J1234", "hcpcs"),
                         {"cuis": ["C0000002"], "matched_snomed_concept_ids": ["67890"]})

    def test_a_source_with_no_crosswalk_configured_degrades_to_empty(self):
        self.assertEqual(MockSource().umls_crosswalk_entry("99213", "cpt"), {})

    def test_real_source_never_raises_and_returns_a_well_formed_result(self):
        """The real, un-mocked AuthoritativeSource. umls_cpt_snomed_crosswalk.json is
        a UMLS-licensed derived artifact that is never committed to git (gitignored,
        same as snomed_concept_terms.json/snomed_icd10_map.json) -- most checkouts
        genuinely lack it, proving the fail-safe-empty path for real rather than via
        a bare/raising test double, but a box that already built it (this repo's own
        EC2 deployment, after tools/build_umls_crosswalk.py has run) legitimately has
        it. Either way this must never raise, and a made-up code the crosswalk almost
        certainly never contains must return {} regardless of which state holds."""
        source = AuthoritativeSource()
        result = source.umls_crosswalk_entry("ZZ_NOT_A_REAL_CODE_ZZ", "cpt")
        self.assertEqual(result, {})


class UmlsCandidatesAccessor(unittest.TestCase):
    """`AuthoritativeSource.umls_candidates` (issue #6 F9-R7 item 2, and its
    F9-R7-C/D remediation -- Codex's independent re-review of 92f4596). A REAL
    (not mocked) source, with `_umls_term_index` set directly to a controlled
    synthetic artifact -- the declared-source loading/binding machinery this
    bypasses is already covered by `UmlsCrosswalkAccessor` above and by every
    other `declared_document_snapshot` consumer; this class is only testing
    `umls_candidates`'s own matching/reachability logic. Synthetic codes/CUIs,
    per this suite's convention."""

    def _source(self, index: dict) -> AuthoritativeSource:
        source = AuthoritativeSource.__new__(AuthoritativeSource)
        source._umls_term_index = index
        source._umls_scan_index = None
        source._db = None
        source.lookup = lambda code, system: {"long_description": f"Synthetic {code} descriptor"}
        return source

    def _index(self) -> dict:
        return {
            "release": "TEST2026", "mrconso_sha256": "deadbeef", "generated": "now",
            "term_to_cuis": {"synthetic assembly service": ["C0000001"]},
            "cui_to_atoms": {"C0000001": [
                {"sab": "CPT", "code": "99999", "term": "Synthetic assembly service",
                 "tty": "PT"}]},
        }

    def test_whole_string_equality_still_matches(self):
        candidates = self._source(self._index()).umls_candidates(
            ["synthetic assembly service"], "cpt", None)
        self.assertEqual([c.code for c in candidates], ["99999"])

    def test_the_term_embedded_mid_sentence_also_matches(self):
        """issue #6 F9-R7-C: the exact reported failure mode -- `resolution.py`
        passes full structured queries and full evidence sentences, not bare
        term-index keys. Reproduced directly before this fix: this returned []."""
        candidates = self._source(self._index()).umls_candidates(
            ["the synthetic assembly service was completed today"], "cpt", None)
        self.assertEqual([c.code for c in candidates], ["99999"])

    def test_an_unmatched_phrase_contributes_no_seed(self):
        candidates = self._source(self._index()).umls_candidates(
            ["an entirely unrelated procedure"], "cpt", None)
        self.assertEqual(candidates, [])

    def _cross_system_index(self) -> dict:
        """A longer CPT-only term ("synthetic assembly service") and a shorter
        HCPCS-only term ("synthetic assembly") that is a PREFIX of it -- the
        exact shape issue #6 F9-R8-B reproduced: one shared scan index let the
        longer, CPT-only match consume the span before the shorter, valid
        HCPCS term was ever tried."""
        return {
            "release": "TEST2026", "mrconso_sha256": "deadbeef", "generated": "now",
            "term_to_cuis": {
                "synthetic assembly service": ["C_CPT_ONLY"],
                "synthetic assembly": ["C_HCPCS_ONLY"],
            },
            "cui_to_atoms": {
                "C_CPT_ONLY": [{"sab": "CPT", "code": "99999",
                               "term": "Synthetic assembly service", "tty": "PT"}],
                "C_HCPCS_ONLY": [{"sab": "HCPCS", "code": "A1234",
                                 "term": "Synthetic assembly", "tty": "PT"}],
            },
        }

    def test_a_cpt_only_term_never_masks_a_shorter_hcpcs_match(self):
        """issue #6 F9-R8-B, Codex's independent re-review of 5ab4a13: querying
        HCPCS for the full phrase must still find the HCPCS-only term, even
        though a longer CPT-only term containing it as a prefix also matches."""
        candidates = self._source(self._cross_system_index()).umls_candidates(
            ["the synthetic assembly service was completed today"], "hcpcs", None)
        self.assertEqual([c.code for c in candidates], ["A1234"])

    def test_an_hcpcs_only_term_never_masks_a_longer_cpt_match(self):
        """The mirror case: querying CPT for the same phrase must still find
        the CPT-only term -- per-system scanning must not weaken the EXISTING
        longest-match behavior within the system that actually owns it."""
        candidates = self._source(self._cross_system_index()).umls_candidates(
            ["the synthetic assembly service was completed today"], "cpt", None)
        self.assertEqual([c.code for c in candidates], ["99999"])

    def test_overlapping_terms_within_one_system_still_prefer_the_longest_match(self):
        """Regression guard: per-system partitioning must not disturb the
        EXISTING longest-match-wins behavior when both overlapping terms
        belong to the SAME system."""
        index = {
            "release": "TEST2026", "mrconso_sha256": "deadbeef", "generated": "now",
            "term_to_cuis": {
                "synthetic assembly service": ["C1"],
                "synthetic assembly": ["C2"],
            },
            "cui_to_atoms": {
                "C1": [{"sab": "CPT", "code": "11111",
                       "term": "Synthetic assembly service", "tty": "PT"}],
                "C2": [{"sab": "CPT", "code": "22222",
                       "term": "Synthetic assembly", "tty": "PT"}],
            },
        }
        candidates = self._source(index).umls_candidates(
            ["the synthetic assembly service was completed today"], "cpt", None)
        self.assertEqual([c.code for c in candidates], ["11111"],
                         "the longer term must win, not both/the shorter one")

    def test_lineage_is_carried_not_discarded(self):
        """issue #6 F9-R7-D: SAB/code/term/TTY/CUI/matched-phrase lineage, plus
        the bound artifact identity, must survive into `CandidateCode.authority`
        -- not just the first match, every one that supports the code."""
        index = self._index()
        index["term_to_cuis"]["a synonym phrase"] = ["C0000002"]
        index["cui_to_atoms"]["C0000002"] = [
            {"sab": "HCPT", "code": "99999", "term": "A synonym phrase", "tty": "SY"}]
        candidates = self._source(index).umls_candidates(
            ["synthetic assembly service", "a synonym phrase"], "cpt", None)
        self.assertEqual(len(candidates), 1)
        authority = candidates[0].authority
        self.assertEqual(authority["release"], "TEST2026")
        self.assertEqual(authority["mrconso_sha256"], "deadbeef")
        cuis = {m["cui"] for m in authority["matches"]}
        self.assertEqual(cuis, {"C0000001", "C0000002"})
        sabs = {m["atom"]["sab"] for m in authority["matches"]}
        self.assertEqual(sabs, {"CPT", "HCPT"})

    def test_a_umls_seed_reaches_verification_despite_eight_higher_scored_rivals(self):
        """issue #6 F9-R7-C: a fixed low UMLS score (0.3, `data_access.py`) must
        not be crowded out of `resolution._propose_then_verify`'s VERIFY_K=8
        shortlist by higher-scoring RAG candidates competing on raw score --
        reproduced directly before the reserved-lane fix (`_MIN_UMLS_SLOTS`,
        `resolution.py`): the UMLS candidate never appeared in `alternatives`
        with 8 higher-scored rivals present."""
        from claude_coder.data_access import MockSource
        from claude_coder.models import CandidateCode, ClinicalFact, EvidenceSpan, FactKind
        from claude_coder import resolution
        from claude_coder.eligibility import (ClaimComponent, ClaimLineIntent,
                                              EligibilityState, RetrievalRequest,
                                              fact_snapshot_digest)

        rivals = [CandidateCode(f"RIVAL{i}", "cpt", f"unrelated service {i}", 0.9,
                                "retrieval") for i in range(8)]
        umls_cand = CandidateCode("99999", "cpt", "Synthetic assembly service", 0.3,
                                  "umls_recall")

        class ScanSource(MockSource):
            def retrieve(self, description, system, top_k=20):
                return rivals if system == "cpt" else []

            def umls_candidates(self, terms, system, dos):
                for t in terms:
                    if "synthetic assembly service" in t.lower():
                        return [umls_cand]
                return []

        records = {(c.code, "cpt"): {"long_description": c.descriptor, "active": True}
                  for c in rivals}
        records[("99999", "cpt")] = {"long_description": "Synthetic assembly service",
                                     "active": True}
        src = ScanSource(records=records)
        text = "the synthetic assembly service was performed today"
        fact = ClinicalFact(FactKind.PROCEDURE, text,
                            evidence=[EvidenceSpan(text, anchored=True, span_id="s1")],
                            confidence=0.99, fact_id="f1")
        intent = ClaimLineIntent(
            intent_id="t-f1", encounter_id="t", component=ClaimComponent.SERVICE,
            clinical_event_ids=["f1"], fact_kind=fact.kind.value,
            clinical_action=fact.description, attributes=dict(fact.attributes),
            date_of_service=None, billing_entity_id=None, source_span_ids=[],
            state=EligibilityState.ELIGIBLE_FOR_RETRIEVAL, fact_digest=fact_snapshot_digest(fact))
        line = resolution.resolve(RetrievalRequest(intent, fact), src,
                                  llm=lambda system, user: '{"candidates": []}')
        self.assertIn("99999", [c.code for c in line.alternatives],
                      "a UMLS-matched candidate must reach verification even with "
                      "8 higher-scored rivals in the pool")

    def test_cross_source_lineage_is_merged_not_replaced(self):
        """issue #6 F9-R8-C, Codex's independent re-review of 5ab4a13:
        `resolution.py`'s cross-source union (`best[c.code] = c` on a higher
        score) REPLACED the loser's authority outright -- when a descriptor-
        retrieval hit (0.9) and a UMLS hit (0.3) proposed the SAME code, the
        winning candidate kept only its own lineage and the other source's
        was discarded before it ever reached the bundle. Reproduced directly
        before the fix: the merged candidate's `authority` held only
        `{"index": "rag-hybrid", ...}`, never the UMLS `matches`."""
        from claude_coder.data_access import MockSource
        from claude_coder.models import CandidateCode, ClinicalFact, EvidenceSpan, FactKind
        from claude_coder import resolution
        from claude_coder.eligibility import (ClaimComponent, ClaimLineIntent,
                                              EligibilityState, RetrievalRequest,
                                              fact_snapshot_digest)

        rag_cand = CandidateCode("99999", "cpt", "Synthetic assembly service", 0.9,
                                 "retrieval", authority={"index": "rag-hybrid", "system": "cpt"})
        umls_cand = CandidateCode("99999", "cpt", "Synthetic assembly service", 0.3,
                                  "umls_recall", authority={"matches": [{"cui": "C0000001"}]})

        class DualSource(MockSource):
            def retrieve(self, description, system, top_k=20):
                return [rag_cand] if system == "cpt" else []

            def umls_candidates(self, terms, system, dos):
                return [umls_cand] if system == "cpt" else []

        src = DualSource(records={("99999", "cpt"): {
            "long_description": "Synthetic assembly service", "active": True}})
        text = "the synthetic assembly service was performed today"
        fact = ClinicalFact(FactKind.PROCEDURE, text,
                            evidence=[EvidenceSpan(text, anchored=True, span_id="s1")],
                            confidence=0.99, fact_id="f1")
        intent = ClaimLineIntent(
            intent_id="t-f1", encounter_id="t", component=ClaimComponent.SERVICE,
            clinical_event_ids=["f1"], fact_kind=fact.kind.value,
            clinical_action=fact.description, attributes=dict(fact.attributes),
            date_of_service=None, billing_entity_id=None, source_span_ids=[],
            state=EligibilityState.ELIGIBLE_FOR_RETRIEVAL, fact_digest=fact_snapshot_digest(fact))
        line = resolution.resolve(RetrievalRequest(intent, fact), src,
                                  llm=lambda system, user: '{"candidates": []}')
        merged = next(c for c in line.alternatives if c.code == "99999")
        self.assertEqual(merged.source, "retrieval")   # higher score still wins identity
        self.assertEqual(set(merged.authority["sources"]), {"retrieval", "umls_recall"})
        self.assertEqual(merged.authority["retrieval"],
                         {"index": "rag-hybrid", "system": "cpt"})
        self.assertEqual(merged.authority["umls_recall"]["matches"],
                         [{"cui": "C0000001"}])

    def test_a_merged_candidate_still_gets_its_umls_reserved_lane_slot(self):
        """issue #6 F9-R9-C, Codex's independent re-review of 6ff2761:
        `_merge_candidate` keeps the higher-scored candidate's scalar
        `.source`, so checking `c.source == "umls_recall"` alone misses a
        merged candidate whenever retrieval outscored UMLS for it -- even
        though its `authority["sources"]` still names `umls_recall`. Eight
        higher-scored rivals plus a target proposed by BOTH retrieval (below
        the rivals) and UMLS: reproduced directly before this fix, the merged
        target's `.source` became "retrieval", so `_MIN_UMLS_SLOTS`'s reserved
        lane never recognized it and it was crowded out before verification."""
        from claude_coder.data_access import MockSource
        from claude_coder.models import CandidateCode, ClinicalFact, EvidenceSpan, FactKind
        from claude_coder import resolution
        from claude_coder.eligibility import (ClaimComponent, ClaimLineIntent,
                                              EligibilityState, RetrievalRequest,
                                              fact_snapshot_digest)

        rivals = [CandidateCode(f"RIVAL{i}", "cpt", f"unrelated service {i}", 0.9,
                                "retrieval") for i in range(8)]
        target_rag = CandidateCode("TARGET", "cpt", "Synthetic target service", 0.2,
                                   "retrieval", authority={"index": "rag-hybrid"})
        target_umls = CandidateCode("TARGET", "cpt", "Synthetic target service", 0.3,
                                    "umls_recall", authority={"matches": [{"cui": "C1"}]})

        class DualSource(MockSource):
            def retrieve(self, description, system, top_k=20):
                return rivals + [target_rag] if system == "cpt" else []

            def umls_candidates(self, terms, system, dos):
                return [target_umls] if system == "cpt" else []

        records = {(c.code, "cpt"): {"long_description": c.descriptor, "active": True}
                  for c in rivals}
        records[("TARGET", "cpt")] = {"long_description": "Synthetic target service",
                                      "active": True}
        src = DualSource(records=records)
        text = "the synthetic target service was performed today"
        fact = ClinicalFact(FactKind.PROCEDURE, text,
                            evidence=[EvidenceSpan(text, anchored=True, span_id="s1")],
                            confidence=0.99, fact_id="f1")
        intent = ClaimLineIntent(
            intent_id="t-f1", encounter_id="t", component=ClaimComponent.SERVICE,
            clinical_event_ids=["f1"], fact_kind=fact.kind.value,
            clinical_action=fact.description, attributes=dict(fact.attributes),
            date_of_service=None, billing_entity_id=None, source_span_ids=[],
            state=EligibilityState.ELIGIBLE_FOR_RETRIEVAL, fact_digest=fact_snapshot_digest(fact))
        line = resolution.resolve(RetrievalRequest(intent, fact), src,
                                  llm=lambda system, user: '{"candidates": []}')
        target = next((c for c in line.alternatives if c.code == "TARGET"), None)
        self.assertIsNotNone(target, "the merged candidate must still reach "
                             "verification despite 8 higher-scored rivals")
        self.assertEqual(set(target.authority["sources"]), {"retrieval", "umls_recall"})


class SemanticClassMatchingRules(unittest.TestCase):
    """`AuthoritativeSource.semantic_class`'s rule interpreter, isolated from real
    data by pre-populating its lazy caches directly (the same objects the real
    file-backed loaders would produce) -- deterministic, no authoritative data file
    dependency, still exercising the REAL matching logic, not a reimplementation."""

    def _source(self, records, rules, icd_chapters=None, cpt_categories=None, gp=None):
        s = AuthoritativeSource()
        s._records = {}   # unused by this path; lookup() reads _reference()
        s._semrules = rules
        s._icdchap = icd_chapters or []
        s._cptcat = cpt_categories or {}
        s._gp = gp or {}
        # `semantic_class` calls `self.lookup`, which reads `self._reference()` --
        # patch `lookup` directly rather than the whole reference-DB machinery.
        s.lookup = lambda code, system: records.get((code, system))
        return s

    def test_descriptor_any_matches_case_insensitively(self):
        s = self._source(
            records={("W1", "cpt"): {"long_description": "Widget Evaluation Visit"}},
            rules={"evaluation_management": {"descriptor_any": ["evaluation visit"]}})
        self.assertEqual(s.semantic_class("W1", "cpt"), "evaluation_management")

    def test_global_days_kind_numeric_matches_only_digit_global_periods(self):
        s = self._source(
            records={("W2", "cpt"): {"long_description": "Widget removal"}},
            rules={"surgical_procedure": {"global_days_kind": "numeric"}},
            gp={"W2": {"global": "090"}})
        self.assertEqual(s.semantic_class("W2", "cpt"), "surgical_procedure")

    def test_global_days_kind_numeric_does_not_match_a_non_numeric_indicator(self):
        s = self._source(
            records={("W3", "cpt"): {"long_description": "Widget imaging"}},
            rules={"surgical_procedure": {"global_days_kind": "numeric"}},
            gp={"W3": {"global": "XXX"}})
        self.assertIsNone(s.semantic_class("W3", "cpt"))

    def test_icd_chapter_ids_matches_by_the_codes_own_category_not_the_full_code(self):
        """Regression: a full code longer than its 3-character category must still
        match a chapter boundary expressed only at the category level (the
        lexicographic-prefix bug this module's own docstring documents)."""
        s = self._source(
            records={},
            rules={"injury_poisoning": {"icd_chapter_ids": [19]}},
            icd_chapters=[(19, "S00", "T88")])
        self.assertEqual(s.semantic_class("T8888XA", "icd10"), "injury_poisoning")
        self.assertIsNone(s.semantic_class("A00", "icd10"))

    def test_cpt_category_membership_matches(self):
        s = self._source(
            records={("W4", "cpt"): {"long_description": "Widget tracking code"}},
            rules={"performance_measure_tracking": {"cpt_category":
                                                     "performance_measure_tracking"}},
            cpt_categories={"performance_measure_tracking": frozenset({"W4"})})
        self.assertEqual(s.semantic_class("W4", "cpt"), "performance_measure_tracking")

    def test_no_rule_matches_returns_none_not_a_guess(self):
        s = self._source(
            records={("W5", "cpt"): {"long_description": "Something unclassified"}},
            rules={"evaluation_management": {"descriptor_any": ["evaluation visit"]}})
        self.assertIsNone(s.semantic_class("W5", "cpt"))

    def test_first_matching_rule_wins_when_more_than_one_could_apply(self):
        s = self._source(
            records={("W6", "cpt"): {"long_description": "Widget evaluation, surgical"}},
            rules={"evaluation_management": {"descriptor_any": ["evaluation"]},
                  "surgical_procedure": {"global_days_kind": "numeric"}},
            gp={"W6": {"global": "090"}})
        # dict iteration order is insertion order in real coding_semantics.json;
        # this proves the interpreter respects that order rather than some other
        # implicit precedence.
        self.assertEqual(s.semantic_class("W6", "cpt"), "evaluation_management")


class ConceptLookupIsAxisAware(unittest.TestCase):
    """`concept_lookup(axis, term)` — the generalized signature (issue #6 item 3)."""

    def test_mock_source_only_answers_the_anatomy_axis_from_its_configured_table(self):
        source = MockSource(concept_lookup={"widget term": {
            "term": "widget term", "candidates": ["C1"], "method": "exact",
            "unique": True, "expansions": ["alt widget term"],
            "source_identity": {"source_id": "mock"}}})
        anatomy = source.concept_lookup("anatomy", "widget term")
        self.assertEqual(anatomy["candidates"], ["C1"])
        # The SAME configured term on a DIFFERENT axis must not silently answer --
        # only "anatomy" was ever configured, matching real AuthoritativeSource
        # behavior where each axis has its own governed source or none at all.
        other_axis = source.concept_lookup("procedure", "widget term")
        self.assertEqual(other_axis["candidates"], [])
        self.assertEqual(other_axis["method"], "none")

    def test_an_ungoverned_axis_is_honestly_empty_not_an_error(self):
        source = MockSource()
        result = source.concept_lookup("approach", "open")
        self.assertEqual(result, {"term": "open", "candidates": [], "method": "none",
                                  "unique": False, "expansions": [],
                                  "source_identity": None})

    def test_real_source_procedure_axis_is_gated_before_touching_the_synonym_index(self):
        """Without a loaded synonym index, the procedure axis degrades exactly like
        anatomy degrades without a concept graph -- honestly empty, never an
        exception reaching the caller."""
        s = AuthoritativeSource()
        s._proc_syn_index = False   # simulate "index unavailable", bypass file I/O
        result = s.concept_lookup("procedure", "anything")
        self.assertEqual(result["candidates"], [])
        self.assertEqual(result["method"], "none")

    def test_procedure_axis_method_is_never_the_governed_concept_method_string(self):
        """The trust-tier distinction is load-bearing: a caller must be able to tell
        a procedure-axis match (AI self-consistency) apart from a real governed
        concept-graph match (anatomy) by `method` alone."""
        s = AuthoritativeSource()
        s._proc_syn_index = True
        s._proc_syn_by_term = {"widget term": {"W1"}}
        s._proc_syn_by_code = {"W1": ("widget term", "alt widget phrase")}
        s._proc_syn_identity = {"source_id": "mock_verified_synonyms"}
        result = s.concept_lookup("procedure", "widget term")
        self.assertEqual(result["method"], "retrieval_consistency_validated")
        self.assertNotIn(result["method"], ("exact", "despaced", "token_set"))
        self.assertEqual(result["candidates"], ["W1"])
        self.assertTrue(result["unique"])
        self.assertIn("alt widget phrase", result["expansions"])
        self.assertNotIn("widget term", result["expansions"],
                         "a term never expands to itself")

    def test_procedure_axis_ambiguous_when_a_term_verifies_against_more_than_one_code(self):
        s = AuthoritativeSource()
        s._proc_syn_index = True
        s._proc_syn_by_term = {"widget term": {"W1", "W2"}}
        s._proc_syn_by_code = {"W1": ("widget term",), "W2": ("widget term",)}
        s._proc_syn_identity = {"source_id": "mock_verified_synonyms"}
        result = s.concept_lookup("procedure", "widget term")
        self.assertFalse(result["unique"])
        self.assertEqual(sorted(result["candidates"]), ["W1", "W2"])
        self.assertEqual(result["expansions"], [],
                         "an ambiguous match must not guess which code's other "
                         "terms to offer as expansions")


class CompiledRecordAssembly(unittest.TestCase):
    """`semantics.compiled_record` -- the item-1 schema, assembled from whatever a
    source (real or mock) answers, never guessed when a source is silent."""

    def test_icd10_record_shape(self):
        # A realistic (synthetic) 3+ character code -- real ICD-10-CM codes are
        # never shorter than a 3-character category, which the ancestor-walk in
        # `excludes1_refs`/`component_relationships` (`range(len(undot), 2, -1)`)
        # assumes.
        source = MockSource(
            records={("D10", "icd10"): {"description": "Widget deficiency syndrome",
                                        "effective_from": "2020-01-01",
                                        "effective_to": "2030-01-01"}},
            excludes1={"D10": {"D20"}})
        rec = semantics.compiled_record("D10", "icd10", source)
        self.assertEqual(rec["code"], "D10")
        self.assertEqual(rec["system"], "icd10")
        self.assertEqual(rec["semantic_class"], "diagnosis")
        self.assertEqual(rec["action_concepts"], [])
        self.assertIn("widget", rec["anatomy_concepts"])
        self.assertEqual(rec["component_relationships"],
                         [{"type": "excludes1", "target_code_refs": ["D20"]}])
        self.assertEqual(rec["effective_period"],
                         {"from": "2020-01-01", "to": "2030-01-01"})

    def test_cpt_record_shape_and_source_identity_passthrough(self):
        source = MockSource(records={("P1", "cpt"): {
            "long_description": "Widgetectomy, sprocket",
            "effective_date": "2026-01-01"}})
        rec = semantics.compiled_record("P1", "cpt", source,
                                        source_identity={"data_fingerprint_sha256": "abc"})
        self.assertEqual(rec["action_concepts"], ["widgetectomy"])
        self.assertEqual(rec["anatomy_concepts"], ["sprocket"])
        self.assertEqual(rec["source_identity"], {"data_fingerprint_sha256": "abc"})
        self.assertEqual(rec["effective_period"], {"from": "2026-01-01", "to": ""})

    def test_missing_source_identity_defaults_to_empty_not_a_live_recompute(self):
        """Regression: `compiled_record` must never call a source's own
        `data_fingerprint()` internally (expensive, uncached, would turn a
        per-candidate call into a per-candidate full-manifest rebuild)."""
        source = MockSource(records={("P2", "cpt"): {"long_description": "Widget"}})
        rec = semantics.compiled_record("P2", "cpt", source)
        self.assertEqual(rec["source_identity"], {})

    def test_unknown_code_returns_none_not_a_placeholder_record(self):
        source = MockSource()
        self.assertIsNone(semantics.compiled_record("NOPE", "cpt", source))

    def test_approach_words_are_not_promoted_without_a_governed_field(self):
        source = MockSource(records={("P3", "cpt"): {
            "long_description": "Widget repair, open approach"}})
        rec = semantics.compiled_record("P3", "cpt", source)
        self.assertEqual(rec["approach"], [])

    def test_required_attributes_flags_a_bounded_measurement_interval(self):
        source = MockSource(records={("P4", "cpt"): {
            "long_description": "Widget repair, size 4 sq cm or less"}})
        rec = semantics.compiled_record("P4", "cpt", source)
        self.assertIn("measurement", rec["required_attributes"])


class RealAuthoritativeDataSmokeTest(unittest.TestCase):
    """One test against the REAL loaded authoritative data, per this plan's own test
    philosophy: the subject is discovered dynamically by a semantic property, never
    asserted as a literal code, so this proves the mechanism works on real data
    without hardcoding a medical-code value in the test."""

    def test_some_real_cpt_code_with_a_numeric_global_period_classifies_as_surgical(self):
        source = AuthoritativeSource()
        table = source._pfs_table()
        candidate = next((code for code, rec in table.items()
                          if str(rec.get("global") or "").isdigit()), None)
        if candidate is None:
            self.skipTest("no code with a numeric global period in the loaded PFS table")
        result = source.semantic_class(candidate, "cpt")
        self.assertEqual(result, "surgical_procedure")


if __name__ == "__main__":
    unittest.main()
