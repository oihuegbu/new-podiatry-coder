"""SNOMED CT -> ICD-10-CM extended-map CONTEXT rules (issue #6, F13-type holds):
`terminology.MapRuleResolver`, its recall wiring in `data_access`, and the builder's
companion output. Synthetic concepts, terms and targets throughout -- the mechanic
turns on the map's own rule grammar, never on a clinical term."""
import importlib.util
import json
import pathlib
import tempfile
import unittest
from unittest.mock import patch

from claude_coder import terminology as term


def _document():
    return {
        "concepts": {
            "B1": ["structure alpha degeneration"],
            "I1": ["insertional structure alpha tendinopathy"],
            "I2": ["right structure alpha degeneration"],
            "I3": ["calcific structure alpha degeneration of right side"],
        },
        "base": {"B1": [
            {"group": 1, "priority": 1, "rule": "IFA", "ifa": "I3", "target": "X01.1", "advice": ""},
            {"group": 1, "priority": 2, "rule": "IFA", "ifa": "I1", "target": "X01.2", "advice": ""},
            {"group": 1, "priority": 3, "rule": "IFA", "ifa": "I2", "target": "X01.3", "advice": ""},
            {"group": 1, "priority": 4, "rule": "OTHERWISE TRUE", "ifa": "", "target": "X01.9",
             "advice": ""},
            {"group": 2, "priority": 1, "rule": "IFA", "ifa": "I2", "target": "X01.4", "advice": ""},
        ]},
        "term_to_base": {"structure alpha degeneration": ["B1"]},
    }


class MapRuleResolverTest(unittest.TestCase):
    def _targets(self, wording, laterality=None):
        hits = term.MapRuleResolver(_document()).resolve(wording, laterality)
        return [(h["group"], h["target"], h["rule"], h["ifa_concept"]) for h in hits]

    def test_a_lexical_ifa_rule_fires_on_the_referenced_concepts_distinctive_word(self):
        self.assertEqual(self._targets("insertional structure alpha degeneration"),
                         [(1, "X01.2", "IFA", "I1")])

    def test_a_laterality_only_ifa_rule_is_resolved_by_typed_laterality_never_lexically(self):
        self.assertEqual(self._targets("structure alpha degeneration", "right"),
                         [(1, "X01.3", "IFA", "I2"), (2, "X01.4", "IFA", "I2")])
        self.assertEqual(self._targets("right structure alpha degeneration", None),
                         [(1, "X01.9", "OTHERWISE TRUE", "")])

    def test_the_wrong_typed_laterality_falls_to_the_default(self):
        self.assertEqual(self._targets("structure alpha degeneration", "left"),
                         [(1, "X01.9", "OTHERWISE TRUE", "")])

    def test_a_rule_carrying_laterality_and_content_needs_both(self):
        self.assertEqual(self._targets("calcific structure alpha degeneration", "right"),
                         [(1, "X01.1", "IFA", "I3"), (2, "X01.4", "IFA", "I2")])

    def test_a_word_shared_across_the_bases_rule_set_selects_nothing(self):
        doc = _document()
        # "tender" recurs across more than three distinct content profiles -> not informative
        for k, extra in (("I1", "tender"), ("I2", "tender"), ("I3", "tender")):
            doc["concepts"][k] = [t + " tender" for t in doc["concepts"][k]]
        doc["concepts"]["I5"] = ["tender structure alpha degeneration"]
        doc["concepts"]["I7"] = ["tender variant structure alpha degeneration"]
        doc["base"]["B1"].insert(0, {"group": 1, "priority": 0, "rule": "IFA", "ifa": "I5",
                                     "target": "X01.5", "advice": ""})
        doc["base"]["B1"].insert(1, {"group": 1, "priority": 0, "rule": "IFA", "ifa": "I7",
                                     "target": "X01.7", "advice": ""})
        hits = term.MapRuleResolver(doc).resolve("tender structure alpha degeneration")
        self.assertEqual([(h["group"], h["target"]) for h in hits], [(1, "X01.9")])

    def test_two_equally_covered_concepts_fire_neither(self):
        doc = _document()
        doc["concepts"]["I6"] = ["insertional structure alpha tendinosis"]
        doc["base"]["B1"].append({"group": 3, "priority": 1, "rule": "IFA", "ifa": "I6",
                                  "target": "X01.6", "advice": ""})
        hits = term.MapRuleResolver(doc).resolve("insertional structure alpha degeneration")
        self.assertEqual([(h["group"], h["target"]) for h in hits], [(1, "X01.9")])
        # content stated, side wrong -> not I3; I1/I2 do not fire -> default
        self.assertEqual(self._targets("calcific structure alpha degeneration", "left"),
                         [(1, "X01.9", "OTHERWISE TRUE", "")])

    def test_side_words_phrased_with_shared_site_scaffolding_are_still_laterality_rules(self):
        doc = _document()
        doc["concepts"]["I8"] = ["structure alpha degeneration of right upper part"]
        doc["concepts"]["I9"] = ["structure alpha degeneration of left upper part"]
        doc["base"]["B1"] = [
            {"group": 1, "priority": 1, "rule": "IFA", "ifa": "I8", "target": "X01.8", "advice": ""},
            {"group": 1, "priority": 2, "rule": "IFA", "ifa": "I9", "target": "X01.7", "advice": ""},
            {"group": 1, "priority": 3, "rule": "OTHERWISE TRUE", "ifa": "", "target": "X01.9", "advice": ""}]
        self.assertEqual([(h["group"], h["target"]) for h in term.MapRuleResolver(doc).resolve(
            "structure alpha degeneration", "left")], [(1, "X01.7")])
        self.assertEqual([(h["group"], h["target"]) for h in term.MapRuleResolver(doc).resolve(
            "structure alpha degeneration", None)], [(1, "X01.9")])
        # a bilateral variant without the site words does not disturb the scaffold
        doc["concepts"]["I11"] = ["bilateral structure alpha degeneration"]
        doc["base"]["B1"].insert(2, {"group": 1, "priority": 2, "rule": "IFA", "ifa": "I11",
                                     "target": "X01.5", "advice": ""})
        self.assertEqual([(h["group"], h["target"]) for h in term.MapRuleResolver(doc).resolve(
            "structure alpha degeneration", "right")], [(1, "X01.8")])
        # a word that also names an UNSIDED variant is content, not scaffolding
        doc["concepts"]["I10"] = ["structure alpha degeneration of upper part"]
        doc["base"]["B1"].insert(0, {"group": 1, "priority": 0, "rule": "IFA", "ifa": "I10",
                                     "target": "X01.6", "advice": ""})
        self.assertEqual([(h["group"], h["target"]) for h in term.MapRuleResolver(doc).resolve(
            "structure alpha degeneration", "left")], [(1, "X01.9")])

    def test_wording_matching_no_rule_bearing_concept_resolves_nothing(self):
        self.assertEqual(self._targets("an unrelated documented condition", "right"), [])

    def test_every_hit_carries_full_rule_provenance(self):
        (hit,) = term.MapRuleResolver(_document()).resolve(
            "insertional structure alpha degeneration")
        self.assertEqual(hit["base_concept"], "B1")
        self.assertEqual(hit["base_terms"], ["structure alpha degeneration"])
        self.assertEqual(hit["ifa_terms"], ["insertional structure alpha tendinopathy"])
        self.assertIn("method", hit["base_match"])
        self.assertIn("['insertional']", hit["why"])


class RecallWiringTest(unittest.TestCase):
    """`AuthoritativeSource.snomed_code_matches` adds context-rule targets beside
    the unconditional term hits, with method "context_rule" and provenance."""

    class _Stub:
        def __init__(self):
            self._snomed = term.TerminologyIndex({"X01.9": ["structure alpha degeneration"]})
            self._snomed_identity = {"source_id": "snomed_crosswalk"}
            self._snomed_rules = term.MapRuleResolver(_document())
            self._snomed_rules_identity = {"source_id": "snomed_crosswalk_rules"}

        def _ensure_snomed_term_index(self):
            return self._snomed

        def _ensure_snomed_rules(self):
            return self._snomed_rules

        def leaf_codes(self, code, system):
            return {code} if code != "X01.4" else set()   # X.4 is not a loaded code

    def test_context_rule_targets_join_the_term_hits(self):
        from claude_coder.data_access import AuthoritativeSource
        out = AuthoritativeSource.snomed_code_matches(
            self._Stub(), "insertional structure alpha degeneration", "icd10",
            laterality="right")
        self.assertNotIn("X01.9", out)                  # the base's own default, superseded
        self.assertIn("X01.2", out)                     # IFA I1, lexical
        self.assertEqual(out["X01.2"]["method"], "context_rule")
        self.assertEqual(out["X01.2"]["context_rule"]["ifa_concept"], "I1")
        self.assertEqual(out["X01.2"]["source_identity"]["source_id"], "snomed_crosswalk_rules")
        self.assertNotIn("X01.4", out)                  # target not in the loaded code set

    def test_the_default_stays_when_no_ifa_row_fires(self):
        from claude_coder.data_access import AuthoritativeSource
        out = AuthoritativeSource.snomed_code_matches(
            self._Stub(), "structure alpha degeneration", "icd10", laterality=None)
        self.assertEqual(set(out), {"X01.9"})
        self.assertEqual(out["X01.9"]["method"], "exact")

    def test_without_the_rules_file_recall_is_exactly_the_term_map(self):
        from claude_coder.data_access import AuthoritativeSource
        stub = self._Stub()
        stub._snomed_rules = False
        out = AuthoritativeSource.snomed_code_matches(
            stub, "insertional structure alpha degeneration", "icd10", laterality="right")
        self.assertEqual(set(out), {"X01.9"})


class DiagnosisRecallPassesTypedLateralityTest(unittest.TestCase):
    def test_resolve_passes_the_claim_authorized_laterality_to_the_crosswalk(self):
        from claude_coder.data_access import MockSource
        from claude_coder.models import (AttributeEvidence, CandidateCode, ClinicalFact,
                                         EvidenceSpan, FactKind, RelationState)
        from claude_coder import resolution
        from tests.test_claude_coder import _request
        seen = {}
        src = MockSource(records={("CODE_A", "icd10"): {"long_description": "Condition alpha, right part",
                                                        "active": True}},
                         retrieval={("*", "icd10"): [CandidateCode("CODE_A", "icd10",
                                                                    "Condition alpha, right part", 0.9)]})

        def snomed_code_matches(description, system, *, laterality=None):
            seen["laterality"] = laterality
            return {}
        src.snomed_code_matches = snomed_code_matches
        span = EvidenceSpan("condition alpha of the right part", anchored=True, span_id="s1")
        fact = ClinicalFact(kind=FactKind.DIAGNOSIS, description="condition alpha",
                            attributes={"laterality": "right"}, evidence=[span], fact_id="F1",
                            attribute_evidence={"laterality": (AttributeEvidence(
                                span=span, scope="local", assertion_state=RelationState.ASSERTED,
                                value="right"),)})
        resolution.resolve(_request(fact), src, llm=None)
        self.assertEqual(seen.get("laterality"), "right")


class BuilderEmitsContextRulesTest(unittest.TestCase):
    """`tools/build_snomed_icd10_map.py` writes the companion rules file from a
    synthetic RF2 release: rows in map order, only rule-bearing bases, only
    targets in the loaded code set, referenced concepts' own terms."""

    HEADER_MAP = ("id\teffectiveTime\tactive\tmoduleId\trefsetId\treferencedComponentId\t"
                  "mapGroup\tmapPriority\tmapRule\tmapAdvice\tmapTarget\tcorrelationId\t"
                  "mapCategoryId\n")
    HEADER_DESC = ("id\teffectiveTime\tactive\tmoduleId\tconceptId\tlanguageCode\ttypeId\t"
                   "term\tcaseSignificanceId\n")

    def _release(self, root):
        rel = pathlib.Path(root) / "SnomedCT_TEST_20260101"
        (rel / "Snapshot" / "Refset" / "Map").mkdir(parents=True)
        (rel / "Snapshot" / "Terminology").mkdir(parents=True)
        rows = [
            ("1", "B1", "1", "2", "IFA 700 | Insertional |", "IF", "X01.2"),
            ("1", "B1", "1", "1", "IFA 800 | Calcific right |", "IF", "X01.1"),
            ("1", "B1", "1", "9", "OTHERWISE TRUE", "ALWAYS", "X01.9"),
            ("1", "B1", "2", "1", "OTHERWISE TRUE", "CANNOT", ""),
            ("1", "B2", "1", "1", "TRUE", "ALWAYS", "Y01.1"),          # no IFA -> not a base
            ("0", "B1", "1", "3", "IFA 900 | inactive |", "IF", "X01.8"),   # inactive row
            ("1", "B3", "1", "1", "IFA 950 | termless |", "IF", "X01.7"),   # IFA concept w/o terms
            ("1", "B3", "1", "2", "OTHERWISE TRUE", "ALWAYS", "X01.6"),
        ]
        with open(rel / "Snapshot" / "Refset" / "Map" / "der2_iisssccRefset_ExtendedMapSnapshot_TEST.txt", "w") as fh:
            fh.write(self.HEADER_MAP)
            for i, (a, cid, g, p, rule, adv, tgt) in enumerate(rows):
                fh.write(f"{i}\t20260101\t{a}\tm\trs\t{cid}\t{g}\t{p}\t{rule}\t{adv}\t{tgt}\tc\tk\n")
        descs = [("B1", "structure alpha degeneration"), ("B1", "Structure alpha degeneration (disorder)"),
                 ("700", "insertional structure alpha tendinopathy"), ("800", "calcific structure alpha degeneration of right side"),
                 ("B2", "condition beta"), ("B3", "condition gamma"), ("900", "inactive concept")]
        with open(rel / "Snapshot" / "Terminology" / "sct2_Description_Snapshot-en_TEST.txt", "w") as fh:
            fh.write(self.HEADER_DESC)
            for i, (cid, t) in enumerate(descs):
                fh.write(f"{i}\t20260101\t1\tm\t{cid}\ten\tsyn\t{t}\tcs\n")
        return rel

    def test_companion_rules_file(self):
        spec = importlib.util.spec_from_file_location(
            "build_snomed_icd10_map",
            pathlib.Path(__file__).resolve().parent.parent / "tools" / "build_snomed_icd10_map.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        with tempfile.TemporaryDirectory() as tmp:
            data = pathlib.Path(tmp) / "data"
            (data / "codes").mkdir(parents=True)
            (data / "codes" / "icd10cm_codes.json").write_text(json.dumps(
                [{"code": c} for c in ("X01.1", "X01.2", "X01.9", "X01.6", "X01.7", "Y01.1")]))
            rel = self._release(tmp)
            with patch.object(mod, "DATA_DIR", data), \
                    patch("sys.argv", ["build", "--release", str(rel)]):
                rc = mod.main()
            self.assertEqual(rc, 0)
            rules = json.loads((data / "codes" / "snomed_icd10_rules.json").read_text())
            self.assertEqual(sorted(rules["base"]), ["B1"])          # B2: no IFA; B3: IFA termless
            rows = rules["base"]["B1"]
            self.assertEqual([(r["group"], r["priority"], r["rule"], r["ifa"], r["target"]) for r in rows],
                             [(1, 1, "IFA", "800", "X01.1"), (1, 2, "IFA", "700", "X01.2"),
                              (1, 9, "OTHERWISE TRUE", "", "X01.9")])
            self.assertEqual(rules["concepts"]["B1"], ["structure alpha degeneration"])
            self.assertEqual(rules["concepts"]["800"], ["calcific structure alpha degeneration of right side"])
            self.assertEqual(rules["term_to_base"], {"structure alpha degeneration": ["B1"]})
            terms = json.loads((data / "codes" / "snomed_icd10_map.json").read_text())["terms"]
            self.assertEqual(terms["structure alpha degeneration"], ["X01.9"])   # unconditional map unchanged


class SpecificityRelativeInheritsTheBasesMatchTest(unittest.TestCase):
    """An immediate-stem sibling (same subcategory, different side character) of an
    authority-matched base inherits the base's term match, annotated; a category-
    wide cousin found by the token-overlap fallback does not."""

    def test_sibling_inherits_and_cousin_does_not(self):
        from claude_coder.data_access import MockSource
        from claude_coder.models import (AttributeEvidence, CandidateCode, ClinicalFact,
                                         EvidenceSpan, FactKind, RelationState)
        from claude_coder import resolution
        # undotted synthetic codes: MockSource.descriptions is dot-sensitive, the
        # specificity helper looks relatives up undotted (the real source is not)
        src = MockSource(records={
            ("X0879", "icd10"): {"long_description": "Fibrous overgrowth of structure alpha, unspecified region", "active": True},
            ("X0871", "icd10"): {"long_description": "Fibrous overgrowth of structure alpha, right region", "active": True},
            ("X0872", "icd10"): {"long_description": "Fibrous overgrowth of structure alpha, left region", "active": True},
            ("X0851", "icd10"): {"long_description": "Fibrous overgrowth of structure alpha, right other region", "active": True}})
        span = EvidenceSpan("fibrous overgrowth of structure alpha of the right region", anchored=True, span_id="s1")
        fact = ClinicalFact(kind=FactKind.DIAGNOSIS, description="fibrous overgrowth of structure alpha",
                            attributes={"laterality": "right"}, evidence=[span], fact_id="F1",
                            attribute_evidence={"laterality": (AttributeEvidence(
                                span=span, scope="local", assertion_state=RelationState.ASSERTED,
                                value="right"),)})
        base = CandidateCode("X0879", "icd10", "Fibrous overgrowth of structure alpha, unspecified region", 1.0,
                             source="snomed-crosswalk",
                             authority={"term_to_code_match": {"method": "context_rule"}})
        relatives = {c.code: c for c in resolution._diagnosis_specificity_candidates(fact, [base], src)}
        self.assertIn("X0871", relatives)
        self.assertIn("X0851", relatives)
        sibling = relatives["X0871"].authority["term_to_code_match"]
        self.assertEqual(sibling["method"], "context_rule")
        self.assertEqual(sibling["specificity_relative_of"], "X0879")
        self.assertNotIn("term_to_code_match", relatives["X0851"].authority)


class SiblingDisambiguationByTheRecordsOwnWordsTest(unittest.TestCase):
    """`refine_diagnosis_specificity` step 2: when the verifier entails two or more
    more-specific siblings of an already-established unspecified-side code (a code
    edition that split the concept by tissue as well as side), the ONE sibling
    whose own distinguishing vocabulary this event's evidence states -- every other
    sibling's being absent -- is selected; anything less clear-cut stays a
    documentation question. Synthetic vocabulary."""

    def _source(self):
        from claude_coder.data_access import MockSource
        return MockSource(records={
            ("RR10", "icd10"): {"long_description": "Widgetopathy of sheath and cord, unspecified region"},
            ("RR11", "icd10"): {"long_description": "Widgetopathy of sheath, right region"},
            ("RR13", "icd10"): {"long_description": "Widgetopathy of cord, right region"},
            ("RR12", "icd10"): {"long_description": "Widgetopathy of sheath, left region"},
            # side-specific but still "unspecified" in type: never a strictly more
            # specific sibling, must not enter the set (it shares both concept words)
            ("RR19", "icd10"): {"long_description": "Unspecified widgetopathy of sheath and cord, right region"}})

    def _line(self, evidence_text):
        from claude_coder.models import (AttributeEvidence, CandidateCode, ClinicalFact,
                                         Disposition, EvidenceSpan, FactKind, RelationState,
                                         ResolutionMethod, ResolvedLine)
        span = EvidenceSpan(evidence_text, anchored=True, span_id="s1")
        f = ClinicalFact(FactKind.DIAGNOSIS, "widgetopathy right region",
                         attributes={"laterality": "right"}, evidence=[span],
                         attribute_evidence={"laterality": (
                             AttributeEvidence(span=span, assertion_state=RelationState.ASSERTED,
                                               value="right"),)},
                         disposition=Disposition.PERFORMED)
        return ResolvedLine(fact=f, chosen=CandidateCode(
            "RR10", "icd10", "Widgetopathy of sheath and cord, unspecified region", 1.0),
            method=ResolutionMethod.VERIFIED, rationale="entailed")

    def _judge(self):
        from tests import shortlist_verdict as _sv
        return _sv.judge(entails=lambda d: "right" in d.lower(),
                         prefer=lambda d: "cord" in d.lower(), reason="more specific")

    def test_the_sibling_the_record_names_is_selected(self):
        from claude_coder import resolution
        from claude_coder.models import ResolutionMethod
        out = resolution.refine_diagnosis_specificity(
            self._line("widgetopathy of the cord of the right region"), self._source(), self._judge())
        self.assertTrue(out.resolved, out.rationale)
        self.assertEqual(out.chosen.code, "RR1.3")
        self.assertIs(out.method, ResolutionMethod.VERIFIED)
        self.assertEqual(out.tie_record["sibling_disambiguation"]["selected"], "RR1.3")
        self.assertIn("RR1.1", out.tie_record["sibling_disambiguation"]["absent"])
        self.assertNotIn("RR1.9", out.tie_record["sibling_disambiguation"]["still_entailed"])

    def test_neither_sibling_named_stays_a_documentation_question(self):
        from claude_coder import resolution
        out = resolution.refine_diagnosis_specificity(
            self._line("widgetopathy of the right region"), self._source(), self._judge())
        self.assertIsNone(out.chosen)
        self.assertIn("document the distinguishing detail", out.documentation_gap or "")

    def test_both_siblings_named_stays_a_documentation_question(self):
        from claude_coder import resolution
        out = resolution.refine_diagnosis_specificity(
            self._line("widgetopathy of the cord and sheath of the right region"),
            self._source(), self._judge())
        self.assertIsNone(out.chosen)


class PartialValuePhraseMatchTest(unittest.TestCase):
    """`ConceptRelationIndex`: an EMBEDDED match of a VALUE PHRASE whose governed
    window sits next to an unmatched qualifying token is partial -- it neither
    identifies the phrase nor grounds a relation -- while side words, connectives
    and action descriptions are untouched. Synthetic concepts."""

    def _index(self):
        return term.ConceptRelationIndex({
            "T": {"terms": ["cord", "cord structure"], "parents": []},
            "A": {"terms": ["alpha cord", "structure of alpha cord"], "parents": ["T"]},
            "B": {"terms": ["beta cord"], "parents": ["T"]},
            # a qualified descendant named by a Latin-form muscle name
            "D": {"terms": ["cord of deltoideus muscle structure", "deltoideus muscle cord"],
                  "parents": ["T"]},
            "D_ENTIRE": {"terms": ["entire deltoideus muscle cord"], "parents": ["D"]},
            # a different structure that merely mentions the head: never the identity
            "D_SHEATH": {"terms": ["deltoideus muscle cord sheath"], "parents": ["T"]},
            # a lineage where one qualifier lives on an ancestor
            "BONE": {"terms": ["bone", "bone structure"], "parents": []},
            "TARSAL": {"terms": ["tarsal bone", "tarsal bone structure"], "parents": ["BONE"]},
            "EPS": {"terms": ["structure of epsilon bone"], "parents": ["TARSAL"]},
            "EPS_ENTIRE": {"terms": ["entire epsilon bone"], "parents": ["EPS"]},
            # a governed muscle window immediately before the head qualifies it
            "FM": {"terms": ["flexor muscle"], "parents": []},
            "FMC": {"terms": ["flexor muscle cord structure"], "parents": ["T"]},
        })

    def test_a_qualified_head_noun_is_partial_and_unresolved(self):
        idx = self._index()
        m = idx.match_longest("gamma cord", phrase=True)
        self.assertTrue(m.partial)
        self.assertFalse(m.unique)
        self.assertEqual(idx.relation_detail("alpha cord", "gamma cord",
                                             embedded=True, phrase=True).verdict,
                         term.CONCEPT_UNRESOLVED)
        # the same text scanned as a DESCRIPTION keeps today's hierarchy verdict
        self.assertEqual(idx.relation_detail("alpha cord", "gamma cord", embedded=True).verdict,
                         term.CONCEPT_RELATED)

    def test_side_words_connectives_and_of_phrases_are_not_qualifiers(self):
        idx = self._index()
        for text in ("right alpha cord", "alpha cord with graft", "insertion of alpha cord",
                     "structure of alpha cord"):
            self.assertEqual(idx.relation_detail("alpha cord", text, embedded=True,
                                                 phrase=True).verdict,
                             term.CONCEPT_SAME, text)

    def test_normalize_gives_a_partial_match_no_expansions(self):
        idx = self._index()
        m, expansions = idx.normalize("gamma cord", embedded=True, phrase=True)
        self.assertTrue(m.partial)
        self.assertEqual(expansions, ())
        m2, expansions2 = idx.normalize("alpha cord", embedded=True, phrase=True)
        self.assertTrue(m2.unique)
        self.assertIn("structure of alpha cord", expansions2)

    def test_qualifiers_resolve_to_the_heads_descendant(self):
        idx = self._index()
        m = idx.match_longest("deltoid cord", phrase=True)        # "deltoid" ~ "deltoideus"
        self.assertEqual(m.candidates, ("D",))                    # the structure, not its "entire" child or the sheath
        self.assertTrue(m.unique)
        self.assertIn("qualified_descendant", m.method)
        _m, expansions = idx.normalize("deltoid cord", embedded=True, phrase=True)
        self.assertIn("cord of deltoideus muscle structure", expansions)
        # two sibling tendons are neither the same nor hierarchically related
        self.assertEqual(idx.relation_detail("alpha cord", "deltoid cord",
                                             embedded=True, phrase=True).verdict,
                         term.CONCEPT_UNRESOLVED)

    def test_a_qualifier_stated_by_an_ancestor_and_the_structure_over_its_entire_child(self):
        idx = self._index()
        m = idx.match_longest("tarsal epsilon bone", phrase=True)
        self.assertEqual(m.candidates, ("EPS",))
        self.assertTrue(m.unique)
        # the structure concept wins over its own "entire" child; the region qualifier
        # is stated by the ancestor, not by the concept's own terms
        self.assertEqual(idx.relation_detail("tarsal bone", "tarsal epsilon bone",
                                             embedded=True, phrase=True).verdict,
                         term.CONCEPT_RELATED)

    def test_a_qualifier_no_descendant_states_stays_partial(self):
        idx = self._index()
        m = idx.match_longest("omega cord", phrase=True)
        self.assertTrue(m.partial)
        self.assertFalse(m.unique)

    def test_an_adjacent_governed_window_qualifies_the_head(self):
        idx = self._index()
        m = idx.match_longest("flexor muscle cord", phrase=True)
        self.assertEqual(m.candidates, ("FMC",))
        self.assertTrue(m.unique)
