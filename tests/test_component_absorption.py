"""`pipeline.apply_convention_component_absorption`: a released procedure the cited
convention says is included in another released procedure of the same operative
episode is excluded, naming the parent and the authority (conventions pack
`claim_controls`, mechanic "component_included_in_released_service"). The shipped
entry: Achilles debridement is reported with the secondary-repair Achilles code.
"""
import json
import tempfile
import unittest
from pathlib import Path

from claude_coder import conventions as _conventions
from claude_coder import pipeline
from claude_coder.data_access import MockSource
from claude_coder.models import (CandidateCode, ClinicalFact, CodingResult, Disposition,
                                 EvidenceSpan, FactKind, ResolutionMethod, ResolvedLine)

PARENT_DESC = "Repair, secondary, Achilles tendon, with or without graft"
COMPONENT_DESC = "Excision of lesion, tendon, tendon sheath, or capsule (including synovectomy); foot"


def _line(fid, code, descriptor, evidence):
    fact = ClinicalFact(kind=FactKind.PROCEDURE, description=evidence.lower(),
                        attributes={"performer_id": "person-1", "billing_entity_id": "person-1"},
                        disposition=Disposition.PERFORMED,
                        evidence=[EvidenceSpan(text=evidence, anchored=True, span_id=f"sp-{fid}")],
                        confidence=0.9, fact_id=fid)
    return ResolvedLine(fact=fact, chosen=CandidateCode(code, "cpt", descriptor),
                        method=ResolutionMethod.VERIFIED)


def _agreed(*span_ids):
    from app.contracts.source_evidence import (ReconciliationStatus, SourceReconciliation,
                                               SpanReconciliation)
    return SourceReconciliation(spans=tuple(
        SpanReconciliation(span_id=s, status=ReconciliationStatus.AGREED, pages=(1,))
        for s in span_ids))


def _result(component_evidence="Abnormal and damaged tendon tissue was removed.", shared=True):
    parent = _line("F3", "CPT_REPAIR", PARENT_DESC, "The tendon was reattached with suture anchors.")
    comp = _line("F2", "CPT_EXCISE", COMPONENT_DESC, component_evidence)
    intents = ([{"intent_id": "si-1", "component_event_ids": ["F1", "F2", "F3"]}] if shared else
               [{"intent_id": "si-1", "component_event_ids": ["F3"]},
                {"intent_id": "si-2", "component_event_ids": ["F2"]}])
    return CodingResult(encounter_id="enc", date_of_service="2026-03-14", lines=[parent, comp],
                        service_intents=intents), parent, comp


class ComponentAbsorptionTest(unittest.TestCase):

    def setUp(self):
        _conventions.load_claim_controls.cache_clear()

    def test_the_documented_debridement_is_included_in_the_released_secondary_repair(self):
        result, parent, comp = _result()
        pipeline.apply_convention_component_absorption(result, MockSource(), _agreed("sp-F2", "sp-F3"))
        self.assertTrue(comp.excluded_reason, comp.excluded_reason)
        self.assertIn("CPT_REPAIR", comp.excluded_reason)
        self.assertIn("CPT Assistant", comp.excluded_reason)
        self.assertIsNone(parent.excluded_reason)
        self.assertEqual(comp.tie_record["claim_level_exclusion"]["parent_fact_id"], "F3")

    def test_a_documented_rupture_keeps_the_component(self):
        result, _, comp = _result("Ruptured tendon tissue was debrided and removed.")
        pipeline.apply_convention_component_absorption(result, MockSource(), _agreed("sp-F2", "sp-F3"))
        self.assertIsNone(comp.excluded_reason)

    def test_a_different_operative_episode_keeps_the_component(self):
        result, _, comp = _result(shared=False)
        pipeline.apply_convention_component_absorption(result, MockSource(), _agreed("sp-F2", "sp-F3"))
        self.assertIsNone(comp.excluded_reason)

    def test_disagreed_evidence_keeps_the_component(self):
        """Fail-closed: the component's quotation was checked against the original
        page and NOT confirmed -- nothing may be read off it."""
        from app.contracts.source_evidence import (ReconciliationStatus, SourceReconciliation,
                                                   SpanReconciliation)
        disagreed = SourceReconciliation(spans=(
            SpanReconciliation(span_id="sp-F2", status=ReconciliationStatus.DISAGREED, pages=(1,)),
            SpanReconciliation(span_id="sp-F3", status=ReconciliationStatus.AGREED, pages=(1,))))
        result, _, comp = _result()
        pipeline.apply_convention_component_absorption(result, MockSource(), disagreed)
        self.assertIsNone(comp.excluded_reason)

    def test_no_released_parent_keeps_the_component(self):
        result, parent, comp = _result()
        parent.excluded_reason = "already bundled elsewhere"
        pipeline.apply_convention_component_absorption(result, MockSource(), _agreed("sp-F2", "sp-F3"))
        self.assertIsNone(comp.excluded_reason)

    def test_a_pack_without_the_entry_is_inert(self):
        result, _, comp = _result()
        with tempfile.TemporaryDirectory() as tmp:
            pack = Path(tmp) / "pack.json"
            pack.write_text(json.dumps({"conventions": [], "claim_controls": []}))
            pipeline.apply_convention_component_absorption(result, MockSource(), _agreed("sp-F2", "sp-F3"),
                                                           pack_path=str(pack))
        self.assertIsNone(comp.excluded_reason)

    def test_the_shipped_entry_cites_its_authority(self):
        controls = _conventions.load_claim_controls("component_included_in_released_service")
        self.assertEqual(len(controls), 1)
        self.assertIn("CPT Assistant", controls[0]["authority"])
        self.assertIn("evidence_regex", controls[0]["applies_when"])


if __name__ == "__main__":
    unittest.main()
