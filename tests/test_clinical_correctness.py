"""Tests for the clinical-correctness contracts added after the
routine_podiatry_00001 expert review:

  1. Section-aware evidence — incidental-anatomy sentences (tourniquet,
     positioning, prep/drape, anesthesia) never DRIVE a claim change,
     while protective evidence still reads the full note.
  2. Removal conservation — a documentation-mismatch removal must
     substitute the family member the documented work supports, or
     escalate loudly; documented work never falls off a claim silently.
  3. Material-correction tracking — every claim-mutating layer action is
     recorded, tagged interpretive vs. data-grounded.
  4. Clinical audit — interpretive corrections are verified by an
     authority-grounded expert-coder pass; any dispute forces REVIEW and
     blocks registry auto-recording (fail closed on malformed/ungrounded
     verdicts).

Everything runs against stub reference data — no live code sets, no
network, no hardcoded rule outcomes beyond the fixture's own descriptors.

Run:  PYTHONPATH=. .venv/bin/python -m pytest tests/test_clinical_correctness.py -q
"""

import unittest
from unittest import mock

from app.validation.validator import CodingValidator


# ---------------------------------------------------------------------------
# stub reference data (descriptors are the fixture's data, not medical rules)
# ---------------------------------------------------------------------------

class _StubDB:
    """Minimal CodeReferenceDB stand-in: descriptor lookups only."""

    def __init__(self, icd10=None, cpt=None):
        self.icd10 = {k.replace(".", "").upper(): {"description": v}
                      for k, v in (icd10 or {}).items()}
        self.cpt = {k: {"long_description": v} for k, v in (cpt or {}).items()}
        self.hcpcs = {}

    def validate_icd10(self, code):
        return self.icd10.get(str(code).replace(".", "").upper())

    def validate_cpt(self, code):
        info = self.cpt.get(str(code).strip())
        return {"description": info["long_description"]} if info else None

    def icd10_siblings(self, prefix):
        p = prefix.replace(".", "").upper()
        return [(c, i["description"]) for c, i in sorted(self.icd10.items())
                if c.startswith(p)]


def _validator(icd10=None, cpt=None):
    v = CodingValidator(_StubDB(icd10=icd10, cpt=cpt), compliance_store=None)
    v.issues = []
    v._bundled_codes_to_suppress = set()
    v._non_billable_codes_to_suppress = set()
    return v


# ---------------------------------------------------------------------------
# 1. section-aware evidence
# ---------------------------------------------------------------------------

class ClinicalNoteViewTest(unittest.TestCase):
    def setUp(self):
        self.v = _validator()

    def test_incidental_sentences_removed(self):
        note = ("Assessment: retrocalcaneal exostosis, right heel. "
                "A well-padded thigh tourniquet was inflated to 300 mmHg. "
                "The Achilles tendon was debrided of degenerative tissue.")
        view = self.v._clinical_note_view(note)
        self.assertIn("retrocalcaneal exostosis", view)
        self.assertIn("Achilles tendon was debrided", view)
        self.assertNotIn("thigh", view)

    def test_positioning_prep_anesthesia_removed(self):
        note = ("The patient was positioned prone on the operating table. "
                "The right lower extremity was prepped and draped in the "
                "usual sterile fashion. Local anesthetic block at the ankle. "
                "Findings: hypertrophic bone at the calcaneus.")
        view = self.v._clinical_note_view(note)
        self.assertNotIn("prone", view)
        self.assertNotIn("draped", view)
        self.assertNotIn("block", view)
        self.assertIn("hypertrophic bone", view)

    def test_clinical_content_untouched_without_markers(self):
        note = "Assessment: hallux valgus, right foot. Plan: bunionectomy."
        self.assertEqual(self.v._clinical_note_view(note), note)


class SiblingSwapAsymmetryTest(unittest.TestCase):
    """The exact regression shape from routine_00001: an acquired-deformity
    code relocated to the THIGH because a tourniquet-placement sentence
    contained the word."""

    ICD = {
        "M21.6X1": "Other acquired deformities of right foot",
        "M21.851": "Other specified acquired deformities of right thigh",
    }

    def _icd_entry(self):
        return {"code": "M21.6X1", "type": "secondary",
                "description": self.ICD["M21.6X1"]}

    def test_tourniquet_thigh_never_drives_a_swap(self):
        v = _validator(icd10=self.ICD)
        icd = [self._icd_entry()]
        note = ("Retrocalcaneal exostosis of the right heel was resected. "
                "A well-padded thigh tourniquet was inflated to 300 mmHg.")
        v._check_icd_sibling_descriptor(icd, note)
        self.assertEqual(icd[0]["code"], "M21.6X1",
                         "incidental 'thigh' must not relocate the diagnosis")
        self.assertNotIn("sibling_matches_note_better",
                         {i.category for i in v.issues})

    def test_clinically_documented_sibling_still_swaps(self):
        v = _validator(icd10=self.ICD)
        icd = [self._icd_entry()]
        note = ("Assessment: acquired deformity of the right thigh with "
                "palpable prominence. No foot involvement.")
        v._check_icd_sibling_descriptor(icd, note)
        self.assertEqual(icd[0]["code"], "M21.851",
                         "clinically asserted sibling evidence still decides")


# ---------------------------------------------------------------------------
# 2. removal conservation
# ---------------------------------------------------------------------------

class RemovalConservationTest(unittest.TestCase):
    CPT = {
        "27650": "Repair, primary, open or percutaneous, ruptured Achilles "
                 "tendon;",
        "27654": "Repair, secondary, Achilles tendon, with or without graft",
        "99213": "Office or other outpatient visit for the evaluation and "
                 "management of an established patient",
    }

    def _setup(self, category="undocumented_procedure_indication"):
        v = _validator(cpt=self.CPT)
        cpt = [{"code": "27650", "description": self.CPT["27650"],
                "modifiers": ["RT"], "units": 1}]
        v._non_billable_codes_to_suppress.add("27650")
        v._add("WARNING", "27650", category,
               "descriptor requires a ruptured tendon; none documented",
               "verify", denial_risk="HIGH")
        return v, cpt

    def test_unique_documented_sibling_substitutes(self):
        v, cpt = self._setup()
        note = ("Secondary repair of the right Achilles tendon was "
                "performed with graft augmentation and anchor fixation.")
        v._check_removal_conservation(cpt, note)
        self.assertEqual(cpt[0]["code"], "27654")
        self.assertNotIn("27650", v._non_billable_codes_to_suppress)
        self.assertTrue(cpt[0].get("needs_review"))
        conserv = [i for i in v.issues
                   if i.category == "removal_conservation"]
        self.assertEqual(len(conserv), 1)
        self.assertTrue(conserv[0].message.startswith("AUTO-CORRECTED"))

    def test_documented_work_without_provable_sibling_escalates(self):
        v, cpt = self._setup()
        # the WORK (repair of the Achilles tendon) is documented, but no
        # sibling's distinguishing attributes are provable
        note = "Repair of the right Achilles tendon was performed."
        v._check_removal_conservation(cpt, note)
        self.assertEqual(cpt[0]["code"], "27650")  # unchanged
        self.assertIn("27650", v._non_billable_codes_to_suppress)
        conserv = [i for i in v.issues
                   if i.category == "removal_conservation"]
        self.assertEqual(len(conserv), 1)
        self.assertIn("uncoded", conserv[0].message)
        self.assertEqual(conserv[0].denial_risk, "HIGH")

    def test_undocumented_work_removal_stands_silently(self):
        v, cpt = self._setup()
        note = "Nail debridement of five toenails performed."
        v._check_removal_conservation(cpt, note)
        self.assertIn("27650", v._non_billable_codes_to_suppress)
        self.assertNotIn("removal_conservation",
                         {i.category for i in v.issues})

    def test_structural_removals_never_conserved(self):
        for category in ("mue_exceeded", "ncci_bundling", "billability"):
            v, cpt = self._setup(category=category)
            note = ("Secondary repair of the Achilles tendon was performed "
                    "with graft augmentation.")
            v._check_removal_conservation(cpt, note)
            self.assertEqual(cpt[0]["code"], "27650",
                             f"{category} removal must stand")
            self.assertIn("27650", v._non_billable_codes_to_suppress)

    def test_incidental_evidence_never_substitutes(self):
        v, cpt = self._setup()
        # sibling attributes appear ONLY inside incidental-context language
        note = ("Repair of the Achilles tendon was performed. The secondary "
                "graft site was prepped and draped in sterile fashion.")
        v._check_removal_conservation(cpt, note)
        self.assertEqual(cpt[0]["code"], "27650",
                         "prep/drape sentence must not elect a sibling")


# ---------------------------------------------------------------------------
# 3. material-correction tracking
# ---------------------------------------------------------------------------

class MaterialCorrectionsTest(unittest.TestCase):
    def test_auto_corrections_and_removals_recorded(self):
        v = _validator()
        v._add("WARNING", "M77.31", "sibling_matches_note_better",
               "AUTO-CORRECTED: swapped", "verify", denial_risk="HIGH")
        v._add("WARNING", "28118", "mue_exceeded",
               "AUTO-CORRECTED: units capped", "verify")
        v._add("INFO", "11720", "some_advisory", "just a note", "none")
        v._non_billable_codes_to_suppress.add("A4570")
        v._add("ERROR", "A4570", "undocumented_supply_indication",
               "supply never documented", "remove")
        mats = v._material_corrections()
        by_code = {m["code"]: m for m in mats}
        self.assertIn("M77.31", by_code)
        self.assertTrue(by_code["M77.31"]["interpretive"])
        self.assertIn("28118", by_code)
        self.assertFalse(by_code["28118"]["interpretive"],
                         "MUE is data-grounded, not interpretive")
        self.assertEqual(by_code["A4570"]["action"], "removal")
        self.assertTrue(by_code["A4570"]["interpretive"])
        self.assertNotIn("11720", by_code)

    def test_no_corrections_means_empty(self):
        v = _validator()
        v._add("WARNING", "99213", "em_level_mismatch",
               "descriptor says low", "verify")
        self.assertEqual(v._material_corrections(), [])


# ---------------------------------------------------------------------------
# 4. clinical audit + registry gate
# ---------------------------------------------------------------------------

def _result(corrections, disposition="CLEAN", audit=None):
    r = {
        "success": True,
        "document_id": "note_x",
        "final_disposition": disposition,
        "auto_coding_tier": "AUTO" if disposition == "CLEAN" else "REVIEW",
        "auto_coding_confidence": 0.9,
        "auto_coding_review_reasons": [],
        "consistency": {"runs": 3, "unanimous": True},
        "material_corrections": corrections,
        "icd_codes": [{"code": "M77.31", "type": "primary",
                       "description": "Calcaneal spur, right foot"}],
        "cpt_codes": [{"code": "28118", "modifiers": ["RT"], "units": 1}],
        "hcpcs_codes": [],
    }
    if audit is not None:
        r["clinical_audit"] = audit
    return r


_INTERP = [{"category": "sibling_matches_note_better", "code": "M77.31",
            "action": "auto_correction", "interpretive": True,
            "message": "AUTO-CORRECTED: swapped"}]
_DATA = [{"category": "mue_exceeded", "code": "28118",
          "action": "auto_correction", "interpretive": False,
          "message": "AUTO-CORRECTED: capped"}]


class RegistryGateTest(unittest.TestCase):
    def _eligible(self, result):
        from tools.claims_registry import eligible_for_auto
        # These tests isolate the clinical-audit gate; certificate behavior
        # has its own release-boundary suite.
        with mock.patch(
                "app.release.claim_readiness.verify_readiness_certificate",
                return_value=(True, "")):
            return eligible_for_auto(result)

    def _upheld_audit(self, result):
        from tools.clinical_auditor import corrections_fingerprint
        return {"verdict": "upheld",
                "fingerprint": corrections_fingerprint(result)}

    def test_unreviewed_claim_blocks(self):
        ok, why = self._eligible(_result(_INTERP))
        self.assertFalse(ok)
        self.assertIn("not yet clinically reviewed", why)

    def test_upheld_audit_unblocks(self):
        r = _result(_INTERP)
        r["clinical_audit"] = self._upheld_audit(r)
        ok, why = self._eligible(r)
        self.assertTrue(ok, why)

    def test_disputed_audit_blocks(self):
        ok, why = self._eligible(_result(_INTERP,
                                         audit={"verdict": "disputed"}))
        self.assertFalse(ok)
        self.assertIn("disputed", why)

    def test_data_grounded_claims_also_need_review(self):
        # The review is UNIVERSAL: the absence of interpretive corrections
        # is exactly what an unreported mutation looks like (routine_00003).
        ok, why = self._eligible(_result(_DATA))
        self.assertFalse(ok)
        self.assertIn("not yet clinically reviewed", why)
        r = _result(_DATA)
        r["clinical_audit"] = self._upheld_audit(r)
        ok, _ = self._eligible(r)
        self.assertTrue(ok)

    def test_stale_audit_fingerprint_blocks(self):
        r = _result(_INTERP)
        r["clinical_audit"] = self._upheld_audit(r)
        # the claim changes after the review -> the verdict is stale
        r["cpt_codes"].append({"code": "99213", "modifiers": [],
                               "units": 1})
        ok, why = self._eligible(r)
        self.assertFalse(ok)
        self.assertIn("stale", why)


class ClinicalAuditorTest(unittest.TestCase):
    def _audit(self, result, verdicts, note="Assessment: calcaneal spur."):
        """Run audit_result with the LLM pass mocked to return verdicts
        (a list — one element per pass, cycled)."""
        import tools.clinical_auditor as ca
        calls = {"n": 0}

        def fake_once(case, pass_idx=0):
            v = verdicts[min(calls["n"], len(verdicts) - 1)]
            calls["n"] += 1
            return dict(v, _model="test", _usage={})

        with mock.patch.object(ca, "_audit_once", side_effect=fake_once):
            block = ca.audit_result("note_x", result, note, rep=object(),
                                    passes=len(verdicts))
        return block, calls["n"]

    def test_no_interpretive_corrections_still_reviewed(self):
        # The whole-claim review ALWAYS runs — a claim without recorded
        # interpretive corrections is exactly where an unreported mutation
        # hides (routine_00003), so there is no vacuous uphold anymore.
        result = _result(_DATA)
        block, n = self._audit(result, [{
            "items": [], "claim_findings": [],
            "claim_level_concerns": "", "overall_rationale": "clean"}])
        self.assertEqual(n, 1, "the review must call the model")
        self.assertEqual(block["verdict"], "upheld")
        self.assertEqual(result["final_disposition"], "CLEAN")

    def test_grounded_uphold_keeps_clean(self):
        result = _result(_INTERP)
        block, n = self._audit(result, [{
            "items": [{"index": 0, "verdict": "uphold",
                       "authority": "ICD-10-CM Index, Haglund -> M77.3-",
                       "note_evidence": "Haglund's deformity documented"}],
            "claim_level_concerns": "", "overall_rationale": "fine"}],
            note="Assessment: Haglund's deformity documented, right foot.")
        self.assertEqual(block["verdict"], "upheld")
        self.assertEqual(result["final_disposition"], "CLEAN")
        self.assertEqual(n, 1)

    def test_overturn_forces_review(self):
        result = _result(_INTERP)
        block, _ = self._audit(result, [{
            "items": [{"index": 0, "verdict": "overturn",
                       "authority": "descriptor mismatch",
                       "note_evidence": "note documents the other variant"}],
            "claim_level_concerns": "", "overall_rationale": "wrong"}])
        self.assertEqual(block["verdict"], "disputed")
        self.assertEqual(result["final_disposition"], "REVIEW")
        self.assertTrue(any("clinical_audit" in s for s in
                            result["auto_coding_review_reasons"]))

    def test_ungrounded_verdict_degrades_to_uncertain(self):
        result = _result(_INTERP)
        block, _ = self._audit(result, [{
            "items": [{"index": 0, "verdict": "uphold",
                       "authority": "", "note_evidence": ""}],
            "claim_level_concerns": "", "overall_rationale": ""}])
        self.assertEqual(block["verdict"], "disputed",
                         "ungrounded verdicts must fail closed")
        self.assertEqual(result["final_disposition"], "REVIEW")

    def test_malformed_verdict_fails_closed(self):
        result = _result(_INTERP)
        block, _ = self._audit(result, [{
            "items": [{"index": 7, "verdict": "uphold",
                       "authority": "x", "note_evidence": "y"}],
            "claim_level_concerns": "", "overall_rationale": ""}])
        self.assertEqual(block["verdict"], "disputed")

    def test_split_passes_degrade_to_uncertain(self):
        result = _result(_INTERP)
        block, n = self._audit(result, [
            {"items": [{"index": 0, "verdict": "uphold",
                        "authority": "a", "note_evidence": "b"}],
             "claim_level_concerns": "", "overall_rationale": ""},
            {"items": [{"index": 0, "verdict": "overturn",
                        "authority": "a", "note_evidence": "b"}],
             "claim_level_concerns": "", "overall_rationale": ""}])
        self.assertEqual(n, 2)
        self.assertEqual(block["verdict"], "disputed")

    def test_advisory_defect_disputes_despite_advisory_materiality(self):
        # a wrong scrubber advisory is a mechanizable dispute: an "upheld"
        # verdict here would ship the record CLEAN with the defect
        # unadjudicated, deadlocking its flip class at
        # awaiting_verification forever
        result = _result(_DATA)
        note = "Assessment: calcaneal spur. Custom orthotic dispensed."
        block, _ = self._audit(result, [{
            "items": [],
            "claim_findings": [{
                "kind": "advisory_defect", "array": "hcpcs_codes",
                "code": "L3000", "materiality": "advisory",
                "finding": "the coverage advisory demands diabetes-"
                           "pathway evidence; the LCD recognizes the "
                           "documented biomechanical pathway",
                "authority": "LCD L33641",
                "note_evidence": "Custom orthotic dispensed"}],
            "claim_level_concerns": "", "overall_rationale": ""}],
            note=note)
        self.assertEqual(block["verdict"], "disputed")
        self.assertEqual(result["final_disposition"], "REVIEW")
        self.assertTrue(any("advisory_defect" in s for s in
                            result["auto_coding_review_reasons"]))

    def test_claim_level_concern_disputes_even_when_items_upheld(self):
        result = _result(_INTERP)
        block, _ = self._audit(result, [{
            "items": [{"index": 0, "verdict": "uphold",
                       "authority": "a", "note_evidence": "b"}],
            "claim_level_concerns": "documented repair work is uncoded",
            "overall_rationale": ""}])
        self.assertEqual(block["verdict"], "disputed")

    def test_fingerprint_changes_with_corrections(self):
        from tools.clinical_auditor import corrections_fingerprint
        a = corrections_fingerprint(_result(_INTERP))
        b = corrections_fingerprint(_result(_DATA))
        self.assertNotEqual(a, b)

    def test_fingerprint_changes_with_adjudication_decisions(self):
        # The full-record review judges the adjudication block too — a
        # changed decision must stale the verdict even when the claim
        # arrays are unchanged.
        from tools.clinical_auditor import corrections_fingerprint
        r1 = _result(_INTERP)
        r2 = _result(_INTERP)
        r2["adjudication"] = {"items": [
            {"array": "cpt_codes", "code": "28118", "kind": "attributes",
             "decision": "set", "fields": {"modifiers": []}}]}
        self.assertNotEqual(corrections_fingerprint(r1),
                            corrections_fingerprint(r2))

    def test_fingerprint_protocol_version_stales_old_verdicts(self):
        # A verdict rendered under an older review protocol (the curated
        # case file) must never release a hold under the current one.
        import tools.clinical_auditor as ca
        r = _result(_INTERP)
        current = ca.corrections_fingerprint(r)
        with mock.patch.object(ca, "_AUDIT_PROTOCOL_VERSION", 1):
            old = ca.corrections_fingerprint(r)
        self.assertNotEqual(current, old)

    # ── note-evidence verification (production hardening) ──
    _NOTE = ("Assessment: retrocalcaneal exostosis of the right heel. "
             "The Achilles tendon was debrided of degenerative tissue.")

    def _uphold_with_evidence(self, ev):
        return [{"items": [{"index": 0, "verdict": "uphold",
                            "authority": "ICD-10-CM Tabular",
                            "note_evidence": ev}],
                 "claim_level_concerns": "", "overall_rationale": ""}]

    def test_fabricated_quote_degrades_to_uncertain(self):
        result = _result(_INTERP)
        block, _ = self._audit(
            result,
            self._uphold_with_evidence(
                "the plantar fascia was released at its origin"),
            note=self._NOTE)
        self.assertEqual(block["verdict"], "disputed",
                         "a quote absent from the note must not uphold")
        self.assertEqual(result["final_disposition"], "REVIEW")

    def test_verbatim_quote_upholds(self):
        result = _result(_INTERP)
        block, _ = self._audit(
            result,
            self._uphold_with_evidence(
                "The Achilles tendon was debrided of degenerative tissue"),
            note=self._NOTE)
        self.assertEqual(block["verdict"], "upheld")

    def test_paraphrased_quote_still_supported(self):
        result = _result(_INTERP)
        # reordered, articles dropped, lower-cased — honest paraphrase
        block, _ = self._audit(
            result,
            self._uphold_with_evidence(
                "achilles tendon debrided of degenerative tissue"),
            note=self._NOTE)
        self.assertEqual(block["verdict"], "upheld")

    def test_absence_evidence_upholds_without_a_quote(self):
        result = _result(_INTERP)
        block, _ = self._audit(
            result,
            self._uphold_with_evidence(
                "no acute rupture is documented anywhere in the note"),
            note=self._NOTE)
        self.assertEqual(block["verdict"], "upheld",
                         "absence-grounded evidence is carved out")

    def test_no_note_text_does_not_manufacture_dispute(self):
        from tools.clinical_auditor import _evidence_supported
        self.assertTrue(_evidence_supported(
            {"note_evidence": "anything at all here"}, ""))

    def test_unit_supported_matcher(self):
        from tools.clinical_auditor import _evidence_supported as sup
        note = self._NOTE
        self.assertTrue(sup({"note_evidence":
                             "retrocalcaneal exostosis"}, note))
        self.assertFalse(sup({"note_evidence":
                              "bunion of the great toe"}, note))
        self.assertTrue(sup({"note_evidence":
                             "no fracture seen"}, note))


class FullRecordCaseTest(unittest.TestCase):
    """The reviewer's case file carries the COMPLETE saved record — the
    same artifact a human reads when handed the output JSON. The curation
    that hid the routine_00008 contradictions (adjudication decision vs
    claim line, run votes vs disposition) from the in-pipeline review must
    never come back."""

    def _case(self):
        from tools.clinical_auditor import assemble_case
        r = _result(_INTERP)
        r["adjudication"] = {"items": [
            {"array": "cpt_codes", "code": "28118", "kind": "attributes",
             "decision": "set", "fields": {"modifiers": []}}]}
        r["clinical_audit"] = {"verdict": "upheld", "fingerprint": "prior"}
        r["rag_context"] = {"note_full_text": "the full note text",
                            "corrections_made": ["kept"]}
        return assemble_case("note_x", r, "Assessment: calcaneal spur.",
                             rep=object())

    def test_full_record_present_with_metadata_fields(self):
        case = self._case()
        rec = case["full_record"]
        self.assertEqual(rec["consistency"], {"runs": 3, "unanimous": True})
        self.assertEqual(
            rec["adjudication"]["items"][0]["fields"], {"modifiers": []})
        self.assertEqual(rec["final_disposition"], "CLEAN")
        self.assertEqual(rec["material_corrections"], _INTERP)

    def test_prior_audit_block_excluded(self):
        # a stale verdict must not anchor the fresh review
        self.assertNotIn("clinical_audit", self._case()["full_record"])

    def test_duplicated_note_text_replaced_not_dropped(self):
        rec = self._case()["full_record"]
        self.assertNotEqual(rec["rag_context"]["note_full_text"],
                            "the full note text")
        self.assertEqual(rec["rag_context"]["corrections_made"], ["kept"])

    def test_source_result_not_mutated(self):
        from tools.clinical_auditor import assemble_case
        r = _result(_INTERP)
        r["clinical_audit"] = {"verdict": "upheld"}
        r["rag_context"] = {"note_full_text": "the full note text"}
        assemble_case("note_x", r, "note", rep=object())
        self.assertEqual(r["clinical_audit"], {"verdict": "upheld"})
        self.assertEqual(r["rag_context"]["note_full_text"],
                         "the full note text")


# ---------------------------------------------------------------------------
# 5. CLEAN requires PASSING the audit (pending hold + promotion)
# ---------------------------------------------------------------------------

class AuditPendingGateTest(unittest.TestCase):
    """EVERY scrub-CLEAN claim is HELD at REVIEW under the pending marker
    until the clinical review upholds it — the review stage sits inside
    the CLEAN path, not after it, and it is universal (the absence of
    recorded corrections is what an unreported mutation looks like)."""

    def _apply(self, corrections, audit=None, clean=True):
        from types import SimpleNamespace
        from enum import Enum
        from app.pipeline import MedicalCodingPipeline

        class D(Enum):
            CLEAN = "CLEAN"
            REVIEW = "REVIEW"
        scrub = SimpleNamespace(
            clean=clean,
            disposition=D.CLEAN if clean else D.REVIEW,
            summary="scrub summary",
            blocking_findings=[],
            # accepts the same kwargs as pydantic's model_dump — the
            # pipeline calls model_dump(mode="json") (Status-enum
            # serialization fix), and a stub that rejects kwargs turns
            # every call through this path into a TypeError
            model_dump=lambda **kw: {"clean": clean,
                                     "disposition":
                                         "CLEAN" if clean else "REVIEW",
                                     "summary": "scrub summary"})
        dump = dict(_result(corrections), clinical_audit=audit or {})
        result = SimpleNamespace(
            claim_scrub=None, final_disposition="", final_summary="",
            material_corrections=corrections,
            clinical_audit=audit or {},
            auto_coding_tier="", auto_coding_summary="",
            auto_coding_review_reasons=[], auto_coding_confidence=0.9,
            model_dump=lambda: dump)
        MedicalCodingPipeline._apply_scrub_verdict(
            MedicalCodingPipeline.__new__(MedicalCodingPipeline), result, scrub)
        return result

    def _current_audit(self, corrections):
        from tools.clinical_auditor import corrections_fingerprint
        return {"verdict": "upheld",
                "fingerprint": corrections_fingerprint(
                    _result(corrections))}

    def test_interpretive_corrections_hold_clean_at_review(self):
        r = self._apply(_INTERP)
        self.assertEqual(r.final_disposition, "REVIEW")
        self.assertEqual(r.auto_coding_tier, "REVIEW")
        self.assertTrue(any("[clinical_audit/pending]" in s
                            for s in r.auto_coding_review_reasons))
        self.assertLessEqual(r.auto_coding_confidence, 0.84)

    def test_current_upheld_audit_passes_straight_to_clean(self):
        r = self._apply(_INTERP, audit=self._current_audit(_INTERP))
        self.assertEqual(r.final_disposition, "CLEAN")
        self.assertEqual(r.auto_coding_tier, "AUTO")

    def test_stale_upheld_audit_still_holds(self):
        r = self._apply(_INTERP, audit={"verdict": "upheld",
                                        "fingerprint": "stale0000000000"})
        self.assertEqual(r.final_disposition, "REVIEW")
        self.assertTrue(any("[clinical_audit/pending]" in s
                            for s in r.auto_coding_review_reasons))

    def test_data_grounded_claims_also_held(self):
        r = self._apply(_DATA)
        self.assertEqual(r.final_disposition, "REVIEW")
        self.assertTrue(any("[clinical_audit/pending]" in s
                            for s in r.auto_coding_review_reasons))
        r = self._apply(_DATA, audit=self._current_audit(_DATA))
        self.assertEqual(r.final_disposition, "CLEAN")

    def test_scrub_review_never_held_pending(self):
        r = self._apply(_INTERP, clean=False)
        self.assertEqual(r.final_disposition, "REVIEW")
        self.assertFalse(any("[clinical_audit/pending]" in s
                             for s in r.auto_coding_review_reasons))


class AuditPromotionTest(unittest.TestCase):
    """The audit's uphold verdict is the ONLY thing that releases the
    pending hold; disputes replace it; other review verdicts (routing)
    are never overridden."""

    def _held(self, routed=False):
        r = _result(_INTERP, disposition="REVIEW")
        r["claim_scrub"] = {"clean": True, "disposition": "CLEAN",
                            "summary": "all filters passed"}
        r["auto_coding_review_reasons"] = [
            "[clinical_audit/pending] 1 interpretive layer correction(s) "
            "await the clinical-correctness audit"]
        r["auto_coding_confidence"] = 0.8
        if routed:
            r["review_routing"] = "routed"
            r["consistency"] = {"runs": 3, "unanimous": False}
        return r

    def _audit(self, result, verdict):
        import tools.clinical_auditor as ca
        v = {"items": [{"index": 0, "verdict": verdict,
                        "authority": "ICD-10-CM Index",
                        "note_evidence": "documented in the note"}],
             "claim_level_concerns": "", "overall_rationale": ""}
        note = "Assessment: the finding is documented in the note, right foot."
        with mock.patch.object(
                ca, "_audit_once",
                side_effect=lambda case, pass_idx=0: dict(v, _model="t")):
            return ca.audit_result("note_x", result, note, rep=object(),
                                   passes=1)

    def test_uphold_promotes_held_claim_to_clean(self):
        r = self._held()
        block = self._audit(r, "uphold")
        self.assertEqual(block["verdict"], "upheld")
        self.assertEqual(r["final_disposition"], "CLEAN")
        self.assertEqual(r["auto_coding_tier"], "AUTO")
        self.assertGreaterEqual(r["auto_coding_confidence"], 0.85)
        self.assertFalse(any("[clinical_audit/pending]" in s
                             for s in r["auto_coding_review_reasons"]))

    def test_dispute_replaces_hold_with_named_correction(self):
        r = self._held()
        block = self._audit(r, "overturn")
        self.assertEqual(block["verdict"], "disputed")
        self.assertEqual(r["final_disposition"], "REVIEW")
        reasons = r["auto_coding_review_reasons"]
        self.assertFalse(any("[clinical_audit/pending]" in s
                             for s in reasons))
        self.assertTrue(any("[clinical_audit/overturn]" in s
                            for s in reasons))

    def test_repeated_dispute_never_duplicates_reasons(self):
        # The convergence loop re-reviews a note every iteration; each
        # verdict SUPERSEDES the last, so its reason entries must replace
        # the previous ones, not accumulate (measured live on
        # routine_00001: three review passes, three copies per finding).
        r = self._held()
        self._audit(r, "overturn")
        first = [s for s in r["auto_coding_review_reasons"]
                 if "[clinical_audit/" in s]
        self._audit(r, "overturn")
        second = [s for s in r["auto_coding_review_reasons"]
                  if "[clinical_audit/" in s]
        self.assertEqual(first, second,
                         "re-reviews must supersede, never duplicate")

    def test_uphold_never_overrides_consistency_routing(self):
        r = self._held(routed=True)
        self._audit(r, "uphold")
        self.assertEqual(r["final_disposition"], "REVIEW",
                         "a routed (non-unanimous) note must stay routed")
        self.assertFalse(any("[clinical_audit/pending]" in s
                             for s in r["auto_coding_review_reasons"]))


class AuditSkipPathTest(unittest.TestCase):
    """audit_batch's fingerprint skip must still REALIZE the stored
    verdict: a re-scrub can re-place the pending hold after the audit
    already upheld these exact corrections — the skip path releases it
    again without LLM spend."""

    def test_skip_path_promotes_re_held_claim(self):
        import json as _json
        import tempfile
        from pathlib import Path
        import tools.clinical_auditor as ca

        r = _result(_INTERP, disposition="REVIEW")
        r["claim_scrub"] = {"clean": True, "disposition": "CLEAN",
                            "summary": "all filters passed"}
        r["auto_coding_review_reasons"] = ["[clinical_audit/pending] held"]
        r["clinical_audit"] = {
            "verdict": "upheld",
            "fingerprint": ca.corrections_fingerprint(r),
            "items": [{"index": 0, "verdict": "uphold",
                       "authority": "a", "note_evidence": "b"}],
            "claim_level_concerns": "",
        }
        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / "note_x_results.json"
            f.write_text(_json.dumps(r))
            with mock.patch.object(
                    ca, "_audit_once",
                    side_effect=AssertionError("must not call the LLM")):
                stats = ca.audit_batch(Path(td), docs=["note_x"],
                                       rep=object())
            saved = _json.loads(f.read_text())
        self.assertEqual(stats["skipped"], 1)
        self.assertEqual(saved["final_disposition"], "CLEAN")
        self.assertEqual(saved["auto_coding_tier"], "AUTO")


# ---------------------------------------------------------------------------
# 6. the auditor GROWS: disputes feed the actuation queue
# ---------------------------------------------------------------------------

class AuditDisputeQueueTest(unittest.TestCase):
    """A disputed correction becomes an audit_dispute flip class — the
    same queue and actuation machinery as consistency flips — held at
    awaiting_verification until a human/adjudicated registry claim
    exists to serve as the realignment target."""

    def _dir_with_disputed(self, td, doc="note_x"):
        import json as _json
        from pathlib import Path
        r = _result(_INTERP, disposition="REVIEW")
        r["document_id"] = doc
        r["clinical_audit"] = {
            "verdict": "disputed",
            "items": [{"index": 0, "verdict": "overturn",
                       "authority": "ICD-10-CM Index",
                       "note_evidence": "tourniquet sentence"}],
            "claim_level_concerns": "",
        }
        r["rag_context"] = {"note_full_text":
                            "Assessment: calcaneal spur. Swapped nothing."}
        (Path(td) / f"{doc}_results.json").write_text(_json.dumps(r))
        return Path(td)

    def test_dispute_enqueues_awaiting_verification(self):
        import tempfile
        from pathlib import Path
        import tools.flip_triage as ft
        with tempfile.TemporaryDirectory() as td:
            results = self._dir_with_disputed(td)
            q = Path(td) / "queue.jsonl"
            with mock.patch.object(ft, "_verified_docs",
                                   return_value=set()):
                stats = ft.scan(results, queue_path=q)
            classes = ft.load_queue(q)
        self.assertEqual(stats["audit_disputes_seen"], 1)
        audit_classes = [c for c in classes
                         if c["kind"] == "audit_dispute"]
        self.assertEqual(len(audit_classes), 1)
        cls = audit_classes[0]
        self.assertEqual(cls["status"], "awaiting_verification")
        self.assertEqual(cls["code"], "M77.31")
        self.assertEqual(cls["array"], "icd_codes")
        d = cls["documents"][0]["disagreement"]
        self.assertEqual(d["kind"], "audit_dispute")
        self.assertIn("audit", d)

    def test_class_opens_once_document_verified(self):
        import tempfile
        from pathlib import Path
        import tools.flip_triage as ft
        with tempfile.TemporaryDirectory() as td:
            results = self._dir_with_disputed(td)
            q = Path(td) / "queue.jsonl"
            with mock.patch.object(ft, "_verified_docs",
                                   return_value=set()):
                ft.scan(results, queue_path=q)
            # a human verifies the claim -> the next scan opens the class
            with mock.patch.object(ft, "_verified_docs",
                                   return_value={"note_x"}):
                stats = ft.scan(results, queue_path=q)
            classes = ft.load_queue(q)
        cls = [c for c in classes if c["kind"] == "audit_dispute"][0]
        self.assertEqual(cls["status"], "open")
        self.assertEqual(stats.get("opened_after_verification"), 1)

    def test_class_opens_once_advisory_target_verified(self):
        # an advisory-shaped dispute can NEVER get a registry claim or a
        # per-code row (the claim is correct as billed) — its verified
        # emission target must open the class, or it deadlocks forever
        import tempfile
        from pathlib import Path
        import tools.flip_triage as ft
        with tempfile.TemporaryDirectory() as td:
            results = self._dir_with_disputed(td)
            q = Path(td) / "queue.jsonl"
            with mock.patch.object(ft, "_verified_docs",
                                   return_value=set()), \
                 mock.patch.object(ft, "_verified_code_targets",
                                   return_value={}), \
                 mock.patch.object(ft, "_verified_advisory_targets",
                                   return_value={}):
                ft.scan(results, queue_path=q)
            with mock.patch.object(ft, "_verified_docs",
                                   return_value=set()), \
                 mock.patch.object(ft, "_verified_code_targets",
                                   return_value={}), \
                 mock.patch.object(ft, "_verified_advisory_targets",
                                   return_value={"note_x": {
                                       ("advisory_emission",
                                        "MEDICAL_NECESSITY|M77.31"):
                                           False}}):
                stats = ft.scan(results, queue_path=q)
            classes = ft.load_queue(q)
        cls = [c for c in classes if c["kind"] == "audit_dispute"][0]
        self.assertEqual(cls["status"], "open")
        self.assertEqual(stats.get("opened_after_verification"), 1)

    def test_upheld_audits_enqueue_nothing(self):
        import json as _json
        import tempfile
        from pathlib import Path
        import tools.flip_triage as ft
        with tempfile.TemporaryDirectory() as td:
            r = _result(_INTERP)
            r["clinical_audit"] = {"verdict": "upheld", "items": [
                {"index": 0, "verdict": "uphold", "authority": "a",
                 "note_evidence": "b"}]}
            (Path(td) / "note_x_results.json").write_text(_json.dumps(r))
            q = Path(td) / "queue.jsonl"
            with mock.patch.object(ft, "_verified_docs",
                                   return_value=set()):
                stats = ft.scan(Path(td), queue_path=q)
            classes = ft.load_queue(q)
        self.assertEqual(stats["audit_disputes_seen"], 0)
        self.assertEqual([c for c in classes
                          if c["kind"] == "audit_dispute"], [])


# ---------------------------------------------------------------------------
# 7. diff-derived mutation capture — no layer can act unseen
# ---------------------------------------------------------------------------

class DerivedMutationTest(unittest.TestCase):
    """material_corrections is derived by diffing claim state, not by
    trusting layer self-reporting — the structural fix for the
    routine_00003 hole (an unreported demotion escaped the audit)."""

    def _diff(self, pre_icd, post_icd, coding_result=None, issues=None):
        v = _validator()
        pre = v._pre_validation_snapshot(pre_icd, [], [])
        for sev, code, cat, msg in (issues or []):
            v._add(sev, code, cat, msg, "verify")
        return v._material_corrections_with_derived(
            pre, post_icd, [], [], coding_result or {})

    def test_unreported_removal_is_derived_interpretive(self):
        pre = [{"code": "B35.1", "type": "primary"},
               {"code": "R26.2", "type": "secondary"}]
        post = [{"code": "B35.1", "type": "primary"}]
        mats = self._diff(pre, post, coding_result={
            "supporting_conditions": [
                {"code": "R26.2",
                 "review_reason": "Demoted from billed diagnoses: NEC "
                                  "residual-category code not documented"}]})
        derived = [m for m in mats if m.get("derived")]
        self.assertEqual(len(derived), 1)
        self.assertEqual(derived[0]["code"], "R26.2")
        self.assertEqual(derived[0]["action"], "derived_removal")
        self.assertTrue(derived[0]["interpretive"],
                        "unknown provenance must be audited — fail closed")
        self.assertIn("Demoted from billed diagnoses",
                      derived[0]["message"])

    def test_reported_removal_is_not_duplicated(self):
        pre = [{"code": "B35.1", "type": "primary"},
               {"code": "L60.3", "type": "secondary"}]
        post = [{"code": "B35.1", "type": "primary"}]
        mats = self._diff(pre, post, issues=[
            ("WARNING", "L60.3", "icd_subsumption",
             "AUTO-CORRECTED: L60.3 removed as subsumed")])
        self.assertEqual([m for m in mats if m.get("derived")], [])

    def test_swap_message_covers_both_sides(self):
        pre = [{"code": "M21.6X1", "type": "secondary"}]
        post = [{"code": "M21.851", "type": "secondary"}]
        mats = self._diff(pre, post, issues=[
            ("WARNING", "M21.851", "sibling_matches_note_better",
             "AUTO-CORRECTED: M21.6X1 -> M21.851 per documented attribute")])
        self.assertEqual([m for m in mats if m.get("derived")], [],
                         "a reported swap accounts for old AND new code")

    def test_primary_retype_covered_by_was_pattern(self):
        pre = [{"code": "M77.31", "type": "primary"},
               {"code": "M76.61", "type": "secondary"}]
        post = [{"code": "M77.31", "type": "secondary"},
                {"code": "M76.61", "type": "primary"}]
        mats = self._diff(pre, post, issues=[
            ("WARNING", "M76.61", "primary_designation",
             "AUTO-CORRECTED: M76.61 designated primary (was M77.31)")])
        self.assertEqual([m for m in mats if m.get("derived")], [])

    def test_unreported_retype_is_derived(self):
        pre = [{"code": "M77.31", "type": "primary"},
               {"code": "M76.61", "type": "secondary"}]
        post = [{"code": "M77.31", "type": "secondary"},
                {"code": "M76.61", "type": "primary"}]
        mats = self._diff(pre, post)
        derived = {m["code"] for m in mats if m.get("derived")}
        self.assertEqual(derived, {"M77.31", "M76.61"},
                         "silent primary flips must surface for audit")

    def test_unreported_addition_is_derived(self):
        pre = [{"code": "B35.1", "type": "primary"}]
        post = [{"code": "B35.1", "type": "primary"},
                {"code": "L60.1", "type": "secondary"}]
        mats = self._diff(pre, post)
        derived = [m for m in mats if m.get("derived")]
        self.assertEqual(len(derived), 1)
        self.assertEqual(derived[0]["action"], "derived_addition")

    def test_auto_added_issues_are_recorded_corrections(self):
        v = _validator()
        v._add("WARNING", "L60.1", "documented_nail_finding_completion",
               "AUTO-ADDED: L60.1 ('Onycholysis') — documented finding",
               "confirm")
        mats = v._material_corrections()
        self.assertEqual(len(mats), 1)
        self.assertEqual(mats[0]["action"], "auto_addition")
        self.assertTrue(mats[0]["interpretive"],
                        "a completion election reads the note")


# ---------------------------------------------------------------------------
# 8. whole-claim findings — gates, routing, growth
# ---------------------------------------------------------------------------

class ClaimFindingsTest(unittest.TestCase):
    _NOTE = ("Assessment: onychomycosis with pain, 4 nails. She reports "
             "pain with closed shoe wear and difficulty ambulating.")

    def _finding(self, materiality="billing_material", code="R26.2",
                 evidence="difficulty ambulating", authority="LCD policy"):
        return {"kind": "missing_code", "array": "icd_codes", "code": code,
                "materiality": materiality, "finding": "covered secondary "
                "for the ambulation pathway is missing from the claim",
                "authority": authority, "note_evidence": evidence}

    def _audit(self, result, findings, items=None):
        import tools.clinical_auditor as ca
        v = {"items": items or [], "claim_findings": findings,
             "claim_level_concerns": "", "overall_rationale": ""}
        with mock.patch.object(
                ca, "_audit_once",
                side_effect=lambda case, pass_idx=0: dict(v, _model="t")):
            return ca.audit_result("note_x", result, self._NOTE,
                                   rep=object(), passes=1)

    def test_billing_material_finding_forces_review(self):
        result = _result([])
        block = self._audit(result, [self._finding()])
        self.assertEqual(block["verdict"], "disputed")
        self.assertEqual(result["final_disposition"], "REVIEW")
        self.assertTrue(any("[clinical_audit/finding/missing_code]" in s
                            for s in result["auto_coding_review_reasons"]))

    def test_advisory_finding_never_routes(self):
        result = _result([])
        block = self._audit(result, [self._finding(
            materiality="advisory")])
        self.assertEqual(block["verdict"], "upheld")
        self.assertEqual(result["final_disposition"], "CLEAN")
        self.assertEqual(len(block["claim_findings"]), 1,
                         "advisory findings are still recorded for growth")

    def test_ungrounded_material_finding_still_routes(self):
        result = _result([])
        block = self._audit(result, [self._finding(authority="",
                                                   evidence="")])
        self.assertEqual(block["verdict"], "disputed",
                         "fail closed: ungrounded but billing-material")
        self.assertEqual(
            block["claim_findings"][0]["materiality"], "uncertain")

    def test_fabricated_quote_in_advisory_finding_drops_it(self):
        result = _result([])
        block = self._audit(result, [self._finding(
            materiality="advisory",
            evidence="the tibial sesamoid was excised intact")])
        self.assertEqual(block["claim_findings"], [],
                         "an ungrounded advisory finding is pure noise")
        self.assertEqual(block["verdict"], "upheld")

    def test_clean_claim_empty_findings_upholds(self):
        result = _result([])
        block = self._audit(result, [])
        self.assertEqual(block["verdict"], "upheld")
        self.assertEqual(result["final_disposition"], "CLEAN")


class FindingQueueTest(unittest.TestCase):
    """Whole-claim findings feed the SAME audit_dispute actuation queue as
    disputed corrections — confirmed review findings become deterministic
    rules/templates through the acceptance gates, never patches."""

    def _dir_with_finding(self, td, materiality, doc="note_x"):
        import json as _json
        from pathlib import Path
        r = _result([], disposition="REVIEW")
        r["document_id"] = doc
        r["clinical_audit"] = {
            "verdict": ("disputed" if materiality == "billing_material"
                        else "upheld"),
            "items": [],
            "claim_findings": [{
                "kind": "missing_code", "array": "icd_codes",
                "code": "R26.2", "materiality": materiality,
                "finding": "covered secondary missing",
                "authority": "LCD", "note_evidence": "difficulty "
                "ambulating"}],
            "claim_level_concerns": "",
        }
        r["rag_context"] = {"note_full_text":
                            "She reports difficulty ambulating."}
        (Path(td) / f"{doc}_results.json").write_text(_json.dumps(r))
        return Path(td)

    def _scan(self, td):
        from pathlib import Path
        import tools.flip_triage as ft
        q = Path(td) / "queue.jsonl"
        with mock.patch.object(ft, "_verified_docs", return_value=set()):
            stats = ft.scan(Path(td), queue_path=q)
        return stats, ft.load_queue(q)

    def test_material_finding_enqueues_audit_class(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            self._dir_with_finding(td, "billing_material")
            stats, classes = self._scan(td)
        cls = [c for c in classes if c["kind"] == "audit_dispute"]
        self.assertEqual(len(cls), 1)
        self.assertEqual(cls[0]["code"], "R26.2")
        self.assertEqual(cls[0]["status"], "awaiting_verification")
        d = cls[0]["documents"][0]["disagreement"]
        self.assertIn("finding", d)
        self.assertEqual(d["finding"]["kind"], "missing_code")

    def test_advisory_finding_also_enqueues(self):
        # An upheld claim with a grounded advisory finding (e.g. a system
        # advisory whose recommendation is wrong) still grows the queue.
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            self._dir_with_finding(td, "advisory")
            stats, classes = self._scan(td)
        cls = [c for c in classes if c["kind"] == "audit_dispute"]
        self.assertEqual(len(cls), 1)
        self.assertEqual(stats["audit_disputes_seen"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
