"""Record coherence gate — every check judges a record's self-agreement,
so every fixture here is a record that disagrees with itself in exactly
one way (codes are placeholders, never real medical codes)."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.record_coherence import (COHERENCE_MARKER,  # noqa: E402
                                    coherence_violations,
                                    enforce_coherence, sweep)


def _record(**over) -> dict:
    base = {
        "document_id": "docz", "success": True,
        "final_disposition": "CLEAN", "auto_coding_tier": "AUTO",
        "auto_coding_confidence": 0.9,
        "claim_scrub": {"clean": True, "disposition": "CLEAN"},
        "consistency": {"runs": 3, "unanimous": True},
        "clinical_audit": {"verdict": "upheld", "fingerprint": "x"},
        "icd_codes": [{"code": "AAA.1", "type": "primary"}],
        "cpt_codes": [{"code": "11111", "modifiers": [], "units": 1,
                       "linked_diagnoses": ["AAA.1"]}],
        "hcpcs_codes": [],
        "material_corrections": [],
    }
    base.update(over)
    return base


class CoherenceChecksTest(unittest.TestCase):
    def test_coherent_record_has_no_violations(self):
        self.assertEqual(coherence_violations(_record()), [])

    def test_clean_with_non_clean_scrub(self):
        r = _record(claim_scrub={"clean": False, "disposition": "REVIEW"})
        self.assertTrue(any("claim scrub" in v
                            for v in coherence_violations(r)))

    def test_clean_with_split_consistency(self):
        r = _record(consistency={"runs": 3, "unanimous": False})
        self.assertTrue(any("consistency runs disagree" in v
                            for v in coherence_violations(r)))

    def test_clean_but_routed_to_human(self):
        r = _record(review_routing="routed")
        self.assertTrue(any("routed to human review" in v
                            for v in coherence_violations(r)))

    def test_clean_without_upheld_review(self):
        r = _record(clinical_audit={"verdict": "disputed"})
        self.assertTrue(any("nothing released the claim" in v
                            for v in coherence_violations(r)))
        # ... unless the caller is the review's own promotion path
        self.assertEqual(coherence_violations(
            r, require_audit_release=False), [])

    def test_overridden_adjudication_is_incoherent(self):
        r = _record(adjudication={"items": [
            {"array": "cpt_codes", "code": "11111", "kind": "attributes",
             "decision": "set", "fields": {"modifiers": ["XX"]},
             "authority": "a", "note_evidence": "e"}]})
        self.assertTrue(any("adjudication decided" in v
                            for v in coherence_violations(r)))

    def test_removed_code_still_billed(self):
        r = _record(material_corrections=[
            {"category": "c", "code": "11111", "action": "removal",
             "message": "line suppressed from the claim"}])
        self.assertTrue(any("still billed" in v
                            for v in coherence_violations(r)))

    def test_added_code_not_billed(self):
        r = _record(material_corrections=[
            {"category": "c", "code": "22222", "action": "auto_addition",
             "message": "AUTO-ADDED fixture"}])
        self.assertTrue(any("not billed" in v
                            for v in coherence_violations(r)))

    def test_fresh_correction_outranks_carried_one(self):
        # prior pass removed the code, the fresh pass re-added it —
        # the fresh decision governs, so a billed code is coherent
        r = _record(material_corrections=[
            {"category": "c", "code": "11111", "action": "auto_addition",
             "message": "AUTO-ADDED fixture"},
            {"category": "c", "code": "11111", "action": "removal",
             "message": "old", "carried_from_prior_pass": True}])
        self.assertEqual(coherence_violations(r), [])

    def test_dangling_linked_diagnosis(self):
        r = _record(cpt_codes=[{"code": "11111", "modifiers": [],
                                "units": 1,
                                "linked_diagnoses": ["BBB.9"]}])
        self.assertTrue(any("not on the claim's diagnosis list" in v
                            for v in coherence_violations(r)))

    def test_zero_and_two_primaries(self):
        r = _record(icd_codes=[{"code": "AAA.1", "type": "secondary"}],
                    cpt_codes=[])
        self.assertTrue(any("first-listed" in v
                            for v in coherence_violations(r)))
        r2 = _record(icd_codes=[{"code": "AAA.1", "type": "primary"},
                                {"code": "BBB.9", "type": "primary"}],
                     cpt_codes=[])
        self.assertTrue(any("first-listed" in v
                            for v in coherence_violations(r2)))


class EnforcementTest(unittest.TestCase):
    def test_enforce_holds_clean_and_names_each_contradiction(self):
        r = _record(review_routing="routed")
        violations = enforce_coherence(r)
        self.assertEqual(len(violations), 1)
        self.assertEqual(r["final_disposition"], "REVIEW")
        self.assertTrue(any(COHERENCE_MARKER in x
                            for x in r["auto_coding_review_reasons"]))

    def test_enforce_leaves_coherent_record_untouched(self):
        r = _record()
        before = json.dumps(r, sort_keys=True)
        self.assertEqual(enforce_coherence(r), [])
        self.assertEqual(json.dumps(r, sort_keys=True), before)

    def test_upheld_review_cannot_promote_incoherent_record(self):
        from tools.clinical_auditor import _enforce_verdict
        r = _record(final_disposition="CLEAN",
                    cpt_codes=[{"code": "11111", "modifiers": [],
                                "units": 1,
                                "linked_diagnoses": ["BBB.9"]}])
        _enforce_verdict(r, {"verdict": "upheld"}, [], {}, "")
        self.assertEqual(r["final_disposition"], "REVIEW")
        self.assertTrue(any(COHERENCE_MARKER in x
                            for x in r["auto_coding_review_reasons"]))

    def test_registry_gate_fails_closed_on_incoherence(self):
        from tools.claims_registry import eligible_for_auto
        from tools.clinical_auditor import corrections_fingerprint
        r = _record(consistency={"runs": 3, "unanimous": True},
                    review_routing="routed")
        r["clinical_audit"]["fingerprint"] = corrections_fingerprint(r)
        ok, why = eligible_for_auto(r)
        self.assertFalse(ok)
        self.assertIn("contradicts itself", why)

    def test_sweep_report_only_never_writes(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            f = d / "docz_results.json"
            f.write_text(json.dumps(_record(review_routing="routed")))
            before = f.read_text()
            stats = sweep(d, report_only=True)
            self.assertEqual(stats["incoherent"], 1)
            self.assertEqual(f.read_text(), before)

    def test_sweep_enforce_holds_violators(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            f = d / "docz_results.json"
            f.write_text(json.dumps(_record(review_routing="routed")))
            stats = sweep(d)
            self.assertEqual(stats["held"], 1)
            saved = json.loads(f.read_text())
            self.assertEqual(saved["final_disposition"], "REVIEW")


if __name__ == "__main__":
    unittest.main()
