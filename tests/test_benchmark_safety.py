"""Gold promotion and coverage metrics fail closed."""

import json
import tempfile
import unittest
from pathlib import Path

from tools.benchmark_ab import METRICS, score_dir, score_note


class BenchmarkSafetyTest(unittest.TestCase):
    def test_missing_candidate_case_scores_zero(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            gold, candidate = root / "gold", root / "candidate"
            gold.mkdir()
            candidate.mkdir()
            payload = {"document_id": "case", "icd_codes": [],
                       "cpt_codes": [], "hcpcs_codes": [],
                       "final_disposition": "REVIEW"}
            (gold / "case_results.json").write_text(json.dumps(payload))
            score = score_dir(candidate, gold, verbose=False)
            self.assertEqual(score["notes_scored"], 1)
            self.assertEqual(score["notes_missing"], 1)
            self.assertTrue(all(score[m] == 0.0 for m in METRICS))

    def test_linkage_and_diagnosis_order_are_scored(self):
        gold = {
            "icd_codes": [{"code": "A", "type": "primary"},
                          {"code": "B", "type": "secondary"}],
            "cpt_codes": [{"code": "P", "linked_diagnoses": ["A"]}],
            "hcpcs_codes": [], "final_disposition": "CLEAN",
        }
        candidate = {
            **gold,
            "icd_codes": list(reversed(gold["icd_codes"])),
            "cpt_codes": [{"code": "P", "linked_diagnoses": ["B"]}],
        }
        score = score_note(gold, candidate)
        self.assertEqual(score["icd_sequence"], 0.0)
        self.assertEqual(score["cpt_linkage"], 0.0)


if __name__ == "__main__":
    unittest.main()
