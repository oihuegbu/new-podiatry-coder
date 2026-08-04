from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from medical_coder.snomed import SnomedIntegrityError, prepare_snomed_us


class SnomedPreparationTestCase(unittest.TestCase):
    def _archive(self, root: Path, *, valid_relationship_header: bool = True) -> Path:
        archive_path = root / "SnomedCT_Test_20260301T120000Z.zip"
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.writestr(
                "release/Snapshot/Terminology/sct2_Concept_Snapshot_Test.txt",
                "id\teffectiveTime\tactive\tmoduleId\tdefinitionStatusId\n"
                "101\t20260301\t1\tmodule\tdefinition\n"
                "102\t20260301\t1\tmodule\tdefinition\n"
                "103\t20260301\t1\tmodule\tdefinition\n"
                "104\t20260301\t0\tmodule\tdefinition\n",
            )
            archive.writestr(
                "release/Snapshot/Terminology/sct2_Description_Snapshot-en_Test.txt",
                "id\teffectiveTime\tactive\tmoduleId\tconceptId\tlanguageCode\ttypeId\tterm\tcaseSignificanceId\n"
                "201\t20260301\t1\tmodule\t101\ten\ttype\tAlpha finding\tcase\n"
                "202\t20260301\t1\tmodule\t101\ten\ttype\tFinding alpha\tcase\n"
                "203\t20260301\t1\tmodule\t102\ten\ttype\tBeta structure\tcase\n"
                "206\t20260301\t1\tmodule\t102\ten\ttype\t\"Quoted synonym\tcase\n"
                "204\t20260301\t1\tmodule\t103\ten\ttype\tRelationship type\tcase\n"
                "205\t20260301\t1\tmodule\t104\ten\ttype\tInactive concept term\tcase\n",
            )
            relationship_header = (
                "id\teffectiveTime\tactive\tmoduleId\tsourceId\tdestinationId\trelationshipGroup\ttypeId\tcharacteristicTypeId\tmodifierId\n"
                if valid_relationship_header else
                "id\teffectiveTime\tactive\n"
            )
            archive.writestr(
                "release/Snapshot/Terminology/sct2_Relationship_Snapshot_Test.txt",
                relationship_header + "301\t20260301\t1\tmodule\t101\t102\t0\t103\tcharacteristic\tmodifier\n",
            )
        return archive_path

    def test_compiles_only_active_english_rf2_content_with_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = self._archive(root)
            output = root / "snomed.json"
            metadata = prepare_snomed_us(
                archive, output, minimum_concepts=3, minimum_terms=5, minimum_relationships=1
            )
            value = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(metadata["release_id"], "20260301")
            self.assertEqual(len(value["concepts"]), 3)
            self.assertEqual(value["concepts"][0]["display_term"], "Alpha finding")
            self.assertNotIn("preferred_term", value["concepts"][0])
            self.assertEqual(value["terms"]["101"], ["Alpha finding", "Finding alpha"])
            self.assertNotIn("104", value["terms"])
            self.assertEqual(metadata["active_relationship_count"], 1)

    def test_rejects_changed_rf2_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(SnomedIntegrityError):
                prepare_snomed_us(
                    self._archive(root, valid_relationship_header=False),
                    root / "snomed.json",
                    minimum_concepts=3,
                    minimum_terms=5,
                    minimum_relationships=1,
                )


if __name__ == "__main__":
    unittest.main()
