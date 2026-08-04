from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from medical_coder.compiler import SourceCompiler, SourceIntegrityError


class SourceFailureTestCase(unittest.TestCase):
    def test_reversed_period_fails_without_explicit_adapter_policy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "source.json").write_text(
                json.dumps(
                    [
                        {
                            "identifier": "test-item",
                            "label": "test description",
                            "start": "2025-01-01",
                            "end": "2024-12-31",
                        }
                    ]
                ),
                encoding="utf-8",
            )
            pack = {
                "pack_id": "failure-pack",
                "schema_version": 1,
                "artifacts": [
                    {
                        "path": "source.json",
                        "authority": "test",
                        "system": "test",
                        "version": "one",
                        "code_field": "identifier",
                        "display_fields": ["label"],
                        "effective_start_field": "start",
                        "effective_end_field": "end",
                        "billable_default": True,
                    }
                ],
            }
            (root / "pack.json").write_text(json.dumps(pack), encoding="utf-8")
            with self.assertRaises(SourceIntegrityError):
                SourceCompiler(root).compile(root / "pack.json", root / "snapshots")


if __name__ == "__main__":
    unittest.main()

