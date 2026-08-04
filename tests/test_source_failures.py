from __future__ import annotations

import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from medical_coder.compiler import SourceCompiler, SourceIntegrityError


class SourceFailureTestCase(unittest.TestCase):
    def test_interrupted_compile_removes_partial_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data").mkdir()
            (root / "data" / "artifacts.json").write_text("[]", encoding="utf-8")
            pack = root / "pack.json"
            pack.write_text(json.dumps({
                "pack_id": "interrupt-test", "schema_version": 3, "capabilities": {},
                "artifacts": [{
                    "path": "data/artifacts.json", "authority": "test", "system": "test",
                    "version": "one", "code_field": "code", "display_fields": ["display"],
                }],
            }), encoding="utf-8")
            output = root / "snapshots"
            compiler = SourceCompiler(root)
            with patch.object(compiler, "_compile_artifacts", side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    compiler.compile(pack, output)
            self.assertEqual(list(output.iterdir()), [])

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
                "schema_version": 3,
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

