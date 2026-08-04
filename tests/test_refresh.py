from __future__ import annotations

import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import Mock, patch

from medical_coder.refresh import (
    AuthorityFetcher,
    Download,
    RefreshError,
    RefreshRunner,
    fetch_nlm_authenticated_release,
    parse_delimited_archives,
    parse_umls_mrconso_candidates,
)


class RefreshTestCase(unittest.TestCase):
    @staticmethod
    def _mrconso_row(cui: str, source: str, code: str, term: str, suppressed: str = "N") -> str:
        values = [""] * 18
        values[0] = cui
        values[1] = "ENG"
        values[11] = source
        values[13] = code
        values[14] = term
        values[16] = suppressed
        return "|".join(values)

    def test_fetcher_rejects_non_authority_url_before_network(self) -> None:
        fetcher = AuthorityFetcher({"authority.example"})
        with self.assertRaises(RefreshError):
            fetcher.fetch("https://untrusted.example/source")
        with self.assertRaises(RefreshError):
            fetcher.fetch("http://authority.example/source")

    def test_header_driven_archive_parser(self) -> None:
        data = io.BytesIO()
        with zipfile.ZipFile(data, "w") as archive:
            archive.writestr(
                "table.txt",
                "banner\nColumn 1\tColumn 2\tModifier\tEffective\tDeletion\n"
                "AAAAA\tBBBBB\t0\t20260101\t*\n",
            )
        config = {
            "identifier_regex": "^[A-Z]{5}$",
            "member_regex": "(?i)\\.txt$",
            "field_aliases": {
                "left_code": ["column 1"],
                "right_code": ["column 2"],
                "modifier_indicator": ["modifier"],
                "effective_start": ["effective"],
                "effective_end": ["deletion"],
            },
        }
        download = Download("https://authority.example/table.zip", data.getvalue(), "digest", None, None)
        rows = parse_delimited_archives(config, [download], "2026-01-01")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["effective_start"], "2026-01-01")
        self.assertIsNone(rows[0]["effective_end"])

    def test_umls_bridge_is_membership_bounded_and_candidate_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "data" / "codes" / "cpt.json"
            source.parent.mkdir(parents=True)
            source.write_text(json.dumps({"codes": [{"identifier": "ALPHA"}]}), encoding="utf-8")
            data = io.BytesIO()
            rows = [
                self._mrconso_row("C1", "TARGET", "ALPHA", "Licensed descriptor"),
                self._mrconso_row("C1", "CLINICAL", "X", "Clinical synonym"),
                self._mrconso_row("C1", "CLINICAL", "X", "Suppressed synonym", "Y"),
                self._mrconso_row("C2", "TARGET", "NOTLICENSED", "Outside membership"),
                self._mrconso_row("C2", "CLINICAL", "X", "Should not cross boundary"),
            ]
            with zipfile.ZipFile(data, "w") as archive:
                archive.writestr("release/META/MRCONSO.RRF", "\n".join(rows) + "\n")
            payload = parse_umls_mrconso_candidates(
                {
                    "membership_source": "data/codes/cpt.json",
                    "membership_code_field": "identifier",
                    "target_sources": ["TARGET"],
                    "clinical_sources": ["CLINICAL"],
                },
                Download("https://authority.example/release.zip", data.getvalue(), "digest", None, None),
                root,
            )
            self.assertEqual(payload["terms"], {"ALPHA": ["Clinical synonym"]})
            self.assertEqual(payload["metadata"]["usage"], "candidate_recall_only")

    def test_nlm_authenticated_fetch_never_returns_or_reports_api_key(self) -> None:
        secret = "do-not-leak-this-key"
        response = Mock()
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        response.geturl.return_value = "https://download.nlm.nih.gov/release.zip"
        response.read.return_value = b"release"
        response.headers = {}
        with patch("medical_coder.refresh.urllib.request.urlopen", return_value=response) as opener:
            result = fetch_nlm_authenticated_release(
                "https://uts-ws.nlm.nih.gov/download",
                "https://download.nlm.nih.gov/release.zip",
                secret,
                {"uts-ws.nlm.nih.gov", "download.nlm.nih.gov"},
            )
        self.assertEqual(result.url, "https://download.nlm.nih.gov/release.zip")
        self.assertNotIn(secret, repr(result))
        self.assertIn(secret, opener.call_args.args[0].full_url)

    def test_nlm_authenticated_fetch_sanitizes_network_errors(self) -> None:
        secret = "do-not-leak-this-key"
        with patch("medical_coder.refresh.urllib.request.urlopen", side_effect=RuntimeError(secret)):
            with self.assertRaises(RefreshError) as caught:
                fetch_nlm_authenticated_release(
                    "https://uts-ws.nlm.nih.gov/download",
                    "https://download.nlm.nih.gov/release.zip",
                    secret,
                    {"uts-ws.nlm.nih.gov", "download.nlm.nih.gov"},
                )
        self.assertNotIn(secret, str(caught.exception))

    def test_batch_refresh_rolls_back_installed_source_when_compile_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "data" / "codes" / "target.json"
            target.parent.mkdir(parents=True)
            target.write_text('{"state":"original"}\n', encoding="utf-8")
            (root / "source-packs").mkdir()
            registry = root / "registry.json"
            registry.write_text(json.dumps({
                "schema_version": 1,
                "sources": [{"id": "source-a"}],
            }), encoding="utf-8")
            pack = root / "source-packs" / "pack.json"
            pack.write_text("{}", encoding="utf-8")
            runner = RefreshRunner(root, registry)
            runner._prepare = Mock(return_value={
                "source_id": "source-a", "config": {"authority": "test", "output_path": "data/codes/target.json"},
                "downloads": [], "effective_start": "2026-01-01", "payload": {"state": "new"},
                "row_count": 1, "target": target,
            })
            with patch("medical_coder.refresh.SourceCompiler.compile", side_effect=[root / "shadow", RuntimeError("fail")]):
                with self.assertRaises(RuntimeError):
                    runner.refresh_many(["source-a"], pack, root / "snapshots")
            self.assertEqual(json.loads(target.read_text(encoding="utf-8")), {"state": "original"})


if __name__ == "__main__":
    unittest.main()

