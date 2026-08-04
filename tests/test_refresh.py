from __future__ import annotations

import io
import unittest
import zipfile

from medical_coder.refresh import AuthorityFetcher, RefreshError, parse_delimited_archives
from medical_coder.refresh import Download


class RefreshTestCase(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()

