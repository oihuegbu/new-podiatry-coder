"""Full refresh-stage regression for snomed_concept_terms (issue #6 F7-R3-C5).

Codex, exact-SHA re-review: `tools/refresh_authoritative_data._verify()` counted only
the "terms"/"codes" container fields, so a correctly-built `snomed_concept_terms.json`
(container field "concepts") was reported as an empty-output FAILURE even though the
build itself succeeded and wrote 43,459 real concepts against the licensed release.

This exercises the real build tool (`tools/build_snomed_concept_terms.py`) against a
minimal, SYNTHETIC RF2-shaped fixture -- not the licensed release, which is not present
in CI -- and the real `_verify()` together, proving the two now agree on what a
successful build looks like. SNOMED's own architecture constants (IS_A typeId,
inferred characteristicTypeId, Synonym/FSN typeIds, the Body Structure root id) are
real RF2 format identifiers, not clinical vocabulary or a medical code the guard covers
-- required for the fixture to be shaped like a real release at all.
"""
from __future__ import annotations

import importlib
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_RELATIONSHIP_HEADER = ("id\teffectiveTime\tactive\tmoduleId\tsourceId\tdestinationId\t"
                        "relationshipGroup\ttypeId\tcharacteristicTypeId\tmodifierId\n")
_DESCRIPTION_HEADER = ("id\teffectiveTime\tactive\tmoduleId\tconceptId\tlanguageCode\t"
                       "typeId\tterm\tcaseSignificanceId\n")
_IS_A = "116680003"
_INFERRED = "900000000000011006"
_FSN = "900000000000003001"
_SYNONYM = "900000000000013009"
_ROOT = "123037004"   # SNOMED's own Body Structure root -- an RF2 format id, not vocabulary.


def _write_fixture(release_dir: Path) -> None:
    snap = release_dir / "Snapshot" / "Terminology"
    snap.mkdir(parents=True)
    rel = ["1\t20260101\t1\t9\tSYN1\t" + _ROOT + f"\t0\t{_IS_A}\t{_INFERRED}\t0",
          "2\t20260101\t1\t9\tSYN2\tSYN1\t0\t" + _IS_A + f"\t{_INFERRED}\t0"]
    (snap / "sct2_Relationship_Snapshot_TEST_20260101.txt").write_text(
        _RELATIONSHIP_HEADER + "\n".join(rel) + "\n")
    desc = [f"10\t20260101\t1\t9\tSYN1\ten\t{_SYNONYM}\tsynthetic structure one\t0",
           f"11\t20260101\t1\t9\tSYN1\ten\t{_FSN}\tsynthetic structure one (body structure)\t0",
           f"12\t20260101\t1\t9\tSYN2\ten\t{_SYNONYM}\tsynthetic structure two\t0"]
    (snap / "sct2_Description_Snapshot-en_TEST_20260101.txt").write_text(
        _DESCRIPTION_HEADER + "\n".join(desc) + "\n")


class SnomedConceptTermsRefreshRegression(unittest.TestCase):

    def test_prepare_and_verify_agree_on_a_real_build(self):
        import tools.build_snomed_concept_terms as builder
        import tools.refresh_authoritative_data as refresh

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            release_dir = tmp / "SnomedCT_TEST_20260101T120000Z"
            _write_fixture(release_dir)
            data_dir = tmp / "data"
            (data_dir / "codes").mkdir(parents=True)

            old_argv, old_data_dir, old_codes = (
                sys.argv, builder.DATA_DIR, refresh.CODES)
            sys.argv = ["build_snomed_concept_terms.py", "--release", str(release_dir)]
            builder.DATA_DIR = data_dir
            refresh.CODES = data_dir / "codes"
            try:
                exit_code = builder.main()
                self.assertEqual(exit_code, 0)
                out = data_dir / "codes" / "snomed_concept_terms.json"
                self.assertTrue(out.exists())

                # This is the exact call the refresh stage makes after `prepare`. It
                # must succeed, not raise "prepared but empty" against a real build.
                record = refresh._verify("snomed_concept_terms.json")
            finally:
                sys.argv, builder.DATA_DIR, refresh.CODES = (
                    old_argv, old_data_dir, old_codes)

        self.assertEqual(record["codes"], 2, record)   # SYN1 (root child) + SYN2 (leaf)
        self.assertTrue(record["provenance"])
        self.assertTrue(record["sha256"])

    def test_verify_rejects_an_unrecognized_container_schema(self):
        import tools.refresh_authoritative_data as refresh
        import json

        with tempfile.TemporaryDirectory() as td:
            codes_dir = Path(td) / "codes"
            codes_dir.mkdir()
            (codes_dir / "mystery.json").write_text(json.dumps({"nonsense": {"x": 1}}))
            old_codes = refresh.CODES
            refresh.CODES = codes_dir
            try:
                with self.assertRaises(RuntimeError) as ctx:
                    refresh._verify("mystery.json")
                self.assertIn("no recognized record container", str(ctx.exception))
            finally:
                refresh.CODES = old_codes

    def test_verify_rejects_an_empty_relationship_digest(self):
        """A present-but-empty release digest is its own build defect, even when the
        record container carries entries -- a partial run must not be reported as a
        clean success."""
        import json

        import tools.refresh_authoritative_data as refresh

        with tempfile.TemporaryDirectory() as td:
            codes_dir = Path(td) / "codes"
            codes_dir.mkdir()
            (codes_dir / "partial.json").write_text(json.dumps(
                {"concepts": {"SYN1": {"terms": ["x"], "parents": []}},
                 "relationship_sha256": ""}))
            old_codes = refresh.CODES
            refresh.CODES = codes_dir
            try:
                with self.assertRaises(RuntimeError) as ctx:
                    refresh._verify("partial.json")
                self.assertIn("relationship_sha256", str(ctx.exception))
            finally:
                refresh.CODES = old_codes


class RunSourcePublicationSafetyTest(unittest.TestCase):
    """Codex F7-R3-C5, exact-SHA re-review, sixth pass, exact reproduction: start with
    a valid prior artifact, make `prepare` write an empty/invalid one, run
    `run_source`. Before this fix it reported `ERROR: ... prepared but empty` but LEFT
    THE INVALID FILE IN PLACE, destroying the prior valid artifact -- every builder
    writes directly to the published path, so a failed run overwrites the last
    known-good file before verification ever runs."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.codes_dir = Path(self.tmp.name) / "codes"
        self.codes_dir.mkdir()

    class _Args:
        dry_run = False

    def test_a_failed_refresh_preserves_the_prior_valid_artifact(self):
        import json

        import tools.refresh_authoritative_data as refresh

        output_path = self.codes_dir / "fake_source.json"
        output_path.write_text(json.dumps(
            {"concepts": {"C1": {"terms": ["x"], "parents": []}}}))
        prior_bytes = output_path.read_bytes()

        def _broken_prepare(tmp, args):
            # `prepare` returns a COMMAND; the write happens when that command is RUN
            # (mirroring every real builder, which writes to the published path as a
            # subprocess, not as a side effect of `prepare` being called) -- the exact
            # failure mode is the builder overwriting the PUBLISHED path with an empty
            # (but validly-shaped) artifact, then exiting 0; `_verify` is what catches
            # the emptiness, not the build command itself.
            script = (f"import json; open({str(output_path)!r}, 'w')"
                     f".write(json.dumps({{'concepts': {{}}}}))")
            return [sys.executable, "-c", script]

        old_codes, old_sources = refresh.CODES, refresh.SOURCES
        refresh.CODES = self.codes_dir
        refresh.SOURCES = dict(refresh.SOURCES)
        refresh.SOURCES["fake_source"] = {"output": "fake_source.json",
                                          "prepare": _broken_prepare}
        try:
            rec = refresh.run_source("fake_source", self._Args())
            self.assertTrue(str(rec["status"]).startswith("ERROR"), rec)
            self.assertEqual(output_path.read_bytes(), prior_bytes,
                             "a failed refresh must not destroy the prior valid artifact")
        finally:
            refresh.CODES, refresh.SOURCES = old_codes, old_sources

    def test_a_failed_first_run_leaves_no_artifact_behind(self):
        """The symmetric case: no prior artifact existed at all. A failed run must not
        leave a broken one in its place either."""
        import json

        import tools.refresh_authoritative_data as refresh

        output_path = self.codes_dir / "fake_source.json"
        self.assertFalse(output_path.exists())

        def _broken_prepare(tmp, args):
            script = (f"import json; open({str(output_path)!r}, 'w')"
                     f".write(json.dumps({{'concepts': {{}}}}))")
            return [sys.executable, "-c", script]

        old_codes, old_sources = refresh.CODES, refresh.SOURCES
        refresh.CODES = self.codes_dir
        refresh.SOURCES = dict(refresh.SOURCES)
        refresh.SOURCES["fake_source"] = {"output": "fake_source.json",
                                          "prepare": _broken_prepare}
        try:
            rec = refresh.run_source("fake_source", self._Args())
            self.assertTrue(str(rec["status"]).startswith("ERROR"), rec)
            self.assertFalse(output_path.exists(),
                             "a failed first-ever run must not leave an invalid "
                             "artifact where none existed before")
        finally:
            refresh.CODES, refresh.SOURCES = old_codes, old_sources

    def test_a_successful_refresh_still_publishes_normally(self):
        """The fix must not block a genuinely successful refresh from publishing."""
        import json

        import tools.refresh_authoritative_data as refresh

        output_path = self.codes_dir / "fake_source.json"

        def _good_prepare(tmp, args):
            script = (f"import json; open({str(output_path)!r}, 'w').write(json.dumps("
                     f"{{'concepts': {{'C1': {{'terms': ['x'], 'parents': []}}}}}}))")
            return [sys.executable, "-c", script]

        old_codes, old_sources = refresh.CODES, refresh.SOURCES
        refresh.CODES = self.codes_dir
        refresh.SOURCES = dict(refresh.SOURCES)
        refresh.SOURCES["fake_source"] = {"output": "fake_source.json",
                                          "prepare": _good_prepare}
        try:
            rec = refresh.run_source("fake_source", self._Args())
            self.assertEqual(rec["status"], "ok", rec)
            self.assertEqual(rec["codes"], 1)
        finally:
            refresh.CODES, refresh.SOURCES = old_codes, old_sources


if __name__ == "__main__":
    unittest.main()
