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


def _staged_builder_script(payload_json: str, *, echo_published: Path | None = None) -> list[str]:
    """A fake builder command matching the REAL contract every builder follows
    (issue #6 F7-R3-C5, exact-SHA re-review, eighth pass): it resolves its OWN output
    path from `PODIATRY_DATA_DIR` (falling back to the real data dir), exactly like
    `app.core.config.DATA_DIR` does -- never a path baked in by the test itself, which
    would silently skip staging and prove nothing.

    `echo_published`, when given, makes the fake builder READ BACK the REAL published
    path WHILE it runs (before it writes anything) and records what it saw to a
    marker file next to it -- the concurrent-reader proof: a reader consulting the
    published path DURING the build must see the untouched prior bytes, never the
    in-progress staged write.
    """
    echo = ""
    if echo_published is not None:
        marker = echo_published.with_suffix(".seen_during_build")
        echo = (f"import shutil\n"
               f"src = {str(echo_published)!r}\n"
               f"try:\n"
               f"    shutil.copy2(src, {str(marker)!r})\n"
               f"except FileNotFoundError:\n"
               f"    open({str(marker)!r}, 'w').write('<absent>')\n")
    script = (
        "import json, os, pathlib\n"
        f"{echo}"
        "base = pathlib.Path(os.environ.get('PODIATRY_DATA_DIR') or '.') / 'codes'\n"
        "base.mkdir(parents=True, exist_ok=True)\n"
        f"(base / 'fake_source.json').write_text({payload_json!r})\n"
    )
    return [sys.executable, "-c", script]


class RunSourcePublicationSafetyTest(unittest.TestCase):
    """Codex F7-R3-C5, exact-SHA re-review, sixth/eighth passes: publication is now
    fully staged -- every builder subprocess runs against a staging copy of the data
    directory (`PODIATRY_DATA_DIR`), so its own write NEVER touches the published path
    at all; only a validated staged artifact is moved into place with one atomic
    `os.replace`. This is a materially stronger guarantee than backup-and-restore (the
    prior artifact is never overwritten in the first place, not overwritten-then-
    restored), and these tests prove it directly: a concurrent reader sees the
    untouched prior bytes for the FULL duration of the build, not just after a
    failure is caught."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.data_dir = Path(self.tmp.name) / "data"
        self.codes_dir = self.data_dir / "codes"
        self.codes_dir.mkdir(parents=True)

    class _Args:
        dry_run = False

    def _patched(self, refresh, prepare):
        old_codes, old_data_dir, old_sources = (
            refresh.CODES, refresh.DATA_DIR, refresh.SOURCES)
        refresh.CODES = self.codes_dir
        refresh.DATA_DIR = self.data_dir
        refresh.SOURCES = dict(refresh.SOURCES)
        refresh.SOURCES["fake_source"] = {"output": "fake_source.json",
                                          "prepare": prepare}
        return old_codes, old_data_dir, old_sources

    def _restore(self, refresh, saved):
        refresh.CODES, refresh.DATA_DIR, refresh.SOURCES = saved

    def test_a_failed_refresh_preserves_the_prior_valid_artifact(self):
        import json

        import tools.refresh_authoritative_data as refresh

        output_path = self.codes_dir / "fake_source.json"
        output_path.write_text(json.dumps(
            {"concepts": {"C1": {"terms": ["x"], "parents": []}}}))
        prior_bytes = output_path.read_bytes()

        def _broken_prepare(tmp, args):
            return _staged_builder_script(json.dumps({"concepts": {}}))

        saved = self._patched(refresh, _broken_prepare)
        try:
            rec = refresh.run_source("fake_source", self._Args())
            self.assertTrue(str(rec["status"]).startswith("ERROR"), rec)
            self.assertEqual(output_path.read_bytes(), prior_bytes,
                             "a failed refresh must not destroy the prior valid artifact")
        finally:
            self._restore(refresh, saved)

    def test_a_failed_first_run_leaves_no_artifact_behind(self):
        """The symmetric case: no prior artifact existed at all. A failed run must not
        leave a broken one in its place either."""
        import json

        import tools.refresh_authoritative_data as refresh

        output_path = self.codes_dir / "fake_source.json"
        self.assertFalse(output_path.exists())

        def _broken_prepare(tmp, args):
            return _staged_builder_script(json.dumps({"concepts": {}}))

        saved = self._patched(refresh, _broken_prepare)
        try:
            rec = refresh.run_source("fake_source", self._Args())
            self.assertTrue(str(rec["status"]).startswith("ERROR"), rec)
            self.assertFalse(output_path.exists(),
                             "a failed first-ever run must not leave an invalid "
                             "artifact where none existed before")
        finally:
            self._restore(refresh, saved)

    def test_a_successful_refresh_still_publishes_normally(self):
        """The fix must not block a genuinely successful refresh from publishing."""
        import json

        import tools.refresh_authoritative_data as refresh

        def _good_prepare(tmp, args):
            return _staged_builder_script(
                json.dumps({"concepts": {"C1": {"terms": ["x"], "parents": []}}}))

        saved = self._patched(refresh, _good_prepare)
        try:
            rec = refresh.run_source("fake_source", self._Args())
            self.assertEqual(rec["status"], "ok", rec)
            self.assertEqual(rec["codes"], 1)
        finally:
            self._restore(refresh, saved)

    def test_a_concurrent_reader_sees_the_untouched_prior_artifact_during_the_build(self):
        """Codex's exact concurrency reproduction: a reader consulting the PUBLISHED
        path WHILE the builder is still running (before verification/replace) must
        see the complete, untouched prior artifact -- never unverified in-progress
        bytes, and never absence. Proven by having the fake builder itself read the
        published path before writing anything."""
        import json

        import tools.refresh_authoritative_data as refresh

        output_path = self.codes_dir / "fake_source.json"
        prior_payload = {"concepts": {"C1": {"terms": ["x"], "parents": []}}}
        output_path.write_text(json.dumps(prior_payload))
        prior_bytes = output_path.read_bytes()

        def _good_prepare(tmp, args):
            return _staged_builder_script(
                json.dumps({"concepts": {"C2": {"terms": ["y"], "parents": []}}}),
                echo_published=output_path)

        saved = self._patched(refresh, _good_prepare)
        try:
            rec = refresh.run_source("fake_source", self._Args())
            self.assertEqual(rec["status"], "ok", rec)
            marker = output_path.with_suffix(".seen_during_build")
            self.assertEqual(marker.read_bytes(), prior_bytes,
                             "the published path must show the untouched prior "
                             "artifact to a reader while the build is in progress")
            # And publication still completed correctly afterward.
            self.assertEqual(json.loads(output_path.read_text())["concepts"],
                             {"C2": {"terms": ["y"], "parents": []}})
        finally:
            self._restore(refresh, saved)

    def test_a_build_that_never_completes_leaves_the_published_path_untouched(self):
        """Codex's exact hard-termination reproduction, exercised at the process
        boundary rather than by actually sending SIGKILL (not reliably reproducible
        in a unit test): a builder subprocess that dies before writing ANYTHING
        (simulating a crash/kill at the earliest possible point) must never have
        touched the published path -- true by construction once the write goes to a
        staging copy first, but proven here rather than merely asserted."""
        import json

        import tools.refresh_authoritative_data as refresh

        output_path = self.codes_dir / "fake_source.json"
        output_path.write_text(json.dumps(
            {"concepts": {"C1": {"terms": ["x"], "parents": []}}}))
        prior_bytes = output_path.read_bytes()
        prior_mtime = output_path.stat().st_mtime_ns

        def _killed_prepare(tmp, args):
            # Exits nonzero before writing anything -- the closest a portable unit
            # test can get to "the process died mid-build".
            return [sys.executable, "-c", "import sys; sys.exit(137)"]

        saved = self._patched(refresh, _killed_prepare)
        try:
            rec = refresh.run_source("fake_source", self._Args())
            self.assertTrue(str(rec["status"]).startswith("ERROR"), rec)
            self.assertEqual(output_path.read_bytes(), prior_bytes)
            self.assertEqual(output_path.stat().st_mtime_ns, prior_mtime,
                             "the published file must not even be TOUCHED (no write,"
                             " no rename) when the build never reaches completion")
        finally:
            self._restore(refresh, saved)

    def test_an_exit_zero_builder_that_writes_nothing_is_never_reported_as_refreshed(self):
        """Codex F7-R3-C5, exact-SHA re-review, ninth pass, exact reproduction: a
        builder that exits 0 without creating or updating its output (a silent
        no-op, a path-redirection defect, or a swallowed producer failure) used to
        inherit the OLD target from the staging mirror -- `_verify` accepted the
        stale bytes as freshly built, and the refresh reported success while
        actually refreshing nothing."""
        import json

        import tools.refresh_authoritative_data as refresh

        output_path = self.codes_dir / "fake_source.json"
        output_path.write_text(json.dumps(
            {"concepts": {"C1": {"terms": ["x"], "parents": []}}}))
        prior_bytes = output_path.read_bytes()

        def _noop_prepare(tmp, args):
            # Exits 0 -- a "successful" run -- but never writes fake_source.json at
            # all, staged or otherwise.
            return [sys.executable, "-c", "pass"]

        saved = self._patched(refresh, _noop_prepare)
        try:
            rec = refresh.run_source("fake_source", self._Args())
            self.assertTrue(str(rec["status"]).startswith("ERROR"), rec)
            self.assertEqual(output_path.read_bytes(), prior_bytes,
                             "a no-op builder must not be reported as a refresh, and "
                             "must not disturb the prior artifact")
        finally:
            self._restore(refresh, saved)


if __name__ == "__main__":
    unittest.main()
