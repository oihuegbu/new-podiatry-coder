"""ComplianceDataStore's PFS (global_period) ingest and DOS-aware selection
(issue #6 F9-R11-H-C, third re-review).

`_ingest_global_periods` treats data/global_periods.json as a single,
asserted-COMPLETE CMS PFS extract: every currently-open row is closed at a
new release's own effective_from (derived ONLY from the release's declared
CMS RVU release-letter convention, never a filesystem timestamp), then the
release's own rows are inserted fresh -- so a code omitted from a newer
release is a real retirement signal, not a data gap, and an already-deployed
database's legacy rows get the same real effective-dating a fresh install
gets. `pfs_record` is a STRICT, single-window lookup (deliberately not
`_asof`'s graceful multi-step degrade) -- a DOS outside every ingested
release's window is an honest None, never the nearest available release
standing in for an era it does not describe.

These tests exercise the ingest and lookup directly against isolated,
schema-only stores -- never the shared module-level `_STORE` other test
files build against the real compliance.db.
"""
import json
import unittest

import app.compliance.datastore.store as store_module
from app.compliance.datastore.store import ComplianceDataStore, _rvu_release_effective_from


def _store(tmp_path):
    s = ComplianceDataStore(db_path=tmp_path / "test_compliance.db")
    s._create_schema()
    return s


def _write_release(tmp_path, monkeypatch, codes, version="RVU26C (2026 July release)",
                   declared_count=None):
    payload = {"version": version, "source": "test", "source_url": "", "codes": codes}
    if declared_count is not None:
        payload["counts"] = {"codes": declared_count}
    path = tmp_path / "global_periods.json"
    path.write_text(json.dumps(payload))
    monkeypatch.setattr(store_module, "GLOBAL_PERIODS_FILE", path)
    return path


class _Isolated(unittest.TestCase):
    """Base: an isolated schema-only store + a GLOBAL_PERIODS_FILE monkeypatch
    shim, without needing the pytest `monkeypatch` fixture in a unittest
    TestCase."""

    def _isolated(self):
        import tempfile
        from pathlib import Path
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmpdir.name)
        self.addCleanup(self._tmpdir.cleanup)
        self.monkeypatch = _MonkeypatchShim()
        self.addCleanup(self.monkeypatch.undo)
        return _store(self.tmp_path)


class _MonkeypatchShim:
    """Minimal setattr/undo, so these unittest.TestCase tests don't need the
    pytest `monkeypatch` fixture just for one module attribute."""
    def __init__(self):
        self._orig = []

    def setattr(self, obj, name, value):
        self._orig.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)

    def undo(self):
        for obj, name, value in reversed(self._orig):
            setattr(obj, name, value)


class RvuReleaseEffectiveFrom(unittest.TestCase):
    def test_parses_the_documented_cms_release_letter_convention(self):
        self.assertEqual(
            _rvu_release_effective_from("RVU26C (2026 July release)"), "2026-07-01")
        self.assertEqual(
            _rvu_release_effective_from("RVU26A (2026 January release)"), "2026-01-01")

    def test_returns_none_when_the_string_does_not_match_no_fallback_date_guessed(self):
        """issue #6 F9-R11-H-C, third re-review: NO filesystem-mtime (or any
        other guessed) fallback -- an unparseable version means the refresh
        must be REJECTED, not silently dated from something else."""
        self.assertIsNone(_rvu_release_effective_from("not a real version"))
        self.assertIsNone(_rvu_release_effective_from(""))


class GlobalPeriodCompleteSnapshotIngest(_Isolated):
    def test_first_ingest_opens_every_row_from_the_releases_own_effective_date(self):
        s = self._isolated()
        _write_release(self.tmp_path, self.monkeypatch,
                       {"64450": {"global_days": "000", "status": "A"}})
        s._ingest_global_periods()
        row = s.pfs_record("64450", dos="2026-08-01")
        self.assertIsNotNone(row)
        self.assertEqual(row["glob_days"], "000")
        self.assertEqual(row["billing_status"], "A")
        self.assertEqual(row["effective_from"], "2026-07-01")
        self.assertEqual(row["source_version"], "RVU26C (2026 July release)")

    def test_a_dos_before_any_release_is_an_honest_gap_not_the_nearest_release(self):
        """The exact defect Codex reproduced against the prior (second
        re-review) design: `pfs_record` must never let a 2026 release answer
        for a 2020 DOS. Strict, no fallback."""
        s = self._isolated()
        _write_release(self.tmp_path, self.monkeypatch,
                       {"64450": {"global_days": "000", "status": "A"}},
                       version="RVU26C (2026 July release)")
        s._ingest_global_periods()
        self.assertIsNone(s.pfs_record("64450", dos="2020-01-01"))

    def test_a_dos_after_the_open_releases_window_still_resolves_it(self):
        """The newest ingested release stays open-ended (effective_to=OPEN)
        until a NEWER one supersedes it -- "currently in force" is not the
        same defect as "answers for an era it doesn't cover"."""
        s = self._isolated()
        _write_release(self.tmp_path, self.monkeypatch,
                       {"64450": {"global_days": "000", "status": "A"}},
                       version="RVU26C (2026 July release)")
        s._ingest_global_periods()
        row = s.pfs_record("64450", dos="2030-01-01")
        self.assertIsNotNone(row)
        self.assertEqual(row["glob_days"], "000")

    def test_an_unknown_code_or_empty_table_is_none_not_an_error(self):
        s = self._isolated()
        self.assertIsNone(s.pfs_record("64450", dos="2026-08-01"))
        _write_release(self.tmp_path, self.monkeypatch,
                       {"64450": {"global_days": "000", "status": "A"}})
        s._ingest_global_periods()
        self.assertIsNone(s.pfs_record("99999", dos="2026-08-01"))

    def test_reingesting_the_identical_release_is_a_no_op(self):
        """Idempotent per (source_id, effective_from) -- matches the
        contract `ingest_snapshot` already established for the live refresh
        path (ncci_ptp/mue/global_period)."""
        s = self._isolated()
        _write_release(self.tmp_path, self.monkeypatch,
                       {"64450": {"global_days": "000", "status": "A"}},
                       version="RVU26C (2026 July release)")
        s._ingest_global_periods()
        first = s.conn.execute(
            "SELECT COUNT(*) c FROM global_period").fetchone()["c"]
        s._ingest_global_periods()
        second = s.conn.execute(
            "SELECT COUNT(*) c FROM global_period").fetchone()["c"]
        self.assertEqual(first, second)

    def test_a_changed_release_closes_every_row_and_each_dos_resolves_its_own_era(self):
        """The exact acceptance condition Codex named: two releases with
        different values must each answer for the DOS they actually
        cover."""
        s = self._isolated()
        _write_release(self.tmp_path, self.monkeypatch,
                       {"64450": {"global_days": "000", "status": "A"}},
                       version="RVU26A (2026 January release)")
        s._ingest_global_periods()
        _write_release(self.tmp_path, self.monkeypatch,
                       {"64450": {"global_days": "090", "status": "A"}},
                       version="RVU26C (2026 July release)")
        s._ingest_global_periods()

        before = s.pfs_record("64450", dos="2026-03-14")
        after = s.pfs_record("64450", dos="2026-08-01")
        self.assertEqual(before["glob_days"], "000")
        self.assertEqual(after["glob_days"], "090")
        rows = s.conn.execute(
            "SELECT COUNT(*) c FROM global_period WHERE code=?", ("64450",)).fetchone()
        self.assertEqual(rows["c"], 2)

    def test_a_code_omitted_from_a_newer_release_is_closed_not_left_falsely_active(self):
        """issue #6 F9-R11-H-C, third re-review, H-C2: a COMPLETE snapshot's
        silence about a code is a real retirement signal. Reproduces
        Codex's exact scenario: RVU26A has X0001, RVU26C omits it."""
        s = self._isolated()
        _write_release(self.tmp_path, self.monkeypatch,
                       {"64450": {"global_days": "000", "status": "A"},
                        "27650": {"global_days": "090", "status": "A"}},
                       version="RVU26A (2026 January release)")
        s._ingest_global_periods()
        _write_release(self.tmp_path, self.monkeypatch,
                       {"27650": {"global_days": "090", "status": "A"}},
                       version="RVU26C (2026 July release)")
        s._ingest_global_periods()

        # Still resolves for a DOS in the era it actually covered:
        self.assertIsNotNone(s.pfs_record("64450", dos="2026-03-14"))
        # No longer active for a DOS in the release that omitted it:
        self.assertIsNone(s.pfs_record("64450", dos="2026-08-01"))
        # The still-present code is unaffected:
        self.assertIsNotNone(s.pfs_record("27650", dos="2026-08-01"))

    def test_an_older_release_arriving_after_a_newer_one_is_rejected(self):
        """issue #6 F9-R11-H-C, third re-review: never silently apply an
        out-of-order release over a newer already-ingested one."""
        s = self._isolated()
        _write_release(self.tmp_path, self.monkeypatch,
                       {"64450": {"global_days": "090", "status": "A"}},
                       version="RVU26C (2026 July release)")
        s._ingest_global_periods()
        _write_release(self.tmp_path, self.monkeypatch,
                       {"64450": {"global_days": "000", "status": "A"}},
                       version="RVU26A (2026 January release)")
        s._ingest_global_periods()

        row = s.pfs_record("64450", dos="2026-08-01")
        self.assertEqual(row["glob_days"], "090",
                         "the older release must not have overwritten the newer one")
        rows = s.conn.execute(
            "SELECT COUNT(*) c FROM global_period WHERE code=?", ("64450",)).fetchone()
        self.assertEqual(rows["c"], 1)

    def test_a_zero_row_snapshot_is_rejected_not_recorded_as_ingested(self):
        s = self._isolated()
        _write_release(self.tmp_path, self.monkeypatch, {},
                       version="RVU26C (2026 July release)")
        s._ingest_global_periods()
        rows = s.conn.execute("SELECT COUNT(*) c FROM global_period").fetchone()["c"]
        self.assertEqual(rows, 0)
        already = s.conn.execute(
            "SELECT 1 FROM data_source_version WHERE source_id=?",
            (s._GLOBAL_PERIODS_SOURCE_ID,)).fetchone()
        self.assertIsNone(already, "a 0-row snapshot must not be recorded as a real ingest")

    def test_a_truncated_snapshot_is_rejected_not_accepted_as_complete(self):
        """issue #6 F9-R11-H-C, fourth re-review: "complete snapshot" was
        asserted in a comment but never validated -- a body with fewer codes
        than the file's own declared "counts.codes" must never be accepted
        as the new complete truth (which would silently retire every code
        the truncation happened to drop)."""
        s = self._isolated()
        _write_release(self.tmp_path, self.monkeypatch,
                       {"64450": {"global_days": "000", "status": "A"},
                        "27650": {"global_days": "090", "status": "A"}},
                       version="RVU26A (2026 January release)", declared_count=2)
        s._ingest_global_periods()
        # Second release CLAIMS count=2 but its body only has 1 -- truncated.
        _write_release(self.tmp_path, self.monkeypatch,
                       {"27650": {"global_days": "090", "status": "A"}},
                       version="RVU26C (2026 July release)", declared_count=2)
        s._ingest_global_periods()

        self.assertIsNotNone(s.pfs_record("64450", dos="2026-08-01"),
                             "the truncated release must have been rejected -- "
                             "the January row must still be open")
        rows = s.conn.execute("SELECT COUNT(*) c FROM global_period").fetchone()["c"]
        self.assertEqual(rows, 2, "the truncated release's row must never have been inserted")

    def test_a_rebuild_does_not_erase_previously_ingested_history(self):
        """Codex's required regression #5: "global_periods" carries no
        "clear" entry in the source-freshness registry, so a refresh can
        never blindly wipe this table out from under its own history."""
        s = self._isolated()
        sources = {src["id"]: src for src in s._data_sources()}
        self.assertEqual(sources["global_periods"]["clear"], [],
                         "global_periods must not be blindly cleared on refresh -- "
                         "its own ingest is what preserves history")

    def test_a_legacy_deployed_row_is_migrated_to_a_real_effective_date(self):
        """issue #6 F9-R11-H-C, third re-review, H-C1: a database seeded
        before this fix carries rows at effective_from='1900-01-01'
        (the old seed baseline). Ingesting ANY real release must close that
        legacy row -- not leave it open forever because its VALUES happen
        to be unchanged (the second re-review's actual bug)."""
        s = self._isolated()
        s.conn.execute(
            "INSERT INTO global_period (code, glob_days, billing_status, "
            "effective_from, effective_to) VALUES (?,?,?,?,?)",
            ("64450", "000", "A", "1900-01-01", "9999-12-31"))
        s.conn.commit()
        _write_release(self.tmp_path, self.monkeypatch,
                       {"64450": {"global_days": "000", "status": "A"}},
                       version="RVU26C (2026 July release)")
        s._ingest_global_periods()

        row = s.pfs_record("64450", dos="2026-08-01")
        self.assertEqual(row["effective_from"], "2026-07-01",
                         "the legacy 1900-01-01 baseline must not still be open")
        legacy_still_open = s.conn.execute(
            "SELECT 1 FROM global_period WHERE code=? AND effective_from=? "
            "AND effective_to=?", ("64450", "1900-01-01", "9999-12-31")).fetchone()
        self.assertIsNone(legacy_still_open, "the legacy row must have been closed")

    def test_the_source_version_migration_preserves_genuine_live_refresh_history(self):
        """issue #6 F9-R11-H-C, fourth re-review: the migration that adds
        the source_version column previously did an unconditional `DELETE
        FROM global_period` before reingesting -- which destroyed genuine
        effective-dated rows the LIVE scheduled refresh (ingest_snapshot,
        source_id "pfs_global") may already have accumulated, not just the
        legacy 1900-01-01 seed baseline. Simulates a pre-migration table
        (source_version column absent, matching a database built before
        this round) carrying ONE genuine historical row with a real,
        non-legacy effective_from -- it must survive the migration."""
        s = self._isolated()
        # Simulate the pre-migration schema: drop and recreate global_period
        # WITHOUT source_version, matching what an already-deployed database
        # (built before this round shipped) actually has on disk.
        s.conn.executescript("""
            DROP TABLE global_period;
            CREATE TABLE global_period (
                code TEXT NOT NULL, glob_days TEXT, billing_status TEXT,
                bilat_surg TEXT, pctc_ind TEXT, mult_proc TEXT, asst_surg TEXT,
                co_surg TEXT, team_surg TEXT,
                effective_from TEXT NOT NULL DEFAULT '1900-01-01',
                effective_to TEXT NOT NULL DEFAULT '9999-12-31'
            );
        """)
        # A GENUINE historical row -- e.g. from a live "pfs_global" refresh
        # -- not the legacy 1900-01-01 baseline. Every indicator populated
        # (not just glob_days/billing_status) so the EARLIER, unrelated
        # bilat_surg/pctc_ind backfill migration -- which itself re-ingests
        # when ANY row has those columns NULL -- does not also fire here and
        # confound what this test is isolating.
        s.conn.execute(
            "INSERT INTO global_period (code, glob_days, billing_status, bilat_surg, "
            "pctc_ind, mult_proc, asst_surg, co_surg, team_surg, "
            "effective_from, effective_to) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            ("27650", "090", "A", "9", "0", "9", "9", "9", "9", "2025-10-01", "9999-12-31"))
        s.conn.commit()
        _write_release(self.tmp_path, self.monkeypatch,
                       {"64450": {"global_days": "000", "status": "A"}},
                       version="RVU26C (2026 July release)")

        s._ensure_migrations()

        survived = s.conn.execute(
            "SELECT effective_from, effective_to FROM global_period WHERE code=?",
            ("27650",)).fetchone()
        self.assertIsNotNone(survived, "the genuine historical row must not be deleted")
        self.assertEqual(survived["effective_from"], "2025-10-01",
                         "the migration must not have rebaselined a real effective date")

    def test_a_pre_change_deployed_database_and_a_fresh_one_converge_to_identical_state(self):
        """Codex's required "fresh-vs-upgraded equivalence" regression: an
        already-deployed database (simulated with a legacy 1900-01-01 seed
        row) and a brand-new one must answer identically once the SAME
        release has been ingested into both."""
        upgraded = self._isolated()
        upgraded.conn.execute(
            "INSERT INTO global_period (code, glob_days, billing_status, "
            "effective_from, effective_to) VALUES (?,?,?,?,?)",
            ("64450", "090", "I", "1900-01-01", "9999-12-31"))   # stale/wrong legacy values
        upgraded.conn.commit()

        import tempfile
        from pathlib import Path
        fresh_tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(fresh_tmpdir.cleanup)
        fresh = _store(Path(fresh_tmpdir.name))

        codes = {"64450": {"global_days": "000", "status": "A"}}
        version = "RVU26C (2026 July release)"
        for store, tmp_path in ((upgraded, self.tmp_path), (fresh, Path(fresh_tmpdir.name))):
            path = tmp_path / "global_periods.json"
            path.write_text(json.dumps({"version": version, "source": "test",
                                        "source_url": "", "codes": codes}))
            self.monkeypatch.setattr(store_module, "GLOBAL_PERIODS_FILE", path)
            store._ingest_global_periods()

        self.assertEqual(upgraded.pfs_record("64450", dos="2026-08-01"),
                         fresh.pfs_record("64450", dos="2026-08-01"))


if __name__ == "__main__":
    unittest.main()
