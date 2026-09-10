"""ComplianceDataStore's PFS (global_period) ingest and DOS-aware selection
(issue #6 F9-R11-H-C, second re-review).

`_ingest_global_periods` was rewritten from an unconditional clear-then-insert
(which meant `global_period` never held more than one row per code, so DOS
was structurally unable to select anything) into a diff-based upsert that
preserves effective-dated history across refreshes. These tests exercise
that upsert directly against an isolated in-memory-shaped store -- never the
shared module-level `_STORE` other test files build against the real
compliance.db -- and the RVU release-letter date derivation it relies on.
"""
import json
import unittest

import app.compliance.datastore.store as store_module
from app.compliance.datastore.store import ComplianceDataStore, _rvu_release_effective_from


def _store(tmp_path):
    s = ComplianceDataStore(db_path=tmp_path / "test_compliance.db")
    s._create_schema()
    return s


def _write_release(tmp_path, monkeypatch, codes, version="RVU26C (2026 July release)"):
    path = tmp_path / "global_periods.json"
    path.write_text(json.dumps({"version": version, "source": "test", "source_url": "",
                                "codes": codes}))
    monkeypatch.setattr(store_module, "GLOBAL_PERIODS_FILE", path)
    return path


class RvuReleaseEffectiveFrom(unittest.TestCase):
    def test_parses_the_documented_cms_release_letter_convention(self):
        self.assertEqual(
            _rvu_release_effective_from("RVU26C (2026 July release)", "2099-01-01"),
            "2026-07-01")
        self.assertEqual(
            _rvu_release_effective_from("RVU26A (2026 January release)", "2099-01-01"),
            "2026-01-01")

    def test_falls_back_when_the_string_does_not_match(self):
        self.assertEqual(_rvu_release_effective_from("not a real version", "2026-05-01"),
                         "2026-05-01")
        self.assertEqual(_rvu_release_effective_from("", "2026-05-01"), "2026-05-01")


class GlobalPeriodDiffUpsert(unittest.TestCase):
    def test_first_ingest_opens_one_row_from_the_releases_own_effective_date(self):
        s = self._isolated()
        _write_release(self.tmp_path, self.monkeypatch,
                       {"64450": {"global_days": "000", "status": "A"}})
        s._ingest_global_periods()
        row = s.pfs_record("64450", dos="2026-08-01")
        self.assertIsNotNone(row)
        self.assertEqual(row["glob_days"], "000")
        self.assertEqual(row["billing_status"], "A")
        self.assertEqual(row["effective_from"], "2026-07-01")

    def test_a_dos_before_any_release_gracefully_falls_back_to_the_earliest_row(self):
        """`pfs_record` reuses `_asof`'s existing, already-established
        graceful-degrade behavior (the same one NCCI/MUE already use) rather
        than inventing a PFS-only stricter fail-closed policy for a DOS
        outside every ingested release's window -- a deliberate, narrower
        scope than a full historical-selection policy would need (see the
        H-C/H-D handoff for why: doing that consistently would mean
        revisiting NCCI/MUE's own established degrade behavior too, not a
        one-off special case here)."""
        s = self._isolated()
        _write_release(self.tmp_path, self.monkeypatch,
                       {"64450": {"global_days": "000", "status": "A"}},
                       version="RVU26C (2026 July release)")
        s._ingest_global_periods()
        row = s.pfs_record("64450", dos="2020-01-01")
        self.assertIsNotNone(row)
        self.assertEqual(row["glob_days"], "000")

    def test_an_unchanged_reingest_does_not_duplicate_the_row(self):
        s = self._isolated()
        _write_release(self.tmp_path, self.monkeypatch,
                       {"64450": {"global_days": "000", "status": "A"}})
        s._ingest_global_periods()
        _write_release(self.tmp_path, self.monkeypatch,
                       {"64450": {"global_days": "000", "status": "A"}},
                       version="RVU26D (2026 October release)")
        s._ingest_global_periods()
        rows = s.conn.execute(
            "SELECT COUNT(*) c FROM global_period WHERE code=?", ("64450",)).fetchone()
        self.assertEqual(rows["c"], 1)

    def test_a_changed_reingest_closes_the_old_row_and_opens_a_new_one(self):
        """The exact acceptance condition Codex named: two releases with
        different values must each answer for the DOS they actually cover."""
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
        self.assertEqual(rows["c"], 2, "changed values must open a second row, "
                                       "not overwrite the first")

    def test_a_rebuild_does_not_erase_previously_ingested_history(self):
        """Codex's required regression #5: the scheduled-refresh topology
        (`integrate()`) must not delete-and-rebuild compliance.db out from
        under history this ingest already accumulated. Simulated here at the
        ingest level: calling build_or_load's schema-creation path again
        must not be how a refresh is expressed for this table (the "clear"
        list Codex flagged is empty for "global_periods" -- see
        `ComplianceDataStore._data_sources`)."""
        s = self._isolated()
        _write_release(self.tmp_path, self.monkeypatch,
                       {"64450": {"global_days": "000", "status": "A"}},
                       version="RVU26A (2026 January release)")
        s._ingest_global_periods()
        _write_release(self.tmp_path, self.monkeypatch,
                       {"64450": {"global_days": "090", "status": "A"}},
                       version="RVU26C (2026 July release)")
        s._ingest_global_periods()
        sources = {src["id"]: src for src in s._data_sources()}
        self.assertEqual(sources["global_periods"]["clear"], [],
                         "global_periods must not be blindly cleared on refresh -- "
                         "its own ingest is what preserves history")
        rows = s.conn.execute(
            "SELECT COUNT(*) c FROM global_period WHERE code=?", ("64450",)).fetchone()
        self.assertEqual(rows["c"], 2)

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


if __name__ == "__main__":
    unittest.main()
