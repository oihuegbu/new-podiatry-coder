"""Full refresh-stage regression for umls_crosswalk (issue #6, UMLS 2026AA ingestion).

Exercises the real build tool (`tools/build_umls_crosswalk.py`) against a minimal,
SYNTHETIC MRCONSO.RRF-shaped fixture -- not the licensed UMLS release, which is not
present in CI -- and the real `refresh_authoritative_data._verify()` together, proving
the two agree on what a successful build looks like (matching
`tests/test_snomed_concept_refresh.py`'s convention for the same reason:
`_KEYED_MAP_CONTAINERS` must list this builder's "crosswalk" container field, or a
correctly-built artifact reports as an empty-output failure).

MRCONSO.RRF's own column order (CUI/LAT/SAB/CODE/STR/SUPPRESS at fixed indices) is a
real RRF format position, not clinical vocabulary or a medical code the no-hardcoding
guard covers -- required for the fixture to be shaped like a real release at all. CUIs,
CPT/HCPCS codes, and SNOMED concept IDs below are synthetic placeholders.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _row(cui, lat, sab, code, s, suppress="N"):
    # CUI|LAT|TS|LUI|STT|SUI|ISPREF|AUI|SAUI|SCUI|SDUI|SAB|TTY|CODE|STR|SRL|SUPPRESS|CVF|
    return (f"{cui}|{lat}|P|L1|PF|S1|Y|A1||{code}||{sab}|PT|{code}|{s}|0|{suppress}||")


def _write_fixture(release_dir: Path) -> None:
    release_dir.mkdir(parents=True)
    rows = [
        # SYNC1: a CPT code and a SNOMED concept sharing a CUI -- must appear.
        _row("CSYN1", "ENG", "CPT", "99999", "synthetic assembly service"),
        _row("CSYN1", "ENG", "SNOMEDCT_US", "1234567", "synthetic assembly (procedure)"),
        # SYNC2: an HCPT (CPT-within-HCPCS) code sharing a CUI with a SECOND SNOMED
        # concept -- proves the HCPT SAB is honored, and a code can gather more than
        # one matched concept.
        _row("CSYN2", "ENG", "HCPT", "88888", "synthetic HCPT service"),
        _row("CSYN2", "ENG", "SNOMEDCT_US", "2345678", "synthetic HCPT concept one"),
        _row("CSYN2", "ENG", "SNOMEDCT_US", "3456789", "synthetic HCPT concept two"),
        # HCPCS code with NO matching SNOMED concept under its CUI -- must be excluded
        # from the crosswalk entirely (billing code alone is not enough).
        _row("CSYN3", "ENG", "HCPCS", "77777", "synthetic unmatched HCPCS service"),
        # Suppressed row -- must never contribute, even though it otherwise matches.
        _row("CSYN4", "ENG", "CPT", "66666", "synthetic suppressed service", suppress="Y"),
        _row("CSYN4", "ENG", "SNOMEDCT_US", "4567890", "synthetic suppressed concept"),
        # Non-English row -- must never contribute.
        _row("CSYN5", "SPA", "CPT", "55555", "servicio sintetico"),
        _row("CSYN5", "ENG", "SNOMEDCT_US", "5678901", "synthetic spanish-paired concept"),
        # Excluded SABs: CPTSP (Spanish CPT) and HCDT (dental) -- must never contribute
        # even though they share a CUI with a real SNOMED row.
        _row("CSYN6", "ENG", "CPTSP", "44444", "cpt espanol"),
        _row("CSYN6", "ENG", "HCDT", "D4444", "dental code"),
        _row("CSYN6", "ENG", "SNOMEDCT_US", "6789012", "synthetic excluded-sab concept"),
    ]
    (release_dir / "MRCONSO.RRF").write_text("\n".join(rows) + "\n")


class UmlsCrosswalkRefreshRegression(unittest.TestCase):

    def test_prepare_and_verify_agree_on_a_real_build(self):
        import tools.build_umls_crosswalk as builder
        import tools.refresh_authoritative_data as refresh

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            release_dir = tmp / "rrf_output"
            _write_fixture(release_dir)
            data_dir = tmp / "data"
            (data_dir / "codes").mkdir(parents=True)

            old_argv, old_data_dir, old_codes = (
                sys.argv, builder.DATA_DIR, refresh.CODES)
            sys.argv = ["build_umls_crosswalk.py", "--release", str(release_dir)]
            builder.DATA_DIR = data_dir
            refresh.CODES = data_dir / "codes"
            try:
                exit_code = builder.main()
                self.assertEqual(exit_code, 0)
                out = data_dir / "codes" / "umls_cpt_snomed_crosswalk.json"
                self.assertTrue(out.exists())

                # The exact call the refresh stage makes after `prepare`. Must succeed
                # against a real build -- proves "crosswalk" is a recognized container
                # in _KEYED_MAP_CONTAINERS, not reported as "prepared but empty".
                record = refresh._verify("umls_cpt_snomed_crosswalk.json")
            finally:
                sys.argv, builder.DATA_DIR, refresh.CODES = (
                    old_argv, old_data_dir, old_codes)

        # Only 99999 (CPT) and 88888 (HCPT) genuinely share a CUI with a SNOMED row.
        self.assertEqual(record["codes"], 2, record)
        self.assertTrue(record["provenance"])
        self.assertTrue(record["sha256"])

    def test_billing_codes_with_no_matching_snomed_concept_are_excluded(self):
        import tools.build_umls_crosswalk as builder
        import json

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            release_dir = tmp / "rrf_output"
            _write_fixture(release_dir)
            data_dir = tmp / "data"
            (data_dir / "codes").mkdir(parents=True)

            old_argv, old_data_dir = sys.argv, builder.DATA_DIR
            sys.argv = ["build_umls_crosswalk.py", "--release", str(release_dir)]
            builder.DATA_DIR = data_dir
            try:
                self.assertEqual(builder.main(), 0)
                payload = json.loads(
                    (data_dir / "codes" / "umls_cpt_snomed_crosswalk.json").read_text())
            finally:
                sys.argv, builder.DATA_DIR = old_argv, old_data_dir

        crosswalk = payload["crosswalk"]
        self.assertIn("99999", crosswalk)
        self.assertEqual(crosswalk["99999"]["matched_snomed_concept_ids"], ["1234567"])
        self.assertIn("88888", crosswalk)
        self.assertEqual(sorted(crosswalk["88888"]["matched_snomed_concept_ids"]),
                         ["2345678", "3456789"])
        # 77777 (no SNOMED match), 66666 (suppressed), 55555 (non-English CPT row),
        # 44444/D4444 (excluded SABs) must all be absent.
        for missing in ("77777", "66666", "55555", "44444", "D4444"):
            self.assertNotIn(missing, crosswalk)

    def test_no_release_found_degrades_to_a_clean_no_op(self):
        import tools.build_umls_crosswalk as builder

        old_argv = sys.argv
        sys.argv = ["build_umls_crosswalk.py", "--release", "/nonexistent/umls/release"]
        try:
            self.assertEqual(builder.main(), 0)
        finally:
            sys.argv = old_argv

    def test_the_term_index_is_actually_built_when_a_synthetic_code_is_current(self):
        """issue #6 F9-R7-C, Codex's independent re-review of 92f4596: every
        test above proves the CROSSWALK builds, but none of them ever patches
        `_load_current_codes` -- so `_build_term_index`'s "is this code in the
        CURRENT authoritative registry" check always says no for a synthetic
        code, `useful_cuis` stays empty, and `umls_term_index.json` is silently
        never written. This suite therefore never actually covered the term-
        index build at all. Patch the current-registry check so 99999 IS
        current, and prove the artifact is genuinely produced with the right
        shape."""
        import json
        import tools.build_umls_crosswalk as builder

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            release_dir = tmp / "rrf_output"
            _write_fixture(release_dir)
            data_dir = tmp / "data"
            (data_dir / "codes").mkdir(parents=True)

            old_argv, old_data_dir, old_current = (
                sys.argv, builder.DATA_DIR, builder._load_current_codes)
            sys.argv = ["build_umls_crosswalk.py", "--release", str(release_dir)]
            builder.DATA_DIR = data_dir
            builder._load_current_codes = lambda: {"cpt": {"99999"}, "hcpcs": set()}
            try:
                self.assertEqual(builder.main(), 0)
                out = data_dir / "codes" / "umls_term_index.json"
                self.assertTrue(out.exists(), "umls_term_index.json was never written")
                payload = json.loads(out.read_text())
            finally:
                (sys.argv, builder.DATA_DIR, builder._load_current_codes) = (
                    old_argv, old_data_dir, old_current)

        self.assertIn("synthetic assembly service", payload["term_to_cuis"])
        self.assertEqual(payload["term_to_cuis"]["synthetic assembly service"], ["CSYN1"])
        self.assertEqual(payload["cui_to_atoms"]["CSYN1"][0]["code"], "99999")
        self.assertEqual(payload["code_to_cuis"]["cpt:99999"], ["CSYN1"])
        # 88888 (HCPT) is current in NO registry per the patched current-codes
        # set above, so it must NOT appear -- proves the current-registry
        # restriction is actually enforced, not merely present in the code.
        self.assertNotIn("cpt:88888", payload["code_to_cuis"])
        self.assertNotIn("hcpcs:88888", payload["code_to_cuis"])

    def test_no_snomed_overlap_at_all_degrades_to_a_clean_no_op(self):
        import tools.build_umls_crosswalk as builder

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            release_dir = tmp / "rrf_output"
            release_dir.mkdir(parents=True)
            (release_dir / "MRCONSO.RRF").write_text(
                _row("CSYNX", "ENG", "CPT", "12321", "synthetic lonely cpt service") + "\n")
            data_dir = tmp / "data"
            (data_dir / "codes").mkdir(parents=True)

            old_argv, old_data_dir = sys.argv, builder.DATA_DIR
            sys.argv = ["build_umls_crosswalk.py", "--release", str(release_dir)]
            builder.DATA_DIR = data_dir
            try:
                self.assertEqual(builder.main(), 0)
                self.assertFalse(
                    (data_dir / "codes" / "umls_cpt_snomed_crosswalk.json").exists())
            finally:
                sys.argv, builder.DATA_DIR = old_argv, old_data_dir


if __name__ == "__main__":
    unittest.main()
