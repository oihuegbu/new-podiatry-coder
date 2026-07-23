"""Tests for policy-corpus grounding — turning the adjudicator's prose
citations from attestations into lookups.

Covers:
  1. Deterministic quote verification against stored policy sources
     (containment, ellipsis fragments, bounded token-overlap tolerance,
     too-short quotes, empty corpus).
  2. Attestation tiers (document_quoted / attested_only / unverified).
  3. Fabricated-quote voiding in the adjudicator's verdict map (fail
     closed: an invented passage voids the pass verdict).
  4. Attestation stamping on recorded per-code and observable targets.
  5. Registry anchoring discipline: attested_only targets are recorded
     but never anchor actuation, and a later attested_only
     re-adjudication RETIRES an earlier anchorable value.
  6. Cross-family second-opinion pass model selection.

Everything runs against temp dirs and stubs — no live corpus, no LLM.

Run:  PYTHONPATH=. python -m pytest tests/test_policy_grounding.py -q
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import tools.policy_corpus as pc
from tools.claims_registry import (record_adjudicated_codes,
                                   record_adjudicated_observables,
                                   verified_code_targets,
                                   verified_observable_targets)
from tools.coder_adjudicator import (_adjudicate_once, _attestation_of,
                                     _quote_fabricated,
                                     _stamp_attestation, _verdict_map)

_DOC_TEXT = (
    "Foot Care. Routine foot care is excluded from coverage. However, "
    "the presence of a systemic condition such as metabolic, "
    "neurologic, or peripheral vascular disease may require scrupulous "
    "foot care by a professional. Services ordinarily considered "
    "routine may be covered when systemic conditions result in severe "
    "circulatory embarrassment.\n" * 40)  # >5000 chars is not required


class _CorpusDir:
    """Context: point policy_corpus at a temp dir holding one source."""

    def __init__(self, text=_DOC_TEXT, empty=False):
        self.text, self.empty = text, empty

    def __enter__(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        if not self.empty:
            (d / "mbpm_ch15.txt").write_text(self.text)
        self.p1 = mock.patch.object(pc, "POLICY_DIR", d)
        self.p1.start()
        pc._norm_cache.clear()
        return d

    def __exit__(self, *exc):
        self.p1.stop()
        self.tmp.cleanup()
        pc._norm_cache.clear()
        return False


# ---------------------------------------------------------------------------
# 1. quote verification
# ---------------------------------------------------------------------------

class VerifyQuoteTest(unittest.TestCase):
    def test_verbatim_quote_verifies(self):
        with _CorpusDir():
            res = pc.verify_quote(
                "Routine foot care is excluded from coverage.")
        self.assertTrue(res["verified"])
        self.assertEqual(res["source_id"], "mbpm_ch15")

    def test_punctuation_and_case_are_normalized(self):
        with _CorpusDir():
            res = pc.verify_quote(
                "routine FOOT care is excluded, from coverage")
        self.assertTrue(res["verified"])

    def test_ellipsis_fragments_each_verify(self):
        with _CorpusDir():
            res = pc.verify_quote(
                "Routine foot care is excluded from coverage ... may "
                "require scrupulous foot care by a professional")
        self.assertTrue(res["verified"])

    def test_invented_passage_fails(self):
        with _CorpusDir():
            res = pc.verify_quote(
                "Nail debridement is always covered when the patient "
                "reports any discomfort whatsoever during ambulation")
        self.assertFalse(res["verified"])

    def test_short_quote_cannot_verify(self):
        with _CorpusDir():
            res = pc.verify_quote("foot care is excluded")
        self.assertFalse(res["verified"])
        self.assertIn("too short", res["why"])

    def test_no_corpus_is_not_verification(self):
        with _CorpusDir(empty=True):
            self.assertFalse(pc.corpus_available())
            res = pc.verify_quote(
                "Routine foot care is excluded from coverage.")
        self.assertFalse(res["verified"])
        self.assertIn("no policy corpus", res["why"])

    def test_light_extraction_artifacts_tolerated(self):
        # one dropped token out of >=7 stays above the 0.85 overlap bar
        with _CorpusDir():
            res = pc.verify_quote(
                "services ordinarily considered routine may be covered "
                "when systemic conditions result in circulatory "
                "embarrassment")
        self.assertTrue(res["verified"])


# ---------------------------------------------------------------------------
# 2. attestation tiers
# ---------------------------------------------------------------------------

class AttestationTierTest(unittest.TestCase):
    def test_verified_quote_is_document_quoted(self):
        with _CorpusDir():
            tier = pc.attest({"authority": "MBPM Ch.15 §290",
                              "authority_quote":
                              "Routine foot care is excluded from "
                              "coverage."})
        self.assertEqual(tier, "document_quoted")

    def test_missing_quote_is_attested_only_when_corpus_present(self):
        with _CorpusDir():
            self.assertEqual(pc.attest({"authority": "MBPM Ch.15"}),
                             "attested_only")

    def test_unverified_quote_is_attested_only(self):
        with _CorpusDir():
            tier = pc.attest({"authority_quote":
                              "a passage that exists in no stored "
                              "policy source anywhere at all"})
        self.assertEqual(tier, "attested_only")

    def test_no_corpus_is_unverified(self):
        with _CorpusDir(empty=True):
            self.assertEqual(pc.attest({"authority_quote": "whatever"}),
                             "unverified")

    def test_declared_reference_data_basis_is_data_backed(self):
        with _CorpusDir():
            tier = pc.attest({"authority": "MUE table",
                              "authority_basis": "reference_data"})
        self.assertEqual(tier, "data_backed")

    def test_basis_declaration_cannot_rescue_a_failed_quote(self):
        # a prose quote that fails verification stays attested_only even
        # if the item ALSO claims reference_data — the quote is the claim
        with _CorpusDir():
            tier = pc.attest({"authority_basis": "reference_data",
                              "authority_quote":
                              "a passage that exists in no stored "
                              "policy source anywhere at all"})
        self.assertEqual(tier, "attested_only")


# ---------------------------------------------------------------------------
# 3. fabricated-quote voiding in the adjudicator
# ---------------------------------------------------------------------------

def _presence_item(**kw):
    d = {"array": "cpt_codes", "code": "11720", "kind": "presence",
         "decision": "include", "authority": "NCCI Ch.1",
         "note_evidence": "debridement of six nails performed"}
    d.update(kw)
    return d


class QuoteVoidingTest(unittest.TestCase):
    def test_no_quote_is_never_fabrication(self):
        with _CorpusDir():
            self.assertFalse(_quote_fabricated(_presence_item()))

    def test_verified_quote_is_not_fabrication(self):
        with _CorpusDir():
            self.assertFalse(_quote_fabricated(_presence_item(
                authority_quote="Routine foot care is excluded from "
                                "coverage.")))

    def test_invented_quote_is_fabrication(self):
        with _CorpusDir():
            self.assertTrue(_quote_fabricated(_presence_item(
                authority_quote="all nail services are covered without "
                                "restriction for every beneficiary")))

    def test_no_corpus_degrades_to_pre_corpus_behavior(self):
        with _CorpusDir(empty=True):
            self.assertFalse(_quote_fabricated(_presence_item(
                authority_quote="anything at all here")))

    def test_broken_corpus_module_never_crashes_adjudication(self):
        with mock.patch.object(pc, "corpus_available",
                               side_effect=RuntimeError("boom")):
            self.assertFalse(_quote_fabricated(_presence_item(
                authority_quote="anything at all here")))

    def test_fabricated_quote_voids_the_pass_verdict(self):
        disputed = [{"array": "cpt_codes", "code": "11720",
                     "kind": "presence"}]
        good = {"items": [_presence_item(
            authority_quote="Routine foot care is excluded from "
                            "coverage.")]}
        bad = {"items": [_presence_item(
            authority_quote="a passage that exists in no stored policy "
                            "source anywhere at all")]}
        with _CorpusDir():
            self.assertIsNotNone(_verdict_map(good, disputed))
            self.assertIsNone(_verdict_map(bad, disputed))


# ---------------------------------------------------------------------------
# 4. attestation stamping on recorded targets
# ---------------------------------------------------------------------------

class StampAttestationTest(unittest.TestCase):
    def test_weakest_matching_item_tier_wins(self):
        targets = [{"array": "cpt_codes", "code": "11720", "row": None}]
        items = [
            _presence_item(authority_quote="Routine foot care is "
                                           "excluded from coverage."),
            {"array": "cpt_codes", "code": "11720", "kind": "attributes",
             "decision": "set", "fields": {"units": 1},
             "authority": "MUE policy", "note_evidence": "six nails"},
        ]
        with _CorpusDir():
            _stamp_attestation(targets, items)
        # presence item verified (document_quoted), attributes item has
        # no quote (attested_only) -> weakest wins
        self.assertEqual(targets[0]["attestation"], "attested_only")

    def test_unmatched_target_gets_conservative_tier(self):
        targets = [{"array": "hcpcs_codes", "code": "A4570", "row": None}]
        with _CorpusDir():
            _stamp_attestation(targets, [])
        self.assertEqual(targets[0]["attestation"], "attested_only")

    def test_em_level_codes_list_matches(self):
        targets = [{"array": "cpt_codes", "code": "99213", "row": None}]
        items = [{"array": "cpt_codes", "kind": "em_level",
                  "codes": ["99213", "99214"], "decision": "select",
                  "decision_code": "99213", "authority": "AMA MDM",
                  "note_evidence": "low complexity",
                  "authority_quote": "Routine foot care is excluded "
                                     "from coverage."}]
        with _CorpusDir():
            _stamp_attestation(targets, items)
        self.assertEqual(targets[0]["attestation"], "document_quoted")

    def test_no_corpus_stamps_unverified(self):
        targets = [{"array": "cpt_codes", "code": "11720", "row": None}]
        with _CorpusDir(empty=True):
            _stamp_attestation(targets, [_presence_item()])
            self.assertEqual(_attestation_of(None), "unverified")
        self.assertEqual(targets[0]["attestation"], "unverified")


# ---------------------------------------------------------------------------
# 5. registry anchoring discipline
# ---------------------------------------------------------------------------

class RegistryAnchoringTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.reg = Path(self.tmp.name) / "registry.jsonl"

    def tearDown(self):
        self.tmp.cleanup()

    def test_attested_only_observable_target_never_anchors(self):
        record_adjudicated_observables(
            "doc1",
            [{"observable": "advisory_emission",
              "key": "MEDICAL_NECESSITY|11720", "emit": False,
              "attestation": "attested_only"},
             {"observable": "advisory_emission",
              "key": "DOCUMENTATION|L60.1", "emit": True,
              "attestation": "document_quoted"}],
            source="t", by="t", registry_path=self.reg)
        got = verified_observable_targets(registry_path=self.reg)
        self.assertEqual(got, {"doc1": {
            ("advisory_emission", "DOCUMENTATION|L60.1"): True}})

    def test_later_attested_only_retires_earlier_anchorable(self):
        record_adjudicated_observables(
            "doc1", [{"observable": "advisory_emission",
                      "key": "MEDICAL_NECESSITY|11720", "emit": False,
                      "attestation": "document_quoted"}],
            source="t", by="t", registry_path=self.reg)
        record_adjudicated_observables(
            "doc1", [{"observable": "advisory_emission",
                      "key": "MEDICAL_NECESSITY|11720", "emit": True,
                      "attestation": "attested_only"}],
            source="t", by="t", registry_path=self.reg)
        self.assertEqual(verified_observable_targets(
            registry_path=self.reg), {})

    def test_legacy_targets_without_tier_stay_anchorable(self):
        record_adjudicated_observables(
            "doc1", [{"observable": "advisory_emission",
                      "key": "MEDICAL_NECESSITY|11720", "emit": False}],
            source="t", by="t", registry_path=self.reg)
        got = verified_observable_targets(registry_path=self.reg)
        self.assertEqual(got["doc1"][("advisory_emission",
                                      "MEDICAL_NECESSITY|11720")], False)

    def test_attested_only_code_target_never_anchors(self):
        record_adjudicated_codes(
            "doc1",
            [{"array": "cpt_codes", "code": "27654",
              "row": {"code": "27654", "modifiers": ["RT"], "units": 1},
              "attestation": "attested_only"},
             {"array": "hcpcs_codes", "code": "A4570", "row": None,
              "attestation": "unverified"}],
            source="t", by="t", registry_path=self.reg)
        got = verified_code_targets(registry_path=self.reg)
        self.assertEqual(set(got["doc1"]), {("hcpcs_codes", "A4570")})


# ---------------------------------------------------------------------------
# 6. autonomous corpus upkeep (ensure)
# ---------------------------------------------------------------------------

class CorpusUpkeepTest(unittest.TestCase):
    def _meta(self, d: Path, sid: str, checked: str) -> None:
        (d / f"{sid}.txt").write_text(_DOC_TEXT)
        (d / f"{sid}.meta.json").write_text(json.dumps(
            {"id": sid, "sha256": "x", "fetched_at": checked,
             "last_checked": checked}))

    def test_missing_source_is_stale(self):
        with _CorpusDir(empty=True):
            stale = pc._stale_ids(30)
        self.assertEqual(set(stale), {s["id"] for s in pc.manifest()})

    def test_fresh_source_is_not_stale(self):
        from datetime import datetime, timezone
        with _CorpusDir(empty=True) as d:
            now = datetime.now(timezone.utc).isoformat(
                timespec="seconds")
            for s in pc.manifest():
                self._meta(d, s["id"], now)
            self.assertEqual(pc._stale_ids(30), [])

    def test_overdue_source_is_stale(self):
        with _CorpusDir(empty=True) as d:
            for s in pc.manifest():
                self._meta(d, s["id"], "2020-01-01T00:00:00+00:00")
            stale = pc._stale_ids(30)
        self.assertEqual(set(stale), {s["id"] for s in pc.manifest()})

    def test_unreadable_provenance_is_stale(self):
        with _CorpusDir(empty=True) as d:
            sid = pc.manifest()[0]["id"]
            (d / f"{sid}.txt").write_text(_DOC_TEXT)
            (d / f"{sid}.meta.json").write_text("{not json")
            self.assertIn(sid, pc._stale_ids(30))

    def test_ensure_fetches_only_the_stale(self):
        with _CorpusDir(empty=True), \
                mock.patch.object(pc, "_stale_ids",
                                  return_value=["mbpm_ch15"]), \
                mock.patch.object(pc, "fetch",
                                  return_value={"fetched":
                                                ["mbpm_ch15"]}) as f, \
                mock.patch.object(pc, "AUTOFETCH", "1"):
            out = pc.ensure()
        f.assert_called_once_with(["mbpm_ch15"])
        self.assertEqual(out["fetched"], ["mbpm_ch15"])

    def test_ensure_is_free_when_fresh(self):
        with mock.patch.object(pc, "_stale_ids", return_value=[]), \
                mock.patch.object(pc, "fetch") as f, \
                mock.patch.object(pc, "AUTOFETCH", "1"):
            out = pc.ensure()
        f.assert_not_called()
        self.assertEqual(out, {"fresh": True})

    def test_ensure_respects_the_kill_switch(self):
        with mock.patch.object(pc, "AUTOFETCH", "0"), \
                mock.patch.object(pc, "fetch") as f:
            out = pc.ensure()
        f.assert_not_called()
        self.assertIn("skipped", out)

    def test_ensure_never_raises(self):
        with mock.patch.object(pc, "_stale_ids",
                               side_effect=RuntimeError("network")), \
                mock.patch.object(pc, "AUTOFETCH", "1"):
            out = pc.ensure()
        self.assertIn("error", out)

    def test_adjudicator_choke_point_never_raises(self):
        from tools.coder_adjudicator import _ensure_policy_corpus
        with mock.patch.object(pc, "ensure",
                               side_effect=RuntimeError("boom")):
            _ensure_policy_corpus()  # must not raise


# ---------------------------------------------------------------------------
# 7. cross-family second opinion
# ---------------------------------------------------------------------------

class AltModelPassTest(unittest.TestCase):
    def _run(self, pass_idx, alt):
        calls = []

        def fake_chat(**kw):
            calls.append(kw)
            return json.dumps({"items": []}), {}

        with mock.patch("app.core.llm_client.chat_completion",
                        side_effect=fake_chat), \
                mock.patch("app.core.config.LLM_PROVIDER", "claude"), \
                mock.patch("tools.coder_adjudicator.ALT_MODEL", alt):
            _adjudicate_once({"case": 1}, pass_idx=pass_idx)
        return calls[0]

    def test_first_pass_always_primary_model(self):
        from tools.coder_adjudicator import CODER_MODEL
        self.assertEqual(self._run(0, "other-family-model")["model"],
                         CODER_MODEL)

    def test_later_pass_uses_alt_family_when_configured(self):
        self.assertEqual(self._run(1, "other-family-model")["model"],
                         "other-family-model")

    def test_later_pass_unchanged_when_not_configured(self):
        from tools.coder_adjudicator import CODER_MODEL
        self.assertEqual(self._run(1, "")["model"], CODER_MODEL)


if __name__ == "__main__":
    unittest.main()
