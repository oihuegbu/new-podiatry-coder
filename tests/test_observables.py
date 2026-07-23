"""Tests for the measurement-observable layer — the replay gates'
extensible vocabulary and its autonomous growth.

Covers:
  1. The built-in advisory_emission observable (identify ambiguity
     discipline, signature purity).
  2. The generic registry events (adjudicated_observables) and the merged
     view over legacy adjudicated_advisories events.
  3. The static safety gate + sandboxed loader for synthesized observable
     modules (same whitelist posture as auto templates).
  4. Fail-closed emission measurement (a crashed/missing observable can
     never silently satisfy a 'must not fire' verdict).
  5. Measurement-gap detection and the deterministic meta-gates of
     observable synthesis (identity, purity, baseline, corpus safety,
     vocabulary discipline) — the LLM design call itself is always
     stubbed.
  6. The convergence loop's stall->grow->continue wiring and the review
     fingerprint staling when the vocabulary grows.

Everything runs against stubs — no live reference data, no network.

Run:  PYTHONPATH=. python -m pytest tests/test_observables.py -q
"""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import tools.observables as obs_mod
from tools.observables import (_advisory_identify, _advisory_signature,
                               all_observables, code_of_key, emission_of,
                               load_auto_observables, observable_name_of,
                               record_signatures,
                               validate_observable_source)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def _record(warns=None):
    return {
        "document_id": "doc1",
        "cpt_codes": [{"code": "11111", "modifiers": [], "units": 1}],
        "icd_codes": [], "hcpcs_codes": [],
        "claim_scrub": {"findings": (warns if warns is not None else [
            {"filter_id": "MEDICAL_NECESSITY", "status": "WARN",
             "codes": ["11111"], "reason": "pathway A required"},
            {"filter_id": "DOCUMENTATION", "status": "PASS",
             "codes": ["11111"], "reason": "fine"},
        ])},
    }


_VALID_OBSERVABLE_SRC = '''\
OBSERVABLE_NAME = "linkage_annotation_emission"
SCHEMA_DOC = "Measures linkage annotations; realized by rules."
FINDING_KINDS = ("linkage_annotation_defect",)


def identify(result, finding):
    code = str(finding.get("code") or "").upper()
    if not code:
        return None, "no code"
    hits = [a for a in (result.get("linkage_annotations") or [])
            if str(a.get("code") or "").upper() == code]
    if len(hits) != 1:
        return None, "zero or several annotations"
    return "LINKAGE|" + code, str(hits[0].get("text") or "")


def signature(result):
    out = set()
    for a in (result.get("linkage_annotations") or []):
        c = str(a.get("code") or "").upper()
        if c:
            out.add("LINKAGE|" + c)
    return out
'''


# ---------------------------------------------------------------------------
# 1. built-in advisory_emission observable
# ---------------------------------------------------------------------------

class AdvisoryEmissionBuiltinTest(unittest.TestCase):
    def test_identify_resolves_unique_warn(self):
        key, why = _advisory_identify(_record(),
                                      {"code": "11111"})
        self.assertEqual(key, "MEDICAL_NECESSITY|11111")
        self.assertIn("pathway A required", why)

    def test_identify_no_warn_is_ambiguous(self):
        key, why = _advisory_identify(_record(warns=[]), {"code": "11111"})
        self.assertIsNone(key)
        self.assertIn("not identifiable", why)

    def test_identify_multiple_warns_is_ambiguous(self):
        rec = _record(warns=[
            {"filter_id": "A", "status": "WARN", "codes": ["11111"]},
            {"filter_id": "B", "status": "WARN", "codes": ["11111"]}])
        key, why = _advisory_identify(rec, {"code": "11111"})
        self.assertIsNone(key)
        self.assertIn("ambiguous", why)

    def test_identify_without_code_declines(self):
        key, _ = _advisory_identify(_record(), {"code": ""})
        self.assertIsNone(key)

    def test_signature_reads_only_warns_and_does_not_mutate(self):
        rec = _record()
        frozen = json.dumps(rec, sort_keys=True)
        self.assertEqual(_advisory_signature(rec),
                         {"MEDICAL_NECESSITY|11111"})
        self.assertEqual(json.dumps(rec, sort_keys=True), frozen)

    def test_builtin_is_registered_and_claims_advisory_defect(self):
        entry = all_observables()["advisory_emission"]
        self.assertTrue(entry["builtin"])
        self.assertIn("advisory_defect", entry["finding_kinds"])

    def test_code_of_key(self):
        self.assertEqual(code_of_key("MEDICAL_NECESSITY|11111"), "11111")
        self.assertEqual(code_of_key("A|B|m77.31"), "M77.31")


# ---------------------------------------------------------------------------
# 2. generic registry events + merged legacy view
# ---------------------------------------------------------------------------

class AdjudicatedObservablesRegistryTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "registry.jsonl"

    def tearDown(self):
        self.tmp.cleanup()

    def _record(self, targets=None):
        from tools.claims_registry import record_adjudicated_observables
        return record_adjudicated_observables(
            "doc1",
            targets or [{"observable": "advisory_emission",
                         "key": "MEDICAL_NECESSITY|11111", "emit": False,
                         "authority": "LCD L12345"}],
            "doc1_results.json", by="coder-llm/test",
            registry_path=self.path)

    def test_records_and_reads_back(self):
        from tools.claims_registry import verified_observable_targets
        self.assertIsNotNone(self._record())
        t = verified_observable_targets(registry_path=self.path)
        self.assertEqual(t["doc1"][("advisory_emission",
                                    "MEDICAL_NECESSITY|11111")], False)

    def test_idempotent_and_latest_wins(self):
        from tools.claims_registry import (load_events,
                                           verified_observable_targets)
        self.assertIsNotNone(self._record())
        self.assertIsNone(self._record())
        self.assertEqual(len(load_events(self.path)), 1)
        self._record([{"observable": "advisory_emission",
                       "key": "MEDICAL_NECESSITY|11111", "emit": True,
                       "authority": "revised"}])
        t = verified_observable_targets(registry_path=self.path)
        self.assertTrue(t["doc1"][("advisory_emission",
                                   "MEDICAL_NECESSITY|11111")])

    def test_malformed_targets_are_dropped(self):
        self.assertIsNone(self._record(
            [{"observable": "", "key": "K|1", "emit": False},
             {"observable": "x", "key": "", "emit": False},
             {"observable": "x", "key": "K|1", "emit": "yes"}]))

    def test_legacy_advisory_events_merge_into_generic_view(self):
        from tools.claims_registry import (record_adjudicated_advisories,
                                           verified_advisory_targets,
                                           verified_observable_targets)
        record_adjudicated_advisories(
            "doc1", [{"filter_id": "MEDICAL_NECESSITY", "code": "11111",
                      "emit": False, "authority": "LCD"}],
            "doc1_results.json", by="coder-llm/test",
            registry_path=self.path)
        t = verified_observable_targets(registry_path=self.path)
        self.assertEqual(t["doc1"][("advisory_emission",
                                    "MEDICAL_NECESSITY|11111")], False)
        # and the advisory-specific view still round-trips
        a = verified_advisory_targets(registry_path=self.path)
        self.assertEqual(a["doc1"][("MEDICAL_NECESSITY", "11111")], False)


# ---------------------------------------------------------------------------
# 3. static gate + loader for synthesized observables
# ---------------------------------------------------------------------------

class ObservableStaticGateTest(unittest.TestCase):
    def test_valid_module_passes(self):
        self.assertEqual(validate_observable_source(_VALID_OBSERVABLE_SRC),
                         [])
        self.assertEqual(observable_name_of(_VALID_OBSERVABLE_SRC),
                         "linkage_annotation_emission")

    def _assert_rejected(self, src, needle):
        problems = validate_observable_source(src)
        self.assertTrue(any(needle in p for p in problems),
                        f"expected {needle!r} in {problems}")

    def test_rejects_missing_exports(self):
        self._assert_rejected("x = 1", "OBSERVABLE_NAME")
        self._assert_rejected("x = 1", "FINDING_KINDS")
        self._assert_rejected("x = 1", "def identify")
        self._assert_rejected("x = 1", "def signature")

    def test_rejects_io_and_dynamic_execution(self):
        self._assert_rejected(
            _VALID_OBSERVABLE_SRC + "\ndef f(result):\n"
            "    return open('/etc/passwd')\n", "open")
        self._assert_rejected(
            _VALID_OBSERVABLE_SRC + "\nimport os\n", "not allowed")

    def test_rejects_literal_medical_codes(self):
        bad = _VALID_OBSERVABLE_SRC.replace(
            '"LINKAGE|" + code', '"LINKAGE|" + "99213"')
        self._assert_rejected(bad, "literal medical code")

    def test_rejects_builtin_name_collision(self):
        bad = _VALID_OBSERVABLE_SRC.replace(
            'OBSERVABLE_NAME = "linkage_annotation_emission"',
            'OBSERVABLE_NAME = "advisory_emission"')
        self._assert_rejected(bad, "collides")

    def test_rejects_wrong_signatures(self):
        bad = _VALID_OBSERVABLE_SRC.replace(
            "def signature(result):", "def signature(result, extra):")
        self._assert_rejected(bad, "signature() signature")

    def test_rejects_while_loops(self):
        self._assert_rejected(
            _VALID_OBSERVABLE_SRC + "\ndef g(result):\n"
            "    while True:\n        pass\n", "while")


class ObservableLoaderTest(unittest.TestCase):
    def test_loads_valid_module_and_skips_defective(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            (d / "good.py").write_text(_VALID_OBSERVABLE_SRC)
            (d / "bad.py").write_text("import os\nOBSERVABLE_NAME='x'\n")
            with mock.patch.object(obs_mod, "AUTO_OBSERVABLES_DIR", d), \
                    mock.patch.dict(obs_mod._cache, {}, clear=True):
                loaded = load_auto_observables()
        self.assertEqual(sorted(loaded), ["linkage_annotation_emission"])
        entry = loaded["linkage_annotation_emission"]
        rec = {"linkage_annotations": [{"code": "11111", "text": "t"}]}
        self.assertEqual(entry["signature"](rec), {"LINKAGE|11111"})
        key, _ = entry["identify"](rec, {"code": "11111"})
        self.assertEqual(key, "LINKAGE|11111")

    def test_synthesized_module_cannot_shadow_builtin(self):
        shadow = _VALID_OBSERVABLE_SRC.replace(
            'OBSERVABLE_NAME = "linkage_annotation_emission"',
            'OBSERVABLE_NAME = "advisory_emission"')
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            (d / "shadow.py").write_text(shadow)
            with mock.patch.object(obs_mod, "AUTO_OBSERVABLES_DIR", d), \
                    mock.patch.dict(obs_mod._cache, {}, clear=True):
                # static gate rejects the collision at load time…
                self.assertEqual(load_auto_observables(), {})
                # …and all_observables prefers builtins regardless
                self.assertTrue(
                    all_observables()["advisory_emission"]["builtin"])


# ---------------------------------------------------------------------------
# 4. fail-closed measurement
# ---------------------------------------------------------------------------

class EmissionMeasurementTest(unittest.TestCase):
    def test_emission_of_reads_the_builtin(self):
        out = emission_of(_record(), {"advisory_emission":
                                      {"MEDICAL_NECESSITY|11111",
                                       "OTHER|99999"}})
        self.assertTrue(out[("advisory_emission",
                             "MEDICAL_NECESSITY|11111")])
        self.assertFalse(out[("advisory_emission", "OTHER|99999")])
        self.assertNotIn(("advisory_emission", "__error__"), out)

    def test_missing_observable_measures_error_marker(self):
        out = emission_of(_record(), {"ghost_observable": {"K|11111"}})
        self.assertFalse(out[("ghost_observable", "K|11111")])
        self.assertTrue(out[("ghost_observable", "__error__")])

    def test_crashing_observable_measures_error_marker(self):
        def boom(result):
            raise RuntimeError("measurement crashed")
        broken = dict(all_observables()["advisory_emission"],
                      signature=boom)
        with mock.patch.object(obs_mod, "_BUILTINS",
                               {"advisory_emission": broken}), \
                mock.patch.object(obs_mod, "load_auto_observables",
                                  return_value={}):
            out = emission_of(_record(), {"advisory_emission":
                                          {"MEDICAL_NECESSITY|11111"}})
        self.assertFalse(out[("advisory_emission",
                              "MEDICAL_NECESSITY|11111")])
        self.assertTrue(out[("advisory_emission", "__error__")])

    def test_record_signatures_sentinel_on_crash(self):
        def boom(result):
            raise RuntimeError("crashed")
        broken = dict(all_observables()["advisory_emission"],
                      signature=boom)
        with mock.patch.object(obs_mod, "_BUILTINS",
                               {"advisory_emission": broken}), \
                mock.patch.object(obs_mod, "load_auto_observables",
                                  return_value={}):
            sigs = record_signatures(_record())
        self.assertEqual(sigs["advisory_emission"],
                         frozenset({"__error__"}))

    def test_replay_gate_hit_fails_closed_on_error_marker(self):
        # a 'must not fire' goal reads emission False when the observable
        # crashed — the __error__ veto must keep that from counting as a
        # hit in gate_replay's convergence arithmetic
        goals = {("advisory_emission", "MEDICAL_NECESSITY|11111"): False}
        crashed = {("advisory_emission", "MEDICAL_NECESSITY|11111"): False,
                   ("advisory_emission", "__error__"): True}
        goal_obs = {k[0] for k in goals}
        hit = all(crashed.get(k) == emit for k, emit in goals.items()) \
            and not any(crashed.get((o, "__error__")) for o in goal_obs)
        self.assertFalse(hit)


# ---------------------------------------------------------------------------
# 5. gap detection + synthesis meta-gates
# ---------------------------------------------------------------------------

def _disputed_payload(kind="other", materiality="billing_material",
                      extra=None):
    rec = _record()
    rec.update(extra or {})
    rec["clinical_audit"] = {
        "verdict": "disputed", "fingerprint": "f" * 16,
        "claim_findings": [{
            "kind": kind, "array": "cpt_codes", "code": "11111",
            "materiality": materiality,
            "finding": "a linkage annotation the authorities contradict",
            "authority": "CMS manual", "note_evidence": "quote"}],
    }
    return rec


class GapDetectionTest(unittest.TestCase):
    def _detect(self, payload, ledger_sigs=None):
        from tools import observable_synthesis as osyn
        with tempfile.TemporaryDirectory() as td:
            rd = Path(td)
            (rd / "note_x_results.json").write_text(
                json.dumps(payload, default=str))
            with mock.patch.object(osyn, "_attempted_sigs",
                                   return_value=ledger_sigs or set()):
                return osyn.detect_gaps(rd, ["note_x"])

    def test_unknown_kind_routing_finding_is_a_gap(self):
        gaps = self._detect(_disputed_payload(kind="other"))
        self.assertEqual(len(gaps), 1)
        self.assertEqual(gaps[0]["document_id"], "note_x")

    def test_covered_kind_is_not_a_gap(self):
        gaps = self._detect(_disputed_payload(kind="advisory_defect"))
        self.assertEqual(gaps, [])

    def test_billing_mechanizable_kind_is_not_a_gap(self):
        gaps = self._detect(_disputed_payload(kind="wrong_code"))
        self.assertEqual(gaps, [])

    def test_advisory_materiality_grounded_finding_is_still_a_gap(self):
        # advisory_defect itself was materiality-"advisory" before its
        # observable existed — grounding, not materiality, is the bar
        gaps = self._detect(_disputed_payload(kind="other",
                                              materiality="advisory"))
        self.assertEqual(len(gaps), 1)

    def test_ungrounded_finding_is_not_a_gap(self):
        p = _disputed_payload(kind="other")
        p["clinical_audit"]["claim_findings"][0]["authority"] = ""
        self.assertEqual(self._detect(p), [])

    def test_undisputed_note_is_not_scanned(self):
        p = _disputed_payload()
        p["clinical_audit"]["verdict"] = "upheld"
        self.assertEqual(self._detect(p), [])

    def test_ledgered_gap_is_never_reburned(self):
        from tools import observable_synthesis as osyn
        p = _disputed_payload()
        sig = osyn._gap_sig("note_x",
                            p["clinical_audit"]["claim_findings"][0])
        self.assertEqual(self._detect(p, ledger_sigs={sig}), [])


class SynthesisMetaGatesTest(unittest.TestCase):
    """gate_design: every deterministic requirement a designed observable
    must satisfy before it may join the measurement vocabulary."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.rd = Path(self.tmp.name)
        self.record = _disputed_payload(
            extra={"linkage_annotations": [{"code": "11111",
                                            "text": "annotation"}]})
        (self.rd / "note_x_results.json").write_text(
            json.dumps(self.record, default=str))
        self.gap = {"document_id": "note_x",
                    "finding": self.record["clinical_audit"]
                    ["claim_findings"][0],
                    "gap_sig": "abc", "record": self.record}

    def tearDown(self):
        self.tmp.cleanup()

    def _gate(self, src, gap=None):
        from tools.observable_synthesis import gate_design
        return gate_design(src, gap or self.gap, self.rd)

    def test_valid_design_passes_all_gates(self):
        self.assertEqual(self._gate(_VALID_OBSERVABLE_SRC), "")

    def test_static_violation_rejected(self):
        self.assertIn("static gate",
                      self._gate("import os\n" + _VALID_OBSERVABLE_SRC))

    def test_finding_kind_clash_rejected(self):
        bad = _VALID_OBSERVABLE_SRC.replace(
            'FINDING_KINDS = ("linkage_annotation_defect",)',
            'FINDING_KINDS = ("advisory_defect",)')
        self.assertIn("vocabulary", self._gate(bad))

    def test_billing_kind_claim_rejected(self):
        bad = _VALID_OBSERVABLE_SRC.replace(
            'FINDING_KINDS = ("linkage_annotation_defect",)',
            'FINDING_KINDS = ("wrong_code",)')
        self.assertIn("vocabulary", self._gate(bad))

    def test_unresolvable_trigger_rejected(self):
        # the record has no linkage_annotations block -> identify returns
        # None -> the design does not close the gap that triggered it
        gap = dict(self.gap, record=_disputed_payload())
        self.assertIn("identity", self._gate(_VALID_OBSERVABLE_SRC, gap))

    def test_baseline_must_fire(self):
        # signature() that never contains the resolved key
        bad = _VALID_OBSERVABLE_SRC.replace(
            'out.add("LINKAGE|" + c)', "pass")
        self.assertIn("baseline", self._gate(bad))

    def test_mutating_signature_rejected(self):
        bad = _VALID_OBSERVABLE_SRC.replace(
            "def signature(result):\n    out = set()",
            "def signature(result):\n"
            "    result[\"poisoned\"] = True\n    out = set()")
        self.assertIn("purity", self._gate(bad))

    def test_corpus_crash_rejected(self):
        # a second saved record lacking the block the observable indexes
        # into unconditionally
        bad = _VALID_OBSERVABLE_SRC.replace(
            "for a in (result.get(\"linkage_annotations\") or []):",
            "for a in result[\"linkage_annotations\"]:")
        (self.rd / "other_results.json").write_text(
            json.dumps(_record(), default=str))
        self.assertIn("corpus", self._gate(bad))


class GrowObservablesDriverTest(unittest.TestCase):
    """The growth driver: designs are gated, installs land in the auto
    dir, declines and failures are ledgered and never re-burned."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.rd = Path(self.tmp.name) / "results"
        self.rd.mkdir()
        self.auto = Path(self.tmp.name) / "auto_observables"
        self.ledger = Path(self.tmp.name) / "ledger.jsonl"
        record = _disputed_payload(
            extra={"linkage_annotations": [{"code": "11111",
                                            "text": "annotation"}]})
        (self.rd / "note_x_results.json").write_text(
            json.dumps(record, default=str))

    def tearDown(self):
        self.tmp.cleanup()

    def _grow(self, design):
        from tools import observable_synthesis as osyn
        with mock.patch.object(osyn, "_design_once",
                               return_value=design), \
                mock.patch.object(osyn, "LEDGER_PATH", self.ledger), \
                mock.patch.object(osyn, "AUTO_OBSERVABLES_DIR",
                                  self.auto), \
                mock.patch.object(obs_mod, "AUTO_OBSERVABLES_DIR",
                                  self.auto), \
                mock.patch.dict(obs_mod._cache, {}, clear=True):
            return osyn.grow_observables(self.rd, ["note_x"])

    def test_accepted_design_installs_and_ledgers(self):
        n = self._grow({"decision": "observable",
                        "observable_code": _VALID_OBSERVABLE_SRC,
                        "rationale": "measurable"})
        self.assertEqual(n, 1)
        self.assertTrue(
            (self.auto / "linkage_annotation_emission.py").exists())
        entries = [json.loads(x) for x in
                   self.ledger.read_text().splitlines()]
        self.assertEqual(entries[0]["outcome"], "installed")

    def test_declined_design_installs_nothing_but_ledgers(self):
        n = self._grow({"decision": "decline",
                        "rationale": "a human judgment case"})
        self.assertEqual(n, 0)
        self.assertFalse(self.auto.exists())
        entries = [json.loads(x) for x in
                   self.ledger.read_text().splitlines()]
        self.assertEqual(entries[0]["outcome"], "declined")

    def test_gated_out_design_is_ledgered_failed(self):
        n = self._grow({"decision": "observable",
                        "observable_code": "import os\n",
                        "rationale": "bad"})
        self.assertEqual(n, 0)
        entries = [json.loads(x) for x in
                   self.ledger.read_text().splitlines()]
        self.assertEqual(entries[0]["outcome"], "failed")

    def test_ledgered_gap_not_reattempted(self):
        self._grow({"decision": "decline", "rationale": "human case"})
        from tools import observable_synthesis as osyn
        # Pin AUTO_OBSERVABLES_DIR to the same sandbox the decline was
        # ledgered under: the no-retry contract is scoped to one vocabulary
        # EPOCH, and the epoch hashes the loaded observables. Without the
        # pin this second call reads the REAL auto-observables directory —
        # once the live system grows its first observable there, the epoch
        # legitimately differs and the decline becomes retryable, which is
        # the retry-on-growth feature, not the no-retry bug this guards.
        with mock.patch.object(osyn, "LEDGER_PATH", self.ledger), \
                mock.patch.object(osyn, "AUTO_OBSERVABLES_DIR", self.auto), \
                mock.patch.object(obs_mod, "AUTO_OBSERVABLES_DIR", self.auto), \
                mock.patch.dict(obs_mod._cache, {}, clear=True), \
                mock.patch.object(osyn, "_design_once") as dsn:
            n = osyn.grow_observables(self.rd, ["note_x"])
        self.assertEqual(n, 0)
        dsn.assert_not_called()


# ---------------------------------------------------------------------------
# 6. loop wiring + fingerprint staling
# ---------------------------------------------------------------------------

class LoopGrowthWiringTest(unittest.TestCase):
    """converge(): a would-be stall first attempts measurement growth; an
    installed observable continues the loop, no growth stalls it."""

    def _converge(self, grown):
        from tools.audit_convergence_loop import converge
        with tempfile.TemporaryDirectory() as td:
            rd = Path(td)
            (rd / "note_x_results.json").write_text(
                json.dumps(_disputed_payload(), default=str))
            fake_triage = mock.MagicMock()
            fake_triage.scan = mock.MagicMock()
            with mock.patch("tools.clinical_auditor.audit_batch",
                            return_value={"audited": 0}), \
                    mock.patch("tools.coder_adjudicator.adjudicate_audit",
                               return_value={"adjudicated": 0,
                                             "partial": 0,
                                             "still_disputed": 1}), \
                    mock.patch.dict("sys.modules",
                                    {"tools.flip_triage": fake_triage}), \
                    mock.patch("tools.auto_actuate.actuate",
                               return_value={"actuated": 0,
                                             "resolved_baseline": 0}), \
                    mock.patch("tools.audit_convergence_loop.replay_scope",
                               return_value=0), \
                    mock.patch("tools.observable_synthesis."
                               "grow_observables",
                               side_effect=grown) as g, \
                    mock.patch("tools.claims_registry.ingest",
                               return_value={}):
                summary = converge(rd, docs=["note_x"], max_iterations=3,
                                   rep=mock.MagicMock())
        return summary, g

    def test_no_growth_stalls(self):
        summary, g = self._converge(grown=[0])
        self.assertEqual(summary["status"], "stalled")
        g.assert_called_once()

    def test_growth_continues_the_loop(self):
        # iteration 1 stalls -> grows 1 observable -> continues;
        # iteration 2 stalls -> no growth -> stalls for a human
        summary, g = self._converge(grown=[1, 0])
        self.assertEqual(summary["status"], "stalled")
        self.assertEqual(g.call_count, 2)
        self.assertEqual(summary.get("observables_installed"), 1)
        self.assertEqual(
            summary["iterations"][0].get("observables_installed"), 1)


class VocabularyFingerprintTest(unittest.TestCase):
    def test_fingerprint_stales_when_vocabulary_grows(self):
        # installing an observable must invalidate every review verdict:
        # the notes re-run against the grown system, mechanically
        from tools import clinical_auditor as ca
        result = _disputed_payload()
        before = ca.corrections_fingerprint(result)
        with mock.patch.object(
                ca, "_observable_kinds",
                return_value={"advisory_defect",
                              "linkage_annotation_defect"}):
            after = ca.corrections_fingerprint(result)
        self.assertNotEqual(before, after)

    def test_prompt_supplement_lists_synthesized_kinds_only(self):
        from tools import clinical_auditor as ca
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            (d / "linkage.py").write_text(_VALID_OBSERVABLE_SRC)
            with mock.patch.object(obs_mod, "AUTO_OBSERVABLES_DIR", d), \
                    mock.patch.dict(obs_mod._cache, {}, clear=True):
                sup = ca._vocabulary_supplement()
        self.assertIn("linkage_annotation_defect", sup)
        self.assertIn("linkage_annotation_emission", sup)
        # builtins are covered by the static prompt, not the supplement
        self.assertNotIn('"advisory_defect"', sup)

    def test_supplement_empty_without_synthesized_observables(self):
        from tools import clinical_auditor as ca
        with tempfile.TemporaryDirectory() as td:
            with mock.patch.object(obs_mod, "AUTO_OBSERVABLES_DIR",
                                   Path(td)), \
                    mock.patch.dict(obs_mod._cache, {}, clear=True):
                self.assertEqual(ca._vocabulary_supplement(), "")


# ---------------------------------------------------------------------------
# vocabulary-epoch ledger: declined gaps retry when the vocabulary grows
# ---------------------------------------------------------------------------

class EpochLedgerTest(unittest.TestCase):
    def _synth(self):
        import tools.observable_synthesis as osyn
        return osyn

    def test_epoch_changes_when_vocabulary_grows(self):
        osyn = self._synth()
        base = {"advisory_emission": {"finding_kinds":
                                      ("advisory_defect",)}}
        grown = dict(base, linkage={"finding_kinds":
                                    ("linkage_annotation_defect",)})
        with mock.patch.object(osyn, "all_observables",
                               return_value=base):
            e1 = osyn._vocab_epoch()
        with mock.patch.object(osyn, "all_observables",
                               return_value=grown):
            e2 = osyn._vocab_epoch()
        self.assertNotEqual(e1, e2)

    def test_declined_gap_retries_in_a_new_epoch_only(self):
        osyn = self._synth()
        with tempfile.TemporaryDirectory() as td:
            ledger = Path(td) / "ledger.jsonl"
            base = {"advisory_emission": {"finding_kinds":
                                          ("advisory_defect",)}}
            grown = dict(base, x={"finding_kinds": ("x_defect",)})
            with mock.patch.object(osyn, "LEDGER_PATH", ledger):
                with mock.patch.object(osyn, "all_observables",
                                       return_value=base):
                    osyn._ledger({"gap_sig": "g1", "outcome": "declined"})
                    # same epoch: attempted, not retried
                    self.assertIn("g1", osyn._attempted_sigs())
                with mock.patch.object(osyn, "all_observables",
                                       return_value=grown):
                    # vocabulary grew: the decline is stale, retryable
                    self.assertNotIn("g1", osyn._attempted_sigs())

    def test_installed_gap_never_retries(self):
        osyn = self._synth()
        with tempfile.TemporaryDirectory() as td:
            ledger = Path(td) / "ledger.jsonl"
            base = {"advisory_emission": {"finding_kinds":
                                          ("advisory_defect",)}}
            grown = dict(base, x={"finding_kinds": ("x_defect",)})
            with mock.patch.object(osyn, "LEDGER_PATH", ledger):
                with mock.patch.object(osyn, "all_observables",
                                       return_value=base):
                    osyn._ledger({"gap_sig": "g1",
                                  "outcome": "installed"})
                with mock.patch.object(osyn, "all_observables",
                                       return_value=grown):
                    self.assertIn("g1", osyn._attempted_sigs())

    def test_pre_epoch_ledger_entries_are_retryable(self):
        osyn = self._synth()
        with tempfile.TemporaryDirectory() as td:
            ledger = Path(td) / "ledger.jsonl"
            ledger.write_text(json.dumps(
                {"gap_sig": "old", "outcome": "declined"}) + "\n")
            with mock.patch.object(osyn, "LEDGER_PATH", ledger):
                self.assertNotIn("old", osyn._attempted_sigs())


if __name__ == "__main__":
    unittest.main()
