"""Tests for the automated expert-coder adjudicator (tools/coder_adjudicator)
and the registry's 'adjudicated' verification tier.

The full adjudication path needs the reference DB + compliance store and a
live model, so the end-to-end flow is exercised in-container. Here we prove
the trust machinery — the parts that make the adjudicator safe to automate:
verdicts are void unless complete, grounded, and inside the disputed scope;
independent passes must agree; mechanical application touches ONLY the
disputed items; and registry precedence (human > adjudicated > auto) can
never be violated by a lower tier.
"""

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.coder_adjudicator import (
    _ADJUDICATOR_PROMPT, _apply_to_run, _item_key, _norm_decision,
    _verdict_map, adjudicate)
from tools import claims_registry as reg


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

_DISPUTED = [
    {"array": "cpt_codes", "code": "99213", "kind": "presence",
     "advisory": False, "present_in_runs": 2, "runs": 3},
    {"array": "cpt_codes", "code": "11056", "kind": "attributes",
     "advisory": False, "fields": ["modifiers"],
     "values": [{"modifiers": ["59"]}, {"modifiers": []}], "runs": 3},
]


def _verdict(decision_99213="exclude", mods_11056=None, grounded=True):
    g = {"authority": "NCCI Policy Manual Ch.1 — E/M inherent to minor "
                      "procedure is bundled",
         "note_evidence": "No separately identifiable E/M documented beyond "
                          "the procedure evaluation."} if grounded else \
        {"authority": "", "note_evidence": ""}
    return {"items": [
        {"array": "cpt_codes", "code": "99213", "kind": "presence",
         "decision": decision_99213, **g},
        {"array": "cpt_codes", "code": "11056", "kind": "attributes",
         "decision": "set",
         "fields": {"modifiers": mods_11056 if mods_11056 is not None else []},
         **g},
    ], "overall_rationale": "test"}


def _run(cpt, icd=None, hcpcs=None):
    return {"cpt_codes": cpt, "icd_codes": icd or [], "hcpcs_codes":
            hcpcs or [], "supporting_conditions": []}


# ---------------------------------------------------------------------------
# verdict validation — incomplete / ungrounded / out-of-scope verdicts void
# ---------------------------------------------------------------------------

class VerdictMapTest(unittest.TestCase):
    def test_complete_grounded_verdict_maps(self):
        m = _verdict_map(_verdict(), _DISPUTED)
        self.assertIsNotNone(m)
        self.assertEqual(len(m), 2)
        self.assertEqual(m[_item_key(_DISPUTED[0])], ("exclude",))

    def test_abstain_voids_the_whole_verdict(self):
        self.assertIsNone(_verdict_map(_verdict("abstain"), _DISPUTED))

    def test_ungrounded_decision_voids_the_verdict(self):
        # authority + note evidence are mandatory: judgment without a cited
        # source is exactly what this role is forbidden to exercise
        self.assertIsNone(_verdict_map(_verdict(grounded=False), _DISPUTED))

    def test_missing_item_voids_the_verdict(self):
        v = _verdict()
        v["items"] = v["items"][:1]  # silently skipped one disputed item
        self.assertIsNone(_verdict_map(v, _DISPUTED))

    def test_invented_scope_voids_the_verdict(self):
        v = _verdict()
        v["items"].append({"array": "cpt_codes", "code": "97597",
                           "kind": "presence", "decision": "exclude",
                           "authority": "x", "note_evidence": "y"})
        self.assertIsNone(_verdict_map(v, _DISPUTED))

    def test_modifier_normalization_makes_passes_comparable(self):
        a = _verdict_map(_verdict(mods_11056=["lt", "59"]), _DISPUTED)
        b = _verdict_map(_verdict(mods_11056=["59", "LT"]), _DISPUTED)
        self.assertEqual(a, b)

    def test_split_decisions_compare_unequal(self):
        a = _verdict_map(_verdict("exclude"), _DISPUTED)
        b = _verdict_map(_verdict("include"), _DISPUTED)
        self.assertNotEqual(a, b)

    def test_em_level_select_requires_decision_code(self):
        item = {"array": "cpt_codes", "kind": "em_level",
                "codes": ["99213", "99214"], "decision": "select"}
        self.assertIsNone(_norm_decision(item))
        item["decision_code"] = "99213"
        self.assertEqual(_norm_decision(item), ("select", "99213"))

    def test_em_level_select_outside_family_is_void(self):
        # the adjudicator may pick among the codes the runs produced —
        # inventing a NEW E/M level is outside its authority
        item = {"array": "cpt_codes", "kind": "em_level",
                "codes": ["99213", "99214"], "decision": "select",
                "decision_code": "99215"}
        self.assertIsNone(_norm_decision(item))

    def test_duplicate_items_void_the_verdict(self):
        v = _verdict()
        v["items"].append(dict(v["items"][0], decision="include"))
        self.assertIsNone(_verdict_map(v, _DISPUTED))


# ---------------------------------------------------------------------------
# mechanical application — only the disputed items can move
# ---------------------------------------------------------------------------

class ApplyVerdictTest(unittest.TestCase):
    def test_presence_exclude_removes_only_that_code(self):
        run = _run([{"code": "99213", "modifiers": ["25"]},
                    {"code": "11056", "modifiers": []}])
        decisions = {_item_key(_DISPUTED[0]): ("exclude",),
                     _item_key(_DISPUTED[1]): ("set", (("modifiers", "[]"),))}
        out = _apply_to_run(run, decisions, _DISPUTED, [run])
        self.assertEqual([e["code"] for e in out["cpt_codes"]], ["11056"])

    def test_presence_include_pulls_entry_from_another_run(self):
        d = [_DISPUTED[0]]
        with_em = _run([{"code": "99213", "modifiers": ["25"],
                         "description": "office visit"}])
        without = _run([{"code": "11056", "modifiers": []}])
        decisions = {_item_key(d[0]): ("include",)}
        out = _apply_to_run(without, decisions, d, [with_em, without])
        codes = {e["code"] for e in out["cpt_codes"]}
        self.assertEqual(codes, {"11056", "99213"})
        added = next(e for e in out["cpt_codes"] if e["code"] == "99213")
        self.assertEqual(added["modifiers"], ["25"])  # full entry, not a stub

    def test_attributes_set_touches_only_disputed_fields(self):
        run = _run([{"code": "11056", "modifiers": ["59"], "units": 2,
                     "description": "paring"}])
        d = [_DISPUTED[1]]
        decisions = {_item_key(d[0]):
                     ("set", (("modifiers", "[]"), ("units", "9")))}
        out = _apply_to_run(run, decisions, d, [run])
        e = out["cpt_codes"][0]
        self.assertEqual(e["modifiers"], [])
        # 'units' was NOT among the disputed fields — the adjudicator has no
        # authority over it, so the attempted write is discarded
        self.assertEqual(e["units"], 2)
        self.assertEqual(e["description"], "paring")

    def test_attributes_with_no_disputed_fields_grants_nothing(self):
        # fail CLOSED: a disputed item missing its 'fields' list is a
        # malformed report — the adjudicator gets no write access at all
        run = _run([{"code": "11056", "modifiers": ["59"], "units": 2}])
        d = [{"array": "cpt_codes", "code": "11056", "kind": "attributes",
              "advisory": False, "runs": 3}]  # no "fields"
        decisions = {_item_key(d[0]): ("set", (("modifiers", "[]"),
                                               ("units", "9")))}
        out = _apply_to_run(run, decisions, d, [run])
        e = out["cpt_codes"][0]
        self.assertEqual(e["modifiers"], ["59"])
        self.assertEqual(e["units"], 2)

    def test_unrealizable_include_is_detected(self):
        from tools.coder_adjudicator import _decisions_applicable
        d = [{"array": "cpt_codes", "code": "97597", "kind": "presence",
              "advisory": False, "runs": 3}]
        runs = [_run([{"code": "11056", "modifiers": []}])]
        decisions = {_item_key(d[0]): ("include",)}
        why = _decisions_applicable(decisions, d, runs)
        self.assertIsNotNone(why)
        self.assertIn("97597", why)
        # exclude needs no source entry — always realizable
        decisions = {_item_key(d[0]): ("exclude",)}
        self.assertIsNone(_decisions_applicable(decisions, d, runs))

    def test_icd_exclusion_demotes_to_supporting_conditions(self):
        d = [{"array": "icd_codes", "code": "E11.9", "kind": "presence",
              "advisory": False, "runs": 3}]
        run = _run([], icd=[{"code": "E11.9", "type": "secondary"}])
        decisions = {_item_key(d[0]): ("exclude",)}
        out = _apply_to_run(run, decisions, d, [run])
        self.assertEqual(out["icd_codes"], [])
        self.assertEqual(out["supporting_conditions"][0]["code"], "E11.9")
        self.assertEqual(out["supporting_conditions"][0]["demoted_by"],
                         "coder_adjudication")

    def test_em_level_select_keeps_exactly_one_family_member(self):
        d = [{"array": "cpt_codes", "kind": "em_level", "advisory": False,
              "codes": ["99213", "99214"], "runs": 3}]
        run214 = _run([{"code": "99214", "modifiers": ["25"]},
                       {"code": "11720", "modifiers": []}])
        run213 = _run([{"code": "99213", "modifiers": ["25"]},
                       {"code": "11720", "modifiers": []}])
        decisions = {_item_key(d[0]): ("select", "99213")}
        out = _apply_to_run(run214, decisions, d, [run214, run213])
        codes = [e["code"] for e in out["cpt_codes"]]
        self.assertIn("99213", codes)
        self.assertNotIn("99214", codes)
        self.assertIn("11720", codes)  # undisputed line untouched

    def test_included_code_is_canonicalized_across_runs(self):
        # A presence flip masks attribute comparison; if each run kept its
        # OWN entry for an included code, a settled presence flip could
        # resurface as an attributes flip (type/modifiers differ) and
        # strand the note split. Every run must get the same entry.
        d = [{"array": "icd_codes", "code": "M71.571", "kind": "presence",
              "advisory": False, "runs": 2}]
        run_a = _run([], icd=[{"code": "M71.571", "type": "primary"}])
        run_b = _run([], icd=[{"code": "M71.571", "type": "secondary"}])
        decisions = {_item_key(d[0]): ("include",)}
        out_a = _apply_to_run(run_a, decisions, d, [run_a, run_b])
        out_b = _apply_to_run(run_b, decisions, d, [run_a, run_b])
        self.assertEqual(out_a["icd_codes"], out_b["icd_codes"])
        self.assertEqual(out_a["icd_codes"][0]["type"], "primary")

    def test_undisputed_arrays_are_untouched(self):
        run = _run([{"code": "99213", "modifiers": []},
                    {"code": "11056", "modifiers": ["59"]}],
                   icd=[{"code": "L84", "type": "primary"}],
                   hcpcs=[{"code": "A5500", "modifiers": []}])
        decisions = {_item_key(_DISPUTED[0]): ("exclude",),
                     _item_key(_DISPUTED[1]): ("set", (("modifiers", "[]"),))}
        out = _apply_to_run(run, decisions, _DISPUTED, [run])
        self.assertEqual(out["icd_codes"], run["icd_codes"])
        self.assertEqual(out["hcpcs_codes"], run["hcpcs_codes"])


# ---------------------------------------------------------------------------
# orchestration — split verdicts and abstentions never touch a result file
# ---------------------------------------------------------------------------

class AdjudicateOrchestrationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.results = Path(self.tmp.name)
        (self.results / "consistency_runs").mkdir()
        result = {
            "document_id": "noteX", "success": True,
            "consistency": {"runs": 2, "unanimous": False,
                            "disagreements": [_DISPUTED[0]]},
            "rag_context": {"note_full_text": "Office visit. Nail care."},
            "final_disposition": "CLEAN",
        }
        (self.results / "noteX_results.json").write_text(json.dumps(result))
        for i, cpt in enumerate(([{"code": "99213", "modifiers": []}], []), 1):
            (self.results / "consistency_runs" / f"noteX_run{i}.json"
             ).write_text(json.dumps(dict(
                 _run(cpt), rag_context={"note_full_text": "Office visit."})))
        self.original = (self.results / "noteX_results.json").read_text()

    def tearDown(self):
        self.tmp.cleanup()

    def _adjudicate(self, verdicts):
        rep = mock.MagicMock()
        seq = iter(verdicts)
        with mock.patch("app.compliance.engine.ClaimScrubber"), \
                mock.patch("app.compliance.agents.build_default_agents"), \
                mock.patch("tools.coder_adjudicator._adjudicate_once",
                           side_effect=lambda case, pass_idx=0: next(seq)):
            return adjudicate(self.results, docs=["noteX"], rep=rep,
                              passes=2)

    def test_split_verdicts_leave_result_untouched(self):
        d = [_DISPUTED[0]]
        v1 = {"items": [dict(_verdict()["items"][0], decision="exclude")]}
        v2 = {"items": [dict(_verdict()["items"][0], decision="include")]}
        stats = self._adjudicate([v1, v2])
        self.assertEqual(stats["split_verdicts"], 1)
        self.assertEqual(stats["adjudicated"], 0)
        self.assertEqual((self.results / "noteX_results.json").read_text(),
                         self.original)

    def test_abstention_leaves_result_untouched(self):
        v = {"items": [dict(_verdict()["items"][0], decision="abstain")]}
        stats = self._adjudicate([v, v])
        self.assertEqual(stats["abstained"], 1)
        self.assertEqual(stats["adjudicated"], 0)
        self.assertEqual((self.results / "noteX_results.json").read_text(),
                         self.original)


# ---------------------------------------------------------------------------
# the adjudication protocol itself — the binding constraints must be present
# ---------------------------------------------------------------------------

class ProtocolContractTest(unittest.TestCase):
    def test_pass_floor_is_two(self):
        # a single pass has nothing to agree with — the floor is structural
        import importlib
        import tools.coder_adjudicator as ca
        with mock.patch.dict("os.environ",
                             {"CODER_ADJUDICATION_PASSES": "1"}):
            importlib.reload(ca)
            self.assertEqual(ca.ADJUDICATION_PASSES, 2)
        importlib.reload(ca)

    def test_prompt_binds_authority_and_abstention(self):
        for clause in ("AUTHORITY, NOT INTUITION",
                       "if it\n   is not documented, it was not done",
                       "NCCI Policy Manual Chapter 1",
                       "ICD-10-CM Official\n   Guidelines",
                       "DECIDE ONLY THE DISPUTED ITEMS",
                       "ABSTAIN WHEN THE AUTHORITIES DO NOT DECIDE"):
            self.assertIn(clause, _ADJUDICATOR_PROMPT)

    def test_prompt_mandates_conservative_defaults(self):
        self.assertIn("the lower level is correct", _ADJUDICATOR_PROMPT)
        self.assertIn("the E/M is NOT separately\n     billable",
                      _ADJUDICATOR_PROMPT)


# ---------------------------------------------------------------------------
# MDM grid — E/M leveling authority as licensed data, not model memory
# ---------------------------------------------------------------------------

class MdmGridTest(unittest.TestCase):
    """store.mdm_requirements(code): level from the code's OWN descriptor,
    row from the grid revision in force on the DOS — no code tables."""

    def setUp(self):
        import sqlite3
        from app.compliance.datastore.store import ComplianceDataStore
        self.store = object.__new__(ComplianceDataStore)
        self.store._conn = sqlite3.connect(":memory:")  # behind .conn property
        self.store._conn.row_factory = sqlite3.Row
        self.store.conn.execute(
            "CREATE TABLE code_set (code_system TEXT, code TEXT, "
            "description TEXT, effective_from TEXT, effective_to TEXT, "
            "status TEXT)")
        self.store.conn.executemany(
            "INSERT INTO code_set VALUES ('CPT', ?, ?, '1900-01-01', "
            "'9999-12-31', 'active')",
            [("99213", "Office visit ... established patient, which "
                       "requires ... low level of medical decision "
                       "making ..."),
             ("99214", "Office visit ... established patient, which "
                       "requires ... moderate level of medical decision "
                       "making ..."),
             ("99211", "Office visit ... that may not require the "
                       "presence of a physician ..."),
             ("11720", "Debridement of nail(s) by any method(s); 1 to 5")])

    def tearDown(self):
        self.store.conn.close()

    def test_level_read_from_descriptor(self):
        self.assertEqual(self.store.em_mdm_level("99213"), "low")
        self.assertEqual(self.store.em_mdm_level("99214"), "moderate")
        self.assertIsNone(self.store.em_mdm_level("99211"))  # no MDM phrase
        self.assertIsNone(self.store.em_mdm_level("11720"))  # not E/M

    def test_requirements_join_level_to_grid_row(self):
        req = self.store.mdm_requirements("99214", dos="2026-07-20")
        if req is None:  # grid file not present in this deployment
            self.skipTest("em_mdm_grid.json not installed")
        self.assertEqual(req["level"], "moderate")
        probs = " ".join(req["requirements"]["problems_addressed"]).lower()
        self.assertIn("2 or more stable, chronic illnesses", probs)
        self.assertIn("2 of the 3 elements", req["selection_rule"])
        risk = req["requirements"]["risk_of_management"]
        self.assertIn("Prescription drug management", risk["examples"])

    def test_effective_dating_selects_revision(self):
        g2021 = self.store.mdm_grid(dos="2021-06-01")
        g2026 = self.store.mdm_grid(dos="2026-07-20")
        if not (g2021 and g2026):
            self.skipTest("em_mdm_grid.json not installed")
        # the 2023 revision added hospital-care rows to the low level
        low_2021 = " ".join(g2021["levels"]["low"]["problems_addressed"])
        low_2026 = " ".join(g2026["levels"]["low"]["problems_addressed"])
        self.assertNotIn("hospital inpatient", low_2021)
        self.assertIn("hospital inpatient", low_2026)

    def test_non_mdm_codes_get_no_row(self):
        self.assertIsNone(self.store.mdm_requirements("11720"))
        self.assertIsNone(self.store.mdm_requirements("99211"))


# ---------------------------------------------------------------------------
# registry — adjudicated tier precedence
# ---------------------------------------------------------------------------

def _event(doc, verification, code="11720"):
    return {"registry_version": 1, "event": "finalized",
            "document_id": doc, "recorded_at": "2026-07-20T00:00:00+00:00",
            "verification": verification, "verified_by": "t",
            "source": "s",
            "claim": {"icd_codes": [], "cpt_codes": [{"code": code}],
                      "hcpcs_codes": [], "final_disposition": "CLEAN"}}


class RegistryTierTest(unittest.TestCase):
    def test_adjudicated_outranks_auto(self):
        view = reg.current_view([_event("n", "adjudicated"),
                                 _event("n", "auto", code="99999")])
        self.assertEqual(view["n"]["verification"], "adjudicated")

    def test_human_outranks_adjudicated(self):
        view = reg.current_view([_event("n", "human"),
                                 _event("n", "adjudicated", code="99999")])
        self.assertEqual(view["n"]["verification"], "human")

    def test_same_tier_latest_wins(self):
        view = reg.current_view([_event("n", "adjudicated"),
                                 _event("n", "adjudicated", code="99999")])
        self.assertEqual(view["n"]["claim"]["cpt_codes"][0]["code"], "99999")

    def test_record_adjudicated_never_displaces_human(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "reg.jsonl"
            reg.append_events([_event("n", "human")], path)
            result = {"document_id": "n", "final_disposition": "CLEAN",
                      "cpt_codes": [{"code": "11720"}], "icd_codes": [],
                      "hcpcs_codes": [], "consistency": {}}
            ev = reg.record_adjudicated("n", result, "src.json",
                                        by="coder-llm/test",
                                        registry_path=path)
            self.assertIsNone(ev)
            view = reg.current_view(reg.load_events(path))
            self.assertEqual(view["n"]["verification"], "human")

    def test_record_adjudicated_is_idempotent_on_unchanged_claim(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "reg.jsonl"
            result = {"document_id": "n", "final_disposition": "CLEAN",
                      "cpt_codes": [{"code": "11720"}], "icd_codes": [],
                      "hcpcs_codes": [], "consistency": {},
                      "adjudication": {"model": "m", "passes": 2}}
            ev1 = reg.record_adjudicated("n", result, "src.json",
                                         by="coder-llm/test",
                                         registry_path=path)
            self.assertIsNotNone(ev1)
            self.assertEqual(ev1["verification"], "adjudicated")
            ev2 = reg.record_adjudicated("n", result, "src.json",
                                         by="coder-llm/test",
                                         registry_path=path)
            self.assertIsNone(ev2)
            self.assertEqual(len(reg.load_events(path)), 1)

    def test_ingest_never_displaces_adjudicated(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "reg.jsonl"
            results = Path(tmp) / "results"
            results.mkdir()
            reg.append_events([_event("n", "adjudicated")], path)
            from tools.clinical_auditor import corrections_fingerprint
            payload = {
                "document_id": "n", "success": True,
                "final_disposition": "CLEAN",
                "consistency": {"runs": 3, "unanimous": True},
                "cpt_codes": [{"code": "99999"}], "icd_codes": [],
                "hcpcs_codes": []}
            # the universal review gate sits before tier protection — this
            # test targets the tier branch, so satisfy the gate
            payload["clinical_audit"] = {
                "verdict": "upheld",
                "fingerprint": corrections_fingerprint(payload)}
            (results / "n_results.json").write_text(json.dumps(payload))
            with mock.patch(
                    "app.release.claim_readiness.verify_readiness_certificate",
                    return_value=(True, "")):
                stats = reg.ingest(results, path)
            self.assertEqual(stats["recorded"], 0)
            self.assertEqual(stats["human_protected"], 1)
            view = reg.current_view(reg.load_events(path))
            self.assertEqual(view["n"]["verification"], "adjudicated")


class ProposedCodeValidationTest(unittest.TestCase):
    """The deterministic authoritative gate on an audit-PROPOSED code (one no
    run bills, that a review/finding/exploratory lead wants added). It grounds
    the fix in DATA, not the reviewer's say-so: a proposal materializes only
    when it exists, its descriptor matches the documented work, it is billable,
    and it raises no unbypassable NCCI conflict. Mock DB/store — the logic, not
    the reference data."""

    import types as _types

    def _rep(self, *, rec, mue=1, gp="090", ncci=None):
        import types
        db = types.SimpleNamespace(
            validate_cpt=lambda c: rec,
            validate_hcpcs=lambda c: rec,
            validate_icd10=lambda c: rec,
            get_mue=lambda c: mue,
            ncci_data_available=lambda dos: True,
            check_ncci=lambda a, b, dos=None: ncci)
        store = types.SimpleNamespace(global_period=lambda c, dos=None: gp)
        return types.SimpleNamespace(db=db, store=store)

    def _ok(self, rep, arr, code, note, procs=("Retrocalcaneal exostectomy "
                                               "of the calcaneus",),
            claim=None):
        from tools.coder_adjudicator import _proposed_code_authoritative_ok
        main = {"cpt_codes": [{"code": c} for c in (claim or ["27654"])],
                "patient_metadata": {"date_of_service": "2026-01-05"},
                "procedures_performed_today": list(procs)}
        return _proposed_code_authoritative_ok(rep, arr, code, main, note)

    _OSTECTOMY = {"long_description": "Ostectomy, calcaneus"}
    _NOTE = "Retrocalcaneal exostectomy with Haglund resection of the calcaneus."

    def test_valid_proposal_passes(self):
        rep = self._rep(rec=self._OSTECTOMY)
        ok, why = self._ok(rep, "cpt_codes", "28118", self._NOTE)
        self.assertTrue(ok, why)

    def test_not_in_reference_data_refused(self):
        rep = self._rep(rec=None)
        ok, why = self._ok(rep, "cpt_codes", "99999", self._NOTE)
        self.assertFalse(ok)
        self.assertIn("reference data", why)

    def test_descriptor_not_grounded_refused(self):
        # a real code whose descriptor has nothing to do with the note
        rep = self._rep(rec={"long_description":
                             "Office or other outpatient visit, established"})
        ok, why = self._ok(rep, "cpt_codes", "99213", self._NOTE)
        self.assertFalse(ok)
        self.assertIn("not grounded", why)

    def test_zero_mue_refused_even_when_documented(self):
        rep = self._rep(rec={"long_description": "Ostectomy, calcaneus"}, mue=0)
        ok, why = self._ok(rep, "cpt_codes", "28118", self._NOTE)
        self.assertFalse(ok)
        self.assertIn("MUE", why)

    def test_missing_global_period_refused(self):
        rep = self._rep(rec=self._OSTECTOMY, gp="")
        ok, why = self._ok(rep, "cpt_codes", "28118", self._NOTE)
        self.assertFalse(ok)
        self.assertIn("global", why)

    def test_ncci_hard_bundle_refused(self):
        rep = self._rep(rec=self._OSTECTOMY,
                        ncci={"code2": "28118", "modifier": "0"})
        ok, why = self._ok(rep, "cpt_codes", "28118", self._NOTE,
                           claim=["27654", "28100"])
        self.assertFalse(ok)
        self.assertIn("NCCI", why)

    def test_ncci_bypassable_edit_allowed(self):
        # indicator 1 (separation modifier may bypass) is NOT a hard bundle
        rep = self._rep(rec=self._OSTECTOMY,
                        ncci={"code2": "28118", "modifier": "1"})
        ok, why = self._ok(rep, "cpt_codes", "28118", self._NOTE,
                           claim=["27654", "28100"])
        self.assertTrue(ok, why)

    def test_diagnosis_only_needs_existence(self):
        # a proposed ICD is not procedure-descriptor matched
        rep = self._rep(rec={"description": "Acquired deformity of foot"})
        ok, why = self._ok(rep, "icd_codes", "M21.6X1", self._NOTE)
        self.assertTrue(ok, why)


if __name__ == "__main__":
    unittest.main()
