#!/usr/bin/env python3
"""Automated actuation: flip classes → declarative rules → converged runs.

This automates the loop that was previously manual engineering: review each
run-to-run billing disagreement against the authoritative sources, decide
what the correct billable coding is, and build a deterministic layer so the
note processes cleanly — repeated until unanimous.

For each OPEN class in the flip queue (tools/flip_triage.py):

  1. DOSSIER    — assemble the evidence a human reviewer would read: the
                  note text, each run's version of the flipping entry, and
                  the authoritative reference data (code descriptors, ICD
                  Tabular category text and instructional conventions, MUE
                  limits, NCCI pair edits) pulled live from the compliance
                  datastore — never from the model's memory.
  2. PROPOSAL   — a reasoning LLM drafts ONE declarative rule for an
                  existing rule-engine template (config, not code), or
                  answers "escalate" when no safe deterministic rule exists.
  3. GATES      — deterministic acceptance, all must pass:
                    structural   rule parses, template known, fields sane
                    no-hardcode  no literal medical codes anywhere in the
                                 rule (descriptor grammar / lexicons only)
                    convergence  replaying the stored per-run artifacts
                                 through the validator WITH the rule reduces
                                 distinct billing signatures on the class's
                                 documents (ideally to 1)…
                    no-harm      …never increases them on any document, and
                    inertness    leaves every already-unanimous note's
                                 replayed claim byte-identical.
  4. PROPOSE    — accepted candidates are written as immutable DRAFT
                  proposal artifacts. They never modify the active rule
                  pack or install executable templates. Human approval,
                  signing, shadow deployment, and rollback rehearsal are
                  separate required lifecycle stages.

Drafts remain inert until an independent promotion workflow reviews and
signs a pack, validates it in shadow mode, and explicitly deploys it.

Runs inside the app container (needs the reference DB + compliance store):
  docker compose run --rm app python tools/auto_actuate.py --limit 5
  --dry-run            evaluate + report, write nothing
  --results-dir PATH   defaults to output/results
  --scope PREFIX       restrict to documents whose id starts with PREFIX
                       (repeatable). Classes without a scoped document are
                       skipped; dossiers and replays read scoped documents
                       only — older corpora sharing the results dir stay
                       out of both the evidence and the control set.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from loguru import logger  # noqa: E402

from tools import flip_triage  # noqa: E402

import os  # noqa: E402

# The deterministic rule pack is a DECLARED release source (`validator_rules`) whose
# bytes are bound into the release fingerprint.  This module is reachable from the
# deployment (`app.release.claim_readiness` -> `tools.clinical_auditor` ->
# `tools.auto_actuate`), so it resolves the pack through the declaration like every
# other production reader rather than composing its path.  (Directive section 6.)
from app.release.source_manifest import declared_source_path  # noqa: E402

RULES_PATH = declared_source_path("validator_rules")
PROPOSALS_DIR = ROOT / "data" / "rules" / "proposals"
NOTES_DIR = Path(os.getenv("NOTES_DIR", str(ROOT / "doctors_notes")))

BUILTIN_TEMPLATES = ("context_gate", "tiered_family_arbitration",
                     "icd_tiered_axis", "companion_completion",
                     "residual_secondary_demotion",
                     "documented_service_completion",
                     "documented_diagnosis_completion")


def _self_authored() -> dict[str, dict]:
    """Every live self-authored template, graduated (trusted, in the app
    tree) and sandboxed (data/rules/auto_templates/) alike — graduated
    entries win a name collision, mirroring the engine's dispatch."""
    from app.validation.auto_templates import load_auto_templates
    from app.validation.graduated import GRADUATED
    return {**load_auto_templates(), **GRADUATED}


def all_templates() -> tuple[str, ...]:
    """The live template vocabulary: hand-written mechanics plus every
    self-authored template (sandboxed candidates and graduated alumni).
    Escalations record this; when it grows, stale 'no template fits'
    verdicts auto-reopen. Graduation keeps the name in the vocabulary,
    so promoting a template never churns escalation records."""
    return BUILTIN_TEMPLATES + tuple(
        t for t in sorted(_self_authored())
        if t not in BUILTIN_TEMPLATES)

# Literal medical codes are forbidden in every field that SELECTS anything
# (regexes, descriptor grammar, tier words, evidence stems) — a rule must
# reason from descriptor grammar and reference-data lookups so it
# generalizes and survives code-set updates. Display-only prose (messages,
# recommendations, citations) may mention codes: those strings are shown to
# humans and select nothing.
_CODE_LITERAL_RE = re.compile(
    r"\b\d{5}\b|\b[A-Z]\d{4}\b|\b[A-TV-Z]\d{2}\.\d{1,4}\b")
_PROSE_FIELDS = {
    "authority", "rationale",
    # action-block display templates (all rendered, never matched)
    "message", "message_added", "message_undocumented", "recommendation",
    "rationale_added", "review_reason", "review_reason_added",
    "review_reason_demoted",
    "evidence_found", "evidence_missing", "action_swap", "action_dup",
}


# ---------------------------------------------------------------------------
# Replay harness
# ---------------------------------------------------------------------------

class Replayer:
    """Replays stored result payloads through a fresh CodingValidator so a
    candidate rule's effect can be measured deterministically — same claim
    arrays, same note text, only the rule pack differs."""

    def __init__(self):
        from app.rag.code_reference import CodeReferenceDB
        from app.compliance.datastore.store import ComplianceDataStore
        logger.info("Replayer: loading reference DB + compliance store...")
        self.db = CodeReferenceDB()
        self.db.load_all()  # constructor builds EMPTY tables — without this
        #                     every descriptor lookup silently returns None
        self.store = ComplianceDataStore()
        self.store.build_or_load()

    def _fresh_validator(self):
        from app.validation.validator import CodingValidator
        return CodingValidator(self.db, self.store)

    @staticmethod
    def signature(icd, cpt, hcpcs) -> tuple:
        """Billing signature — the claim-form-visible content of a result.
        Two runs with equal signatures produce the same claim."""
        def norm(entries, with_type=False):
            out = []
            for e in entries or []:
                code = str(e.get("code") or "").strip().upper()
                if not code:
                    continue
                row = (code,
                       tuple(sorted(str(m) for m in (e.get("modifiers") or [])
                                    if m)),
                       str(e.get("units") or ""))
                if with_type:
                    row += (str(e.get("type") or "").strip().lower(),)
                out.append(row)
            return tuple(sorted(out))
        return (norm(icd, with_type=True), norm(cpt), norm(hcpcs))

    def replay_arrays(self, payload: dict,
                      note_text: str) -> tuple[dict, dict]:
        """One stored run payload → (post-validation claim arrays, validation
        report) under the CURRENT rule pack. Claim context (payer, DOS, DOB,
        physician-documented codes) is reconstructed from the payload's own
        metadata so the replayed validator sees what the original one saw."""
        coding_result = {
            "icd10_codes": copy.deepcopy(payload.get("icd_codes") or []),
            "cpt_codes": copy.deepcopy(payload.get("cpt_codes") or []),
            "hcpcs_codes": copy.deepcopy(payload.get("hcpcs_codes") or []),
            "snomed_codes": copy.deepcopy(payload.get("snomed_codes") or []),
            "supporting_conditions":
                copy.deepcopy(payload.get("supporting_conditions") or []),
        }
        meta = payload.get("patient_metadata") or {}
        from app.compliance.payer_registry import (PayerRegistryUnavailable,
                                                    parse_insurance_text)
        try:
            from app.compliance.engine import _parse_dos
            dos = _parse_dos(meta)
            follows_medicare = parse_insurance_text(
                str(meta.get("insurance") or "")).follows_medicare_coverage
        except PayerRegistryUnavailable:
            # NOT swallowed into `follows_medicare = False`: that is the answer for a
            # commercial payer, so an unreadable registry would silently validate every
            # note against the WRONG coverage floor (no LCD necessity, no routine-foot-care
            # class findings, no status-I check). (Codex F6-R5-A, round 6.)
            raise
        except Exception:
            dos, follows_medicare = None, False
        v = self._fresh_validator()
        report = v.validate(
            coding_result,
            note_full_text=note_text,
            physician_documented_codes=(
                payload.get("physician_documented_codes") or []),
            dos=dos,
            note_category=str(payload.get("note_category") or ""),
            patient_dob=str(meta.get("date_of_birth") or ""),
            payer_follows_medicare_coverage=follows_medicare,
            note_assessment_text=_assessment_slice(note_text),
            # The completeness invariant must re-run on the REPLAYED claim —
            # replay is what realizes the final shipped claim, and a code
            # dropped/added during reconciliation changes what is accounted
            # for. Read the documented procedures back from the stored
            # payload (persisted on the record for exactly this) so the
            # replayed validation_issues carry the completeness flag the
            # coherence gate reads; without it the check no-ops on the claim
            # that actually ships.
            procedures_performed=(
                payload.get("procedures_performed_today") or None),
        )
        return coding_result, report

    def replay(self, payload: dict, note_text: str) -> tuple:
        """One stored run payload → post-validation billing signature."""
        coding_result, _ = self.replay_arrays(payload, note_text)
        return self.signature(coding_result["icd10_codes"],
                              coding_result["cpt_codes"],
                              coding_result["hcpcs_codes"])



def _assessment_slice(note_text: str) -> str:
    m = re.search(r"assessment[^\n]*\n(.*?)(?:\n\s*(?:plan|procedure)\b|\Z)",
                  note_text or "", re.IGNORECASE | re.DOTALL)
    return m.group(1)[:4000] if m else ""


def _note_text_for(doc: str, results_dir: Path, runs: list[dict],
                   main: dict | None) -> str:
    for src in ([main] if main else []) + runs:
        t = ((src.get("rag_context") or {}).get("note_full_text")) or ""
        if t:
            return t
    pdf = NOTES_DIR / f"{doc}.pdf"
    if pdf.exists():
        try:
            import pdfplumber
            with pdfplumber.open(pdf) as p:
                return "\n".join((pg.extract_text() or "") for pg in p.pages)
        except Exception as exc:
            logger.warning(f"{doc}: pdfplumber fallback failed: {exc}")
    return ""


def _load_runs(doc: str, results_dir: Path) -> list[dict]:
    runs_dir = results_dir / "consistency_runs"
    out = []
    for i in range(1, 10):
        f = runs_dir / f"{doc}_run{i}.json"
        if not f.exists():
            break
        out.append(json.loads(f.read_text()))
    return out


def _load_main(doc: str, results_dir: Path) -> dict | None:
    f = results_dir / f"{doc}_results.json"
    if not f.exists():
        return None
    data = json.loads(f.read_text())
    return data if isinstance(data, dict) else None  # excludes all_results


# ---------------------------------------------------------------------------
# Dossier — evidence grounded in the authoritative sources
# ---------------------------------------------------------------------------

def _authoritative_evidence(rep: Replayer, array: str, codes: list[str],
                            desc_fallback: dict | None = None) -> list[dict]:
    out = []
    for code in codes:
        c = str(code).strip().upper()
        if not c or "/" in c:
            continue
        row: dict = {"code": c}
        try:
            if array == "icd_codes":
                info = rep.db.validate_icd10(c) or {}
                row["descriptor"] = info.get("description", "")
                row["tabular_category"] = \
                    rep.store.icd10_tabular_description(
                        c.replace(".", "")[:3]) or ""
                groups = rep.store.use_additional_code_groups(c)
                if groups:
                    row["use_additional_code"] = [
                        {"instruction": carrier,
                         "targets": [f"{r} {d[:60]}" for r, d in refs[:5]]}
                        for carrier, refs in groups[:3]]
            else:
                info = (rep.db.validate_cpt(c) if c.isdigit()
                        else rep.db.validate_hcpcs(c)) or {}
                row["descriptor"] = (info.get("long_description")
                                     or info.get("description", "")
                                     # the RAG subset omits many surgical
                                     # CPTs — the billed line's own
                                     # description still grounds the rule
                                     or (desc_fallback or {}).get(c, ""))
                mue = rep.store.mue(c)
                if mue:
                    row["mue"] = mue
                # MDM-leveled E/M code: attach the licensed AMA MDM table
                # row its own descriptor requires — the structured
                # authority for every leveling / documentation-sufficiency
                # judgment, citeable like any other reference row instead
                # of relying on model memory of the grid
                mdm = rep.store.mdm_requirements(c)
                if mdm:
                    row["mdm_requirements"] = mdm
        except Exception as exc:
            row["lookup_error"] = str(exc)
        out.append(row)
    return out


def _sig_view(sig: tuple) -> dict:
    """A billing signature rendered as readable claim arrays — the shape
    proposers/designers see in dossiers and rejection diffs. Signatures
    are the gate's comparison unit; this is the same content spelled out
    (code, modifiers, units, and type for diagnoses)."""
    def rows(arr, with_type=False):
        out = []
        for row in arr:
            d = {"code": row[0], "modifiers": list(row[1]),
                 "units": row[2]}
            if with_type and len(row) > 3:
                d["type"] = row[3]
            out.append(d)
        return out
    icd, cpt, hcpcs = sig
    return {"icd_codes": rows(icd, with_type=True),
            "cpt_codes": rows(cpt), "hcpcs_codes": rows(hcpcs)}


def _sig_diff(cand_sigs: list, goal: tuple) -> dict:
    """What the candidate's replays produced vs. the target they must
    land on, reduced to the ROWS that differ — the exact, minimal fact a
    designer needs to repair a rule that moved a verified claim."""
    def rows_of(sig):
        return {row for arr in sig for row in arr}

    def plain(row):  # JSON-native: nested modifier tuple becomes a list
        return [list(x) if isinstance(x, tuple) else x for x in row]

    goal_rows = rows_of(goal)
    produced = []
    for sig in dict.fromkeys(cand_sigs):  # unique, order-preserving
        rows = rows_of(sig)
        extra = sorted(rows - goal_rows)
        missing = sorted(goal_rows - rows)
        if extra or missing:
            produced.append({
                "rows_your_replay_added_or_changed":
                    [plain(r) for r in extra],
                "rows_the_target_requires_but_replay_lost":
                    [plain(r) for r in missing]})
    return {"replay_vs_target_diffs": produced,
            "target_claim": _sig_view(goal)}


def _implicated_rules(cls: dict, results_dir: Path) -> list[dict]:
    """Enabled auto-generated pack rules whose action.category matches a
    material correction recorded on the class's own codes in a target
    document — the deployed rules that ACTED on the disputed content.
    These are the amendment candidates an audit-dispute proposal may name
    (amend_rule / disable_rule); everything else in the pack is out of
    reach."""
    class_codes = {str(c).upper() for d in cls["documents"]
                   for c in ((d.get("disagreement") or {}).get("codes")
                             or [cls["code"]])}
    from tools.clinical_auditor import material_corrections_of
    categories: set[str] = set()
    for d in cls["documents"]:
        main = _load_main(d["document_id"], results_dir) or {}
        for m in material_corrections_of(main):
            if isinstance(m, dict) and \
                    str(m.get("code") or "").upper() in class_codes:
                cat = str(m.get("category") or "")
                if cat:
                    categories.add(cat)
    if not categories:
        return []
    try:
        pack = json.loads(RULES_PATH.read_text())
    except Exception:
        return []
    return [r for r in pack.get("rules", [])
            if r.get("auto_generated") and r.get("enabled", True)
            and str((r.get("action") or {}).get("category") or "")
            in categories]


def build_dossier(cls: dict, rep: Replayer, results_dir: Path,
                  max_docs: int = 3) -> dict:
    codes = sorted({c for d in cls["documents"]
                    for c in ((d.get("disagreement") or {}).get("codes")
                              or [cls["code"]])})
    # Billed-line descriptions from the per-run evidence back-fill any code
    # the curated reference subset doesn't carry a descriptor for.
    desc_fallback: dict = {}
    for d in cls["documents"]:
        for e in d.get("per_run_entry") or []:
            for x in (e if isinstance(e, list) else [e]):
                if x and x.get("code") and x.get("description"):
                    desc_fallback.setdefault(
                        str(x["code"]).upper(), x["description"])
    registry = _registry_verified_claims()
    audit_kind = cls.get("kind") == "audit_dispute"
    code_targets = _per_code_targets() if audit_kind else {}
    advisory_targets = _advisory_targets() if audit_kind else {}
    docs = []
    for d in cls["documents"]:
        doc = d["document_id"]
        runs = _load_runs(doc, results_dir)
        main = _load_main(doc, results_dir)
        entry = {
            "document_id": doc,
            "disagreement": d.get("disagreement"),
            "per_run_entry": d.get("per_run_entry"),
            "note_text": _note_text_for(doc, results_dir, runs, main)[:9000],
            "_replayable": bool(runs),
        }
        # Ground truth is a design INPUT, not just a post-hoc gate: when a
        # verified note flips again, the only acceptable resolution is the
        # one that lands every run byte-identical on this claim — give the
        # proposer/designer the target instead of letting it discover the
        # constraint through a gate rejection.
        if doc in registry:
            entry["registry_verified_claim"] = dict(
                _sig_view(registry[doc]),
                constraint="This document's claim is registry-VERIFIED "
                           "ground truth. Any rule that changes this "
                           "document's replay MUST make every run land "
                           "EXACTLY on this claim (same codes, same "
                           "modifiers, same units, same diagnosis types) "
                           "— touch nothing beyond what realigning to it "
                           "requires.")
        elif doc in code_targets:
            # Scoped ground truth from a partial adjudication: only the
            # class's own codes are verified, and only their exact rows.
            rows = {c: (None if t is None else
                        {"code": t[0], "modifiers": list(t[1]),
                         "units": t[2],
                         **({"type": t[3]} if len(t) > 3 else {})})
                    for (_a, c), t in code_targets[doc].items()
                    if c in {str(x).upper() for x in codes}}
            if rows:
                entry["adjudicated_code_targets"] = {
                    "targets": rows,
                    "constraint": "These specific codes are VERIFIED "
                                  "per-code ground truth (expert-coder "
                                  "adjudication, unanimous and authority-"
                                  "grounded, on a note that cannot verify "
                                  "whole). Every replay of this document "
                                  "must land these codes EXACTLY on these "
                                  "rows (null = the code must be absent). "
                                  "The rest of the claim is NOT verified — "
                                  "touch nothing beyond what landing these "
                                  "codes requires."}
        try:
            from tools.observables import all_observables, code_of_key
            _obs_docs = {n: e["schema_doc"]
                         for n, e in all_observables().items()}
        except Exception:
            _obs_docs, code_of_key = {}, lambda k: str(k).rsplit(
                "|", 1)[-1].upper()
        adv_goals = {k: e for k, e in (advisory_targets.get(doc)
                                       or {}).items()
                     if code_of_key(k[1]) in {str(x).upper()
                                              for x in codes}}
        if adv_goals:
            # Observable-emission ground truth: the claim is correct as
            # billed; what the adjudicator verified is whether a MEASURED
            # PHENOMENON (e.g. a scrubber advisory) may fire. Give the
            # designer the emission goal and each observable's own
            # realization documentation, instead of letting it guess at
            # a billing-shaped fix the gate can never accept.
            entry["adjudicated_observable_targets"] = {
                "targets": [{"observable": k[0], "key": k[1],
                             "verified_state":
                                 "must_fire" if e else "must_not_fire"}
                            for k, e in sorted(adv_goals.items())],
                "observable_docs": {o: _obs_docs.get(o, "")
                                    for o in {k[0] for k in adv_goals}},
                "constraint": "This dispute is about a measured system "
                              "PHENOMENON, NOT a claim line — the claim "
                              "is correct as billed and every claim line "
                              "must replay byte-identical. The "
                              "adjudicated verified state above is the "
                              "target; each observable's doc describes "
                              "the surface that realizes an emission "
                              "change (e.g. advisory_emission is "
                              "realized via v.suppress_scrub_advisory("
                              "filter_id, code, rule_id, authority, "
                              "note) — WARN-only), gated on the note "
                              "evidence your rule requires — never by "
                              "touching billing rows.",
            }
        docs.append(entry)
    # Best-evidenced documents first: the proposer reasons from note text,
    # and the replay gate can only verify docs with per-run artifacts.
    docs.sort(key=lambda x: (not x["_replayable"], not x["note_text"]))
    docs = docs[:max_docs]
    for x in docs:
        x.pop("_replayable")
    dossier = {
        "flip_class": {k: cls[k] for k in ("class_key", "kind", "array",
                                           "code")},
        "codes_involved": codes,
        "authoritative_reference_data":
            _authoritative_evidence(rep, cls["array"], codes, desc_fallback),
        "documents": docs,
    }
    if audit_kind:
        implicated = _implicated_rules(cls, results_dir)
        if implicated:
            dossier["implicated_rules"] = {
                "rules": [{k: v for k, v in r.items() if k != "provenance"}
                          for r in implicated],
                "note": "These DEPLOYED auto-generated rules acted on this "
                        "class's codes (their action.category matches a "
                        "material correction recorded on them). If the "
                        "dispute's root cause is one of these rules — its "
                        "evidence grammar too narrow, its descriptor "
                        "reading wrong for a documented fact pattern an "
                        "authority recognizes — you may amend or disable "
                        "it instead of authoring a new rule.",
            }
    return dossier


# ---------------------------------------------------------------------------
# Proposal — one declarative rule, or escalate
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a medical-coding compliance engineer for a podiatry claims pipeline.
The pipeline runs each note 3 times; a FLIP is a code that some runs billed
and others didn't (or billed with different attributes). Flips are resolved
by DECLARATIVE VALIDATOR RULES: versioned config executed by generic rule
templates, exactly like the hand-written deterministic layers.

Your job: given one flip class (the disagreement, each run's entry, the note
text, and the authoritative reference data), either author ONE rule that
resolves the flip deterministically, or escalate.

A class whose kind is "audit_dispute" is different: the runs AGREED, but
the clinical-correctness review found the claim clinically wrong — either
a deterministic layer's correction was overturned (the dossier carries the
disputed correction) or the whole-claim expert review reported a grounded
finding (the dossier carries the finding: wrong/missing code, wrong
primary designation, a coverage requirement the claim fails, or a system
advisory whose recommendation is authoritatively wrong for the fact
pattern). Either way the dossier includes the review's authority citation
and note evidence, plus the verified ground truth to land on — one of:
- "registry_verified_claim": the document's full human/adjudicated claim.
  Every replay must land byte-identical on it.
- "adjudicated_code_targets": PER-CODE verified rows from an expert-coder
  adjudication on a note that cannot verify whole (split runs, or a
  deployed rule overriding the verdict). Every replay must land exactly
  those codes on exactly those rows (null = absent) and touch nothing
  else.
- "adjudicated_observable_targets": the dispute is about a MEASURED
  PHENOMENON of the saved record (named by an observable namespace and
  key — e.g. observable "advisory_emission", key "FILTER_ID|CODE" for a
  compliance-scrubber WARN advisory), NOT a claim line — the claim is
  correct as billed and every claim line must replay byte-identical. The
  verified state says whether the phenomenon must fire ("must_fire") or
  must not ("must_not_fire") for this documented fact pattern; the
  accompanying observable_docs describe each observable's realization
  surface. For advisory_emission, a "must_not_fire" target is realized
  ONLY through a rule whose template calls
  v.suppress_scrub_advisory(filter_id, code, rule_id,
  authority, note) when the note documents the authority-recognized
  pathway — never by adding/removing/altering billed lines.
Resolve the class by generalizing the MECHANISM of the error (e.g. the
kind of note context that must not count as documentation), never by
memorizing this note or its codes.

AMENDING A DEPLOYED RULE (audit_dispute classes only): when the dossier
carries "implicated_rules", those deployed auto-generated rules acted on
the disputed codes. If the dispute's root cause is one of them — most
often an evidence grammar too narrow for a documented fact pattern the
cited authority recognizes — respond with an amendment instead of a new
rule:
  {"decision": "amend_rule", "target_rule_id": "<implicated rule id>",
   "rationale": "...", "rule": {...the full corrected replacement...}}
or, when the rule's whole premise is wrong and no correction preserves it:
  {"decision": "disable_rule", "target_rule_id": "<implicated rule id>",
   "rationale": "..."}
The replacement rule obeys every constraint a new rule does (template
schema, no code literals, authority citation). Amend/disable may only
name a rule listed in implicated_rules; prefer amending (the mechanic
stays, the grammar widens) over disabling. The old version is kept
disabled in the pack as the audit trail.

Available templates (choose exactly one):
- context_gate: suppress a line whose only note mentions occur inside
  non-billable contexts (each context a labeled regex; optionally gated on
  the claim carrying a surgery code in a range).
- tiered_family_arbitration: a CPT family whose descriptors encode an
  ordered attribute axis (depth, extent...); the billable member is the one
  the note's evidence sentences support; no evidence -> lowest tier.
- icd_tiered_axis: same mechanic over an ICD family whose FINAL character
  encodes a severity axis spelled in the descriptors' own tier words.
- companion_completion: when the claim carries a carrier-condition code and
  a trigger code, the Alphabetic Index 'with' linkage supplies the mandated
  combination code.
- residual_secondary_demotion: a non-primary residual-category diagnosis
  stays billed only when the encounter's assessment documents it; otherwise
  demote to supporting_conditions.
- documented_service_completion: presence arbitration for a service/supply
  family (CPT or HCPCS) selected by descriptor grammar. When the note's own
  sentences affirmatively document the service (lexicon stems, negation-
  scrubbed, outside exclusion contexts), the family member whose descriptor
  the documentation best matches is added if absent; with no documenting
  sentence, billed family members are suppressed. This is the template for
  supply/service codes that flap between runs while the note is unambiguous
  (dispensed DME, casting supplies, strapping, unna boots...). Its schema:
  {"id", "template", "enabled", "authority",
   "applies_to": {"array": "hcpcs_codes"|"cpt_codes"},
   "family": {"code_regex": broad-structural-only (e.g. "^A45\\\\d\\\\d$" is
              NOT allowed — too specific; "^[AQ]\\\\d{4}$" is), and/or
              "descriptor_prefix", "descriptor_requires_any",
              "descriptor_excludes" lists},
   "evidence": {"service_stems": [note words that name the service],
                "exclusions": [{"label", "regex"}...],
                "min_descriptor_tokens": 2, "scrub_negation": true},
   "action": {"severity", "category", "message_added",
              "message_undocumented", "rationale_added",
              "review_reason_added", "recommendation", "denial_risk"}}
- documented_diagnosis_completion: the ICD sibling of
  documented_service_completion — presence arbitration for a DIAGNOSIS
  family selected by descriptor grammar. When the note's negation-scrubbed
  sentences affirmatively document the condition (condition_stems, outside
  exclusion contexts) and no family member is billed, the member whose
  descriptor the documentation best matches is added (secondary by
  default; primary only when action.add_as_primary is true AND the claim
  has no primary). With no documenting sentence, billed non-primary
  members are demoted to supporting_conditions. Use this for diagnoses
  the note documents unambiguously but that flap between runs
  (metatarsalgia, enthesopathy...). Its schema:
  {"id", "template", "enabled", "authority",
   "applies_to": {"array": "icd_codes"},
   "family": {"code_regex": broad structural regex over the DOTTED form —
              a category prefix like "^M77\\\\." is acceptable when the
              descriptor grammar does the real selection; a full code like
              "^M77\\\\.41$" is NOT (that is a hardcoded code),
              and/or "descriptor_prefix", "descriptor_requires_any",
              "descriptor_excludes" lists},
   "evidence": {"condition_stems": [note words naming the condition],
                "exclusions": [{"label", "regex"}...],
                "min_descriptor_tokens": 2, "scrub_negation": true},
   "action": {"severity", "category", "message_added",
              "message_undocumented", "rationale_added",
              "review_reason_added", "review_reason_demoted",
              "recommendation", "denial_risk", "add_as_primary": false}}

HARD CONSTRAINTS:
- NEVER put a literal medical code (CPT, HCPCS, ICD) anywhere in the rule.
  Select codes via descriptor grammar (descriptor_prefix, tier words,
  category-description contains) or broad structural regexes only
  (e.g. a leading-digit family class). Violations are auto-rejected.
- The rule must generalize the MECHANISM, not memorize these notes: base it
  on what the authoritative reference data mandates (descriptors, Tabular
  conventions, MUE), quoting the authority in the "authority" field.
- REGISTRY GROUND TRUTH: when a dossier document carries a
  "registry_verified_claim", that claim is settled, human-relevant truth.
  A rule that changes such a document's replay is accepted ONLY if every
  run lands byte-identical on that exact claim. Author the rule so its
  entire effect on that document is realigning to the verified claim —
  same codes, same modifiers, same units, same diagnosis types — and
  nothing else.
- PRECISION: fix exactly the disagreement and stop. A rule arbitrating one
  attribute (a modifier, the units, the primary flag) must leave every
  OTHER attribute of the line — and every other line — untouched.
- If no template fits, if the correct behavior is genuinely ambiguous, or
  if resolving it would require judgment about THIS note only, escalate.

The example rules you receive define the exact JSON schema each template
expects (field names, action block, message placeholders). Follow them
precisely. Every rule needs: id (kebab-case), template, enabled=true,
authority, applies_to.array, and its template's fields.

Respond with JSON only:
  {"decision": "rule", "rationale": "...", "rule": {...}}
or
  {"decision": "escalate", "reason": "...",
   "missing_template": {"name": "snake_case_template_name (3-41 chars)",
                        "mechanism": "a one-paragraph spec of the GENERIC
                        deterministic mechanic a new template would need
                        to resolve this class of flips"}}

Include "missing_template" ONLY when your blocker is that no existing
template's mechanic fits but a safe deterministic mechanic clearly exists
— the system can then DESIGN that template automatically. Omit it when
the flip is judgment-shaped (genuinely ambiguous, note-specific, or
requiring clinical judgment): those must reach a human.

NEVER return decision="rule" citing a template name that is not in the
vocabulary above — such a rule is structurally invalid. If the mechanic
you need has no template yet, escalate with a missing_template hint (you
may sketch the rule you wish you could write inside the mechanism
paragraph)."""


def _system_prompt() -> str:
    """The proposer prompt with the CURRENT template vocabulary: the
    hand-written templates above plus the schema documentation of every
    self-authored template (sandboxed and graduated)."""
    auto = _self_authored()
    if not auto:
        return _SYSTEM_PROMPT
    docs = "\n".join(
        f"- {name} (self-authored template):\n  "
        + t["schema_doc"].strip().replace("\n", "\n  ")
        for name, t in sorted(auto.items()))
    head, sep, tail = _SYSTEM_PROMPT.partition("\nHARD CONSTRAINTS:")
    return f"{head}\n{docs}\n{sep.lstrip()}{tail}"


def _template_examples(pack: dict) -> str:
    seen, out = set(), []
    for r in pack.get("rules", []):
        t = r.get("template")
        if t in seen:
            continue
        seen.add(t)
        out.append(f"--- example {t} rule ---\n"
                   + json.dumps(r, indent=1)[:2600])
    return "\n".join(out)


# Rule proposals are the highest-stakes LLM call in the system — the output
# becomes a deployed production layer — so they run on the most capable
# reasoning model available rather than the batch pipeline's default.
# Override with AUTO_ACTUATE_MODEL; only meaningful under the claude
# provider (OpenAI deployments keep their configured model).
PROPOSAL_MODEL = os.getenv("AUTO_ACTUATE_MODEL", "claude-fable-5")

# Version of the proposal PROTOCOL — what an escalation verdict means.
# Bump whenever the proposer gains a capability that could overturn old
# escalations (v2: structured missing_template hints + template
# synthesis; v3: pack audit no longer false-positives on provenance,
# which had rolled back a healthy synthesized template; v4: audit skips
# disabled rules, whose rollback corpses were failing every later
# acceptance; v5: registry protection gained its directional exception —
# replays may converge exactly ONTO a verified claim, so registry-blocked
# verdicts from v4 and earlier are stale; v6: dossiers now carry each
# verified document's exact claim as a design target, replay rejections
# feed back the produced-vs-target row diff, and the design contract
# gained single-axis-mutation + reference-data-only code-class
# constraints — modifier/attribute classes that failed synthesis blind
# under v5 are worth a sighted re-attempt). Escalations record the
# protocol they were judged under; a different protocol makes the
# verdict stale, so the class reopens.
# v9: audit-dispute proposals gained amend_rule/disable_rule (a deployed
# auto-generated rule implicated in the dispute can be corrected or
# retired through the same gates, instead of every proposal being a new
# appended rule that the implicated rule then fights), and per-code
# verified targets from partial adjudications now serve as scoped
# realignment goals — classes that escalated because the note could
# never verify whole are worth a fresh attempt. v10: the structural
# gate's rule-id cap grew 60->96 chars — it was judging length, not
# structure, and rejected a healthy v9 amendment live over a 62-char id.
# v11: a rule proposal citing a NONEXISTENT template now converts into a
# structured missing-template hint and enters template synthesis instead
# of dying in the human queue (observed live: a proposal citing
# 'laterality_modifier_arbitration' escalated as 'unknown template'
# under v10, though it was a vocabulary blocker with a full mechanic
# spec attached), and the design prompt documents the compliance
# store's bilateral-surgery/laterality API (bilat_surg,
# modifier_laterality) so anatomic-modifier mechanics are designed
# against authoritative indicators, never section/prefix guesses.
# v12 adds advisory-shaped audit disputes: adjudicated_advisory_targets
# as a third ground-truth form (the claim is correct as billed; the
# verified state is whether a scrubber ADVISORY fires), realized via
# v.suppress_scrub_advisory and measured by the emission-aware replay
# gate. Proposals authored before v12 could not know the mechanic.
# v13 generalizes v12 to MEASUREMENT OBSERVABLES: the ground-truth block
# is now adjudicated_observable_targets ({observable, key, verified
# state} plus each observable's realization doc), advisory emission
# being merely the first observable — the measurement vocabulary itself
# now grows autonomously (tools/observable_synthesis.py), so proposals
# must be re-attempted whenever the dossier's shape they were authored
# against is superseded.
PROPOSAL_PROTOCOL = 13


def propose_rule(dossier: dict, pack: dict) -> dict:
    from app.core.config import LLM_PROVIDER
    from app.core.llm_client import chat_completion
    user = (f"TEMPLATE SCHEMAS (existing production rules as examples):\n"
            f"{_template_examples(pack)}\n\n"
            f"FLIP CLASS DOSSIER:\n{json.dumps(dossier, indent=1)}\n\n"
            f"Author the rule or escalate.")
    model = PROPOSAL_MODEL if LLM_PROVIDER == "claude" else None
    system = _system_prompt()
    try:
        text, usage = chat_completion(
            system_prompt=system, user_prompt=user,
            model=model, max_tokens=8192, json_mode=True, effort="high")
    except Exception as exc:
        if model is None:
            raise
        # claude-fable-5 can be regionally/temporarily unavailable (it has
        # been suspended before); a proposal is better made by the default
        # model than not made at all. chat_completion already exhausted its
        # transient retries before raising, so this is one clean fallback.
        logger.warning(f"Proposal model {model!r} failed ({exc}) — "
                       f"falling back to the pipeline default")
        model = None
        text, usage = chat_completion(
            system_prompt=system, user_prompt=user,
            max_tokens=8192, json_mode=True, effort="high")
    proposal = json.loads(text)
    proposal["_usage"] = usage
    proposal["_model"] = model or "pipeline-default"
    return proposal


# ---------------------------------------------------------------------------
# Acceptance gates
# ---------------------------------------------------------------------------

def gate_structural(rule: dict) -> str:
    if not isinstance(rule, dict):
        return "rule is not an object"
    # kebab-case, up to 96 chars: descriptive ids routinely pass 60 chars
    # (a live amendment escalated at 62 — the cap was judging length, not
    # structure) and the suffixing on collision needs headroom too
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{2,95}", str(rule.get("id", ""))):
        return f"bad rule id {rule.get('id')!r}"
    # RuleEngine keys rules by id (last wins) — a colliding candidate would
    # silently REPLACE an existing rule during the trial replay.
    existing = {r.get("id") for r in
                json.loads(RULES_PATH.read_text()).get("rules", [])}
    if rule["id"] in existing:
        return f"rule id {rule['id']!r} already exists in the pack"
    if rule.get("template") not in all_templates():
        return f"unknown template {rule.get('template')!r}"
    arr = (rule.get("applies_to") or {}).get("array", "cpt_codes")
    if arr not in ("icd_codes", "cpt_codes", "hcpcs_codes"):
        return f"bad applies_to.array {arr!r}"
    tmpl = rule["template"]
    if tmpl == "documented_service_completion" and arr == "icd_codes":
        return "documented_service_completion is CPT/HCPCS-only — use " \
               "documented_diagnosis_completion for diagnoses"
    if tmpl == "documented_diagnosis_completion" and arr != "icd_codes":
        return "documented_diagnosis_completion applies to icd_codes only"
    if not rule.get("authority"):
        return "missing authority citation"
    return ""


def gate_no_code_literals(rule: dict) -> str:
    def scan(s: str, key: str):
        # An escaped dot is still a dot: 'M77\.41' in a pattern IS the
        # dotted ICD literal — restore it before blanking the remaining
        # escape sequences (\b, \d, \s...), which sit flush against any
        # digits they wrap and would otherwise defeat \b word boundaries.
        m = _CODE_LITERAL_RE.search(
            re.sub(r"\\[A-Za-z]", " ", re.sub(r"\\+\.", ".", s)))
        if m:
            yield f"literal code {m.group()!r} in field {key!r}"

    def walk(obj, key=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                # Provenance is the acceptance's own audit trail (flip
                # class key, document ids, replay detail) — it records
                # codes BY DESIGN and selects nothing. Scanning it made
                # the first live pack audit roll back a healthy rule.
                if k == "provenance":
                    continue
                yield from scan(str(k), key)  # tier words etc. live in KEYS
                yield from walk(v, k)
        elif isinstance(obj, list):
            for v in obj:
                yield from walk(v, key)
        elif isinstance(obj, str) and key not in _PROSE_FIELDS:
            yield from scan(obj, key)
    return "; ".join(walk(rule))


def _code_rows(sig: tuple, codes: set[str]) -> tuple:
    """The signature rows belonging to the flip class's own codes — the
    unit the convergence gate judges. A document carrying SEVERAL flip
    classes never converges whole-claim from one rule; what one rule must
    achieve is agreement on ITS codes across runs."""
    return tuple(row for arr in sig for row in arr
                 if str(row[0]).upper() in codes)


def _registry_verified_claims() -> dict[str, tuple]:
    """{document_id: billing signature of its VERIFIED claim} from the
    finalized-claims registry (auto-recorded unanimous+CLEAN, or a human
    coder's record). These are the ground truth the whole system exists
    to produce — a candidate rule may not move their replays, with ONE
    directional exception judged against the signature returned here:
    landing every replay byte-identical on the verified claim itself."""
    try:
        from tools.claims_registry import load_events, current_view
        out = {}
        for doc, e in current_view(load_events()).items():
            c = e.get("claim") or {}
            out[doc] = Replayer.signature(c.get("icd_codes"),
                                          c.get("cpt_codes"),
                                          c.get("hcpcs_codes"))
        return out
    except Exception:
        return {}


def _realigns(cand_sigs: list, goal: tuple) -> bool:
    """True when every candidate replay lands exactly on the verified
    claim's signature — convergence ONTO ground truth. That is agreement
    with the registry, not alteration of it; any other change to a
    verified document's replay remains an automatic rejection."""
    return bool(cand_sigs) and all(s == goal for s in cand_sigs)


def _per_code_targets() -> dict[str, dict[tuple, tuple | None]]:
    """{doc: {(array, CODE): signature-row-or-None}} — the PER-CODE
    verified targets partial adjudications record (see claims_registry.
    verified_code_targets). Rows are converted to billing-signature row
    tuples so they compare directly against _code_rows output; None means
    the verdict is that the code must be ABSENT. These are the scoped
    realignment goals for audit-dispute classes on notes that cannot
    verify whole — a full-claim human/adjudicated record supersedes them
    (the registry accessor already enforces that)."""
    try:
        from tools.claims_registry import verified_code_targets
        raw = verified_code_targets()
    except Exception:
        return {}
    out: dict[str, dict[tuple, tuple | None]] = {}
    for doc, m in raw.items():
        for (array, code), row in m.items():
            if row is None:
                t = None
            else:
                t = (str(row.get("code") or "").strip().upper(),
                     tuple(sorted(str(x) for x in (row.get("modifiers")
                                                   or []) if x)),
                     str(row.get("units") or ""))
                if array == "icd_codes":
                    t += (str(row.get("type") or "").strip().lower(),)
            out.setdefault(doc, {})[(array, code)] = t
    return out


def _advisory_targets() -> dict[str, dict[tuple, bool]]:
    """{doc: {(OBSERVABLE, KEY): emit}} — the verified observable-emission
    targets audit-dispute adjudications record (see claims_registry.
    verified_observable_targets; advisory emission was the first
    observable and legacy advisory events are merged into this view).
    These are the realignment goals for observable-shaped audit-dispute
    classes: the claim is correct as billed (billing-signature
    realignment can never measure the fix), and the verified state is
    whether the disputed measured phenomenon FIRES."""
    try:
        from tools.claims_registry import verified_observable_targets
        return verified_observable_targets()
    except Exception:
        return {}


def _class_advisory_goals(advisory_targets: dict, target_docs: set[str],
                          class_codes: set[str]) -> dict[str, dict]:
    """{target_doc: {(observable, key): emit}} restricted to the class's
    own codes (an observable key ends with '|<CODE>' by contract) — the
    emission goals THIS class's candidate must realize."""
    from tools.observables import code_of_key
    out = {}
    for doc in target_docs:
        goals = {k: e for k, e in (advisory_targets.get(doc) or {}).items()
                 if code_of_key(k[1]) in class_codes}
        if goals:
            out[doc] = goals
    return out


def _audit_class_anchored(cls: dict) -> bool:
    """Whether ANY of this audit-dispute class's documents still carries a
    verified realignment target the gates could converge on: a whole-claim
    registry record, per-code verified rows covering the class's own codes,
    or an observable-emission goal scoped to them.

    A class opened while targets existed can LOSE them later (a registry
    wipe, a voided verdict) — its 'open' status is then stale. Proposing
    for it is structurally futile: gate_replay's convergence criterion for
    audit disputes is landing ON a verified target, so with none in the
    registry every candidate (and every synthesized template's trial rule)
    is doomed to 'does not land'. Measured live on routine_00003: a stale
    pre-wipe class consumed the whole template-synthesis budget while the
    freshly-adjudicated class sat behind it. Unanchored classes park back
    at awaiting_verification, and flip_triage's existing graduation re-opens
    them the moment a target reappears."""
    target_docs = {d["document_id"] for d in cls["documents"]}
    class_codes = {str(c).upper() for d in cls["documents"]
                   for c in ((d.get("disagreement") or {}).get("codes")
                             or [cls["code"]])}
    registry = _registry_verified_claims()
    if target_docs & set(registry):
        return True
    code_targets = _per_code_targets()
    for doc in target_docs:
        if any(c in class_codes for (_a, c) in code_targets.get(doc, {})):
            return True
    return bool(_class_advisory_goals(_advisory_targets(), target_docs,
                                      class_codes))


def _advisory_scrubber(rep: Replayer):
    """One ClaimScrubber per Replayer, built lazily — only emission-target
    classes ever pay for it."""
    scr = getattr(rep, "_advisory_scrubber", None)
    if scr is None:
        from app.compliance.agents import build_default_agents
        from app.compliance.engine import ClaimScrubber
        scr = ClaimScrubber(rep.store, agents=build_default_agents(rep.store))
        rep._advisory_scrubber = scr
    return scr


def _advisory_emission(payload: dict, arrays: dict, report: dict,
                       note: str, scrubber, keys: set[tuple]
                       ) -> dict[tuple, bool]:
    """{(observable, key): fires} for the watched keys, measured on ONE
    replayed run assembled exactly the way production assembles it
    (_rebuild_run: validation spread + scrub) — the emission state the
    gate judges is the one the pipeline would actually save. Measurement
    is delegated to the observables' own signature() functions (fail
    closed on a crashed observable — see observables.emission_of)."""
    from tools.observables import emission_of
    from tools.replay_reconcile import _rebuild_run
    out = _rebuild_run(payload, arrays, report, scrubber, note)
    by_obs: dict[str, set[str]] = {}
    for obs, key in keys:
        by_obs.setdefault(obs, set()).add(key)
    fires = emission_of(out, by_obs)
    return {k: v for k, v in fires.items() if k in keys
            or k[1] == "__error__"}


def _replay_with_advisories(rep: Replayer, scrubber, payloads: list[dict],
                            note: str, keys: set[tuple]
                            ) -> tuple[list[tuple], list[dict]]:
    """(billing signatures, observable emission maps) for a batch of
    stored runs under the CURRENTLY-POINTED rule pack — the
    emission-aware sibling of [rep.replay(p, note) for p in payloads]."""
    sigs, advs = [], []
    for p in payloads:
        arrays, report = rep.replay_arrays(p, note)
        sigs.append(Replayer.signature(arrays["icd10_codes"],
                                       arrays["cpt_codes"],
                                       arrays["hcpcs_codes"]))
        advs.append(_advisory_emission(p, arrays, report, note,
                                       scrubber, keys))
    return sigs, advs


def _code_target_rows(targets: dict[tuple, tuple | None],
                      class_codes: set[str]) -> tuple[set, set] | None:
    """(covered codes, expected rows) for the class codes a document's
    per-code targets cover — or None when they cover none of them. The
    expected rows are what _code_rows over the covered codes must equal:
    each present-target contributes its row; an absent-target contributes
    nothing (the code simply must not appear)."""
    covered = {c for (_a, c) in targets if c in class_codes}
    if not covered:
        return None
    rows = {row for (_a, c), row in targets.items()
            if c in covered and row is not None}
    return covered, rows


def _lands_on_code_targets(sigs: list, targets: dict[tuple, tuple | None],
                           class_codes: set[str]) -> bool | None:
    """Whether every replay signature lands the class's covered codes
    exactly on their per-code verified rows. None when the targets cover
    none of the class codes (no judgment possible)."""
    ct = _code_target_rows(targets, class_codes)
    if ct is None:
        return None
    covered, expected = ct
    return bool(sigs) and all(
        set(_code_rows(s, covered)) == expected for s in sigs)


def _project_code_targets(payloads: list[dict],
                          targets: dict[tuple, tuple | None],
                          class_codes: set[str],
                          rep: Replayer) -> list[dict]:
    """Stored run payloads with the per-code verified rows PRE-APPLIED —
    the replay input for a document whose verification is scoped.

    The stored runs are the generative stage's output and may not carry
    the adjudicated content at all (an 'include' verdict materializes
    from a donor at adjudication time — measured live, routine_00001:
    27654 was a missing-code finding, so no stored run bills it, and no
    pack mutation could ever conjure the line from them). The adjudicated
    verdicts were applied MECHANICALLY and are re-applied by every
    adjudication pass; what the pack must prove is that it lets them
    SURVIVE the deterministic stack. So the trial reproduces the same
    mechanical application — insert/remove/set exactly the covered
    codes' rows — and replays THAT through baseline and candidate packs:
    a baseline layer that strips a verified row is the defect, a
    candidate under which every row survives is the fix. Entry identity
    comes from the row plus the reference-DB descriptor, never
    invention."""
    covered = {(a, c): row for (a, c), row in targets.items()
               if c in class_codes}
    if not covered:
        return payloads
    out = []
    for p in payloads:
        q = json.loads(json.dumps(p, default=str))
        for (array, code), row in covered.items():
            entries = [e for e in (q.get(array) or [])
                       if not (isinstance(e, dict) and
                               str(e.get("code") or "").upper() == code)]
            if row is not None:
                entry: dict = {"code": row[0],
                               "modifiers": list(row[1])}
                if row[2]:
                    # signature rows stringify units; claim entries carry
                    # them numerically (validator layers compare against
                    # MUE limits) — restore the numeric form
                    entry["units"] = (int(row[2])
                                      if str(row[2]).isdigit() else row[2])
                if array == "icd_codes" and len(row) > 3 and row[3]:
                    entry["type"] = row[3]
                try:
                    rows = _authoritative_evidence(rep, array, [row[0]])
                    desc = (rows[0].get("descriptor") or "") if rows else ""
                except Exception:
                    desc = ""
                if desc:
                    entry["description"] = desc
                entries.append(entry)
            q[array] = entries
        out.append(q)
    return out


def gate_replay(rule: dict | None, cls: dict, queue: list[dict],
                rep: Replayer, results_dir: Path,
                scope: tuple[str, ...] = (),
                baseline_cache: dict | None = None,
                disable_rule_id: str = "") -> tuple[str, dict]:
    """Convergence of the class's own codes + whole-claim no-harm on flip
    documents; whole-claim inertness everywhere else — and inertness on
    every registry-verified claim, which outranks all other categories,
    with one directional exception: a candidate whose replays land
    byte-identical on the verified claim itself is agreeing with ground
    truth, not moving it. Both sides are REPLAYED (baseline pack vs
    baseline+candidate) so the comparison isolates exactly the
    candidate's effect.

    The candidate is normally one NEW rule (appended for the trial). For
    audit-dispute amendments it can also be a pack MUTATION: disable an
    existing auto-generated rule (`disable_rule_id`, with rule=None) or
    replace it (`disable_rule_id` + the amended rule). Same gate either
    way — the mutation must realign its targets and move nothing else."""
    import app.validation.rule_engine as re_mod

    target_docs = {d["document_id"] for d in cls["documents"]}
    other_flip_docs = {d["document_id"] for c in queue
                       for d in c["documents"] if c is not cls}
    registry_docs = _registry_verified_claims()
    # audit_dispute classes carry no run-to-run disagreement (the wrong
    # claim was unanimous); their convergence criterion is REALIGNMENT —
    # every replay of a target document must land byte-identical on its
    # verified registry claim, or (scoped verification from a partial
    # adjudication) land the class's own codes exactly on their per-code
    # verified rows.
    audit_kind = cls.get("kind") == "audit_dispute"
    code_targets = _per_code_targets() if audit_kind else {}
    class_codes = {str(c).upper() for d in cls["documents"]
                   for c in ((d.get("disagreement") or {}).get("codes")
                             or [cls["code"]])}
    # Advisory-shaped audit disputes: the claim is correct as billed, and
    # the verified target is the disputed advisory's EMISSION STATE. The
    # convergence criterion becomes: byte-identical claim lines, with the
    # advisory's emission matching the adjudicated verdict on target docs
    # and unchanged everywhere else — measured by replaying through the
    # full production assembly (validator + scrubber), because a billing
    # signature is structurally blind to advisories.
    adv_goals: dict[str, dict] = {}
    adv_keys: set[tuple] = set()
    scrubber = None
    if audit_kind:
        adv_goals = _class_advisory_goals(_advisory_targets(),
                                          target_docs, class_codes)
        adv_keys = {k for goals in adv_goals.values() for k in goals}
        if adv_goals:
            scrubber = _advisory_scrubber(rep)

    # Every replayable doc: flip docs get per-run convergence measured;
    # all other docs (incl. unanimous ones) get single-replay inertness.
    jobs: dict[str, list[dict]] = {}
    for f in sorted(results_dir.glob("*_results.json")):
        if f.name == "all_results.json":
            continue
        doc = f.stem.removesuffix("_results")
        if not _in_scope(doc, scope):
            continue
        runs = _load_runs(doc, results_dir)
        main = _load_main(doc, results_dir)
        payloads = runs or ([main] if main else [])
        if payloads:
            jobs[doc] = payloads

    candidate_pack = json.loads(RULES_PATH.read_text())
    if disable_rule_id:
        for r in candidate_pack["rules"]:
            if r.get("id") == disable_rule_id:
                r["enabled"] = False
    # The generic dispatcher executes only auto_generated+enabled rules —
    # the candidate must carry the flags DURING the trial, not just after
    # acceptance, or the replay measures a pack in which it never runs.
    if rule is not None:
        candidate_pack["rules"].append(
            dict(rule, auto_generated=True, enabled=True))
    tmp = RULES_PATH.parent / "_candidate_pack.json"
    tmp.write_text(json.dumps(candidate_pack))

    detail: dict = {"documents": {}}
    verdict = ""
    converged_a_target = False
    try:
        for doc, payloads in jobs.items():
            note = _note_text_for(doc, results_dir, payloads,
                                  _load_main(doc, results_dir))
            if not note:
                # No replay material — convergence can't be judged here;
                # the converged_a_target requirement below still demands
                # proof on at least one replayable target document.
                detail["documents"][doc] = {"skipped": "no note text"}
                continue

            # A target doc whose verification is scoped (per-code rows
            # from a partial adjudication) replays with those rows
            # PRE-APPLIED to the stored runs: the stored runs may not
            # carry the adjudicated content at all (an include verdict
            # materializes from a donor), and what the trial must prove
            # is that the candidate pack lets the verified rows SURVIVE
            # the deterministic stack where the baseline strips them.
            projected = None
            if audit_kind and doc in target_docs \
                    and doc not in registry_docs and doc in code_targets:
                pj = _project_code_targets(payloads, code_targets[doc],
                                           class_codes, rep)
                if pj is not payloads:
                    projected = pj

            src = projected or payloads
            base_adv = cand_adv = None
            re_mod.RULES_FILE = RULES_PATH
            re_mod.load_rule_pack.cache_clear()
            if adv_keys:
                # Emission-aware replay on EVERY document: target docs to
                # judge realignment, all others to prove the disputed
                # advisory's emission is inert where nobody adjudicated it.
                base, base_adv = _replay_with_advisories(
                    rep, scrubber, src, note, adv_keys)
            elif projected is not None:
                base = [rep.replay(p, note) for p in src]
            else:
                base = _baseline_sigs(doc, payloads, note, rep,
                                      baseline_cache)

            re_mod.RULES_FILE = tmp
            re_mod.load_rule_pack.cache_clear()
            if adv_keys:
                cand, cand_adv = _replay_with_advisories(
                    rep, scrubber, src, note, adv_keys)
            else:
                cand = [rep.replay(p, note) for p in src]

            n0, n1 = len(set(base)), len(set(cand))
            # Convergence on the class's own codes: did the rule make the
            # runs agree about THIS flip, regardless of other flips on the
            # same document?
            k0 = len({_code_rows(s, class_codes) for s in base})
            k1 = len({_code_rows(s, class_codes) for s in cand})
            detail["documents"][doc] = {
                "distinct_signatures_before": n0,
                "distinct_signatures_after": n1,
                "class_code_variants_before": k0,
                "class_code_variants_after": k1,
                "changed": base != cand,
            }
            if projected is not None:
                detail["documents"][doc][
                    "per_code_targets_pre_applied"] = True
            # Registry protection outranks everything: a verified claim
            # (especially a human coder's) is settled ground truth — a rule
            # that moves it fails, even if it would converge its own flip.
            # Directional exception: replays that all land EXACTLY on the
            # verified signature are converging onto that ground truth —
            # required when a verified note flips again (it sits in both
            # camps: frozen claim AND live flip document), where any
            # effective rule necessarily changes its replay.
            realigned = False
            if doc in registry_docs and base != cand:
                if _realigns(cand, registry_docs[doc]):
                    realigned = True
                    detail["documents"][doc]["registry_realigned"] = True
                else:
                    verdict = f"{doc}: alters a registry-VERIFIED claim " \
                              f"without landing on it — a deployed rule " \
                              f"may never move settled ground truth " \
                              f"(only converge onto it exactly)"
                    # The repair loop needs the exact miss, not just the
                    # verdict: which rows the replay produced that the
                    # verified claim doesn't carry, and vice versa.
                    detail["violation"] = dict(
                        _sig_diff(cand, registry_docs[doc]),
                        document_id=doc, kind="registry")
                    break
            if doc in target_docs:
                if n1 > n0 or k1 > k0:
                    verdict = f"{doc}: candidate WORSENS convergence " \
                              f"({n0}->{n1} claim signatures, " \
                              f"{k0}->{k1} class-code variants)"
                    break
                if audit_kind:
                    if realigned:
                        converged_a_target = True
                    elif doc not in registry_docs and doc in code_targets:
                        landed = _lands_on_code_targets(
                            cand, code_targets[doc], class_codes)
                        if landed:
                            detail["documents"][doc][
                                "code_targets_realigned"] = True
                            converged_a_target = True
                        elif landed is False:
                            ct = _code_target_rows(code_targets[doc],
                                                   class_codes)
                            covered, expected = ct
                            detail["documents"][doc]["code_target_miss"] = {
                                "covered_codes": sorted(covered),
                                "target_rows": [
                                    [list(x) if isinstance(x, tuple)
                                     else x for x in r]
                                    for r in sorted(expected)],
                                "produced_rows": [
                                    [list(x) if isinstance(x, tuple)
                                     else x for x in r]
                                    for s in dict.fromkeys(cand)
                                    for r in _code_rows(s, covered)],
                            }
                    if doc in adv_goals and cand_adv is not None:
                        # Advisory-emission realignment: every replayed run's
                        # emission of each adjudicated advisory must match the
                        # verdict, AND the claim lines must stay byte-identical
                        # to baseline — an advisory fix that also moves billing
                        # rows is doing something nobody adjudicated (a billing
                        # move is judged only by the sig-level targets above).
                        goals = adv_goals[doc]
                        goal_obs = {k[0] for k in goals}
                        # a crashed observable measures every key False —
                        # which would silently satisfy a 'must not fire'
                        # goal. Its __error__ marker vetoes the hit
                        # (fail closed, never fail silent).
                        hit = all(a.get(k) == emit for a in cand_adv
                                  for k, emit in goals.items()) \
                            and not any(a.get((o, "__error__"))
                                        for a in cand_adv
                                        for o in goal_obs)
                        base_hit = all(a.get(k) == emit
                                       for a in (base_adv or [])
                                       for k, emit in goals.items()) \
                            and not any(a.get((o, "__error__"))
                                        for a in (base_adv or [])
                                        for o in goal_obs)
                        # convergence must be the CANDIDATE's doing: a
                        # baseline that already satisfies the goals is
                        # baseline_resolves' case, and crediting it here
                        # would accept a rule that did nothing.
                        if hit and not base_hit and base == cand:
                            detail["documents"][doc][
                                "advisory_emission_realigned"] = True
                            converged_a_target = True
                        elif not hit:
                            detail["documents"][doc][
                                "advisory_emission_miss"] = {
                                "targets": {f"{k[0]}|{k[1]}":
                                            ("emit" if emit else "suppress")
                                            for k, emit in goals.items()},
                                "candidate_emission_per_run": [
                                    {f"{k[0]}|{k[1]}": a.get(k)
                                     for k in goals} for a in cand_adv],
                            }
                elif k1 < k0 or (k0 <= 1 and n1 < n0):
                    converged_a_target = True
            elif doc in other_flip_docs:
                if n1 > n0:
                    verdict = f"{doc}: harms another flip class's document " \
                              f"({n0} -> {n1} signatures)"
                    break
            else:
                if base != cand and not realigned:
                    verdict = f"{doc}: not inert on an already-unanimous " \
                              f"document"
                    # Same principle as the registry diff: the baseline
                    # replay IS the target an inert rule must preserve.
                    detail["violation"] = dict(
                        _sig_diff(cand, base[0]),
                        document_id=doc, kind="inertness")
                    break
            # Advisory inertness is claim-invisible, so the signature
            # checks above cannot see it: on any document WITHOUT a
            # verified emission target for this class, the watched
            # advisories must fire exactly as they did at baseline.
            if cand_adv is not None and doc not in adv_goals \
                    and cand_adv != base_adv:
                verdict = f"{doc}: changes a disputed advisory's emission " \
                          f"on a document with no verified emission " \
                          f"target — advisory inertness violated"
                detail["violation"] = {
                    "document_id": doc, "kind": "advisory_inertness",
                    "baseline_emission_per_run": [
                        {f"{k[0]}|{k[1]}": a.get(k) for k in adv_keys}
                        for a in (base_adv or [])],
                    "candidate_emission_per_run": [
                        {f"{k[0]}|{k[1]}": a.get(k) for k in adv_keys}
                        for a in cand_adv],
                }
                break
        if not verdict and not converged_a_target:
            verdict = (("candidate does not land any disputed document's "
                        "replay on its verified target (registry claim, "
                        "per-code adjudicated rows, or adjudicated "
                        "advisory-emission state) — the clinically "
                        "wrong (unanimous) outcome survives it")
                       if audit_kind else
                       ("rule is inert on its own flip class — the runs' "
                        "disagreement about these codes survives the rule"))
    finally:
        re_mod.RULES_FILE = RULES_PATH
        re_mod.load_rule_pack.cache_clear()
        tmp.unlink(missing_ok=True)
    return verdict, detail


# ---------------------------------------------------------------------------
# Template synthesis — when no existing template's mechanic fits, design one
# ---------------------------------------------------------------------------
#
# The proposer escalating with a structured missing_template hint means the
# blocker is VOCABULARY, not judgment: a safe deterministic mechanic exists,
# the engine just has no template for it. This phase asks the reasoning
# model to author that template as a sandboxed Python module PLUS the first
# rule that uses it, then verifies the pair through every gate the system
# has: static AST safety, no-hardcoded-codes (source and rule), structural,
# and the full replay gates (convergence on the flip class's own codes,
# no-harm, inertness on unanimous notes, registry protection). A template
# that fails a gate gets one repair attempt with the exact failure fed
# back; still failing, the file is removed and the class stays escalated.
# Accepted templates persist in data/rules/auto_templates/, join the
# proposer's vocabulary immediately, and auto-reopen every stale
# "no template fits" escalation on the next scan.

_DESIGN_SYSTEM_PROMPT = """\
You are a medical-coding compliance engineer designing a NEW rule-engine
TEMPLATE for a podiatry claims pipeline. Templates are generic
deterministic mechanics (Python); rules are versioned config a template
interprets. The existing vocabulary could not resolve a class of run-to-run
billing flips, and an analysis identified the missing mechanic. Author it.

Deliver a single Python module with EXACTLY these top-level definitions:

  TEMPLATE_NAME = "snake_case_name"       # what rules cite as "template";
      # snake_case, 3-41 characters — longer names fail the static gate
  SCHEMA_DOC = \"\"\"...\"\"\"            # rule-JSON schema documentation:
      # field names, semantics, and constraints, written for a future rule
      # author. It will be appended verbatim to the rule proposer's prompt.
  def execute(engine, rule, icd, cpt, hcpcs, coding_result,
              note_full_text, note_assessment_text):
      # the mechanic. icd/cpt/hcpcs are the claim's live entry lists
      # (dicts with code, description, modifiers, units, type,
      # rationale...); mutate them in place. rule is this rule's config
      # dict. Return value is ignored.

SANDBOX — the module executes under a strict static gate; violations are
auto-rejected:
- only `import re` is allowed (no other imports)
- available builtins: abs all any bool dict divmod enumerate filter float
  format frozenset int isinstance issubclass len list map max min next
  range repr reversed round set sorted str sum tuple zip, plus common
  exception types. NOTHING else (no open/eval/exec/getattr/type/print).
- forbidden constructs: while loops, with blocks, class definitions,
  async, global/nonlocal, any dunder identifier or attribute
- NEVER a literal medical code (CPT/HCPCS/ICD) anywhere in the source —
  the template must be fully generic; codes are selected by the RULE's
  descriptor grammar and reference-data lookups.

ENGINE API available through the `engine` parameter:
  v = engine.v                       # the CodingValidator
  v._add(severity, code, category, message, recommendation,
         denial_risk="LOW"|"MEDIUM"|"HIGH")     # report an issue
  v._non_billable_codes_to_suppress.add(code)   # remove a billed
      # CPT/HCPCS line (the validator strips it after rules run).
      # ICD lines are never deleted: demote by moving the entry from
      # icd to coding_result["supporting_conditions"] instead.
  v._tokens(text) -> set[str]        # lowercase word set
  v._stem(token) -> str              # light suffix stemmer
  v._DESC_STOPWORDS                  # frozenset of descriptor stopwords
  v._note_evidence(text) -> (word_set_with_stems,
                             negation_scrubbed_lowercase_text)
  v.db.icd10 / v.db.cpt / v.db.hcpcs # {code: info} descriptor tables
      # (ICD keys are UNdotted; info["description"] or
      # info["long_description"])
  v.db.validate_icd10(dotted) / v.db.validate_cpt(c) /
  v.db.validate_hcpcs(c) -> info dict or None
  v.store.icd10_tabular_description(category3) -> str
  v.store.use_additional_code_groups(code) / v.store.code_also_groups(code)
  v.store.code_first_etiology_refs(code)   # ICD Tabular conventions
  v.store.mue(code) -> int|None      # Medicare units-of-service limit
  v.store.ncci_data_available(dos) -> bool
  v.store.ncci_pair(c1, c2, dos) -> {"col1","col2","modifier_indicator"}|None
      # the NCCI PTP edit between two claim lines, if one exists — THE
      # authority on whether two procedures bundle (modifier_indicator
      # "1" = a distinct-service modifier may bypass; "0" = never)
  v.store.bilat_surg(code) -> "0"|"1"|"2"|"3"|"9"|None
      # CMS bilateral-surgery indicator (Physician Fee Schedule) — THE
      # code-specific authority on whether a laterality modifier
      # (RT/LT/50) is expected on a procedure ("1" = unilateral
      # procedure on a sided structure). Never guess laterality
      # applicability from a code's section or descriptor.
  v.store.modifier_laterality(mod) -> "RT"|"LT"|None
      # whether a modifier's own AMA/CMS name denotes a body side —
      # covers RT/LT themselves AND site-specific digit modifiers
      # (e.g. toe modifiers), so "already sided" is a data lookup,
      # never a modifier list
  v.suppress_scrub_advisory(filter_id, code, rule_id=..., authority=...,
                            note=...)
      # record that a compliance-scrubber ADVISORY (the WARN finding the
      # named filter emits about the named code) must not fire on this
      # claim — the ONLY correct action for an adjudicated_advisory_
      # targets dispute with verified_state "must_not_fire". WARN-only
      # by contract (a FAIL can never be suppressed); the scrubber
      # records the suppression as its own PASS finding carrying rule_id
      # and authority. NEVER also touch billed lines for these disputes:
      # the claim is verified correct as billed.
  engine.rules                       # {rule_id: rule config} (whole pack)

DESIGN REQUIREMENTS:
- deterministic: same inputs, same outputs; no randomness, no ordering
  dependence on dict iteration where it changes the outcome (sort).
- conservative: when evidence is ambiguous (ties, missing lookups), DO
  NOTHING — a template must never guess. Any entry it adds must carry
  needs_review=True and a review_reason.
- generic: the mechanic must generalize beyond the triggering notes.
  Selection is by descriptor grammar / Tabular conventions from the rule
  config, never by code identity.
- defensive: entries may lack any key — use .get() with defaults.
- SINGLE-AXIS MUTATION: a template that arbitrates one attribute (a
  modifier class, units, the primary/secondary flag) must mutate ONLY
  that attribute on ONLY the lines its rule selects. It must never
  rewrite a line's other modifiers, touch other lines, or normalize
  anything as a side effect — the replay gates verify this byte-for-byte
  against every unanimous and registry-verified document.
- CODE-CLASS FACTS FROM REFERENCE DATA ONLY: when the mechanic needs to
  know what KIND of code a line is (an evaluation-and-management visit,
  a drug/supply, an add-on, a paired-organ procedure...), derive that
  from the engine's reference data — the code's own descriptor grammar
  via v.db lookups, or the compliance store's conventions — NEVER from
  literal code ranges or prefixes baked into the source (that is a
  hardcoded medical code and auto-rejects). Line RELATIONSHIPS follow
  the same rule: e.g. whether two claim lines bundle is a compliance-
  store fact, not a pattern to encode.
- REGISTRY GROUND TRUTH AS TARGET: dossier documents carrying a
  "registry_verified_claim" show the exact claim every replay of that
  document must land on if your mechanic touches it at all. Design the
  mechanic so its full effect on such documents is realignment to that
  claim — byte-identical: same codes, same modifiers, same units, same
  diagnosis types.

Also author the FIRST RULE (JSON) for the triggering flip class, following
your own SCHEMA_DOC. Rule constraints: id kebab-case; template equals
TEMPLATE_NAME; enabled true; an "authority" citation of the governing
source (descriptor text, ICD-10-CM conventions, NCCI policy...); an
applies_to.array of icd_codes|cpt_codes|hcpcs_codes; NO literal medical
codes in any selecting field (broad structural regexes and descriptor
grammar only — prose fields like authority/message/recommendation may
mention codes).

Respond with JSON only:
  {"decision": "template", "rationale": "...",
   "template_code": "the complete Python module source",
   "rule": {...the first rule...}}
or, if no safe generic deterministic mechanic exists after all:
  {"decision": "decline", "reason": "..."}"""

DESIGN_ATTEMPTS = int(os.getenv("AUTO_TEMPLATE_ATTEMPTS", "2"))
TEMPLATE_LIMIT = int(os.getenv("AUTO_TEMPLATE_LIMIT", "2"))


def _unknown_template_hint(rule: dict, rationale: str = "") -> dict | None:
    """Recover a structured missing-template hint from a rule proposal
    whose structural defect is citing a template the engine doesn't have.

    The proposer names a missing mechanic two ways: the documented way
    (decision=escalate with a missing_template hint) and this way —
    authoring the rule it WISHES it could write, against a template name
    it invented. Both mean the same thing: the blocker is VOCABULARY,
    not judgment. Before this helper, only the first form reached
    template synthesis; the second died in the human queue as a plain
    'unknown template' escalation (observed live: a rule citing
    'laterality_modifier_arbitration', protocol 10) — even though an
    attempted rule is the best-specified synthesis candidate there is,
    a concrete spec of the fields the mechanic must interpret.

    Returns None when the cited name could never be a TEMPLATE_NAME
    (the module static gate would reject it) or when it actually exists
    (then 'unknown template' was not the failure and there is nothing
    to synthesize)."""
    name = str(rule.get("template") or "")
    if name in all_templates() or \
            not re.fullmatch(r"[a-z][a-z0-9_]{2,40}", name):
        return None
    return {
        "name": name,
        "mechanism": ("The rule proposer authored a rule citing this "
                      "template before it existed. Design the generic "
                      "deterministic mechanic that rule implies — the "
                      "attempted rule (attached) is a SPEC of the "
                      "mechanic, not a contract: author your own "
                      "SCHEMA_DOC and first rule. Proposer's rationale: "
                      + str(rationale or "(none)")).strip(),
        "attempted_rule": rule,
    }


def _repair_feedback(reason: str, detail: dict) -> str:
    """The exact repair brief a rejected design gets back. A bare verdict
    ('not inert', 'alters a verified claim', 'does not land') forces the
    designer to guess what its rule actually did; the replay gate's
    structured violation — the rows/emissions the replay produced vs. the
    target's — makes the second attempt a correction instead of a blind
    retry."""
    v = (detail or {}).get("violation")
    if v:
        return (f"{reason}\n\nEXACT VIOLATION (from the replay gate) — your "
                f"rule's replay of document {v.get('document_id')} diverged "
                f"from the {'registry-VERIFIED claim' if v.get('kind') == 'registry' else 'baseline claim it must leave untouched'}. "
                f"Row format: [code, [modifiers...], units(, type)]:\n"
                f"{json.dumps({k: v[k] for k in ('replay_vs_target_diffs', 'target_claim') if k in v}, indent=1)[:4000]}\n"
                f"Repair so the replay touches ONLY what the flip class "
                f"requires and nothing else on this document.")
    # 'does not land' rejections carry per-document miss diagnostics
    # instead of a violation block: which verified rows/emission states
    # the trial replay was supposed to land on, and what it produced.
    misses = {doc: {k: d[k] for k in ("advisory_emission_miss",
                                      "code_target_miss") if k in d}
              for doc, d in ((detail or {}).get("documents") or {}).items()
              if isinstance(d, dict)
              and ("advisory_emission_miss" in d or "code_target_miss" in d)}
    if not misses:
        return reason
    return (f"{reason}\n\nEXACT MISS (from the replay gate), per target "
            f"document — 'targets' is the verified state your rule's "
            f"replay must produce; the paired entries are what it "
            f"actually produced:\n"
            f"{json.dumps(misses, indent=1, default=str)[:4000]}\n"
            f"Repair so the replay lands the targets exactly (an "
            f"advisory 'suppress' target is realized ONLY via "
            f"v.suppress_scrub_advisory) while every claim line stays "
            f"byte-identical to baseline.")


def design_template(hint: dict, dossiers: list[dict], pack: dict,
                    feedback: str = "") -> dict:
    from app.core.config import LLM_PROVIDER
    from app.core.llm_client import chat_completion
    user = (f"MISSING MECHANIC (identified during rule proposal):\n"
            f"{json.dumps(hint, indent=1)}\n\n"
            f"FLIP CLASS DOSSIER(S) the template must resolve:\n"
            f"{json.dumps(dossiers, indent=1)[:60000]}\n\n"
            f"EXISTING RULES (style/JSON conventions reference):\n"
            f"{_template_examples(pack)[:8000]}\n"
            + (f"\nYOUR PREVIOUS ATTEMPT WAS REJECTED — fix exactly this "
               f"and resubmit:\n{feedback}\n" if feedback else "")
            + "\nAuthor the template module and its first rule, or "
              "decline.")
    model = PROPOSAL_MODEL if LLM_PROVIDER == "claude" else None
    try:
        text, usage = chat_completion(
            system_prompt=_DESIGN_SYSTEM_PROMPT, user_prompt=user,
            model=model, max_tokens=16384, json_mode=True, effort="high")
    except Exception as exc:
        if model is None:
            raise
        logger.warning(f"Design model {model!r} failed ({exc}) — "
                       f"falling back to the pipeline default")
        model = None
        text, usage = chat_completion(
            system_prompt=_DESIGN_SYSTEM_PROMPT, user_prompt=user,
            max_tokens=16384, json_mode=True, effort="high")
    design = json.loads(text)
    design["_usage"] = usage
    design["_model"] = model or "pipeline-default"
    return design


def _gate_template_pair(code: str, rule: dict, cls: dict, queue: list[dict],
                        rep: Replayer, results_dir: Path,
                        scope: tuple[str, ...],
                        baseline_cache: dict) -> tuple[str, dict, str | None]:
    """All gates for a (template module, first rule) pair.

    Candidate source is executable only inside an isolated temporary loader
    directory for the duration of replay. It is never placed in the live
    auto-template directory, even briefly.
    """
    import app.validation.auto_templates as auto_templates
    from app.validation.auto_templates import (
        load_auto_templates, template_name_of,
        validate_template_clause_tagging, validate_template_source)

    problems = validate_template_source(code)
    if problems:
        return ("template source rejected: " + "; ".join(problems[:6]),
                {}, None)
    # Admission-time only (see validate_template_clause_tagging): newly
    # synthesized source may not add untagged emission sites to the
    # surface tests/check_clause_coverage.py is draining. Already-
    # installed templates are untouched — this gate never runs at load.
    clause_problems = validate_template_clause_tagging(code)
    if clause_problems:
        return ("template source rejected (clause tagging): "
                + "; ".join(clause_problems[:6]), {}, None)
    name = template_name_of(code)
    if name in BUILTIN_TEMPLATES:
        return (f"TEMPLATE_NAME {name!r} collides with a built-in "
                f"template", {}, None)
    if name in _self_authored():
        return (f"TEMPLATE_NAME {name!r} already installed (or "
                f"graduated)", {}, None)
    if rule.get("template") != name:
        return (f"rule.template {rule.get('template')!r} != "
                f"TEMPLATE_NAME {name!r}", {}, None)

    live_dir = auto_templates.AUTO_TEMPLATES_DIR
    with tempfile.TemporaryDirectory(prefix="rule-proposal-") as tmp:
        sandbox_dir = Path(tmp)
        path = sandbox_dir / f"{name}.py"
        path.write_text(code, encoding="utf-8")
        auto_templates.AUTO_TEMPLATES_DIR = sandbox_dir
        try:
            if name not in load_auto_templates():
                return ("template failed to load after passing the static gate "
                        "(missing exports at execution)", {}, None)
            reason = gate_structural(rule) or gate_no_code_literals(rule)
            detail: dict = {}
            if not reason:
                reason, detail = gate_replay(
                    rule, cls, queue, rep, results_dir, scope,
                    baseline_cache=baseline_cache)
        except Exception as exc:
            reason, detail = f"gating raised {exc!r}", {}
        finally:
            auto_templates.AUTO_TEMPLATES_DIR = live_dir
    if reason:
        return reason, detail, None
    return "", detail, code


def _clamp_template_name(name: str) -> str:
    """The hinted mechanic name, normalized to something the module static
    gate can accept (snake_case, 3-41 chars). Hints come from the rule
    proposer's free text — measured live on routine_00003, a 45-char hint
    name was adopted verbatim by the designer and burned a whole design
    attempt on the length check. Over-long names lose whole trailing
    _segments (never mid-word truncation); a hint that can't be normalized
    falls back to a generic stem the designer is free to improve on."""
    s = re.sub(r"[^a-z0-9_]+", "_", str(name or "").strip().lower())
    s = re.sub(r"_+", "_", s).strip("_")
    while len(s) > 41 and "_" in s:
        s = s.rsplit("_", 1)[0]
    s = s[:41].rstrip("_")
    if not re.fullmatch(r"[a-z][a-z0-9_]{2,40}", s):
        return "synthesized_mechanic"
    return s


def synthesize_templates(candidates: list[tuple[dict, dict, dict]],
                         queue: list[dict], rep: Replayer,
                         results_dir: Path, scope: tuple[str, ...],
                         baseline_cache: dict, dry_run: bool,
                         summary: dict) -> int:
    """candidates: (cls, dossier, missing_template_hint) for every class
    that escalated with a structured hint this pass. Groups them by the
    hinted mechanic, designs at most TEMPLATE_LIMIT new templates, gates
    each with its first rule, and deploys the survivors. Returns the
    number of templates installed. Sibling classes sharing an accepted
    mechanic are NOT re-proposed here — their recorded template
    vocabulary is now stale, so the next scan reopens them for a normal
    rule proposal against the new template."""
    groups: dict[str, list[tuple[dict, dict, dict]]] = {}
    for cls, dossier, hint in candidates:
        # Normalize the hinted name to a gate-acceptable TEMPLATE_NAME
        # before it reaches the designer (who adopts hint names verbatim).
        hint = dict(hint, name=_clamp_template_name(hint.get("name")))
        groups.setdefault(hint["name"], []).append((cls, dossier, hint))
    installed = 0
    pack = json.loads(RULES_PATH.read_text())
    ordered = sorted(groups.values(), key=len, reverse=True)
    for group in ordered[:TEMPLATE_LIMIT]:
        # Best-evidenced class drives design and gating; the rest ride as
        # supporting dossiers and reopen automatically once the vocabulary
        # grows.
        group.sort(key=lambda t: -len(t[0]["documents"]))
        cls, _dossier, hint = group[0]
        dossiers = [d for _c, d, _h in group[:3]]
        key = cls["class_key"]
        logger.info(f"=== Synthesizing template for {hint.get('name')!r} "
                    f"(driver class {key}, {len(group)} class(es)) ===")
        feedback = ""
        for attempt in range(1, DESIGN_ATTEMPTS + 1):
            try:
                design = design_template(hint, dossiers, pack,
                                         feedback=feedback)
            except Exception as exc:
                logger.warning(f"  design attempt {attempt} failed: {exc}")
                feedback = f"the design call itself failed: {exc}"
                continue
            if design.get("decision") != "template":
                logger.info(f"  designer declined: "
                            f"{design.get('reason', '')[:300]}")
                break
            code = str(design.get("template_code") or "")
            rule = design.get("rule") or {}
            reason, detail, template_source = _gate_template_pair(
                code, rule, cls, queue, rep, results_dir, scope,
                baseline_cache)
            if reason:
                logger.info(f"  attempt {attempt} rejected: {reason[:400]}")
                feedback = _repair_feedback(reason, detail)
                continue
            if dry_run:
                logger.info(f"  DRY RUN: template {rule['template']!r} + "
                            f"rule {rule['id']!r} would be proposed")
                installed += 1
                break
            # Persist source and replay proof in an inert proposal.
            accept_rule(rule, cls, design.get("rationale", ""),
                        dict(detail,
                             proposal_model=design.get("_model"),
                             synthesized_template=rule["template"],
                             template_source=template_source))
            installed += 1
            summary["templates_created"] = \
                summary.get("templates_created", 0) + 1
            summary["proposed"] += 1
            summary["escalated"] = max(0, summary["escalated"] - 1)
            for c in summary["classes"]:
                if c.get("class_key") == key:
                    c.update(status="proposed", rule_id=rule["id"],
                             synthesized_template=rule["template"])
            flip_triage.set_status(key, "proposed", {
                "rule_id": rule["id"],
                "synthesized_template": rule["template"],
                "replay": detail})
            pack = json.loads(RULES_PATH.read_text())
            logger.info(f"  -> TEMPLATE PROPOSED: {rule['template']} "
                        f"(rule {rule['id']}) after {attempt} attempt(s)")
            break
    return installed


# ---------------------------------------------------------------------------
# Actuation
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _baseline_sigs(doc: str, payloads: list[dict], note: str,
                   rep: Replayer, cache: dict | None) -> list[tuple]:
    """Baseline-pack replay signatures for one document, memoized across
    flip classes: the baseline pack is invariant between rule acceptances,
    so replaying it once per document per acceptance-epoch is enough.
    Callers must have pointed RULES_FILE at the real pack already."""
    if cache is not None and doc in cache:
        return cache[doc]
    sigs = [rep.replay(p, note) for p in payloads]
    if cache is not None:
        cache[doc] = sigs
    return sigs


def baseline_resolves(cls: dict, rep: Replayer, results_dir: Path,
                      cache: dict | None = None) -> bool:
    """Staleness pre-check: flips are recorded against the validator that
    ran the batch, but layers keep landing. If replaying the stored runs
    through the CURRENT pack already converges this class's codes on every
    replayable target document, the flip is already fixed — record that
    instead of spending a proposal on it. (A rerun of the batch will retire
    it from the queue naturally.)"""
    import app.validation.rule_engine as re_mod
    class_codes = {str(c).upper() for d in cls["documents"]
                   for c in ((d.get("disagreement") or {}).get("codes")
                             or [cls["code"]])}
    audit_kind = cls.get("kind") == "audit_dispute"
    registry = _registry_verified_claims() if audit_kind else {}
    code_targets = _per_code_targets() if audit_kind else {}
    advisory_targets = _advisory_targets() if audit_kind else {}
    re_mod.RULES_FILE = RULES_PATH
    re_mod.load_rule_pack.cache_clear()
    checked = False
    for d in cls["documents"]:
        doc = d["document_id"]
        runs = _load_runs(doc, results_dir)
        if audit_kind:
            # An audit dispute's runs already AGREE (the error is unanimous
            # by construction) — 'resolved' means the CURRENT pack's replay
            # lands every run exactly on the verified registry claim, or
            # (scoped verification from a partial adjudication) lands the
            # class's own codes exactly on their per-code verified rows,
            # or (advisory-shaped dispute) already emits every adjudicated
            # advisory in its verified state.
            goal = registry.get(doc)
            adv_goals = (_class_advisory_goals(advisory_targets, {doc},
                                               class_codes)
                         .get(doc) or {})
            payloads = runs or [m for m in [_load_main(doc, results_dir)]
                                if m]
            if (goal is None and doc not in code_targets
                    and not adv_goals) or not payloads:
                continue
            note = _note_text_for(doc, results_dir, payloads,
                                  _load_main(doc, results_dir))
            if not note:
                continue
            if adv_goals:
                scrubber = _advisory_scrubber(rep)
                _, advs = _replay_with_advisories(
                    rep, scrubber, payloads, note, set(adv_goals))
                goal_obs = {k[0] for k in adv_goals}
                resolved = all(a.get(k) == emit for a in advs
                               for k, emit in adv_goals.items()) \
                    and not any(a.get((o, "__error__")) for a in advs
                                for o in goal_obs)
                if not resolved:
                    return False  # the disputed phenomenon still misfires
                checked = True
            if goal is None and doc not in code_targets:
                continue  # advisory-only verification on this doc
            sigs = _baseline_sigs(doc, payloads, note, rep, cache)
            if goal is not None:
                if not _realigns(sigs, goal):
                    return False  # the wrong claim is alive in replay space
            elif doc in code_targets:
                # scoped verification replays with the verified rows
                # PRE-APPLIED (same mechanism as the gate trial): resolved
                # means the CURRENT pack already lets them survive
                projected = _project_code_targets(
                    payloads, code_targets[doc], class_codes, rep)
                if projected is not payloads:
                    sigs = [rep.replay(p, note) for p in projected]
                landed = _lands_on_code_targets(
                    sigs, code_targets[doc], class_codes)
                if landed is None:
                    continue  # targets cover none of this class's codes
                if not landed:
                    return False
            checked = True
            continue
        if len(runs) < 2:
            continue
        note = _note_text_for(doc, results_dir, runs,
                              _load_main(doc, results_dir))
        if not note:
            continue
        sigs = _baseline_sigs(doc, runs, note, rep, cache)
        if len({_code_rows(s, class_codes) for s in sigs}) > 1:
            return False  # the flip is alive in replay space
        checked = True
    return checked


def accept_rule(rule: dict, cls: dict, rationale: str,
                replay_detail: dict, amends: str = "") -> None:
    """Persist a governed draft; never mutate the production rule pack."""
    rule = copy.deepcopy(rule)
    rule["auto_generated"] = True
    rule["enabled"] = False
    rule["provenance"] = {
        "proposed_at": _now(),
        "flip_class": cls["class_key"],
        "documents": [d["document_id"] for d in cls["documents"]],
        "rationale": rationale,
        "replay": replay_detail,
    }
    if amends:
        rule["provenance"]["amends"] = amends
    body = {
        "proposal_version": 1,
        "status": "draft",
        "proposal_type": ("retire_rule" if rule.get("target_rule_id") else
                          "amend_rule" if amends else "add_rule"),
        "rule": rule,
        "required_lifecycle": [
            "independent_human_review", "signed_pack",
            "sandbox_replay", "shadow_deployment", "rollback_rehearsal",
        ],
    }
    fingerprint_body = copy.deepcopy(body)
    fingerprint_body["rule"]["provenance"].pop("proposed_at", None)
    encoded = json.dumps(fingerprint_body, sort_keys=True, separators=(",", ":"),
                         default=str).encode()
    body["proposal_fingerprint"] = "sha256:" + hashlib.sha256(
        encoded).hexdigest()
    PROPOSALS_DIR.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(rule["id"]))
    path = PROPOSALS_DIR / f"{safe}-{body['proposal_fingerprint'][7:19]}.json"
    if not path.exists():
        path.write_text(json.dumps(body, indent=2, sort_keys=True,
                                   default=str))


def _disable_rule(rule_id: str,
                  reason: str = "post-deployment pack audit failed",
                  superseded_by: str = "") -> None:
    """Create a retirement proposal; never disable a live rule directly."""
    proposal = {
        "id": f"retire-{rule_id}", "template": "governance_only",
        "target_rule_id": rule_id, "reason": reason,
        "superseded_by": superseded_by,
    }
    accept_rule(proposal, {"class_key": "governance/retirement",
                           "documents": []}, reason, {})


def _reenable_rule(rule_id: str) -> None:
    """No-op: draft proposals never change the live rule's state."""
    logger.info(f"Rule {rule_id} remained live; no rollback was necessary")


def audit_pack() -> list[str]:
    """Post-deployment bug check for the WHOLE live rule pack, run after
    every acceptance: every auto-generated rule must still parse, name a
    known template, carry no code literals in its selecting fields, and
    have a unique id — and every installed self-authored template module
    must still pass its full static gate (sandbox constraints AND the
    no-hardcoded-medical-codes scan). Catches anything that slipped in
    outside the gates (hand edits, older-vintage rules, merge accidents)
    the moment the pack changes — not on some future failing batch."""
    problems: list[str] = []
    try:
        from app.validation.auto_templates import (AUTO_TEMPLATES_DIR,
                                                   validate_template_source)
        if AUTO_TEMPLATES_DIR.exists():
            for f in sorted(AUTO_TEMPLATES_DIR.glob("*.py")):
                bad = validate_template_source(f.read_text(
                    encoding="utf-8"))
                problems += [f"template {f.name}: {p}" for p in bad[:4]]
    except Exception as exc:
        problems.append(f"template audit failed: {exc}")
    try:
        pack = json.loads(RULES_PATH.read_text())
    except Exception as exc:
        return problems + [f"rule pack unreadable: {exc}"]
    seen_ids: set[str] = set()
    for r in pack.get("rules", []):
        rid = str(r.get("id", "?"))
        if rid in seen_ids:
            problems.append(f"duplicate rule id {rid!r}")
        seen_ids.add(rid)
        if not r.get("auto_generated") or not r.get("enabled", True):
            # Disabled rules never execute — they are rollback audit
            # trails. Auditing one (e.g. a rolled-back rule referencing
            # its removed template) rolled back a HEALTHY later
            # acceptance in production; only live rules can be defects.
            continue
        if r.get("template") not in all_templates():
            problems.append(f"{rid}: unknown template {r.get('template')!r}")
        hit = gate_no_code_literals(r)
        if hit:
            problems.append(f"{rid}: {hit}")
    return problems


def _in_scope(doc: str, scope: tuple[str, ...]) -> bool:
    return not scope or any(doc.startswith(p) for p in scope)


def _replayable(cls: dict, results_dir: Path) -> bool:
    """A class is actionable only if at least one of its documents has the
    replay material (per-run artifacts) the convergence gate needs — no
    point spending a proposal on evidence we can't verify against."""
    return any(_load_runs(d["document_id"], results_dir)
               for d in cls["documents"])


def actuate(results_dir: Path, limit: int, dry_run: bool,
            scope: tuple[str, ...] = ()) -> dict:
    flip_triage.scan(results_dir)
    queue = flip_triage.load_queue()
    # Scope restriction: every class is narrowed to its in-scope documents
    # BEFORE eligibility, dossier assembly, and replay — evidence and
    # verification both stay inside the requested corpus.
    if scope:
        queue = [dict(c, documents=[d for d in c["documents"]
                                    if _in_scope(d["document_id"], scope)])
                 for c in queue]
        queue = [c for c in queue if c["documents"]]
    open_classes = [c for c in queue
                    if c["status"] == "open" and _replayable(c, results_dir)]
    # Template-vocabulary growth automatically reopens escalations: a class
    # escalated because "no template fits" was judged against the templates
    # of ITS day. Every escalation records the vocabulary it saw; when the
    # current vocabulary differs, the verdict is stale and the class earns
    # a fresh proposal — no human re-queuing.
    for c in queue:
        if c["status"] != "escalated" or not _replayable(c, results_dir):
            continue
        act = c.get("actuation") or {}
        seen = set(act.get("templates_available") or ())
        proto = act.get("proposal_protocol") or 1
        # 'proposal failed:' escalations are infrastructure outages
        # (credit exhaustion, API errors) recorded before transient
        # failures learned to skip — never a judgment; always retry.
        transient = str(act.get("reason") or "").startswith(
            "proposal failed:")
        if (seen != set(all_templates()) or proto != PROPOSAL_PROTOCOL
                or transient):
            why = ("a transient infrastructure failure" if transient
                   else "a smaller template vocabulary"
                   if seen != set(all_templates())
                   else f"proposal protocol {proto} (now "
                        f"{PROPOSAL_PROTOCOL})")
            logger.info(f"Reopening {c['class_key']}: escalated under "
                        f"{why}")
            c["status"] = "open"
            open_classes.append(c)
    # Most-recurrent first: a class seen on several documents is both the
    # highest-value fix and the best-evidenced one.
    open_classes.sort(key=lambda c: -len(c["documents"]))
    open_classes = open_classes[:limit]
    summary = {"considered": len(open_classes), "proposed": 0,
               "escalated": 0, "classes": []}
    if not open_classes:
        logger.info("Flip queue has no open classes — nothing to actuate")
        return summary

    rep = Replayer()
    pack = json.loads(RULES_PATH.read_text())
    # Baseline replay signatures per document, valid until the pack changes
    # (an acceptance) — replaying the same unchanged pack against the same
    # stored runs for every class was the actuation loop's dominant cost.
    baseline_cache: dict = {}

    # Phase 1 (sequential): staleness pre-check + dossier assembly. Both
    # touch the shared SQLite connection and the global rule-pack pointer,
    # neither of which is thread-safe — and both are cheap local work.
    to_propose: list[tuple[dict, dict]] = []  # (cls, dossier)
    for cls in open_classes:
        key = cls["class_key"]
        try:
            if cls.get("kind") == "audit_dispute" \
                    and not _audit_class_anchored(cls):
                outcome = {
                    "class_key": key, "status": "awaiting_verification",
                    "reason": "no verified realignment target remains in "
                              "the registry for this class (registry "
                              "wipe or voided verdict) — parked until "
                              "adjudication re-verifies one"}
                if not dry_run:
                    flip_triage.set_status(key, "awaiting_verification", {
                        k: v for k, v in outcome.items()
                        if k != "class_key"})
                summary["classes"].append(outcome)
                summary["unanchored"] = summary.get("unanchored", 0) + 1
                logger.info(f"{key} -> UNANCHORED (no verified target in "
                            f"the registry) — parked awaiting verification")
                continue
            if baseline_resolves(cls, rep, results_dir,
                                 cache=baseline_cache):
                outcome = {
                    "class_key": key, "status": "resolved_baseline",
                    "reason": "replaying the stored runs through the "
                              "CURRENT rule pack already converges this "
                              "class's codes — fixed by layers accepted "
                              "since the batch ran"}
                if not dry_run:
                    flip_triage.set_status(key, "resolved_baseline", {
                        k: v for k, v in outcome.items()
                        if k != "class_key"})
                summary["classes"].append(outcome)
                summary["resolved_baseline"] = \
                    summary.get("resolved_baseline", 0) + 1
                logger.info(f"{key} -> RESOLVED at baseline "
                            f"(no rule needed)")
                continue
            to_propose.append((cls, build_dossier(cls, rep, results_dir)))
        except Exception as exc:
            to_propose.append((cls, {"_dossier_error": str(exc)}))

    # Phase 2 (parallel): the LLM proposals — pure network calls, the
    # dominant sequential cost of the loop. Order is preserved so the
    # acceptance phase below stays deterministic.
    from concurrent.futures import ThreadPoolExecutor

    def _propose(item):
        cls, dossier = item
        if "_dossier_error" in dossier:
            return {"decision": "escalate",
                    "reason": f"proposal failed: {dossier['_dossier_error']}"}
        try:
            return propose_rule(dossier, pack)
        except Exception as exc:
            # An infrastructure failure (credit exhaustion, outage, JSON
            # mangling) is NOT a judgment about the class — recording it
            # as 'escalated' under the current protocol would stop it
            # from ever auto-reopening. Leave the class untouched in the
            # queue so the next pass simply retries.
            return {"decision": "escalate", "_transient": True,
                    "reason": f"proposal failed: {exc}"}

    n_workers = min(int(os.getenv("AUTO_ACTUATE_PROPOSAL_WORKERS", "4")),
                    max(len(to_propose), 1))
    if to_propose:
        logger.info(f"Proposing rules for {len(to_propose)} class(es) "
                    f"with {n_workers} parallel worker(s)...")
        # Pre-warm the provider client: it's a lazily-initialized module
        # global, and letting the pool's threads race its first init could
        # construct it twice.
        from app.core import llm_client
        from app.core.config import LLM_PROVIDER
        (llm_client.get_anthropic_client if LLM_PROVIDER == "claude"
         else llm_client.get_openai_client)()
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        proposals = list(ex.map(_propose, to_propose))

    # Phase 3 (sequential): deterministic gates + draft persistence. The
    # live baseline never changes during this pass.
    synth_candidates: list[tuple[dict, dict, dict]] = []
    for (cls, dossier), proposal in zip(to_propose, proposals):
        key = cls["class_key"]
        logger.info(f"=== Actuating flip class {key} "
                    f"({len(cls['documents'])} doc(s)) ===")
        outcome = {"class_key": key}

        if proposal.get("_transient"):
            # Infrastructure failure, not a judgment: the class stays in
            # its current status and the next pass retries it. Writing an
            # 'escalated' verdict here would burn the protocol's reopen
            # trigger on an outage.
            outcome.update(status=cls.get("status", "open"), skipped=True,
                           reason=proposal.get("reason", ""))
            summary["classes"].append(outcome)
            summary["transient_failures"] = \
                summary.get("transient_failures", 0) + 1
            logger.warning(f"  -> SKIPPED (transient): "
                           f"{outcome['reason'][:200]}")
            continue

        decision = str(proposal.get("decision") or "")
        if decision in ("amend_rule", "disable_rule"):
            # Amendment of a DEPLOYED rule — audit_dispute classes only,
            # and only rules the dossier named as implicated. Same gates
            # as a new rule; the pack mutation (old rule disabled, and for
            # amend_rule the replacement appended) is trialed as a whole
            # by gate_replay before anything is written.
            target_id = str(proposal.get("target_rule_id") or "")
            implicated = {str(r.get("id")) for r in
                          ((dossier.get("implicated_rules") or {})
                           .get("rules") or [])}
            rule = proposal.get("rule") or {}
            reason, detail = "", {}
            if cls.get("kind") != "audit_dispute":
                reason = ("amend/disable is reserved for audit_dispute "
                          "classes — consistency flips stay append-only")
            elif target_id not in implicated:
                reason = (f"target_rule_id {target_id!r} is not an "
                          f"implicated rule for this class")
            elif decision == "amend_rule":
                existing = {r.get("id") for r in json.loads(
                    RULES_PATH.read_text()).get("rules", [])}
                if rule.get("id") in existing:
                    base_id, n = str(rule["id"]), 2
                    while f"{base_id}-r{n}" in existing:
                        n += 1
                    rule = dict(rule, id=f"{base_id}-r{n}")
                reason = (gate_structural(rule)
                          or gate_no_code_literals(rule))
                if not reason:
                    reason, detail = gate_replay(
                        rule, cls, queue, rep, results_dir, scope,
                        baseline_cache=baseline_cache,
                        disable_rule_id=target_id)
            else:
                reason, detail = gate_replay(
                    None, cls, queue, rep, results_dir, scope,
                    baseline_cache=baseline_cache,
                    disable_rule_id=target_id)
            if reason:
                outcome.update(status="escalated", reason=reason,
                               proposed_amendment=decision,
                               target_rule_id=target_id,
                               proposed_rule=rule or None,
                               replay=detail or None,
                               templates_available=sorted(all_templates()),
                               proposal_protocol=PROPOSAL_PROTOCOL)
            else:
                outcome.update(status="proposed", replay=detail,
                               amendment=decision,
                               superseded_rule_id=target_id)
                if decision == "amend_rule":
                    outcome["rule_id"] = rule["id"]
                if not dry_run:
                    rationale = proposal.get("rationale", "")
                    if decision == "amend_rule":
                        accept_rule(rule, cls, rationale,
                                    dict(detail, proposal_model=proposal
                                         .get("_model")),
                                    amends=target_id)
                    else:
                        _disable_rule(
                            target_id,
                            reason=(f"retirement proposed by audit-dispute "
                                    f"class {key}: "
                                    f"{str(rationale)[:300]}"))
        elif decision != "rule":
            reason = proposal.get("reason", "model chose to escalate")
            # Record the template vocabulary this escalation was judged
            # against — when a later release grows it, the stale verdict
            # auto-reopens for a fresh proposal (see actuate()'s reopen).
            outcome.update(status="escalated", reason=reason,
                           templates_available=sorted(all_templates()),
                           proposal_protocol=PROPOSAL_PROTOCOL)
            hint = proposal.get("missing_template")
            if isinstance(hint, dict) and hint.get("mechanism"):
                outcome["missing_template"] = hint
                synth_candidates.append((cls, dossier, hint))
        else:
            rule = proposal.get("rule") or {}
            reason = (gate_structural(rule) or gate_no_code_literals(rule))
            detail: dict = {}
            if not reason:
                reason, detail = gate_replay(rule, cls, queue, rep,
                                             results_dir, scope,
                                             baseline_cache=baseline_cache)
            if reason:
                # The full rejected rule rides in the queue detail — flip
                # forensics need to see WHAT was tried, not just that it
                # failed, and a reopened class should not re-derive it blind.
                outcome.update(status="escalated", reason=reason,
                               proposed_rule_id=rule.get("id"),
                               proposed_rule=rule,
                               replay=detail or None,
                               templates_available=sorted(all_templates()),
                               proposal_protocol=PROPOSAL_PROTOCOL)
                # A rule citing a nonexistent template is a missing-
                # template hint in disguise — route it into synthesis
                # instead of leaving it for the human queue.
                if reason.startswith("unknown template"):
                    hint = _unknown_template_hint(
                        rule, proposal.get("rationale", ""))
                    if hint:
                        outcome["missing_template"] = hint
                        synth_candidates.append((cls, dossier, hint))
            else:
                outcome.update(status="proposed", rule_id=rule["id"],
                               replay=detail)
                if not dry_run:
                    accept_rule(rule, cls, proposal.get("rationale", ""),
                                dict(detail,
                                     proposal_model=proposal.get("_model")))

        if not dry_run:
            flip_triage.set_status(
                key, outcome["status"],
                {k: v for k, v in outcome.items() if k != "class_key"})
        summary["classes"].append(outcome)
        summary[outcome["status"]] += 1
        if outcome["status"] == "escalated":
            tail = f": {outcome.get('reason')}"
        elif outcome.get("amendment") == "disable_rule":
            tail = f" (disabled rule {outcome.get('superseded_rule_id')})"
        elif outcome.get("amendment") == "amend_rule":
            tail = (f" (rule {outcome.get('rule_id')} amends "
                    f"{outcome.get('superseded_rule_id')})")
        else:
            tail = f" (rule {outcome.get('rule_id')})"
        logger.info(f"  -> {outcome['status'].upper()}{tail}")

    # Phase 4: escalations whose blocker was VOCABULARY (a structured
    # missing-template hint) get their template designed and gated into an
    # inert proposal. The live vocabulary never changes in this process.
    if synth_candidates and os.getenv("AUTO_TEMPLATE_SYNTH", "1") == "1":
        logger.info(f"{len(synth_candidates)} escalation(s) carry a "
                    f"missing-template hint — entering template synthesis")
        try:
            synthesize_templates(synth_candidates, queue, rep, results_dir,
                                 scope, baseline_cache, dry_run, summary)
        except Exception as exc:
            logger.error(f"Template synthesis failed: {exc!r} — classes "
                         f"remain escalated")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", default=str(flip_triage.DEFAULT_RESULTS))
    p.add_argument("--limit", type=int, default=5,
                   help="max flip classes to actuate this invocation")
    p.add_argument("--dry-run", action="store_true",
                   help="evaluate and report; write neither pack nor queue")
    p.add_argument("--scope", action="append", default=[],
                   help="document-id prefix to restrict to (repeatable)")
    args = p.parse_args()
    summary = actuate(Path(args.results_dir), args.limit, args.dry_run,
                      scope=tuple(args.scope))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
