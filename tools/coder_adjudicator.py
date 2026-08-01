#!/usr/bin/env python3
"""Automated expert-coder adjudication for judgment-shaped holdouts.

The growth loop (rules + templates) converges every disagreement that a
generic deterministic mechanic can decide. What remains — modifier-25
significance, E/M MDM leveling, documentation-sufficiency calls — is
coder judgment. This module automates that role the only defensible way:
as RULE APPLICATION, never intuition. A reasoning model (Fable 5) acts
as an expert coder whose every determination must be derived from the
authoritative sources the system already holds — NCCI Policy Manual
Chapter 1, the AMA 2021 E/M MDM framework as published in the CPT
descriptors, CMS documentation principles, ICD-10-CM Official
Guidelines, and the claim's own reference data (descriptors, PTP edits,
MUE, Tabular conventions) assembled into the case file. No authority,
no verdict: the adjudicator must abstain, and abstention routes the
note to the human queue exactly as before.

Trust is engineered, not assumed:
  - N independent adjudication passes (default 2) must agree on every
    disputed item — a split adjudication is treated as no adjudication.
  - Every decision must cite its authority AND the note evidence (or
    the documented ABSENCE of evidence, per CMS's "if it is not
    documented, it was not done").
  - Verdicts are applied MECHANICALLY, and only to the disputed items
    from the consistency report — the adjudicator cannot touch any
    other line, attribute, or array (enforced in code, not by trust).
  - The realigned runs are replayed through the full deterministic
    stack (validator rule pack + claim scrubber); the note is saved
    only if the replay is unanimous, and auto-recorded to the registry
    only if the scrub disposition is CLEAN.
  - Registry precedence: adjudicated claims outrank the pipeline's
    auto records but NEVER a human coder's record.

CLI (inside the app container):
  docker compose run --rm app python tools/coder_adjudicator.py \
      [results_dir] [--docs stem1,stem2] [--dry-run]
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from loguru import logger  # noqa: E402

from tools.auto_actuate import (  # noqa: E402
    Replayer, _authoritative_evidence, _load_main, _load_runs,
    _note_text_for, _registry_verified_claims, _sig_view)

DEFAULT_RESULTS = ROOT / "output" / "results"

# Agreement across independent passes is the adjudicator's whole basis
# for trust — a single pass has nothing to agree WITH, so 2 is a floor,
# not a default.
ADJUDICATION_PASSES = max(2, int(os.getenv("CODER_ADJUDICATION_PASSES",
                                           "2")))
CODER_MODEL = os.getenv("CODER_ADJUDICATOR_MODEL", "claude-fable-5")
# Cross-family second opinion: when set, passes AFTER the first run on
# this model instead of CODER_MODEL. Unanimity between different model
# FAMILIES is materially stronger evidence that a verdict is
# authority-determined than unanimity between temperature re-rolls of one
# family (shared training biases agree with themselves). Optional — unset
# keeps the existing same-model varied-temperature protocol.
ALT_MODEL = os.getenv("CODER_ADJUDICATOR_ALT_MODEL", "").strip()
# Per-invocation cost ceiling: adjudication runs at batch finalization,
# and a misconfigured batch over a huge corpus must not fan out into
# thousands of reasoning-model calls. Holdouts beyond the cap stay
# deferred and route to human review exactly as before.
ADJUDICATION_LIMIT = int(os.getenv("CODER_ADJUDICATION_LIMIT", "25"))

_BILLING_ARRAYS = ("icd_codes", "cpt_codes", "hcpcs_codes")


# --------------------------------------------------------------------------
# The adjudication protocol — the licensed-coder decision procedures
# --------------------------------------------------------------------------

_ADJUDICATOR_PROMPT = """\
You are an expert certified professional coder (CPC) adjudicating a
podiatry claim for a private practice (professional claims, office
setting). The pipeline ran this note multiple times; the runs agree on
everything EXCEPT the disputed items listed. Your job is to decide each
disputed item the way an auditor-proof coder would.

BINDING RULES OF THIS ROLE:
1. AUTHORITY, NOT INTUITION. Every decision must be DERIVED from an
   authoritative source: the CPT/HCPCS/ICD descriptors and reference
   data in the case file, NCCI Policy Manual Chapter 1 (and the PTP
   edit data provided), the AMA E/M medical-decision-making framework
   spelled in the E/M descriptors themselves, ICD-10-CM Official
   Guidelines (including IV.G: first-listed diagnosis is the condition
   chiefly responsible for the services provided), and CMS
   documentation policy. Name the source and the specific principle in
   every decision's "authority" field.
2. THE NOTE IS THE ONLY EVIDENCE. CMS documentation principle: if it
   is not documented, it was not done. Quote the note sentence(s) that
   ground each decision in "note_evidence". A decision to EXCLUDE for
   lack of documentation must state what documentation is absent.
3. CONSERVATIVE DEFAULTS, AS THE AUTHORITIES MANDATE:
   - Same-day E/M with a minor procedure (modifier 25, NCCI Ch.1): the
     evaluation inherent to the procedure (assessing the problem, the
     decision to perform it, consent, immediate follow-up) is BUNDLED.
     Report the E/M only when the documentation shows significant,
     separately identifiable work BEYOND that inherent evaluation —
     e.g. a distinct problem worked up, a new diagnostic evaluation
     with its own MDM, a documented change in management beyond the
     procedure. The same diagnosis alone does not disqualify, but
     routine pre-procedure history/exam language alone does not
     qualify. Insufficient documentation -> the E/M is NOT separately
     billable.
   - E/M LEVEL: each disputed E/M code's case-file entry carries
     "mdm_requirements" — the licensed AMA MDM table row that code's
     own descriptor requires (problems addressed, data categories with
     their combination thresholds, risk examples, and the 2-of-3
     selection rule). Score the note against THAT row, element by
     element, citing the specific table language you matched. Time may
     substitute ONLY when total time is documented in the note (the
     minute threshold is in the code's descriptor). When the
     documented elements do not meet the higher level's row,
     the lower level is correct.
   - Distinct-service modifiers (59/X): valid only to bypass an actual
     NCCI PTP edit with another PROCEDURE on this claim whose edit
     data (provided) permits bypass, with documented distinct
     site/session/service. No qualifying edit -> the modifier is
     meaningless and must be removed. E/M bundling is modifier-25
     territory, never 59.
   - Anatomic modifiers (RT/LT/50, toe modifiers): only on lines whose
     descriptor has an anatomic axis; follow the reference data.
   - Diagnoses: a code is reportable only when the note documents the
     condition (or its ICD-mandated linkage in the provided Tabular
     conventions). First-listed per ICD-10-CM IV.G. A documented
     condition supported by the note stays; an undocumented one goes.
4. DECIDE ONLY THE DISPUTED ITEMS. Everything else on this claim is
   settled and outside your authority. Do not propose changes to any
   other line, attribute, or array.
5. ABSTAIN WHEN THE AUTHORITIES DO NOT DECIDE. If the note's
   documentation genuinely cannot support a determination either way
   under the rules above, mark the item "abstain" with the reason. An
   abstained note goes to a human coder — that is the correct outcome,
   not a failure.
6. A registry-verified prior claim for THIS note, if present, is
   settled ground truth from a previous adjudication of the same
   encounter — your decisions must be consistent with it unless the
   note evidence you quote contradicts it outright.

7. QUOTE THE POLICY, NOT JUST ITS NAME. Declare each decision's
   grounding in "authority_basis":
   - "reference_data" — the decision is fully derivable from the case
     file's own reference data (descriptors, MDM rows, PTP/MUE edit
     data, Tabular conventions). Leave "authority_quote" empty; the
     deterministic gates re-verify these against the data itself.
   - "policy_prose" — the decision rests on prose policy (a coverage
     pathway, a documentation principle, a manual chapter). When the
     policy document appears in the case file's
     "quotable_policy_sources" list, put the VERBATIM passage you are
     relying on in "authority_quote" — the exact sentence(s), not a
     paraphrase. Every quote is mechanically checked against those
     stored documents; a quote that exists in none of them voids the
     verdict, so NEVER put text in "authority_quote" from any source
     outside that list (cite such sources in "authority" and leave the
     quote empty). A prose decision without a verified quote records at
     the weakest attestation tier and cannot anchor deterministic
     realignment.

Respond with JSON only:
{"items": [
   // one entry per disputed item, same array/code identity as given:
   {"array": "cpt_codes", "code": "<code>", "kind": "presence",
    "decision": "include" | "exclude" | "abstain",
    "authority": "<source + principle>",
    "authority_basis": "reference_data" | "policy_prose",
    "authority_quote": "<verbatim policy passage when policy_prose, "
                       "else empty>",
    "note_evidence": "<verbatim quote, or the absence being relied on>"},
   {"array": "...", "code": "<code>", "kind": "attributes",
    "decision": "set" | "abstain",
    "fields": {"<disputed field>": <adjudicated value>},
    "authority": "...", "authority_basis": "...",
    "authority_quote": "...", "note_evidence": "..."},
   {"array": "cpt_codes", "kind": "em_level", "codes": [...as given...],
    "decision": "select" | "exclude_all" | "abstain",
    "decision_code": "<the correct family member, when select>",
    "authority": "...", "authority_basis": "...",
    "authority_quote": "...", "note_evidence": "..."}
 ],
 "overall_rationale": "<2-4 sentences>"}"""


# --------------------------------------------------------------------------
# Case assembly — everything an adjudicator may rely on, from the data
# --------------------------------------------------------------------------

def _claim_lines(run: dict) -> dict:
    return {arr: [{k: e.get(k) for k in ("code", "description",
                                         "modifiers", "units", "type")}
                  for e in (run.get(arr) or []) if isinstance(e, dict)]
            for arr in _BILLING_ARRAYS}


def _ptp_evidence(rep: Replayer, runs: list[dict],
                  disputed_codes: set[str]) -> list[dict]:
    """NCCI PTP edits between every disputed CPT/HCPCS code and every
    other procedure line any run billed — the authority for every
    bundling / distinct-service-modifier decision."""
    all_codes = sorted({str(e.get("code") or "").upper()
                        for run in runs
                        for arr in ("cpt_codes", "hcpcs_codes")
                        for e in (run.get(arr) or [])
                        if isinstance(e, dict) and e.get("code")})
    out = []
    for c in sorted(disputed_codes):
        for other in all_codes:
            if other == c:
                continue
            try:
                pair = rep.store.ncci_pair(c, other)
            except Exception:
                pair = None
            if pair:
                out.append({"codes": [c, other], "ptp_edit": pair})
    return out


def assemble_case(doc: str, results_dir: Path, rep: Replayer) -> dict | None:
    main = _load_main(doc, results_dir)
    runs = _load_runs(doc, results_dir)
    if not main or len(runs) < 2:
        return None
    report = main.get("consistency") or {}
    billing = [d for d in (report.get("disagreements") or [])
               if not d.get("advisory")]
    if not billing:
        return None
    note = _note_text_for(doc, results_dir, runs, main)
    if not note:
        return None

    disputed_by_array: dict[str, set[str]] = {}
    for d in billing:
        codes = d.get("codes") or [d.get("code")]
        disputed_by_array.setdefault(d["array"], set()).update(
            str(c).upper() for c in codes if c)
    evidence = {arr: _authoritative_evidence(rep, arr, sorted(codes))
                for arr, codes in disputed_by_array.items()}
    proc_disputed = (disputed_by_array.get("cpt_codes", set())
                     | disputed_by_array.get("hcpcs_codes", set()))

    case = {
        "document_id": doc,
        "note_text": note[:12000],
        "disputed_items": billing,
        "per_run_claims": [_claim_lines(r) for r in runs],
        "authoritative_reference_data": evidence,
        "ncci_ptp_edits_on_this_claim":
            _ptp_evidence(rep, runs, proc_disputed),
        "quotable_policy_sources": _quotable_sources(),
    }
    anchor = _registry_verified_claims().get(doc)
    if anchor:
        case["registry_verified_claim"] = _sig_view(anchor)
    return case


def _ensure_policy_corpus() -> None:
    """Autonomous corpus upkeep at the verdict choke point: every driver
    (run.py batch, unanimity loop, finalize scope, audit-convergence
    loop, direct CLI) reaches adjudication through adjudicate() /
    adjudicate_audit(), so calling ensure() here activates the
    quote-verification ground truth for ALL of them with no manual fetch
    and no per-driver wiring. Cheap when fresh (stat/JSON reads only);
    fetches missing/stale sources otherwise; never raises — a fetch
    failure degrades to verification against whatever is stored, which
    is the same fail-closed posture the verifier already has."""
    try:
        from tools.policy_corpus import ensure
        stats = ensure()
        if stats.get("fetched") or stats.get("failed"):
            logger.info(f"policy corpus ensure: {stats}")
    except Exception as exc:
        logger.warning(f"policy corpus ensure unavailable ({exc})")


def _quotable_sources() -> list[str]:
    """Titles of the stored policy documents (tools/policy_corpus.py) —
    surfaced in the case file so the adjudicator knows exactly which
    sources authority_quote may draw from. A model quoting a document
    the corpus does not hold would be voided by the fabricated-quote
    check even when the passage is real; telling it the boundary keeps
    honest verdicts out of that trap."""
    try:
        from tools.policy_corpus import catalog
        return catalog()
    except Exception as exc:
        logger.warning(f"policy catalog unavailable ({exc})")
        return []


# --------------------------------------------------------------------------
# Adjudication passes + agreement
# --------------------------------------------------------------------------

def _adjudicate_once(case: dict, pass_idx: int = 0,
                     system_suffix: str = "") -> dict:
    from app.core.config import LLM_PROVIDER
    from app.core.llm_client import chat_completion
    system = _ADJUDICATOR_PROMPT + system_suffix
    user = (f"CASE FILE:\n{json.dumps(case, indent=1, default=str)}\n\n"
            f"Adjudicate every disputed item, or abstain per item.")
    model = CODER_MODEL if LLM_PROVIDER == "claude" else None
    if pass_idx > 0 and ALT_MODEL and LLM_PROVIDER == "claude":
        model = ALT_MODEL
    # Later passes sample at a higher temperature: a second opinion at
    # near-greedy temperature is largely a re-roll of the first, and
    # agreement between re-rolls proves little. Agreement between a
    # greedy pass and a genuinely varied one (or, when configured, a
    # different model family) is evidence the verdict is
    # authority-determined rather than a sampling artifact.
    temperature = 0.05 if pass_idx == 0 else 0.4
    try:
        text, usage = chat_completion(
            system_prompt=system, user_prompt=user,
            model=model, temperature=temperature, max_tokens=8192,
            json_mode=True, effort="high")
    except Exception as exc:
        if model is None:
            raise
        logger.warning(f"Adjudicator model {model!r} failed ({exc}) — "
                       f"falling back to the pipeline default")
        model = None
        text, usage = chat_completion(
            system_prompt=system, user_prompt=user,
            temperature=temperature, max_tokens=8192,
            json_mode=True, effort="high")
    verdict = json.loads(text)
    verdict["_model"] = model or "pipeline-default"
    verdict["_usage"] = usage
    return verdict


def _item_key(d: dict) -> tuple:
    if d.get("kind") == "em_level":
        return (d.get("array", "cpt_codes"), "em_level",
                tuple(sorted(str(c).upper() for c in (d.get("codes") or []))))
    return (d.get("array"), str(d.get("kind")),
            (str(d.get("code") or "").upper(),))


def _norm_decision(item: dict) -> tuple | None:
    """A verdict item reduced to its comparable, applicable content.
    None = abstain / malformed — either voids the adjudication for the
    whole note (atomic: partial judgment is no judgment)."""
    dec = str(item.get("decision") or "").lower()
    if item.get("kind") == "em_level":
        family = {str(c).upper() for c in (item.get("codes") or [])}
        chosen = str(item.get("decision_code") or "").upper()
        # "select" must name a member of the disputed family — the codes
        # the runs actually produced. Anything else is the adjudicator
        # inventing a NEW code, which is outside its authority (and would
        # silently drop the E/M when no run carries an entry for it).
        if dec == "select" and chosen in family:
            return ("select", chosen)
        if dec == "exclude_all":
            return ("exclude_all",)
        return None
    if item.get("kind") == "presence":
        return (dec,) if dec in ("include", "exclude") else None
    if item.get("kind") in ("advisory", "observable"):
        # emission verdict on a measured phenomenon (scrubber advisory,
        # or any synthesized observable) — changes no claim line.
        # "advisory" is the pre-generalization name, kept so stored
        # adjudication items from those records still reconstruct.
        return (dec,) if dec in ("suppress", "stand") else None
    if item.get("kind") == "attributes":
        if dec != "set" or not isinstance(item.get("fields"), dict):
            return None
        fields = []
        for k, v in sorted(item["fields"].items()):
            if k == "modifiers":
                v = sorted(str(m).upper() for m in (v or []))
            fields.append((k, json.dumps(v, sort_keys=True, default=str)))
        return ("set", tuple(fields))
    return None


def _grounded(item: dict) -> bool:
    return bool(str(item.get("authority") or "").strip()) and \
        bool(str(item.get("note_evidence") or "").strip())


def _quote_fabricated(item: dict) -> bool:
    """True when the item cites a verbatim policy passage that does NOT
    occur in any stored policy source (tools/policy_corpus.py). A
    citation is an attestation; a quote is a lookup — and a quote that
    fails the lookup is the one hard signal the model invented its
    authority, so it voids the pass verdict (fail closed: the dispute
    stays held, nothing wrong ships). No corpus stored, or no quote
    given, is NOT fabrication — those degrade to the attestation tier
    on the recorded target instead. A broken corpus module degrades to
    the pre-corpus behavior (no voiding), never to a crashed
    adjudication."""
    quote = str(item.get("authority_quote") or "").strip()
    if not quote:
        return False
    try:
        from tools.policy_corpus import corpus_available, verify_quote
        if not corpus_available():
            return False
        res = verify_quote(quote, str(item.get("authority") or ""))
        if not res["verified"]:
            logger.warning(f"  fabricated-quote check: no stored source "
                           f"contains {quote[:120]!r} ({res['why']})")
        return not res["verified"]
    except Exception as exc:
        logger.warning(f"  quote verification unavailable ({exc}) — "
                       f"skipped")
        return False


def _attestation_of(item: dict | None) -> str:
    """The policy-grounding tier of one agreed adjudication item
    (document_quoted / attested_only / unverified) — stamped onto every
    recorded registry target so the actuation gates can refuse to anchor
    on quote-less prose verdicts. See tools/policy_corpus.attest."""
    try:
        from tools.policy_corpus import attest
        return attest(item or {})
    except Exception as exc:
        logger.warning(f"  attestation unavailable ({exc}) — recorded as "
                       f"unverified")
        return "unverified"


def _verdict_map(verdict: dict, disputed: list[dict]) -> dict | None:
    """{item_key: normalized decision} — None when the verdict is
    incomplete, ungrounded, abstaining, quote-fabricating, or names items
    that were never disputed (an adjudicator inventing scope voids its
    verdict)."""
    wanted = {_item_key(d) for d in disputed}
    out: dict[tuple, tuple] = {}
    for item in (verdict.get("items") or []):
        key = _item_key(item)
        if key not in wanted or key in out:
            # outside the disputed scope, or two conflicting decisions
            # for the same item — either way the verdict is not a
            # coherent judgment
            return None
        norm = _norm_decision(item)
        if norm is None or not _grounded(item) or _quote_fabricated(item):
            return None
        out[key] = norm
    return out if set(out) == wanted else None


# --------------------------------------------------------------------------
# Mechanical application — verdicts touch ONLY the disputed items
# --------------------------------------------------------------------------

def _find_entry(runs: list[dict], array: str, code: str) -> dict | None:
    for run in runs:
        for e in run.get(array) or []:
            if isinstance(e, dict) and \
                    str(e.get("code") or "").upper() == code:
                return copy.deepcopy(e)
    return None


def _apply_to_run(run: dict, decisions: dict[tuple, tuple],
                  disputed: list[dict], all_runs: list[dict]) -> dict:
    out = json.loads(json.dumps(run, default=str))
    for d in disputed:
        if d.get("kind") in ("advisory", "observable"):
            # an emission verdict mutates no claim array — its
            # realization is a registry emission target the actuation
            # gates converge on, never a mechanical claim edit
            continue
        key = _item_key(d)
        dec = decisions[key]
        array = d["array"]
        entries = [e for e in (out.get(array) or []) if isinstance(e, dict)]

        # A decided code becomes ONE canonical entry in every run (first
        # run carrying it wins, deterministically) — never each run's own
        # variant. A presence flip masks attribute comparison (attributes
        # are only compared for codes present in ALL runs), so letting
        # each run keep its own entry can silently convert a settled
        # presence flip into a fresh attributes flip (e.g. ICD type,
        # modifiers) and strand the note split after adjudication.
        if d.get("kind") == "em_level":
            family = {str(c).upper() for c in (d.get("codes") or [])}
            kept = [e for e in entries
                    if str(e.get("code") or "").upper() not in family]
            if dec[0] == "select":
                tpl = _find_entry(all_runs, array, dec[1])
                if tpl:
                    kept.append(tpl)
            out[array] = kept

        elif d["kind"] == "presence":
            code = str(d.get("code") or "").upper()
            kept = [e for e in entries
                    if str(e.get("code") or "").upper() != code]
            removed = [e for e in entries if e not in kept]
            if dec[0] == "include":
                ent = _find_entry(all_runs, array, code)
                if ent:
                    kept.append(ent)
            elif array == "icd_codes" and removed:
                # system invariant: diagnoses demote, never vanish
                sup = out.setdefault("supporting_conditions", [])
                for e in removed:
                    sup.append(dict(e, demoted_by="coder_adjudication"))
            out[array] = kept

        elif d["kind"] == "attributes":
            code = str(d.get("code") or "").upper()
            allowed = set(d.get("fields") or [])
            for e in entries:
                if str(e.get("code") or "").upper() != code:
                    continue
                for fname, fval in dict(dec[1]).items():
                    # fail CLOSED: only fields the consistency report
                    # itself flagged as disputed are writable — an empty
                    # allowlist grants nothing
                    if fname not in allowed:
                        continue
                    e[fname] = json.loads(fval)
    return out


def _decisions_applicable(decisions: dict, disputed: list[dict],
                          runs: list[dict]) -> str | None:
    """Reason the verdicts CANNOT be realized on the stored runs, or None.

    An 'include'/'select' decision needs a real entry to materialize from
    (some run must carry the code — true by construction for a live
    presence flip, but the embedded report can be stale relative to the
    run dumps on disk). Without this check a stale include would silently
    no-op, the signature guard would see aligned runs, and the note would
    save WITHOUT the code its verdict said to include."""
    for d in disputed:
        dec = decisions[_item_key(d)]
        if d.get("kind") == "em_level" and dec[0] == "select":
            if _find_entry(runs, d["array"], dec[1]) is None:
                return (f"selected E/M {dec[1]} has no entry in any "
                        f"stored run")
        elif d.get("kind") == "presence" and dec[0] == "include":
            code = str(d.get("code") or "").upper()
            if _find_entry(runs, d["array"], code) is None:
                return (f"included code {code} has no entry in any "
                        f"stored run (stale consistency report?)")
    return None


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def adjudicate(results_dir: Path, docs: list[str] | None = None,
               rep: Replayer | None = None, dry_run: bool = False,
               passes: int = ADJUDICATION_PASSES) -> dict:
    """Adjudicate every (scoped) non-unanimous note. Returns
    {"considered", "adjudicated", "abstained", "split_verdicts",
     "failed_replay", "docs": {...}}."""
    from app.compliance.agents import build_default_agents
    from app.compliance.engine import ClaimScrubber
    from app.validation.consistency import (annotate_result, compare_runs,
                                            select_canonical)
    from tools.replay_reconcile import _rebuild_run

    stats = {"considered": 0, "adjudicated": 0, "abstained": 0,
             "split_verdicts": 0, "failed_replay": 0, "docs": {}}
    targets = []
    for f in sorted(results_dir.glob("*_results.json")):
        if f.name == "all_results.json":
            continue
        doc = f.stem.removesuffix("_results")
        if docs is not None and doc not in docs:
            continue
        try:
            main = json.loads(f.read_text())
        except Exception:
            continue
        if not isinstance(main, dict):
            continue
        cons = main.get("consistency") or {}
        if cons and not cons.get("unanimous"):
            targets.append((doc, f, main))
    if not targets:
        return stats
    _ensure_policy_corpus()
    if len(targets) > ADJUDICATION_LIMIT:
        logger.warning(
            f"Adjudication capped at {ADJUDICATION_LIMIT} of "
            f"{len(targets)} holdout(s) (CODER_ADJUDICATION_LIMIT); "
            f"the rest keep their deferred/review status")
        targets = targets[:ADJUDICATION_LIMIT]

    rep = rep or Replayer()
    scrubber = ClaimScrubber(rep.store,
                             agents=build_default_agents(rep.store))
    for doc, f, main in targets:
        case = assemble_case(doc, results_dir, rep)
        if case is None:
            stats["docs"][doc] = "no adjudicable billing disagreement"
            continue
        stats["considered"] += 1
        disputed = case["disputed_items"]
        logger.info(f"Adjudicating {doc}: {len(disputed)} disputed "
                    f"item(s), {passes} independent pass(es)")

        maps, verdicts = [], []
        for i in range(passes):
            try:
                v = _adjudicate_once(case, pass_idx=i)
            except Exception as exc:
                logger.warning(f"  pass {i + 1} failed: {exc}")
                maps.append(None)
                continue
            verdicts.append(v)
            maps.append(_verdict_map(v, disputed))
        if any(m is None for m in maps):
            stats["abstained"] += 1
            stats["docs"][doc] = ("abstained/incomplete verdict — human "
                                  "review stands")
            logger.info(f"  -> ABSTAINED (at least one pass did not fully "
                        f"ground every item)")
            continue
        if any(m != maps[0] for m in maps[1:]):
            stats["split_verdicts"] += 1
            stats["docs"][doc] = ("independent adjudications disagree — "
                                  "human review stands")
            logger.info(f"  -> SPLIT VERDICTS across {passes} passes")
            continue

        decisions = maps[0]
        runs = _load_runs(doc, results_dir)
        why_na = _decisions_applicable(decisions, disputed, runs)
        if why_na:
            stats["abstained"] += 1
            stats["docs"][doc] = f"verdict not realizable: {why_na}"
            logger.info(f"  -> verdict not realizable ({why_na})")
            continue
        # replay against the FULL note text (the case file's copy is
        # truncated for the prompt; the validator must see what the
        # original pipeline run saw)
        note = _note_text_for(doc, results_dir, runs, main)
        try:
            aligned = [_apply_to_run(run, decisions, disputed, runs)
                       for run in runs]
            # Deterministic guard: the verdicts must fully settle the
            # billing-compared content. If the applied runs' signatures
            # still differ, the adjudication did not decide the dispute —
            # void it, exactly like an abstention.
            sigs = {Replayer.signature(a.get("icd_codes"),
                                       a.get("cpt_codes"),
                                       a.get("hcpcs_codes"))
                    for a in aligned}
            if len(sigs) > 1:
                stats["abstained"] += 1
                stats["docs"][doc] = ("verdicts did not align the runs' "
                                      "billing content — human review "
                                      "stands")
                logger.info("  -> verdicts left billing signatures split")
                continue
            # ONE adjudicated claim. The runs' billing signatures now
            # agree, but uncompared per-run metadata (dx_pointers, line
            # descriptions) can still steer validator re-derivations
            # (e.g. primary-diagnosis designation from procedure linkage)
            # differently per run — variance the consistency gate never
            # measures and the adjudicator was never asked about. The
            # saved result has always been exactly one run's arrays
            # (select_canonical); adjudication standardizes on the same
            # canonical run BEFORE replay so the deterministic stack
            # judges the actual claim being submitted.
            canon = aligned[select_canonical(aligned)]
            for a in aligned:
                for arr in ("icd_codes", "cpt_codes", "hcpcs_codes",
                            "snomed_codes", "supporting_conditions"):
                    a[arr] = copy.deepcopy(canon.get(arr) or [])
            rebuilt = []
            for a in aligned:
                arrays, report = rep.replay_arrays(a, note)
                rebuilt.append(_rebuild_run(a, arrays, report,
                                            scrubber, note))
            new_report = compare_runs(rebuilt, store=rep.store)
        except Exception as exc:
            stats["failed_replay"] += 1
            stats["docs"][doc] = f"replay failed: {exc}"
            logger.warning(f"  -> replay failed: {exc}")
            continue
        if not new_report["unanimous"]:
            residue = [d for d in new_report["disagreements"]
                       if not d.get("advisory")]
            stats["failed_replay"] += 1
            stats["docs"][doc] = ("verdicts applied but replay still "
                                  "split — human review stands")
            logger.info(f"  -> replay still split after verdicts; "
                        f"residual disagreements: "
                        f"{json.dumps(residue, default=str)[:2000]}; "
                        f"dispositions={new_report['dispositions']}")
            continue

        if dry_run:
            stats["adjudicated"] += 1
            stats["docs"][doc] = "DRY RUN: would adjudicate"
            logger.info("  -> DRY RUN: adjudication would apply cleanly")
            continue

        idx = select_canonical(rebuilt)
        payload = annotate_result(rebuilt[idx], new_report)
        # annotate_result embeds via setdefault — force the FRESH report
        # so no stale key inside a stored run dump can survive into the
        # saved result
        payload["consistency"] = new_report
        payload["adjudication"] = {
            "at": datetime.now(timezone.utc).isoformat(),
            "model": verdicts[0].get("_model"),
            "passes": passes,
            "items": verdicts[0].get("items"),
            "overall_rationale": verdicts[0].get("overall_rationale", ""),
            "protocol": "authority-grounded expert-coder adjudication; "
                        "unanimous across independent passes; realigned "
                        "runs replayed through validator + scrubber",
        }
        # Clinical-correctness audit BEFORE the registry record: the
        # adjudicated claim's replay may carry interpretive layer
        # corrections (note-text-grounded auto-corrections), and those
        # must be clinically verified before this claim becomes verified
        # truth — a disputed correction forces the disposition to REVIEW
        # here, so the record below never happens.
        fresh_block = None
        if os.getenv("CLINICAL_AUDIT", "1") == "1":
            try:
                from tools.clinical_auditor import audit_result
                block = audit_result(doc, payload, note, rep)
                fresh_block = block
                if block.get("verdict") == "disputed":
                    logger.warning(
                        f"  clinical review disputed the adjudicated "
                        f"claim — routed to REVIEW")
            except Exception as exc:
                # fail closed for EVERY claim: the review is now the
                # universal CLEAN gate (whole-claim, not corrections-only),
                # so an unreviewed claim never becomes verified truth. The
                # result stays adjudicated on disk; the post-batch audit
                # retries and the registry ingest gate holds until it
                # upholds.
                payload["final_disposition"] = "REVIEW"
                payload["auto_coding_tier"] = "REVIEW"
                payload["auto_coding_review_reasons"] = (
                    list(payload.get("auto_coding_review_reasons") or [])
                    + [f"[clinical_audit/error] the clinical review could "
                       f"not run ({exc}) — claim unverified"])
                logger.warning(f"  clinical review failed ({exc}) — "
                               f"failing closed to REVIEW")

        # SURVIVAL INVARIANT (deterministic, runs AFTER the audit so no
        # upheld verdict can re-promote past it): every adjudicated
        # decision must be realized on the final claim. A replay layer
        # that overrode the verdict holds the claim and blocks the record.
        conflicts = _adjudication_conflicts(
            payload, decisions, disputed,
            (verdicts[0].get("items") if verdicts else None))
        if conflicts:
            _apply_override_hold(payload, conflicts)
            logger.warning(
                f"  -> {len(conflicts)} adjudicated decision(s) were "
                f"overridden by replay layers — held at REVIEW, "
                f"not recorded")

        f.write_text(json.dumps(payload, indent=2, default=str))
        stats["adjudicated"] += 1
        disp = payload.get("final_disposition", "")
        stats["docs"][doc] = f"adjudicated (disposition {disp})"
        logger.info(f"  -> ADJUDICATED: unanimous under verdicts, "
                    f"disposition {disp}")

        try:
            from tools.claims_registry import record_adjudicated
            if str(disp).upper() == "CLEAN" and not conflicts:
                record_adjudicated(doc, payload, f.name,
                                   by=f"coder-llm/{payload['adjudication']['model']}")
                logger.info("  -> registry: recorded (adjudicated tier)")
            else:
                # The full claim cannot be recorded (a replay override or
                # a disputed/held disposition) — record the settled
                # verdicts as scoped per-code targets instead, so the
                # layer-vs-adjudicator conflict that is blocking the note
                # gains the verified realignment goal actuation needs.
                _record_partial_targets(doc, f.name, payload, disputed,
                                        decisions, aligned, fresh_block)
        except Exception as exc:
            logger.warning(f"  registry record failed: {exc}")
    return stats


# --------------------------------------------------------------------------
# Adjudication survival — no layer silently outvotes the adjudicator
# --------------------------------------------------------------------------
#
# Measured live (routine_00008): the adjudicator removed modifier RT from
# 97597 citing the CPT descriptor (a per-session, total-wound-surface-area
# service has no laterality axis), the replay's deterministic modifier
# layer RE-ADDED it, the fresh review graded the recurrence advisory, and
# the claim shipped CLEAN carrying the modifier the authorities had just
# ruled inapplicable — and the registry anchored that claim, which then
# REJECTED corrective rules for the same defect on other notes. An LLM
# review pass can only flag; this is the deterministic invariant: every
# adjudicated decision must be REALIZED on the final claim, or the claim
# is held, never recorded, and the conflict is growth-queued.

def _norm_field_value(v):
    if isinstance(v, list):
        return sorted(str(x).upper() for x in v)
    return str(v)


def _adjudication_conflicts(payload: dict, decisions: dict,
                            disputed: list[dict],
                            items: list[dict] | None = None) -> list[dict]:
    """Every decision the adjudicator issued, checked against the FINAL
    claim arrays. Returns one conflict row per violated decision."""
    authority = {}
    for it in (items or []):
        authority[_item_key(it)] = str(it.get("authority") or "")[:300]

    def entry(array: str, code: str) -> dict | None:
        for e in payload.get(array) or []:
            if isinstance(e, dict) and \
                    str(e.get("code") or "").upper() == code:
                return e
        return None

    conflicts = []

    def conflict(d, decision: str, observed: str, code: str = "") -> None:
        conflicts.append({
            "array": d.get("array"),
            "code": code or str(d.get("code") or "").upper(),
            "kind": d.get("kind"),
            "decision": decision,
            "observed": observed,
            "authority": authority.get(_item_key(d), ""),
        })

    for d in disputed:
        dec = decisions.get(_item_key(d))
        if not dec:
            continue
        array = d.get("array")
        if d.get("kind") == "em_level":
            family = {str(c).upper() for c in (d.get("codes") or [])}
            present = {c for c in family if entry(array, c)}
            want = {dec[1]} if dec[0] == "select" else set()
            if present != want:
                conflict(d, f"family {sorted(family)} resolved to "
                            f"{sorted(want) or 'none'}",
                         f"claim carries {sorted(present) or 'none'}",
                         code=(dec[1] if dec[0] == "select"
                               else ",".join(sorted(family))))
        elif d.get("kind") == "presence":
            code = str(d.get("code") or "").upper()
            e = entry(array, code)
            if dec[0] == "include" and e is None:
                conflict(d, f"{code} belongs on the claim",
                         "absent from the final claim")
            elif dec[0] == "exclude" and e is not None:
                conflict(d, f"{code} does not belong on the claim",
                         "still on the final claim")
        elif d.get("kind") == "attributes":
            code = str(d.get("code") or "").upper()
            e = entry(array, code)
            if e is None:
                conflict(d, "attributes were adjudicated on this line",
                         "the line itself is gone from the final claim")
                continue
            for fname, fval in dict(dec[1]).items():
                try:
                    want = json.loads(fval)
                except (ValueError, TypeError):
                    want = fval
                if _norm_field_value(e.get(fname)) != _norm_field_value(want):
                    conflict(d, f"{fname} = {want!r}",
                             f"{fname} = {e.get(fname)!r}")
    return conflicts


def _apply_override_hold(payload: dict, conflicts: list[dict]) -> None:
    """A layer-vs-adjudicator conflict holds the claim at REVIEW with every
    violated decision named, stores the conflicts where the promotion path
    (clinical_auditor._enforce_verdict), the registry gate
    (eligible_for_auto), and the triage scan can all see them — three
    independent fail-closed checks, none of them an LLM."""
    payload.setdefault("adjudication", {})["overridden_by_replay"] = \
        conflicts
    payload["final_disposition"] = "REVIEW"
    payload["auto_coding_tier"] = "REVIEW"
    payload["auto_coding_confidence"] = min(
        float(payload.get("auto_coding_confidence") or 0.0), 0.84)
    marker = "[adjudication/overridden]"
    reasons = [r for r in (payload.get("auto_coding_review_reasons") or [])
               if marker not in str(r)]
    payload["auto_coding_review_reasons"] = reasons + [
        f"{marker} {c['array']}/{c['code']}: the replay's deterministic "
        f"layers produced {c['observed']} where the authority-grounded "
        f"verdict required {c['decision']} — layer-vs-adjudicator "
        f"conflict, human decision required" for c in conflicts]


def survival_conflicts_of(result: dict) -> list[dict]:
    """Reconstruct the adjudicated decisions stored on a saved result and
    re-verify them against its final claim. Empty list means every
    decision survived (or the record carries no adjudication)."""
    items = ((result.get("adjudication") or {}).get("items")
             if isinstance(result, dict) else None) or []
    disputed, decisions = [], {}
    for it in items:
        if not isinstance(it, dict):
            continue
        norm = _norm_decision(it)
        if norm is None:  # abstained items were never applied
            continue
        d = {k: it.get(k) for k in ("array", "code", "kind", "codes")
             if it.get(k) is not None}
        if it.get("kind") == "attributes":
            d["fields"] = sorted((it.get("fields") or {}).keys())
        disputed.append(d)
        decisions[_item_key(d)] = norm
    if not disputed:
        return []
    return _adjudication_conflicts(result, decisions, disputed, items)


def recheck_survival(results_dir: Path, docs: list[str] | None = None,
                     registry_path: Path | None = None) -> dict:
    """Retroactively apply the survival invariant to every saved result
    that carries an adjudication block: recompute conflicts from the
    stored verdict items against the claim as saved, hold any violator at
    REVIEW, and quarantine its adjudicated registry anchor (backed up
    first) so no corrective rule is ever gated against a claim the
    adjudicator did not actually produce. Deterministic, zero LLM."""
    import shutil

    stats = {"checked": 0, "conflicted": 0, "quarantined_records": 0,
             "docs": {}}
    conflicted_docs: list[str] = []
    for f in sorted(results_dir.glob("*_results.json")):
        if f.name == "all_results.json":
            continue
        doc = f.stem.removesuffix("_results")
        if docs is not None and doc not in docs:
            continue
        try:
            main = json.loads(f.read_text())
        except Exception:
            continue
        if not isinstance(main, dict) or \
                not (main.get("adjudication") or {}).get("items"):
            continue
        stats["checked"] += 1
        conflicts = survival_conflicts_of(main)
        if not conflicts:
            continue
        stats["conflicted"] += 1
        stats["docs"][doc] = [f"{c['code']}: required {c['decision']}, "
                              f"observed {c['observed']}"
                              for c in conflicts]
        _apply_override_hold(main, conflicts)
        f.write_text(json.dumps(main, indent=2, default=str))
        conflicted_docs.append(doc)
        logger.warning(f"Survival recheck {doc}: {len(conflicts)} "
                       f"overridden decision(s) — held at REVIEW")

    if conflicted_docs:
        from tools.claims_registry import REGISTRY_PATH, load_events
        path = registry_path or REGISTRY_PATH
        events = load_events(path)
        keep = [e for e in events
                if not (e.get("verification") == "adjudicated"
                        and e.get("document_id") in conflicted_docs)]
        removed = len(events) - len(keep)
        if removed:
            backup = path.parent / (
                path.name + ".bak_survival_"
                + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))
            shutil.copy(path, backup)
            with open(path, "w") as fh:
                for e in keep:
                    fh.write(json.dumps(e, sort_keys=True, default=str)
                             + "\n")
            stats["quarantined_records"] = removed
            logger.warning(f"Registry: quarantined {removed} adjudicated "
                           f"anchor(s) for overridden claims "
                           f"(backup: {backup.name})")
    return stats


# --------------------------------------------------------------------------
# Audit-dispute adjudication — settling the clinical review's findings
# --------------------------------------------------------------------------
#
# A consistency dispute and an audit dispute are different shapes of the
# same problem. Consistency disputes are REPEATABILITY failures: the runs
# disagree, and the adjudicator picks between them. Audit disputes are
# CORRECTNESS failures the runs can never expose — a deterministic layer's
# wrong decision is unanimous by construction — so the clinical review's
# grounded finding is the ALLEGATION, and this mode has the same expert
# coder decide it independently against the same authoritative sources.
# A confirmed decision is applied mechanically, replayed through the full
# deterministic stack, re-reviewed, and recorded at the adjudicated tier —
# which is exactly the verified realignment target the actuation queue's
# audit_dispute classes wait for. That closes the growth loop without a
# human in the path (a human record still outranks everything).

_AUDIT_MODE_SUPPLEMENT = """

AUDIT-DISPUTE MODE — this case differs from the standard one:
The pipeline's runs AGREE on this claim; there is no run-vs-run dispute.
Instead, an independent clinical review alleged specific defects, each
listed as a disputed item with its "allegation" (the reviewer's finding,
authority, and note evidence). Your job is to decide each item on the
authorities and the note ALONE — the allegation is a hypothesis to test,
not a verdict to ratify. Confirming an allegation and rejecting one are
equally correct outcomes when the authorities support them.
- "final_claim" is the claim as currently billed. For a presence item,
  "include" means the code belongs on the final claim and "exclude" means
  it does not — regardless of whether it is currently present.
- For an attributes item, only the listed "fields" are writable; "set"
  states the correct value for this claim (e.g. {"type": "primary"}).
- An item whose kind is "observable" disputes a measured system
  PHENOMENON — most commonly a compliance-scrubber ADVISORY (a
  non-blocking WARN finding), identified by the item's "observable"
  (the measurement namespace) and "key" (its machine identity; the
  phenomenon's full text rides in the allegation) — not any billed
  line: the claim is correct as billed and only the phenomenon's
  guidance is contested. Decide from the governing authority whether
  its requirement actually applies to THIS note's documented fact
  pattern:
    "suppress" — the authority recognizes a distinct pathway/exception
      the note affirmatively documents, so the phenomenon's demand is
      wrong here and it must not fire;
    "stand" — its requirement does apply; the reviewer's allegation is
      rejected.
  Respond for it as {"array", "code", "kind": "observable",
  "decision": "suppress" | "stand" | "abstain", "authority",
  "authority_basis", "authority_quote", "note_evidence"}. An observable decision changes
  NO claim line — it is recorded as a verified emission target for the
  deterministic stack. These verdicts almost always rest on prose
  policy (a coverage pathway, a documentation principle), so
  "authority_quote" (binding rule 7) matters most here: without the
  verbatim verified passage the target records at the weakest
  attestation tier and cannot anchor deterministic realignment.
- Abstain per item when the authorities genuinely do not decide it."""


def _array_of_code(c: str) -> str:
    if "." in c or (c[:1].isalpha() and not (len(c) == 5
                                             and c[1:].isdigit())):
        return "icd_codes"
    if c[:1].isalpha():
        return "hcpcs_codes"
    return "cpt_codes"


def _split_disagreement_keys(cons: dict) -> set[tuple]:
    """(array, CODE) for every billing code the consistency runs disagree
    about — the codes the unanimity machinery owns. Advisory variance
    never claims a code; em_level entries contribute every sibling."""
    keys: set[tuple] = set()
    for d in (cons.get("disagreements") or []):
        if not isinstance(d, dict) or d.get("advisory"):
            continue
        arr = str(d.get("array") or "")
        for c in (d.get("codes") or [d.get("code")]):
            if c:
                keys.add((arr, str(c).upper()))
    return keys


def _observable_for_finding(kind: str) -> dict | None:
    """The measurement observable (tools/observables.py) that can resolve
    a reviewer finding of this kind to a machine identity, or None. Kept
    as a wrapper so a broken observables module degrades to 'the finding
    stays residual' instead of sinking the adjudication."""
    try:
        from tools.observables import observable_for_finding
        return observable_for_finding(kind)
    except Exception as exc:
        logger.warning(f"observable lookup failed for {kind!r}: {exc}")
        return None


def _audit_disputed_items(main: dict) -> tuple[list[dict], list[str]]:
    """Translate the clinical review's disputed verdicts and routing-grade
    findings into the adjudicator's disputed-item vocabulary. Returns
    (mechanizable items, residual descriptions that no mechanical decision
    can realize — those keep the note in the human queue)."""
    from tools.clinical_auditor import material_corrections_of
    audit = main.get("clinical_audit") or {}
    mats = material_corrections_of(main)
    by_key: dict[tuple, dict] = {}
    residual: list[str] = []

    def push(item: dict) -> None:
        key = _item_key(item)
        cur = by_key.get(key)
        if cur is None:
            by_key[key] = item
        elif item.get("kind") == "attributes":
            # merge writable fields for repeat findings on the same code
            cur["fields"] = sorted(set(cur.get("fields") or [])
                                   | set(item.get("fields") or []))

    for it in (audit.get("items") or []):
        if str(it.get("verdict") or "").lower() == "uphold":
            continue
        try:
            m = mats[int(it.get("index"))]
        except (TypeError, ValueError, IndexError):
            continue
        if not m.get("interpretive"):
            continue
        code = str(m.get("code") or "").upper()
        allegation = {
            "source": "clinical_review/correction_verdict",
            "reviewer_verdict": it.get("verdict"),
            "the_systems_correction": str(m.get("message") or "")[:400],
            "authority": str(it.get("authority") or "")[:400],
            "note_evidence": str(it.get("note_evidence") or "")[:400],
        }
        if not code:
            residual.append(f"disputed correction without a code: "
                            f"{str(m.get('message') or '')[:160]}")
            continue
        action = str(m.get("action") or "")
        if "removal" in action or "addition" in action:
            push({"array": _array_of_code(code), "code": code,
                  "kind": "presence", "allegation": allegation})
        else:
            push({"array": _array_of_code(code), "code": code,
                  "kind": "attributes",
                  "fields": ["type", "modifiers", "units"],
                  "allegation": allegation})

    _FIELDS_BY_KIND = {"primary_designation": ["type"],
                       "modifier": ["modifiers"], "units": ["units"],
                       "linkage": ["linked_diagnoses"]}
    for fnd in (audit.get("claim_findings") or []):
        if not isinstance(fnd, dict):
            continue
        fkind = str(fnd.get("kind") or "").lower()
        obs = _observable_for_finding(fkind)
        if obs is not None:
            # The reviewer disputes a measured system PHENOMENON (e.g. a
            # scrubber advisory), not a claim line — the claim is correct
            # as billed. This IS mechanizable: the adjudicator rules on
            # the phenomenon's emission state and the verdict records as
            # a verified observable-emission target — the realignment
            # goal observable-shaped actuation waits for. Identity must
            # resolve deterministically to exactly ONE live phenomenon
            # (the observable's identify() contract), or the dispute
            # stays residual (never guess which phenomenon a prose
            # finding means).
            key, why = obs["identify"](main, fnd)
            code = str(fnd.get("code") or "").upper()
            if key is None:
                residual.append(f"{fkind} on {code or '(no code)'}: {why}")
                continue
            from tools.observables import code_of_key
            code = code_of_key(key) or code
            arr = str(fnd.get("array") or "")
            if arr not in _BILLING_ARRAYS:
                arr = _array_of_code(code)
            push({"array": arr, "code": code, "kind": "observable",
                  "observable": obs["name"], "key": key,
                  "allegation": {
                      "source": "clinical_review/claim_finding",
                      "kind": fkind,
                      "finding": str(fnd.get("finding") or "")[:400],
                      "authority": str(fnd.get("authority") or "")[:400],
                      "note_evidence":
                          str(fnd.get("note_evidence") or "")[:400],
                      "disputed_phenomenon": why,
                      "measured_by": obs["name"],
                  }})
            continue
        if fnd.get("materiality") not in ("billing_material", "uncertain"):
            continue  # advisory findings grow rules, never mutate billing
        code = str(fnd.get("code") or "").upper()
        arr = str(fnd.get("array") or "claim")
        kind = str(fnd.get("kind") or "other")
        allegation = {
            "source": "clinical_review/claim_finding",
            "kind": kind,
            "finding": str(fnd.get("finding") or "")[:400],
            "authority": str(fnd.get("authority") or "")[:400],
            "note_evidence": str(fnd.get("note_evidence") or "")[:400],
        }
        if not code or arr == "claim":
            residual.append(f"claim-level finding ({kind}): "
                            f"{str(fnd.get('finding') or '')[:160]}")
            continue
        if arr not in _BILLING_ARRAYS:
            arr = _array_of_code(code)
        if kind in _FIELDS_BY_KIND:
            push({"array": arr, "code": code, "kind": "attributes",
                  "fields": list(_FIELDS_BY_KIND[kind]),
                  "allegation": allegation})
        else:
            # wrong_code / missing_code / coverage / other — all reduce to
            # "does this code belong on this claim?"
            push({"array": arr, "code": code, "kind": "presence",
                  "allegation": allegation})
    return list(by_key.values()), residual


def _proposed_code_authoritative_ok(rep: Replayer, arr: str, code: str,
                                    main: dict, note_text: str
                                    ) -> tuple[bool, str]:
    """DETERMINISTIC authoritative validation for an audit-PROPOSED code —
    one no stored run bills, that a review verdict, a whole-claim finding, or
    an exploratory lead wants ADDED to the claim. The materialization would
    otherwise build the entry from the reference-DB descriptor on the
    reviewer's say-so alone; this gate grounds the fix in the DATA instead,
    so a hallucinated addition never reaches the claim:

      1. EXISTS — a real code with a real descriptor in the authoritative
         reference data.
      2. DESCRIPTOR MATCHES THE DOCUMENTED WORK — the code's own distinctive
         descriptor tokens overlap the note's documented procedures / text
         (the completeness invariant's grounding, run in the additive
         direction: a proposed procedure code with no footing in what the
         surgeon documented is refused).
      3. BILLABLE — its MUE is not 0 (a 0-MUE code is not separately
         reportable on a professional claim) and, for a CPT procedure, it
         carries a global-period assignment (a real procedure, not an
         unpriceable shell).
      4. NO UNBYPASSABLE NCCI CONFLICT — it is not the bundled (column-2)
         side of a hard PTP edit (modifier indicator 0) against a code
         already billed on the claim.

    Laterality and any residual PTP arrangement are enforced DOWNSTREAM when
    the materialized code is replayed through validate(); this gate covers
    the checks that replay cannot make (descriptor relevance) plus a fast
    fail-closed billability/NCCI pre-screen. Every value is queried from the
    authoritative data — no hardcoded codes. Fail-closed: a check that
    cannot run returns False. Returns (ok, reason)."""
    from app.validation.validator import CodingValidator as _V
    db = rep.db
    code = str(code or "").upper()
    if not code:
        return False, "empty code"
    # 1. EXISTS in the authoritative reference data, with a descriptor.
    if arr == "cpt_codes":
        rec = db.validate_cpt(code)
    elif arr == "hcpcs_codes":
        rec = db.validate_hcpcs(code)
    else:
        rec = (db.validate_icd10(code)
               if hasattr(db, "validate_icd10") else None)
    if not rec:
        return False, "not present in the authoritative reference data"
    desc = str(rec.get("long_description")
               or rec.get("description") or "").strip()
    if not desc:
        return False, "no authoritative descriptor"
    # A proposed DIAGNOSIS is grounded by the documented condition, which the
    # adjudicator's N-pass + authority-quote review judges; this gate only
    # confirms it is a real code (the procedure-descriptor match below does
    # not apply to a diagnosis).
    if arr == "icd_codes":
        return True, "diagnosis present in reference data"
    # 2. DESCRIPTOR MATCHES THE DOCUMENTED WORK. Matched on shared ROOTS, not
    # the suffix stemmer: a terse CPT descriptor ('Ostectomy, calcaneus') and
    # the surgeon's wording ('exostectomy', 'retrocalcaneal') share medical
    # roots ('ostectom', 'calcane') that suffix-stemming alone misses — the
    # same vocabulary gap the completeness invariant hit. A distinctive
    # descriptor token counts as grounded when its 6+char root appears inside
    # some documented-work token (or vice-versa). Lenient by design (need
    # ONE root, and the audit already supplies a mechanically-verified note
    # quote): this is a backstop against a wildly-unrelated proposed code,
    # not a second clinical judgment.
    documented = " ".join(
        str(p) for p in (main.get("procedures_performed_today") or [])
    ) + " " + (note_text or "")
    doc_toks = [t for t in _V._tokens(documented) if len(t) >= 4]
    desc_toks = [t for t in _V._tokens(desc)
                 if len(t) >= 5 and t not in _V._DESC_STOPWORDS]

    def _root_grounded(dt: str) -> bool:
        root = dt[:7] if len(dt) >= 8 else dt[:6] if len(dt) >= 6 else dt
        if len(root) < 5:
            return _V._stem(dt) in {_V._stem(o) for o in doc_toks}
        return any(root in ot or (len(ot) >= 6 and ot[:6] in dt)
                   for ot in doc_toks)

    hits = sum(1 for t in desc_toks if _root_grounded(t))
    need = 1 if len(desc_toks) <= 2 else 2
    if hits < need:
        return False, (f"descriptor {desc[:60]!r} is not grounded in the "
                       f"documented work ({hits}/{need} distinctive roots)")
    # 3. BILLABLE.
    mue = db.get_mue(code)
    if mue == 0:
        return False, "MUE is 0 — not separately reportable on this claim"
    if arr == "cpt_codes":
        try:
            gp = rep.store.global_period(code)
        except Exception:
            gp = None
        if not str(gp or "").strip():
            return False, "no global-period assignment — not an established " \
                          "billable procedure"
    # 4. NO UNBYPASSABLE NCCI CONFLICT with a code already on the claim.
    for e in (main.get("cpt_codes") or []):
        other = str(e.get("code") or "").upper() if isinstance(e, dict) else ""
        if not other or other == code:
            continue
        edit = db.check_ncci(other, code)
        if edit and str(edit.get("code2") or "").upper() == code \
                and str(edit.get("modifier") or "").strip() == "0":
            return False, (f"NCCI hard bundle (indicator 0): {code} is "
                           f"included in billed {other}, not separately "
                           f"reportable")
    return True, "validated against the authoritative reference data"


def _materialize_donor(rep: Replayer, main: dict, runs: list[dict],
                       items: list[dict], note_text: str = "") -> dict:
    """A synthetic 'run' carrying an entry for every disputed code that no
    stored run bills, so an 'include' decision has something mechanical to
    materialize from. Identity comes from the data, never invention: a
    demoted diagnosis is rebuilt from its supporting_conditions entry, and
    anything else from its reference-DB descriptor — but a genuinely NEW
    audit-proposed code (no run, no demoted-dx donor) materializes ONLY when
    it passes _proposed_code_authoritative_ok, so a reviewer cannot conjure a
    code that is not grounded in the authoritative data. A code that stays
    unmaterializable voids its decision (fail closed)."""
    donor = {"icd_codes": [], "cpt_codes": [], "hcpcs_codes": []}
    for d in items:
        code = str(d.get("code") or "").upper()
        arr = d.get("array")
        if not code or arr not in _BILLING_ARRAYS:
            continue
        if _find_entry(runs + [main], arr, code):
            continue
        ent = None
        if arr == "icd_codes":
            for src in [main] + runs:
                for e in (src.get("supporting_conditions") or []):
                    if isinstance(e, dict) and \
                            str(e.get("code") or "").upper() == code:
                        ent = {"code": code,
                               "description": e.get("description", ""),
                               "type": "secondary"}
                        break
                if ent:
                    break
        if ent is None:
            try:
                rows = _authoritative_evidence(rep, arr, [code])
                desc = (rows[0].get("descriptor") or "") if rows else ""
            except Exception:
                desc = ""
            if not desc:
                continue
            # AUDIT-PROPOSED code (no run bills it, no demoted-dx donor): it
            # may materialize ONLY if it validates against the authoritative
            # data — real code, descriptor matching the documented work,
            # billable, no unbypassable NCCI conflict. Anti-hallucination by
            # DATA: a proposal that fails is not materialized, so an 'include'
            # decision on it cannot realize and the item fails closed to a
            # human instead of putting an ungrounded code on the claim.
            try:
                ok, why = _proposed_code_authoritative_ok(
                    rep, arr, code, main, note_text)
            except Exception as exc:  # unverifiable ≠ valid (fail closed)
                ok, why = False, f"validation could not run ({exc})"
            if not ok:
                logger.info(f"  audit-proposed {code} refused "
                            f"materialization — {why}")
                continue
            logger.info(f"  audit-proposed {code} validated against "
                        f"authoritative data — {why}")
            if arr == "icd_codes":
                ent = {"code": code, "description": desc,
                       "type": "secondary"}
            else:
                ent = {"code": code, "description": desc,
                       "modifiers": [], "units": 1}
        donor[arr].append(ent)
    return donor


def _sig_row(array: str, entry: dict | None) -> dict | None:
    """One claim entry reduced to its billing-signature row (the exact
    normalization Replayer.signature applies) — the unit a per-code
    verified target freezes. None = the code is absent."""
    if not isinstance(entry, dict):
        return None
    row = {"code": str(entry.get("code") or "").strip().upper(),
           "modifiers": sorted(str(m) for m in (entry.get("modifiers") or [])
                               if m),
           "units": str(entry.get("units") or "")}
    if array == "icd_codes":
        row["type"] = str(entry.get("type") or "").strip().lower()
    return row


def _adjudicated_code_targets(disputed: list[dict], decisions: dict,
                              aligned_runs: list[dict]) -> list[dict]:
    """The adjudicated verdicts rendered as per-code claim targets:
    {"array", "code", "row"} where row is the exact billing row the
    verdict mandates (read from the runs AFTER mechanical application,
    BEFORE replay — the verdict itself, not whatever a replay layer left
    of it) or None when the verdict is that the code must be absent.
    A code whose applied row differs across runs yields no target: the
    verdict did not realize identically, so there is nothing verified to
    freeze."""
    def entry_of(run, array, code):
        for e in run.get(array) or []:
            if isinstance(e, dict) and \
                    str(e.get("code") or "").upper() == code:
                return e
        return None

    targets: dict[tuple, dict] = {}
    for d in disputed:
        if d.get("kind") in ("advisory", "observable"):
            # an emission verdict verifies an EMISSION state, never the
            # claim row it fires on — freezing the row here would verify
            # content nobody adjudicated
            continue
        dec = decisions.get(_item_key(d))
        if not dec:
            continue
        array = d["array"]
        codes = ([str(c).upper() for c in (d.get("codes") or [])]
                 if d.get("kind") == "em_level"
                 else [str(d.get("code") or "").upper()])
        for code in codes:
            rows = [_sig_row(array, entry_of(run, array, code))
                    for run in aligned_runs]
            if not rows or any(r != rows[0] for r in rows[1:]):
                continue
            # a code under several verdicts (presence + attributes, the
            # routine_00001/27654 shape) yields ONE target — both read
            # the same applied entry
            targets[(array, code)] = {"array": array, "code": code,
                                      "row": rows[0]}
    return list(targets.values())


def _fresh_review_contradicts(block: dict, disputed: list[dict],
                              decisions: dict, payload: dict) -> set[tuple]:
    """(array, CODE) adjudicated keys the FRESH post-adjudication review
    sides AGAINST. A reviewer-vs-adjudicator disagreement is a genuine
    human case — such a code must never become a verified target. Two
    contradiction sources: (a) a non-advisory whole-claim finding on the
    code pointing the opposite way from the verdict (a missing_code
    finding AGREES with an include verdict a layer stripped; a
    wrong_code/coverage finding AGREES with an exclude verdict still on
    the claim; everything else disputes it), and (b) an UPHELD correction
    verdict endorsing the removal of a code the adjudicator ruled present
    and that is indeed absent from the claim."""
    directions: dict[tuple, str] = {}
    for d in disputed:
        if d.get("kind") in ("advisory", "observable"):
            continue  # no claim-line direction — nothing to contradict
        dec = decisions.get(_item_key(d))
        if not dec:
            continue
        if d.get("kind") == "em_level":
            chosen = dec[1] if dec[0] == "select" else None
            for c in (d.get("codes") or []):
                c = str(c).upper()
                directions[(d["array"], c)] = ("present" if c == chosen
                                               else "absent")
        elif d["kind"] == "presence":
            directions[(d["array"], str(d["code"]).upper())] = (
                "present" if dec[0] == "include" else "absent")
        else:  # attributes — the verdict presumes the line stays billed
            directions[(d["array"], str(d["code"]).upper())] = "present"

    out: set[tuple] = set()
    for f in (block.get("claim_findings") or []):
        if not isinstance(f, dict) or f.get("materiality") == "advisory":
            continue
        key = (str(f.get("array") or ""), str(f.get("code") or "").upper())
        want = directions.get(key)
        if want is None:
            continue
        kind = str(f.get("kind") or "").lower()
        agrees = (kind == "missing_code" if want == "present"
                  else kind in ("wrong_code", "coverage"))
        if not agrees:
            out.add(key)

    from tools.clinical_auditor import material_corrections_of
    mats = material_corrections_of(payload)  # the list block items index
    billed = {(a, str(e.get("code") or "").upper())
              for a in _BILLING_ARRAYS
              for e in (payload.get(a) or []) if isinstance(e, dict)}
    for item in (block.get("items") or []):
        if str(item.get("verdict") or "").lower() != "uphold":
            continue
        try:
            m = mats[int(item.get("index"))]
        except (TypeError, ValueError, IndexError):
            continue
        code = str(m.get("code") or "").upper()
        for (arr, c), want in directions.items():
            if c == code and want == "present" and (arr, c) not in billed:
                out.add((arr, c))
    return out


_TIER_RANK = {"document_quoted": 3, "data_backed": 2, "unverified": 1,
              "attested_only": 0}


def _stamp_attestation(targets: list[dict], items: list) -> None:
    """Stamp each per-code target with the weakest attestation tier of
    the adjudication items that decided its code — the registry accessors
    (claims_registry._anchorable) refuse attested_only targets, so a
    prose verdict without a verified policy quote can be recorded and
    audited but never anchors actuation. A target no item matches (should
    not happen; targets derive from decisions, decisions from items) gets
    the no-quote tier — conservative, never silently strong."""
    for t in targets:
        code = str(t.get("code") or "").upper()
        arr = str(t.get("array") or "")
        tiers = []
        for it in items or []:
            if not isinstance(it, dict) or str(it.get("array")) != arr:
                continue
            codes = {str(it.get("code") or "").upper()} | \
                {str(c).upper() for c in (it.get("codes") or [])}
            if code in codes:
                tiers.append(_attestation_of(it))
        t["attestation"] = (min(tiers, key=lambda x: _TIER_RANK.get(x, 0))
                            if tiers else _attestation_of(None))


def _record_partial_targets(doc: str, source: str, payload: dict,
                            disputed: list[dict], decisions: dict,
                            aligned_runs: list[dict],
                            fresh_block: dict | None) -> None:
    """Record the adjudicated verdicts as PER-CODE verified targets when
    the note cannot be recorded whole (split runs, a replay override, or
    an unrelated residual dispute). The verdicts themselves are settled —
    unanimous across independent passes, authority-grounded, mechanically
    realized identically on every run — and recording them scoped gives
    the matching audit-dispute classes the realignment goal actuation
    needs. This breaks the deadlock where a wrong deterministic rule can
    only be fixed against a verified target, but the note can never
    verify while that same rule keeps overriding the verdict (measured
    live, routine_00001/27654). Fail closed: no fresh review, no targets;
    a code the fresh review sides AGAINST the adjudicator on is excluded
    (that disagreement is a human case, never a target)."""
    if fresh_block is None:
        return
    try:
        contested = _fresh_review_contradicts(
            fresh_block, disputed, decisions, payload)
        targets = [t for t in _adjudicated_code_targets(
                       disputed, decisions, aligned_runs)
                   if (t["array"], t["code"]) not in contested]
        _stamp_attestation(
            targets, (payload.get("adjudication") or {}).get("items"))
        weak = [t for t in targets if t["attestation"] == "attested_only"]
        if weak:
            logger.info(
                f"  -> {len(weak)} per-code target(s) recorded at "
                f"attested_only (prose verdict without a verified policy "
                f"quote) — visible for audit, excluded from actuation "
                f"anchoring: "
                + ", ".join(f"{t['array']}/{t['code']}" for t in weak))
        if targets:
            from tools.claims_registry import record_adjudicated_codes
            ev = record_adjudicated_codes(
                doc, targets, source,
                by=f"coder-llm/{payload['adjudication']['model']}",
                adjudication=payload["adjudication"])
            if ev:
                logger.info(
                    f"  -> registry: {len(targets)} per-code verified "
                    f"target(s) recorded — scoped realignment goal(s) "
                    f"for audit_dispute actuation")
        if contested:
            logger.info(
                f"  -> {len(contested)} adjudicated code(s) contested by "
                f"the fresh review — no target recorded for: "
                + ", ".join(f"{a}/{c}" for a, c in sorted(contested)))
    except Exception as exc:
        logger.warning(f"  per-code target record failed: {exc}")


def _record_observable_targets(doc: str, source: str, payload: dict,
                               disputed: list[dict],
                               decisions: dict) -> None:
    """Record the adjudicated emission verdicts as verified observable
    targets. An emission verdict can NEVER be realized mechanically at
    adjudication time — the measured surface (e.g. the scrubber)
    recomputes its findings from data, so a 'suppress' verdict only
    materializes once actuation mints the deterministic rule that
    satisfies it. Recording the target is therefore the verdict's ONLY
    realization path, and the emission-aware replay gate uses it as the
    convergence goal. Both directions record: suppress (emit=false)
    gives the class its realignment goal; stand (emit=true) closes the
    class at baseline — the phenomenon already fires, which now IS the
    verified state."""
    targets = []
    for d in disputed:
        if d.get("kind") not in ("advisory", "observable"):
            continue
        dec = decisions.get(_item_key(d))
        if not dec or dec[0] not in ("suppress", "stand"):
            continue
        auth, tier = "", None
        for it in ((payload.get("adjudication") or {}).get("items") or []):
            if isinstance(it, dict) and _item_key(it) == _item_key(d):
                auth = str(it.get("authority") or "")[:300]
                tier = _attestation_of(it)
        key = str(d.get("key") or "")
        if not key and d.get("filter_id"):  # pre-generalization item shape
            key = f"{d['filter_id']}|{str(d.get('code') or '').upper()}"
        if not key:
            continue
        targets.append({"observable":
                        str(d.get("observable") or "advisory_emission"),
                        "key": key,
                        "emit": dec[0] == "stand",
                        "authority": auth,
                        "attestation": tier or _attestation_of(None)})
    if not targets:
        return
    weak = [t for t in targets if t["attestation"] == "attested_only"]
    if weak:
        logger.info(
            f"  -> {len(weak)} observable target(s) recorded at "
            f"attested_only (prose verdict without a verified policy "
            f"quote) — visible for audit, excluded from actuation "
            f"anchoring: " + ", ".join(t["key"] for t in weak))
    try:
        from tools.claims_registry import record_adjudicated_observables
        ev = record_adjudicated_observables(
            doc, targets, source,
            by=f"coder-llm/{payload['adjudication']['model']}",
            adjudication=payload["adjudication"])
        if ev:
            logger.info(
                f"  -> registry: {len(targets)} verified observable-"
                f"emission target(s) recorded — emission-state realignment "
                f"goal(s) for observable-shaped audit_dispute actuation")
    except Exception as exc:
        logger.warning(f"  observable target record failed: {exc}")


def _enforce_single_primary(run: dict, decisions: dict,
                            disputed: list[dict]) -> None:
    """ICD-10-CM I.B/IV.G: one first-listed diagnosis. When a verdict set
    a code's type to primary, every OTHER primary demotes to secondary —
    mechanical bookkeeping of the verdict, not a new judgment."""
    promoted = set()
    for d in disputed:
        if d.get("kind") != "attributes" or d.get("array") != "icd_codes":
            continue
        dec = decisions.get(_item_key(d))
        if not dec or dec[0] != "set":
            continue
        fields = dict(dec[1])
        try:
            if json.loads(fields.get("type", '""')) == "primary":
                promoted.add(str(d.get("code") or "").upper())
        except (ValueError, TypeError):
            continue
    if not promoted:
        return
    entries = [e for e in (run.get("icd_codes") or [])
               if isinstance(e, dict)]
    for e in entries:
        if str(e.get("type") or "").lower() == "primary" and \
                str(e.get("code") or "").upper() not in promoted:
            e["type"] = "secondary"
    # first-listed means first: the promoted code leads the array
    entries.sort(key=lambda e: 0 if str(e.get("code") or "").upper()
                 in promoted else 1)
    run["icd_codes"] = entries


def adjudicate_audit(results_dir: Path, docs: list[str] | None = None,
                     rep: Replayer | None = None, dry_run: bool = False,
                     passes: int = ADJUDICATION_PASSES) -> dict:
    """Adjudicate every (scoped) note whose clinical review disputed the
    claim. Confirmed decisions are applied mechanically, replayed through
    the full deterministic stack, re-reviewed, and — when the fresh review
    upholds — recorded at the adjudicated tier, giving the actuation
    queue's audit_dispute classes their verified realignment target.

    Jurisdiction on consistency holdouts: the codes the runs disagree
    about stay with the unanimity machinery, but review findings about
    codes every run AGREES on are adjudicated here too — a wrong
    deterministic decision is unanimous by construction, so no amount of
    consistency work can ever decide it. Such a partial adjudication
    corrects the claim in place; the note stays routed and is never
    promoted or registry-recorded from this path."""
    from app.compliance.agents import build_default_agents
    from app.compliance.engine import ClaimScrubber
    from app.validation.consistency import (annotate_result, compare_runs,
                                            select_canonical)
    from tools.clinical_auditor import audit_result
    from tools.replay_reconcile import _rebuild_run

    stats = {"considered": 0, "adjudicated": 0, "partial": 0,
             "abstained": 0, "split_verdicts": 0, "failed_replay": 0,
             "still_disputed": 0, "docs": {}}
    targets = []
    for f in sorted(results_dir.glob("*_results.json")):
        if f.name == "all_results.json":
            continue
        doc = f.stem.removesuffix("_results")
        if docs is not None and doc not in docs:
            continue
        try:
            main = json.loads(f.read_text())
        except Exception:
            continue
        if not isinstance(main, dict):
            continue
        if (main.get("clinical_audit") or {}).get("verdict") != "disputed":
            continue
        targets.append((doc, f, main))
    if not targets:
        return stats
    _ensure_policy_corpus()
    if len(targets) > ADJUDICATION_LIMIT:
        logger.warning(
            f"Audit adjudication capped at {ADJUDICATION_LIMIT} of "
            f"{len(targets)} disputed note(s) (CODER_ADJUDICATION_LIMIT)")
        targets = targets[:ADJUDICATION_LIMIT]

    rep = rep or Replayer()
    scrubber = ClaimScrubber(rep.store,
                             agents=build_default_agents(rep.store))
    for doc, f, main in targets:
        disputed, residual = _audit_disputed_items(main)
        cons = main.get("consistency") or {}
        unanimous_note = not cons or bool(cons.get("unanimous"))
        if not unanimous_note:
            # A consistency holdout is NOT skipped wholesale: the codes the
            # runs disagree about belong to the unanimity machinery
            # (adjudicate() above — mutating them here would paper over the
            # repeatability failure), but the review's findings about codes
            # every run AGREES on are correctness failures that are
            # unanimous by construction — the unanimity machinery can never
            # decide them, and the blanket skip stranded them with nobody
            # ruling (measured live, routine_00001: the A4570 MUE-0 and
            # missing-27654 findings sat unadjudicated because OTHER codes
            # were split). Those items are adjudicated here; the note stays
            # routed for its run disagreement and is never promoted or
            # registry-recorded from this path.
            split_keys = _split_disagreement_keys(cons)
            deferred = [d for d in disputed
                        if (d["array"], d["code"]) in split_keys]
            disputed = [d for d in disputed
                        if (d["array"], d["code"]) not in split_keys]
            residual = residual + [
                f"{d['array']}/{d['code']}: part of the run disagreement "
                f"— the consistency machinery owns it" for d in deferred]
        if not disputed:
            stats["docs"][doc] = ("no mechanizable disputed item — human "
                                  "review stands"
                                  + (f" ({len(residual)} claim-level)"
                                     if residual else ""))
            continue
        runs = _load_runs(doc, results_dir)
        note = _note_text_for(doc, results_dir, runs or [main], main)
        if not note:
            stats["docs"][doc] = "note text unavailable — cannot adjudicate"
            continue
        stats["considered"] += 1
        logger.info(f"Audit adjudication {doc}: {len(disputed)} disputed "
                    f"item(s) from the clinical review, {passes} "
                    f"independent pass(es)")

        donor = _materialize_donor(rep, main, runs, disputed, note_text=note)
        disputed_by_array: dict[str, set[str]] = {}
        for d in disputed:
            disputed_by_array.setdefault(d["array"], set()).add(d["code"])
        evidence = {arr: _authoritative_evidence(rep, arr, sorted(codes))
                    for arr, codes in disputed_by_array.items()}
        proc_disputed = (disputed_by_array.get("cpt_codes", set())
                         | disputed_by_array.get("hcpcs_codes", set()))
        audit = main.get("clinical_audit") or {}
        case = {
            "document_id": doc,
            "note_text": note[:12000],
            "final_claim": _claim_lines(main),
            "supporting_conditions_not_billed": [
                {k: e.get(k) for k in ("code", "description", "type",
                                       "review_reason") if e.get(k)}
                for e in (main.get("supporting_conditions") or [])
                if isinstance(e, dict)],
            "disputed_items": disputed,
            "clinical_review_context": {
                "overall_rationale": audit.get("overall_rationale", ""),
                "claim_level_concerns": audit.get("claim_level_concerns",
                                                  ""),
            },
            "authoritative_reference_data": evidence,
            "ncci_ptp_edits_on_this_claim":
                _ptp_evidence(rep, [main], proc_disputed),
            "quotable_policy_sources": _quotable_sources(),
        }
        anchor = _registry_verified_claims().get(doc)
        if anchor:
            case["registry_verified_claim"] = _sig_view(anchor)

        maps, verdicts = [], []
        for i in range(passes):
            try:
                v = _adjudicate_once(case, pass_idx=i,
                                     system_suffix=_AUDIT_MODE_SUPPLEMENT)
            except Exception as exc:
                logger.warning(f"  pass {i + 1} failed: {exc}")
                maps.append(None)
                continue
            verdicts.append(v)
            maps.append(_verdict_map(v, disputed))
        if any(m is None for m in maps):
            stats["abstained"] += 1
            stats["docs"][doc] = ("abstained/incomplete verdict — human "
                                  "review stands")
            logger.info("  -> ABSTAINED (at least one pass did not fully "
                        "ground every item)")
            continue
        if any(m != maps[0] for m in maps[1:]):
            stats["split_verdicts"] += 1
            stats["docs"][doc] = ("independent adjudications disagree — "
                                  "human review stands")
            logger.info(f"  -> SPLIT VERDICTS across {passes} passes")
            continue

        decisions = maps[0]
        source_runs = (runs if len(runs) >= 2 else [main])
        lookup_runs = source_runs + [donor]
        why_na = _decisions_applicable(decisions, disputed, lookup_runs)
        if why_na:
            stats["abstained"] += 1
            stats["docs"][doc] = f"verdict not realizable: {why_na}"
            logger.info(f"  -> verdict not realizable ({why_na})")
            continue

        if dry_run:
            stats["adjudicated"] += 1
            stats["docs"][doc] = "DRY RUN: would adjudicate"
            continue

        try:
            aligned = [_apply_to_run(run, decisions, disputed, lookup_runs)
                       for run in source_runs]
            for a in aligned:
                _enforce_single_primary(a, decisions, disputed)
            if unanimous_note:
                # the runs already agree, and every run received the
                # identical verdicts — canonicalize anyway so replay
                # judges one claim. NEVER done on a split note: copying
                # one run's arrays over the others would fabricate the
                # unanimity the runs don't have.
                canon = aligned[select_canonical(aligned)]
                for a in aligned:
                    for arr in ("icd_codes", "cpt_codes", "hcpcs_codes",
                                "snomed_codes", "supporting_conditions"):
                        a[arr] = copy.deepcopy(canon.get(arr) or [])
            rebuilt = []
            for a in aligned:
                arrays, report = rep.replay_arrays(a, note)
                rebuilt.append(_rebuild_run(a, arrays, report,
                                            scrubber, note))
            if len(rebuilt) >= 2:
                new_report = compare_runs(rebuilt, store=rep.store)
            else:
                new_report = main.get("consistency") or {}
        except Exception as exc:
            stats["failed_replay"] += 1
            stats["docs"][doc] = f"replay failed: {exc}"
            logger.warning(f"  -> replay failed: {exc}")
            continue
        if unanimous_note and len(rebuilt) >= 2 \
                and not new_report.get("unanimous"):
            stats["failed_replay"] += 1
            stats["docs"][doc] = ("verdicts applied but replay split the "
                                  "runs — human review stands")
            logger.info("  -> replay split the runs after verdicts")
            continue
        if not unanimous_note and len(rebuilt) >= 2:
            # a split note stays split on ITS codes — but the adjudicated
            # items themselves must replay identically across every run,
            # or the verdict was not realized deterministically
            replay_split = {(str(d.get("array") or ""), str(c).upper())
                            for d in (new_report.get("disagreements") or [])
                            if isinstance(d, dict) and not d.get("advisory")
                            for c in (d.get("codes") or [d.get("code")])
                            if c}
            adj_keys = {(d["array"], d["code"]) for d in disputed}
            if adj_keys & replay_split:
                stats["failed_replay"] += 1
                stats["docs"][doc] = (
                    "adjudicated items did not replay identically across "
                    "the split runs — human review stands")
                logger.info("  -> adjudicated items split on replay")
                continue

        idx = select_canonical(rebuilt) if len(rebuilt) >= 2 else 0
        if len(rebuilt) >= 2:
            payload = annotate_result(rebuilt[idx], new_report)
            payload["consistency"] = new_report
        else:
            payload = rebuilt[idx]
            if new_report:
                payload["consistency"] = new_report
        payload["adjudication"] = {
            "at": datetime.now(timezone.utc).isoformat(),
            "model": verdicts[0].get("_model"),
            "passes": passes,
            "items": verdicts[0].get("items"),
            "overall_rationale": verdicts[0].get("overall_rationale", ""),
            "protocol": "authority-grounded expert-coder adjudication of "
                        "clinical-review disputes; unanimous across "
                        "independent passes; applied mechanically to the "
                        "disputed items only; replayed through validator "
                        "+ scrubber; re-reviewed before recording",
            "residual_unmechanizable": residual,
        }
        # SURVIVAL INVARIANT first (deterministic): if a replay layer
        # overrode any adjudicated decision, the claim is held regardless
        # of what the review says — the review grades clinical substance,
        # not verdict fidelity, and it demonstrably (routine_00008) graded
        # a re-added modifier as advisory.
        conflicts = _adjudication_conflicts(
            payload, decisions, disputed,
            (verdicts[0].get("items") if verdicts else None))
        if conflicts:
            _apply_override_hold(payload, conflicts)
            logger.warning(
                f"  -> {len(conflicts)} adjudicated decision(s) were "
                f"overridden by replay layers — held at REVIEW")

        # The FRESH clinical review is the gate: the adjudicated claim is
        # a new claim, and only an upheld whole-claim review of it may
        # promote and record. A repeat dispute means the reviewer and the
        # adjudicator disagree — that is a genuine human case.
        promoted = False
        fresh_block = None
        try:
            block = audit_result(doc, payload, note, rep)
            fresh_block = block
            # a split note is never promoted from this path — its run
            # disagreement still stands regardless of what the fresh
            # review says about the adjudicated items
            promoted = (block.get("verdict") == "upheld"
                        and not conflicts and unanimous_note)
        except Exception as exc:
            payload["final_disposition"] = "REVIEW"
            payload["auto_coding_tier"] = "REVIEW"
            payload["auto_coding_review_reasons"] = (
                list(payload.get("auto_coding_review_reasons") or [])
                + [f"[clinical_audit/error] the post-adjudication review "
                   f"could not run ({exc}) — claim unverified"])
            logger.warning(f"  post-adjudication review failed ({exc}) — "
                           f"failing closed to REVIEW")
        if residual and promoted:
            # a claim-level allegation no mechanical decision could touch
            # is still unresolved — the note stays with a human even if
            # the re-review upholds the mutated items
            promoted = False
            payload["final_disposition"] = "REVIEW"
            payload["auto_coding_tier"] = "REVIEW"
            payload["auto_coding_review_reasons"] = (
                list(payload.get("auto_coding_review_reasons") or [])
                + [f"[audit_adjudication/residual] {r}" for r in residual])

        f.write_text(json.dumps(payload, indent=2, default=str))
        # Observable-emission verdicts record in EVERY outcome: they
        # cannot be realized mechanically (the measured surface recomputes
        # its findings, so a 'suppress' verdict only materializes once
        # actuation mints the rule) — the target IS the verdict's
        # realization path, and a promoted/held disposition changes
        # nothing about that.
        _record_observable_targets(doc, f.name, payload, disputed,
                                   decisions)
        disp = payload.get("final_disposition", "")
        if promoted and str(disp).upper() == "CLEAN":
            stats["adjudicated"] += 1
            stats["docs"][doc] = f"adjudicated (disposition {disp})"
            logger.info(f"  -> ADJUDICATED: review upholds the corrected "
                        f"claim, disposition {disp}")
            try:
                from tools.claims_registry import record_adjudicated
                record_adjudicated(
                    doc, payload, f.name,
                    by=f"coder-llm/{payload['adjudication']['model']}")
                logger.info("  -> registry: recorded (adjudicated tier) — "
                            "realignment target for audit_dispute classes")
            except Exception as exc:
                logger.warning(f"  registry record failed: {exc}")
            continue

        _record_partial_targets(doc, f.name, payload, disputed, decisions,
                                aligned, fresh_block)

        if not unanimous_note:
            stats["partial"] += 1
            stats["docs"][doc] = (
                f"{len(disputed)} review finding(s) adjudicated on the "
                f"codes the runs agree on; note stays routed for its run "
                f"disagreement")
            logger.info(f"  -> PARTIAL: {len(disputed)} finding(s) "
                        f"realized on the agreed codes; run disagreement "
                        f"stays with the consistency machinery")
        else:
            stats["still_disputed"] += 1
            stats["docs"][doc] = ("adjudicated but the fresh review still "
                                  "withholds CLEAN — human review stands")
            logger.info("  -> adjudicated, but the claim remains held "
                        "(review dispute or residual claim-level item)")
    return stats


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results_dir", nargs="?", default=str(DEFAULT_RESULTS))
    p.add_argument("--docs", default="",
                   help="comma-separated note stems to restrict to")
    p.add_argument("--audit-disputes", action="store_true",
                   help="adjudicate clinical-review disputes instead of "
                        "consistency holdouts")
    p.add_argument("--recheck-survival", action="store_true",
                   help="deterministically re-verify every saved "
                        "adjudication against its final claim; hold and "
                        "de-anchor any claim a replay layer overrode")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    docs = [s.strip() for s in args.docs.split(",") if s.strip()] or None
    if args.recheck_survival:
        stats = recheck_survival(Path(args.results_dir), docs=docs)
    else:
        fn = adjudicate_audit if args.audit_disputes else adjudicate
        stats = fn(Path(args.results_dir), docs=docs, dry_run=args.dry_run)
    print(json.dumps(stats, indent=2, default=str))


if __name__ == "__main__":
    main()
