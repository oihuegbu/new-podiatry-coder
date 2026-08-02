#!/usr/bin/env python3
"""Iterate batch -> triage -> actuation -> batch until every note is CLEAN.

This automates the convergence discipline used to take the first corpus to
10/10 unanimous, end to end — and keeps going until the claims are as clean
as autonomy can make them. Two nested loops:

INNER (batch iterations): each iteration reprocesses the holdouts with 3x
consistency (run.py already triages every disagreement and auto-actuates
declarative rules post-batch), then the next iteration runs with those
rules live. New flip classes surfaced by fresh LLM variance are captured
and actuated on the following pass. Iterates while unanimity or structure
still grows.

OUTER (convergence cycles): after the inner loop settles, finalization
runs — expert-coder adjudication, review routing, the audit-convergence
loop (clinical review disputes -> adjudication -> rules / amendments /
synthesized templates), registry ingest. Finalization itself MINTS
deterministic structure; any note that failed under the old stack deserves
a fresh generative pass against the new one (a template that didn't exist
when the note failed must get its shot at the note). So: if the cycle
changed the structure signature or raised the CLEAN count, and notes
remain non-CLEAN, the next cycle reprocesses exactly those notes fresh.

Termination (dynamic — no fixed cycle count):
  SUCCESS  — every note in scope is CLEAN (unanimous, scrub-clean, and the
             clinical review upholds the claim).
  STALL    — a full cycle neither moved the progress vector (CLEAN,
             unanimity, verified targets satisfied/recorded, splits,
             material findings) nor changed the deterministic structure
             (enabled rules + templates). Before stopping, one grace
             cycle is granted if anchored audit classes hold verified
             targets that actuation has not yet proposed against —
             adjudication records targets AFTER the cycle's actuation
             pass, so a stall verdict without the grace would strand
             actionable ground truth. What remains after that is
             judgment-shaped (the escalated human queue) or pure
             stochastic variance; more passes would only spend money.
  PATIENCE — --patience consecutive cycles minted structure yet never
             raised the CLEAN count. Structure growth justifies the next
             fresh pass, but not unboundedly: if new rules/templates keep
             landing while every claim stays non-CLEAN, the remaining
             blockers are not structure-shaped and a human should look.
  MAX      — only if --max-cycles is set (> 0); default is unlimited.

Runs inside the app container:
  docker compose run --rm app python tools/unanimity_loop.py \
      --start-at routine_podiatry_00001 --end-at routine_podiatry_00010 \
      --max-iters 4 --consistency 3 --workers 30
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RESULTS_DIR = ROOT / "output" / "results"
RULES_PATH = ROOT / "data" / "rules" / "validator_rules.json"
AUTO_TEMPLATES_DIR = ROOT / "data" / "rules" / "auto_templates"
GRADUATED_DIR = ROOT / "app" / "validation" / "graduated"


def _n_auto_rules() -> int:
    """Enabled auto rules only: a rule accepted then rolled back by the
    post-deployment audit is not progress, and counting it would let a
    stalled loop spend another iteration on the strength of a no-op."""
    pack = json.loads(RULES_PATH.read_text())
    return sum(1 for r in pack.get("rules", [])
               if r.get("auto_generated") and r.get("enabled", True))


def _n_auto_templates() -> int:
    """Self-authored template modules count as loop progress too: a fresh
    template means previously judgment-stalled escalations reopen next
    pass with a bigger vocabulary — stopping on that iteration's flat
    unanimity would abandon the very capability just built. Graduated
    templates count alongside sandboxed ones: graduation moves a module
    between the two directories mid-loop, and counting only one side
    would misread the move as a template appearing or vanishing."""
    n = 0
    if AUTO_TEMPLATES_DIR.exists():
        n += sum(1 for _ in AUTO_TEMPLATES_DIR.glob("*.py"))
    if GRADUATED_DIR.exists():
        n += sum(1 for f in GRADUATED_DIR.glob("*.py")
                 if f.name != "__init__.py")
    return n


def _structure_sig() -> tuple:
    """Identity of the live deterministic structure: the ENABLED rule ids
    plus every self-authored template module (sandboxed and graduated).
    Counting rules is not enough — an amendment disables the old rule and
    appends its replacement (net count zero), and a disable-only actuation
    lowers the count while still being progress. Any change to this
    signature means the stack a note failed under no longer exists, so the
    note has earned a fresh generative pass."""
    pack = json.loads(RULES_PATH.read_text())
    rules = tuple(sorted(str(r.get("id", "")) for r in pack.get("rules", [])
                         if r.get("enabled", True)))
    tmpls: list[str] = []
    if AUTO_TEMPLATES_DIR.exists():
        tmpls += [f.name for f in AUTO_TEMPLATES_DIR.glob("*.py")]
    if GRADUATED_DIR.exists():
        tmpls += [f.name for f in GRADUATED_DIR.glob("*.py")
                  if f.name != "__init__.py"]
    return rules, tuple(sorted(tmpls))


def _cleanliness(docs: list[str]) -> tuple[int, int, list[str]]:
    """(clean, total, non_clean_docs) by final_disposition — the loop's
    real goal. Unanimity is necessary but not sufficient: a unanimous
    claim the clinical review disputes is not done."""
    clean, non_clean = 0, []
    for doc in docs:
        f = RESULTS_DIR / f"{doc}_results.json"
        try:
            r = json.loads(f.read_text())
        except Exception:
            non_clean.append(doc)
            continue
        if str(r.get("final_disposition", "")).upper() == "CLEAN":
            clean += 1
        else:
            non_clean.append(doc)
    return clean, len(docs), non_clean


def _target_progress(docs: list[str]) -> tuple[int, int]:
    """(satisfied, recorded) over the scope's verified realignment targets
    — the registry-grounded progress measure. `recorded` counts every
    verified target the registry holds for these documents (whole-claim
    records, per-code rows, observable-emission verdicts): adjudication
    recording a new target IS progress, even when the claim itself hasn't
    moved yet — the next cycle's actuation is what realizes it (measured
    live on routine_00003, where cycle 2's targets landed AFTER that
    cycle's actuation pass and the loop stalled sitting on actionable
    ground truth). `satisfied` counts targets the CURRENT saved record
    already realizes (row present/absent as verified, emission matching
    the verdict) — monotone measures of convergence toward adjudicated
    truth that, unlike the review's finding count, cannot read a claim
    getting richer as regression."""
    from tools.auto_actuate import (Replayer, _advisory_targets,
                                    _code_rows, _per_code_targets,
                                    _realigns, _registry_verified_claims)
    from tools.observables import emission_of
    satisfied = recorded = 0
    registry = _registry_verified_claims()
    code_t = _per_code_targets()
    obs_t = _advisory_targets()
    for doc in docs:
        try:
            r = json.loads((RESULTS_DIR / f"{doc}_results.json")
                           .read_text())
        except Exception:
            r = None
        sig = (Replayer.signature(r.get("icd_codes"), r.get("cpt_codes"),
                                  r.get("hcpcs_codes"))
               if isinstance(r, dict) else None)
        goal = registry.get(doc)
        if goal is not None:
            recorded += 1
            if sig is not None and _realigns([sig], goal):
                satisfied += 1
        for (_array, code), row in (code_t.get(doc) or {}).items():
            recorded += 1
            if sig is None:
                continue
            rows = set(_code_rows(sig, {code}))
            if rows == ({row} if row is not None else set()):
                satisfied += 1
        goals = obs_t.get(doc) or {}
        if goals:
            recorded += len(goals)
            if isinstance(r, dict):
                by_obs: dict[str, set] = {}
                for (obs, key) in goals:
                    by_obs.setdefault(obs, set()).add(key)
                fires = emission_of(r, by_obs)
                satisfied += sum(
                    1 for k, emit in goals.items()
                    if fires.get(k) == emit
                    and not fires.get((k[0], "__error__")))
    return satisfied, recorded


def _progress_vector(docs: list[str]) -> tuple[int, ...]:
    """(clean, unanimous, targets_satisfied, targets_recorded,
    -billing_splits, -material_findings): every component so that HIGHER
    IS BETTER. A convergence cycle made progress when ANY component rose
    — the CLEAN count alone is too coarse: a single-note scope can never
    gain CLEAN until it is completely done, so patience keyed on CLEAN
    would silently cap a lone claim at --patience cycles, exactly the
    fixed budget this loop exists to remove.

    The two target components exist because the finding count alone
    INVERTS on a claim getting better (measured live, routine_00003
    cycle 2): a richer claim gives the reviewer more surface, so
    satisfying an adjudicated target (L60.8 landing on the claim) can
    RAISE the finding count while the claim converges. Targets satisfied
    and targets recorded are grounded, monotone-under-improvement
    measures that credit exactly that."""
    clean = unanimous = splits = findings = 0
    for doc in docs:
        try:
            r = json.loads((RESULTS_DIR / f"{doc}_results.json")
                           .read_text())
        except Exception:
            continue
        if str(r.get("final_disposition", "")).upper() == "CLEAN":
            clean += 1
        cons = r.get("consistency") or {}
        if cons.get("unanimous"):
            unanimous += 1
        splits += sum(1 for d in (cons.get("disagreements") or [])
                      if isinstance(d, dict) and not d.get("advisory"))
        ca = r.get("clinical_audit") or {}
        if str(ca.get("verdict") or "") == "disputed":
            mat = sum(1 for f in (ca.get("claim_findings") or [])
                      if isinstance(f, dict)
                      and str(f.get("materiality")) == "billing_material")
            # a disputed review with no parseable material findings still
            # counts: the dispute itself is the remaining blocker
            findings += mat or 1
    try:
        satisfied, recorded = _target_progress(docs)
    except Exception:
        satisfied = recorded = 0
    return clean, unanimous, satisfied, recorded, -splits, -findings


def _anchored_unworked_classes() -> list[str]:
    """Audit-dispute flip classes that hold a verified realignment target
    but have not been actuated — the classes a STALL verdict would strand.
    Targets can be recorded AFTER a cycle's actuation pass already ran
    (adjudication happens inside audit convergence, which follows
    actuation), so 'no structure and no progress' can coexist with
    actionable, anchored ground truth nobody has proposed against yet."""
    try:
        from tools import flip_triage
        from tools.auto_actuate import _audit_class_anchored
        return [c["class_key"] for c in flip_triage.load_queue()
                if c.get("kind") == "audit_dispute"
                and c.get("status") in ("open", "awaiting_verification")
                and _audit_class_anchored(c)]
    except Exception:
        return []


def _scope_docs(start_at: str, end_at: str) -> list[str]:
    docs = []
    for f in sorted(RESULTS_DIR.glob("*_results.json")):
        if f.name == "all_results.json":
            continue
        doc = f.stem.removesuffix("_results")
        if start_at and doc < start_at and not doc.startswith(start_at):
            continue
        if end_at and doc > end_at and not doc.startswith(end_at):
            continue
        docs.append(doc)
    return docs


def _unanimity(docs: list[str]) -> tuple[int, int, list[str]]:
    unanimous, holdouts = 0, []
    for doc in docs:
        f = RESULTS_DIR / f"{doc}_results.json"
        try:
            r = json.loads(f.read_text())
        except Exception:
            holdouts.append(doc)
            continue
        if (r.get("consistency") or {}).get("unanimous"):
            unanimous += 1
        else:
            holdouts.append(doc)
    return unanimous, len(docs), holdouts


def _final_audit_only(scope: list[str]) -> None:
    """The pre-convergence behavior: one clinical review pass, disputes
    stay held for a human. Used when AUDIT_CONVERGENCE=0 or the loop
    errors."""
    try:
        from tools.clinical_auditor import audit_batch
        caud = audit_batch(RESULTS_DIR, docs=scope)
        print(f"\nClinical audit (final): {caud['audited']} audited "
              f"({caud['upheld']} upheld, {caud['disputed']} "
              f"disputed), {caud['skipped']} unchanged/skipped",
              flush=True)
        for doc, msg in sorted(caud["docs"].items()):
            if "disputed" in msg:
                print(f"  [clinical-audit] {doc}: {msg}", flush=True)
    except Exception as exc:
        print(f"Clinical audit (final) failed ({exc})", flush=True)


def _run_inner_iterations(args, holdouts: list[str],
                          history: list) -> list[str]:
    """The batch loop: reprocess holdouts with fresh generative runs until
    100% unanimity, a stall, or --max-iters. Returns the remaining
    non-unanimous holdouts. (Extracted verbatim from the original single-
    cycle main; the outer convergence-cycle loop calls it per cycle.)"""
    prev_unanimous = -1
    prev_rules, prev_templates = _n_auto_rules(), _n_auto_templates()
    for it in range(1, args.max_iters + 1):
        target = (f"{len(holdouts)} holdout note(s)" if holdouts
                  else "full scope")
        print(f"\n{'=' * 66}\nITERATION {it}/{args.max_iters} — {target} "
              f"with {_n_auto_rules()} auto rule(s) live\n{'=' * 66}",
              flush=True)
        env = dict(os.environ,
                   AUTO_ACTUATE="1",
                   AUTO_ACTUATE_LIMIT=os.getenv("AUTO_ACTUATE_LIMIT", "12"),
                   # Interactive Claude calls during iteration: the Batch
                   # API's 50% discount costs queue+polling latency on every
                   # call, and iteration speed is the point here. Production
                   # batches keep the discount (their own env sets it).
                   ANTHROPIC_USE_BATCH="0",
                   # Disagreements stay OFF the human review queue while the
                   # loop is still working them — the next accepted rule may
                   # converge the note. Routing happens only at finalization
                   # below, when the loop stops without full unanimity.
                   DEFER_REVIEW_ROUTING="1",
                   # Same discipline for the clinical-correctness audit: a
                   # holdout's interpretive corrections change every pass,
                   # so auditing mid-loop re-spends the LLM on states that
                   # will change and lets a dispute be overwritten before
                   # the growth queue captures it. Audit ONCE at
                   # finalization on the settled claims (below).
                   DEFER_CLINICAL_AUDIT="1")
        # After the first pass, only reprocess the holdouts: unanimous
        # notes' results stand (acceptance gates prove new rules are inert
        # on them), so reprocessing them buys nothing but wall time.
        # Actuation scope stays the FULL corpus — a rule proposed off a
        # holdout must still prove inert on every unanimous note's stored
        # runs, or reprocessing-only-holdouts would blind the no-harm gate.
        select = ["--start-at", args.start_at, "--end-at", args.end_at]
        if holdouts:
            select = ["--only", ",".join(holdouts)]
            env["AUTO_ACTUATE_SCOPE"] = ",".join(
                _scope_docs(args.start_at, args.end_at))
        proc = subprocess.run(
            [sys.executable, "run.py",
             "--consistency", str(args.consistency),
             "--consistency-workers", str(args.workers), *select],
            cwd=ROOT, env=env)
        if proc.returncode != 0:
            print(f"ITERATION {it}: batch failed "
                  f"(exit {proc.returncode}) — stopping", flush=True)
            break

        docs = _scope_docs(args.start_at, args.end_at)
        unanimous, total, holdouts = _unanimity(docs)
        rules, templates = _n_auto_rules(), _n_auto_templates()
        gained_rules = rules - prev_rules
        gained_templates = templates - prev_templates
        history.append({"iteration": it, "unanimous": unanimous,
                        "total": total, "auto_rules": rules,
                        "auto_templates": templates,
                        "holdouts": holdouts})
        print(f"\nITERATION {it}: {unanimous}/{total} unanimous, "
              f"{gained_rules} new rule(s), {gained_templates} new "
              f"template(s) ({rules} auto rules / {templates} auto "
              f"templates total)", flush=True)
        for h in holdouts:
            print(f"  holdout: {h}", flush=True)

        if unanimous == total:
            print("\nSUCCESS: 100% unanimity reached", flush=True)
            break
        if (unanimous <= prev_unanimous and gained_rules == 0
                and gained_templates == 0):
            print("\nSTALL: no unanimity gain, no new rules, no new "
                  "templates — remaining flips are judgment-shaped (see "
                  "the escalated queue) or stochastic; stopping",
                  flush=True)
            break
        prev_unanimous = unanimous
        prev_rules, prev_templates = rules, templates
    return holdouts


def _finalize(args, holdouts: list[str]) -> None:
    """Post-iteration settlement: expert-coder adjudication of the
    judgment-shaped holdouts, review routing, the audit-convergence loop
    (clinical review disputes -> adjudication -> deterministic structure),
    registry ingest, aggregate refresh. Runs once per convergence cycle —
    the structure it mints is exactly what the NEXT cycle's fresh
    generative pass runs against."""
    # Whatever is still non-unanimous NOW — stall, max-iters, or a failed
    # batch — is beyond what deterministic RULES can converge:
    # judgment-shaped disagreements. Before any human sees them, the
    # automated expert coder adjudicates each one strictly from the
    # authoritative sources (independent passes must agree, every decision
    # must cite authority + note evidence, verdicts replay through the
    # full validator+scrubber stack). Only what the adjudicator abstains
    # on or splits over reaches the human queue.
    if holdouts and os.getenv("CODER_ADJUDICATION", "1") == "1":
        try:
            from tools.coder_adjudicator import adjudicate
            print(f"\nCoder adjudication: {len(holdouts)} judgment-shaped "
                  f"holdout(s)", flush=True)
            astats = adjudicate(RESULTS_DIR, docs=holdouts)
            print(json.dumps(astats, indent=2, default=str), flush=True)
            _, _, holdouts = _unanimity(holdouts)
        except Exception as exc:
            # adjudication is an enhancement to finalization, never a
            # blocker for it — a crash here must not strand the holdouts
            # unrouted
            print(f"Coder adjudication failed ({exc}) — routing all "
                  f"holdouts to review", flush=True)

    if holdouts:
        from tools.replay_reconcile import finalize_review_routing
        print(f"\nFinalizing: routing {len(holdouts)} remaining holdout(s) "
              f"to human review", flush=True)
        finalize_review_routing(RESULTS_DIR, holdouts)

    # Clinical-correctness audit — deferred during iterations, run ONCE
    # here on the SETTLED full scope. It promotes every held scrub-CLEAN
    # claim whose interpretive corrections uphold (releasing the pending
    # hold to CLEAN) and routes any dispute to REVIEW with the correction
    # named. Runs after routing so an already-routed holdout is never
    # promoted; runs before the registry ingest below so promotions are
    # recorded. Adjudicated holdouts were already audited per-doc by the
    # adjudicator, so the fingerprint check skips them here.
    scope = _scope_docs(args.start_at, args.end_at)
    if os.getenv("CLINICAL_AUDIT", "1") == "1":
        # AUDIT_CONVERGENCE=1 (default) upgrades the single final audit to
        # the audit-convergence loop: every grounded dispute is adjudicated
        # against the authorities, actuated into deterministic structure,
        # and the notes replay until the review upholds them or the loop
        # stalls (only then do the remaining disputes stay with a human).
        if os.getenv("AUDIT_CONVERGENCE", "1") == "1":
            try:
                from tools.audit_convergence_loop import converge
                summary = converge(RESULTS_DIR, docs=scope)
                print(f"\nAudit convergence (final): "
                      f"{summary['status']} after "
                      f"{len(summary['iterations'])} iteration(s); "
                      f"remaining dispute(s): "
                      f"{summary.get('final_disputed', [])}", flush=True)
            except Exception as exc:
                print(f"Audit convergence failed ({exc}) — falling back "
                      f"to the single final audit", flush=True)
                _final_audit_only(scope)
        else:
            _final_audit_only(scope)

    # Pack consolidation — the growth loop's maintenance counterpart,
    # run once per finalization AFTER all structure for this cycle has
    # landed: leave-one-out exercise scan over the stored corpus (cached
    # by pack+corpus hash, so an unchanged system costs nothing),
    # dormancy tags, and behavior-preserving merges gated on
    # byte-identical corpus replay. Never a blocker: consolidation can
    # only prove equivalence or do nothing, so a crash here degrades to
    # an unconsolidated (still correct) pack. PACK_CONSOLIDATION=0
    # disables; PACK_CONSOLIDATION_MERGE=0 keeps the scan/tags but skips
    # the LLM merge phase.
    if os.getenv("PACK_CONSOLIDATION", "0") == "1":
        try:
            from tools.pack_consolidation import consolidate
            csum = consolidate(
                RESULTS_DIR,
                merge=os.getenv("PACK_CONSOLIDATION_MERGE", "0") == "1")
            print(f"\nPack consolidation: "
                  f"{len(csum['dormancy'].get('tagged', []))} newly "
                  f"dormant, {len(csum['merges'])} merge(s) accepted, "
                  f"{len(csum['declined']) + len(csum['rejected'])} "
                  f"declined/rejected (accepted merges remain draft proposals)",
                  flush=True)
        except Exception as exc:
            print(f"Pack consolidation failed ({exc}) — pack left "
                  f"unconsolidated", flush=True)

    # Registry ingest — re-run after the audit so claims it just promoted
    # to CLEAN are recorded. run.py's per-iteration ingest ran with the
    # audit deferred, so interpretive-correction claims were held back
    # then (eligible_for_auto fails closed without an upheld audit).
    try:
        from tools.claims_registry import ingest as registry_ingest
        rstats = registry_ingest(RESULTS_DIR)
        print(f"Claims registry (final): {rstats['recorded']} recorded, "
              f"{rstats['unchanged']} unchanged, "
              f"{rstats['skipped']} awaiting review/ineligible", flush=True)
    except Exception as exc:
        print(f"Claims registry ingest (final) failed ({exc})", flush=True)

    # Adjudication/finalization rewrote per-note files after run.py built
    # the aggregate — refresh all_results.json so it matches disk.
    combined = []
    for f in sorted(RESULTS_DIR.glob("*_results.json")):
        if f.name == "all_results.json":
            continue
        try:
            combined.append(json.loads(f.read_text()))
        except Exception:
            pass
    (RESULTS_DIR / "all_results.json").write_text(
        json.dumps(combined, indent=2, default=str))


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start-at", required=True)
    p.add_argument("--end-at", required=True)
    p.add_argument("--max-iters", type=int, default=8,
                   help="max batch iterations per convergence cycle")
    p.add_argument("--max-cycles", type=int, default=0,
                   help="hard cap on convergence cycles; 0 (default) = "
                        "unlimited — the loop runs until SUCCESS, STALL, "
                        "or --patience is exhausted")
    p.add_argument("--patience", type=int, default=3,
                   help="stop after this many CONSECUTIVE cycles without "
                        "measurable claim progress (CLEAN count, unanimity "
                        "count, billing splits, material review findings), "
                        "even if structure keeps being minted — bounded "
                        "spend on claims whose blockers turn out not to "
                        "be structure-shaped")
    p.add_argument("--consistency", type=int, default=3)
    p.add_argument("--workers", type=int, default=30)
    p.add_argument("--resume", action="store_true",
                   help="Seed holdouts from the results already on disk "
                        "instead of starting with a full-scope batch — "
                        "continue a loop that was interrupted, without "
                        "repaying the notes it already converged")
    args = p.parse_args()

    history: list = []
    scope = _scope_docs(args.start_at, args.end_at)
    holdouts: list[str] = []  # empty = full scope (first batch)
    if args.resume:
        c0, t0, non_clean = _cleanliness(scope)
        u0 = _unanimity(scope)[0]
        print(f"--resume: {c0}/{t0} CLEAN on disk ({u0} unanimous), "
              f"{len(non_clean)} non-CLEAN holdout(s) to iterate",
              flush=True)
        if t0 and not non_clean:
            print("SUCCESS: every note in scope already CLEAN", flush=True)
            return
        # Holdouts are the NON-CLEAN notes, not just the non-unanimous
        # ones: a unanimous-but-review-disputed note still needs cycles,
        # and seeding only non-unanimous docs would either drop it or —
        # worse, with zero non-unanimous docs — trigger a full-scope
        # batch that repays every already-CLEAN note.
        holdouts = non_clean

    cycle, cycles_without_progress = 0, 0
    stall_grace_used = False
    while True:
        cycle += 1
        cap = f"/{args.max_cycles}" if args.max_cycles else ""
        sig_before = _structure_sig()
        vec_before = _progress_vector(scope)
        print(f"\n{'#' * 66}\nCONVERGENCE CYCLE {cycle}{cap} — "
              f"{vec_before[0]} CLEAN of {len(scope) or '?'} on disk"
              f"\n{'#' * 66}", flush=True)

        holdouts = _run_inner_iterations(args, holdouts, history)
        _finalize(args, holdouts)

        scope = _scope_docs(args.start_at, args.end_at)
        clean, total, non_clean = _cleanliness(scope)
        vec = _progress_vector(scope)
        sig_changed = _structure_sig() != sig_before
        # Progress = ANY component of the vector rose: a note going CLEAN
        # or unanimous, billing splits shrinking, material review findings
        # shrinking. Regressions elsewhere don't cancel it — fresh
        # generative runs are stochastic, and the guard only needs to
        # distinguish "getting somewhere" from "flat".
        progressed = any(a > b for a, b in zip(vec, vec_before))
        cycles_without_progress = (0 if progressed
                                   else cycles_without_progress + 1)
        if progressed or sig_changed:
            stall_grace_used = False
        history.append({"cycle": cycle, "clean": clean, "total": total,
                        "structure_changed": sig_changed,
                        "progress_vector": list(vec),
                        "non_clean": non_clean})
        print(f"\nCYCLE {cycle}: {clean}/{total} CLEAN, structure "
              f"{'CHANGED' if sig_changed else 'unchanged'}, progress "
              f"{'YES' if progressed else 'none'} "
              f"(unanimous={vec[1]}, targets_satisfied={vec[2]}, "
              f"targets_recorded={vec[3]}, billing_splits={-vec[4]}, "
              f"material_findings={-vec[5]})", flush=True)

        if total and not non_clean:
            print("\nSUCCESS: every note in scope is CLEAN", flush=True)
            break
        if not total:
            print("\nNo results in scope — nothing to converge; stopping",
                  flush=True)
            break
        if not sig_changed and not progressed:
            # A stall verdict must not strand anchored ground truth:
            # adjudication records targets AFTER actuation ran (audit
            # convergence follows actuation inside finalization), so
            # anchored-but-unworked classes deserve exactly one grace
            # cycle for actuation to propose against them. Bounded: the
            # grace is single-use until progress or structure resets it,
            # and --patience remains the hard ceiling.
            pending = _anchored_unworked_classes()
            capped = bool(args.max_cycles) and cycle >= args.max_cycles
            if pending and not stall_grace_used and not capped:
                stall_grace_used = True
                print(f"\nSTALL DEFERRED: {len(pending)} anchored audit "
                      f"class(es) await actuation against recorded "
                      f"verified targets — granting one grace cycle "
                      f"({', '.join(pending[:4])}"
                      f"{'...' if len(pending) > 4 else ''})", flush=True)
                holdouts = non_clean
                continue
            print("\nSTALL: no new deterministic structure and no claim "
                  "progress this cycle — what remains is judgment-shaped "
                  "(human queue) or stochastic; stopping", flush=True)
            break
        if cycles_without_progress >= args.patience:
            print(f"\nPATIENCE: {cycles_without_progress} consecutive "
                  f"cycle(s) minted structure without any claim progress "
                  f"— the remaining blockers are not structure-shaped; "
                  f"stopping (--patience {args.patience})", flush=True)
            break
        if args.max_cycles and cycle >= args.max_cycles:
            print(f"\nMAX: --max-cycles {args.max_cycles} reached with "
                  f"{len(non_clean)} note(s) still non-CLEAN", flush=True)
            break
        # The cycle minted structure (or promoted claims). Every non-CLEAN
        # note earned a fresh generative pass against the stack that did
        # not exist when it failed — that INCLUDES unanimous-but-disputed
        # notes: fresh runs let the corrected claim emerge natively
        # instead of by adjudication patch.
        holdouts = non_clean
        print(f"\nRe-running {len(non_clean)} non-CLEAN note(s) against "
              f"the new structure...", flush=True)

    print("\n" + json.dumps({"history": history}, indent=2), flush=True)


if __name__ == "__main__":
    main()
