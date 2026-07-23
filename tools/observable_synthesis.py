#!/usr/bin/env python3
"""Observable synthesis — autonomous growth of the MEASUREMENT layer.

Rules grow the deterministic stack; templates grow the rule mechanics;
this module grows what the acceptance gates can SEE. The advisory-emission
observable was hand-built when a live dispute (the claim correct as
billed, only a scrubber advisory wrong) was invisible to billing-signature
realignment by construction. This is that growth automated, with the same
posture as template synthesis: an LLM designs the artifact, deterministic
meta-gates decide whether it deploys, one repair attempt with the exact
failure fed back, and every attempt is ledgered so a declined or failed
gap is never re-burned.

Trigger (tools/audit_convergence_loop.converge): the loop is about to
STALL — no adjudications, no accepted rules, no claim changes — while
routing-grade review findings remain that NO observable's vocabulary can
resolve (kind "other"/unknown, i.e. the reviewer saw a defect the
measurement layer has no name for). Each such finding is a
measurement-gap CANDIDATE; the designer judges whether it actually
disputes a measurable phenomenon of the saved record (versus a genuine
human judgment case, which it must decline).

Meta-gates a design must pass before installing (all deterministic):
  static      the same whitelist AST posture as auto templates (no I/O,
              no dunders, no while/recursion, no literal medical codes)
              plus the observable contract (OBSERVABLE_NAME, SCHEMA_DOC,
              FINDING_KINDS, identify(result, finding),
              signature(result)) — tools/observables.py
  vocabulary  FINDING_KINDS must be NEW names: no overlap with existing
              observables or with the reviewer's billing-mechanizable
              kinds (those resolve to claim-line disputes already)
  identity    identify() on the triggering (record, finding) resolves a
              key — twice, identically (a gap the design cannot resolve
              on its own trigger is not closed by it)
  purity      signature() twice on the record → identical key sets; the
              record is byte-identical after both calls (measurement
              must never mutate what it measures)
  baseline    the resolved key IS in signature(record) — the disputed
              phenomenon fires at baseline, so an adjudicated verdict
              has a measurable state to realign
  corpus      signature() runs on EVERY saved result without raising and
              without mutating it — a measurement that crashes on any
              record would poison every future gate replay

On install the observable joins tools/observables.all_observables()
immediately: the clinical reviewer's finding vocabulary grows (its kinds
are offered in the audit prompt and salted into the review fingerprint,
so every verdict goes stale and the notes are RE-REVIEWED under the
grown vocabulary), the adjudicator can mechanize the finding into an
emission verdict, the verified target records, and the emission-aware
replay gates converge actuation on it — the note re-runs against the
grown system end to end, exactly like it does when a rule or template
lands.
"""

from __future__ import annotations

import copy
import json
import sys
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loguru import logger  # noqa: E402

from tools.observables import (AUTO_OBSERVABLES_DIR, all_observables,
                               observable_name_of,
                               validate_observable_source)  # noqa: E402

LEDGER_PATH = ROOT / "data" / "registry" / "observable_synthesis.jsonl"
SYNTHESIS_LIMIT = 2          # max new observables per convergence run
_MAX_ATTEMPTS = 2            # initial design + one repair

# Reviewer finding kinds that already resolve to claim-line disputed
# items (presence/attributes) in tools/coder_adjudicator — a finding of
# these kinds is mechanizable TODAY, so it is never a measurement gap,
# and a synthesized observable may not claim them.
_BILLING_MECHANIZABLE_KINDS = {
    "wrong_code", "missing_code", "coverage", "primary_designation",
    "modifier", "units", "linkage",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Gap detection
# ---------------------------------------------------------------------------

def _gap_sig(doc: str, finding: dict) -> str:
    """Stable identity of one measurement-gap candidate — what the
    attempt ledger keys on so a declined/failed gap is not re-burned
    within one vocabulary epoch."""
    basis = json.dumps([doc, str(finding.get("kind") or ""),
                        str(finding.get("code") or "").upper(),
                        str(finding.get("finding") or "")[:200]],
                       sort_keys=True)
    return sha256(basis.encode()).hexdigest()[:16]


def _vocab_epoch() -> str:
    """Content hash of the CURRENT measurement vocabulary (every
    observable's name + claimed finding kinds). The attempt ledger keys
    declines on this: a gap declined as unmeasurable is a verdict about
    the vocabulary that judged it, so when the vocabulary grows (a new
    observable installs) the epoch changes and every declined/failed gap
    becomes attemptable again — exactly once per epoch. Installed gaps
    never retry (the observable exists)."""
    try:
        vocab = sorted((name, tuple(sorted(e["finding_kinds"])))
                       for name, e in all_observables().items())
    except Exception:
        vocab = []
    return sha256(json.dumps(vocab).encode()).hexdigest()[:16]


def _attempted_sigs() -> set[str]:
    """Gap signatures not worth re-attempting NOW: everything that ever
    INSTALLED, plus declines/failures ledgered under the current
    vocabulary epoch. Entries from older epochs (or from before epochs
    existed) are retryable — the vocabulary that declined them is not the
    vocabulary that would judge them today."""
    epoch = _vocab_epoch()
    out = set()
    if LEDGER_PATH.exists():
        for line in LEDGER_PATH.read_text().splitlines():
            try:
                e = json.loads(line)
            except Exception:
                continue
            if str(e.get("outcome")) == "installed" \
                    or str(e.get("epoch")) == epoch:
                out.add(str(e.get("gap_sig")))
    return out


def _ledger(entry: dict) -> None:
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER_PATH.open("a") as fh:
        fh.write(json.dumps(dict(entry, at=_now(), epoch=_vocab_epoch()),
                            default=str) + "\n")


def detect_gaps(results_dir: Path, docs: list[str]) -> list[dict]:
    """Measurement-gap candidates among the still-disputed notes:
    GROUNDED review findings (authority + note evidence, the same bar
    every routed finding meets) whose kind NO observable and no
    billing-mechanizable translation covers. Kind 'other' is exactly the
    reviewer saying 'I see a defect the vocabulary has no name for'.
    This runs only when the loop is about to STALL — everything still
    disputed here has already failed mechanization by definition — and
    most candidates are genuine human cases; the designer's job is to
    tell the measurable ones apart and decline the rest (each candidate
    is ledgered and attempted at most once per vocabulary epoch — a
    declined gap becomes attemptable again when a new observable installs
    and changes what is measurable)."""
    covered = {k for e in all_observables().values()
               for k in e["finding_kinds"]}
    attempted = _attempted_sigs()
    gaps = []
    for doc in docs:
        f = results_dir / f"{doc}_results.json"
        try:
            payload = json.loads(f.read_text())
        except Exception:
            continue
        main = payload.get("main_result") if isinstance(
            payload.get("main_result"), dict) else payload
        audit = main.get("clinical_audit") or {}
        if audit.get("verdict") != "disputed":
            continue
        for fnd in (audit.get("claim_findings") or []):
            if not isinstance(fnd, dict):
                continue
            kind = str(fnd.get("kind") or "").lower()
            if kind in covered or kind in _BILLING_MECHANIZABLE_KINDS:
                continue  # mechanizable already — not a measurement gap
            if not (str(fnd.get("authority") or "").strip()
                    and str(fnd.get("note_evidence") or "").strip()):
                continue  # ungrounded — noise, not a gap
            sig = _gap_sig(doc, fnd)
            if sig in attempted:
                continue
            gaps.append({"document_id": doc, "finding": fnd,
                         "gap_sig": sig, "record": main})
    return gaps


# ---------------------------------------------------------------------------
# Design (LLM)
# ---------------------------------------------------------------------------

_DESIGN_SYSTEM_PROMPT = """\
You are a medical-coding compliance engineer growing the MEASUREMENT
vocabulary of a podiatry claims pipeline's acceptance gates. The gates
can only converge deterministic fixes on phenomena they can MEASURE on a
saved result record. Their vocabulary is a set of OBSERVABLES — small
pure Python modules that (a) resolve a reviewer's prose finding to the
machine identity of a phenomenon in the record and (b) compute which
phenomena of that class currently fire.

An independent clinical review disputed a note, the dispute could not be
mechanized, and its finding matches no existing observable. Decide:

- If the finding disputes a MEASURABLE PHENOMENON of the saved record —
  something a specific record block emits (a finding, an annotation, a
  flag) that is either present or absent, where the claim lines
  themselves are correct as billed — author the observable that
  measures it.
- If it is a genuine judgment call about what should be BILLED (code
  choice, units, linkage, medical necessity of a line), DECLINE — that
  is a claim-line dispute or a human case, never a measurement gap.

Deliver JSON: {"decision": "observable" | "decline",
               "rationale": "...",
               "observable_code": "..."}  (code only when authoring)

The module contract (enforced by a static gate — violations are
rejected):

    OBSERVABLE_NAME : str   snake_case, 3-41 chars, unique
    SCHEMA_DOC      : str   what it measures, the key format, and what
                            deterministic surface REALIZES an emission
                            change (rules can only be gated on what they
                            can change)
    FINDING_KINDS   : tuple NEW reviewer finding-kind names (snake_case)
                            this observable resolves — never an existing
                            kind
    def identify(result, finding) -> (key | None, why)
        # deterministic; key MUST end with "|<CLAIM CODE>"; ambiguity
        # (zero or several candidates) returns None — never guess
    def signature(result) -> set of keys currently firing

Authoring rules (violations are rejected mechanically):
- import re only; no I/O, no while loops, no recursion, no classes,
  no dunder access, no getattr/setattr/eval/exec
- NEVER a literal medical code (CPT/HCPCS/ICD) anywhere — observables
  are generic mechanics; identity comes from the record's own data
- read-only: signature()/identify() must not mutate the record
- fail closed: when the record lacks the block you measure, return the
  empty set / None — never invent
"""


def _design_once(gap: dict, existing_docs: dict[str, str],
                 feedback: str) -> dict:
    from app.core.config import LLM_PROVIDER
    from app.core.llm_client import chat_completion
    from tools.auto_actuate import PROPOSAL_MODEL

    record = gap["record"]
    slim = {k: record.get(k) for k in
            ("icd10_codes", "cpt_codes", "hcpcs_codes", "claim_scrub",
             "validation_issues", "code_justifications", "status")
            if k in record}
    user = json.dumps({
        "disputed_finding": gap["finding"],
        "document_id": gap["document_id"],
        "saved_record_excerpt": slim,
        "record_top_level_keys": sorted(record.keys()),
        "existing_observables": existing_docs,
        "repair_feedback": feedback or None,
    }, indent=1, default=str)[:60000]
    model = PROPOSAL_MODEL if LLM_PROVIDER == "claude" else None
    try:
        text, usage = chat_completion(
            system_prompt=_DESIGN_SYSTEM_PROMPT, user_prompt=user,
            model=model, max_tokens=16384, json_mode=True, effort="high")
    except Exception as exc:
        if model is None:
            raise
        logger.warning(f"Observable design model {model!r} failed "
                       f"({exc}) — falling back to the pipeline default")
        model = None
        text, usage = chat_completion(
            system_prompt=_DESIGN_SYSTEM_PROMPT, user_prompt=user,
            max_tokens=16384, json_mode=True, effort="high")
    design = json.loads(text)
    design["_model"] = model or "pipeline-default"
    return design


# ---------------------------------------------------------------------------
# Meta-gates
# ---------------------------------------------------------------------------

def _exec_candidate(src: str) -> dict | None:
    from tools.observables import _exec_module
    return _exec_module(src, Path("<candidate>"))


def gate_design(src: str, gap: dict, results_dir: Path) -> str:
    """Every deterministic meta-gate for a candidate observable module.
    Returns '' on acceptance or the exact violation (the repair brief)."""
    problems = validate_observable_source(src)
    if problems:
        return "static gate: " + "; ".join(problems[:5])

    name = observable_name_of(src)
    existing = all_observables()
    if name in existing:
        return f"vocabulary: OBSERVABLE_NAME {name!r} already exists"

    try:
        entry = _exec_candidate(src)
    except Exception as exc:
        return f"load: module raised at import time: {exc!r}"
    if entry is None:
        return "load: missing exports after execution"

    kinds = set(entry["finding_kinds"])
    if not kinds:
        return "vocabulary: FINDING_KINDS is empty"
    taken = {k for e in existing.values() for k in e["finding_kinds"]}
    clash = kinds & (taken | _BILLING_MECHANIZABLE_KINDS | {"other"})
    if clash:
        return (f"vocabulary: FINDING_KINDS {sorted(clash)} already "
                f"resolve elsewhere — an observable must name NEW kinds")

    record, finding = gap["record"], gap["finding"]
    # identity: resolves its own trigger, deterministically
    try:
        k1 = entry["identify"](copy.deepcopy(record), dict(finding))
        k2 = entry["identify"](copy.deepcopy(record), dict(finding))
    except Exception as exc:
        return f"identity: identify() raised on the triggering gap: {exc!r}"
    if not (isinstance(k1, tuple) and len(k1) == 2):
        return "identity: identify() must return a (key, why) pair"
    if k1[0] is None:
        return (f"identity: identify() cannot resolve the very finding "
                f"that triggered this gap ({k1[1]}) — the design does "
                f"not close it")
    if k1[0] != k2[0]:
        return "identity: identify() is not deterministic on its trigger"
    key = str(k1[0])
    if "|" not in key or not key.rsplit("|", 1)[-1].strip():
        return (f"identity: key {key!r} must end with '|<CLAIM CODE>' so "
                f"actuation can scope it to flip classes")

    # purity + baseline on the triggering record
    frozen = json.dumps(record, sort_keys=True, default=str)
    try:
        s1 = set(entry["signature"](record))
        s2 = set(entry["signature"](record))
    except Exception as exc:
        return f"purity: signature() raised on the triggering record: " \
               f"{exc!r}"
    if s1 != s2:
        return "purity: signature() is not deterministic (two calls on " \
               "the same record differ)"
    if json.dumps(record, sort_keys=True, default=str) != frozen:
        return "purity: signature() MUTATED the record it measured"
    if not all(isinstance(x, str) for x in s1):
        return "purity: signature() must return a set of strings"
    if key not in s1:
        return (f"baseline: resolved key {key!r} is not in the record's "
                f"signature — the disputed phenomenon must FIRE at "
                f"baseline for a verdict to have a measurable state")

    # corpus safety: measurement must survive every saved record
    for f in sorted(results_dir.glob("*_results.json")):
        if f.name == "all_results.json":
            continue
        try:
            payload = json.loads(f.read_text())
        except Exception:
            continue
        rec = payload.get("main_result") if isinstance(
            payload.get("main_result"), dict) else payload
        frozen = json.dumps(rec, sort_keys=True, default=str)
        try:
            sig = set(entry["signature"](rec))
        except Exception as exc:
            return f"corpus: signature() raised on {f.name}: {exc!r}"
        if not all(isinstance(x, str) for x in sig):
            return f"corpus: non-string keys on {f.name}"
        if json.dumps(rec, sort_keys=True, default=str) != frozen:
            return f"corpus: signature() mutated {f.name}"
    return ""


# ---------------------------------------------------------------------------
# Growth driver
# ---------------------------------------------------------------------------

def grow_observables(results_dir: Path, docs: list[str],
                     limit: int = SYNTHESIS_LIMIT) -> int:
    """Detect measurement gaps among the still-disputed notes, design at
    most `limit` new observables, meta-gate each (one repair attempt with
    the exact failure fed back), install survivors, and ledger every
    attempt. Returns the number installed — the convergence loop
    continues (re-review under the grown vocabulary, adjudicate, actuate,
    replay) instead of stalling when this is nonzero."""
    gaps = detect_gaps(results_dir, docs)
    if not gaps:
        return 0
    logger.info(f"[observable-synthesis] {len(gaps)} measurement-gap "
                f"candidate(s) among the stalled disputes")
    existing_docs = {n: e["schema_doc"]
                     for n, e in all_observables().items()}
    installed = 0
    for gap in gaps:
        if installed >= limit:
            break
        doc, fnd = gap["document_id"], gap["finding"]
        logger.info(f"[observable-synthesis] designing for {doc} / "
                    f"{fnd.get('kind')} on {fnd.get('code') or 'claim'}")
        feedback = ""
        outcome = "failed"
        detail = ""
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                design = _design_once(gap, existing_docs, feedback)
            except Exception as exc:
                detail = f"design call failed: {exc}"
                logger.warning(f"  attempt {attempt}: {detail}")
                break
            if design.get("decision") != "observable":
                outcome = "declined"
                detail = str(design.get("rationale") or "")[:400]
                logger.info(f"  designer declined: {detail[:200]} — a "
                            f"human case, not a measurement gap")
                break
            src = str(design.get("observable_code") or "")
            reason = gate_design(src, gap, results_dir)
            if reason:
                logger.info(f"  attempt {attempt} rejected: {reason[:300]}")
                feedback = (f"Your previous design was REJECTED by a "
                            f"deterministic meta-gate:\n{reason}\n"
                            f"Repair the module (or decline if the gap "
                            f"is not actually measurable).")
                detail = reason
                continue
            name = observable_name_of(src)
            AUTO_OBSERVABLES_DIR.mkdir(parents=True, exist_ok=True)
            path = AUTO_OBSERVABLES_DIR / f"{name}.py"
            path.write_text(src, encoding="utf-8")
            # the loader re-gates on read; a module that fails there is
            # skipped, so verify it actually joined the vocabulary
            if name not in all_observables():
                path.unlink(missing_ok=True)
                detail = "loader refused the module after install"
                logger.warning(f"  {detail} — rolled back")
                break
            installed += 1
            outcome = "installed"
            detail = name
            existing_docs = {n: e["schema_doc"]
                             for n, e in all_observables().items()}
            logger.info(
                f"  INSTALLED observable {name!r} "
                f"(kinds: {sorted(set(observable_kinds_of(src)))}) — the "
                f"gates' measurement vocabulary grew; every review "
                f"verdict is now stale and the notes re-run against the "
                f"grown system")
            break
        _ledger({"gap_sig": gap["gap_sig"], "document_id": doc,
                 "finding_kind": fnd.get("kind"),
                 "code": fnd.get("code"), "outcome": outcome,
                 "detail": detail})
    return installed


def observable_kinds_of(src: str) -> tuple:
    """FINDING_KINDS of a module source, best-effort (for logging)."""
    try:
        entry = _exec_candidate(src)
        return entry["finding_kinds"] if entry else ()
    except Exception:
        return ()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("results_dir", nargs="?",
                   default=str(ROOT / "output" / "results"))
    p.add_argument("--docs", default="")
    args = p.parse_args()
    rd = Path(args.results_dir)
    docs = [s.strip() for s in args.docs.split(",") if s.strip()] or [
        f.stem.removesuffix("_results")
        for f in sorted(rd.glob("*_results.json"))
        if f.name != "all_results.json"]
    n = grow_observables(rd, docs)
    print(json.dumps({"installed": n}))
