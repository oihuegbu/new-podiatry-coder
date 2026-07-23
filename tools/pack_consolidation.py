#!/usr/bin/env python3
"""Rule-pack consolidation — the growth loop's maintenance counterpart.

Actuation only ever ADDS structure: every convergence cycle can mint new
rules, templates, and observables, and nothing ever asks whether the pack
it grew is still the smallest pack that produces the same behavior. At 45
rules after two notes that is cosmetic; at 50 notes it is a real
maintenance problem (near-duplicate rules that drift apart under
amendment, dormant rules nobody can safely delete because nobody knows
they are dormant).

This module closes that gap with the same discipline the growth side
uses — deterministic replay evidence, never judgment:

  exercise   For every enabled auto-generated rule, replay the ENTIRE
             stored corpus (all consistency runs of every note) with just
             that rule disabled and compare against the live pack's
             replay fingerprint (billing signature + every measurement
             observable's emission signature). A rule whose absence
             changes nothing anywhere is LOAD-BEARING NOWHERE on the
             evidence the system owns. Results persist in
             data/registry/rule_exercise.json.

  dormancy   Rules load-bearing nowhere are TAGGED dormant_on_corpus in
             the pack (metadata only — they stay enabled, because "inert
             on the stored corpus" is not "inert on next week's note").
             The tag gives a human, and the merge phase, an evidence-
             backed shortlist; it is cleared automatically the moment a
             later scan finds the rule load-bearing.

  merge      Enabled auto rules sharing a template are candidate
             families. An LLM proposes a single merged rule per family
             (or declines); the proposal passes the same structural and
             no-code-literal gates as actuation, and then the decisive
             gate: the corpus replay fingerprint under {family disabled +
             merged rule} must be BYTE-IDENTICAL to the live pack's on
             every run of every note. Equivalence is proven, never
             assumed; a proposal that changes anything anywhere is
             rejected and ledgered so it is not re-asked until the pack
             changes. Accepted merges disable the originals in place
             (append-only history, superseded_by set) and append the
             merged rule with full provenance.

Usage:
  python tools/pack_consolidation.py [--results-dir DIR] [--no-merge]
                                     [--limit N] [--force]
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loguru import logger  # noqa: E402

STATE_PATH = ROOT / "data" / "registry" / "rule_exercise.json"
RESULTS_DIR = ROOT / "output" / "results"
# LLM merge proposals per invocation — consolidation runs inside the
# convergence loop, and a huge pack must not fan out into unbounded
# reasoning-model calls in one pass.
MERGE_LIMIT = 3


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Corpus + replay fingerprints
# ---------------------------------------------------------------------------

def _pack() -> dict:
    from tools.auto_actuate import RULES_PATH
    return json.loads(RULES_PATH.read_text())


def _auto_rules(pack: dict) -> list[dict]:
    return [r for r in pack.get("rules", [])
            if r.get("auto_generated") and r.get("enabled", True)]


def _pack_hash(pack: dict) -> str:
    """Content hash of the ENABLED rule set — the scan's cache key. Any
    accepted rule, amendment, or disable changes it and invalidates the
    stored exercise data. Dormancy tags (and provenance bookkeeping) are
    excluded: they are metadata this module itself writes FROM scan
    results — hashing them would make tag_dormancy invalidate the very
    scan it was derived from and force one wasted full rescan per run."""
    def strip(r: dict) -> dict:
        return {k: v for k, v in r.items()
                if k not in ("dormant_on_corpus", "dormant_since",
                             "provenance")}
    rules = sorted(json.dumps(strip(r), sort_keys=True, default=str)
                   for r in pack.get("rules", [])
                   if r.get("enabled", True))
    return hashlib.sha256("\n".join(rules).encode()).hexdigest()[:16]


def _corpus(results_dir: Path) -> list[tuple[str, list[dict], str]]:
    """[(doc, run payloads, note text)] for every stored result — every
    consistency run of every note, falling back to the main payload for
    results saved without stored runs."""
    from tools.auto_actuate import _load_main, _load_runs, _note_text_for
    out = []
    for f in sorted(results_dir.glob("*_results.json")):
        if f.name == "all_results.json":
            continue
        doc = f.stem.removesuffix("_results")
        runs = _load_runs(doc, results_dir)
        if not runs:
            main = _load_main(doc, results_dir)
            runs = [main] if main else []
        if not runs:
            continue
        note = _note_text_for(doc, results_dir, runs, runs[0])
        if note:
            out.append((doc, runs, note))
    return out


def _corpus_hash(corpus: list) -> str:
    """Content hash of the whole stored corpus — run PAYLOADS included,
    not just names/counts: a fresh generative cycle overwrites the
    consistency runs with the same count, and a stale scan silently
    reused against new runs would tag the wrong rules dormant."""
    h = hashlib.sha256()
    for doc, runs, note in corpus:
        h.update(doc.encode())
        h.update(hashlib.sha256(note.encode()).digest())
        for r in runs:
            h.update(hashlib.sha256(
                json.dumps(r, sort_keys=True, default=str).encode()
            ).digest())
    return h.hexdigest()[:16]


def _run_fingerprint(rep, scrubber, payload: dict, note: str) -> str:
    """One run's full behavioral fingerprint under the CURRENTLY-POINTED
    rule pack: billing signature (claim-form content) plus every
    measurement observable's emission signature (advisories and anything
    the vocabulary has grown to). Two packs with equal fingerprints on
    every run of every note are behaviorally indistinguishable on all the
    evidence the system owns."""
    from tools.observables import record_signatures
    from tools.replay_reconcile import _rebuild_run
    arrays, report = rep.replay_arrays(payload, note)
    sig = rep.signature(arrays["icd10_codes"], arrays["cpt_codes"],
                        arrays["hcpcs_codes"])
    rebuilt = _rebuild_run(payload, arrays, report, scrubber, note)
    obs = {name: sorted(keys) for name, keys
           in record_signatures(rebuilt).items()}
    return json.dumps({"billing": sig, "observables": obs},
                      sort_keys=True, default=str)


def _fingerprints(rep, scrubber, corpus: list) -> dict[str, list[str]]:
    return {doc: [_run_fingerprint(rep, scrubber, p, note) for p in runs]
            for doc, runs, note in corpus}


class _temp_pack:
    """Context manager: point the rule engine at a modified copy of the
    live pack for the duration, restoring the real pack (and cache) on
    exit no matter what."""

    def __init__(self, pack: dict):
        self.pack = pack

    def __enter__(self):
        import app.validation.rule_engine as re_mod
        from tools.auto_actuate import RULES_PATH
        self._re_mod = re_mod
        self._real = RULES_PATH
        fd = tempfile.NamedTemporaryFile(
            "w", suffix=".json", prefix="pack_consolidation_",
            delete=False)
        json.dump(self.pack, fd)
        fd.close()
        self._tmp = Path(fd.name)
        re_mod.RULES_FILE = self._tmp
        re_mod.load_rule_pack.cache_clear()
        return self

    def __exit__(self, *exc):
        self._re_mod.RULES_FILE = self._real
        self._re_mod.load_rule_pack.cache_clear()
        self._tmp.unlink(missing_ok=True)
        return False


def _pack_without(pack: dict, rule_ids: set[str],
                  extra_rule: dict | None = None) -> dict:
    out = copy.deepcopy(pack)
    for r in out.get("rules", []):
        if r.get("id") in rule_ids:
            r["enabled"] = False
    if extra_rule is not None:
        out["rules"].append(dict(extra_rule, enabled=True,
                                 auto_generated=True))
    return out


# ---------------------------------------------------------------------------
# Phase 1: exercise scan
# ---------------------------------------------------------------------------

def _state() -> dict:
    try:
        return json.loads(STATE_PATH.read_text())
    except Exception:
        return {}


def _save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=1, default=str))


def exercise_scan(results_dir: Path, rep=None,
                  force: bool = False) -> dict:
    """{rule_id: {"load_bearing_on": [docs]}} for every enabled auto
    rule, by leave-one-out corpus replay. Cached by (pack hash, corpus
    hash): any pack edit or corpus growth invalidates it. This is the
    evidence base for dormancy tags AND the coding memorandum's
    load-bearing filter."""
    pack = _pack()
    corpus = _corpus(results_dir)
    phash, chash = _pack_hash(pack), _corpus_hash(corpus)
    state = _state()
    scan = state.get("scan") or {}
    if not force and scan.get("pack_hash") == phash \
            and scan.get("corpus_hash") == chash:
        logger.info("exercise scan: cached (pack and corpus unchanged)")
        return scan
    rules = _auto_rules(pack)
    if not corpus or not rules:
        logger.info(f"exercise scan: nothing to scan "
                    f"({len(rules)} auto rules, {len(corpus)} notes)")
        scan = {"pack_hash": phash, "corpus_hash": chash,
                "scanned_at": _now(), "rules": {}}
        state["scan"] = scan
        _save_state(state)
        return scan

    if rep is None:
        from tools.auto_actuate import Replayer
        rep = Replayer()
    from tools.auto_actuate import RULES_PATH, _advisory_scrubber
    scrubber = _advisory_scrubber(rep)

    logger.info(f"exercise scan: {len(rules)} auto rule(s) x "
                f"{sum(len(r) for _, r, _ in corpus)} stored run(s)")
    # Pin the engine to the LIVE pack before the baseline — a previous
    # tool crashing inside a temp-pack swap must not leak into the scan.
    import app.validation.rule_engine as re_mod
    re_mod.RULES_FILE = RULES_PATH
    re_mod.load_rule_pack.cache_clear()
    baseline = _fingerprints(rep, scrubber, corpus)
    out: dict[str, dict] = {}
    for r in rules:
        rid = r["id"]
        with _temp_pack(_pack_without(pack, {rid})):
            fps = _fingerprints(rep, scrubber, corpus)
        bearing = sorted(doc for doc in baseline
                         if fps.get(doc) != baseline[doc])
        out[rid] = {"load_bearing_on": bearing}
        logger.info(f"  {rid}: "
                    + (f"load-bearing on {len(bearing)} note(s)"
                       if bearing else "inert on the stored corpus"))
    scan = {"pack_hash": phash, "corpus_hash": chash,
            "scanned_at": _now(), "rules": out}
    state["scan"] = scan
    _save_state(state)
    return scan


def tag_dormancy(scan: dict) -> dict:
    """Write dormant_on_corpus tags into the live pack from scan
    evidence. Metadata only — enabled flags never change here (a rule
    inert on the stored corpus may be load-bearing on the next note; only
    a human, or a superseding merge, retires a rule)."""
    from tools.auto_actuate import RULES_PATH
    pack = _pack()
    tagged, cleared = [], []
    for r in pack.get("rules", []):
        rid = r.get("id")
        info = (scan.get("rules") or {}).get(rid)
        if info is None or not r.get("enabled", True):
            continue
        if info["load_bearing_on"]:
            if r.pop("dormant_on_corpus", None):
                r.pop("dormant_since", None)
                cleared.append(rid)
        elif not r.get("dormant_on_corpus"):
            r["dormant_on_corpus"] = True
            r["dormant_since"] = _now()
            tagged.append(rid)
    if tagged or cleared:
        RULES_PATH.write_text(json.dumps(pack, indent=1))
        logger.info(f"dormancy: tagged {tagged or '[]'}, "
                    f"cleared {cleared or '[]'}")
    return {"tagged": tagged, "cleared": cleared}


# ---------------------------------------------------------------------------
# Phase 2: merge
# ---------------------------------------------------------------------------

_MERGE_SYSTEM_PROMPT = """\
You are consolidating a validator rule pack for a medical-coding
pipeline. You are given several ENABLED rules that share one template
(one generic mechanic; the rules differ only in config: lexicons,
descriptor grammar, context regexes, messages, authority citations).

Decide whether ONE merged rule of the SAME template can replace ALL of
them with EXACTLY the same behavior — not similar, identical. Merging is
only possible when the configs are genuinely unifiable (e.g. unionable
lexicons that cannot interact, identical structure with disjoint
vocabularies). When the rules encode different clinical policies, or a
union could fire where no original fired, DECLINE.

The merged rule must:
- use the same "template" as the originals
- carry NO literal medical codes beyond what the originals already carry
  in the same fields
- cite every original's authority (concatenate/merge the citations)
- keep every original's message semantics (a message template per
  original context is acceptable if the template supports it)

Your proposal is verified mechanically: the whole stored corpus is
replayed under {originals disabled + your merged rule} and must be
byte-identical (claim lines AND advisory emissions) to the current pack.
An unverifiable merge is simply rejected — when in doubt, decline.

Respond with JSON only:
{"decision": "merge" | "decline",
 "why": "<one or two sentences>",
 "rule": { ...complete merged rule config, when decision=merge... }}"""


def merge_candidates(pack: dict, scan: dict) -> list[list[dict]]:
    """Families of >=2 enabled auto rules sharing a template — the only
    shape a behavior-preserving merge can take (the template IS the
    mechanic; rules of different templates have nothing to merge)."""
    by_tpl: dict[str, list[dict]] = {}
    for r in _auto_rules(pack):
        by_tpl.setdefault(str(r.get("template") or ""), []).append(r)
    return [sorted(rs, key=lambda r: r["id"])
            for tpl, rs in sorted(by_tpl.items()) if tpl and len(rs) >= 2]


def _family_key(family: list[dict]) -> str:
    return hashlib.sha256("|".join(sorted(r["id"] for r in family))
                          .encode()).hexdigest()[:16]


def _declined(state: dict, phash: str) -> set[str]:
    return {d["family"] for d in state.get("declined_merges", [])
            if d.get("pack_hash") == phash}


def propose_merge(family: list[dict]) -> dict:
    from app.core.llm_client import chat_completion
    user = json.dumps({"template": family[0].get("template"),
                       "rules": family}, indent=1, default=str)
    text, _usage = chat_completion(
        system_prompt=_MERGE_SYSTEM_PROMPT,
        user_prompt=f"RULE FAMILY:\n{user}",
        temperature=0.05, max_tokens=8192, json_mode=True, effort="high")
    return json.loads(text)


def gate_merge(merged: dict, family: list[dict], pack: dict,
               baseline: dict, rep, scrubber, corpus: list) -> str:
    """Empty string when the merged rule is accepted; otherwise the
    rejection reason. Structural + no-code-literals first (same gates as
    actuation), then the decisive corpus-equivalence replay."""
    from tools.auto_actuate import gate_no_code_literals, gate_structural
    merged = dict(merged)
    merged.setdefault("enabled", True)
    if str(merged.get("template") or "") != \
            str(family[0].get("template") or ""):
        return "merged rule must keep the family's template"
    why = gate_structural(merged)
    if why:
        return f"structural: {why}"
    why = gate_no_code_literals(merged)
    if why:
        return f"code literals: {why}"
    ids = {r["id"] for r in family}
    if merged.get("id") in {r.get("id") for r in pack.get("rules", [])}:
        return f"merged rule id {merged.get('id')!r} collides with an " \
               f"existing rule"
    with _temp_pack(_pack_without(pack, ids, extra_rule=merged)):
        fps = _fingerprints(rep, scrubber, corpus)
    diff = [doc for doc in baseline if fps.get(doc) != baseline[doc]]
    if diff:
        return ("corpus replay is not byte-identical under the merge "
                f"(differs on: {', '.join(sorted(diff))}) — behavior "
                "preservation is unproven")
    return ""


def apply_merge(merged: dict, family: list[dict]) -> None:
    """Disable the originals in place (history preserved, superseded_by
    set) and append the merged rule with full provenance."""
    from tools.auto_actuate import RULES_PATH
    pack = _pack()
    ids = {r["id"] for r in family}
    for r in pack.get("rules", []):
        if r.get("id") in ids:
            r["enabled"] = False
            prov = r.setdefault("provenance", {})
            prov["disabled_reason"] = "consolidated into merged rule"
            prov["superseded_by"] = merged["id"]
    rule = dict(merged, auto_generated=True, enabled=True)
    rule["provenance"] = {
        "consolidated_at": _now(),
        "consolidated_from": sorted(ids),
        "protocol": "behavior-preserving merge; corpus replay "
                    "byte-identical (billing + observable emissions) "
                    "under {originals disabled + merged rule}",
    }
    pack["rules"].append(rule)
    RULES_PATH.write_text(json.dumps(pack, indent=1))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def consolidate(results_dir: Path, merge: bool = True,
                limit: int = MERGE_LIMIT, force: bool = False,
                rep=None) -> dict:
    summary = {"scan": {}, "dormancy": {}, "merges": [],
               "declined": [], "rejected": []}
    scan = exercise_scan(results_dir, rep=rep, force=force)
    summary["scan"] = {rid: info["load_bearing_on"]
                       for rid, info in (scan.get("rules") or {}).items()}
    summary["dormancy"] = tag_dormancy(scan)

    if not merge:
        return summary
    pack = _pack()
    fams = merge_candidates(pack, scan)
    if not fams:
        logger.info("consolidation: no mergeable families")
        return summary
    state = _state()
    phash = _pack_hash(pack)
    declined = _declined(state, phash)
    fams = [f for f in fams if _family_key(f) not in declined][:limit]
    if not fams:
        logger.info("consolidation: every family already declined for "
                    "this pack")
        return summary

    if rep is None:
        from tools.auto_actuate import Replayer
        rep = Replayer()
    from tools.auto_actuate import RULES_PATH, _advisory_scrubber
    scrubber = _advisory_scrubber(rep)
    corpus = _corpus(results_dir)
    import app.validation.rule_engine as re_mod
    re_mod.RULES_FILE = RULES_PATH
    re_mod.load_rule_pack.cache_clear()
    baseline = _fingerprints(rep, scrubber, corpus)

    for family in fams:
        ids = sorted(r["id"] for r in family)
        logger.info(f"=== Merge candidate ({family[0].get('template')}): "
                    f"{', '.join(ids)} ===")
        fkey = _family_key(family)

        def _decline(why: str, bucket: str) -> None:
            state.setdefault("declined_merges", []).append(
                {"family": fkey, "rule_ids": ids, "pack_hash": phash,
                 "why": why, "at": _now()})
            summary[bucket].append({"rule_ids": ids, "why": why})
            logger.info(f"  -> {bucket.upper()}: {why}")

        try:
            proposal = propose_merge(family)
        except Exception as exc:
            # transient (LLM/network) — no ledger entry, retry next run
            summary["rejected"].append({"rule_ids": ids,
                                        "why": f"proposal failed: {exc}"})
            logger.warning(f"  merge proposal failed: {exc}")
            continue
        if str(proposal.get("decision")) != "merge" \
                or not isinstance(proposal.get("rule"), dict):
            _decline(str(proposal.get("why") or "proposer declined"),
                     "declined")
            continue
        why = gate_merge(proposal["rule"], family, pack, baseline,
                         rep, scrubber, corpus)
        if why:
            _decline(why, "rejected")
            continue

        apply_merge(proposal["rule"], family)
        # Post-write verification against the LIVE pack file — the same
        # paranoia as actuation's audit_pack: prove the written pack
        # still replays byte-identically, roll back wholesale otherwise.
        import app.validation.rule_engine as re_mod
        re_mod.load_rule_pack.cache_clear()
        after = _fingerprints(rep, scrubber, corpus)
        if any(after.get(doc) != baseline[doc] for doc in baseline):
            from tools.auto_actuate import _disable_rule, _reenable_rule
            _disable_rule(proposal["rule"]["id"],
                          reason="post-merge live verification failed")
            for rid in ids:
                _reenable_rule(rid)
            re_mod.load_rule_pack.cache_clear()
            _decline("post-write live verification failed — rolled back",
                     "rejected")
            continue
        summary["merges"].append({"rule_ids": ids,
                                  "merged_id": proposal["rule"]["id"]})
        logger.info(f"  -> MERGED into {proposal['rule']['id']}")
        # pack changed: later families must gate against the new baseline
        pack = _pack()
        phash = _pack_hash(pack)
        baseline = after

    _save_state(state)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    ap.add_argument("--no-merge", action="store_true")
    ap.add_argument("--limit", type=int, default=MERGE_LIMIT)
    ap.add_argument("--force", action="store_true",
                    help="ignore the scan cache")
    a = ap.parse_args()
    summary = consolidate(a.results_dir, merge=not a.no_merge,
                          limit=a.limit, force=a.force)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
