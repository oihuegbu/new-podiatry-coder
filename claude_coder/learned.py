"""Learned verified-resolution index — turns propose-then-verify from probabilistic
into deterministic-on-repeat, license-clean, with a per-entry audit trail.

The loop:
  OBSERVE  — every time propose-then-verify accepts a code (its authoritative
             descriptor entailed by the documentation), the (normalized phrase ->
             code) is appended to an observation log with its evidence. Append-only
             and small, so concurrent consistency workers can write safely.
  PROMOTE  — an offline step (tools/build_learned_index.py) aggregates the log and
             promotes a phrase -> code mapping to DETERMINISTIC trust ONLY when it
             is confirmed across at least PROMOTE_AT DISTINCT encounters and is
             unambiguous (the winning code dominates any competitor). No single
             note — however self-consistent — can promote itself; cross-encounter
             agreement is the automated gate (there is no human sign-off).
  RESOLVE  — data_access.learned_index_codes reads the promoted crosswalk and
             resolves deterministically, but only while the entry is still valid:
             the code must still exist AND its current authoritative descriptor must
             still match the descriptor that was verified (self-invalidating against
             code deletions and descriptor revisions — no edition bookkeeping).

No medical code is authored here: entries are DATA accreted from verified
resolutions, and every promoted entry cites the descriptor + evidence it rests on.
"""
from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from pathlib import Path

OBS_FILE = "learned_observations.jsonl"
INDEX_FILE = "learned_cpt_index.json"
# Distinct encounters that must agree before a mapping earns deterministic trust.
# Cross-encounter agreement is the only automated guard against caching a
# systematic LLM error; tune down for faster learning, up for more caution.
PROMOTE_AT = int(os.environ.get("LEARNED_PROMOTE_AT", "3"))


def _norm(phrase: str) -> str:
    from .terminology import _norm as _n
    return _n(phrase)


def _codes_dir() -> Path:
    from app.core.config import DATA_DIR
    return DATA_DIR / "codes"


def observe(encounter_id: str, phrase: str, code: str, system: str,
            descriptor: str, evidence: list[str] | None = None) -> None:
    """Append one verified resolution to the observation log. Fail-safe: any
    problem is swallowed so learning never breaks coding."""
    try:
        rec = {"enc": str(encounter_id), "phrase": _norm(phrase), "code": str(code),
               "system": system, "descriptor": str(descriptor),
               "evidence": [str(e) for e in (evidence or [])[:1]],
               "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        if not rec["phrase"] or not rec["code"]:
            return
        with open(_codes_dir() / OBS_FILE, "a") as fh:      # O_APPEND: atomic per line
            fh.write(json.dumps(rec) + "\n")
    except Exception:
        pass


def entry_current(entry: dict, current_descriptor: str) -> bool:
    """Is a promoted entry still trustworthy given the code's CURRENT authoritative
    descriptor? (The caller has already confirmed the code still exists.) False when
    the descriptor was revised since the mapping was verified — so the coder re-
    verifies against the new descriptor instead of resolving to a stale meaning."""
    stored = str(entry.get("descriptor") or "")
    if not stored or not current_descriptor:
        return True
    return _norm(current_descriptor) == _norm(stored)


def load_observations(path: Path | None = None) -> list[dict]:
    path = path or (_codes_dir() / OBS_FILE)
    if not path.exists():
        return []
    out: list[dict] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def promote(observations: list[dict], promote_at: int = PROMOTE_AT) -> dict[str, dict]:
    """Aggregate observations into a promoted phrase -> entry crosswalk. A code is
    promoted for a phrase only when DISTINCT encounters agreeing on it reach
    `promote_at` AND it dominates any competing code (>= 2x the runner-up), so a
    contested phrasing stays unpromoted (keeps falling to propose-then-verify)."""
    votes: dict[str, dict[str, set]] = defaultdict(lambda: defaultdict(set))
    meta: dict[tuple, dict] = {}
    for o in observations:
        ph, code = o.get("phrase"), o.get("code")
        if not ph or not code:
            continue
        votes[ph][code].add(o.get("enc"))
        m = meta.setdefault((ph, code), {"descriptor": o.get("descriptor", ""),
                                         "system": o.get("system", "cpt"),
                                         "evidence": []})
        for e in o.get("evidence", []):
            if e and e not in m["evidence"]:
                m["evidence"] = (m["evidence"] + [e])[:3]
    entries: dict[str, dict] = {}
    for ph, by_code in votes.items():
        ranked = sorted(by_code.items(), key=lambda kv: len(kv[1]), reverse=True)
        top_code, top_encs = ranked[0]
        runner_up = len(ranked[1][1]) if len(ranked) > 1 else 0
        if len(top_encs) >= promote_at and len(top_encs) >= 2 * runner_up:
            m = meta[(ph, top_code)]
            entries[ph] = {"code": top_code, "system": m["system"],
                           "descriptor": m["descriptor"], "encounters": len(top_encs),
                           "evidence": m["evidence"]}
    return entries


def build_index(promote_at: int = PROMOTE_AT, out: Path | None = None) -> dict:
    """Compile the observation log into the promoted crosswalk file."""
    entries = promote(load_observations(), promote_at)
    payload = {
        "version": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": OBS_FILE,
        "provenance": (f"learned verified-resolution index: phrase->code mappings each "
                       f"confirmed by >= {promote_at} distinct encounters and "
                       f"unambiguous; grounded in the verified descriptor + evidence"),
        "promote_at": promote_at,
        "entries": entries,
    }
    (out or (_codes_dir() / INDEX_FILE)).write_text(json.dumps(payload, indent=1))
    return payload
