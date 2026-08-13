"""Coding memorandum — feeding the pack's learned corrections back
upstream to the generative coder.

Convergence works by the deterministic stack correcting the generative
stage, but the generative stage itself never learns: each new note pays
full price (disagreement, adjudication, actuation) for error classes the
pack already encodes. This module compiles the pack's PROVEN corrections
into a compact prompt block the 4-pass coder sees on every run, so the
expensive stage stops re-making mistakes the cheap stage already knows
how to fix.

Everything here is deterministic and derived from artifacts that already
carry provenance:

  source     data/rules/validator_rules.json — every enabled
             auto-generated rule's authority citation and actuation
             rationale (prose written when the rule was accepted against
             a verified target, describing exactly the coding error it
             corrects).
  filter     data/registry/rule_exercise.json (written by
             tools/pack_consolidation.py) — rules the exercise scan
             PROVED inert on the stored corpus are dropped (a rule that
             changes nothing anywhere teaches nothing). Rules the scan
             has not seen yet (minted after it ran) stay included — a
             stale scan must never silence fresh corrections. Without
             scan data every enabled auto rule qualifies.
  freshness  compiled on demand and cached by the pack file's
             (mtime, size) — a pack change (acceptance, amendment,
             merge) is visible to the very next run with no
             regeneration step to forget.

The memorandum NEVER makes decisions: the deterministic stack still
validates everything downstream, so a stale or ignored memorandum
degrades to exactly today's behavior (the pack corrects the output).
Toggle: CODING_MEMORANDUM=0 disables injection — the measurement knob
for comparing disagreement rates with and without it.
"""

import json
import os
import re
from pathlib import Path

from loguru import logger

from app.release.source_manifest import declared_source_path

# Declared identities, not filenames composed here (Codex F6-R5-A, round 6).
RULES_PATH = declared_source_path("validator_rules")
EXERCISE_PATH = declared_source_path("rule_exercise")

MAX_ENTRIES = 40
MAX_GUIDANCE_CHARS = 320

_cache: dict = {"key": None, "block": ""}


def _proven_inert_ids() -> set[str]:
    """Rule ids the last exercise scan PROVED inert on the stored corpus
    — the only rules the memorandum drops. Rules absent from the scan
    (minted after it ran) are unknown, not inert, and stay included: a
    stale scan must never silence the pack's freshest corrections."""
    try:
        scan = (json.loads(EXERCISE_PATH.read_text()) or {}).get("scan")
        rules = (scan or {}).get("rules")
        if not isinstance(rules, dict):
            return set()
        return {rid for rid, info in rules.items()
                if not (info or {}).get("load_bearing_on")}
    except Exception:
        return set()


def _clean(text: str, limit: int) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()[:limit]


def _entries(pack: dict) -> list[dict]:
    inert = _proven_inert_ids()
    out = []
    for r in pack.get("rules", []):
        if not r.get("auto_generated") or not r.get("enabled", True):
            continue
        rid = str(r.get("id") or "")
        if rid in inert:
            continue
        prov = r.get("provenance") or {}
        guidance = _clean(prov.get("rationale"), MAX_GUIDANCE_CHARS)
        authority = _clean(r.get("authority"), 160)
        if not guidance:
            continue
        out.append({"rule_id": rid, "authority": authority,
                    "guidance": guidance,
                    # scan-proven rules sort ahead of unproven ones
                    "weight": len((prov.get("documents") or []))})
    out.sort(key=lambda e: (-e["weight"], e["rule_id"]))
    return out[:MAX_ENTRIES]


def memorandum_block() -> str:
    """The prompt block (empty string when disabled or nothing to say),
    recompiled automatically whenever the pack file changes."""
    if os.getenv("CODING_MEMORANDUM", "1") != "1":
        return ""
    try:
        st = RULES_PATH.stat()
        key = (st.st_mtime_ns, st.st_size)
    except OSError:
        return ""
    if _cache["key"] == key:
        return _cache["block"]
    try:
        pack = json.loads(RULES_PATH.read_text())
        entries = _entries(pack)
    except Exception as exc:
        logger.warning(f"coding memorandum unavailable ({exc})")
        entries = []
    block = ""
    if entries:
        lines = [f"- {e['guidance']}"
                 + (f" [{e['authority']}]" if e['authority'] else "")
                 for e in entries]
        block = (
            "## CODING MEMORANDUM (corrections this practice's validation"
            " stack has already had to make — verified against"
            " authoritative sources; apply them proactively so your"
            " output does not need the same correction)\n"
            + "\n".join(lines))
        logger.info(f"  Coding memorandum: {len(entries)} learned "
                    f"correction(s) injected into the coder context")
    _cache["key"], _cache["block"] = key, block
    return block
