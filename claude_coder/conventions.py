"""Governed CODING CONVENTIONS -- config, not code (issue #6, product-owner
decision 2026-09-22, option 2 of the release-policy reset).

A coder resolves a descriptor axis the record never states -- "primary vs
secondary", "with vs without", "simple vs complex" -- by an established coding
convention, cited to an authority (CPT Assistant, an AAOS/specialty coding
guide, a CMS manual chapter). This module lets the resolver do the same, with
three constraints that keep it agnostic and auditable:

  * Every convention lives in `data/rules/coding_conventions.json` -- a
    versioned pack an owner can review, cite, enable/disable, and extend --
    never in Python. This module reads the pack; it knows no clinical term,
    no code, no scenario of its own (`tests/check_no_hardcoding.py`).
  * A convention names its AUTHORITY and its `verification_status`; a
    convention that cannot name its source is a guess, not a rule
    (CLAUDE.md, "New deterministic rules are config, not code"). The pack
    itself never names a medical code -- it is expressed in descriptor
    grammar (which AXIS, which VALUE) and in what the record documents
    (`applies_when`), exactly like `data/rules/validator_rules.json`.
  * A convention only ever AUTHORIZES ONE VALUE ON ONE AXIS for a fact whose
    own source-confirmed evidence matches its `applies_when`; it never
    selects a code, never overrides a documented value, and every use is
    recorded in the line's audit record naming the convention and its
    authority, so a reviewer can see which releases rest on a convention
    rather than on the record's own words.

Consumers: `resolution._chosen_own_requirements_confirmed` (a selected
candidate's own required axis) and `tiebreak.narrow` (an axis the page states
nothing about, so a genuine tie can settle) -- both only reached AFTER the
record's own evidence has been checked and found silent.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.core import config

from . import graph_consensus as _gc

# The DECLARED `coding_conventions` release source (`app.core.config`,
# registered in `app.release.source_manifest`), not a parallel path literal
# that could drift from the identity the release fingerprint attests -- the
# same rule `app.validation.rule_engine.RULES_FILE` is held to. A module
# attribute (not inlined at each call) so an offline tool can repoint it to
# replay a candidate pack.
PACK_PATH = config.CODING_CONVENTIONS_FILE


@dataclass(frozen=True)
class ConventionMatch:
    convention_id: str
    axis: str
    value: str
    authority: str
    verification_status: str
    matched_text: str

    def as_record(self) -> dict[str, Any]:
        return {"convention_id": self.convention_id, "axis": self.axis,
                "value": self.value, "authority": self.authority,
                "verification_status": self.verification_status,
                "matched_text": self.matched_text}

    def describe(self) -> str:
        return (f"axis {self.axis!r} = {self.value!r} authorized by governed coding "
                f"convention {self.convention_id!r} [{self.authority}; "
                f"verification_status={self.verification_status}] -- the record states "
                f"{self.matched_text!r}")


@lru_cache(maxsize=8)
def load_pack(path: str | None = None) -> tuple[dict, ...]:
    """Every ENABLED convention in the pack, in pack order. An absent pack is an
    empty tuple, never an error -- a deployment without conventions simply has
    none."""
    target = Path(path) if path else PACK_PATH
    if not target.exists():
        return ()
    payload = json.loads(target.read_text())
    rules = payload.get("conventions") or ()
    return tuple(dict(r) for r in rules if r.get("enabled", True))


def authorized_value(fact, axis: str, reconciliation=None, *,
                     candidate_descriptor: str = "",
                     pack_path: str | None = None) -> ConventionMatch | None:
    """The value a governed convention authorizes for `axis` on `fact`, or None.

    Fail-closed at every step: no pack, no convention for this axis, the
    fact's evidence not source-confirmed, a `fact_kinds`/`candidate_descriptor_
    regex` scope that does not apply, an `evidence_regex` that does not match
    the fact's OWN confirmed evidence, or an `absent_regex` that does -- each
    returns None. Matching is against the fact's own reconciled evidence text
    only, never the whole document and never a rival fact's evidence.
    """
    rules = [r for r in load_pack(pack_path) if str(r.get("axis") or "") == axis]
    if not rules:
        return None
    supported, _proof, text, _spans = _gc.source_support(fact, reconciliation)
    if not supported or not text:
        return None
    kind = str(getattr(getattr(fact, "kind", None), "value", "") or "")
    for rule in rules:
        when = rule.get("applies_when") or {}
        kinds = when.get("fact_kinds")
        if kinds and kind not in kinds:
            continue
        cand_re = when.get("candidate_descriptor_regex")
        if cand_re and not re.search(cand_re, candidate_descriptor or "", re.IGNORECASE):
            continue
        ev_re = when.get("evidence_regex")
        if not ev_re:
            continue
        match = re.search(ev_re, text, re.IGNORECASE)
        if not match:
            continue
        absent = when.get("absent_regex")
        if absent and re.search(absent, text, re.IGNORECASE):
            continue
        return ConventionMatch(
            convention_id=str(rule.get("id") or ""), axis=axis,
            value=str(rule.get("value") or ""),
            authority=str(rule.get("authority") or ""),
            verification_status=str(rule.get("verification_status") or ""),
            matched_text=match.group(0))
    return None
