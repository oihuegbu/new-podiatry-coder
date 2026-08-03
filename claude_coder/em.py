"""E/M leveling — data-driven from the MDM grid and the code descriptors.

The repo's `em_mdm_grid.json` is deliberately code-free: it holds the level
definitions and the 2-of-3 selection rule, and each E/M CODE's own descriptor
states which level it requires ("… LOW level of medical decision making …").
So E/M resolution is: (1) determine the MDM level from the documented elements
via the grid's 2-of-3 rule; (2) pick the retrieved E/M code whose descriptor
states that level and matches the new/established setting. No code is named
here, and no level table is hardcoded beyond the four standard MDM level names
(fixed AMA terminology, the same kind of generic vocabulary as left/right).

The three MDM elements (problems addressed, data reviewed, risk) come from the
extractor as documented levels; incomplete MDM -> no level -> review (fail-closed).
"""
from __future__ import annotations

from .data_access import CodeSource
from .models import ClinicalFact, ResolutionMethod, ResolvedLine

# Ordered MDM levels — the grid's own vocabulary, not codes.
LEVELS = ("straightforward", "low", "moderate", "high")


def mdm_level(problems: str, data: str, risk: str) -> str | None:
    """The MDM level met by 2 of the 3 elements (the grid's selection rule):
    the highest level L such that at least two elements are at L or above."""
    idx = [LEVELS.index(v) for v in (problems, data, risk) if v in LEVELS]
    if len(idx) < 3:
        return None                       # incomplete documentation -> cannot level
    best = None
    for L in range(len(LEVELS)):
        if sum(1 for i in idx if i >= L) >= 2:
            best = L
    return LEVELS[best] if best is not None else None


def _select(level: str, new_patient: bool | None, candidates) -> "object | None":
    """Among retrieved E/M candidates, the one whose descriptor states this MDM
    level and matches the new/established setting — read from the descriptor."""
    for c in candidates:
        d = c.descriptor.lower()
        if level not in d:
            continue
        if new_patient is None:
            return c
        is_new = "new patient" in d
        is_est = "established patient" in d
        if (new_patient and is_new) or (not new_patient and is_est) or not (is_new or is_est):
            return c
    return None


def resolve_em(fact: ClinicalFact, source: CodeSource, top_k: int = 40) -> ResolvedLine:
    """Resolve an evaluation-and-management fact to a level-appropriate code."""
    a = fact.attributes
    level = mdm_level(str(a.get("problems", "")).lower(),
                      str(a.get("data", "")).lower(),
                      str(a.get("risk", "")).lower())
    # time-based selection is the documented alternative; when total time is
    # recorded it can override MDM — a documented next step, not faked here.
    if level is None:
        return ResolvedLine(
            fact=fact, chosen=None, method=ResolutionMethod.ABSTAINED,
            rationale="MDM elements not fully documented — E/M level cannot be set")

    pool = source.retrieve(f"evaluation and management office visit {level} "
                           f"medical decision making", fact.system, top_k=top_k)
    new_patient = a.get("new_patient")
    if isinstance(new_patient, str):
        new_patient = new_patient.strip().lower() in ("true", "yes", "new")
    chosen = _select(level, new_patient, pool)
    if chosen is None:
        return ResolvedLine(
            fact=fact, chosen=None, alternatives=pool[:5],
            method=ResolutionMethod.ABSTAINED,
            rationale=f"no retrieved E/M code matches MDM level '{level}' — review")
    return ResolvedLine(
        fact=fact, chosen=chosen, method=ResolutionMethod.DETERMINISTIC,
        rationale=f"MDM level '{level}' by the 2-of-3 rule; descriptor states it")
