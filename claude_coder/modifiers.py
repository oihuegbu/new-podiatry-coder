"""Modifier assignment — data-driven, no modifier literal in the code.

The modifier VALUES (right-side, left-side, bilateral) are DISCOVERED from
modifiers.json by matching each modifier's own description ("Right side…"), so
the engine carries no hardcoded modifier and stays valid if the modifier set
changes. The LOGIC is agnostic: a documented laterality that the chosen code's
descriptor does not already encode earns the corresponding side/bilateral
modifier.

Scope today: laterality (RT/LT) and bilateral. Distinct-service (59 / X{EPSU})
and E/M-separate (25) are the next mechanics — assigned the same way, from
NCCI signals and the E/M-with-procedure relationship — and are noted, not faked.
"""
from __future__ import annotations

from .models import ClinicalFact


def load_modifier_defs() -> dict:
    """{modifier_code: {description: ...}} from the authoritative modifier file.
    Fail-safe: any problem (no app config, missing file) yields {} so the engine
    simply assigns no modifiers rather than erroring."""
    try:
        import json
        from app.core.config import DATA_DIR
        with open(DATA_DIR / "codes" / "modifiers.json") as fh:
            data = json.load(fh)
        return data.get("modifiers", {}) or {}
    except Exception:
        return {}


def _descr(entry) -> str:
    if isinstance(entry, dict):
        return str(entry.get("description") or entry.get("descriptor")
                   or entry.get("long_description") or "")
    return str(entry)


def _discover(defs: dict, *needles: str) -> str | None:
    """The modifier whose OWN description matches a needle — a data lookup, so
    the code below never names a modifier."""
    for code, entry in defs.items():
        low = _descr(entry).lower()
        if any(n in low for n in needles):
            return code
    return None


class ModifierEngine:
    def __init__(self, defs: dict | None = None) -> None:
        self._defs = defs if defs is not None else load_modifier_defs()
        self._right = _discover(self._defs, "right side")
        self._left = _discover(self._defs, "left side")
        self._bilateral = _discover(self._defs, "bilateral")

    def assign(self, fact: ClinicalFact, descriptor: str) -> list[str]:
        """Modifiers a resolved line earns from the documented facts. Empty when
        the descriptor already encodes the side/bilaterality (no double-coding)."""
        lat = str(fact.attributes.get("laterality", "")).lower().strip()
        desc = descriptor.lower()
        if "bilateral" in desc:
            return []
        if lat == "bilateral" and self._bilateral:
            return [self._bilateral]
        if lat == "right" and "right" not in desc and self._right:
            return [self._right]
        if lat == "left" and "left" not in desc and self._left:
            return [self._left]
        return []
