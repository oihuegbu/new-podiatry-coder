"""Descriptor ontology — structure DERIVED from the authoritative descriptors.

A deterministic coder must decide by STRUCTURE, not by fuzzy similarity. But the
structure need not be authored by hand: the authoritative descriptor already
encodes it. "Collagen dressing, sterile, size more than 16 sq. in. but less than
or equal to 48 sq. in., each" contains a measurement INTERVAL (16, 48]; a
"...right foot" descriptor encodes laterality; "single, each" encodes
cardinality. This module parses those features out of the descriptor text so the
resolver can match a fact's documented attributes against them field by field.

This is how the size-range family selection (A6020-style) that the old pipeline
HARDCODED becomes deterministic and data-driven: the interval comes from each
code's own descriptor, so it self-updates and needs no code list. There is no
medical code in this file — only a parser over descriptor grammar.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_LATERALITY = {"left", "right", "bilateral"}
_CARDINALITY = ("bilateral", "pair", "single", "each", "per")

# words that qualify (not the core concept) — stripped when comparing concepts
_QUALIFIER = _LATERALITY | {
    "single", "each", "per", "pair", "sterile", "size", "sq", "in", "cm", "mm",
    "square", "inch", "inches", "more", "than", "less", "greater", "equal", "to",
    "or", "and", "but", "up", "at", "least", "not", "the", "of", "with", "without",
    "for", "only", "including", "follow", "supply", "unspecified",
}


@dataclass(frozen=True)
class Interval:
    low: float | None = None
    high: float | None = None
    low_inc: bool = True
    high_inc: bool = True
    unit: str | None = None

    def contains(self, x: float) -> bool:
        if self.low is not None:
            if (x < self.low) or (not self.low_inc and x == self.low):
                return False
        if self.high is not None:
            if (x > self.high) or (not self.high_inc and x == self.high):
                return False
        return True

    def bounded(self) -> bool:
        return self.low is not None or self.high is not None


@dataclass
class DescriptorFeatures:
    raw: str
    core_tokens: set[str] = field(default_factory=set)
    laterality: set[str] = field(default_factory=set)
    cardinality: str | None = None
    interval: Interval | None = None


_NUM = r"(\d+(?:\.\d+)?)"
_UNIT = re.compile(r"(sq\.?\s*(?:in|cm)\.?|square\s+(?:inch|centimeter)s?|cm|mm)")


def _to_float(s: str) -> float:
    return float(s)


def _parse_interval(text: str) -> Interval | None:
    t = re.sub(r"\s+", " ", text.lower())
    low = high = None
    low_inc = high_inc = True

    # upper bounds
    m = re.search(rf"less than or equal to {_NUM}", t) or re.search(rf"{_NUM}\s*(?:sq\.?\s*in\.?|sq\.?\s*cm\.?|cm|mm)?\s*or less", t) or re.search(rf"up to {_NUM}", t) or re.search(rf"not more than {_NUM}", t)
    if m:
        high, high_inc = _to_float(m.group(1)), True
    else:
        m = re.search(rf"less than {_NUM}", t)
        if m:
            high, high_inc = _to_float(m.group(1)), False

    # lower bounds
    m = re.search(rf"more than {_NUM}", t) or re.search(rf"greater than {_NUM}", t) or re.search(rf"over {_NUM}", t)
    if m:
        low, low_inc = _to_float(m.group(1)), False
    else:
        m = re.search(rf"at least {_NUM}", t) or re.search(rf"{_NUM}\s*(?:sq\.?\s*in\.?|cm|mm)?\s*or more", t)
        if m:
            low, low_inc = _to_float(m.group(1)), True

    if low is None and high is None:
        return None
    unit_m = _UNIT.search(t)
    return Interval(low=low, high=high, low_inc=low_inc, high_inc=high_inc,
                    unit=(unit_m.group(1) if unit_m else None))


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def parse_descriptor(descriptor: str) -> DescriptorFeatures:
    laterality = {w for w in _LATERALITY
                  if re.search(rf"\b{w}\b", descriptor.lower())}
    cardinality = next((c for c in _CARDINALITY
                        if re.search(rf"\b{c}\b", descriptor.lower())), None)
    interval = _parse_interval(descriptor)
    core = {t for t in _tokens(descriptor)
            if t not in _QUALIFIER and not t.isdigit() and len(t) > 2}
    return DescriptorFeatures(raw=descriptor, core_tokens=core,
                              laterality=laterality, cardinality=cardinality,
                              interval=interval)


_COUNT_RANGE = [
    re.compile(rf"{_NUM}\s*(?:to|through|-|–)\s*{_NUM}"),   # "2 to 4", "2-4"
]


def count_range(descriptor: str) -> tuple[int, int | None] | None:
    """A quantity RANGE the descriptor covers as one unit, e.g. '2 to 4 items'
    -> (2, 4), 'up to 4' -> (1, 4), 'N or more' -> (N, None). Parsed from the
    descriptor — this is how 'one unit for 2-4 items' is known without a code
    list."""
    d = re.sub(r"\s+", " ", descriptor.lower())
    for rx in _COUNT_RANGE:
        m = rx.search(d)
        if m:
            return int(float(m.group(1))), int(float(m.group(2)))
    m = re.search(rf"up to {_NUM}", d)
    if m:
        return 1, int(float(m.group(1)))
    m = re.search(rf"{_NUM}\s*or more", d)
    if m:
        return int(float(m.group(1))), None
    return None


def billing_units(documented_count: int, descriptor: str) -> int:
    """Billing UNITS for a line — not the raw documented count. A descriptor
    whose quantity RANGE covers the count is a single unit (e.g. a '2-4 items'
    code billed once for 2 items); an 'each'/'per' descriptor bills per item;
    otherwise a single unit. Purely descriptor-driven, so it self-updates."""
    n = max(1, int(documented_count or 1))
    rng = count_range(descriptor)
    if rng:
        lo, hi = rng
        if n >= lo and (hi is None or n <= hi):
            return 1                        # the code IS the range -> one unit
        return 1                            # outside range -> add-on territory; stay safe
    d = descriptor.lower()
    if re.search(r"\beach\b|\bper\b|\bsingle\b", d):
        return n
    return 1


# ── CPT section applicability (mechanic 1) ────────────────────────────────────
# A code's SECTION is not a field in the data, but the authoritative descriptor
# names it: the CPT Anesthesia section's descriptors are formulaic ("Anesthesia
# for procedures on …"). We read that grammar — exactly like reading a descriptor
# for its measurement interval or laterality — never a code range. The table is
# descriptor GRAMMAR, extensible as other sections gain a detectable signature;
# there is no medical code in it.
_SECTION_SIGNATURES: dict[str, tuple[str, ...]] = {
    # section name : descriptor-leading phrases that identify it
    "anesthesia": ("anesthesia for", "anesthesia,"),
}


def code_section(descriptor: str) -> str | None:
    """The CPT section a code belongs to, inferred from its authoritative
    descriptor grammar (not a code range). None when no signature matches."""
    d = re.sub(r"\s+", " ", str(descriptor or "").lower()).strip()
    for section, sigs in _SECTION_SIGNATURES.items():
        if any(d.startswith(s) for s in sigs):
            return section
    return None


def is_separate_procedure(descriptor: str) -> bool:
    """CPT '(separate procedure)' designation: the service is bundled when
    performed with a more extensive procedure of the same session. Read straight
    from the descriptor — a real CPT convention, no code list."""
    return "(separate procedure)" in str(descriptor or "").lower()


# ── descriptor ↔ fact token support (mechanic 2, ranking only) ────────────────
def support_score(descriptor: str, text: str) -> int:
    """How many of the DESCRIPTOR's distinctive concept tokens the documented
    text (fact description + evidence) also names — a safe RANK signal used only
    to break near-ties in recall. It never eliminates a candidate (terse/generic
    authoritative descriptors legitimately share few tokens with clinician
    phrasing — the same reason the resolver has no token FLOOR), so a correct but
    tersely-worded code can never be dropped by this; it only helps a
    concept-matching code win when recall is otherwise a wash."""
    from .terminology import _sing
    dtok = {_sing(t) for t in _tokens(descriptor)
            if t not in _QUALIFIER and not t.isdigit() and len(t) > 2}
    ttok = {_sing(t) for t in _tokens(text) if len(t) > 2}
    return len(dtok & ttok)


# ── drug dose -> billing units ────────────────────────────────────────────────
# Mass-unit conversion to a common base (mg). Volume/activity units ('ml', 'units',
# 'iu') only convert to themselves. Generic dosing vocabulary, not codes.
_MASS_TO_MG = {"mg": 1.0, "milligram": 1.0, "g": 1000.0, "gram": 1000.0,
               "mcg": 0.001, "microgram": 0.001, "ug": 0.001, "µg": 0.001}
_DOSE_RX = re.compile(
    r"(\d+(?:\.\d+)?)\s*(mg|milligram|mcg|microgram|ug|µg|g|gram|ml|milliliter|"
    r"cc|units?|iu|meq|mmol)\b", re.I)


def parse_dose(text: str) -> tuple[float, str] | None:
    """(amount, unit) for the first dose in free text ('30 mg', '1 g'), else None."""
    m = _DOSE_RX.search(str(text or ""))
    if not m:
        return None
    return float(m.group(1)), m.group(2).lower()


def _to_base(amount: float, unit: str) -> tuple[float, str]:
    """Normalize a mass dose to mg; leave other units as-is (self-comparable)."""
    u = unit.lower()
    if u in _MASS_TO_MG:
        return amount * _MASS_TO_MG[u], "mg"
    return amount, u


def drug_billing_units(documented: str, per_unit: dict | None) -> int | None:
    """Billing units for a dosed drug = documented total dose / the code's per-unit
    dose (e.g. 30 mg documented, 'per 15 mg' code -> 2 units). Unit-aware (mg/mcg/g
    convert; ml/units compare in-kind). None when the dose isn't documented or the
    units are incompatible — the caller then keeps the safe default of 1 unit."""
    if not per_unit:
        return None
    doc = parse_dose(documented)
    if not doc:
        return None
    d_amt, d_u = _to_base(doc[0], doc[1])
    p_amt, p_u = _to_base(float(per_unit.get("amount") or 0), str(per_unit.get("unit") or ""))
    if p_amt <= 0 or d_u != p_u:
        return None
    return max(1, round(d_amt / p_amt))


def measurement_of(attributes: dict) -> float | None:
    """Pull a single numeric measurement out of a fact's attributes (area,
    size, depth, length…). Structural, unit-agnostic here; a production build
    would carry units through and convert."""
    for key, val in attributes.items():
        k = str(key).lower()
        if any(w in k for w in ("area", "size", "measure", "depth", "length", "sq")):
            m = re.search(_NUM, str(val))
            if m:
                return float(m.group(1))
    return None
