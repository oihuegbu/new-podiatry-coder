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
