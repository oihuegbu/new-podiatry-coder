"""Typed measurements with dimension-guarded comparison (Phase-0).

`ontology.measurement_of` returns a BARE number; its own docstring notes a production
build must "carry units through and convert." A unitless number, or one whose dimension
differs from a descriptor's interval, must NOT eliminate or deterministically prefer a
candidate (a 30 mm length is not comparable to a "<=16 sq in" area). This module carries
value + unit + DIMENSION, and comparison is allowed ONLY when dimensions match and units
are convertible — otherwise it returns None (the caller then declines to act on it).

Agnostic: generic unit vocabulary (area/length/mass), never a medical code or a code
range. The unit tables are ordinary physical conversions.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_NUM = r"(-?\d+(?:\.\d+)?)"

# unit (normalized: lowercased, dots/spaces/underscores removed) -> factor to the
# dimension's CANONICAL unit. Area -> sq cm; length -> mm; mass -> mg.
_AREA = {"sqin": 6.4516, "squareinch": 6.4516, "squareinches": 6.4516, "in2": 6.4516,
         "sqcm": 1.0, "squarecentimeter": 1.0, "squarecentimeters": 1.0, "cm2": 1.0}
_LENGTH = {"mm": 1.0, "cm": 10.0, "m": 1000.0, "in": 25.4, "inch": 25.4, "inches": 25.4}
_MASS = {"mcg": 0.001, "microgram": 0.001, "mg": 1.0, "milligram": 1.0,
         "g": 1000.0, "gram": 1000.0, "grams": 1000.0, "kg": 1_000_000.0}
_DIM: dict[str, dict[str, float]] = {"area": _AREA, "length": _LENGTH, "mass": _MASS}

# The single source of truth for descriptor/measurement ROLE vocabulary. Each role word
# maps to its physical dimension; a role whose value differs from the word itself is a
# SPECIFIC sub-axis of that dimension (depth/width/diameter/thickness/height are sub-axes
# of length), while a role equal to its dimension (area, length) is dimension-level.
# ontology.parse_descriptor derives its recognized-role grammar from RECOGNIZED_ROLES, so
# the two modules can never diverge (Codex F6-R6: height was recognized by the grammar but
# absent here, letting a width satisfy a height-constrained descriptor).
_DIM_WORDS = {"area": "area", "depth": "length", "length": "length",
              "thickness": "length", "diameter": "length", "width": "length",
              "height": "length"}
# Generic magnitude words that name no specific sub-axis (their dimension follows from the
# unit, not the word); dimension-level, never a distinguishing sub-axis.
_GENERIC_ROLES = ("size",)
# Every role word the descriptor grammar and the measurement selector both recognize.
RECOGNIZED_ROLES = tuple(_DIM_WORDS) + _GENERIC_ROLES


def _norm_unit(text: str) -> str:
    return re.sub(r"[.\s_]+", "", str(text).lower())


def unit_dimension(unit: str | None) -> tuple[str | None, float | None]:
    """(dimension, factor-to-canonical) for a unit token, or (None, None) if unknown."""
    if not unit:
        return None, None
    u = _norm_unit(unit)
    for dim, table in _DIM.items():
        if u in table:
            return dim, table[u]
    return None, None


def _detect_unit(text: str) -> tuple[str | None, str | None]:
    """Find a known unit in free text -> (canonical-lookup unit, dimension), matching
    WHOLE tokens only (never a substring), so 'dressing' does not read as 'g' and
    'things' does not read as 'in'. Adjacent tokens are also joined ('sq'+'in' ->
    'sqin') so multi-word units are caught. Area/mass are checked before length so
    'square inch' is area, not 'inch' length."""
    toks = re.findall(r"[a-z0-9]+", str(text).lower())
    cands = set(toks)
    for i in range(len(toks) - 1):
        cands.add(toks[i] + toks[i + 1])            # 'sq'+'in' -> 'sqin'
    for dim in ("area", "mass", "length"):          # area/mass first (multi-word units)
        for u in _DIM[dim]:
            if u in cands:
                return u, dim
    return None, None


@dataclass(frozen=True)
class Measurement:
    value: float
    unit: str | None = None
    dimension: str | None = None
    semantic_role: str | None = None
    source_attribute: str | None = None
    evidence_span_id: str | None = None


def parse_measurement(raw, key: str | None = None,
                      role: str | None = None) -> Measurement | None:
    """A typed measurement from a value (and optional attribute key that may carry the
    unit, e.g. 'size_sqin'). Dimension precedence: explicit unit in the value, then a
    unit token in the key, then a dimension WORD in the key. Unknown -> dimension None
    (unitless: not comparable)."""
    m = re.search(_NUM, str(raw))
    if not m:
        return None
    value = float(m.group(1))
    unit, dim = _detect_unit(str(raw))
    if dim is None and key:
        unit, dim = _detect_unit(str(key))
    if dim is None and key:
        for w, d in _DIM_WORDS.items():
            if w in str(key).lower():
                dim = d
                break
    return Measurement(value=value, unit=unit, dimension=dim,
                       semantic_role=role or (str(key).lower() if key else None),
                       source_attribute=(str(key) if key else None))


def typed_measurement_of(attributes: dict) -> Measurement | None:
    """The typed analog of ontology.measurement_of: the first size/area/length-like
    attribute, carrying its unit + dimension (from the value or the key). None when no
    such attribute exists."""
    for key, val in (attributes or {}).items():
        k = str(key).lower()
        if any(w in k for w in ("area", "size", "measure", "depth", "length", "sq")):
            mm = parse_measurement(val, key=key)
            if mm is not None:
                return mm
    return None


def same_dimension(a: Measurement, b: Measurement) -> bool:
    return (a.dimension is not None and b.dimension is not None
            and a.dimension == b.dimension)


def convert(value: float, dimension: str, from_unit: str | None,
            to_unit: str | None) -> float | None:
    """Convert a value within ONE dimension. None if either unit is unknown for that
    dimension (never a cross-dimension or unit-blind conversion)."""
    table = _DIM.get(dimension or "")
    if not table:
        return None
    fu, tu = _norm_unit(from_unit or ""), _norm_unit(to_unit or "")
    if fu not in table or tu not in table:
        return None
    return value * table[fu] / table[tu]


def compare(a: Measurement, b: Measurement) -> int | None:
    """-1/0/1 ordering of a vs b, ONLY when dimensions match and units convert; else
    None (incomparable — the caller must not act on it)."""
    if not same_dimension(a, b):
        return None
    bv = convert(b.value, b.dimension, b.unit, a.unit) if a.unit and b.unit else None
    if bv is None:
        # same dimension but a unit missing -> comparable only if units are identical
        if a.unit != b.unit:
            return None
        bv = b.value
    return (a.value > bv) - (a.value < bv)


def measurements_of(attributes: dict) -> list["Measurement"]:
    """EVERY typed measurement in a fact's attributes (each carrying value/unit/dimension/
    role). Unlike typed_measurement_of (first match), this returns all of them so the
    caller can select the one whose DIMENSION fits a descriptor axis."""
    out: list[Measurement] = []
    for key, val in (attributes or {}).items():
        k = str(key).lower()
        if any(w in k for w in ("area", "size", "measure", "depth", "length", "sq",
                                "width", "diameter", "thickness", "height")):
            m = parse_measurement(val, key=key)
            if m is not None:
                out.append(m)
    return out


def measurement_for_dimension(attributes: dict, dimension: str):
    """The UNIQUE documented measurement of `dimension`, or None. None when NO measurement
    matches (incompatible dimension) OR when MORE THAN ONE does (ambiguous role -- e.g.
    width vs depth against a length axis): an ambiguous or absent match must never drive a
    deterministic comparison."""
    matches = [m for m in measurements_of(attributes) if m.dimension == dimension]
    return matches[0] if len(matches) == 1 else None


def measurement_for_constraint(attributes: dict, dimension: str,
                               semantic_role: str | None):
    """The unique measurement satisfying a descriptor's dimensional constraint.

    Always gated on physical DIMENSION (a length never satisfies an area constraint) and
    on UNIQUENESS (an ambiguous match cannot drive deterministic/verified selection). Role
    gating is asymmetric:

    - A SPECIFIC sub-axis constraint -- depth / width / diameter / thickness, i.e. a role
      word present in _DIM_WORDS whose dimension differs from the word itself -- must be met
      by a measurement carrying that SAME axis. A width is not a depth merely because both
      are lengths; among several same-dimension measurements the matching axis is selected.
    - A DIMENSION-LEVEL constraint -- the role names the dimension itself (e.g. "area"), or
      is a generic magnitude ("size") or absent -- is satisfied by the UNIQUE measurement of
      that dimension regardless of the documenting vocabulary, so "size 10 sq cm" satisfies
      an "area ... sq cm" descriptor. Two measurements of the dimension remain ambiguous.

    (Codex review F4-R1; role-vocabulary robustness -- descriptor and note need not use the
    same word for the same physical dimension.)
    """
    role = str(semantic_role or "").strip().lower()
    specific = role in _DIM_WORDS and _DIM_WORDS[role] != role
    matches = []
    for m in measurements_of(attributes):
        if m.dimension != dimension:
            continue
        if specific:
            source = str(m.source_attribute or m.semantic_role or "").lower()
            words = re.findall(r"[a-z]+", source)
            fact_role = next(
                (w for w in words if w in _DIM_WORDS or w in _GENERIC_ROLES), None)
            if fact_role != role:
                continue
        matches.append(m)
    return matches[0] if len(matches) == 1 else None
