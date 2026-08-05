"""Typed measurements + dimension-guarded elimination (Phase-0).

The safety property (finding 10): a unitless value, or one whose dimension differs from
a descriptor's interval, must NOT eliminate or deterministically prefer a candidate.
Agnostic — synthetic descriptors and generic units, no medical code."""
from claude_coder import measurement as meas
from claude_coder.models import CandidateCode, ClinicalFact, Disposition, EvidenceSpan, FactKind
from claude_coder.resolution import resolve
from claude_coder.data_access import MockSource
from claude_coder.models import ResolutionMethod


# ---------------------------------------------------------------- the primitive
def test_parse_detects_dimension_from_value_or_key():
    assert meas.parse_measurement("30 sq in").dimension == "area"
    assert meas.parse_measurement("5 mm").dimension == "length"
    assert meas.parse_measurement("30 mg").dimension == "mass"
    assert meas.parse_measurement(30, key="size_sqin").dimension == "area"   # unit from key
    assert meas.parse_measurement(5, key="depth_mm").dimension == "length"
    # a bare number with no unit anywhere -> unitless (dimension unknown)
    assert meas.parse_measurement(30, key="size").dimension is None
    assert meas.parse_measurement("no number here") is None


def test_unit_detection_is_whole_token_not_substring():
    # a stray word containing a unit's letters must NOT be read as that unit
    assert meas.parse_measurement("30 dressings").dimension is None       # not mass 'g'
    assert meas.parse_measurement("wound with 12 things").dimension is None  # not 'in'
    # real adjacent-token / multi-word units still parse correctly
    assert meas.parse_measurement("30 sq in").dimension == "area"
    assert meas.parse_measurement("square inch wound 12").dimension == "area"  # not 'inch' length


def test_typed_measurement_of_attributes():
    assert meas.typed_measurement_of({"size_sqin": 30}).dimension == "area"
    assert meas.typed_measurement_of({"depth_mm": 5}).dimension == "length"
    assert meas.typed_measurement_of({"size": 30}).dimension is None       # unitless
    assert meas.typed_measurement_of({"note": "hello"}) is None            # not a measure key


def test_convert_within_dimension_only():
    assert meas.convert(3, "length", "cm", "mm") == 30                     # 3 cm = 30 mm
    assert abs(meas.convert(1, "area", "sqin", "sqcm") - 6.4516) < 1e-6
    assert meas.convert(3, "length", "cm", "sqin") is None                 # cross-dimension
    assert meas.convert(3, "length", "cm", "furlong") is None              # unknown unit


def test_compare_is_dimension_guarded():
    area = meas.Measurement(30, "sqin", "area")
    length = meas.Measurement(30, "mm", "length")
    assert meas.compare(area, length) is None                             # incompatible -> None
    a3cm = meas.Measurement(3, "cm", "length")
    a30mm = meas.Measurement(30, "mm", "length")
    assert meas.compare(a3cm, a30mm) == 0                                 # 3 cm == 30 mm
    assert meas.compare(meas.Measurement(4, "cm", "length"), a30mm) == 1
    # unitless never compares
    assert meas.compare(meas.Measurement(30, None, None), area) is None


# ---------------------------------------------------------------- integration
def _dressing_candidates():
    return [
        CandidateCode("SUP_SMALL", "hcpcs",
                      "Wound dressing, sterile, size 16 sq. in. or less, each", 0.9),
        CandidateCode("SUP_MED", "hcpcs",
                      "Wound dressing, sterile, size more than 16 sq. in. but less than "
                      "or equal to 48 sq. in., each", 0.9),
        CandidateCode("SUP_LARGE", "hcpcs",
                      "Wound dressing, sterile, size more than 48 sq. in., each", 0.9),
    ]


def test_same_dimension_measurement_still_eliminates():
    """The valid case is preserved: 30 sq in eliminates the <=16 and >48 leaves and
    deterministically selects the 16-48 leaf."""
    src = MockSource(retrieval={("*", "hcpcs"): _dressing_candidates()})
    fact = ClinicalFact(FactKind.SUPPLY, "wound dressing", attributes={"size_sqin": 30},
                        disposition=Disposition.PERFORMED,
                        evidence=[EvidenceSpan("wound dressing 30 sq in applied")],
                        confidence=0.99)
    line = resolve(fact, src)
    assert line.method is ResolutionMethod.DETERMINISTIC and line.chosen.code == "SUP_MED"


def test_incompatible_dimension_measurement_does_not_eliminate():
    """Safety: a 30 mm LENGTH must not be compared against 'sq in' AREA intervals, so it
    must NOT collapse to the deterministic SUP_MED pick that the sq-in value produced."""
    src = MockSource(retrieval={("*", "hcpcs"): _dressing_candidates()})
    fact = ClinicalFact(FactKind.SUPPLY, "wound dressing", attributes={"depth_mm": 30},
                        disposition=Disposition.PERFORMED,
                        evidence=[EvidenceSpan("wound dressing applied, depth 30 mm")],
                        confidence=0.99)
    line = resolve(fact, src)
    assert not (line.method is ResolutionMethod.DETERMINISTIC
                and line.chosen is not None and line.chosen.code == "SUP_MED")


def test_unitless_measurement_does_not_eliminate():
    """A bare, unitless number must not eliminate a candidate against a unit-bearing
    interval."""
    src = MockSource(retrieval={("*", "hcpcs"): _dressing_candidates()})
    fact = ClinicalFact(FactKind.SUPPLY, "wound dressing", attributes={"size": 30},
                        disposition=Disposition.PERFORMED,
                        evidence=[EvidenceSpan("wound dressing size 30 applied")],
                        confidence=0.99)
    line = resolve(fact, src)
    assert not (line.method is ResolutionMethod.DETERMINISTIC
                and line.chosen is not None and line.chosen.code == "SUP_MED")
