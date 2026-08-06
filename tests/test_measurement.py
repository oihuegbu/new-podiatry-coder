"""Typed measurements + dimension-guarded elimination (Phase-0).

The safety property (finding 10): a unitless value, or one whose dimension differs from
a descriptor's interval, must NOT eliminate or deterministically prefer a candidate.
Agnostic — synthetic descriptors and generic units, no medical code."""
from claude_coder import measurement as meas
from claude_coder.models import CandidateCode, ClinicalFact, Disposition, EvidenceSpan, FactKind
from claude_coder.resolution import resolve
from claude_coder.data_access import MockSource
from claude_coder.models import ResolutionMethod


def _request(fact):
    from claude_coder.eligibility import (ClaimComponent, ClaimLineIntent,
                                          EligibilityState, RetrievalRequest)
    if not fact.fact_id:
        fact.fact_id = "fact"
    intent = ClaimLineIntent(
        intent_id=f"test-{fact.fact_id}", encounter_id="test",
        component=ClaimComponent.SERVICE, clinical_event_ids=[fact.fact_id],
        fact_kind=fact.kind.value, clinical_action=fact.description,
        attributes=dict(fact.attributes), date_of_service=None,
        billing_entity_id=None, source_span_ids=[],
        state=EligibilityState.ELIGIBLE_FOR_RETRIEVAL)
    return RetrievalRequest(intent, fact)


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
    line = resolve(_request(fact), src)
    assert line.method is ResolutionMethod.DETERMINISTIC and line.chosen.code == "SUP_MED"


def test_incompatible_dimension_measurement_does_not_eliminate():
    """Safety: a 30 mm LENGTH must not be compared against 'sq in' AREA intervals, so it
    must NOT collapse to the deterministic SUP_MED pick that the sq-in value produced."""
    src = MockSource(retrieval={("*", "hcpcs"): _dressing_candidates()})
    fact = ClinicalFact(FactKind.SUPPLY, "wound dressing", attributes={"depth_mm": 30},
                        disposition=Disposition.PERFORMED,
                        evidence=[EvidenceSpan("wound dressing applied, depth 30 mm")],
                        confidence=0.99)
    line = resolve(_request(fact), src)
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
    line = resolve(_request(fact), src)
    assert not (line.method is ResolutionMethod.DETERMINISTIC
                and line.chosen is not None and line.chosen.code == "SUP_MED")


# ---- Codex review F4: dimension-guarded specificity + semantic-role matching ----
def test_measurement_for_dimension_selects_by_dimension():
    assert meas.measurement_for_dimension({"size_sqin": 30}, "area").value == 30
    assert meas.measurement_for_dimension({"depth_mm": 30}, "area") is None       # no area value
    assert meas.measurement_for_dimension({"depth_mm": 30}, "length").value == 30


def test_measurement_for_dimension_ambiguous_role_returns_none():
    """Two same-dimension measurements (width vs depth) against one axis are ambiguous ->
    no deterministic comparison."""
    assert meas.measurement_for_dimension({"width_cm": 3, "depth_cm": 5}, "length") is None
    assert len(meas.measurements_of({"width_cm": 3, "depth_cm": 5})) == 2


def test_length_measurement_gives_no_specificity_to_area_candidate():
    """Codex F4 reproduction: an area-qualified candidate and a plain one at equal recall;
    a LENGTH measurement must NOT award the area candidate specificity and deterministically
    select it (the old unit-blind specificity path did)."""
    area = CandidateCode("AREA_C", "hcpcs",
                         "wound dressing, sterile, size 16 sq. in. or less, each", 0.9)
    plain = CandidateCode("PLAIN_C", "hcpcs", "wound dressing, sterile, each", 0.9)
    src = MockSource(retrieval={("*", "hcpcs"): [area, plain]})
    fact = ClinicalFact(FactKind.SUPPLY, "wound dressing", attributes={"depth_mm": 5},
                        disposition=Disposition.PERFORMED,
                        evidence=[EvidenceSpan("wound dressing applied, depth 5 mm")],
                        confidence=0.99)
    line = resolve(_request(fact), src)
    assert not (line.method is ResolutionMethod.DETERMINISTIC
                and line.chosen is not None and line.chosen.code == "AREA_C")


# ---- Codex review F4-R1: bounded interval must not close without a supporting measurement -
def test_interval_unsupported_helper():
    from claude_coder.resolution import _interval_unsupported
    from claude_coder.ontology import parse_descriptor
    feats = parse_descriptor("size 16 sq. in. or less")
    length = ClinicalFact(FactKind.SUPPLY, "d", attributes={"depth_mm": 5})
    area = ClinicalFact(FactKind.SUPPLY, "d", attributes={"size_sqin": 10})
    missing = ClinicalFact(FactKind.SUPPLY, "d", attributes={})
    assert _interval_unsupported(length, feats) is True        # wrong dimension
    assert _interval_unsupported(missing, feats) is True       # no measurement at all
    assert _interval_unsupported(area, feats) is False         # compatible + in range
    assert _interval_unsupported(missing, parse_descriptor("plain dressing")) is False  # no interval


def test_single_authoritative_interval_hit_without_supporting_measurement_abstains():
    """Codex F4-R1: a LONE authoritative interval-qualified SUPPLY (bypasses propose-then-
    verify) must NOT close deterministically when the documentation has no dimension-
    compatible measurement -- it abstains with a provider-query documentation gap."""
    area = CandidateCode("AREA_C", "hcpcs",
                         "wound dressing, sterile, size 16 sq. in. or less", 0.9)
    src = MockSource(retrieval={("*", "hcpcs"): [area]})
    fact = ClinicalFact(FactKind.SUPPLY, "wound dressing", attributes={"depth_mm": 5},
                        disposition=Disposition.PERFORMED,
                        evidence=[EvidenceSpan("wound dressing applied, depth 5 mm")],
                        confidence=0.99)
    line = resolve(_request(fact), src)
    assert line.chosen is None                                 # not billed deterministically
    assert line.documentation_gap                              # provider query for the measurement


def test_supported_interval_hit_still_resolves():
    area = CandidateCode("AREA_C", "hcpcs",
                         "wound dressing, sterile, size 16 sq. in. or less", 0.9)
    src = MockSource(retrieval={("*", "hcpcs"): [area]})
    fact = ClinicalFact(FactKind.SUPPLY, "wound dressing", attributes={"size_sqin": 10},
                        disposition=Disposition.PERFORMED,
                        evidence=[EvidenceSpan("wound dressing 10 sq in applied")],
                        confidence=0.99)
    line = resolve(_request(fact), src)
    assert line.chosen is not None and line.chosen.code == "AREA_C"


def test_same_dimension_wrong_semantic_role_does_not_support_interval():
    cand = CandidateCode("DEPTH_C", "hcpcs", "Dressing, depth more than 1 mm", 1.0)
    src = MockSource(records={("DEPTH_C", "hcpcs"): {"active": True}},
                     retrieval={("*", "hcpcs"): [cand]})
    fact = ClinicalFact(FactKind.SUPPLY, "dressing", attributes={"width_mm": "2 mm"},
                        disposition=Disposition.PERFORMED,
                        evidence=[EvidenceSpan("dressing width 2 mm")], confidence=0.99)
    line = resolve(_request(fact), src)
    assert line.chosen is None and line.documentation_gap


def test_verified_path_cannot_override_unsupported_interval():
    cand = CandidateCode("AREA_C", "hcpcs", "Dressing, area more than 1 sq in", 1.0)
    src = MockSource(records={("AREA_C", "hcpcs"): {"active": True}},
                     retrieval={("*", "hcpcs"): [cand]})
    fact = ClinicalFact(FactKind.SUPPLY, "dressing", attributes={"depth_mm": "2 mm"},
                        disposition=Disposition.PERFORMED,
                        evidence=[EvidenceSpan("dressing depth 2 mm")], confidence=0.99)

    def agree(system, _user):
        if "propose" in system.lower():
            return '{"codes":[]}'
        if "independently" in system.lower():
            return '{"entailed":true,"missing_element":false,"reason":"agree"}'
        return '{"choice":1,"reason":"agree"}'

    line = resolve(_request(fact), src, llm=agree, corroborate=agree)
    assert line.chosen is None and line.documentation_gap


# ---- Codex F4-R1 re-review: verified path must also abstain (model agreement != support) -
def test_unsupported_interval_not_billed_via_verified_path():
    """Even with select + independent corroboration AGREEING, a code whose bounded interval
    lacks a dimension-compatible documented measurement must NOT bill through the verified
    path. The required-constraint gate applies regardless of model agreement, fact kind, or
    candidate source. Covers both the one-model and corroborated verified paths."""
    proc = CandidateCode("PROC_RANGE", "cpt", "excision, area 16 sq. cm. or less", 0.9)
    src = MockSource(
        records={("PROC_RANGE", "cpt"): {"long_description": "excision, area 16 sq. cm. or less",
                                         "active": True}},
        retrieval={("*", "cpt"): [proc]})
    fact = ClinicalFact(FactKind.PROCEDURE, "excision", attributes={"depth_mm": 5},
                        disposition=Disposition.PERFORMED, fact_id="fx",
                        evidence=[EvidenceSpan("excision performed, depth 5 mm")],
                        confidence=0.99)

    def agree(system, user):
        sl = system.lower()
        if "propose" in sl:
            return '{"codes": []}'
        if "independently" in sl:
            return '{"entailed": true, "missing_element": false, "reason": "x"}'
        return '{"choice": 1, "reason": "x"}'

    corroborated = resolve(_request(fact), src, llm=agree, corroborate=agree)
    assert corroborated.chosen is None and corroborated.documentation_gap
    one_model = resolve(_request(fact), src, llm=agree)
    assert one_model.chosen is None and one_model.documentation_gap


# ---- role-vocabulary robustness: descriptor and note need not use the same role word ----
def test_constraint_dimension_level_role_accepts_any_vocabulary():
    """A dimension-level axis ("area") is satisfied by a measurement of that dimension
    however it is documented -- "size 10 sq cm" satisfies an "area ... sq cm" descriptor.
    Regression: the old exact role-word gate falsely abstained this valid, in-range case."""
    assert meas.measurement_for_constraint({"size_sqcm": 10}, "area", "area").value == 10
    assert meas.measurement_for_constraint({"area_sqcm": 10}, "area", "size").value == 10
    assert meas.measurement_for_constraint({"size_sqin": 30}, "area", None).value == 30


def test_constraint_specific_subaxis_still_requires_matching_axis():
    """A specific sub-axis constraint keeps its safety: a width does not satisfy a depth,
    the matching axis is selected among several lengths, and a bare dimension role over two
    same-dimension measurements is ambiguous -> None."""
    assert meas.measurement_for_constraint({"width_cm": 3}, "length", "depth") is None
    assert meas.measurement_for_constraint(
        {"width_cm": 3, "depth_mm": 5}, "length", "depth").value == 5
    assert meas.measurement_for_constraint(
        {"width_cm": 3, "depth_mm": 5}, "length", "length") is None


def test_supported_area_vocabulary_measurement_resolves_end_to_end():
    """End-to-end regression: an area-qualified candidate ("area 16 sq cm or less") with a
    documented in-range measurement written as "size" (size_sqcm=10) must RESOLVE, not
    abstain. The descriptor and the note used different words for the same physical
    dimension; that vocabulary gap must not hold a valid, in-range claim."""
    cand = CandidateCode("AREA_C", "cpt", "excision, area 16 sq. cm. or less", 0.9)
    src = MockSource(
        records={("AREA_C", "cpt"): {"long_description": "excision, area 16 sq. cm. or less",
                                     "active": True}},
        retrieval={("*", "cpt"): [cand]})
    fact = ClinicalFact(FactKind.PROCEDURE, "excision", attributes={"size_sqcm": 10},
                        disposition=Disposition.PERFORMED,
                        evidence=[EvidenceSpan("excision performed")], confidence=0.99)
    line = resolve(_request(fact), src)
    assert line.chosen is not None and line.chosen.code == "AREA_C"
