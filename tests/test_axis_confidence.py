"""Phase-0: per-axis confidence — a strong overall read must not mask a weak axis.

The gate uses the WEAKEST axis (min), never an average. Shadow-safe: with no per-axis
values, min_confidence == the scalar. Agnostic — synthetic axes, no medical code."""
from claude_coder.models import ClinicalFact, FactKind, CandidateCode, ResolvedLine, \
    ResolutionMethod, CodingResult, Disposition
from claude_coder.autonomy import decide, SHAKY_EXTRACTION
from claude_coder.data_access import MockSource


def _fact(conf, axes=None):
    return ClinicalFact(FactKind.PROCEDURE, "a performed service", confidence=conf,
                        axis_confidence=axes or {}, disposition=Disposition.PERFORMED)


# ---------------------------------------------------------------- the accessor
def test_min_confidence_falls_back_to_scalar_without_axes():
    assert _fact(0.9).min_confidence == 0.9
    assert _fact(0.9).weakest_axis is None


def test_min_confidence_is_the_weakest_axis_not_an_average():
    f = _fact(0.99, {"occurrence": 0.95, "action": 0.9, "performer": 0.2})
    assert f.min_confidence == 0.2               # NOT the ~0.76 average
    assert f.weakest_axis == "performer"


def test_scalar_included_in_the_minimum():
    f = _fact(0.3, {"occurrence": 0.9})
    assert f.min_confidence == 0.3               # scalar can be the weakest signal


# ---------------------------------------------------------------- autonomy gate
def _verified_line(fact):
    return ResolvedLine(fact=fact, chosen=CandidateCode("AA111", "cpt", "svc"),
                        method=ResolutionMethod.VERIFIED)


def test_weak_axis_routes_to_review_despite_high_overall():
    """A high scalar (0.99) with a weak performer axis (0.2) must still route to REVIEW —
    the strong average cannot conceal the weak axis."""
    res = CodingResult("enc", "2026-08-01",
                       lines=[_verified_line(_fact(0.99, {"performer": 0.2}))], gates=[])
    decide(res, source=MockSource())
    hits = [r for r in res.routing
            if r["destination"] == "REVIEW" and "weakest axis 'performer'" in r["reason"]]
    assert hits, res.routing


def test_high_all_axes_not_routed_by_shaky_floor():
    res = CodingResult("enc", "2026-08-01",
                       lines=[_verified_line(_fact(0.99, {"performer": 0.95, "action": 0.9}))],
                       gates=[])
    decide(res, source=MockSource())
    assert not [r for r in res.routing if "barely documents" in r["reason"]]


def test_phase0_no_axes_behaves_as_before():
    """Shadow safety: with no per-axis values, a high scalar is not routed by the floor
    and a low scalar still is — identical to the pre-change behavior."""
    hi = CodingResult("e", "2026-08-01", lines=[_verified_line(_fact(0.99))], gates=[])
    decide(hi, source=MockSource())
    assert not [r for r in hi.routing if "barely documents" in r["reason"]]
    lo = CodingResult("e", "2026-08-01", lines=[_verified_line(_fact(0.2))], gates=[])
    decide(lo, source=MockSource())
    assert [r for r in lo.routing if "barely documents" in r["reason"]]
