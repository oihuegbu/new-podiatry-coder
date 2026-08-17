"""Directive section 4 — the code-selection TIE POLICY, steps 3 to 5.

The directive states the whole policy as five ordered steps:

  1. eliminate candidates that fail a required constraint;
  2. select automatically ONLY if one candidate uniquely satisfies every required
     documented axis;
  3. if several remain, re-inspect ONLY their DISCRIMINATING axes against the
     ORIGINAL DOCUMENT;
  4. if one becomes uniquely entailed, release it;
  5. if the record genuinely lacks the distinguishing fact, issue ONE targeted
     provider query, or exclude/hold that line — never a generic coder review
     because candidates tied or two models disagreed.

Steps 1 and 2 belong to whoever owns the candidate pool (`resolution._evaluate`
eliminates a candidate that contradicts a documented axis; `resolution._decide`
selects only a UNIQUE satisfier). Steps 3 to 5 are this module, and they are the
part the pipeline had no implementation of: a tie previously fell through to a
retrieval-score margin, to a lexical-overlap tiebreak, to a single bounded model
pick, or to the generic coder-review recommendation — and the directive names
all four as things that must not decide a code.

Two properties make this a VERIFICATION rather than another guess:

* The axes are DERIVED from the tied candidates' own authoritative descriptors —
  an axis is whatever one descriptor asserts that the others do not. Nothing is
  enumerated by hand, and no code, code family, or clinical term appears here.
* The only text an axis is proven against is text the ORIGINAL PAGE was proven to
  say. That definition is not re-implemented here: it is
  `graph_consensus.source_support`, the same one the fact-extraction consensus
  uses, so a code tie and an extraction disagreement can never be settled on two
  different standards of evidence. A model's own prose, a retrieval score, and an
  agreement between two models are all inadmissible.

The two consensus mechanisms compose but never merge: `graph_consensus` settles
what the record SAYS (fact axes), this module settles which CODE those settled
facts uniquely support. Neither counts votes.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .models import CandidateCode
from .ontology import parse_descriptor
from .terminology import _sing

#: Axis identifiers for the audit record. This is NOT a closed list of things that
#: can differ: `AXIS_DESCRIPTOR_TERM` is open-ended by construction — it reports
#: whatever words the authoritative descriptors actually disagree on.
AXIS_LATERALITY = "laterality"
AXIS_MEASUREMENT = "measurement"
AXIS_DESCRIPTOR_TERM = "descriptor_term"

#: English and coding GRAMMAR that can never be a discriminating clinical axis:
#: function words; the classification grammar of a residual bucket (a record states a
#: condition, never the bucket its code lives in, so those words can never be proven
#: from a page); and the counting/unit vocabulary the measurement axis owns. Dropping
#: them keeps a tie's axis list down to facts a provider could actually be asked to
#: document. Every word here is grammar — there is no clinical vocabulary in it, and
#: "without" is deliberately KEPT, because a descriptor's negation is a real axis.
_GRAMMAR = frozenset({
    "a", "an", "and", "any", "are", "as", "at", "be", "by", "each", "for", "from",
    "in", "into", "is", "it", "its", "of", "on", "or", "per", "that", "than", "the",
    "these", "this", "those", "to", "was", "were", "when", "which", "with",
    "other", "specified", "unspecified", "elsewhere", "classified", "nos", "nec",
    "cm", "mm", "sq", "inch", "inche", "square", "size", "less", "greater", "more",
    "equal", "least", "up", "over", "not",
})


def _text_tokens(text: str) -> set[str]:
    """Every token of a PROVEN quotation, plus its singular form, so documented plural
    wording proves a singular descriptor term and vice versa. Short tokens are kept
    deliberately — a laterality word is short, and the alignment tokenizer's length
    floor would make every short axis unprovable (the same reasoning as
    `graph_consensus._value_tokens`)."""
    raw = {t for t in re.split(r"[^a-z0-9]+", (text or "").lower()) if t}
    return raw | {_sing(t) for t in raw}


def _descriptor_tokens(descriptor: str) -> set[str]:
    """A descriptor's potentially DISCRIMINATING words: singularized, with grammar and
    bare numbers removed. Laterality words are left in and stripped by the caller,
    which reports them as their own named axis."""
    out: set[str] = set()
    for tok in re.split(r"[^a-z0-9]+", str(descriptor or "").lower()):
        if not tok or tok.isdigit():
            continue
        s = _sing(tok)
        if s in _GRAMMAR or tok in _GRAMMAR:
            continue
        out.add(s)
    return out


@dataclass(frozen=True)
class AxisProbe:
    """One axis on which the tied candidates' authoritative descriptors say DIFFERENT
    things.

    `terms_by_code` maps each candidate to the words that — if the ORIGINAL DOCUMENT
    states them — show the document is describing THAT candidate. A candidate with no
    term on an axis (its descriptor is the silent, unqualified one) can never be
    positively proven by that axis: the absence of a word is not documentation, and
    quietly taking the less-qualified code because nobody wrote the qualifier is the
    same under-coding the propose-then-verify path already refuses to do silently.
    """

    axis: str
    terms_by_code: dict[str, tuple[str, ...]]
    #: False for an axis no quotation's WORDS can settle (a numeric interval, whose
    #: satisfaction is a typed, unit-converted comparison owned by `resolution`). Such
    #: an axis still names itself in the provider query, and — because release requires
    #: every axis to be settled — it always holds the line. It never selects a code.
    provable: bool = True

    def as_record(self) -> dict[str, Any]:
        return {"axis": self.axis, "provable": self.provable,
                "terms_by_code": {k: list(v)
                                  for k, v in sorted(self.terms_by_code.items())}}


def discriminating_axes(candidates: list[CandidateCode]) -> tuple[AxisProbe, ...]:
    """The axes on which the tied candidates' AUTHORITATIVE descriptors differ.

    Derived, never declared: a token every candidate shares says nothing about which of
    them the document means, so only the DIFFERENCE survives. That is what makes the
    re-inspection targeted — the directive asks for the discriminating axes only, not
    for the whole descriptor to be re-read.
    """
    if len(candidates) < 2:
        return ()
    feats = {c.code: parse_descriptor(c.descriptor) for c in candidates}
    probes: list[AxisProbe] = []

    lat = {code: tuple(sorted(f.laterality)) for code, f in feats.items()}
    if len(set(lat.values())) > 1:
        probes.append(AxisProbe(AXIS_LATERALITY, lat))

    def _interval_key(f) -> str:
        iv = f.interval
        if not (iv and iv.bounded()):
            return ""
        return f"{iv.low}:{iv.low_inc}:{iv.high}:{iv.high_inc}:{iv.unit}"

    ivs = {code: _interval_key(f) for code, f in feats.items()}
    if len(set(ivs.values())) > 1:
        probes.append(AxisProbe(
            AXIS_MEASUREMENT,
            {code: ((key,) if key else ()) for code, key in ivs.items()},
            provable=False))

    lat_words = {_sing(w) for terms in lat.values() for w in terms}
    toks = {c.code: _descriptor_tokens(c.descriptor) - lat_words for c in candidates}
    shared = set.intersection(*toks.values()) if toks else set()
    distinct = {code: tuple(sorted(t - shared)) for code, t in toks.items()}
    if any(distinct.values()):
        probes.append(AxisProbe(AXIS_DESCRIPTOR_TERM, distinct))
    return tuple(probes)


@dataclass
class TieOutcome:
    """The result of steps 3 and 4, and the material step 5 needs."""

    winner: CandidateCode | None = None
    axes: tuple[AxisProbe, ...] = ()
    #: Axes the original document did NOT settle to exactly one candidate.
    unsettled: tuple[str, ...] = ()
    #: candidate code -> the proven words of the document that support it.
    support: dict[str, tuple[str, ...]] = field(default_factory=dict)
    #: `graph_consensus.PROOF_ORIGINAL_PAGE` / `PROOF_ANCHORED_TEXT` / "".
    proof: str = ""
    detail: str = ""
    #: Step 5: ONE targeted question naming exactly the fact that would settle it.
    provider_question: str = ""
    #: True when the tie could not be inspected AT ALL because the original document
    #: does not confirm this event's quotations. That is a source-integrity stop owned
    #: by the source-evidence control (directive section 1), not a documentation gap:
    #: asking a provider to document something would answer the wrong question, exactly
    #: as `graph_consensus.resolve` reasons about the identical situation.
    source_integrity: bool = False

    def as_record(self) -> dict[str, Any]:
        return {
            "winner": (self.winner.code if self.winner else ""),
            "axes": [a.as_record() for a in self.axes],
            "unsettled_axes": list(self.unsettled),
            "support": {k: list(v) for k, v in sorted(self.support.items())},
            "proof": self.proof,
            "detail": self.detail,
            "provider_question": self.provider_question,
            "source_integrity": self.source_integrity,
        }


def _question(fact, axes: tuple[AxisProbe, ...]) -> str:
    """ONE targeted provider query naming exactly what the record must state.

    Assembled from the candidates' own authoritative descriptors, so it asks for the
    real distinguishing fact instead of "please clarify"."""
    named = []
    for probe in axes:
        options = sorted({" ".join(terms) for terms in probe.terms_by_code.values()
                          if terms})
        if options:
            named.append(probe.axis + " (" + " vs ".join(options) + ")")
    what = ("; ".join(named) if named
            else "which of the retrieved candidate services was performed")
    subject = str(getattr(fact, "description", "") or "this service")
    return (f"The record does not state the fact that distinguishes the candidate "
            f"codes for {subject!r}. Please document: {what}.")


def narrow(fact, candidates: list[CandidateCode],
           reconciliation=None) -> TieOutcome:
    """Tie policy steps 3 and 4 — re-inspect ONLY the discriminating axes against the
    original document and return the candidate that becomes UNIQUELY ENTAILED.

    "Uniquely entailed" has exactly one definition, and it lives here so no caller can
    invent a weaker one: EVERY discriminating axis is settled to a single candidate by
    the proven text, and all of them settle to the SAME candidate. An axis nobody
    documented, an axis two candidates each partly claim, and an axis words cannot
    settle at all each leave the tie open — which is step 5, not a coder queue.

    Fail-closed at every branch: no discriminating axis, no source-confirmed text,
    nothing documented, or more than one candidate documented all return no winner
    together with the reason the caller must report.
    """
    from . import graph_consensus as _gc

    unique = list({c.code: c for c in candidates if c is not None and c.code}.values())
    if len(unique) < 2:
        return TieOutcome(winner=(unique[0] if unique else None),
                          detail="no tie to narrow")
    axes = discriminating_axes(unique)
    if not axes:
        return TieOutcome(
            detail="the tied candidates' authoritative descriptors state no differing "
                   "axis, so no documented fact could distinguish them")

    supported, proof, proven_text, _spans = _gc.source_support(fact, reconciliation)
    if not supported:
        return TieOutcome(
            axes=axes, unsettled=tuple(a.axis for a in axes), source_integrity=True,
            detail="the original document does not confirm this event's quotations, so "
                   "its discriminating axes cannot be re-inspected against the page")

    words = _text_tokens(proven_text)
    support: dict[str, list[str]] = {c.code: [] for c in unique}
    settled: set[str] = set()
    for probe in axes:
        if not probe.provable:
            continue
        hits = {code: tuple(t for t in terms if t in words)
                for code, terms in probe.terms_by_code.items()}
        documented = [code for code, found in hits.items() if found]
        if len(documented) == 1:
            settled.add(probe.axis)
        for code, found in hits.items():
            support[code].extend(found)

    frozen_support = {k: tuple(dict.fromkeys(v)) for k, v in support.items()}
    documented_codes = [c for c in unique if frozen_support.get(c.code)]
    unsettled = tuple(a.axis for a in axes if a.axis not in settled)

    if len(documented_codes) == 1 and not unsettled:
        winner = documented_codes[0]
        stated = ", ".join(frozen_support[winner.code])
        axis_names = ", ".join(sorted(settled))
        return TieOutcome(
            winner=winner, axes=axes, support=frozen_support, proof=proof,
            detail=(f"every discriminating axis ({axis_names}) is settled by the "
                    f"original document, which states {stated!r} — asserted only by "
                    f"this candidate's authoritative descriptor"))

    if not documented_codes:
        detail = ("the original document states none of the axes that distinguish the "
                  "tied candidates")
    elif len(documented_codes) > 1:
        detail = (f"{len(documented_codes)} candidates are each positively documented "
                  f"on a distinguishing axis, so the document singles out none of them")
    else:
        detail = ("one candidate is documented but these discriminating axes remain "
                  "unsettled: " + ", ".join(unsettled))
    return TieOutcome(
        axes=axes, unsettled=unsettled, support=frozen_support, proof=proof,
        detail=detail, provider_question=_question(fact, axes))
