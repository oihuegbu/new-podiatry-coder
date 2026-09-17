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
from collections import defaultdict
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
#: issue #6, Codex's independent re-review (F9-R19-A): a candidate's own
#: authoritative descriptor stating "except"/"excluding"/"other than" names a
#: condition under which THIS candidate does NOT apply — the opposite polarity
#: of every other axis here. Genuinely DOCUMENTING that condition ELIMINATES
#: the candidate that carries it; it never selects anything (see
#: `requirement.RequirementRole.EXCLUSION`, `resolution._grounded_elimination`).
#: Never `selectable` here (kept out of `narrow()`'s literal-presence winner
#: logic, which assumes positive-for-this-candidate polarity) — only
#: `compile_requirements` special-cases this axis name to emit the flipped
#: `EXCLUSION` role instead of the default `MUST_SUPPORT`/`POSITIVE_ALIAS`.
AXIS_EXCLUSION_CLAUSE = "exclusion_clause"
#: issue #6, Codex's independent re-review (F9-R19-A): CPT's own family-
#: indentation convention — several candidates' descriptors share an identical
#: stem up to and including a semicolon, and each member's own qualifying
#: clause after it is what a record must state to bill THAT member rather
#: than a sibling. A real, structural, governed pattern (never an arbitrary
#: leftover-word difference — see `AXIS_DESCRIPTOR_TERM`'s own docstring for
#: why raw token differences were deliberately made non-selecting), so this
#: axis IS `selectable`: the whole qualifying clause is checked as one phrase
#: via the same `asserted_status`/`validated_requirement` machinery already
#: proven safe for `laterality`/`inclusion_term`, never bag-of-words.
AXIS_QUALIFIED_CHILD = "qualified_child"
#: issue #6, independent root-cause investigation (real-data replay against the
#: designated note, F1/28118 vs 28120): a candidate's own authoritative
#: descriptor stating a positive indication via "(eg, ...)"/"(e.g., ...)"/
#: "(for example, ...)" grammar. An earlier version of this axis compiled it
#: as a required, `selectable`/`queryable` MUST_SUPPORT precondition (the same
#: polarity as `AXIS_QUALIFIED_CHILD`'s differential) so it could win a tie or
#: ground a provider question. Codex's independent re-review (P1-2) found that
#: unsafe: by AMA/CPT convention a "(eg, ...)" clause is a non-exhaustive
#: ILLUSTRATIVE EXAMPLE, not an exhaustive checklist, and `_indication_clause`'s
#: own docstring already said so -- compiling it as a discriminating/queryable
#: axis let an undocumented example alone decide a tie or manufacture a
#: provider question from punctuation/English grammar alone, with no typed,
#: authoritative requirement field or source-governed rule behind it. This
#: axis is retained (`provable=True`) for recall/audit/explanation ONLY --
#: `selectable=False, queryable=False`, exactly like `AXIS_DESCRIPTOR_TERM`'s
#: own bag-of-words residual -- never eliminates, never wins a tie, never
#: becomes a provider question. Absent a genuine two-evaluator corroboration
#: mechanism for code selection (removed separately, issue #6 single-evaluator
#: change) or a real typed requirement field, a disagreement that turns
#: specifically on an illustrative clause like this has no governed axis to
#: adjudicate through and correctly falls to whatever generic
#: disagreement/tie handling already exists -- never a fabricated, unearned
#: specific question.
AXIS_INDICATION_CLAUSE = "indication_clause"
#: issue #6, Codex's independent re-review (F9-R19-A Finding 2): the shared
#: pre-semicolon stem itself, as a REQUIRED precondition every member of a
#: qualified-child family must clear before any of them may be considered
#: supported. Compiles to `MUST_SUPPORT` exactly like `AXIS_QUALIFIED_CHILD`
#: (same `selectable=True` treatment) -- but because every sibling in a
#: family shares the IDENTICAL term here, a validated NOT_DOCUMENTED verdict
#: eliminates the WHOLE family together, never just the one candidate whose
#: own differentiator happened to be checked. This is what stops a
#: differential question (e.g. "with material A vs with material B") from
#: ever being asked before the shared family itself is confirmed to apply
#: at all.
AXIS_FAMILY_VIABILITY = "family_viability"
#: Axes `narrow()`'s own positive-presence winner logic never evaluates at
#: all -- both are pre-elimination GATES checked exclusively through
#: `compile_requirements`/`resolution._grounded_elimination`/
#: `_requirement_grounded_status`, either because their polarity is
#: inverted (`AXIS_EXCLUSION_CLAUSE`) or because every sibling in a family
#: shares the identical term (`AXIS_FAMILY_VIABILITY`, which could never
#: settle to exactly one candidate through THIS mechanism by construction).
#: Excluded from `narrow()`'s `support`/`documented_axes`/`settled`
#: bookkeeping AND from `unsettled` -- a gate axis that can never settle
#: must never be allowed to permanently block `narrow()`'s overall winner
#: decision (`not unsettled`) for every OTHER, genuinely resolvable axis.
_GATE_ONLY_AXES = frozenset({AXIS_EXCLUSION_CLAUSE, AXIS_FAMILY_VIABILITY})

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


def _token_sequence(text: str) -> tuple[str, ...]:
    """Every token of TEXT, in ORDER -- the ordered counterpart to `_text_tokens`'s
    unordered set, needed wherever a phrase's word ADJACENCY must actually be
    checked (issue #6 F9-R6-R2 re-review), not merely which words occur somewhere
    in the text."""
    return tuple(t for t in re.split(r"[^a-z0-9]+", (text or "").lower()) if t)


def _phrase_present(term: str, tokens: tuple[str, ...]) -> bool:
    """Whether TERM occurs as a CONTIGUOUS, ORDERED run of tokens within `tokens`
    (the ordered output of `_token_sequence`, never `_text_tokens`'s unordered set).

    Every axis before issue #6 F9-R6 Phase 4 (laterality, measurement, descriptor
    words) stated its terms as single tokens, so a bare set-membership test was
    equivalent to a real phrase match. `requirement`-derived axes (currently
    ICD-10-CM `inclusion_term`) state theirs as whole clinical PHRASES ("classic
    presentation") -- a single-token set membership test would silently never
    match a real, multi-word inclusion term at all, and the FIRST bag-of-words
    fix for that (checking each word's presence anywhere, unordered) turned out
    to have its own gap (Codex F9-R6-R2 re-review): "classic" and "presentation"
    occurring anywhere in the document, in any order, arbitrarily far apart, would
    wrongly count as the document stating the phrase "classic presentation" --
    the same masking-by-approximation bug class F9-R4-R1/F9-R4-R2 already found
    and fixed for embedded action-phrase matching elsewhere in this codebase. A
    genuine sliding-window CONTIGUOUS match is the correct fix: for a single-token
    term it reduces to exact membership, unchanged from before this axis kind
    existed. Each position tolerates the text token's own spelling OR its
    singular form -- the SAME plural/singular tolerance the prior bag-of-words
    check gave via `_text_tokens`'s union, preserved exactly, checked per
    position instead of as a global set (the TERM side stays unsingularized,
    matching the prior asymmetric direction).
    """
    return bool(_phrase_positions(term, tokens))


def _phrase_positions(term: str, tokens: tuple[str, ...]) -> tuple[int, ...]:
    """Every start position where TERM occurs as a contiguous, ordered run
    within `tokens` -- the position-returning core `_phrase_present` and
    `asserted_status`'s negation-window check both build on."""
    term_tokens = tuple(t for t in re.split(r"[^a-z0-9]+", (term or "").lower()) if t)
    if not term_tokens:
        return ()
    n = len(term_tokens)
    return tuple(start for start in range(len(tokens) - n + 1)
                if all(tt in (tokens[start + j], _sing(tokens[start + j]))
                      for j, tt in enumerate(term_tokens)))


#: Clinical-documentation negation cues (issue #6 F9-R6-R2/R6-R6, Codex
#: re-review) -- English documentation grammar, not medical vocabulary, the
#: SAME exemption `_GRAMMAR` above already relies on. A phrase occurring only
#: after one of these within its own clause is NEGATED, not stated -- "no
#: classic presentation" must never count as the document stating "classic
#: presentation". Deliberately NOT sourced from `provenance.py`'s
#: `relation_evidence_grammar.json`: that manifest-bound config is real and
#: authoritative but scoped to a narrower purpose (CMS-cited relation-linkage
#: grammar between two event mentions) and is fail-closed by design -- making
#: `narrow()`, a foundational function with zero external file dependencies
#: today, hard-depend on that file for every single axis match would be a far
#: bigger, more fragile change than fixing negation-blindness requires.
_NEGATION = (
    "no", "not", "never", "without", "denies", "denied", "denying",
    "negative for", "absent", "absence of", "ruled out", "rule out",
    "r/o", "unremarkable for",
)
#: Clause boundaries -- a negation cue in an EARLIER sentence must never
#: negate a phrase in a LATER, unrelated one.
_CLAUSE_TERMINATORS = frozenset({".", ";", ":", "!", "?", "\n", "\r", "|", "•"})


def _split_clauses(text: str) -> list[str]:
    pattern = "[" + re.escape("".join(_CLAUSE_TERMINATORS)) + "]"
    return [c for c in re.split(pattern, text or "") if c.strip()]


#: How far back (in tokens, within the SAME clause) a negation cue can still
#: reach a phrase -- "denies any history of X" (4 tokens back) still negates
#: X, but an unrelated EARLIER clause-internal aside must not. A short,
#: bounded window (not "anywhere in the clause") is deliberate: "not on the
#: right, but the left" must negate "right" without also negating "left",
#: even though both share one clause and "not" precedes both positionally --
#: the corrective "but the left" sits well outside a short negation window
#: from "not", the same scope convention standard clinical-negation detectors
#: (e.g. NegEx) use.
_NEGATION_WINDOW = 5


def _term_length(term: str) -> int:
    return len([t for t in re.split(r"[^a-z0-9]+", (term or "").lower()) if t])


def _negated_nearby(tokens: tuple[str, ...], start: int, length: int) -> bool:
    """Whether any negation cue occurs within `_NEGATION_WINDOW` tokens
    immediately BEFORE or AFTER a phrase match, bounded to the SAME clause.
    Covers both pre-positive clinical negation ("no X", "denies X") and
    post-positive ("X ruled out", "X denied") -- checking only one direction
    would miss the other's ordinary phrasing entirely."""
    before = tokens[max(0, start - _NEGATION_WINDOW):start]
    after = tokens[start + length:start + length + _NEGATION_WINDOW]
    return (any(_phrase_positions(marker, before) for marker in _NEGATION) or
           any(_phrase_positions(marker, after) for marker in _NEGATION))


def asserted_status(term_options: tuple[str, ...], text: str) -> str:
    """"supported" | "negated" | "absent" -- the TRUE, clause-scoped,
    negation-aware status of ANY of `term_options` within `text`. Never trusts
    a bare contiguous match on its own: a phrase occurring only alongside a
    negation cue (before OR after it) within its own clause is NEGATED, not
    asserted. Any ONE unnegated occurrence anywhere is enough to call it
    "supported" -- a negated mention elsewhere in the same text does not
    retract a genuine, separate assertion. Only when EVERY occurrence found is
    negated (or none occur at all) does the negative/absent distinction matter
    to the caller.
    """
    found_negated = False
    for clause in _split_clauses(text):
        tokens = _token_sequence(clause)
        for term in term_options:
            length = _term_length(term)
            for pos in _phrase_positions(term, tokens):
                if _negated_nearby(tokens, pos, length):
                    found_negated = True
                else:
                    return "supported"
    return "negated" if found_negated else "absent"


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
    provable: bool
    #: Whether this axis may eliminate alternatives or produce a winner. Raw
    #: descriptor-token differences are auditable recall evidence, not typed
    #: clinical facts, so they are deliberately non-selecting.
    selectable: bool
    #: Whether an undocumented value on this axis can become a provider question.
    #: An open-ended descriptor-token bag is never provider-answerable.
    queryable: bool

    def as_record(self) -> dict[str, Any]:
        return {"axis": self.axis, "provable": self.provable,
                "selectable": self.selectable, "queryable": self.queryable,
                "terms_by_code": {k: list(v)
                                  for k, v in sorted(self.terms_by_code.items())}}


#: issue #6, Codex's independent re-review (F9-R19-A): the three English
#: markers a CPT/HCPCS descriptor uses to state a condition under which the
#: code does NOT apply. Documentation grammar, not clinical vocabulary — the
#: same kind of exemption `_GRAMMAR`/`_NEGATION` above already rely on.
_EXCLUSION_MARKERS = ("except", "excluding", "other than")
_EXCLUSION_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(m) for m in _EXCLUSION_MARKERS) + r")\b\s*(.+)",
    re.IGNORECASE)

# Descriptor coordination grammar, never clinical vocabulary.  A coordinator
# joins alternatives only when it is not part of a comparative threshold or
# the conventional optional phrase "with or without".  The alternatives are
# still copied from the authoritative descriptor; this code merely preserves
# their boolean shape instead of treating an entire compound clause as one
# literal phrase.
_ALTERNATIVE_RE = re.compile(r"\s+(?:or|and/or)\s+", re.IGNORECASE)
_OPTIONAL_COORDINATION_RE = re.compile(r"\bwith\s+or\s+without\b", re.IGNORECASE)
_COMPARATIVE_VALUE_RE = re.compile(
    r"^(?:greater|less|more|fewer|equal|above|below)\b", re.IGNORECASE)


def _descriptor_alternatives(text: str) -> tuple[str, ...]:
    """Return descriptor-stated OR alternatives without inventing synonyms.

    The returned strings are verbatim fragments of ``text``.  This is a
    boolean/grammar parse only: it contains no code, specialty, anatomy,
    procedure, product, or diagnosis vocabulary.  Threshold grammar and
    ``with or without`` remain indivisible because their ``or`` does not join
    two alternative clinical values.
    """
    phrase = (text or "").strip(" ,;.")
    if not phrase:
        return ()
    cuts: list[tuple[int, int]] = []
    for match in _ALTERNATIVE_RE.finditer(phrase):
        before = phrase[:match.start()]
        after = phrase[match.end():]
        # Protect only THIS coordinator. Other genuine alternatives in the
        # same clause must still be split.
        optional = (bool(re.search(r"\bwith\s*$", before, re.IGNORECASE))
                    and bool(re.match(r"^without\b", after, re.IGNORECASE)))
        comparative = bool(_COMPARATIVE_VALUE_RE.match(after))
        if not optional and not comparative:
            cuts.append((match.start(), match.end()))
    if not cuts:
        return (phrase,)
    parts: list[str] = []
    start = 0
    for left, right in cuts:
        part = phrase[start:left].strip(" ,;.")
        if part:
            parts.append(part)
        start = right
    final = phrase[start:].strip(" ,;.")
    if final:
        parts.append(final)
    return tuple(parts) or (phrase,)


def _positive_qualifier_clause(text: str) -> str:
    """The positive portion of a family qualifier, excluding its exception.

    Exclusion semantics are compiled independently as
    :data:`AXIS_EXCLUSION_CLAUSE`.  Leaving the exception attached to the
    positive child axis duplicates opposite-polarity facts in one selectable
    phrase and can manufacture a provider question from a candidate's own
    exclusion.  The split is driven only by the descriptor's grammar marker.
    """
    match = _EXCLUSION_RE.search(text or "")
    if not match:
        return (text or "").strip(" ,;.")
    return (text or "")[:match.start()].strip(" ,;.")


def _exclusion_clause(descriptor: str) -> str | None:
    """The clause following an "except"/"excluding"/"other than" marker within
    ONE clause of `descriptor` (never spanning a clause boundary — a marker in
    an earlier clause must not swallow unrelated later text), or None when the
    descriptor states no such condition. Reproduces verbatim from the
    descriptor by construction (a regex match on the real string), so a
    caller compiling this into a `requirement.DescriptorRequirement` can
    always find it again via `_find_clause`."""
    for clause in _split_clauses(descriptor or ""):
        m = _EXCLUSION_RE.search(clause)
        if m:
            text = m.group(1).strip(" ,;.")
            if text:
                return text
    return None


#: The markers a CPT/HCPCS/ICD-10-CM descriptor uses to introduce a positive,
#: illustrative INDICATION clause -- documentation grammar, not clinical
#: vocabulary, exactly like `_EXCLUSION_MARKERS` above.
_INDICATION_RE = re.compile(
    r"\((?:eg|e\.g\.?|for example)[,.]?\s*([^()]+)\)", re.IGNORECASE)


def _indication_clause(descriptor: str) -> str | None:
    """The clause inside a "(eg, ...)"/"(e.g., ...)"/"(for example, ...)"
    parenthetical, or None when `descriptor` states no such clause.

    Matched directly against the WHOLE descriptor (never per-clause like
    `_exclusion_clause`) because the marker itself is parenthesis-scoped, not
    comma/semicolon-scoped -- a real CPT descriptor commonly nests an
    "(eg, ...)" example clause inside a larger comma-separated phrase (e.g.
    "Partial excision (...) structure (eg, variant condition alpha or
    variant condition beta); structure gamma or structure alpha"), so
    splitting on commas first would sever the marker from its own content,
    exactly the way it would for `_exclusion_clause` if that
    marker were ever nested in parentheses too. Reproduces verbatim from the
    descriptor by construction (a regex match on the real string), so a
    caller compiling this into a `requirement.DescriptorRequirement` can
    always find it again via `_find_clause`, the same guarantee
    `_exclusion_clause` already gives."""
    m = _INDICATION_RE.search(descriptor or "")
    if not m:
        return None
    text = m.group(1).strip(" ,;.")
    return text or None


def _semicolon_prefix(descriptor: str) -> tuple[str, str] | None:
    """(normalized stem including the semicolon, the candidate's own remaining
    qualifying clause) for a descriptor stating CPT's family-indentation
    convention (a shared stem, then a semicolon, then the member-specific
    differentiator) — or None when `descriptor` carries no semicolon, or
    nothing follows it."""
    idx = (descriptor or "").find(";")
    if idx < 0:
        return None
    prefix = descriptor[:idx + 1].strip().lower()
    remainder = descriptor[idx + 1:].strip(" ,;.")
    if not prefix or not remainder:
        return None
    return prefix, remainder


def _qualified_child_groups(candidates: list[CandidateCode]
                            ) -> dict[str, list[CandidateCode]]:
    """{normalized pre-semicolon stem -> members}, restricted to stems at
    least two of the tied candidates genuinely share — CPT's own family
    convention, never an arbitrary prefix match. A candidate with no
    semicolon, or whose stem no sibling in THIS shortlist shares, is not
    part of a documented family at all and appears in no group."""
    by_prefix: dict[str, list[CandidateCode]] = defaultdict(list)
    for c in candidates:
        got = _semicolon_prefix(c.descriptor)
        if got is None:
            continue
        by_prefix[got[0]].append(c)
    return {prefix: members for prefix, members in by_prefix.items() if len(members) >= 2}


def _qualified_child_terms(candidates: list[CandidateCode]) -> dict[str, tuple[str, ...]]:
    """Each family member's own post-semicolon qualifying clause -- the
    DIFFERENTIAL that distinguishes it from its siblings once the shared
    stem itself is established (see `_family_viability_terms` for that
    separate, prerequisite check -- issue #6, Codex's independent re-review,
    F9-R19-A Finding 2: a differential clause may narrow WHICH sibling, but
    it must never be evaluated as if it alone proves the shared family
    applies at all). A candidate with no entry here is not part of a
    documented family needing this differentiation, so nothing is required
    of it on this axis (silence is never proof, same as every other axis
    here)."""
    out: dict[str, tuple[str, ...]] = {}
    for prefix, members in _qualified_child_groups(candidates).items():
        for c in members:
            parsed = _semicolon_prefix(c.descriptor)
            positive = _positive_qualifier_clause(parsed[1])
            out[c.code] = _descriptor_alternatives(positive)
    return out


def _family_viability_terms(candidates: list[CandidateCode]) -> dict[str, tuple[str, ...]]:
    """issue #6, Codex's independent re-review (F9-R19-A Finding 2): the
    shared pre-semicolon stem itself, as a REQUIRED precondition every
    member of a qualified-child family must clear before any of them may be
    considered supported -- never omitted just because it is IDENTICAL
    across siblings (a shared term proves nothing about which sibling, but
    it is not therefore optional: the family as a whole must still be
    genuinely documented). Every member of a >=2 family gets the SAME term
    (its own family's shared stem) -- a candidate with no family membership
    (see `_qualified_child_groups`) gets no entry, exactly like the
    differential axis."""
    out: dict[str, tuple[str, ...]] = {}
    for prefix, members in _qualified_child_groups(candidates).items():
        for c in members:
            out[c.code] = (prefix.rstrip(";").strip(),)
    return out


def discriminating_axes(candidates: list[CandidateCode]) -> tuple[AxisProbe, ...]:
    """The axes on which the tied candidates' AUTHORITATIVE descriptors differ.

    Derived, never declared: a token every candidate shares says nothing about which of
    them the document means, so only the DIFFERENCE survives. That is what makes the
    re-inspection targeted — the directive asks for the discriminating axes only, not
    for the whole descriptor to be re-read.

    issue #6 F9-R2-B, fourth pass: an APPROACH axis compiled from
    `semantics._APPROACH_WORDS` was added and then REVERTED. Codex's re-review
    found it exactly right: that word list has no versioned artifact identity, no
    semantic-role parse, and no authoritative qualifier field behind it -- it is a
    fixed token scanner wearing a governed field's name. Two real reproductions:
    ordinary category wording using "open"/"closed" produced a labelled `approach`
    provider question, and one candidate's silence on those words was treated as
    the OPPOSING approach and used to deterministically select its sibling --
    absence is not evidence, and it was being treated as some. Those six words now
    fall back into the untyped `AXIS_DESCRIPTOR_TERM` bucket like any other
    leftover token, which `provider_query` already never names. Only `laterality`
    (a real closed enumeration) and `measurement` (a real typed, unit-converted
    interval) are genuinely governed enough to select a code or reach a provider.
    """
    if len(candidates) < 2:
        return ()
    feats = {c.code: parse_descriptor(c.descriptor) for c in candidates}
    probes: list[AxisProbe] = []

    lat = {code: tuple(sorted(f.laterality)) for code, f in feats.items()}
    if len(set(lat.values())) > 1:
        probes.append(AxisProbe(
            AXIS_LATERALITY, lat,
            provable=True, selectable=True, queryable=True))

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
            provable=False, selectable=False, queryable=True))

    # issue #6, Codex's independent re-review (F9-R19-A): candidate-differential
    # selection from the candidates' own authoritative descriptor STRUCTURE
    # (never an arbitrary word list) -- an exclusion clause each candidate
    # states independently, and a family-qualifying clause candidates sharing
    # a semicolon stem each state relative to their siblings. Computed BEFORE
    # `AXIS_DESCRIPTOR_TERM` below so those axes' own governed words can be
    # subtracted from its raw bag-of-words the same way laterality's already
    # are -- otherwise the identical words that let a SELECTABLE axis settle
    # the tie would ALSO surface as a non-selectable, never-settling
    # `descriptor_term` axis, which used to permanently block `narrow()`'s
    # winner decision even after the real, governed axis had resolved it.
    excl = {c.code: _descriptor_alternatives(_exclusion_clause(c.descriptor) or "")
            for c in candidates}
    indic = {c.code: _descriptor_alternatives(_indication_clause(c.descriptor) or "")
             for c in candidates}
    qualified = _qualified_child_terms(candidates)
    viability = _family_viability_terms(candidates) if qualified else {}

    lat_words = {_sing(w) for terms in lat.values() for w in terms}
    # The marker itself belongs to the exclusion/indication grammar just as
    # surely as the clause it introduces.  If left in the residual token bag
    # it creates a permanently-unsettled audit-only axis even after the
    # structured positive/exclusion/indication clauses have resolved the
    # candidate family.
    governed_words = (set(lat_words) | {_sing(word) for word in _EXCLUSION_MARKERS}
                      | {"eg", "example"})
    for terms in (*excl.values(), *qualified.values(), *viability.values(), *indic.values()):
        governed_words |= {_sing(w) for phrase in terms
                           for w in re.split(r"[^a-z0-9]+", phrase.lower()) if w}
    toks = {c.code: _descriptor_tokens(c.descriptor) - governed_words for c in candidates}
    shared = set.intersection(*toks.values()) if toks else set()
    distinct = {code: tuple(sorted(t - shared)) for code, t in toks.items()}
    if any(distinct.values()):
        probes.append(AxisProbe(
            AXIS_DESCRIPTOR_TERM, distinct,
            provable=True, selectable=False, queryable=False))

    if any(excl.values()):
        # issue #6, Codex's independent re-review (F9-R23 root finding 3):
        # `queryable=True` let a gate-only exclusion clause leak into a
        # provider question asking the provider to "document" a fact the
        # note ALREADY states (that is exactly what makes it eligible to
        # eliminate a sibling candidate in the first place) -- reproduced
        # live on the designated note. `AXIS_EXCLUSION_CLAUSE` is a member
        # of `_GATE_ONLY_AXES`: it can eliminate an inapplicable candidate,
        # it can never become a provider-answerable open question.
        probes.append(AxisProbe(
            AXIS_EXCLUSION_CLAUSE, excl,
            provable=True, selectable=False, queryable=False))

    if any(indic.values()):
        # issue #6, Codex's independent re-review (P1-2 correction): a
        # "(eg, ...)" clause is, BY AMA/CPT CONVENTION, a non-exhaustive
        # ILLUSTRATIVE EXAMPLE -- `_indication_clause`'s own docstring already
        # called it that -- never a checklist item a coder or provider could
        # be asked to confirm/deny. Compiling it as `selectable=True` let an
        # undocumented example alone win a tie or ground a provider question
        # from punctuation/English grammar alone, with no typed, authoritative
        # requirement field or source-governed rule behind it. Retained here
        # for recall/audit/explanation only -- `provable=True` so it still
        # surfaces in `documented_axes`/audit bookkeeping -- but
        # `selectable=False, queryable=False` exactly like `AXIS_DESCRIPTOR_TERM`
        # above: it can never decide `narrow()`'s winner and never leaks into
        # a provider question. A real, typed requirement field (were one ever
        # added to this repo's authoritative sources) is the only thing that
        # should flip this back.
        probes.append(AxisProbe(
            AXIS_INDICATION_CLAUSE, indic,
            provable=True, selectable=False, queryable=False))

    if qualified:
        full = {c.code: qualified.get(c.code, ()) for c in candidates}
        # A child qualifier can distinguish the tied candidates only when
        # EVERY candidate states a child value and at least two values differ.
        # Silence is not the opposite of a qualifier.  The old probe included
        # unrelated/silent candidates and could turn one candidate's residual
        # suffix into a provider question even though the other candidate had
        # no competing value on this axis.
        if all(full.values()) and len(set(full.values())) > 1:
            probes.append(AxisProbe(
                AXIS_QUALIFIED_CHILD, full,
                provable=True, selectable=True, queryable=True))
        # issue #6, Codex's independent re-review (F9-R21-B): the shared
        # stem is recorded as AUDIT CONTEXT only -- `provable=False,
        # selectable=False, queryable=False` -- never a literal-absence
        # MUST_SUPPORT proxy. An exact-text prefix match reproduces the
        # exact unsafe shape `AXIS_DESCRIPTOR_TERM`'s own history already
        # rejected (a synonymous/paraphrased note can support the clinical
        # family while never repeating the descriptor's exact wording), and
        # `queryable=True` let it leak into a provider question asking the
        # provider to "document" a parent service the note already states.
        # `compile_requirements` filters on `provable`, so this axis never
        # produces a requirement, never grounds an elimination, and (with
        # `_GATE_ONLY_AXES`) never reaches a provider question. Family
        # viability is instead established by the now-corrected disposition
        # layer itself (issue #6, Codex's independent re-review, F9-R21-A):
        # an "entailed" verdict only counts as SUPPORTED once BOTH
        # evaluators independently cite validated, reconciled evidence for
        # the candidate's OWN COMPLETE descriptor -- stem and differential
        # together -- so a genuinely inapplicable family can no longer
        # slip through merely because no lexical proxy caught it, and a
        # genuinely applicable, paraphrased family is no longer falsely
        # eliminated by one.
        probes.append(AxisProbe(
            AXIS_FAMILY_VIABILITY, {c.code: viability.get(c.code, ()) for c in candidates},
            provable=False, selectable=False, queryable=False))
    return tuple(probes)


def _axes_from_requirements(requirements: tuple, codes: set[str],
                            existing_axes: frozenset) -> tuple[AxisProbe, ...]:
    """New discriminating axes a compiled REQUIREMENT set states that `discriminating_axes`
    itself never derives from the candidates alone -- currently the ICD-10-CM
    `inclusion_term` axis (issue #6 F9-R6 Phase 4), compiled from the Tabular's own
    governed `inclusionTerm` field via `requirement.compile_requirements`, never a
    hardcoded term list. Duck-typed against `requirement.DescriptorRequirement` (`.axis`,
    `.candidate_code`, `.expected`, `.selectable`, `.queryable`) instead of importing
    that module, which already imports THIS one to build its requirements FROM these
    same discriminating axes -- importing back here would be circular.

    Grouped exactly the way `discriminating_axes` groups its own probes: an axis counts
    as discriminating only when the tied candidates' terms actually differ, never when
    every candidate states the same thing (which proves nothing about which one the
    document means) or when only one candidate has any term on it at all (silence is
    never proof, the same rule `AxisProbe` itself documents).
    """
    if not requirements:
        return ()
    by_axis: dict[str, dict[str, set[str]]] = {}
    flags: dict[str, list[bool]] = {}
    for req in requirements:
        if req.axis in existing_axes or req.candidate_code not in codes:
            continue
        by_axis.setdefault(req.axis, {}).setdefault(
            req.candidate_code, set()).update(req.expected)
        sel, qry = flags.get(req.axis, (False, False))
        flags[req.axis] = (sel or req.selectable, qry or req.queryable)
    probes: list[AxisProbe] = []
    for axis, terms_by_code in sorted(by_axis.items()):
        full = {code: tuple(sorted(terms_by_code.get(code, ()))) for code in codes}
        if not all(full.values()) or len(set(full.values())) < 2:
            # A missing value is absence, not an opposing value.  Selection or
            # a provider question is valid only when the authority states a
            # genuinely differing value for every candidate still in the tie.
            continue
        selectable, queryable = flags[axis]
        probes.append(AxisProbe(axis, full, provable=True,
                                selectable=selectable, queryable=queryable))
    return tuple(probes)


@dataclass
class TieOutcome:
    """The result of steps 3 and 4, and the material step 5 needs."""

    winner: CandidateCode | None = None
    axes: tuple[AxisProbe, ...] = ()
    #: Axes the original document did NOT settle to exactly one candidate.
    unsettled: tuple[str, ...] = ()
    #: Axes where the proven text confirms AT LEAST ONE candidate's term (issue #6
    #: F9-R2-B) -- distinct from `unsettled`: two candidates can each state a word
    #: the page also states, which never settles the axis but the VALUE is
    #: genuinely documented. Never named in a provider question; see `provider_query`.
    documented: tuple[str, ...] = ()
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
            "documented_axes": list(self.documented),
            "support": {k: list(v) for k, v in sorted(self.support.items())},
            "proof": self.proof,
            "detail": self.detail,
            "provider_question": self.provider_question,
            "source_integrity": self.source_integrity,
        }


def provider_query(fact, axes: tuple[AxisProbe, ...]) -> str:
    """ONE targeted provider query naming exactly what the record must state, or ""
    when nothing left to ask about is a governed, typed distinguishing fact.

    Public because step 5 is reachable from more than one place: a tie the page could
    not settle (below), and a tie the page settled onto a candidate the entailment
    judgements did not both accept (`resolution._tie_escalation`). Both owe the
    provider the SAME question, built once here from the same descriptors.

    Assembled from the candidates' own authoritative descriptors, so it asks for the
    real distinguishing fact instead of "please clarify". AXIS_DESCRIPTOR_TERM is
    NEVER named here, regardless of shape (issue #6 F9-R2-B, third pass): an earlier
    version tried to salvage it when it collapsed to one clean word per candidate,
    but token CARDINALITY does not establish clinical meaning, code-changing status,
    or a provider-answerable field -- an arbitrary single leftover word (two
    synonyms retrieval happened to phrase differently, say) is exactly as
    unpromotable as a whole bag of them. Only a GOVERNED, versioned typed axis
    compiled from authoritative descriptor data -- currently laterality and
    measurement -- may ever reach a provider. When there is nothing left to name,
    this returns "" instead of the old generic "please clarify which service was
    performed" filler -- both callers already treat an empty question as "no
    documentation gap", which correctly routes the line to the coder queue
    (`autonomy.decide`) instead of manufacturing a provider question out of
    retrieval's own leftover vocabulary."""
    named = []
    for probe in axes:
        # issue #6, Codex's independent re-review (F9-R23 root finding 3):
        # a second, independent defense against a gate-only axis reaching a
        # provider question -- never trust `queryable` alone to have been
        # set correctly at every construction site; a gate-only axis is
        # refused here even if it were.
        if probe.axis in _GATE_ONLY_AXES or not probe.queryable:
            continue
        options = sorted({" ".join(terms) for terms in probe.terms_by_code.values()
                          if terms})
        if options:
            named.append(probe.axis + " (" + " vs ".join(options) + ")")
    if not named:
        return ""
    subject = str(getattr(fact, "description", "") or "this service")
    return (f"The record does not state the fact that distinguishes the candidate "
            f"codes for {subject!r}. Please document: {'; '.join(named)}.")


def _typed_laterality_support(fact, probe: AxisProbe, reconciliation
                              ) -> dict[str, tuple[str, ...]] | None:
    """Laterality settled via the FACT's own typed `attributes["laterality"]`
    value -- never re-derived by lexical matching against raw text, which cannot
    correctly resolve arbitrary-distance negation scope (a fixed token window
    around a phrase match cannot tell "left but not right" from "right but not
    left" once the negation cue falls outside the window).

    issue #6 F9-R6-R2, fifth re-review: round 4's "now safe to trust" claim was
    wrong -- `graph_consensus.resolve()`'s matching fix ALSO used a token-window
    heuristic (`tiebreak.asserted_status`) to decide what gets written into
    `attributes["laterality"]`, so it had the identical blind spot one layer
    earlier, AND a fact whose two readings simply AGREE on a wrong value never
    even reaches `resolve()` (no disagreement to resolve), leaving this function
    the last, and only, line of defense either way. It now independently
    re-verifies the typed value against `fact`'s own extraction-time
    `attribute_evidence` (`graph_consensus.asserted_attribute_support` --
    genuinely ASSERTED, scope-validated, source-reconciled evidence, judged by the
    model with the complete sentence in view, never re-derived from raw text)
    EVERY time, regardless of whether a cross-reading disagreement ever ran
    through `resolve()` at all.

    Returns None when this probe is not the laterality axis, the fact carries no
    laterality value at all, OR that value is not genuinely, provably asserted;
    the caller must NOT fall back to lexical matching for this axis when None --
    fail closed, never guess."""
    if probe.axis != AXIS_LATERALITY:
        return None
    value = str((getattr(fact, "attributes", None) or {}).get("laterality") or "").strip().lower()
    if not value:
        return None
    from . import graph_consensus as _gc
    if not _gc.asserted_attribute_support(fact, "laterality", value, reconciliation):
        return None
    return {code: tuple(t for t in terms if t == value)
           for code, terms in probe.terms_by_code.items()}


def narrow(fact, candidates: list[CandidateCode],
           reconciliation=None, requirements: tuple = ()) -> TieOutcome:
    """Tie policy steps 3 and 4 — re-inspect ONLY the discriminating axes against the
    original document and return the candidate that becomes UNIQUELY ENTAILED.

    "Uniquely entailed" has exactly one definition, and it lives here so no caller can
    invent a weaker one: EVERY discriminating axis is settled to a single candidate by
    the proven text, and all of them settle to the SAME candidate. An axis nobody
    documented, an axis two candidates each partly claim, and an axis words cannot
    settle at all each leave the tie open — which is step 5, not a coder queue.

    `requirements` (issue #6 F9-R6 Phase 4, optional): a caller that already compiled
    `requirement.compile_requirements(candidates, source)` for elimination purposes may
    pass it here too, so the SAME governed axes (currently: ICD-10-CM `inclusion_term`)
    that can eliminate a candidate can also NARROW a genuine multi-candidate tie —
    matching laterality's existing dual role instead of leaving it the only axis that
    can both select and narrow. Merged with `discriminating_axes`' own axes via
    `_axes_from_requirements`, which only ever adds an axis `discriminating_axes` does
    not already derive on its own — never a duplicate, never a hardcoded name. Omitted
    (the default), this behaves exactly as it always has.

    Fail-closed at every branch: no discriminating axis, no source-confirmed text,
    nothing documented, or more than one candidate documented all return no winner
    together with the reason the caller must report.

    issue #6 F9-R6-R2/R6-R6 re-review: axis matching is negation-aware
    (`asserted_status`) -- "not on the right, but the left" settles to LEFT,
    never treats the negated mention of "right" as the document stating it.

    issue #6 F9-R6-R2, fourth re-review: laterality specifically no longer
    uses lexical matching at all -- it is settled from the fact's own typed
    `attributes["laterality"]` value (see `_typed_laterality_support`), fail-
    closed when absent, since a fixed-token-window lexical heuristic cannot
    correctly resolve arbitrary-distance grammatical negation scope.
    """
    from . import graph_consensus as _gc

    unique = list({c.code: c for c in candidates if c is not None and c.code}.values())
    if len(unique) < 2:
        return TieOutcome(winner=(unique[0] if unique else None),
                          detail="no tie to narrow")
    axes = discriminating_axes(unique)
    if requirements:
        axes = axes + _axes_from_requirements(
            requirements, {c.code for c in unique},
            frozenset(a.axis for a in axes))
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

    support: dict[str, list[str]] = {c.code: [] for c in unique}
    settled: set[str] = set()
    #: Axes where the proven text confirms AT LEAST ONE candidate's term -- issue #6
    #: F9-R2-B: settling to exactly ONE candidate (`settled`, above) and the fact
    #: itself being DOCUMENTED are different questions. Two candidates can each state
    #: a word the page also states (e.g. both "right"), which never settles the axis
    #: (len(documented) != 1) but the VALUE is genuinely in the record -- asking the
    #: provider to "document" a fact already there would be asking for something that
    #: exists. That is a candidate-mapping ambiguity for a coder, not a documentation
    #: gap for a provider.
    documented_axes: set[str] = set()
    for probe in axes:
        if not probe.provable:
            continue
        if probe.axis in _GATE_ONLY_AXES:
            # issue #6, Codex's independent re-review (F9-R19-A): both
            # `AXIS_EXCLUSION_CLAUSE` (opposite polarity -- documenting it
            # argues AGAINST the candidate that carries it) and
            # `AXIS_FAMILY_VIABILITY` (every sibling in a family shares the
            # IDENTICAL term, so a hit here would add EVERY sibling to
            # `documented_codes` at once) are pre-elimination GATES, never
            # part of this positive-selection bookkeeping -- both are
            # checked exclusively through `compile_requirements`/
            # `resolution._grounded_elimination`/`_requirement_grounded_
            # status`, never here. Skipped entirely (never added to
            # `support`/`documented_axes`/`settled`) -- see `unsettled`
            # below for why they must ALSO be kept out of that check.
            continue
        # issue #6 F9-R6-R2, fourth re-review: laterality is settled EXCLUSIVELY
        # from the fact's own typed attribute now, never re-derived lexically --
        # a fixed-token-window heuristic cannot correctly resolve arbitrary-
        # distance negation scope (proven: "right ... ultimately ruled out" wins
        # RIGHT under a window-based check). A fact with no typed laterality
        # value fails closed (no hits) instead of falling back to the unsafe
        # lexical path. Every other axis kind is unaffected and keeps the
        # existing negation-aware lexical match ("not on the right, but the
        # left" must never count "right" as the document stating it).
        typed = _typed_laterality_support(fact, probe, reconciliation)
        if typed is not None:
            hits = typed
        elif probe.axis == AXIS_LATERALITY:
            hits = {code: () for code in probe.terms_by_code}
        else:
            hits = {code: tuple(t for t in terms if asserted_status((t,), proven_text) == "supported")
                    for code, terms in probe.terms_by_code.items()}
        documented = [code for code, found in hits.items() if found]
        if documented:
            documented_axes.add(probe.axis)
        if probe.selectable and len(documented) == 1:
            settled.add(probe.axis)
        for code, found in hits.items():
            support[code].extend(found)

    frozen_support = {k: tuple(dict.fromkeys(v)) for k, v in support.items()}
    documented_codes = [c for c in unique if frozen_support.get(c.code)]
    unsettled = tuple(a.axis for a in axes
                      if a.axis not in settled and a.axis not in _GATE_ONLY_AXES)

    if len(documented_codes) == 1 and not unsettled:
        winner = documented_codes[0]
        stated = ", ".join(frozen_support[winner.code])
        axis_names = ", ".join(sorted(settled))
        return TieOutcome(
            winner=winner, axes=axes, support=frozen_support, proof=proof,
            documented=tuple(sorted(documented_axes)),
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
        # issue #6, Codex's independent re-review (F9-R19-A Finding 4): a
        # non-selectable axis (`descriptor_term`, ICD's `inclusion_term`, ...)
        # can NEVER settle through this mechanism at all -- it is an internal
        # schema/audit identifier, not a fact a coder or provider could ever
        # act on by "documenting" it, and reporting it as "unsettled" implies
        # a resolvable gap that does not exist. Only genuinely SELECTABLE
        # axes (the ones that could actually decide this tie) are named here;
        # `unsettled` itself (the typed field/audit record) is unchanged, so
        # a caller that needs the complete picture still has it.
        _reportable_unsettled = tuple(a.axis for a in axes
                                      if a.selectable and a.axis not in settled)
        if _reportable_unsettled:
            detail = ("one candidate is documented but these discriminating axes "
                      "remain unsettled: " + ", ".join(_reportable_unsettled))
        else:
            detail = ("one candidate is documented, but no governed, provider-"
                      "answerable axis distinguishes the tied candidates")
    # Never build a provider question that names an axis the record already
    # documents (issue #6 F9-R2-B) -- only genuinely UNDOCUMENTED axes are askable.
    askable = tuple(a for a in axes if a.axis not in documented_axes)
    return TieOutcome(
        axes=axes, unsettled=unsettled, support=frozen_support, proof=proof,
        documented=tuple(sorted(documented_axes)),
        detail=detail, provider_question=provider_query(fact, askable))
