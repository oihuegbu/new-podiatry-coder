"""Deterministic ICD-10-CM resolution via the authoritative Alphabetic Index.

This is the permanent fix for the terse-descriptor / eponym recall gap. The
ICD-10-CM Alphabetic Index (published by NCHS/CMS, parsed into
data/codes/icd10cm_index_terms.json by tools/parse_icd10cm_index.py) is the
AUTHORITATIVE map from clinician vocabulary — eponyms, lay terms, abbreviations,
synonyms — to codes. It is exactly the knowledge an
embedding only approximates. We look a diagnosis up here FIRST, deterministically
and with provenance; the embedding index is only the fallback for phrasings the
Index does not carry.

No code is authored here — the term→code map is inverted from the authoritative
data at load time, and self-updates when the annual Index is re-ingested.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ",
                  str(s).lower().replace("'", ""))).strip()


def _sing(tok: str) -> str:
    """Light singularization so plural note vocabulary matches Index terms
    ('words'->'word', 'boxes'->'box'). Conservative: only trims a
    trailing 's' on longer words."""
    if len(tok) > 4 and tok.endswith("es") and tok[-3] in "sxzo":
        return tok[:-2]
    if len(tok) > 4 and tok.endswith("s") and not tok.endswith("ss"):
        return tok[:-1]
    return tok


def _dot(code: str) -> str:
    """ICD-10-CM display form: 3-char category + '.' + remainder (undotted->dotted)."""
    c = str(code).upper().replace(".", "")
    return c if len(c) <= 3 else f"{c[:3]}.{c[3:]}"


class TerminologyIndex:
    """Inverts the authoritative {code: [index terms]} into term→codes lookups:
    an exact normalized-term map, and an order-independent token-set map that
    handles the Index's inverted phrasing ('Entity, qualifying-term') against a
    note's natural phrasing ('qualifying-term entity')."""

    def __init__(self, terms_by_code: dict[str, list[str]]):
        self._exact: dict[str, set[str]] = {}
        self._despaced: dict[str, set[str]] = {}   # 'two words' <-> 'twowords'
        self._byset: dict[frozenset[str], set[str]] = {}   # order + plural independent
        for code, terms in terms_by_code.items():
            dotted = _dot(code)
            for term in terms or []:
                n = _norm(term)
                if not n:
                    continue
                self._exact.setdefault(n, set()).add(dotted)
                self._despaced.setdefault(n.replace(" ", ""), set()).add(dotted)
                toks = frozenset(_sing(t) for t in n.split() if len(t) > 2)
                if toks:
                    self._byset.setdefault(toks, set()).add(dotted)

    def candidates(self, description: str) -> set[str]:
        """Authoritative ICD-10-CM codes for a clinician term (dotted). Matches in
        order: exact normalized, compound-word (despaced), then order/plural-
        independent token set. Empty if the Index does not carry the phrasing
        (→ caller falls back to retrieval)."""
        n = _norm(description)
        if not n:
            return set()
        if n in self._exact:
            return set(self._exact[n])
        despaced = n.replace(" ", "")
        if despaced in self._despaced:
            return set(self._despaced[despaced])
        toks = frozenset(_sing(t) for t in n.split() if len(t) > 2)
        return set(self._byset.get(toks, set())) if toks else set()

    @classmethod
    def load_snapshot(cls) -> tuple["TerminologyIndex", dict]:
        """(index, content identity of the exact bytes parsed).

        The identity is captured AT THE PARSE because the index is then answered from
        memory: a file replaced afterwards would otherwise be re-hashed at certification
        time and attested to for retrievals it never served. (Codex F6-R5-B.)
        """
        from app.release.source_manifest import (DeclaredSourceUnavailable,
                                                 declared_document_snapshot)
        document, identity = declared_document_snapshot("index_terms",
                                                        DeclaredSourceUnavailable)
        return cls(document.get("terms", {})), identity

    @classmethod
    def load(cls) -> "TerminologyIndex":
        return cls.load_snapshot()[0]


# ---- governed concept identity for OPEN clinical vocabulary (issue #6 F7-R3-C) ----
# `claude_coder.coreference` compares open-vocabulary axis values (anatomy, approach,
# site, ...) by lexical shape, which can only ever answer "identical string" or "not
# identical" -- it cannot tell a synonym pair apart from a genuine distinction. This is
# the authoritative alternative for anatomy: a versioned concept graph (term ->
# concept, concept -> parent concepts) from SNOMED CT's Body Structure hierarchy,
# compiled by tools/build_snomed_concept_terms.py. REVIEWED-OPTIONAL, same disposition
# as every other licensed recall aid on this adapter: absence degrades the axis
# comparison to its existing conservative behavior, never to a wrong relation.
#
# RELATION SEMANTICS (Codex F7-R3-C3, exact-SHA re-review): SNOMED's IS_A hierarchy is
# a subsumption graph, not a partition. Its own OWL/NNF specification does not encode
# class disjointness, and this codebase's licensed release audits confirm why that
# matters operationally: 19,547 Body Structure concepts have more than one parent, and
# tens of thousands of sibling (non-ancestor) concept pairs still share a descendant --
# "no subsumption edge between these two concepts" is NOT "these are opposed", it is
# simply what an IS_A hierarchy alone cannot tell you. So this graph never asserts
# CONCEPT_DISJOINT (removed as a reachable verdict below; the two open axes it could
# have distinguished stay UNDETERMINED, exactly as they already do with no concept
# source at all -- absence of proof is not proof of absence). Likewise, an exact term
# is sometimes itself AMBIGUOUS in the graph (65 exact terms and 543 token-set keys in
# the current release resolve to more than one concept): two ambiguous terms whose
# candidate sets merely intersect are NOT a confirmed match either -- SAME requires
# each term to resolve to exactly ONE concept, and for those concepts to be equal.

#: The concept graph resolved BOTH terms to exactly ONE, EQUAL concept -- the only
#: confirmed-identity verdict this graph may assert. Requires a UNIQUE match on both
#: sides: an ambiguous term (more than one candidate concept) can share a candidate
#: with the other side without actually naming the same real-world structure.
CONCEPT_SAME = "same"
#: Real evidence of a relation, but not a confirmed verdict either way: either an
#: ancestor/descendant pair (a descendant can be a real, more specific distinction; an
#: ancestor can be the same site described less specifically), or at least one side
#: resolved ambiguously and the candidate sets merely overlap.
CONCEPT_RELATED = "ancestor_descendant"
#: RESERVED, never returned by `ConceptRelationIndex` (an IS_A hierarchy alone cannot
#: establish disjointness -- see the module note above). Kept as a named verdict only
#: for a future source built from an authoritative relation that actually asserts
#: opposition (e.g. a modeled SEP triple or an explicit exclusion axiom), should one
#: ever be added; no caller may treat its mere existence as a signal.
CONCEPT_DISJOINT = "disjoint"
#: Either term did not resolve to a known concept in this graph, or resolved with no
#: relation this graph can state. Never a relation on its own -- the caller's existing
#: lexical-identity fallback applies.
CONCEPT_UNRESOLVED = "unresolved"


@dataclass(frozen=True)
class ConceptMatch:
    """How ONE term resolved against the concept graph: which concept id(s) matched, by
    which matching strategy, and (via `unique`) whether the match was unambiguous."""
    term: str
    candidates: tuple[str, ...]
    method: str   # "exact" | "despaced" | "token_set" | "none"

    @property
    def unique(self) -> bool:
        return len(self.candidates) == 1


@dataclass(frozen=True)
class ConceptRelationDetail:
    """The full, auditable basis for one relation verdict: both sides' matches (concept
    ids, matching strategy, ambiguity), the verdict itself, and a confidence a caller
    can weigh rather than trust blindly. `alternatives_*` is empty exactly when that
    side matched uniquely -- there was nothing else it could have meant."""
    verdict: str
    match_a: ConceptMatch
    match_b: ConceptMatch
    confidence: float

    @property
    def alternatives_a(self) -> tuple[str, ...]:
        return self.match_a.candidates if not self.match_a.unique else ()

    @property
    def alternatives_b(self) -> tuple[str, ...]:
        return self.match_b.candidates if not self.match_b.unique else ()


class ConceptRelationIndex:
    """SAME / ancestor-descendant-or-ambiguous-overlap / unresolved for two clinical
    terms, from an authoritative concept graph -- never from lexical shape, and never a
    fabricated DISJOINT (see the module note above).

    No term or concept identity is named here: the graph itself (loaded from data) is
    the only place clinical vocabulary appears, matching every other terminology index
    on this adapter.
    """

    def __init__(self, concepts: dict):
        self._parents: dict[str, tuple[str, ...]] = {
            cid: tuple(rec.get("parents") or []) for cid, rec in (concepts or {}).items()}
        #: concept id -> every governed term that names it (issue #6 F7-R3-C4): the
        #: REVERSE of the term->concept indexes below, so a UNIQUELY resolved term can
        #: be expanded to its concept's other known names for retrieval, independent of
        #: any comparison against a second value.
        self._terms_by_concept: dict[str, tuple[str, ...]] = {
            cid: tuple(sorted({_norm(t) for t in (rec.get("terms") or []) if _norm(t)}))
            for cid, rec in (concepts or {}).items()}
        self._exact: dict[str, set[str]] = {}
        self._despaced: dict[str, set[str]] = {}
        self._byset: dict[frozenset[str], set[str]] = {}
        for cid, rec in (concepts or {}).items():
            for term in rec.get("terms") or []:
                n = _norm(term)
                if not n:
                    continue
                self._exact.setdefault(n, set()).add(cid)
                self._despaced.setdefault(n.replace(" ", ""), set()).add(cid)
                toks = frozenset(_sing(t) for t in n.split() if len(t) > 2)
                if toks:
                    self._byset.setdefault(toks, set()).add(cid)

    def terms_for_concept(self, concept_id: str) -> tuple[str, ...]:
        """Every governed term that names `concept_id` -- the reverse lookup a
        single-entity normalization expands through (issue #6 F7-R3-C4)."""
        return self._terms_by_concept.get(concept_id, ())

    def match(self, term: str) -> ConceptMatch:
        """One term's full match record: candidates plus WHICH strategy found them
        (exact, compound-word, or order/plural-independent token set -- the same
        strategy order as `TerminologyIndex`), so a caller can audit not just what
        matched but how confidently."""
        n = _norm(term)
        if not n:
            return ConceptMatch(term, (), "none")
        if n in self._exact:
            return ConceptMatch(term, tuple(sorted(self._exact[n])), "exact")
        despaced = n.replace(" ", "")
        if despaced in self._despaced:
            return ConceptMatch(term, tuple(sorted(self._despaced[despaced])), "despaced")
        toks = frozenset(_sing(t) for t in n.split() if len(t) > 2)
        if toks and toks in self._byset:
            return ConceptMatch(term, tuple(sorted(self._byset[toks])), "token_set")
        return ConceptMatch(term, (), "none")

    def candidates(self, term: str) -> set[str]:
        """Candidate concept ids a clinical term could name -- the SAME matching
        strategy as `TerminologyIndex.candidates` (exact, compound-word, then
        order/plural-independent token set), just against concept ids instead of
        authoritative codes."""
        return set(self.match(term).candidates)

    def _ancestors(self, concept_id: str) -> set[str]:
        seen: set[str] = set()
        stack = [concept_id]
        while stack:
            cid = stack.pop()
            for parent in self._parents.get(cid, ()):
                if parent not in seen:
                    seen.add(parent)
                    stack.append(parent)
        return seen

    def relation_detail(self, term_a: str, term_b: str) -> ConceptRelationDetail:
        """The full auditable basis (see `ConceptRelationDetail`) for one relation
        verdict between two clinical terms.

        SAME only after a UNIQUE, EQUAL resolution on both sides (Codex F7-R3-C3):
        an ambiguous term's candidate set merely intersecting the other side's is not
        a confirmed match, because the ambiguous term itself might name a different
        one of its own candidates. An ancestor/descendant relation between ANY
        candidate pair is real hierarchy evidence, surfaced as RELATED -- never
        promoted to SAME (a descendant can be a genuinely more specific, distinct
        structure) and never to a difference. A shared candidate between two
        otherwise-unrelated, ambiguous resolutions is likewise RELATED, not SAME: real
        overlap, not a confirmed identity. Absence of any of the above is UNRESOLVED,
        never DISJOINT -- an IS_A hierarchy has no basis to assert opposition.
        """
        ma, mb = self.match(term_a), self.match(term_b)
        a, b = set(ma.candidates), set(mb.candidates)
        if not a or not b:
            return ConceptRelationDetail(CONCEPT_UNRESOLVED, ma, mb, 0.0)
        if ma.unique and mb.unique and a == b:
            return ConceptRelationDetail(CONCEPT_SAME, ma, mb, 1.0)
        for ca in a:
            ancestors_a = self._ancestors(ca)
            for cb in b:
                if cb in ancestors_a or ca in self._ancestors(cb):
                    return ConceptRelationDetail(CONCEPT_RELATED, ma, mb, 0.0)
        if a & b:
            return ConceptRelationDetail(CONCEPT_RELATED, ma, mb, 0.0)
        return ConceptRelationDetail(CONCEPT_UNRESOLVED, ma, mb, 0.0)

    def relation(self, term_a: str, term_b: str) -> str:
        """One of the CONCEPT_* verdicts above for two clinical terms -- see
        `relation_detail` for the full auditable basis behind it."""
        return self.relation_detail(term_a, term_b).verdict

    def normalize(self, term: str) -> tuple[ConceptMatch, tuple[str, ...]]:
        """This ONE term's match, plus every OTHER term the SAME concept is known by
        (issue #6 F7-R3-C4) -- independent of any comparison against a second value,
        so a single mention (or an abbreviation both readings used identically) is
        still normalized. `expansions` is empty unless the match is UNIQUE: an
        ambiguous term's "other names" would be a guess at which of several real
        concepts it actually meant.
        """
        m = self.match(term)
        expansions = self.terms_for_concept(m.candidates[0]) if m.unique else ()
        return m, tuple(t for t in expansions if t != _norm(term))

    @classmethod
    def load_snapshot(cls, source_id: str = "snomed_concept_terms"
                      ) -> tuple["ConceptRelationIndex", dict]:
        """(index, content identity of the exact bytes parsed) -- same binding
        discipline as `TerminologyIndex.load_snapshot`: the identity is captured at
        the parse because the graph then answers every later relation from memory.

        `source_id` selects WHICH governed concept graph to load -- this class is
        root-concept-agnostic (issue #6 F9-R4), so the same code loads the Body
        Structure snapshot (the default, preserving every existing call site) or the
        Procedure snapshot (`source_id="snomed_procedure_terms"`) without any class
        changes; only the declared source differs.
        """
        from app.release.source_manifest import (DeclaredSourceUnavailable,
                                                 declared_document_snapshot)
        document, identity = declared_document_snapshot(source_id,
                                                        DeclaredSourceUnavailable)
        return cls(document.get("concepts", {})), identity
