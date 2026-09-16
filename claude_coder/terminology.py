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

from .ontology import _LATERALITY


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ",
                  str(s).lower().replace("'", ""))).strip()


#: Public alias (issue #6 F9-R7 item 2): `tools/build_umls_crosswalk.py` and
#: `data_access.AuthoritativeSource.umls_candidates` must normalize a term the
#: SAME way this index does when it builds/reads `term_to_cuis` -- a mismatched
#: normalization is a silent, undetectable recall gap, so both sides import
#: this ONE function rather than each keeping its own private copy in sync by
#: hand.
normalize_term = _norm


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


_MIN_DISTINCTIVE_TOKEN_LENGTH = 5
_MAX_DISTINCTIVE_SOURCE_TERMS = 3
# A governed source phrase may occur inside a clinician's more detailed phrase
# ("qualifier source-term extra-detail") without being a fuzzy match.  This is
# deliberately stricter than token overlap: every source token has to be present
# verbatim, and a one-token source phrase is never enough to seed recall.  The
# production consumer is the SNOMED-to-ICD mapping, whose output is still only a
# verification-required candidate, never a coding decision.
_MIN_CONTAINED_SOURCE_TOKENS = 2
# issue #6, independent review: a laterality/orientation word ("right", "left",
# "bilateral", "side") carries no clinical specificity on its own -- it appears as a
# contained phrase in THOUSANDS of unrelated Alphabetic Index entries across every
# body system and chapter, purely because most injury/condition entries state which
# side they're on. Requiring only "at least two contained tokens" let a source term
# built ENTIRELY from these words (e.g. "right side") satisfy that bar while
# contributing zero real diagnostic meaning, seeding recall with an unbounded,
# essentially random set of codes for ANY fact description that happens to end in a
# side qualifier. Reproduced directly: "condition alpha of the right side" (a generic
# synthetic fixture with no real diagnostic content) matched "right side" against a
# real Alphabetic Index term and recalled pathologic-fracture, congenital-anomaly,
# skull-fracture, and traumatic-brain-injury codes alike -- all sharing nothing with
# the fact except the word "side". A contained match must contribute at least one
# token beyond pure orientation/laterality wording. Not a code or clinical-term list --
# generic English orientation words, the same class already excluded elsewhere in
# this codebase's own distinctive-token conventions.
_ORIENTATION_ONLY_TOKENS = set(_LATERALITY) | {"side"}


class TerminologyIndex:
    """Inverts the authoritative {code: [index terms]} into term→codes lookups:
    an exact normalized-term map, and an order-independent token-set map that
    handles the Index's inverted phrasing ('Entity, qualifying-term') against a
    note's natural phrasing ('qualifying-term entity').

    Takes two DISTINCT term sources (issue #6 F9-R12-A): `terms_by_code`
    (direct Index entries) and `cross_reference_terms_by_code` (redirect
    aliases, e.g. "paronychia" resolving through a `<seeAlso>` chain to the
    cellulitis family). Both are folded into the SAME lookup structures here
    -- `candidates()` answers exact-term Index lookup, where a redirect alias
    is legitimate signal (deferred to candidate narrowing downstream, never
    trusted deterministically for a multi-code hit -- see
    `claude_coder.resolution`). They are kept as separate keys in the source
    JSON, and loaded separately here, specifically so callers building
    PER-CODE EMBEDDING TEXT (`app/rag/vector_store.py`) can use `terms_by_code`
    alone and never flatten a broad redirect alias into a code's embedding
    vector -- this class itself does not embed anything, so it is safe to
    merge both for its own exact-lookup purpose.

    issue #6 F9-R12-A, REOPENED (Codex's re-review): merging both tiers into
    ONE untyped lookup meant a caller had no way to tell a DIRECT hit from a
    redirect-only hit -- `resolution.resolve()`'s single-hit deterministic
    gate (`len(idx) == 1`) trusted a `seeAlso` hit exactly like a direct
    one, even though a redirect is supplementary navigation, not proof the
    note supports that code. `direct_candidates()` answers the SAME lookup
    using ONLY `terms_by_code`, via a second, wholly separate index built
    from direct terms alone -- so a caller can require a single-hit match to
    ALSO be direct before trusting it deterministically."""

    def __init__(self, terms_by_code: dict[str, list[str]],
                cross_reference_terms_by_code: dict[str, list[str]] | None = None):
        self._exact: dict[str, set[str]] = {}
        self._despaced: dict[str, set[str]] = {}   # 'two words' <-> 'twowords'
        self._byset: dict[frozenset[str], set[str]] = {}   # order + plural independent
        # Exact-token statistics support a deliberately narrow RECALL fallback for
        # governed synonym maps. This is not fuzzy matching: a query token must occur
        # verbatim in the source terminology, be rare in that terminology, and point
        # to exactly one code. The only production caller is the SNOMED-to-ICD recall
        # layer, whose result is always descriptor-entailment verified before selection.
        # The direct ICD Alphabetic Index path never calls this fallback.
        self._token_codes: dict[str, set[str]] = {}
        self._token_terms: dict[str, set[str]] = {}
        self._terms_by_code: dict[str, set[str]] = {}
        # Exact multi-token source phrases, indexed by each source token.  Recall
        # chooses the smallest query-token bucket before checking token-set
        # containment, so longer terminology tables do not turn a diagnosis lookup
        # into a full-table scan.
        self._contained_by_token: dict[str, list[tuple[str, frozenset[str], str]]] = {}
        self._index_terms(terms_by_code)
        # A separate, direct-only index (issue #6 F9-R12-A, reopened) --
        # recursing with no cross-reference argument terminates immediately
        # (its own `_direct` becomes itself), so this never merges the two
        # tiers' underlying sets.
        self._direct = self if not cross_reference_terms_by_code \
            else TerminologyIndex(terms_by_code)
        if cross_reference_terms_by_code:
            self._index_terms(cross_reference_terms_by_code)

    def _index_terms(self, terms_by_code: dict[str, list[str]]) -> None:
        for code, terms in terms_by_code.items():
            dotted = _dot(code)
            for term in terms or []:
                n = _norm(term)
                if not n:
                    continue
                self._exact.setdefault(n, set()).add(dotted)
                self._terms_by_code.setdefault(dotted, set()).add(n)
                self._despaced.setdefault(n.replace(" ", ""), set()).add(dotted)
                toks = frozenset(_sing(t) for t in n.split() if len(t) > 2)
                if toks:
                    for token in toks:
                        self._token_codes.setdefault(token, set()).add(dotted)
                        self._token_terms.setdefault(token, set()).add(n)
                    self._byset.setdefault(toks, set()).add(dotted)
                    if len(toks) >= _MIN_CONTAINED_SOURCE_TOKENS:
                        entry = (dotted, toks, n)
                        for token in toks:
                            self._contained_by_token.setdefault(token, []).append(entry)

    def _whole_match(self, description: str) -> tuple[set[str], str, object]:
        """(codes, method, comparison key) for the established whole-term match."""
        n = _norm(description)
        if not n:
            return set(), "none", ""
        if n in self._exact:
            return set(self._exact[n]), "exact", n
        despaced = n.replace(" ", "")
        if despaced in self._despaced:
            return set(self._despaced[despaced]), "despaced", despaced
        toks = frozenset(_sing(t) for t in n.split() if len(t) > 2)
        if toks and toks in self._byset:
            return set(self._byset[toks]), "token_set", toks
        return set(), "none", ""

    def candidates(self, description: str) -> set[str]:
        """Authoritative ICD-10-CM codes for a clinician term (dotted). Matches in
        order: exact normalized, compound-word (despaced), then order/plural-
        independent token set. Empty if the Index does not carry the phrasing
        (→ caller falls back to retrieval)."""
        return self._whole_match(description)[0]

    def direct_candidates(self, description: str) -> set[str]:
        """Like `candidates()`, but ONLY through a DIRECT Index entry --
        never a `<see>`/`<seeAlso>` cross-reference redirect alias (issue #6
        F9-R12-A, reopened). A caller uses this to require a hit be direct
        before trusting a single-code match deterministically."""
        return self._direct.candidates(description)

    def recall_matches(self, description: str) -> dict[str, dict]:
        """Auditable source-derived matches for a longer clinical phrase.

        Whole-term matching remains authoritative and is tried first. If it has no
        hit, an exact query token may seed a candidate only when the loaded terminology
        itself proves that token is distinctive using the generic limits above. Each
        result records the source terms and method that produced it so a later selector
        can distinguish a governed term-to-code mapping from vector similarity. This is
        recall only; callers must independently verify the current descriptor and all
        documented requirements.
        """
        normalized = _norm(description)
        codes, method, key = self._whole_match(description)
        if codes:
            matches: dict[str, dict] = {}
            for code in sorted(codes):
                source_terms = self._terms_by_code.get(code, set())
                if method == "exact":
                    terms = [term for term in source_terms if term == key]
                elif method == "despaced":
                    terms = [term for term in source_terms if term.replace(" ", "") == key]
                else:
                    terms = [term for term in source_terms
                             if frozenset(_sing(t) for t in term.split() if len(t) > 2) == key]
                matches[code] = {
                    "method": method,
                    "normalized_query": normalized,
                    "source_terms": sorted(terms),
                }
            return matches

        matches: dict[str, dict] = {}
        query_tokens = {_sing(t) for t in normalized.split() if len(t) > 2}

        # Exact contained governed phrase recall.  Clinical extraction often
        # preserves additional documented qualifiers around a terminology term;
        # requiring whole-string equality in that situation turns a real source
        # mapping into a needless recall gap.  This comparison remains entirely
        # lexical and source-bounded: it neither edits/stems medical vocabulary nor
        # treats a near string as equivalent.  A matching term must contribute at
        # least two of its own normalized tokens, all of which appear verbatim in
        # the query.  As with every other result from this method, callers use it
        # to propose candidates for independent descriptor/evidence verification.
        contained: dict[str, list[str]] = {}
        # A contained source phrase can be anchored by *any* one of its tokens.
        # Choosing only the globally smallest query-token bucket looks efficient,
        # but is incorrect: an incidental, rare qualifier in the note can select a
        # bucket unrelated to the actual contained source phrase and silently lose
        # its candidate.  Union the bounded token buckets, then de-duplicate the
        # exact source entries before testing full containment.  This remains a
        # source-bounded lookup (not a table scan or fuzzy match), while making
        # every exact contained phrase discoverable regardless of extra wording.
        entries, seen_entries = [], set()
        for token in sorted(query_tokens):
            for entry in self._contained_by_token.get(token, ()):
                if entry not in seen_entries:
                    seen_entries.add(entry)
                    entries.append(entry)
        for code, source_tokens, source_term in entries:
            if source_tokens <= query_tokens and (source_tokens - _ORIENTATION_ONLY_TOKENS):
                contained.setdefault(code, []).append(source_term)
        for code, source_terms in contained.items():
            matches[code] = {
                "method": "contained_source_phrase",
                "normalized_query": normalized,
                "source_terms": sorted(set(source_terms)),
            }

        tokens = {_sing(t) for t in normalized.split()
                  if len(t) >= _MIN_DISTINCTIVE_TOKEN_LENGTH}
        for token in sorted(tokens):
            # A pure orientation/laterality word is never "distinctive," no
            # matter how few codes it happens to map to -- the same principle
            # as the contained-phrase guard above, defense-in-depth for this
            # separate fallback path.
            if token in _ORIENTATION_ONLY_TOKENS:
                continue
            codes = self._token_codes.get(token) or set()
            terms = self._token_terms.get(token) or set()
            if (len(codes) != 1 or not terms
                    or len(terms) > _MAX_DISTINCTIVE_SOURCE_TERMS):
                continue
            code = next(iter(codes))
            # A contained multi-token phrase is stronger than a one-token
            # distinctiveness fallback.  Preserve it as the primary method while
            # recording any additional rare-token lineage, rather than letting a
            # weaker match overwrite the audit explanation.
            record = matches.get(code)
            if record is None:
                record = {
                    "method": "distinctive_source_token",
                    "normalized_query": normalized,
                    "matched_tokens": [],
                    "source_terms": [],
                }
                matches[code] = record
            if record["method"] == "distinctive_source_token":
                record["matched_tokens"].append(token)
            record["source_terms"] = sorted(set(record["source_terms"]) | set(terms))
        return matches

    def recall_candidates(self, description: str) -> set[str]:
        """Candidate-code compatibility wrapper over :meth:`recall_matches`."""
        return set(self.recall_matches(description))

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
        return cls(document.get("terms", {}),
                   document.get("cross_reference_terms", {})), identity

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
    # Coverage of the Cartesian product of candidate concepts.  "all" means
    # every possible interpretation on both sides is equal or linked by the
    # governed hierarchy; "some" means only at least one interpretation is;
    # "none" means none are.  This keeps an ambiguous compound term from being
    # treated like an arbitrary single match while allowing callers to
    # distinguish universally compatible ambiguity from partial overlap.
    pair_coverage: str = "none"
    related_pair_count: int = 0
    total_pair_count: int = 0

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
        #: {phrase length in tokens -> {token tuple -> concept ids}} (issue #6 F9-R4-R1):
        #: the index `match_longest` scans, built from the SAME governed terms as every
        #: other lookup here -- no separate vocabulary, just a different access pattern
        #: (an exact multi-token PHRASE occurring anywhere in a longer string, not the
        #: whole string matching a term).
        self._phrases: dict[int, dict[tuple[str, ...], set[str]]] = {}
        self._max_phrase_tokens = 0
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
                phrase = tuple(n.split())
                if phrase:
                    self._phrases.setdefault(len(phrase), {}).setdefault(phrase, set()).add(cid)
                    self._max_phrase_tokens = max(self._max_phrase_tokens, len(phrase))

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

    def match_longest(self, text: str) -> ConceptMatch:
        """Like `match`, but for a full action DESCRIPTION rather than a bare term
        (issue #6 F9-R4-R1): tries a whole-string `match` first (unchanged), and only
        when that finds nothing, scans `text` LEFT TO RIGHT for governed phrases,
        taking the LONGEST governed phrase that starts AT EACH POSITION and unioning
        concepts from every non-overlapping match found along the way. Deliberately
        NOT fuzzy/edit-distance/phonetic -- only an EXACT token-boundary phrase from
        the governed term table. A window (or two different windows at the same
        position) matching more than one concept leaves `candidates` with all of them,
        which `ConceptMatch.unique` already turns into a non-unique, non-SAME-eligible
        result -- no separate ambiguity handling needed here.

        issue #6 F9-R4-R2 (Codex re-review): an earlier version stopped scanning as
        soon as ANY width had ANY hit anywhere in the text, so a longer governed
        phrase silently masked a separate, shorter governed phrase mentioned
        elsewhere in the same compound action ("removal of bone spur and tenotomy
        were performed" only ever found "removal of bone spur", never "tenotomy").
        Scanning every position and only skipping past a match's own consumed
        tokens fixes that while keeping "longest wins" exactly where it actually
        applies: two phrases that could both start at the SAME position.
        """
        direct = self.match(text)
        if direct.candidates:
            return direct
        tokens = tuple(_norm(text).split())
        hits: set[str] = set()
        matched_widths: list[int] = []
        cursor = 0
        while cursor < len(tokens):
            found = False
            for width in range(min(self._max_phrase_tokens, len(tokens) - cursor), 0, -1):
                concepts = self._phrases.get(width, {}).get(tokens[cursor:cursor + width])
                if concepts:
                    hits.update(concepts)
                    matched_widths.append(width)
                    cursor += width
                    found = True
                    break
            if not found:
                cursor += 1
        method = ("token_scan:" + ",".join(map(str, matched_widths))) if hits else "none"
        return ConceptMatch(text, tuple(sorted(hits)), method)

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

    def relation_detail(self, term_a: str, term_b: str, *,
                        embedded: bool = False) -> ConceptRelationDetail:
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

        `embedded` (issue #6 F9-R4-R1) routes matching through `match_longest`
        instead of `match` -- for a full action DESCRIPTION (a summary line or a
        narrative sentence), which is very unlikely to be, in its entirety, exactly
        one governed term, versus a single free-text VALUE (e.g. an anatomy string),
        which usually is. Only the caller (`data_access.procedure_relation_detail`)
        opts into this; every other governed axis keeps the stricter whole-string
        match unchanged.
        """
        matcher = self.match_longest if embedded else self.match
        ma, mb = matcher(term_a), matcher(term_b)
        a, b = set(ma.candidates), set(mb.candidates)
        if not a or not b:
            return ConceptRelationDetail(CONCEPT_UNRESOLVED, ma, mb, 0.0,
                                         "none", 0, len(a) * len(b))
        if ma.unique and mb.unique and a == b:
            return ConceptRelationDetail(CONCEPT_SAME, ma, mb, 1.0, "all", 1, 1)
        total = len(a) * len(b)
        related = 0
        for ca in a:
            ancestors_a = self._ancestors(ca)
            for cb in b:
                if ca == cb or cb in ancestors_a or ca in self._ancestors(cb):
                    related += 1
        if related:
            coverage = "all" if related == total else "some"
            return ConceptRelationDetail(CONCEPT_RELATED, ma, mb, 0.0,
                                         coverage, related, total)
        return ConceptRelationDetail(CONCEPT_UNRESOLVED, ma, mb, 0.0,
                                     "none", 0, total)

    def relation(self, term_a: str, term_b: str) -> str:
        """One of the CONCEPT_* verdicts above for two clinical terms -- see
        `relation_detail` for the full auditable basis behind it."""
        return self.relation_detail(term_a, term_b).verdict

    def normalize(self, term: str, *, embedded: bool = False
                 ) -> tuple[ConceptMatch, tuple[str, ...]]:
        """This ONE term's match, plus every OTHER term the SAME concept is known by
        (issue #6 F7-R3-C4) -- independent of any comparison against a second value,
        so a single mention (or an abbreviation both readings used identically) is
        still normalized. `expansions` is empty unless the match is UNIQUE: an
        ambiguous term's "other names" would be a guess at which of several real
        concepts it actually meant.

        `embedded=True` (issue #6, F9-R13-D architectural-gap follow-up): matches via
        `match_longest` instead of the strict whole-string `match`, for a
        DESCRIPTIVE value that contains a governed term rather than being one in its
        entirety -- the same widening `relation_detail(..., embedded=True)` already
        applies to a two-value comparison, extended here to the single-value
        normalization/retrieval-expansion path so a verbose phrase can still expand
        under its governed synonyms.
        """
        m = (self.match_longest if embedded else self.match)(term)
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
