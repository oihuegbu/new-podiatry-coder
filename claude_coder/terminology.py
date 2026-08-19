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

#: The concept graph resolved BOTH terms to the SAME concept -- a confirmed match.
CONCEPT_SAME = "same"
#: One concept is an ancestor of the other -- related, but NOT proof of sameness (a
#: descendant can be a real, more specific distinction) and NOT proof of a difference
#: (an ancestor can be the same site described less specifically).
CONCEPT_RELATED = "ancestor_descendant"
#: Both terms resolved to concepts, and neither concept is an ancestor of the other --
#: a genuine, structural opposition the authoritative graph itself asserts.
CONCEPT_DISJOINT = "disjoint"
#: Either term did not resolve to a known concept in this graph. Never a relation on
#: its own -- the caller's existing lexical-identity fallback applies.
CONCEPT_UNRESOLVED = "unresolved"


class ConceptRelationIndex:
    """SAME / ancestor-descendant / DISJOINT / unresolved for two clinical terms,
    from an authoritative concept graph -- never from lexical shape.

    No term or concept identity is named here: the graph itself (loaded from data) is
    the only place clinical vocabulary appears, matching every other terminology index
    on this adapter.
    """

    def __init__(self, concepts: dict):
        self._parents: dict[str, tuple[str, ...]] = {
            cid: tuple(rec.get("parents") or []) for cid, rec in (concepts or {}).items()}
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

    def candidates(self, term: str) -> set[str]:
        """Candidate concept ids a clinical term could name -- the SAME matching
        strategy as `TerminologyIndex.candidates` (exact, compound-word, then
        order/plural-independent token set), just against concept ids instead of
        authoritative codes."""
        n = _norm(term)
        if not n:
            return set()
        if n in self._exact:
            return set(self._exact[n])
        despaced = n.replace(" ", "")
        if despaced in self._despaced:
            return set(self._despaced[despaced])
        toks = frozenset(_sing(t) for t in n.split() if len(t) > 2)
        return set(self._byset.get(toks, set())) if toks else set()

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

    def relation(self, term_a: str, term_b: str) -> str:
        """One of the CONCEPT_* verdicts above for two clinical terms.

        UNRESOLVED whenever either term does not resolve to a known concept in this
        graph -- including a term resolving to several concepts with no relation among
        them, which is the term itself being ambiguous within the graph, not a
        confirmed anything.
        """
        a, b = self.candidates(term_a), self.candidates(term_b)
        if not a or not b:
            return CONCEPT_UNRESOLVED
        if a & b:
            return CONCEPT_SAME
        for ca in a:
            ancestors_a = self._ancestors(ca)
            for cb in b:
                if cb in ancestors_a or ca in self._ancestors(cb):
                    return CONCEPT_RELATED
        return CONCEPT_DISJOINT

    @classmethod
    def load_snapshot(cls) -> tuple["ConceptRelationIndex", dict]:
        """(index, content identity of the exact bytes parsed) -- same binding
        discipline as `TerminologyIndex.load_snapshot`: the identity is captured at
        the parse because the graph then answers every later relation from memory.
        """
        from app.release.source_manifest import (DeclaredSourceUnavailable,
                                                 declared_document_snapshot)
        document, identity = declared_document_snapshot("snomed_concept_terms",
                                                        DeclaredSourceUnavailable)
        return cls(document.get("concepts", {})), identity
