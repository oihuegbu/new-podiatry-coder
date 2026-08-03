"""Deterministic ICD-10-CM resolution via the authoritative Alphabetic Index.

This is the permanent fix for the terse-descriptor / eponym recall gap. The
ICD-10-CM Alphabetic Index (published by NCHS/CMS, parsed into
data/codes/icd10cm_index_terms.json by tools/parse_icd10cm_index.py) is the
AUTHORITATIVE map from clinician vocabulary — "onychomycosis", "Morton's
neuroma", eponyms, lay terms — to codes. It is exactly the knowledge an
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


def _dot(code: str) -> str:
    """ICD-10-CM display form: 3-char category + '.' + remainder (undotted->dotted)."""
    c = str(code).upper().replace(".", "")
    return c if len(c) <= 3 else f"{c[:3]}.{c[3:]}"


class TerminologyIndex:
    """Inverts the authoritative {code: [index terms]} into term→codes lookups:
    an exact normalized-term map, and an order-independent token-set map that
    handles the Index's inverted phrasing ('Neuroma, Morton's') against a note's
    natural phrasing ('Morton's neuroma')."""

    def __init__(self, terms_by_code: dict[str, list[str]]):
        self._exact: dict[str, set[str]] = {}
        self._byset: dict[frozenset[str], set[str]] = {}
        for code, terms in terms_by_code.items():
            dotted = _dot(code)
            for term in terms or []:
                n = _norm(term)
                if not n:
                    continue
                self._exact.setdefault(n, set()).add(dotted)
                toks = frozenset(t for t in n.split() if len(t) > 2)
                if toks:
                    self._byset.setdefault(toks, set()).add(dotted)

    def candidates(self, description: str) -> set[str]:
        """Authoritative ICD-10-CM codes for a clinician term (dotted). Empty if
        the Index does not carry the phrasing (→ caller falls back to retrieval)."""
        n = _norm(description)
        if not n:
            return set()
        exact = self._exact.get(n)
        if exact:
            return set(exact)
        toks = frozenset(t for t in n.split() if len(t) > 2)
        return set(self._byset.get(toks, set())) if toks else set()

    @classmethod
    def load(cls) -> "TerminologyIndex":
        import json
        from app.core.config import DATA_DIR
        with open(DATA_DIR / "codes" / "icd10cm_index_terms.json") as fh:
            terms = json.load(fh).get("terms", {})
        return cls(terms)
