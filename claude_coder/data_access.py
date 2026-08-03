"""The authoritative source — the ONLY place code knowledge lives.

Every code, descriptor, activity window, NCCI edit, MUE limit and global period
is READ from the real data already in this repo (CodeReferenceDB / compliance.db
/ the RAG index). The coder logic asks this layer questions; it never contains
codes itself. Swapping in the next quarter's data files changes the answers with
no code change — the property Nym gets from encoding AMA/CMS/WHO as data rather
than logic.

`CodeSource` is a Protocol so the pipeline can run against the real data
(`AuthoritativeSource`) in production and against a `MockSource` in tests — which
is also how the tests avoid embedding any real medical code.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from .models import CandidateCode, Outcome


@runtime_checkable
class CodeSource(Protocol):
    def retrieve(self, description: str, system: str,
                 top_k: int = 20) -> list[CandidateCode]: ...

    def lookup(self, code: str, system: str) -> dict[str, Any] | None: ...

    def active_on(self, code: str, system: str,
                  dos: str | None) -> Outcome: ...

    def ncci_indicator(self, col1: str, col2: str,
                       dos: str | None) -> str | None: ...

    def mue_limit(self, code: str, dos: str | None) -> int | None: ...


class AuthoritativeSource:
    """Adapter over the repo's existing authoritative components. Imports are
    lazy so `claude_coder.models`/logic stay importable without the heavy RAG
    stack (and so the unit tests can run on the mock alone)."""

    def __init__(self) -> None:
        self._db = None
        self._store = None
        self._syn: dict[str, dict] | None = None

    def _synonyms_for(self, code: str, system: str) -> tuple[str, ...]:
        """Clinician synonyms/index terms for a code, from the generated synonym
        layers. These are the bridge between note vocabulary ('Morton's neuroma')
        and a terse descriptor ('Lesion of plantar nerve') — used for concept
        matching only, never as a coding authority."""
        if self._syn is None:
            import json
            from app.core.config import DATA_DIR
            self._syn = {}
            for name, fname in (("cpt", "cpt_synonyms.json"),
                                ("hcpcs", "hcpcs_synonyms.json"),
                                ("icd10", "icd10_synonyms.json")):
                try:
                    self._syn[name] = json.load(
                        open(DATA_DIR / "codes" / fname)).get("terms", {})
                except Exception:
                    self._syn[name] = {}
        terms = self._syn.get(system, {})
        got = terms.get(code) or terms.get(code.replace(".", "")) or ()
        return tuple(got)

    # -- retrieval (concept -> candidate codes) --------------------------------
    def _vector_store(self):
        if self._store is None:
            from app.rag.vector_store import MedicalCodeVectorStore
            self._store = MedicalCodeVectorStore()
            self._store.build_or_load()
        return self._store

    def retrieve(self, description: str, system: str,
                 top_k: int = 20) -> list[CandidateCode]:
        hits = self._vector_store().search(description, system, top_k=top_k)
        out: list[CandidateCode] = []
        for h in hits:
            code = str(h.get("code") or "")
            if not code:
                continue
            # The DESCRIPTOR is authoritative — read it from the code record, not
            # from the index payload (which may omit it). Retrieval only supplies
            # the code identity (recall); the record supplies the truth.
            rec = self.lookup(code, system) or {}
            descriptor = (rec.get("long_description") or rec.get("description")
                          or rec.get("short_description")
                          or h.get("description") or h.get("descriptor") or "")
            out.append(CandidateCode(
                code=code,
                system=system,
                descriptor=str(descriptor),
                score=float(h.get("score") or h.get("rrf") or 0.0),
                source="retrieval",
                aliases=self._synonyms_for(code, system),
                authority={"index": "rag-hybrid", "system": system},
            ))
        return out

    # -- authoritative record lookup ------------------------------------------
    def _reference(self):
        if self._db is None:
            from app.rag.code_reference import CodeReferenceDB
            self._db = CodeReferenceDB()
            self._db.load_all()
        return self._db

    def lookup(self, code: str, system: str) -> dict[str, Any] | None:
        table = getattr(self._reference(), system, {})
        rec = table.get(code) or table.get(code.replace(".", ""))
        return rec if isinstance(rec, dict) else None

    def active_on(self, code: str, system: str, dos: str | None) -> Outcome:
        """POSITIVE, tri-state activity check. Missing DOS or a record without
        a usable activity window is UNKNOWN — never silently "active"."""
        rec = self.lookup(code, system)
        if rec is None:
            return Outcome.BLOCKED            # code does not exist -> hard stop
        if not dos:
            return Outcome.UNKNOWN            # cannot assert activity without DOS
        status = str(rec.get("status", "active")).lower()
        if status not in ("active", ""):
            return Outcome.BLOCKED
        # effective-window fields vary by source; assert only when present.
        start = rec.get("effective_from") or rec.get("add_date")
        end = rec.get("effective_to") or rec.get("term_date") or rec.get("end_date")
        if start and dos < str(start):
            return Outcome.BLOCKED
        if end and dos > str(end):
            return Outcome.BLOCKED
        if not (start or end):
            return Outcome.UNKNOWN            # no window to verify against
        return Outcome.PASS

    def ncci_indicator(self, col1: str, col2: str, dos: str | None) -> str | None:
        db = self._reference()
        fn = getattr(db, "ncci_ptp_indicator", None) or getattr(db, "ncci_edit", None)
        if fn is None:
            return None                       # source can't answer -> caller -> UNKNOWN
        try:
            return fn(col1, col2, dos)         # type: ignore[misc]
        except Exception:
            return None

    def mue_limit(self, code: str, dos: str | None) -> int | None:
        db = self._reference()
        fn = getattr(db, "mue_limit", None) or getattr(db, "mue", None)
        if fn is None:
            return None
        try:
            return fn(code, dos)               # type: ignore[misc]
        except Exception:
            return None


class MockSource:
    """In-memory source for tests. Records are supplied by the test with
    SYNTHETIC identifiers so the test suite contains no real medical code."""

    def __init__(self, records: dict[tuple[str, str], dict[str, Any]] | None = None,
                 retrieval: dict[tuple[str, str], list[CandidateCode]] | None = None,
                 ncci: dict[tuple[str, str], str] | None = None,
                 mue: dict[str, int] | None = None) -> None:
        self._records = records or {}
        self._retrieval = retrieval or {}
        self._ncci = ncci or {}
        self._mue = mue or {}

    def retrieve(self, description, system, top_k=20):
        hits = (self._retrieval.get((description, system))
                or self._retrieval.get(("*", system)) or [])
        return hits[:top_k]

    def lookup(self, code, system):
        return self._records.get((code, system))

    def active_on(self, code, system, dos):
        rec = self._records.get((code, system))
        if rec is None:
            return Outcome.BLOCKED
        if not dos:
            return Outcome.UNKNOWN
        return Outcome.PASS if rec.get("active", True) else Outcome.BLOCKED

    def ncci_indicator(self, col1, col2, dos):
        return self._ncci.get((col1, col2))

    def mue_limit(self, code, dos):
        return self._mue.get(code)
