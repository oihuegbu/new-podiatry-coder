"""The authoritative source — the ONLY place code knowledge lives.

Every code, descriptor, activity window, NCCI edit, MUE limit and separately-
payable status is READ from the real data already in this repo (CodeReferenceDB /
compliance.db / the RAG index). The coder logic asks this layer questions; it
never contains codes itself. Swapping in next quarter's data files changes the
answers with no code change.

RAG review (before reuse): the repo's `MedicalCodeVectorStore.search` is a hybrid
dense (bge) + sparse (BM25) retriever with RRF and a cosine threshold, measured
~98% recall@20. It is sound to reuse AS RECALL, with two integration notes acted
on below: (1) it returns the recall signal under `similarity_score` (not
`score`); (2) the descriptor in the payload varies by system, so the DESCRIPTOR
is read from the authoritative record instead. Retrieval supplies code identity +
a relevance score; the record supplies the truth.

`CodeSource` is a Protocol so the pipeline runs against real data
(`AuthoritativeSource`) in production and a `MockSource` in tests — which is also
how the tests avoid embedding any real medical code.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from .models import CandidateCode, Outcome


@runtime_checkable
class CodeSource(Protocol):
    def retrieve(self, description: str, system: str,
                 top_k: int = 20) -> list[CandidateCode]: ...

    def lookup(self, code: str, system: str) -> dict[str, Any] | None: ...

    def active_on(self, code: str, system: str, dos: str | None) -> Outcome: ...

    def separately_billable(self, code: str, system: str, dos: str | None) -> Outcome: ...

    def ncci_indicator(self, col1: str, col2: str, dos: str | None) -> str | None: ...

    def mue_limit(self, code: str, dos: str | None) -> int | None: ...


# Data-driven signals that a code is NOT a separately reportable line. These are
# generic terms found in coverage/status fields or the descriptor itself (e.g.
# a descriptor that declares a "noncovered" or "bundled" service) — not a code
# list. Any code carrying such a signal is excluded from the claim.
_NOT_SEPARATELY = ("noncovered", "non-covered", "not separately", "bundled",
                   "packaged", "not payable", "included in", "not billed separately")


class AuthoritativeSource:
    """Adapter over the repo's existing authoritative components. Imports are
    lazy so the models/logic stay importable without the heavy RAG stack."""

    def __init__(self) -> None:
        self._db = None
        self._store = None

    # -- retrieval: RECALL only (concept -> candidate code identities) ---------
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
            rec = self.lookup(code, system) or {}
            descriptor = (rec.get("long_description") or rec.get("description")
                          or rec.get("short_description") or "")
            out.append(CandidateCode(
                code=code,
                system=system,
                descriptor=str(descriptor),
                score=float(h.get("similarity_score") or h.get("rerank_score") or 0.0),
                source="retrieval",
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
        """POSITIVE activity check. A code present and active in the current
        authoritative edition, with no window that excludes the DOS, is active.
        Missing code -> BLOCKED; explicit termination before DOS -> BLOCKED;
        missing DOS -> UNKNOWN (cannot assert)."""
        rec = self.lookup(code, system)
        if rec is None:
            return Outcome.BLOCKED
        if not dos:
            return Outcome.UNKNOWN
        status = str(rec.get("status", "active")).lower()
        if status not in ("active", ""):
            return Outcome.BLOCKED
        start = rec.get("effective_from") or rec.get("add_date")
        end = rec.get("effective_to") or rec.get("term_date") or rec.get("end_date")
        if start and dos < str(start):
            return Outcome.BLOCKED
        if end and dos > str(end):
            return Outcome.BLOCKED
        return Outcome.PASS

    def separately_billable(self, code: str, system: str, dos: str | None) -> Outcome:
        """Is this code a separately payable LINE? Data-driven: an explicit
        non-covered / bundled / packaged signal (in a status field or the
        descriptor), or an MUE of 0, means it is not separately reportable and
        must not appear as its own line. No such signal -> UNKNOWN (kept)."""
        rec = self.lookup(code, system) or {}
        blob = " ".join(str(rec.get(k, "")).lower() for k in (
            "coverage", "coverage_status", "payment_indicator", "status_indicator",
            "separately_payable", "billable", "long_description", "description"))
        if any(w in blob for w in _NOT_SEPARATELY):
            return Outcome.BLOCKED
        mue = self.mue_limit(code, dos)
        if mue is not None and mue == 0:
            return Outcome.BLOCKED
        return Outcome.UNKNOWN

    def ncci_indicator(self, col1: str, col2: str, dos: str | None) -> str | None:
        db = self._reference()
        fn = getattr(db, "ncci_ptp_indicator", None) or getattr(db, "ncci_edit", None)
        if fn is None:
            return None
        try:
            return fn(col1, col2, dos)          # type: ignore[misc]
        except Exception:
            return None

    def mue_limit(self, code: str, dos: str | None) -> int | None:
        db = self._reference()
        fn = getattr(db, "mue_limit", None) or getattr(db, "mue", None)
        if fn is None:
            return None
        try:
            return fn(code, dos)                # type: ignore[misc]
        except Exception:
            return None


class MockSource:
    """In-memory source for tests. Records use SYNTHETIC identifiers so the test
    suite contains no real medical code."""

    def __init__(self, records: dict[tuple[str, str], dict[str, Any]] | None = None,
                 retrieval: dict[tuple[str, str], list[CandidateCode]] | None = None,
                 ncci: dict[tuple[str, str], str] | None = None,
                 mue: dict[str, int] | None = None,
                 nonbillable: set[str] | None = None) -> None:
        self._records = records or {}
        self._retrieval = retrieval or {}
        self._ncci = ncci or {}
        self._mue = mue or {}
        self._nonbillable = nonbillable or set()

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

    def separately_billable(self, code, system, dos):
        return Outcome.BLOCKED if code in self._nonbillable else Outcome.UNKNOWN

    def ncci_indicator(self, col1, col2, dos):
        return self._ncci.get((col1, col2))

    def mue_limit(self, code, dos):
        return self._mue.get(code)
