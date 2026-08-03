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

    def global_period(self, code: str) -> str | None: ...

    def bilat_indicator(self, code: str) -> str | None: ...

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
        self._gp: dict | None = None

    def _pfs(self, code: str) -> dict:
        """The CMS PFS indicator record for a code ({'global':…, 'bilat':…})."""
        if self._gp is None:
            try:
                import json
                from app.core.config import DATA_DIR
                with open(DATA_DIR / "codes" / "global_period.json") as fh:
                    self._gp = json.load(fh).get("codes", {})
            except Exception:
                self._gp = {}
        rec = self._gp.get(code) or self._gp.get(code.replace(".", ""))
        return rec if isinstance(rec, dict) else {}

    def global_period(self, code: str) -> str | None:
        """CMS global-surgical-package days (000/010/090/XXX/YYY/ZZZ/MMM). None if
        unknown. Source: PFS RVU file (tools/build_global_period.py)."""
        return self._pfs(code).get("global")

    def bilat_indicator(self, code: str) -> str | None:
        """CMS bilateral-surgery indicator: '1' = bilateral eligible (modifier 50
        applies), '0'/'2'/'3' = 50 not appropriate, '9' = concept does not apply
        (no laterality modifier). None if unknown."""
        return self._pfs(code).get("bilat")

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
        """PTP modifier indicator ('0' no bypass, '1' bypass-with-modifier, '9'
        deleted) via the real edit method `check_ncci`. None = no edit for the
        pair. Robust to the edit's return shape."""
        db = self._reference()
        try:
            edit = db.check_ncci(col1, col2)     # type: ignore[attr-defined]
        except Exception:
            return None
        if not edit:
            return None
        if isinstance(edit, dict):
            for k in ("modifier_indicator", "indicator", "mi", "ptp_modifier"):
                if k in edit:
                    return str(edit[k])
            return "0"                            # an edit exists but no bypass field -> treat as hard
        return str(edit)

    def mue_limit(self, code: str, dos: str | None) -> int | None:
        """Max medically-unlikely units from the MUE table (a dict keyed by
        code -> {'mue_value': N})."""
        db = self._reference()
        table = getattr(db, "mue", None)
        if not isinstance(table, dict):
            return None
        rec = table.get(code) or table.get(code.replace(".", ""))
        if isinstance(rec, dict):
            val = rec.get("mue_value")
            return int(val) if val is not None else None
        return int(rec) if isinstance(rec, int) else None


class MockSource:
    """In-memory source for tests. Records use SYNTHETIC identifiers so the test
    suite contains no real medical code."""

    def __init__(self, records: dict[tuple[str, str], dict[str, Any]] | None = None,
                 retrieval: dict[tuple[str, str], list[CandidateCode]] | None = None,
                 ncci: dict[tuple[str, str], str] | None = None,
                 mue: dict[str, int] | None = None,
                 nonbillable: set[str] | None = None,
                 gp: dict[str, str] | None = None,
                 bilat: dict[str, str] | None = None) -> None:
        self._records = records or {}
        self._retrieval = retrieval or {}
        self._ncci = ncci or {}
        self._mue = mue or {}
        self._nonbillable = nonbillable or set()
        self._gp = gp or {}
        self._bilat = bilat or {}

    def global_period(self, code):
        return self._gp.get(code)

    def bilat_indicator(self, code):
        return self._bilat.get(code)

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
