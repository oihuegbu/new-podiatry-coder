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

# Sentinel distinguishing an authority check that COULD NOT RUN (data missing / a
# lookup error) from a check that ran and found no restriction. A caller must never
# read "unavailable" as "no edit": an unrunnable check stops autonomy (fail-closed).
AUTHORITY_UNAVAILABLE = "__unavailable__"


@runtime_checkable
class CodeSource(Protocol):
    def retrieve(self, description: str, system: str,
                 top_k: int = 20) -> list[CandidateCode]: ...

    def lookup(self, code: str, system: str) -> dict[str, Any] | None: ...

    def descriptions(self, code: str, system: str) -> list[str]: ...

    def active_on(self, code: str, system: str, dos: str | None) -> Outcome: ...

    def separately_billable(self, code: str, system: str, dos: str | None) -> Outcome: ...

    def global_period(self, code: str) -> str | None: ...

    def bilat_indicator(self, code: str) -> str | None: ...

    def index_codes(self, description: str, system: str) -> set[str]: ...

    def snomed_codes(self, description: str, system: str) -> set[str]: ...

    def cpt_index_codes(self, description: str, system: str) -> set[str]: ...

    def learned_index_codes(self, description: str, system: str) -> set[str]: ...

    def drug_index_codes(self, description: str, system: str) -> set[str]: ...

    def drug_unit(self, code: str) -> dict | None: ...

    def procedure_index_codes(self, description: str, system: str) -> set[str]: ...

    def leaf_codes(self, stem: str, system: str) -> set[str]: ...

    def ncci_indicator(self, col1: str, col2: str, dos: str | None) -> str | None: ...

    def ncci_edit(self, a: str, b: str, dos: str | None) -> dict | None: ...

    def mue_limit(self, code: str, dos: str | None) -> int | None: ...

    def mue_available(self) -> bool: ...

    def excludes1_refs(self, code: str, system: str) -> set[str]: ...

    def data_fingerprint(self) -> dict[str, Any]: ...

    def qualifying_dx_for(self, code: str, system: str = "cpt") -> set[str] | None: ...


# Data-driven signals that a code is NOT a separately reportable line. These are
# generic terms found in coverage/status fields or the descriptor itself (e.g.
# a descriptor that declares a "noncovered" or "bundled" service) — not a code
# list. Any code carrying such a signal is excluded from the claim.
_NOT_SEPARATELY = ("noncovered", "non-covered", "not separately", "bundled",
                   "packaged", "not payable", "included in", "not billed separately")


class AuthoritativeSource:
    """Adapter over the repo's existing authoritative components. Imports are
    lazy so the models/logic stay importable without the heavy RAG stack."""

    # Authoritative description TIERS, richest first. The AMA CPT data package
    # ships several parallel descriptors per code (full clinical, medium, and the
    # plain-language 'consumer' descriptor); using them ALL gives the deterministic
    # matcher more authoritative surface — e.g. the consumer wording distinguishes
    # one act from a similarly-worded but clinically different act,
    # which the terse clinical descriptor alone blurs. These are real authoritative
    # fields (never the walled-off, llm-generated synonym retrieval aid).
    _DESC_TIERS = ("long_description", "medium_description",
                   "consumer_description", "short_description", "description")

    def __init__(self) -> None:
        self._db = None
        self._store = None
        self._gp: dict | None = None
        self._idx = None
        self._snomed = None
        self._cptidx = None
        self._learned = None
        self._drug = None
        self._drug_units: dict | None = None
        self._rich: dict | None = None

    def _rich_records(self, system: str) -> dict:
        """Raw {code: record} straight from data/codes/<system>_codes.json, which
        carries the FULL set of authoritative description tiers (the in-memory
        CodeReferenceDB keeps only long+short). Cached per system; {} if absent."""
        if self._rich is None:
            self._rich = {}
        if system not in self._rich:
            table: dict[str, dict] = {}
            try:
                import json
                from app.core.config import DATA_DIR
                with open(DATA_DIR / "codes" / f"{system}_codes.json") as fh:
                    data = json.load(fh)
                rows = (data if isinstance(data, list)
                        else data.get("codes") or data.get(system)
                        or next((v for v in data.values() if isinstance(v, list)), []))
                for r in rows:
                    if isinstance(r, dict) and r.get("code"):
                        table[str(r["code"])] = r
            except Exception:
                table = {}
            self._rich[system] = table
        return self._rich[system]

    def descriptions(self, code: str, system: str) -> list[str]:
        """Every authoritative descriptor TIER for a code (long/medium/consumer/
        short), de-duplicated, richest first. Falls back to the in-memory record
        when the rich file is absent. Empty if the code is unknown."""
        rec = self._rich_records(system).get(code) \
            or self._rich_records(system).get(str(code).replace(".", "")) \
            or self.lookup(code, system) or {}
        out, seen = [], set()
        for k in self._DESC_TIERS:
            v = str(rec.get(k) or "").strip()
            if v and v.lower() not in seen:
                seen.add(v.lower())
                out.append(v)
        return out

    def snomed_codes(self, description: str, system: str) -> set[str]:
        """Long-tail authoritative term->ICD-10-CM via the SNOMED CT -> ICD-10-CM
        map (NLM/UMLS): the comprehensive clinical-synonym/eponym layer that
        resolves long-tail eponym/synonym phrasings the ICD Alphabetic Index does
        not carry. Fail-safe: empty when the map file is absent — it needs a (free)
        UMLS license to build; see tools/build_snomed_icd10_map.py."""
        if system != "icd10":
            return set()
        if self._snomed is None:
            try:
                import json
                from app.core.config import DATA_DIR
                from .terminology import TerminologyIndex
                with open(DATA_DIR / "codes" / "snomed_icd10_map.json") as fh:
                    term_to_codes = json.load(fh).get("terms", {})
                # Invert term->codes into code->terms and reuse the SAME robust
                # matcher as the ICD Index (exact / compound / token-set + plural)
                # so variant phrasings (possessive / plural / word order) hit.
                inv: dict[str, list[str]] = {}
                for term, codes in term_to_codes.items():
                    for c in codes:
                        inv.setdefault(c, []).append(term)
                self._snomed = TerminologyIndex(inv)
            except Exception:
                self._snomed = False
        if not self._snomed:
            return set()
        return {c for c in self._snomed.candidates(description)
                if self.leaf_codes(c, "icd10")}

    def cpt_index_codes(self, description: str, system: str) -> set[str]:
        """AUTHORITATIVE procedure term -> CPT code, via the AMA CPT Alphabetic
        Index (the CPT-axis analog of the NCHS ICD Alphabetic Index). This is the
        real term->code map — a documented procedure phrase -> the
        code — that no descriptor/embedding heuristic can reproduce (a note's
        specific value vs a descriptor phrased as 'other than <a different value>').

        Source: data/codes/cpt_index_terms.json, prepared by tools/parse_cpt_index.py
        from the LICENSED AMA CPT Link 'Index file'. Fail-safe: empty set until that
        file is ingested (it is AMA-licensed and not publicly downloadable), so the
        coder simply falls back to the descriptor index + embedding until then. CPT
        only — the AMA Index does not cover HCPCS Level II."""
        if system != "cpt":
            return set()
        if getattr(self, "_cptidx", None) is None:
            try:
                import json
                from app.core.config import DATA_DIR
                from .terminology import TerminologyIndex
                with open(DATA_DIR / "codes" / "cpt_index_terms.json") as fh:
                    terms = json.load(fh).get("terms", {})   # {code: [index phrases]}
                self._cptidx = TerminologyIndex(terms) if terms else False
            except Exception:
                self._cptidx = False
        if not self._cptidx:
            return set()
        out = set()
        for c in self._cptidx.candidates(description):
            code = str(c).replace(".", "")
            if self.lookup(code, system):
                out.add(code)
        return out

    def _load_drug_table(self) -> None:
        if self._drug is not None:
            return
        try:
            import json
            from app.core.config import DATA_DIR
            from .terminology import TerminologyIndex
            data = json.loads((DATA_DIR / "codes" / "hcpcs_drug_table.json").read_text())
            terms = data.get("terms", {})       # {code: [drug names]}
            self._drug = TerminologyIndex(terms) if terms else False
            self._drug_units = data.get("units", {}) or {}
        except Exception:
            self._drug = False
            self._drug_units = {}

    def learned_index_codes(self, description: str, system: str) -> set[str]:
        """DETERMINISTIC phrase -> code via the LEARNED verified-resolution index
        (data/codes/learned_cpt_index.json, promoted by tools/build_learned_index.py
        from prior propose-then-verify results). Self-invalidating: a promoted entry
        is honored only while the code still exists AND its CURRENT authoritative
        descriptor still matches the descriptor that was verified — so a deleted or
        revised code silently falls back to re-verification instead of resolving to
        a stale mapping. Empty until mappings are promoted."""
        if getattr(self, "_learned", None) is None:
            try:
                import json
                from app.core.config import DATA_DIR
                with open(DATA_DIR / "codes" / "learned_cpt_index.json") as fh:
                    self._learned = json.load(fh).get("entries", {}) or {}
            except Exception:
                self._learned = {}
        if not self._learned:
            return set()
        from .terminology import _norm
        entry = self._learned.get(_norm(description))
        if not entry or entry.get("system") != system:
            return set()
        code = str(entry.get("code") or "")
        if not code or not self.lookup(code, system):
            return set()                        # code deleted -> invalidate
        from . import learned
        current = (self.descriptions(code, system) or [""])[0]
        if not learned.entry_current(entry, current):
            return set()                        # descriptor revised -> re-verify
        return {code}

    def drug_index_codes(self, description: str, system: str) -> set[str]:
        """AUTHORITATIVE drug NAME -> HCPCS code, via the CMS Table of Drugs &
        Biologicals (public-domain; prepared by tools/build_hcpcs_drug_table.py).
        The name->code map for J/A/Q drug codes — a documented drug name -> its
        code — the drug analog of the ICD/CPT alphabetic indexes. Fail-safe empty
        until the table is prepared. HCPCS only."""
        if system != "hcpcs":
            return set()
        self._load_drug_table()
        if not self._drug:
            return set()
        out = set()
        for c in self._drug.candidates(description):
            code = str(c).replace(".", "")
            if self.lookup(code, system):
                out.add(code)
        return out

    def drug_unit(self, code: str) -> dict | None:
        """The code's authoritative per-unit dose ({'amount': N, 'unit': 'mg'}) from
        the drug table — used to convert a documented total dose into billing units.
        None if the code is not a dosed drug or the table is absent."""
        self._load_drug_table()
        if not self._drug_units:
            return None
        rec = self._drug_units.get(code) or self._drug_units.get(code.replace(".", ""))
        return rec if isinstance(rec, dict) else None

    def procedure_index_codes(self, description: str, system: str) -> set[str]:
        """Deterministic CPT/HCPCS grounding, the procedure-axis analog of the ICD
        Alphabetic Index (mechanic 5). Builds a TerminologyIndex over the OFFICIAL
        descriptors already in the registry (long + short) and returns the codes a
        documented procedure phrase matches by exact / compound / order-and-plural-
        independent token set. Provenance-clean and deterministic wherever a phrase
        cleanly matches a descriptor; the embedding recall remains the fallback for
        everything else. No code authored here — the index is inverted from the
        descriptor data at load time and self-updates with the code set."""
        if system not in ("cpt", "hcpcs"):
            return set()
        cache = getattr(self, "_pidx", None)
        if cache is None:
            cache = self._pidx = {}
        if system not in cache:
            try:
                from .terminology import TerminologyIndex
                # Prefer the rich file (all description tiers); fall back to the
                # in-memory registry (long+short) when it is not present.
                rich = self._rich_records(system)
                by_code: dict[str, list[str]] = {}
                if rich:
                    for code, rec in rich.items():
                        terms = self.descriptions(code, system)
                        if terms:
                            by_code[str(code)] = terms
                else:
                    for code, rec in getattr(self._reference(), system, {}).items():
                        if not isinstance(rec, dict):
                            continue
                        terms = [d for d in (str(rec.get("long_description") or ""),
                                             str(rec.get("short_description") or "")) if d]
                        if terms:
                            by_code[str(code)] = terms
                cache[system] = TerminologyIndex(by_code) if by_code else False
            except Exception:
                cache[system] = False
        idx = cache[system]
        if not idx:
            return set()
        # TerminologyIndex dots codes (ICD form); CPT/HCPCS have no dot, so strip
        # it back off, and keep only codes that actually exist in this system.
        out = set()
        for c in idx.candidates(description):
            code = str(c).replace(".", "")
            if self.lookup(code, system):
                out.add(code)
        return out

    def index_codes(self, description: str, system: str) -> set[str]:
        """Authoritative ICD-10-CM codes for a clinician term, via the Alphabetic
        Index. ICD-10-CM only; empty set otherwise or when the Index is absent."""
        if system != "icd10":
            return set()
        if self._idx is None:
            try:
                from .terminology import TerminologyIndex
                self._idx = TerminologyIndex.load()
            except Exception:
                self._idx = False
        if not self._idx:
            return set()
        # Keep candidates that resolve to real billable code(s) — a valid leaf OR a
        # category with billable children (expanded downstream). Drops Index noise.
        return {c for c in self._idx.candidates(description) if self.leaf_codes(c, "icd10")}

    def leaf_codes(self, stem: str, system: str) -> set[str]:
        """The billable code(s) at/under a code stem: the code itself if it is a
        billable leaf, otherwise its more-specific billable children — so a
        category the Index returns (e.g. a 4-char category) becomes its leaves (its
        5+-char billable children),
        which the resolver then disambiguates by documented laterality."""
        table = getattr(self._reference(), system, {})
        undot = str(stem).replace(".", "").upper()
        under = {str(k).replace(".", "").upper() for k in table
                 if str(k).replace(".", "").upper().startswith(undot)}
        if not under:
            return set()
        from .terminology import _dot
        if undot in under:                       # the stem is itself a billable leaf
            return {_dot(undot)}
        return {_dot(u) for u in under}          # category -> its billable children

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
        deleted) via the real edit method `check_ncci`. None = the check RAN and
        found no edit for the pair (the loaded snapshot COVERS this DOS);
        AUTHORITY_UNAVAILABLE = the check could not run — a lookup error OR the loaded
        quarterly snapshot does not COVER this DOS — the caller treats that as UNKNOWN,
        never as 'no edit', so an off-quarter DOS fails closed. Robust to return shape."""
        ref = self._reference()
        try:
            # A quarterly NCCI snapshot only speaks for the dates it covers. If it does
            # NOT cover this DOS, the check cannot run -> UNAVAILABLE (UNKNOWN/SYSTEM_HOLD),
            # never a clean 'no edit'. Pass the DOS so the lookup is scoped to the release
            # covering it (edit set and direction can change quarter to quarter).
            available = getattr(ref, "ncci_data_available", None)
            if available is not None and not available(dos):
                return AUTHORITY_UNAVAILABLE                   # snapshot does not cover DOS
            edit = ref.check_ncci(col1, col2, dos)   # type: ignore[attr-defined]
        except Exception:
            return AUTHORITY_UNAVAILABLE                       # check could not run
        if not edit:
            return None                                        # ran; no edit for this pair
        if isinstance(edit, dict):
            for k in ("modifier_indicator", "indicator", "mi", "ptp_modifier"):
                if k in edit:
                    return str(edit[k])
            return "0"                            # an edit exists but no bypass field -> treat as hard
        return str(edit)

    def ncci_edit(self, a: str, b: str, dos: str | None) -> dict | None:
        """The DIRECTIONAL PTP edit for a code pair: which code is payable
        (column 1 / comprehensive) vs the bundled component (column 2), plus the
        modifier indicator. Returns {'payable', 'component', 'modifier'} or None.
        The direction comes from the authoritative row (check_ncci echoes the
        real col1/col2), so a caller can DEMOTE the component rather than block
        the whole claim."""
        db = self._reference()
        try:
            edit = db.check_ncci(a, b, dos)      # type: ignore[attr-defined]
        except Exception:
            return None
        if not isinstance(edit, dict):
            return None
        payable = edit.get("code1")
        component = edit.get("code2")
        if not payable or not component:
            return None
        mod = edit.get("modifier")
        for k in ("modifier", "modifier_indicator", "indicator", "mi"):
            if edit.get(k) is not None:
                mod = edit.get(k)
                break
        return {"payable": str(payable), "component": str(component),
                "modifier": ("0" if mod is None else str(mod))}

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

    def mue_available(self) -> bool:
        """Whether the MUE table is loaded at all. Lets the gate distinguish 'this
        code has no published MUE' (limit None, fine) from 'the MUE table could not
        be loaded' (assert nothing -> UNKNOWN)."""
        try:
            return isinstance(getattr(self._reference(), "mue", None), dict)
        except Exception:
            return False

    def data_fingerprint(self) -> dict[str, Any]:
        """A content fingerprint of the loaded authoritative editions, so the
        release certificate binds the exact data the claim was coded against — a
        changed code set / descriptor edition changes the fingerprint. Uses the
        deployment's codes checksum when present, else per-system code counts."""
        fp: dict[str, Any] = {}
        try:
            db = self._reference()
            fp["counts"] = {s: len(getattr(db, s, {}) or {})
                            for s in ("icd10", "cpt", "hcpcs")}
        except Exception:
            fp["counts"] = {}
        try:
            from app.core.config import DATA_DIR
            chk = DATA_DIR / "qdrant_store" / "codes_checksum.txt"
            if chk.exists():
                fp["codes_checksum"] = chk.read_text().strip()[:64]
        except Exception:
            pass
        return fp

    def _excludes1_map(self) -> dict[str, set[str]]:
        """{undotted category key -> set of undotted Excludes1 target prefixes} from
        the ICD-10-CM Tabular instructional notes (icd10cm_instructional_notes.json).
        Excludes1 = 'NOT CODED HERE': the two conditions are mutually exclusive and
        should not be reported together (unless genuinely unrelated — a human
        judgement). Cached; {} if the file is absent."""
        cache = getattr(self, "_excl1", None)
        if cache is not None:
            return cache
        cache = {}
        try:
            import json
            from app.core.config import DATA_DIR
            with open(DATA_DIR / "codes" / "icd10cm_instructional_notes.json") as fh:
                entries = json.load(fh).get("codes", {})
            for key, rec in entries.items():
                if not isinstance(rec, dict):
                    continue
                refs = rec.get("excludes1_code_refs") or []
                pref = {str(r).replace(".", "").upper() for r in refs if r}
                if pref:
                    cache[str(key).replace(".", "").upper()] = pref
        except Exception:
            cache = {}
        self._excl1 = cache
        return cache

    def _coverage_map(self) -> dict[str, set[str]]:
        """{governed CPT/HCPCS code -> set of qualifying ICD-10 codes} from the CMS
        LCD/Article coverage data (podiatry_lcd.json: governed_cpts x qualifying_dx) —
        the authoritative dx->procedure MEDICAL-NECESSITY linkage. Cached; {} if the
        file is absent (necessity then falls back to the structural check)."""
        cache = getattr(self, "_cov", None)
        if cache is not None:
            return cache
        cache = {}
        try:
            import json
            from app.core.config import DATA_DIR
            d = json.load(open(DATA_DIR / "codes" / "podiatry_lcd.json"))
            for row in (d.get("lcd") or []) + (d.get("article") or []):
                qd = {str(x).replace(".", "").upper()
                      for x in (row.get("qualifying_dx") or [])}
                if not qd:
                    continue
                for c in (row.get("governed_cpts") or []):
                    cache.setdefault(str(c).replace(".", "").upper(), set()).update(qd)
        except Exception:
            cache = {}
        self._cov = cache
        return cache

    def qualifying_dx_for(self, code: str, system: str = "cpt") -> set[str] | None:
        """Authoritative qualifying (medically-necessary) ICD-10 codes for a procedure
        per CMS LCD/Article coverage — the dx->procedure necessity linkage, undotted.
        None when the procedure is governed by NO coverage policy (its necessity
        cannot be confirmed from this source -> callers fail closed); a set when it is
        governed."""
        cov = self._coverage_map()
        return cov.get(str(code).replace(".", "").upper()) if cov else None

    def excludes1_refs(self, code: str, system: str) -> set[str]:
        """The Excludes1 target code-prefixes that apply to a diagnosis — gathered
        from the code AND its ancestor categories (an Excludes1 at a 3-character
        category governs every child code under it). A billed diagnosis whose code
        starts with any
        returned prefix is in an Excludes1 relationship with `code`. ICD-10-CM only;
        empty when the notes file is absent (gate degrades to NOT_APPLICABLE)."""
        if system != "icd10":
            return set()
        table = self._excludes1_map()
        if not table:
            return set()
        undot = str(code).replace(".", "").upper()
        out: set[str] = set()
        for length in range(len(undot), 2, -1):        # code, then each ancestor category
            out |= table.get(undot[:length], set())
        return out


class MockSource:
    """In-memory source for tests. Records use SYNTHETIC identifiers so the test
    suite contains no real medical code."""

    def __init__(self, records: dict[tuple[str, str], dict[str, Any]] | None = None,
                 retrieval: dict[tuple[str, str], list[CandidateCode]] | None = None,
                 ncci: dict[tuple[str, str], str] | None = None,
                 mue: dict[str, int] | None = None,
                 nonbillable: set[str] | None = None,
                 gp: dict[str, str] | None = None,
                 bilat: dict[str, str] | None = None,
                 index: dict[str, set] | None = None,
                 snomed: dict[str, set] | None = None,
                 proc_index: dict[str, set] | None = None,
                 cpt_index: dict[str, set] | None = None,
                 drug_index: dict[str, set] | None = None,
                 drug_units: dict[str, dict] | None = None,
                 learned_index: dict[str, set] | None = None,
                 excludes1: dict[str, set] | None = None,
                 coverage: dict[str, set] | None = None) -> None:
        self._records = records or {}
        self._retrieval = retrieval or {}
        self._ncci = ncci or {}
        self._mue = mue or {}
        self._nonbillable = nonbillable or set()
        self._gp = gp or {}
        self._bilat = bilat or {}
        self._index = index or {}
        self._snomed_map = snomed or {}
        self._proc_index = proc_index or {}
        self._cpt_index = cpt_index or {}
        self._drug_map = drug_index or {}
        self._drug_units = drug_units or {}
        self._learned_map = learned_index or {}
        self._excl1_data = {str(k).replace(".", "").upper():
                            {str(r).replace(".", "").upper() for r in v}
                            for k, v in (excludes1 or {}).items()}
        self._coverage = {str(k).replace(".", "").upper():
                          {str(x).replace(".", "").upper() for x in v}
                          for k, v in (coverage or {}).items()}

    def global_period(self, code):
        return self._gp.get(code)

    def bilat_indicator(self, code):
        return self._bilat.get(code)

    def index_codes(self, description, system):
        return set(self._index.get(description, set())) if system == "icd10" else set()

    def snomed_codes(self, description, system):
        return set(self._snomed_map.get(description, set())) if system == "icd10" else set()

    def cpt_index_codes(self, description, system):
        return set(self._cpt_index.get(description, set())) if system == "cpt" else set()

    def learned_index_codes(self, description, system):
        got = self._learned_map.get(description)
        return {got} if isinstance(got, str) else set(got or set())

    def drug_index_codes(self, description, system):
        return set(self._drug_map.get(description, set())) if system == "hcpcs" else set()

    def drug_unit(self, code):
        return self._drug_units.get(code)

    def procedure_index_codes(self, description, system):
        return set(self._proc_index.get(description, set())) if system in ("cpt", "hcpcs") else set()

    def leaf_codes(self, stem, system):
        undot = str(stem).replace(".", "").upper()
        under = {c for (c, s) in self._records
                 if s == system and c.replace(".", "").upper().startswith(undot)}
        if not under:
            return set()
        exact = {c for c in under if c.replace(".", "").upper() == undot}
        return exact or under

    def retrieve(self, description, system, top_k=20):
        hits = (self._retrieval.get((description, system))
                or self._retrieval.get(("*", system)) or [])
        return hits[:top_k]

    def lookup(self, code, system):
        return self._records.get((code, system))

    def descriptions(self, code, system):
        rec = self._records.get((code, system)) or {}
        tiers = ("long_description", "medium_description", "consumer_description",
                 "short_description", "description")
        return [str(rec[k]) for k in tiers if rec.get(k)]

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

    def ncci_edit(self, a, b, dos):
        # Mock stores directional edits under the payable/comprehensive key
        # (a, b) -> modifier indicator, mirroring the real col1/col2 direction.
        if (a, b) in self._ncci:
            return {"payable": a, "component": b, "modifier": str(self._ncci[(a, b)])}
        if (b, a) in self._ncci:
            return {"payable": b, "component": a, "modifier": str(self._ncci[(b, a)])}
        return None

    def mue_limit(self, code, dos):
        return self._mue.get(code)

    def mue_available(self):
        return True

    def data_fingerprint(self):
        return {"source": "mock"}

    def qualifying_dx_for(self, code, system="cpt"):
        c = str(code).replace(".", "").upper()
        return set(self._coverage[c]) if c in self._coverage else None

    def excludes1_refs(self, code, system):
        if system != "icd10" or not self._excl1_data:
            return set()
        undot = str(code).replace(".", "").upper()
        out = set()
        for length in range(len(undot), 2, -1):
            out |= self._excl1_data.get(undot[:length], set())
        return out
