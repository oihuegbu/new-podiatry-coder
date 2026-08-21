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

import hashlib
from typing import Any, Protocol, runtime_checkable

# The ONE typed source registry, shared by both halves of the deployed image. Importing
# the base error and the fail-closed readers from it (rather than defining a second set
# here) is what makes "declared source" mean the same thing in `app/**` and in
# `claude_coder/**`. (Codex F6-R5-A, round 6.)
from app.release.source_manifest import (
    DeclaredSourceUnavailable, SourceSnapshotSet,
    declared_document as _declared_document,
    declared_document_snapshot as _declared_document_snapshot,
    declared_json_snapshot as _declared_json_snapshot,
    declared_table as _declared_table,
    declared_table_snapshot as _declared_table_snapshot)

from .models import CandidateCode, Outcome

# Sentinel distinguishing an authority check that COULD NOT RUN (data missing / a
# lookup error) from a check that ran and found no restriction. A caller must never
# read "unavailable" as "no edit": an unrunnable check stops autonomy (fail-closed).
AUTHORITY_UNAVAILABLE = "__unavailable__"


class AuthoritativeDataUnavailable(DeclaredSourceUnavailable):
    """A REQUIRED authoritative source could not be read as authoritative data.

    ABSENCE of a required source is caught upstream: the capability manifest reports it as
    `missing_required` and `source_manifest_gate` BLOCKS before any certificate exists.
    PRESENCE is what this exception exists for. A file that is present but malformed,
    truncated, or structurally wrong is hashed happily by the manifest -- it is there, it
    has bytes, it has an identity -- and every one of these read paths used to swallow the
    parse failure into an EMPTY table.

    An empty table is never a neutral answer here; for each of these sources it is the
    PERMISSIVE one: "this code has no global period" (so the global surgical package is not
    applied), "no modifier applies" (so the claim goes out bare), "this diagnosis carries no
    Excludes1 note" (so the conflict gate passes), "this service is governed by no coverage
    policy" (so the less restrictive ungoverned path is taken). Corruption therefore RELAXED
    the claim instead of holding it -- absence blocked, corruption did not.

    Every consumer converts this into a hold: the pipeline's fail-closed boundary for the
    tables read during claim ASSEMBLY (before any gate runs), and the owning gate for the
    tables a gate itself reads. (Round 5, phase 4.)
    """


class CoverageDataUnavailable(AuthoritativeDataUnavailable):
    """The authoritative coverage policy could not be read. Raised (never degraded to an
    empty map) because "no coverage data" and "this service is governed by no policy" are
    opposite conclusions: the second releases a claim the first must hold."""


class PfsIndicatorsUnavailable(AuthoritativeDataUnavailable):
    """The CMS PFS indicator table (global period + bilateral indicator) could not be read.
    Raised because an empty table makes every code look like it has NO global period -- so
    `apply_global_package` stops bundling a related same-day E/M into the global surgical
    package -- and like the bilateral concept does not apply, changing which laterality
    modifiers the claim carries."""


class InstructionalNotesUnavailable(AuthoritativeDataUnavailable):
    """The ICD-10-CM Tabular instructional notes could not be read. Raised because an empty
    note table makes every diagnosis pair look Excludes1-clean, which does not merely lose a
    lookup: it RELAXES the Excludes1 compliance gate from a real check into a silent pass."""


class SemanticClassUnavailable(AuthoritativeDataUnavailable):
    """The authoritative code-classification config (`coding_semantics.json`) or one
    of the reference tables its rules read (ICD-10-CM chapter boundaries, the
    licensed CPT category membership) could not be read. Raised rather than
    degraded to "no class": an unclassified code is never a proxy for "not a
    surgical procedure" or "not evaluation and management" -- those are real,
    consequential distinctions for family-eligibility narrowing (issue #6,
    compiled-semantic-layer plan item 1), and losing the classifier must hold
    that narrowing, never silently widen it back to the whole code space."""


class ModifierDefinitionsUnavailable(AuthoritativeDataUnavailable):
    """The authoritative modifier definitions could not be read. Raised because the engine
    DISCOVERS every modifier it emits from these descriptions, so an empty table is
    indistinguishable from "no modifier is warranted" and silently ships a claim stripped of
    the laterality / distinct-service / separately-identifiable-E-M modifiers it needs."""


def _source_path(source_id: str):
    """The declared path of an authoritative source the decision path reads.

    Resolving through the release-source declaration -- instead of composing a filename
    literal here -- is what keeps "the bytes the certificate attests" and "the bytes the
    coder reads" the same object: an identity that is not registered and dispositioned
    raises rather than becoming a file nobody certifies. (Codex F6-R5, round 5.)
    """
    from app.release.source_manifest import declared_source_path
    return declared_source_path(source_id)


def declared_document(source_id: str,
                      error: type[AuthoritativeDataUnavailable]) -> dict:
    """The parsed JSON document a REQUIRED authoritative source publishes, read FAIL-CLOSED.

    Delegates to the implementation that now lives beside the declaration itself
    (`app.release.source_manifest.declared_document`) so `app/**`'s readers and this
    module's share ONE mechanic rather than two that can drift.  Re-exported here under
    the name the coder's modules already import.  (Codex F6-R5-A, round 6.)
    """
    return _declared_document(source_id, error)


def declared_table(source_id: str, key: str,
                   error: type[AuthoritativeDataUnavailable]) -> dict:
    """The non-empty mapping a REQUIRED authoritative source publishes under `key`.

    NON-EMPTY is part of the contract, not a nicety: a document that parses but carries no
    table (wrong schema, truncated write, an extract whose builder failed) yields exactly
    the same `{}` the swallowed-exception path used to yield, and `{}` is the permissive
    answer for every one of these sources.  See `declared_document` above for why the
    implementation lives with the declaration.
    """
    return _declared_table(source_id, key, error)


def declared_document_snapshot(source_id: str,
                               error: type[AuthoritativeDataUnavailable]) -> tuple:
    """(document, identity of the exact bytes parsed) -- for a read whose result is CACHED.

    A cached read answers every later decision from memory, so the file at the path when
    the certificate is built is not what answered anything.  The identity captured at the
    parse is bound into the release fingerprint and the certificate must match it.
    (Codex F6-R5-B.)
    """
    return _declared_document_snapshot(source_id, error)


def declared_table_snapshot(source_id: str, key: str,
                            error: type[AuthoritativeDataUnavailable]) -> tuple:
    """(non-empty table, identity of the exact bytes parsed) -- see
    `declared_document_snapshot`."""
    return _declared_table_snapshot(source_id, key, error)


def declared_json_snapshot(source_id: str,
                           error: type[AuthoritativeDataUnavailable]) -> tuple:
    """(parsed JSON of ANY shape, identity of the exact bytes parsed).

    Any shape because the authoritative code tables are JSON ARRAYS, not objects.
    """
    return _declared_json_snapshot(source_id, error)


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

    def drug_dose_table_available(self) -> bool: ...

    def procedure_index_codes(self, description: str, system: str) -> set[str]: ...

    def leaf_codes(self, stem: str, system: str) -> set[str]: ...

    def ncci_indicator(self, col1: str, col2: str, dos: str | None) -> str | None: ...

    def ncci_edit(self, a: str, b: str, dos: str | None) -> dict | None: ...

    def mue_limit(self, code: str, dos: str | None) -> int | None: ...

    def mue_available(self) -> bool: ...

    def excludes1_refs(self, code: str, system: str) -> set[str]: ...

    def component_relationships(self, code: str, system: str) -> dict[str, set[str]]: ...

    def semantic_class(self, code: str, system: str) -> str | None: ...

    def assert_claim_assembly_data_readable(self) -> None: ...

    def data_fingerprint(self) -> dict[str, Any]: ...

    def qualifying_dx_for(self, code: str, system: str = "cpt") -> set[str] | None: ...

    def concept_relation(self, term_a: str, term_b: str) -> str: ...

    def concept_relation_detail(self, term_a: str, term_b: str) -> dict: ...

    def concept_lookup(self, axis: str, term: str) -> dict: ...


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
        self._concept_relation_index = None
        self._concept_relation_identity = None
        self._cptidx = None
        self._learned = None
        self._drug = None
        self._drug_units: dict | None = None
        self._rich: dict | None = None
        # The content identities of the declared sources THIS adapter parsed and cached.
        # The code/edit tables are bound by `CodeReferenceDB` (which parses them); the
        # lazily-read policy and rule documents below are parsed here, so their identities
        # are captured here and merged into one set at `data_fingerprint`. Same defect and
        # same technique as the compiled database: the bytes that produced the cached copy,
        # not a re-hash of the path at certification time. (Codex F6-R5-B.)
        self._bound_sources = SourceSnapshotSet()

    def prepare(self, force_rebuild_index: bool = False) -> None:
        """Build/load every heavy dependency ONCE, up front, and fail loudly if it
        cannot be built.

        Everything this adapter reads is lazy by design, so a batch that never calls
        this still works — it just pays the vector-store build (up to ~60-90 min on a
        cold Qdrant) and the reference-DB load inside the first encounter it codes,
        where the cost looks like a hung first note and where a *missing* dependency
        surfaces as that note's system hold rather than as a startup failure.

        This is the entrypoint's `--setup-only` step (and its pre-batch warm-up):

          * the Qdrant hybrid collections — rebuilt from scratch when
            `force_rebuild_index` is set, otherwise loaded/refreshed by checksum;
          * the authoritative code-reference tables;
          * the claim-assembly data (PFS indicators + modifier definitions), asserted
            readable here for the same reason `code_encounter` asserts it: an empty
            table is the PERMISSIVE answer for both, so its unavailability must be an
            error, never a default. Raises `AuthoritativeDataUnavailable`.
        """
        if force_rebuild_index:
            from app.rag.vector_store import MedicalCodeVectorStore
            self._store = MedicalCodeVectorStore()
            self._store.build_or_load(force_rebuild=True)
        else:
            self._vector_store()
        self._reference()
        self.assert_claim_assembly_data_readable()

    #: {code system -> the DECLARED release-source identity publishing its code table}.
    #: Identities, not paths and not medical codes: the reader resolves (and
    #: content-addresses) the file through the declaration, so the bytes that become the
    #: rich-descriptor table are the bytes the certificate names.  Resolving from CENTRAL
    #: CONFIG rather than an f-string was the earlier fix here (the ICD file is
    #: 'icd10cm_codes.json', so f"{system}_codes.json" built a non-existent path and
    #: silently lost every ICD descriptor tier); resolving from the DECLARATION is the same
    #: fix carried one step further, to the one registry the manifest is built from.
    _RICH_RECORD_SOURCES = {"icd10": "icd10_codes", "cpt": "cpt_codes",
                            "hcpcs": "hcpcs_codes"}

    def _rich_records(self, system: str) -> dict:
        """Raw {code: record} straight from the declared code table, which carries the FULL
        set of authoritative description tiers (the in-memory CodeReferenceDB keeps only
        long+short). Cached per system; {} if absent.

        This is a SECOND in-memory copy of the same three declared tables, parsed at a
        different moment from `CodeReferenceDB.load_all()`'s, so it is bound like every
        other parse -- and binding it is what makes a refresh landing BETWEEN the two loads
        a reported conflict instead of two editions silently answering one claim.
        (Codex F6-R5-B, post-fix review.)
        """
        if self._rich is None:
            self._rich = {}
        if system not in self._rich:
            table: dict[str, dict] = {}
            try:
                source_id = self._RICH_RECORD_SOURCES.get(system)
                if source_id is None:
                    raise AuthoritativeDataUnavailable(
                        f"no declared rich-record source for code system {system!r}")
                data, identity = declared_json_snapshot(source_id,
                                                        AuthoritativeDataUnavailable)
                rows = (data if isinstance(data, list)
                        else data.get("codes") or data.get(system)
                        or next((v for v in data.values() if isinstance(v, list)), []))
                for r in rows:
                    if isinstance(r, dict) and r.get("code"):
                        table[str(r["code"])] = r
                self._bound_sources.bind(identity)
            except Exception:
                # Descriptor TIERS only: without them `descriptions()` falls back to the
                # in-memory registry's long+short, which removes matching surface rather
                # than admitting anything, so this degrades instead of holding -- and binds
                # nothing, because no bytes were parsed to attest to.
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
                from .terminology import TerminologyIndex
                document, identity = declared_document_snapshot(
                    "snomed_crosswalk", AuthoritativeDataUnavailable)
                term_to_codes = document.get("terms", {})
                # Invert term->codes into code->terms and reuse the SAME robust
                # matcher as the ICD Index (exact / compound / token-set + plural)
                # so variant phrasings (possessive / plural / word order) hit.
                inv: dict[str, list[str]] = {}
                for term, codes in term_to_codes.items():
                    for c in codes:
                        inv.setdefault(c, []).append(term)
                self._snomed = TerminologyIndex(inv)
                self._bound_sources.bind(identity)
            except Exception:
                # A reviewed-OPTIONAL recall aid: absence removes candidates and can never
                # admit one, so it degrades rather than holding. Nothing is bound in that
                # case, which is correct -- no bytes were parsed to attest to.
                self._snomed = False
        if not self._snomed:
            return set()
        return {c for c in self._snomed.candidates(description)
                if self.leaf_codes(c, "icd10")}

    def _ensure_concept_relation_index(self):
        from . import terminology as _term
        if self._concept_relation_index is None:
            try:
                index, identity = _term.ConceptRelationIndex.load_snapshot()
                self._concept_relation_index = index
                self._concept_relation_identity = identity
                self._bound_sources.bind(identity)
            except Exception:
                # A reviewed-OPTIONAL concept graph: absence can only leave a relation
                # UNRESOLVED, never assert a wrong one, so it degrades rather than
                # holding. Nothing is bound in that case -- no bytes were parsed.
                self._concept_relation_index = False
        return self._concept_relation_index

    def concept_relation(self, term_a: str, term_b: str) -> str:
        """SAME / ancestor-descendant / DISJOINT / unresolved for two clinical terms,
        from the authoritative SNOMED CT Body Structure concept graph (issue #6
        F7-R3-C) -- never from lexical shape. Coreference axis comparison uses this
        to tell a synonym pair apart from a genuine distinction where lexical shape
        alone cannot. REVIEWED-OPTIONAL: absence returns `terminology.CONCEPT_
        UNRESOLVED`, which callers already treat as "cannot establish a relation" --
        the same conservative direction this axis already defaults to without
        concept data; see tools/build_snomed_concept_terms.py.
        """
        from . import terminology as _term
        index = self._ensure_concept_relation_index()
        if not index:
            return _term.CONCEPT_UNRESOLVED
        return index.relation(term_a, term_b)

    def concept_relation_detail(self, term_a: str, term_b: str) -> dict:
        """The full auditable basis (issue #6 F7-R3-C4) behind `concept_relation`'s
        verdict: each side's matched concept candidates, matching method and
        uniqueness, a confidence, and the exact concept-graph release identity the
        verdict was read from -- a JSON-safe record for binding into the audit trail
        and certificate, not just a bare string. `concept_relation`'s string verdict
        alone cannot answer "which concept, from which release, decided this" --
        this can.
        """
        from . import terminology as _term
        index = self._ensure_concept_relation_index()
        if not index:
            return {"verdict": _term.CONCEPT_UNRESOLVED, "source_identity": None}
        detail = index.relation_detail(term_a, term_b)
        return {
            "verdict": detail.verdict,
            "confidence": detail.confidence,
            "term_a": {"term": detail.match_a.term,
                      "candidates": list(detail.match_a.candidates),
                      "method": detail.match_a.method,
                      "unique": detail.match_a.unique},
            "term_b": {"term": detail.match_b.term,
                      "candidates": list(detail.match_b.candidates),
                      "method": detail.match_b.method,
                      "unique": detail.match_b.unique},
            "source_identity": dict(self._concept_relation_identity or {}),
        }

    #: Governed axes THIS source can route `concept_lookup` to a real concept graph
    #: for. Only "anatomy" has one today (the SNOMED CT Body Structure hierarchy) --
    #: adding a "procedure" axis needs a real licensed/verified procedure-terminology
    #: source, which does not exist yet: `cpt_synonyms.json`'s own provenance field
    #: says outright it is "llm-generated... RETRIEVAL AID ONLY -- NOT an authoritative
    #: source and never a coding decision input", and CPT's `concept_id` field is 1:1
    #: with each code (verified: 11,601 codes, 11,601 distinct concept_ids, zero
    #: shared) -- it names the code itself, not a reverse term->concept grouping the
    #: way SNOMED's does. "procedure" (below) is a genuinely weaker trust tier, not
    #: a governed concept graph -- see `_ensure_procedure_synonym_index`. Any OTHER
    #: axis honestly reports no governed source rather than being backed by data
    #: explicitly disclaimed as non-authoritative.
    _GOVERNED_LOOKUP_AXES = frozenset({"anatomy", "procedure"})

    _EMPTY_CONCEPT_LOOKUP = {"candidates": [], "method": "none", "unique": False,
                             "expansions": [], "source_identity": None}

    def _ensure_procedure_synonym_index(self):
        """Lazy-loaded, cached {normalized term -> set of CPT codes} /
        {code -> its own verified terms} pair from `cpt_verified_synonyms.json`
        (issue #6, compiled-semantic-layer plan item 3).

        REVIEWED-OPTIONAL, same discipline as `_ensure_concept_relation_index`:
        absence degrades this axis to "no expansion", never a held encounter --
        this is a retrieval-recall aid, not a claim-changing relation the way a
        governed anatomy SAME verdict is. Returns `False` (cached) when unavailable.
        """
        if getattr(self, "_proc_syn_index", None) is not None:
            return self._proc_syn_index
        try:
            doc, identity = declared_document_snapshot(
                "cpt_verified_synonyms", AuthoritativeDataUnavailable)
            terms_by_code = doc.get("terms") or {}
            if not isinstance(terms_by_code, dict) or not terms_by_code:
                raise AuthoritativeDataUnavailable("empty terms table")
            by_term: dict[str, set[str]] = {}
            code_terms: dict[str, tuple[str, ...]] = {}
            for code, synonyms in terms_by_code.items():
                norm_terms = tuple(sorted({str(t).strip().casefold()
                                          for t in (synonyms or []) if str(t).strip()}))
                if norm_terms:
                    code_terms[code] = norm_terms
                for t in norm_terms:
                    by_term.setdefault(t, set()).add(code)
            self._proc_syn_by_term = by_term
            self._proc_syn_by_code = code_terms
            self._proc_syn_identity = identity
            self._bound_sources.bind(identity)
            self._proc_syn_index = True
        except Exception:
            self._proc_syn_index = False
        return self._proc_syn_index

    def _concept_lookup_anatomy(self, term: str) -> dict:
        from . import terminology as _term
        index = self._ensure_concept_relation_index()
        if not index:
            return dict(self._EMPTY_CONCEPT_LOOKUP)
        match, expansions = index.normalize(term)
        return {
            "term": match.term,
            "candidates": list(match.candidates),
            "method": match.method,
            "unique": match.unique,
            "expansions": list(expansions),
            # Every candidate's own known governed terms, ambiguous or not -- so a
            # caller building a provider-facing message never has to show a bare
            # concept id (issue #6 F7-R3-C4, exact-SHA re-review, tenth pass: an
            # ambiguity hold routed to a human is only actionable if what it asks
            # about is nameable).
            "candidate_terms": {cid: list(index.terms_for_concept(cid))
                               for cid in match.candidates},
            "source_identity": dict(self._concept_relation_identity or {}),
        }

    def _concept_lookup_procedure(self, term: str) -> dict:
        """A DELIBERATELY WEAKER match tier than `_concept_lookup_anatomy`: this
        does not read a governed concept graph, only a table of LLM-generated
        candidate terms that were kept ONLY when they independently round-trip to
        their own code through the authoritative retrieval index
        (`tools/verify_cpt_synonyms.py`). That is corroboration BETWEEN two
        machine-learned artifacts (the generating LLM and the retrieval embedding
        model), not independent grounding in a licensed, human-curated standard
        the way SNOMED CT backs the anatomy axis -- so `method` names this
        explicitly (`"retrieval_consistency_validated"`, never a method string
        that could be mistaken for a governed-concept match) and a caller must
        never treat a "unique" match here with the same weight as a SAME verdict
        from a real concept graph.
        """
        if not self._ensure_procedure_synonym_index():
            return dict(self._EMPTY_CONCEPT_LOOKUP)
        norm = str(term).strip().casefold()
        candidates = tuple(sorted(self._proc_syn_by_term.get(norm, ())))
        unique = len(candidates) == 1
        expansions = (self._proc_syn_by_code.get(candidates[0], ())
                     if unique else ())
        expansions = tuple(e for e in expansions if e != norm)
        return {
            "term": term,
            "candidates": list(candidates),
            "method": "retrieval_consistency_validated" if candidates else "none",
            "unique": unique,
            "expansions": list(expansions),
            "candidate_terms": {cid: list(self._proc_syn_by_code.get(cid, ()))
                               for cid in candidates},
            "source_identity": dict(self._proc_syn_identity or {}),
        }

    def concept_lookup(self, axis: str, term: str) -> dict:
        """This ONE term's governed identity on `axis`, independent of any comparison
        (issue #6 F7-R3-C4/plan item 3): a single-entity normalization, so a
        documented value is resolved against the concept graph even when no second
        reading worded it differently to compare against. `expansions` is every OTHER
        term the SAME concept is known by (empty unless the match is unique) -- the
        canonical alternate phrasings retrieval may additionally query under. A
        JSON-safe record, same discipline as `concept_relation_detail`.

        Each axis's `method` values are its own trust tier -- "anatomy" can return a
        real governed-concept match; "procedure" can only ever return
        `"retrieval_consistency_validated"` (see `_concept_lookup_procedure`), never
        a method string that looks like the former. A caller must not conflate them.
        """
        if axis not in self._GOVERNED_LOOKUP_AXES:
            return dict(self._EMPTY_CONCEPT_LOOKUP)
        if axis == "anatomy":
            return self._concept_lookup_anatomy(term)
        return self._concept_lookup_procedure(term)

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
                from .terminology import TerminologyIndex
                document, identity = declared_document_snapshot(
                    "cpt_index_terms", AuthoritativeDataUnavailable)
                terms = document.get("terms", {})            # {code: [index phrases]}
                self._cptidx = TerminologyIndex(terms) if terms else False
                self._bound_sources.bind(identity)
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
            from .terminology import TerminologyIndex
            data, identity = declared_document_snapshot("hcpcs_drug_table",
                                                        AuthoritativeDataUnavailable)
            terms = data.get("terms", {})       # {code: [drug names]}
            self._drug = TerminologyIndex(terms) if terms else False
            self._drug_units = data.get("units", {}) or {}
            self._bound_sources.bind(identity)
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
        a stale mapping. Empty until mappings are promoted.

        Read and bound through the same exact-byte snapshot primitive as every other
        cached source on this adapter: the parsed copy answers every later call from
        memory, so the certificate must attest to the bytes that PARSE produced, not a
        re-hash of whatever is at the path when the manifest is built. Without this, a
        cached stale edition could keep contributing a candidate after the file was
        replaced, while the certificate names the replacement's bytes -- the same
        old-bytes-used/new-bytes-certified defect this binding closes for every other
        source on this adapter. (Codex F6-R5-B, learned-index gap.)"""
        if getattr(self, "_learned", None) is None:
            try:
                document, identity = declared_document_snapshot(
                    "learned_cpt_index", AuthoritativeDataUnavailable)
                self._learned = document.get("entries", {}) or {}
                self._bound_sources.bind(identity)
            except Exception:
                # REVIEWED-OPTIONAL (see docstring): absence/corruption degrades to
                # re-verification rather than holding, same as every other recall aid on
                # this adapter -- and nothing is bound, because no bytes were parsed to
                # attest to.
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

    def drug_dose_table_available(self) -> bool:
        """Whether the authoritative per-unit dose table LOADED — the difference between
        'this code is not a dosed drug' and 'nobody could tell us the per-unit dose'.
        Without it, `drug_unit` returning None is ambiguous and the unit computation
        silently falls back to a count, which changes billed units. The dose table is a
        reviewed-optional source precisely BECAUSE this distinction is available and the
        gate holds on it. (Codex F6-R5, round 5.)"""
        self._load_drug_table()
        return bool(self._drug_units)

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
                self._idx, identity = TerminologyIndex.load_snapshot()
                self._bound_sources.bind(identity)
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

    def _pfs_table(self) -> dict:
        """The whole CMS PFS indicator table, read FAIL-CLOSED and cached.

        Absence of this REQUIRED source is caught by the capability gate before any
        certificate exists. PRESENT-BUT-CORRUPT is not: the manifest hashes the bytes
        happily, and this read used to swallow the parse failure into `{}` -- at which point
        every code reports global period None and bilateral indicator None, so
        `apply_global_package` bundles nothing and the laterality modifiers change. That is
        corruption RELAXING the claim, so it raises. (Round 5, phase 4.)
        """
        if self._gp is None:
            self._gp, identity = declared_table_snapshot(
                "pfs_indicators", "codes", PfsIndicatorsUnavailable)
            self._bound_sources.bind(identity)
        return self._gp

    def _pfs(self, code: str) -> dict:
        """The CMS PFS indicator record for a code ({'global':…, 'bilat':…}). {} when the
        table has no record for this code -- a real answer from readable data, never the
        table failing to load (that raises `PfsIndicatorsUnavailable`)."""
        table = self._pfs_table()
        rec = table.get(code) or table.get(code.replace(".", ""))
        return rec if isinstance(rec, dict) else {}

    def assert_claim_assembly_data_readable(self) -> None:
        """Prove the authoritative tables CLAIM ASSEMBLY reads are readable, before it runs.

        The PFS indicator table is consumed while the claim is being BUILT (per-line
        bilateral modifiers, then the global surgical package) -- before the first gate
        executes -- so unlike coverage policy or the Tabular notes there is no gate
        downstream that could convert its unavailability into a hold. Asserting it here,
        once, is what makes the raise land on the pipeline's fail-closed boundary rather
        than escaping assembly as a crash.

        This is the SAME cached load assembly performs, not a second one that could
        disagree with it: once this returns, `_pfs` is served from `self._gp` and cannot
        raise later within the same encounter.
        """
        self._pfs_table()

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
        # A HCPCS code paid under OPPS (Social Security Act 1833(t)) -- e.g. a device
        # pass-through code -- is a facility / hospital-outpatient charge, NOT separately
        # reportable on the practitioner's PROFESSIONAL claim (paid under 1848). Read the
        # authoritative statute field; agnostic (no code named).
        if "1833(t)" in str(rec.get("statute", "")).lower():
            return Outcome.BLOCKED
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
            # None here means "no directional edit to apply", and that is SAFE only
            # because of what this function drives: a DEMOTION that can only ever remove
            # a line from the claim, plus an integral-bundling decision that can only
            # ever convert an escalation into an exclusion. Neither can bill anything.
            # The gate that decides RELEASABILITY reads the same authority through
            # `ncci_indicator` above, which reports AUTHORITY_UNAVAILABLE and holds the
            # encounter -- and directive section 6's certificate now BLOCKS on an
            # unusable compiled database independently of either. Raising instead would
            # abort assembly mid-claim and turn a structured hold into a crash artifact.
            # (Proved by `test_an_uncertifiable_compiled_database_stops_the_deployed_
            # entrypoint`, which asserts the hold through the real entrypoint.)
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
        be loaded' (assert nothing -> UNKNOWN).

        NON-EMPTY, not merely `isinstance(..., dict)`: an empty dict is exactly what a
        present-but-structurally-wrong MUE file produces, and it answers "no MUE is
        published for any code" -- so `mue_gate` reported NOT_APPLICABLE and every unit
        count went unchecked. Same class as the corruption fixes above; an empty table is
        indistinguishable from an unloaded one, so it must read as unloaded.
        (Round 5, phase 4.)"""
        try:
            return bool(getattr(self._reference(), "mue", None))
        except Exception:
            return False

    def data_fingerprint(self) -> dict[str, Any]:
        """A CONTENT-ADDRESSED fingerprint of the loaded authoritative editions, so the
        release certificate binds the exact BYTES the claim was coded against.

        Row counts are not an identity: two materially different source files of the same
        cardinality would fingerprint identically, leaving same-count source drift invisible
        to the attestation. The identity is therefore the capability manifest's per-source
        SHA-256 digests, sizes and release/effective windows, summarised into one
        `fingerprint_sha256`; counts and the deployment's optional Qdrant checksum remain as
        corroborating detail, never as the identity. (Codex F6-R5.)

        REQUIRED components fail loudly: a release must never be certified against unknown
        authoritative data. A swallowed failure that returned {} or partial counts is a
        silent-fallback certification hole.
        """
        fp: dict[str, Any] = {"fingerprint_version": "release-data-fingerprint-v3"}
        db = self._reference()                       # load failure raises -> pipeline holds
        # The identity of the database snapshot that ANSWERED this encounter, taken from
        # the querying object itself and carried forward into the certificate.  The
        # manifest below independently re-hashes whatever is at that path NOW; requiring
        # the two to agree is what turns "the database was replaced after the last query"
        # from a silent pass into a hold, with no later query obliged to notice it.
        # Raises if the database cannot be identified at all -- a release must never be
        # certified against data nobody can name. (Codex F6-R5-A.)
        fp["database_snapshot"] = snapshot = db.database_snapshot()
        # ... and the identities of every JSON source that was PARSED INTO MEMORY and has
        # answered from that parsed copy ever since: the code/limit tables (bound by the
        # object that loaded them) and whichever policy/rule documents this encounter had to
        # read (bound here). Exactly the same defect as the compiled database's, for every
        # other claim-affecting source: a file replaced after it was loaded leaves the
        # in-memory data speaking for the OLD bytes while the manifest below hashes the NEW
        # ones. Requiring the two to agree makes that a hold. (Codex F6-R5-B.)
        bound_sources = SourceSnapshotSet()
        bound_sources.merge(db.source_snapshots())
        bound_sources.merge(self._bound_sources)
        # The reviewed CONTROL CONFIGURATION is cached at MODULE level (one edition per
        # process, by construction), so its binding is read from the modules that hold it
        # rather than from an object. Bound only WHEN it was consulted: an encounter that
        # never reached the necessity gate legitimately has no control binding, and
        # inventing one would attest to bytes that decided nothing.
        from .gates import necessity_control_snapshot
        from .provenance import relation_grammar_snapshot
        for control in (necessity_control_snapshot(), relation_grammar_snapshot()):
            if control:
                bound_sources.bind(control)
        fp["source_snapshots"] = bound_sources.identities
        fp["counts"] = {s: len(getattr(db, s, {}) or {})
                        for s in ("icd10", "cpt", "hcpcs")}
        if not all(fp["counts"].get(s) for s in ("icd10", "cpt", "hcpcs")):
            raise RuntimeError(
                "authoritative code counts unavailable/empty; cannot fingerprint release")
        try:
            from app.release.source_manifest import declared_source_path
            chk = declared_source_path("retrieval_index_checksum")
            if chk.exists():
                fp["codes_checksum"] = chk.read_text().strip()[:64]
        except Exception:
            pass                                     # codes_checksum is corroborating only
        from .capability import build_manifest, fingerprint_digest  # raises if unavailable
        manifest = build_manifest(bound_database=snapshot, bound_sources=bound_sources)
        fp["source_manifest"] = manifest
        fp["fingerprint_sha256"] = fingerprint_digest(fp["counts"], manifest)
        return fp

    def _excludes1_map(self) -> dict[str, set[str]]:
        """{undotted category key -> set of undotted Excludes1 target prefixes} from
        the ICD-10-CM Tabular instructional notes. Excludes1 = 'NOT CODED HERE': the two
        conditions are mutually exclusive and should not be reported together (unless
        genuinely unrelated — a human judgement). Cached.

        FAIL-CLOSED: unreadable, unparseable or noteless instructional data RAISES
        `InstructionalNotesUnavailable`. It used to degrade to {}, and {} does not merely
        lose a lookup — it makes every diagnosis pair look Excludes1-clean, turning
        `icd_excludes_gate` from a compliance check into a silent PASS. A gate that cannot
        read its authority must hold, not clear the claim. (Round 5, phase 4.)
        """
        cache = getattr(self, "_excl1", None)
        if cache is not None:
            return cache
        entries, identity = declared_table_snapshot("instructional_notes", "codes",
                                                    InstructionalNotesUnavailable)
        cache = {}
        for key, rec in entries.items():
            if not isinstance(rec, dict):
                continue
            refs = rec.get("excludes1_code_refs") or []
            pref = {str(r).replace(".", "").upper() for r in refs if r}
            if pref:
                cache[str(key).replace(".", "").upper()] = pref
        if not cache:
            # The document parsed and carries code entries, but not one of them declares an
            # Excludes1 reference -- a schema/field drift in the extract, not a real edition
            # of the ICD-10-CM Tabular. Indistinguishable at the gate from "no conflicts".
            raise InstructionalNotesUnavailable(
                f"authoritative instructional_notes at {_source_path('instructional_notes')} "
                f"declares no Excludes1 reference for any code")
        self._bound_sources.bind(identity)
        self._excl1 = cache
        return cache

    def _coverage_map(self) -> dict[str, set[str]]:
        """{governed CPT/HCPCS code -> set of qualifying ICD-10 codes} from the CMS
        LCD/Article coverage data (governed_cpts x qualifying_dx) — the authoritative
        dx->procedure MEDICAL-NECESSITY linkage. Cached.

        FAIL-CLOSED: unreadable, unparseable or structurally empty coverage data RAISES.
        It used to degrade to {}, which `qualifying_dx_for` reports as "this service is
        governed by no policy" — moving every service onto the LESS restrictive
        ungoverned path exactly when the coverage authority was unavailable. The caller
        (the necessity gate) converts the raise into a HOLD. (Codex F6-R5, round 5.)
        """
        cache = getattr(self, "_cov", None)
        if cache is not None:
            return cache
        d, identity = declared_document_snapshot("coverage_policy",
                                                 CoverageDataUnavailable)
        path = identity["path"]                  # the bytes actually parsed, not a re-read
        cache = {}
        try:
            for row in (d.get("lcd") or []) + (d.get("article") or []):
                qd = {str(x).replace(".", "").upper()
                      for x in (row.get("qualifying_dx") or [])}
                if not qd:
                    continue
                for c in (row.get("governed_cpts") or []):
                    cache.setdefault(str(c).replace(".", "").upper(), set()).update(qd)
        except Exception as exc:
            raise CoverageDataUnavailable(
                f"authoritative coverage policy unreadable at {path}: {exc}") from exc
        if not cache:
            raise CoverageDataUnavailable(
                f"authoritative coverage policy at {path} declares no governed "
                f"procedure/qualifying-diagnosis linkage")
        self._bound_sources.bind(identity)
        self._cov = cache
        return cache

    def qualifying_dx_for(self, code: str, system: str = "cpt") -> set[str] | None:
        """Authoritative qualifying (medically-necessary) ICD-10 codes for a procedure
        per CMS LCD/Article coverage — the dx->procedure necessity linkage, undotted.
        None when the procedure is governed by NO coverage policy (its necessity
        cannot be confirmed from this source -> callers fail closed); a set when it is
        governed. Raises `CoverageDataUnavailable` when the coverage authority itself
        could not be read -- "unavailable" must never be reported as "ungoverned"."""
        cov = self._coverage_map()
        return cov.get(str(code).replace(".", "").upper())

    def excludes1_refs(self, code: str, system: str) -> set[str]:
        """The Excludes1 target code-prefixes that apply to a diagnosis — gathered
        from the code AND its ancestor categories (an Excludes1 at a 3-character
        category governs every child code under it). A billed diagnosis whose code
        starts with any
        returned prefix is in an Excludes1 relationship with `code`. ICD-10-CM only.

        Raises `InstructionalNotesUnavailable` when the Tabular notes cannot be read: an
        empty result must mean "this diagnosis carries no Excludes1 note", never "nobody
        could tell us" -- the caller (`gates.icd_excludes_gate`) holds on the second."""
        if system != "icd10":
            return set()
        table = self._excludes1_map()
        undot = str(code).replace(".", "").upper()
        out: set[str] = set()
        for length in range(len(undot), 2, -1):        # code, then each ancestor category
            out |= table.get(undot[:length], set())
        return out

    #: The ICD-10-CM Tabular relationship types this reads, alongside Excludes1
    #: (already its own dedicated map/gate above -- unchanged). Same source file,
    #: same `*_code_refs` shape (issue #6, compiled-semantic-layer plan item 1).
    _RELATIONSHIP_FIELDS = ("excludes2", "codeFirst", "useAdditionalCode", "codeAlso")

    def _relationship_map(self) -> dict[str, dict[str, set[str]]]:
        """{undotted category key -> {relationship type -> target code-prefixes}} for
        every Tabular relationship type OTHER than Excludes1 (which keeps its own
        cache above; both read the same underlying table). Cached.

        FAIL-CLOSED for the same reason `_excludes1_map` is: a table that parses but
        declares no relationship at all is a schema/field drift, not a real Tabular
        edition, and must not be read as "these codes carry no notes".
        """
        cache = getattr(self, "_relmap", None)
        if cache is not None:
            return cache
        entries, identity = declared_table_snapshot("instructional_notes", "codes",
                                                    InstructionalNotesUnavailable)
        cache = {}
        any_ref = False
        for key, rec in entries.items():
            if not isinstance(rec, dict):
                continue
            per_code: dict[str, set[str]] = {}
            for field in self._RELATIONSHIP_FIELDS:
                refs = rec.get(f"{field}_code_refs") or []
                pref = {str(r).replace(".", "").upper() for r in refs if r}
                if pref:
                    per_code[field] = pref
                    any_ref = True
            if per_code:
                cache[str(key).replace(".", "").upper()] = per_code
        if not any_ref:
            raise InstructionalNotesUnavailable(
                f"authoritative instructional_notes at {_source_path('instructional_notes')} "
                f"declares no Excludes2/codeFirst/useAdditionalCode/codeAlso reference "
                f"for any code")
        self._bound_sources.bind(identity)
        self._relmap = cache
        return cache

    def component_relationships(self, code: str, system: str) -> dict[str, set[str]]:
        """{relationship type -> target code-prefixes} for a diagnosis, gathered from
        the code AND its ancestor categories exactly like `excludes1_refs` (a note at
        a 3-character category governs every child code under it). Relationship types:
        `excludes1` (mutually exclusive -- never code both), `excludes2` (not included
        here, but may be coded together when the record supports both), `codeFirst`
        (a sequencing requirement), `useAdditionalCode` (a required companion code
        when the documentation supports it), `codeAlso` (a paired code, either order).
        ICD-10-CM only; {} for any other system.

        Raises `InstructionalNotesUnavailable` when the Tabular notes cannot be read,
        matching `excludes1_refs` -- an empty result must mean "this diagnosis carries
        no relationship note", never "nobody could tell us".
        """
        if system != "icd10":
            return {}
        undot = str(code).replace(".", "").upper()
        excl1 = self.excludes1_refs(code, system)
        out: dict[str, set[str]] = {"excludes1": excl1} if excl1 else {}
        table = self._relationship_map()
        for length in range(len(undot), 2, -1):
            for field, refs in table.get(undot[:length], {}).items():
                out.setdefault(field, set()).update(refs)
        return out

    # -- semantic classification (issue #6, compiled-semantic-layer plan item 1) --
    def _coding_semantics(self) -> dict:
        """The `code_classes` classification config: {class name -> matching rule}.
        Reviewed, versioned, already-ingested data (`data/codes/coding_semantics.json`)
        that had no `claude_coder`-reachable consumer before this. Cached."""
        cache = getattr(self, "_semrules", None)
        if cache is not None:
            return cache
        doc, identity = declared_document_snapshot("coding_semantics",
                                                    SemanticClassUnavailable)
        classes = doc.get("code_classes")
        if not isinstance(classes, dict) or not classes:
            raise SemanticClassUnavailable(
                f"authoritative coding_semantics at {_source_path('coding_semantics')} "
                f"declares no non-empty code_classes table")
        self._bound_sources.bind(identity)
        self._semrules = classes
        return classes

    def _icd_chapter_ranges(self) -> list[tuple[int, str, str]]:
        """[(chapter id, undotted start code, undotted end code)] from the CDC/NCHS
        chapter boundaries. Cached."""
        cache = getattr(self, "_icdchap", None)
        if cache is not None:
            return cache
        doc, identity = declared_document_snapshot("icd10cm_chapters",
                                                    SemanticClassUnavailable)
        chapters = doc.get("chapters")
        if not isinstance(chapters, list) or not chapters:
            raise SemanticClassUnavailable(
                f"authoritative icd10cm_chapters at {_source_path('icd10cm_chapters')} "
                f"declares no non-empty chapters table")
        cache = [(int(c["id"]), str(c["start"]).replace(".", "").upper(),
                 str(c["end"]).replace(".", "").upper())
                for c in chapters if isinstance(c, dict) and "id" in c]
        self._bound_sources.bind(identity)
        self._icdchap = cache
        return cache

    def _cpt_categories(self) -> dict[str, frozenset[str]]:
        """{category name -> frozenset of member codes} from the licensed CPT
        category membership. Cached."""
        cache = getattr(self, "_cptcat", None)
        if cache is not None:
            return cache
        doc, identity = declared_document_snapshot("cpt_categories",
                                                    SemanticClassUnavailable)
        cats = doc.get("categories")
        if not isinstance(cats, dict) or not cats:
            raise SemanticClassUnavailable(
                f"authoritative cpt_categories at {_source_path('cpt_categories')} "
                f"declares no non-empty categories table")
        cache = {str(name): frozenset(str(c) for c in (members or []))
                 for name, members in cats.items()}
        self._bound_sources.bind(identity)
        self._cptcat = cache
        return cache

    def semantic_class(self, code: str, system: str) -> str | None:
        """The `coding_semantics.json` class this code matches, or None when no rule
        matches -- an honest "unclassified", never a guess (issue #6 item 1: "unknown
        semantics remain unknown").

        Each class declares exactly one authoritative-data rule:
          - `descriptor_any`: the code's own long/short description contains one of
            the listed phrases (case-insensitive substring).
          - `global_days_kind: "numeric"`: the CMS global-surgical-package indicator
            is a numeric day count (000/010/090), not XXX/YYY/ZZZ/MMM.
          - `icd_chapter_ids`: the code's 3-character category falls inside one of
            the listed CDC/NCHS ICD-10-CM chapters.
          - `cpt_category`: the code is a member of the licensed CPT category.
          - `pfs_status_any`: NOT YET SATISFIABLE here -- `claude_coder`'s own PFS
            table (`_pfs_table`, built by `tools/build_global_period.py`) does not
            carry the CMS status-indicator field this rule reads, only global/bilat.
            A class whose ONLY rule is `pfs_status_any` is honestly unclassifiable
            from this source today rather than approximated; the first match wins
            when a code matches more than one class, so which class table order
            settles it is a config decision, not a code one.
        """
        rules = self._coding_semantics()
        undot = str(code).replace(".", "").upper()
        rec = self.lookup(code, system) or {}
        descriptor = str(rec.get("long_description") or rec.get("description")
                         or rec.get("short_description") or "").casefold()
        for name, rule in rules.items():
            if not isinstance(rule, dict):
                continue
            terms = rule.get("descriptor_any")
            if terms and descriptor and any(str(t).casefold() in descriptor for t in terms):
                return name
            if rule.get("global_days_kind") == "numeric" and system in ("cpt", "hcpcs"):
                gp = self.global_period(code)
                if gp and gp.isdigit():
                    return name
            chapter_ids = rule.get("icd_chapter_ids")
            if chapter_ids and system == "icd10":
                # Compare the code's own 3-character CATEGORY, never the full code,
                # against the (always 3-character) chapter boundaries -- a full code
                # longer than its category (e.g. "B9999") sorts lexicographically
                # AFTER a 3-character upper bound ("B99") even though it is a real
                # member of that chapter, since Python string comparison treats a
                # proper prefix as "less than" the longer string it prefixes.
                category = undot[:3]
                for cid, start, end in self._icd_chapter_ranges():
                    if cid in chapter_ids and start <= category <= end:
                        return name
            category = rule.get("cpt_category")
            if category and system in ("cpt", "hcpcs"):
                if code in self._cpt_categories().get(str(category), frozenset()):
                    return name
        return None


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
                 coverage: dict[str, set] | None = None,
                 concept_relation: dict[tuple[str, str], str] | None = None,
                 concept_lookup: dict[str, dict] | None = None,
                 component_relationships: dict[str, dict[str, set]] | None = None,
                 semantic_class: dict[str, str] | None = None) -> None:
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
        self._concept_relation_map = concept_relation or {}
        self._concept_lookup_map = concept_lookup or {}
        self._component_rel_data = {
            str(k).replace(".", "").upper():
                {rel: {str(r).replace(".", "").upper() for r in refs}
                 for rel, refs in (v or {}).items()}
            for k, v in (component_relationships or {}).items()}
        self._semantic_class_data = {str(k): str(v)
                                     for k, v in (semantic_class or {}).items()}

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

    def drug_dose_table_available(self) -> bool:
        return bool(self._drug_units)

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

    def assert_claim_assembly_data_readable(self) -> None:
        """No-op: the mock's tables are the in-memory ones it was constructed with, so
        there is no file whose readability could be asserted. Implemented (rather than
        left off) so the mock satisfies the full `CodeSource` contract."""
        return None

    def data_fingerprint(self):
        """A schema-COMPLETE synthetic fingerprint: the mock must satisfy exactly the same
        content-addressed contract as the real source, so tests cannot pass against a shape
        production would reject. (Codex F6-R5.)

        "Complete" now means the COMPLETE required-source set: the identities, roles and
        release-metadata expectations are read from the same versioned registry
        declaration the real manifest is built from, so a mock can never satisfy the
        validator with a set of synthetic sources production would reject -- and a change
        to the required set fails the mock too, instead of leaving tests green against a
        set that no longer exists."""
        from app.release.source_manifest import (
            COMPLIANCE_DATABASE_SOURCE_ID, REQUIRED_SOURCE_SCHEMA_VERSION,
            SNAPSHOT_BOUND_SOURCES, required_release_sources)
        from .capability import MANIFEST_VERSION, fingerprint_digest, manifest_digest
        sources = []
        for source_id, spec in required_release_sources().items():
            release = ({"effective_from": "2026-01-01", "effective_to": "9999-12-31",
                        "version": "mock"}
                       if spec["release_metadata_required"]
                       else {"effective_from": "", "effective_to": "", "version": ""})
            sources.append({
                "source": source_id, "source_id": source_id, "required": True,
                "present": True, "status": "loaded", "role": spec["role"],
                "path": f"/mock/{source_id}.json", "bytes": 1,
                "sha256": "sha256:" + hashlib.sha256(source_id.encode()).hexdigest(),
                "release": release})
        manifest = {"manifest_version": MANIFEST_VERSION,
                    "required_sources_schema": REQUIRED_SOURCE_SCHEMA_VERSION,
                    "sources": sources, "missing_required": [], "degraded_optional": [],
                    "integrity_errors": [], "status": "OK",
                    "manifest_sha256": manifest_digest(sources)}
        counts = {"icd10": 1, "cpt": 1, "hcpcs": 1}
        # The compiled database's query-time binding, mirrored from the mock's own record
        # for it: the real producer propagates the identity of the snapshot that answered
        # the queries and the validator requires the manifest to agree with it, so a mock
        # without one would pass a shape production rejects. (Codex F6-R5-A.)
        database = next(s for s in sources
                        if s["source_id"] == COMPLIANCE_DATABASE_SOURCE_ID)
        # The in-memory sources' load-time bindings, mirrored from the mock's own records
        # for them, for the same reason as the database's: the real producer propagates the
        # identity captured when each source was PARSED and the validator requires the
        # manifest to agree with it, so a mock without them would pass a shape production
        # rejects. Read from the same declaration, so a change to the bound set fails the
        # mock too. (Codex F6-R5-B.)
        by_id = {s["source_id"]: s for s in sources}
        snapshots = {source_id: {"source_id": source_id, "path": by_id[source_id]["path"],
                                 "sha256": by_id[source_id]["sha256"],
                                 "size": by_id[source_id]["bytes"]}
                     for source_id in SNAPSHOT_BOUND_SOURCES}
        return {"source": "mock", "fingerprint_version": "release-data-fingerprint-v3",
                "counts": counts, "source_manifest": manifest,
                "database_snapshot": {"source_id": COMPLIANCE_DATABASE_SOURCE_ID,
                                      "path": database["path"],
                                      "sha256": database["sha256"],
                                      "size": database["bytes"]},
                "source_snapshots": snapshots,
                "fingerprint_sha256": fingerprint_digest(counts, manifest)}

    def qualifying_dx_for(self, code, system="cpt"):
        c = str(code).replace(".", "").upper()
        return set(self._coverage[c]) if c in self._coverage else None

    def concept_relation(self, term_a, term_b):
        from .terminology import CONCEPT_UNRESOLVED
        if (term_a, term_b) in self._concept_relation_map:
            return self._concept_relation_map[(term_a, term_b)]
        if (term_b, term_a) in self._concept_relation_map:
            return self._concept_relation_map[(term_b, term_a)]
        return CONCEPT_UNRESOLVED

    def concept_relation_detail(self, term_a, term_b):
        return {"verdict": self.concept_relation(term_a, term_b),
               "confidence": 1.0, "term_a": {"term": term_a}, "term_b": {"term": term_b},
               "source_identity": {"source_id": "mock_concept_relation"}}

    def concept_lookup(self, axis, term):
        # Test fixtures configure ONE flat term->record table (the one governed axis
        # this suite exercises, anatomy) -- gating on axis here mirrors what the real
        # `AuthoritativeSource` actually does (only "anatomy" has a governed source
        # today), not a shim papering over a shape mismatch.
        if axis == "anatomy" and term in self._concept_lookup_map:
            return dict(self._concept_lookup_map[term])
        return {"term": term, "candidates": [], "method": "none", "unique": False,
               "expansions": [], "source_identity": None}

    def excludes1_refs(self, code, system):
        if system != "icd10" or not self._excl1_data:
            return set()
        undot = str(code).replace(".", "").upper()
        out = set()
        for length in range(len(undot), 2, -1):
            out |= self._excl1_data.get(undot[:length], set())
        return out

    def component_relationships(self, code, system):
        if system != "icd10":
            return {}
        excl1 = self.excludes1_refs(code, system)
        out = {"excludes1": excl1} if excl1 else {}
        undot = str(code).replace(".", "").upper()
        for length in range(len(undot), 2, -1):
            for rel, refs in self._component_rel_data.get(undot[:length], {}).items():
                out.setdefault(rel, set()).update(refs)
        return out

    def semantic_class(self, code, system):
        return self._semantic_class_data.get(str(code))
