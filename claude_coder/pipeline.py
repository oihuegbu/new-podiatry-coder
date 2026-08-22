"""End-to-end orchestration.

    note ─► extract facts (CLU, code-free)
         ─► resolve each fact -> code (deterministic, from authoritative data)
         ─► arbitrate only the ambiguous ones (bounded LLM over retrieved codes)
         ─► positive release gates (fail-closed, data-backed)
         ─► autonomy controller (AUTO_READY | REVIEW_REQUIRED | BLOCKED)

Every step is pluggable: pass a `MockSource` and stub LLMs to run the whole
pipeline deterministically in a test, or the real `AuthoritativeSource` in
production. The result carries its own audit trail (evidence -> fact -> code ->
method -> authority) so any decision can be explained.
"""
from __future__ import annotations

import logging
import re

from . import arbitration, certificate, em, extraction, gates, ontology, resolution
from .arbitration import LLMFn
from .autonomy import decide
from .data_access import AuthoritativeSource, CodeSource
from .models import CodingResult, ResolutionMethod, ResolvedLine


logger = logging.getLogger(__name__)

_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")

# The second reading is its OWN assertion origin: a distinct run id keeps it from being
# folded into the primary reading's origin, so a relation both readings assert is
# recorded as multiply-asserted rather than as one call repeating itself.
_SECOND_READING_RUN_ID = "second-reading"


def _fingerprint_certifiable(fp) -> bool:
    """A release may be certified only against a fingerprint that actually IDENTIFIES the
    authoritative data — not merely asserts that some data was there.

    The COMPLETE schema is validated (Codex F6-R5):
      - non-empty per-system code counts;
      - a versioned source manifest whose status is OK, with `missing_required == []` and
        `integrity_errors == []` (both explicitly present as lists, not merely falsy);
      - the manifest accounts for EXACTLY the required source identities and roles declared
        by the versioned registry (`app.release.source_manifest.required_release_sources`) —
        no omitted required source, no duplicate identity, no unregistered required source,
        no role that disagrees with the declaration.  Requiring merely that SOME source
        marks itself `required` let one synthetic source certify a release while every real
        claim-affecting source (edit policy, unit limits, code sets, global periods) was
        absent from a self-consistent manifest;
      - every required source present, non-empty, and carrying a `sha256:<64 hex>` content
        digest — so two different files with equal row counts cannot share an identity;
      - a non-empty upstream effective/edition window on every required source the authority
        publishes one for; sources it publishes none for are reviewed exemptions recorded in
        the declaration, not blanks silently accepted;
      - a `database_snapshot` binding: the content identity the compiled database was
        bound to when it ANSWERED this encounter, which the manifest's own record for that
        database must equal. A digest re-read from the path at certification time is a
        different fact about a different moment, and on its own would certify whatever
        file happens to be there rather than the one the decisions came from;
      - `source_snapshots`: the same binding for every OTHER claim-affecting source that is
        PARSED INTO MEMORY and answered from that copy afterwards -- the ICD/CPT/HCPCS and
        MUE tables and the SNOMED control table always, plus whichever policy/rule documents
        this encounter had to read. Each must be present for the sources the declaration
        says are eagerly loaded, and each must equal the manifest's own record for it;
      - a `manifest_sha256` that still matches the recorded sources, and a top-level
        `fingerprint_sha256` RECOMPUTED from the canonical counts + manifest and compared —
        pattern-matching its shape accepted an arbitrary all-zero / all-`f` / plausible-but-
        wrong digest as the identity of the data, and accepted a reordered or partially
        copied manifest.
    A status-only manifest (`{"status": "OK"}`) is therefore NOT certifiable.

    TOTAL by construction: this validator is called outside the fingerprint try/except, so it
    must never raise. An unexpected shape (or an unavailable digest helper) resolves to "not
    certifiable", which routes the structured retryable hold below -- not an exception that
    escapes `code_encounter` and loses the hold entirely.
    """
    try:
        return _fingerprint_schema_ok(fp)
    except Exception:
        return False


def _fingerprint_schema_ok(fp) -> bool:
    from app.release.source_manifest import (
        COMPLIANCE_DATABASE_SOURCE_ID, REQUIRED_SOURCE_SCHEMA_VERSION,
        SNAPSHOT_BOUND_SOURCES, is_content_digest, release_window_populated,
        required_release_sources)

    if not isinstance(fp, dict) or not fp:
        return False
    counts = fp.get("counts")
    if not isinstance(counts, dict) or not all(counts.get(s) for s in ("icd10", "cpt", "hcpcs")):
        return False
    manifest = fp.get("source_manifest")
    if not isinstance(manifest, dict) or manifest.get("status") != "OK":
        return False
    if not str(manifest.get("manifest_version") or "").strip():
        return False
    # Which required-source DEFINITION the manifest was built against. A manifest that
    # does not name one, or names another, is not comparable to the current declaration.
    if manifest.get("required_sources_schema") != REQUIRED_SOURCE_SCHEMA_VERSION:
        return False
    for key in ("missing_required", "integrity_errors"):
        value = manifest.get(key)
        if not isinstance(value, list) or value:
            return False
    sources = manifest.get("sources")
    if not isinstance(sources, list) or not sources:
        return False

    # Identity index: a duplicated source_id makes "which bytes" ambiguous, so it is
    # rejected before any per-source check rather than silently last-one-wins.
    declared: dict[str, dict] = {}
    for s in sources:
        if not isinstance(s, dict):
            return False
        source_id = str(s.get("source_id") or "").strip()
        if not source_id or source_id in declared:
            return False
        declared[source_id] = s

    # The manifest must account for EXACTLY the declared required set -- neither an
    # omitted required source (a partial/failed manifest certifying anyway) nor an
    # unregistered one asserting itself required (a synthetic source standing in for
    # the real claim-affecting inputs).
    expected = required_release_sources()
    if {sid for sid, s in declared.items() if s.get("required")} != set(expected):
        return False
    for source_id, spec in expected.items():
        s = declared[source_id]
        if str(s.get("role") or "") != spec["role"]:
            return False
        if not s.get("present") or not isinstance(s.get("bytes"), int) or s["bytes"] <= 0:
            return False
        if not _SHA256_RE.fullmatch(str(s.get("sha256") or "")):
            return False
        if (spec["release_metadata_required"]
                and not release_window_populated(s.get("release"))):
            return False

    # The compiled database the certificate attests to must be the SNAPSHOT that answered
    # this encounter -- the identity the querying object bound and propagated here -- not
    # an independently timed re-hash of whatever is at that path now. A fingerprint with no
    # such binding, or one its own manifest record disagrees with, is not certifiable.
    # (Codex F6-R5-A.)
    snapshot = fp.get("database_snapshot")
    if not isinstance(snapshot, dict) or not is_content_digest(snapshot.get("sha256")):
        return False
    bound_record = declared.get(COMPLIANCE_DATABASE_SOURCE_ID)
    if not isinstance(bound_record, dict):
        return False
    if str(bound_record.get("sha256") or "") != str(snapshot.get("sha256") or ""):
        return False
    if bound_record.get("bytes") != snapshot.get("size"):
        return False

    # The same requirement for every source held IN MEMORY: the certificate must name the
    # bytes each table was parsed from, not whatever is at its path now. Absence of a
    # binding for an eagerly-loaded source is itself disqualifying -- "nobody identified the
    # bytes that became the in-memory table" is not a clean result, and accepting it would
    # let a loader that stops binding certify exactly as before. (Codex F6-R5-B.)
    snapshots = fp.get("source_snapshots")
    if not isinstance(snapshots, dict) or not snapshots:
        return False
    if any(source_id not in snapshots for source_id in SNAPSHOT_BOUND_SOURCES):
        return False
    for source_id, identity in snapshots.items():
        if not isinstance(identity, dict) or not is_content_digest(identity.get("sha256")):
            return False
        record = declared.get(str(source_id))
        if not isinstance(record, dict):
            return False
        if str(record.get("sha256") or "") != str(identity.get("sha256") or ""):
            return False
        if record.get("bytes") != identity.get("size"):
            return False

    from .capability import fingerprint_digest, manifest_digest
    if manifest.get("manifest_sha256") != manifest_digest(sources):
        return False
    # Recompute the aggregate identity: the declared fingerprint must BE the digest of
    # the counts + manifest it claims to identify, not merely digest-shaped.
    if str(fp.get("fingerprint_sha256") or "") != fingerprint_digest(counts, manifest):
        return False
    return True


def code_encounter(
    encounter_id: str,
    note_text: str,
    date_of_service: str | None,
    source: CodeSource | None = None,
    extract_llm: LLMFn | None = None,
    extract_llm_b: LLMFn | None = None,
    arbitrate_llm: LLMFn | None = None,
    verify_llm: LLMFn | None = None,
    corroborate_llm: LLMFn | None = None,
    modifier_engine: "ModifierEngine | None" = None,
    billing_context: dict | None = None,
    audit_repository=None,
    document_version: str | None = None,
    model_profiles: dict | None = None,
    source_evidence=None,
    source_reader=None,
    service_date_binding: dict | None = None,
) -> CodingResult:
    """`source_evidence` is a `contracts.source_evidence.SourceEvidenceDocument`: the
    ORIGINAL document as read by more than one channel. Without it the note text is one
    model's transcription with nothing to check it against, and
    `gates.source_evidence_gate` holds every encounter that came from a document
    (issue #6 F6-R6-A). `source_reader` is the OPTIONAL, lazily-invoked second model
    read used only for pages no deterministic channel could read — see the escalation
    below, which is the cost control that keeps this from doubling the price of notes
    whose text layer already covers them."""
    from .models import GateResult, Outcome
    from .modifiers import ModifierEngine
    source = source or AuthoritativeSource()

    # ---- Trust boundary for a CALLER-SUPPLIED source_evidence document ---------------
    # `source_evidence` is directly caller-suppliable, so a caller can hand in a
    # fully-formed `SourceEvidenceDocument` whose reads never passed through
    # `with_channel`'s trust-boundary revalidation at all (issue #6 F7-R3-A, exact-SHA
    # re-review, seventh pass). Revalidating unconditionally here -- not only for reads
    # added later -- closes that gap: a genuinely valid document reconstructs
    # identically, so this costs nothing for the ordinary case (a document from the
    # trusted compiler) and refuses a forged one before a single quotation is
    # reconciled against it.
    if source_evidence is not None:
        try:
            source_evidence = source_evidence.revalidated()
        except Exception as exc:
            return _system_hold_result(encounter_id, date_of_service,
                                       "source_evidence_integrity", exc, source)

    # ---- Fail-closed boundary for the data CLAIM ASSEMBLY reads ----------------------
    # `pfs_indicators` (global period + bilateral indicator) and `modifier_definitions` are
    # REQUIRED release sources consumed while the claim is being BUILT -- per-line
    # modifiers, then the global surgical package -- which is BEFORE the first gate runs.
    # So, unlike coverage policy or the Tabular notes, no gate downstream can convert their
    # unavailability into a hold; a present-but-corrupt file used to degrade to an empty
    # table right here, and an empty table is the PERMISSIVE answer for both (nothing has a
    # global period, no modifier applies). They are therefore proven readable ONCE, up
    # front -- before any LLM call is spent -- and any typed unavailability becomes the same
    # retryable system hold every other enforced boundary in this function produces.
    # (Round 5, phase 4.)
    from .data_access import AuthoritativeDataUnavailable
    try:
        probe = getattr(source, "assert_claim_assembly_data_readable", None)
        if callable(probe):
            probe()
        # A caller-supplied engine carries its own reviewed definitions; only the default
        # engine reads the authoritative file, and that read is the assertion.
        modifier_engine = modifier_engine or ModifierEngine()
    except AuthoritativeDataUnavailable as exc:
        return _system_hold_result(encounter_id, date_of_service,
                                   "authoritative_data_integrity", exc, source)

    # Propose-then-verify is enabled in real mode (no stubbed LLMs). It grounds every
    # procedure code in an authoritative descriptor the documentation entails — the
    # license-clean substitute for the CPT Index. In real mode it is also corroborated
    # by an INDEPENDENT second model, so a procedure bills only when two independent
    # judgements agree. Tests pass stub LLMs and leave these None -> deterministic
    # path unchanged, no corroboration.
    if verify_llm is None and arbitrate_llm is None:
        from .verify import default_corroborate_llm, default_verify_llm
        verify_llm = default_verify_llm
        if corroborate_llm is None:
            corroborate_llm = default_corroborate_llm
    # Two independent readings of the note (directive section 3). Enabled in real mode
    # for the same reason and by the same rule as corroboration above: a caller that
    # supplies its own extractor (every test) opts out, so the deterministic path is
    # unchanged. `config.GRAPH_CONSENSUS=0` disables the control explicitly and the
    # audit record then says only one reading was taken.
    # Whether the second reading is this pipeline's OWN independence control or a
    # caller-supplied disagreement detector. Only the former promises independence, so
    # only the former fails closed when it turns out not to be independent (F7-R5).
    enforce_second_reading_independence = False
    if extract_llm is None and extract_llm_b is None:
        from app.core import config as _config
        if getattr(_config, "GRAPH_CONSENSUS", True):
            extract_llm_b = extraction.default_second_extract_llm
            enforce_second_reading_independence = True
    profiles = model_profiles or _model_profile_identity(
        extract_llm, verify_llm, corroborate_llm, extract_llm_b)

    from .models import FactKind
    # Enforced evidence/service graph. Any extraction, anchoring, graph-integrity,
    # eligibility, or durable-audit failure stops before the first retrieval call.
    try:
        from . import provenance as _prov
        from . import eligibility as _elig
        # The extraction call's own recorded identity travels WITH the graph it produced:
        # every relation is stamped with this call's assertion origin, so corroboration
        # downstream counts distinct sources rather than repetitions. (Codex F6-R3.)
        extracted = extraction.extract_note(note_text, extract_llm, billing_context,
                                            model_profile=profiles.get("extraction"))
        facts = extracted.facts
        _prov.anchor_facts(note_text, facts, document_version=document_version)
        # ---- Structural composition (issue #6 items 2/3) -----------------------------
        # A DIFFERENT signal from extraction's own relation calls: which events the note
        # documents together, under the same heading, purely from the document's own
        # structure -- never a judgement about which actions are usually integral to
        # which (extraction.py reserves that judgement and never makes it either). Runs
        # over the now-anchored primary facts so it can locate each one in `note_text`;
        # its output joins extraction's own relations into the SAME grounding/validation
        # pass below, so a structurally-derived edge is held to the identical standard
        # as an extracted one -- never a parallel, less-verified path.
        from . import composition as _compose
        structural_relations = _compose.compose(facts, note_text)
        relations = _prov.bind_relation_evidence(
            list(extracted.relations) + structural_relations, facts)
        relations = _prov.validate_relations(relations, facts, note_text)
        if audit_repository is None:
            from app.core.config import PROVENANCE_DB
            audit_repository = _prov.SqliteAuditRepository(PROVENANCE_DB, strict=True)
        audit_hashes = [audit_repository.append(
            encounter_id, "evidence_anchoring", _prov.anchoring_report(facts))]
        # ---- Second independent reading, compared on GRAPH AXES ---------------------
        # Not a vote. The second reading DETECTS a disagreement on a code-changing axis;
        # the ORIGINAL PAGE settles it (directive section 3). Differently worded prose
        # for the same event aligns and produces no disagreement at all, so nothing can
        # be routed anywhere merely because two models phrased a finding differently.
        consensus = None
        recovery = None
        recall = None
        #: reading channel id -> the exact text facts anchored in it were verified
        #: against. The primary transcription is the implicit "" reading and is never
        #: listed here (see `provenance.readings_map`).
        readings: dict[str, str] = {}
        if extract_llm_b is not None:
            consensus, source_evidence, recovery, recall = _run_graph_consensus(
                note_text, facts, billing_context, extract_llm_b, profiles,
                document_version, source_evidence, source_reader,
                enforce_independence=enforce_second_reading_independence,
                source=source)
            # Every reading a fact may now be anchored in. The relation kernel re-reads
            # the document between two endpoint mentions to prove an edge's DIRECTION,
            # and it can only do that against the string those mentions were verified
            # in — so a recovered event's edges are proven in the reading that found it,
            # never mislocated in a transcription that may not contain the passage.
            if recall is not None:
                readings[recall.channel_id] = recall.text
            # ---- RECALL redundancy, not only axis agreement (issue #6 F7-R3) ---------
            # A performed service the PRIMARY extractor missed entirely used to be
            # recorded in consensus metadata and dropped, so the encounter proceeded with
            # an incomplete graph, no integrity complaint, and a silently under-coded
            # claim. An event the union ADMITTED -- proven against the original page and
            # found to rest on document text no primary event rests on -- is appended to
            # the primary fact list HERE, before source reconciliation, eligibility, graph
            # construction and retrieval. From this line on it is not a recovered event at
            # all: it is a fact, decided by exactly the code every primary fact is decided
            # by, with no parallel path that could drift.
            if recovery.facts:
                # On a TRIAL copy first. The second reading's edges naming a recovered
                # event go through the SAME binding and validation as primary edges -- so
                # a component the record calls PART_OF another service is demoted, not
                # billed twice -- and that validation fails CLOSED by raising, which at
                # this function's boundary is a whole-encounter hold. A malformed edge
                # from the SECOND reading must take the recovered events out, not the
                # encounter down: if the trial does not validate, the admissions are
                # withdrawn (recorded, and held by the gate below) and the primary
                # encounter proceeds exactly as it would have without a second reading.
                trial_facts = list(facts) + list(recovery.facts)
                try:
                    trial_relations = (
                        _prov.validate_relations(
                            list(relations) + _prov.bind_relation_evidence(
                                recovery.relations, trial_facts),
                            trial_facts, note_text, readings=readings)
                        if recovery.relations else relations)
                except _prov.RelationIntegrityError as exc:
                    recovery.withdraw(
                        f"the second reading's relational context for this event could "
                        f"not be validated against the relation kernel "
                        f"({type(exc).__name__}), so it was not added to the graph")
                else:
                    facts.extend(recovery.facts)
                    relations = trial_relations
            # Recorded AFTER admission is final, so the audit record and the graph carry
            # the verdict that actually held -- never an admission the trial withdrew.
            consensus.recovered_events = recovery.as_records()
            audit_hashes.append(audit_repository.append(
                encounter_id, "graph_consensus", consensus.as_record()))
        # ---- Source evidence: the transcription is a CANDIDATE reading ---------------
        # Anchoring above proves each quotation is verbatim in the TRANSCRIPTION. This
        # proves it is verbatim in the ORIGINAL DOCUMENT, by reconciling it against an
        # independent reading of the very page it sits on, and writes that page, its
        # image digest and the region back onto the span. Every deterministic channel
        # already in the document is free, so this runs for every quotation; the PAID
        # channel is escalated to later, and only where it can change the answer.
        source_reconciliation = None
        if source_evidence is not None:
            source_reconciliation = _reconcile_readings(source_evidence, facts, recall)
            _prov.apply_reconciliation(facts, source_reconciliation)
            audit_hashes.append(audit_repository.append(
                encounter_id, "source_evidence_reconciliation",
                source_reconciliation.certificate_record()))
        audit_hashes.append(audit_repository.append(encounter_id, "relation_graph", {
            "schema_version": extracted.schema_version,
            # the extraction call this graph came from -- the unit of assertion independence
            "assertion_origin": (extracted.origin.as_record() if extracted.origin else None),
            "relation_grammar_version": _prov.load_relation_grammar()["version"],
            "relations": [{"relation_id": r.relation_id,
                           "subject_event_id": r.subject_event_id,
                           "predicate": r.predicate.value,
                           "object_event_id": r.object_event_id,
                           "state": r.state.value,
                           "evidence_span_ids": r.evidence_span_ids,
                           "confidence": r.confidence,
                           # GROUNDING: what the RECORD establishes about the edge, and the
                           # spans that establish it -- the necessity control reads this, so
                           # it must be auditable (F6-R3)
                           "reconciliation_status": r.reconciliation_status,
                           "reconciliation_evidence": list(r.reconciliation_evidence or []),
                           # AGREEMENT: recorded for audit/confidence, never justification
                           "corroboration_status": r.corroboration_status,
                           "assertion_origins": sorted(str(o) for o in (r.assertion_origins or [])),
                           "independent_support": r.independent_support,
                           "support": r.support} for r in relations],
        }))
        # ---- Single-entity terminology normalization (issue #6 F7-R3-C4) -----------
        # Runs over EVERY fact -- including any the second-reading union recovered
        # above -- regardless of whether a second reading exists at all, and
        # regardless of whether two readings agreed on the wording: the cross-reading
        # pairwise match in `graph_consensus.compare_axes` only ever fires on a
        # MISMATCH, so an abbreviation both readings wrote identically (or a note
        # extracted from only one reading) never reached a concept lookup before.
        # Placed BEFORE eligibility so `fact_snapshot_digest` captures the
        # normalized `governed_terms` it already covers, not a pre-normalization
        # snapshot a later mutation could diverge from.
        from . import coreference as _coref
        terminology_normalizations: list[dict] = []
        for _fact in facts:
            terminology_normalizations.extend(
                _coref.normalize_fact_terminology(_fact, source, encounter_id))
        if terminology_normalizations:
            audit_hashes.append(audit_repository.append(
                encounter_id, "terminology_normalization",
                {"normalizations": terminology_normalizations}))
        intents = _elig.evaluate(facts, relations, encounter_id, date_of_service,
                                 source=source)
        audit_hashes.append(audit_repository.append(encounter_id, "eligibility_enforced", {
            "control_mode": "ENFORCED_FAIL_CLOSED",
            "model_profiles": profiles,
            "summary": _elig.summary(intents),
            "diff": _elig.shadow_diff(facts, intents),
        }))
        # ---- THE single clinical representation --------------------------------------
        # Everything above -- anchored evidence, the reconciled relation kernel, the
        # eligibility roles, the service episodes and the cannot-link constraints -- is
        # compiled into ONE addressable graph here. Retrieval below is authorized by its
        # intents, the certificate binds its record, and claim assembly reads it to say
        # exactly which nodes and edges each released line rests on.
        from . import graph as _graph
        _episodes, _ = _elig.build_episodes(facts, relations, encounter_id,
                                            date_of_service)
        clinical_graph = _graph.build_graph(
            facts, relations, intents, encounter_id=encounter_id,
            date_of_service=date_of_service, episodes=_episodes,
            extraction_schema_version=extracted.schema_version,
            relation_grammar_version=_prov.load_relation_grammar()["version"],
            axis_resolutions=(consensus.resolutions if consensus is not None else ()),
            # The UNION's verdict on every second-reading-only event, not the raw
            # unmatched list: an admitted one is already a node above, and the rest are
            # recorded here with the reason they are not.
            unmatched_second_reading=(consensus.recovered_events
                                      if consensus is not None else ()))
        graph_problems = clinical_graph.integrity_problems()
        audit_hashes.append(audit_repository.append(
            encounter_id, "clinical_graph",
            {"graph_sha256": clinical_graph.graph_sha256(),
             **clinical_graph.as_record()}))
    except Exception as exc:
        return _system_hold_result(encounter_id, date_of_service,
                                   "pre_retrieval_integrity", exc, source)

    # Hard capability boundary: only the canonical event of an ELIGIBLE intent can
    # construct a RetrievalRequest. Holds, non-claim evidence, missing intents and
    # merged mentions are accounted for without invoking any retrieval implementation.
    from .eligibility import EligibilityState as _ES
    from .eligibility import RetrievalRequest
    _elig_state = {e: it for it in intents for e in it.clinical_event_ids}
    pre_retrieval_gates = []
    # A graph that contradicts itself is an integrity state, not a retry and not a
    # coding judgement: every later stage would be reasoning about a representation
    # that disagrees with itself. Non-retryable -> BLOCKED (directive section 8).
    if graph_problems:
        pre_retrieval_gates.append(GateResult(
            "clinical_graph_integrity", Outcome.BLOCKED, "; ".join(graph_problems),
            "clinical evidence/service graph", retryable=False))
    # An event an independent reading of the ORIGINAL DOCUMENT reported, that this run
    # could neither confirm nor refute, must not let the encounter present as a complete
    # claim -- that is the silent omission this control exists to prevent, one step
    # later. It is SYSTEM work, not coding work and not a documentation gap: what is
    # missing is a reading of the page, so it is retryable and never routed to a coder.
    for _held in (recovery.holds if recovery is not None else ()):
        pre_retrieval_gates.append(GateResult(
            f"second_reading_event_unverified:{_held.second_event_id}",
            Outcome.UNKNOWN, _held.reason,
            "event-candidate union (product directive section 3)", retryable=True))
    # Codex F7-R3, exact-SHA re-review, defect A: a page no independent reading could
    # cover used to be RECORDED (`recall_uncovered_pages`) but never BLOCKED anything --
    # a failed proactive read, or a reader that covered no page, still let the encounter
    # proceed to a fully resolved, presentable claim. A service documented only on that
    # page is then silently absent, with every control reporting clean.
    #
    # The first version of this gate exempted a page whose AGGREGATE `PageStatus` is
    # BLANK -- but the compiler assigns that status whenever the PRIMARY transcription
    # and the embedded-text layer both found zero tokens (round-2 re-review). Neither
    # of those is a read of the rendered PAGE IMAGE: a blank vision transcription may
    # mean the model omitted the page's visible content, and an empty text layer says
    # nothing about text drawn as an image. "Two channels found nothing" is not proof a
    # human looking at the page would find nothing too.
    #
    # The second version exempted a page once ANY independent image-capable channel had
    # a `PageRead` record for it at all -- but a record existing is not the same as that
    # record reporting BLANK (exact-SHA re-review, second pass): an independent reader
    # can return a read whose OWN `PageRead.status` is `UNREADABLE` (it tried and could
    # not get text) or `MISSING` (it did not return the page at all), and both used to
    # count as "inspected" purely because a record was present. A page is exempt only
    # when an independent, image-capable channel's read of it POSITIVELY reports
    # `PageStatus.BLANK` -- a genuine inspection that found nothing, not merely an
    # attempt that failed or never happened.
    if consensus is not None and source_evidence is not None:
        from app.contracts.source_evidence import ChannelKind as _ChannelKind
        from app.contracts.source_evidence import PageStatus as _PageStatus
        from app.contracts.source_evidence import independent_of as _independent_of
        primary_channel = source_evidence.primary_channel

        def _blank_verifying_read(page_number: int):
            """The independent, image-capable read that positively certified this
            page BLANK, or None. Returned (not just a bool) so the exempting read's
            own channel/detail/digest -- already validated non-empty by
            `build_page_read` -- can be named in the durable record rather than only
            implied by a boolean gate outcome (Codex F7-R3-A, exact-SHA re-review,
            fourth pass)."""
            page = source_evidence.page(page_number)
            if page is None or primary_channel is None:
                return None
            for read in page.reads:
                if (read.channel_id == source_evidence.primary_channel_id
                        or read.status is not _PageStatus.BLANK):
                    continue
                channel = source_evidence.channel(read.channel_id)
                if (channel is not None
                        and channel.kind in (_ChannelKind.VISION, _ChannelKind.OCR)
                        and _independent_of(channel, primary_channel)):
                    return read
            return None

        # Durable, per-page provenance for every page exempted as blank -- the
        # channel and its own validated detail, not merely a boolean the gate
        # consumed and discarded (Codex F7-R3-A, exact-SHA re-review, fourth pass).
        consensus.recall_blank_pages = tuple(
            {"page": p, "channel_id": read.channel_id, "detail": read.detail,
             "text_sha256": read.text_sha256}
            for p in consensus.recall_uncovered_pages
            for read in [_blank_verifying_read(p)] if read is not None)
        _blocking_pages = [p for p in consensus.recall_uncovered_pages
                           if _blank_verifying_read(p) is None]
        if _blocking_pages:
            pre_retrieval_gates.append(GateResult(
                "recall_page_coverage", Outcome.UNKNOWN,
                f"page(s) {_blocking_pages} of the original document have no "
                f"independent reading; a service documented only there would be "
                f"silently omitted from the claim",
                "independent document recall (issue #6 F7-R3)", retryable=True))
    # issue #6 item 5/F8-R2: computed BEFORE the retrieval loop (not only for
    # audit afterward) so each `RetrievalRequest` below can carry every OTHER
    # fact its own service intent groups it with -- semantic eligibility then
    # reads what the whole documented service states, never just one isolated
    # fact. `facts`/`relations` are already final at this point (structural
    # composition already ran above), so this is the same computation the
    # audit-only version at the end of this function would produce; computing
    # it once, here, and reusing it there removes the duplicate work.
    _service_intents = _compose.service_intents(facts, relations)
    _facts_by_id = {f.fact_id: f for f in facts if f.fact_id}
    _intent_facts_by_event: dict[str, tuple] = {}
    for _si in _service_intents:
        _members = tuple(_facts_by_id[eid] for eid in _si.component_event_ids
                         if eid in _facts_by_id)
        for eid in _si.component_event_ids:
            _intent_facts_by_event[eid] = _members

    lines = []
    for fact in facts:
        # issue #6 item 8: reset every iteration -- a loop-local carried across facts
        # would otherwise attach the PREVIOUS fact's candidate eligibility report to
        # a line that never went through resolve() at all this iteration.
        _candidate_eligibility = None
        _it = _elig_state.get(fact.fact_id)
        if _it is None:
            line = ResolvedLine(
                fact=fact, chosen=None, method=ResolutionMethod.ABSTAINED,
                rationale="missing ClaimLineIntent — retrieval prohibited",
                excluded_reason="pre-retrieval integrity hold")
            pre_retrieval_gates.append(GateResult(
                f"eligibility_intent:{fact.fact_id}", Outcome.UNKNOWN,
                "no eligibility intent was produced for the extracted event",
                "eligibility-before-retrieval", retryable=True))
        elif _it.state is not _ES.ELIGIBLE_FOR_RETRIEVAL:
            # ALL non-PASS decisions for the human-readable rationale; only the
            # decisions that actually produced the state for the ROUTING decision. A
            # recorded-but-non-blocking note (a diagnosis with no explicit documented
            # linkage) must not make a single-cause hold look multi-cause.
            _non_pass = [d for d in _it.decisions if d.outcome is not Outcome.PASS]
            _blocking = _elig.blocking_decisions(_it)
            _r = "; ".join(f"{d.gate}: {d.detail}" for d in _non_pass) or _it.state.value
            # WHO must act on each hold is DECLARED by the eligibility engine
            # (`eligibility._HOLD_OWNERS`), never re-derived here from gate names.
            #
            # This used to special-case exactly one gate: `axis_consensus` (a
            # code-changing axis two independent readings read differently that the
            # original page could not settle) became a provider question, and EVERY
            # other unresolved gate fell through to generic REVIEW. Two of those were
            # the same defect class -- an unsettled PART_OF/SEPARATE_FROM relation, and
            # a duplicate mention that cannot be assigned to either of two explicitly
            # distinct services -- both a code-changing fact the record does not state.
            # One was not a coding question at all: an extracted event with no clinical
            # action is an unusable graph node, i.e. an integrity state. Declaring the
            # owner beside the gate is what stops the NEXT gate from inheriting the
            # coder queue by omission. (Product directive section 8.)
            _owners = {_elig.hold_owner(d) for d in _blocking}
            # EVERY blocking hold is a question only the provider can answer -> ONE
            # precise query. Deliberately no GateResult and no excluded_reason: a gate
            # here would route the encounter to generic REVIEW, and an excluded_reason
            # would drop the item from routing entirely -- both forbidden for this case.
            _query_only = bool(_blocking) and _owners == {_elig.OWNER_PROVIDER_QUERY}
            line = ResolvedLine(
                fact=fact, chosen=None, method=ResolutionMethod.ABSTAINED,
                rationale=(f"held for a targeted provider query ({_r})" if _query_only
                           else f"diverted before retrieval ({_it.state.value}: {_r})"),
                excluded_reason=(None if _query_only
                                 else f"eligibility state {_it.state.value}"),
                documentation_gap=("; ".join(d.detail for d in _blocking)
                                   if _query_only else None))
            if _it.state is _ES.AUTO_HOLD and not _query_only:
                # Integrity beats retry beats judgement, and the gate carries that:
                # an unverifiable state is BLOCKED, an unresolved DEPENDENCY is a
                # retryable UNKNOWN (SYSTEM_HOLD), and only a hold no declared owner
                # claims stays the non-retryable UNKNOWN that reaches a coder.
                _integrity = _elig.OWNER_INTEGRITY in _owners
                _retryable = (not _integrity and _elig.OWNER_SYSTEM in _owners
                              and _elig.OWNER_CODER not in _owners)
                pre_retrieval_gates.append(GateResult(
                    f"eligibility_hold:{fact.fact_id}",
                    Outcome.BLOCKED if _integrity else Outcome.UNKNOWN, _r,
                    "eligibility-before-retrieval", retryable=_retryable))
        elif fact.fact_id != _it.clinical_event_ids[0]:
            line = ResolvedLine(
                fact=fact, chosen=None, method=ResolutionMethod.ABSTAINED,
                rationale=f"merged into eligible intent {_it.intent_id}; no duplicate retrieval",
                excluded_reason="duplicate mention represented by canonical claim-line intent")
        else:
            # issue #6 item 5/F8-R2: every other fact this event's own service
            # intent groups it with, so semantic eligibility (inside resolve())
            # reads the whole documented service, not just this one fact in
            # isolation. Empty tuple (falls back to the fact alone) when this
            # fact belongs to no multi-member intent -- the common case.
            _intent_facts = _intent_facts_by_event.get(fact.fact_id, ())
            try:
                if fact.kind is FactKind.EM:
                    line = em.resolve_em(
                        RetrievalRequest(_it, fact, intent_facts=_intent_facts), source)
                else:
                    line = resolution.resolve(
                        RetrievalRequest(_it, fact, intent_facts=_intent_facts), source,
                        llm=verify_llm, corroborate=corroborate_llm,
                        dos=date_of_service, reconciliation=source_reconciliation)
            except Exception as exc:
                return _system_hold_result(encounter_id, date_of_service,
                                           f"retrieval_execution:{fact.fact_id}", exc, source)
            # issue #6 item 8: captured here, before arbitration/refinement below MAY
            # reconstruct `line` (see the item 7 comment at the end of this loop for
            # why that matters) -- `em.resolve_em` does not run semantic eligibility
            # at all, so this is honestly None for an EM line, never guessed.
            _candidate_eligibility = getattr(line, "candidate_eligibility", None)
        # A fact that went through propose-then-verify is already resolved-or-
        # escalated on authoritative entailment; don't second-guess it with the
        # weaker arbitration fallback. (Diagnoses verify too when they reach the
        # embedding fallback.) Other kinds still arbitrate residual ambiguity.
        went_through_pv = (verify_llm is not None
                           and fact.kind in (FactKind.PROCEDURE, FactKind.IMAGING,
                                             FactKind.DIAGNOSIS))
        # A line that carries a `documentation_gap` was held by a DETERMINISTIC
        # constraint that failed (an unsupported descriptor interval) or by the tie
        # policy's targeted provider query. Neither may be overturned by a bounded model
        # pick: the directive states that the LLM verifier "may not invent a code or
        # override a failed deterministic constraint," and that a tie is settled by the
        # document or asked about -- not decided by a model. Arbitration therefore only
        # ever sees residual ambiguity that no constraint and no page has already
        # answered. (Before this, a SUPPLY/DRUG line held for an unsupported bounded
        # measurement went straight to arbitration, which could re-select the very
        # candidate the interval check had just refused.)
        if ((not line.resolved) and line.alternatives and fact.billable
                and not went_through_pv and not line.documentation_gap):
            line = arbitration.arbitrate(line, arbitrate_llm)
        # AUDIT: a tie that several candidates survived is a claim-affecting decision
        # in its own right -- which axes distinguished them, what the ORIGINAL DOCUMENT
        # was proven to say about each, and whether the page settled it or the provider
        # was asked. Recorded whether the tie released a code or held the line, so the
        # certificate's audit chain can answer "why not the other candidate?".
        if line.tie_record:
            try:
                audit_hashes.append(audit_repository.append(
                    encounter_id, "code_tie_resolution",
                    {"fact_id": fact.fact_id, "intent_id": _it.intent_id,
                     "released": bool(line.resolved),
                     "code": (line.chosen.code if line.chosen else ""),
                     **line.tie_record}))
            except Exception as exc:              # durable audit is enforced
                return _system_hold_result(
                    encounter_id, date_of_service,
                    f"code_tie_audit_persistence:{fact.fact_id}", exc, source)
        # OBSERVE: feed a propose-then-verify success into the learned index so that,
        # once the same phrase->code is confirmed across enough distinct encounters,
        # it resolves deterministically next time. Real mode only; fail-safe.
        if (verify_llm is not None and line.resolved
                and line.method is ResolutionMethod.VERIFIED
                and fact.kind in (FactKind.PROCEDURE, FactKind.IMAGING)):
            from . import learned
            learned.observe(encounter_id, fact.description, line.chosen.code,
                            line.chosen.system, line.chosen.descriptor,
                            [s.text for s in fact.evidence])
        # ICD-10-CM 'highest documented specificity': sharpen an unspecified/NOS
        # diagnosis to the most-specific code the documentation entails — a
        # structural laterality upgrade, then (in real mode) a verified upgrade past
        # a broad catch-all to a specific on-concept relative. Authoritative and
        # entailment-checked; escalates rather than billing an unspecified code when
        # the record supports a specific one but verification is split.
        if line.resolved and fact.kind is FactKind.DIAGNOSIS:
            line = resolution.refine_diagnosis_specificity(
                line, source, verify_llm, corroborate_llm,
                reconciliation=source_reconciliation)
        if line.resolved and line.fact.billable:
            # Data-driven bundling filter: a resolved code the source declares
            # NOT separately reportable (bundled / non-covered / MUE 0) is kept
            # for the audit trail but dropped from the claim. Agnostic.
            if source.separately_billable(
                    line.chosen.code, line.chosen.system, date_of_service) is Outcome.BLOCKED:
                line.excluded_reason = "not separately reportable per authoritative data"
            elif line.chosen.system in ("cpt", "hcpcs"):
                # Laterality/bilateral modifiers and billing UNITS belong to
                # procedure/supply codes only. An ICD-10 DIAGNOSIS encodes laterality
                # IN the code (right vs left vs unspecified) and never takes an RT/LT
                # modifier or a unit count — so this whole block is skipped for it.
                # Data-driven per-line modifiers (laterality) + billing units
                # (descriptor-driven, so a "2-4 items" code bills as one unit).
                line.modifiers = modifier_engine.assign(
                    line.fact, line.chosen.descriptor,
                    bilat=source.bilat_indicator(line.chosen.code))
                cnt = line.fact.attributes.get("count") or line.fact.attributes.get("quantity") or 1
                try:
                    cnt = int(cnt)
                except (TypeError, ValueError):
                    cnt = 1
                line.units = ontology.billing_units(cnt, line.chosen.descriptor)
                # A dosed drug bills by dose, not count: documented total dose /
                # the code's authoritative per-unit dose (e.g. 30 mg / 'per 15 mg'
                # = 2 units). Falls back to the count-based units above when the
                # dose or per-unit is unavailable.
                if line.fact.kind is FactKind.DRUG:
                    du = ontology.drug_billing_units(
                        ontology.documented_dose_text(line.fact),
                        source.drug_unit(line.chosen.code))
                    if du is not None:
                        line.units = du
        # issue #6 items 7/8: stamped here, LAST, after every helper above that may
        # reconstruct `line` (`arbitration.arbitrate`,
        # `resolution.refine_diagnosis_specificity`) rather than mutate it in place
        # -- either would otherwise silently drop these fields back to their
        # dataclass defaults. None of `resolve`/`em.resolve_em`/arbitration/
        # refinement decide a submission status; it always comes from the intent
        # eligibility already computed in `eligibility.evaluate`.
        if _it is not None:
            line.claim_submission_status = _it.claim_submission_status
        # issue #6 item 8: restored here for the SAME reason -- captured right after
        # resolve()/em.resolve_em() returned, before arbitration/refinement could
        # reconstruct `line` and silently drop it back to the dataclass default.
        if _candidate_eligibility is not None:
            line.candidate_eligibility = _candidate_eligibility
        lines.append(line)

    # issue #6 item 8: the SAME grouping already computed above (and already fed
    # into retrieval's semantic eligibility, item 5) -- serialized here for the
    # audit trail, preserved regardless of whether any of these events ended up
    # billed. Never recomputed: one grouping, used for both retrieval and audit,
    # so the two can never silently drift apart.
    service_intents = [{"intent_id": si.intent_id, "component_event_ids": si.component_event_ids}
                       for si in _service_intents]

    result = CodingResult(
        encounter_id=encounter_id,
        date_of_service=date_of_service,
        service_date_binding=(dict(service_date_binding)
                              if service_date_binding else None),
        lines=lines,
        claim_line_intents=intents,
        relations=relations,
        audit_record_hashes=audit_hashes,
        graph=clinical_graph,
        consensus=(consensus.as_record() if consensus is not None else None),
        terminology_normalizations=tuple(terminology_normalizations),
        service_intents=service_intents,
    )
    # Mechanic 4 — collapse duplicate resolved codes into one line before anything
    # downstream reasons about the claim as a set.
    dedup_lines(result, source)
    # Mechanic 1 — code-type/section applicability (e.g. an anesthesia-section code
    # is not separately reportable by the operating provider).
    apply_section_applicability(result)
    # Claim-level modifiers (E/M-25, distinct-service 59/X) once all lines exist —
    # this records which PTP pairs a justified modifier bypasses.
    modifier_engine.assign_claim(result, source)
    # Mechanic 3 — resolve NCCI PTP conflicts by DEMOTING the bundled component
    # (not blocking the claim) whenever no distinct-service modifier is justified.
    apply_ncci_bundling(result, source)
    # An ancillary procedure that ESCALATED but is an NCCI 'always-bundled' component
    # of a billed primary is INTEGRAL — decide it (bundle), don't send it to review.
    apply_integral_bundling(result, source)

    apply_global_package(result, source)
    # ---- Escalation to a PAID independent read, scoped to where it matters -----------
    # Only now is it known WHICH quotations justify a released line, so only now can the
    # second read be aimed. A page is re-read only when (a) a quotation behind a billed
    # line sits on it and (b) no deterministic channel could read it — an image-only or
    # low-yield page. A document whose text layer covers it therefore costs nothing
    # extra; a scanned one costs a read of the few pages the claim rests on, never a
    # second read of the whole note.
    result.document_version = document_version
    if (source_evidence is not None and source_reader is not None
            and source_reconciliation is not None):
        from app.contracts.source_evidence import pages_needing_independent_read
        billed_span_ids = {s.span_id for ln in result.billable_lines
                           for s in (ln.fact.evidence or []) if s.span_id}
        wanted = pages_needing_independent_read(
            source_evidence, source_reconciliation, billed_span_ids)
        if wanted:
            try:
                extra = source_reader.read_pages(wanted)
                channel = source_reader.channel()
            except Exception as exc:
                # A failed second read proves nothing, so it must not look like one:
                # the affected quotations stay UNVERIFIABLE and the gate holds.
                extra, channel = {}, None
                result.notes.append(
                    f"independent page read unavailable for pages {list(wanted)} "
                    f"({type(exc).__name__}); those quotations remain unverified")
            if extra and channel is not None:
                try:
                    escalated = source_evidence.with_channel(channel, extra)
                    reconciled = _reconcile_readings(escalated, facts, recall)
                except Exception as exc:
                    # A second read that cannot be INCORPORATED proves nothing either.
                    # The first reconciliation stands (those quotations are still
                    # UNVERIFIABLE and still hold), and the reason is recorded rather
                    # than raised past the claim that has already been computed.
                    result.notes.append(
                        f"independent page read could not be incorporated "
                        f"({type(exc).__name__}: {exc}); those quotations remain "
                        f"unverified")
                else:
                    source_evidence = escalated
                    source_reconciliation = reconciled
                    _prov.apply_reconciliation(facts, source_reconciliation)
                    try:
                        audit_hashes.append(audit_repository.append(
                            encounter_id, "source_evidence_reconciliation",
                            source_reconciliation.certificate_record()))
                    except Exception as exc:              # durable audit is enforced
                        return _system_hold_result(
                            encounter_id, date_of_service,
                            "source_evidence_audit_persistence", exc, source)
    result.source_reconciliation = source_reconciliation
    result.gates = pre_retrieval_gates + gates.run_gates(result, note_text, source,
                                                        readings=readings)
    decide(result, source=source)
    # Actionable documentation guidance for whatever could not be coded confidently.
    from . import recommendations as _recs
    result.recommendations = _recs.build_recommendations(result)
    # Make each routed item self-contained: attach its provider-facing suggested
    # solution to the routing entry (joined by stable fact_id) so a PROVIDER_QUERY
    # carries the exact question to send — no fragile description-based join needed.
    _attach_recommendations(result)
    # ---- Fail-closed, order-safe release attestation (Codex F6-R5) --------------------
    # Invariant: the RETURNED verdict, the CERTIFICATE, the data fingerprint, and the LAST
    # durable audit decision can never disagree. Achieved by (a) treating a missing data
    # fingerprint as an integrity hold that prevents certification, (b) building the
    # certificate in memory BEFORE any terminal persistence, and (c) persisting the terminal
    # release decision LAST -- reflecting the final (possibly downgraded) verdict and binding
    # the certificate hash -- so AUTO_READY is never persisted before it is real, and a
    # certificate never exists without its durable record.
    terminal_extra: dict = {}
    try:
        fingerprint = source.data_fingerprint()
    except Exception:
        fingerprint = None
    if not _fingerprint_certifiable(fingerprint):
        # Without a fingerprint that identifies the authoritative data (non-empty counts +
        # OK manifest) the claim is not certifiable -- a swallowed/partial/empty fingerprint
        # must not certify. Retryable hold, no certificate.
        result.gates.append(GateResult(
            "data_fingerprint", Outcome.UNKNOWN,
            "authoritative-data fingerprint unavailable or incomplete; provenance cannot be attested",
            "audit/certificate integrity", retryable=True))
        decide(result, source=source)
        result.certificate = None
    else:
        try:
            cert = certificate.build_certificate(
                result, note_text,
                source_identity={"source": type(source).__name__, "data": fingerprint,
                                 "models": profiles,
                                 "extraction_schema": extracted.schema_version})
        except Exception as exc:
            result.gates.append(GateResult(
                "release_evidence_persistence", Outcome.UNKNOWN,
                f"certificate could not be built: {type(exc).__name__}",
                "audit/certificate integrity", retryable=True))
            decide(result, source=source)
            result.certificate = None
        else:
            result.certificate = cert
            terminal_extra = {"certificate_sha256": cert.get("certificate_sha256")}
    # Terminal release decision is persisted LAST, carrying the FINAL verdict. If this
    # durable write fails, the release is not attestable: drop the certificate and downgrade
    # so a returned AUTO_READY can never exist without a matching durable record.
    try:
        result.audit_record_hashes.append(audit_repository.append(
            encounter_id, "release_decision", {
                "verdict": result.verdict.value,
                "destination": result.destination.value if result.destination else None,
                "billable_event_ids": [ln.fact.fact_id for ln in result.billable_lines],
                "gate_outcomes": [{"name": g.name, "outcome": g.outcome.value}
                                  for g in result.gates],
                # What the chain-of-custody guarantee ACTUALLY was for this release: which
                # terminal-head anchor backend was in force and whether it is a real external
                # trust boundary. Recorded so no artifact can imply an integrity property that
                # was not enforced -- when no anchor is configured this says so, in the record
                # itself. (Codex F6-R4-A.)
                "terminal_head_anchor": _terminal_head_anchor(audit_repository),
                **terminal_extra,
            }))
    except Exception as exc:
        result.certificate = None
        result.gates.append(GateResult(
            "release_evidence_persistence", Outcome.UNKNOWN,
            f"terminal release decision could not be persisted: {type(exc).__name__}",
            "audit/certificate integrity", retryable=True))
        decide(result, source=source)
    return result


def _terminal_head_anchor(audit_repository) -> dict:
    """Declare, in the durable release record, what the terminal-head trust boundary was.

    Total by construction: a repository implementation without an anchor (or one whose
    status cannot be read) is DESCRIBED as such, never omitted -- an absent field would
    read as 'nothing to report' when the honest answer is 'not externally anchored'.
    Enforcement lives in the repository's append path, which raises and holds the release;
    this is the attestation of what was in force, not the control itself.
    """
    unknown = {"backend": "unavailable", "configured": False,
               "external_trust_boundary": False}
    status_fn = getattr(audit_repository, "checkpoint_status", None)
    if not callable(status_fn):
        return unknown
    try:
        status = status_fn()
    except Exception as exc:                       # pragma: no cover - defensive
        return {**unknown, "backend": "error", "problems": [type(exc).__name__]}
    if not isinstance(status, dict):               # pragma: no cover - defensive
        return unknown
    # `location`/`guarantee` are backend-agnostic (every backend names where its
    # checkpoints live, and one that claims an external boundary states what earns the
    # claim), so the durable record says WHICH store anchored this release and on what
    # basis -- not merely that some anchor was configured.
    # `adoption_allowed` is carried for the same reason as `limitation`: a release certified
    # during a one-run legacy-adoption migration was NOT anchored for the whole of its
    # journal's history, and the durable record has to say so rather than let a later reader
    # infer coverage the anchor never had. (Codex F6-R4-A finding A.)
    return {k: status.get(k) for k in
            ("backend", "configured", "external_trust_boundary", "required",
             "adoption_allowed", "store_id", "journal_seq", "anchored_seq", "location",
             "guarantee", "limitation", "problems")
            if k in status}


def _system_hold_result(encounter_id: str, date_of_service: str | None,
                        stage: str, exc: Exception, source) -> CodingResult:
    """Typed fail-closed result for any pre-retrieval operational/integrity failure.

    LOUD as well as fail-closed. Holding here is correct and already typed, but the
    hold alone is not diagnosable: the ClaimBundle carries no gate detail, so this
    stage reaches an operator only as a coarse `release.reason_codes` entry, and the
    deployed batch log says nothing beyond `SYSTEM_RETRY | 0 diagnosis line(s) | 0
    service line(s)`. Every note in the batch then fails identically with the cause
    stated nowhere -- which is how a missing second-reading credential, an unreachable
    provider or a malformed extractor response all present as the same silent zero.

    So the CAUSE and its traceback go to the log. They deliberately do NOT go into the
    artifact: an exception message can quote the note (`ExtractionSchemaError` embeds
    the offending value), and a claim artifact is a different distribution boundary
    from an operator log. The artifact keeps the exception TYPE only.
    """
    from .models import GateResult, Outcome
    logger.error("  %s: held with zero retrieval at the %s boundary - %s: %s",
                 encounter_id, stage, type(exc).__name__, exc, exc_info=True)
    result = CodingResult(encounter_id=encounter_id, date_of_service=date_of_service)
    result.gates = [GateResult(stage, Outcome.UNKNOWN,
                               f"{stage} failed ({type(exc).__name__})",
                               "enforced pipeline boundary", retryable=True)]
    decide(result, source=source)
    return result


def _reconcile_readings(document, facts, recall, *, only=None):
    """Prove every anchored quotation against the ORIGINAL DOCUMENT — each one in the
    reading its character offsets actually belong to (issue #6 F7-R3).

    A quotation the primary transcription proposed is located by the transcription's own
    page arithmetic; a quotation an INDEPENDENT reading of the document proposed is
    located by that reading's. Proving both with one set of offsets would not be a
    smaller error than proving neither — it would name a page confidently and wrongly.
    The two records are then ONE record, joined on span id (which is salted with the
    reading, so the union cannot collide).
    """
    from app.contracts.source_evidence import (merge_reconciliations, reconcile_reading,
                                               reconcile_spans)
    from . import provenance as _prov

    by_reading = _prov.span_targets_by_reading(facts)

    def _wanted(targets):
        return [t for t in targets if only is None or t.span_id in only]

    # ALWAYS produced, even with no targets: the record carries the per-page outcomes and
    # channel identities every downstream gate and certificate reads, and an encounter
    # with nothing to prove still has to say which pages were read by whom.
    parts = [reconcile_spans(document, _wanted(by_reading.get("", [])))]
    if recall is not None:
        targets = _wanted(by_reading.get(recall.channel_id, []))
        if targets:
            parts.append(reconcile_reading(document, targets, recall))
    # A quotation anchored in a reading this call was not given cannot be located on any
    # page, and the honest outcome is a loud stop rather than a quietly unreconciled
    # span. `source_evidence_gate` would already hold such a span as unproven, but a
    # reading that exists and is not routed is a defect in THIS function, not an
    # encounter-level fact, and it must not present as one.
    known = {"", recall.channel_id if recall is not None else ""}
    stray = sorted(key for key in by_reading if key not in known)
    if stray:
        raise RuntimeError(
            f"evidence is anchored in reading(s) {stray} that this reconciliation was "
            f"not given; a quotation cannot be proven against a page in a reading "
            f"nobody supplied")
    return merge_reconciliations(*parts)


def _run_graph_consensus(note_text, facts, billing_context, extract_llm_b, profiles,
                         document_version, source_evidence, source_reader, *,
                         enforce_independence: bool = False,
                         source: CodeSource | None = None):
    """Second reading -> EVENT-CANDIDATE UNION + axis comparison -> TARGETED
    original-page verification.

    Returns `(report, source_evidence, recovery)`. The document is returned because a
    targeted page verification may have added a paid independent channel to it: carrying
    that forward means the later release-time escalation never pays to read the same page
    twice, and the reconciliation the claim is finally proven against is the strongest
    reading obtained anywhere in the run. The `recovery` is the union's decision about
    every event only the second reading found (`claude_coder.event_union`): the caller
    appends its facts/edges to the PRIMARY graph inputs, so a service the primary
    extractor missed is decided by the ordinary pipeline instead of being recorded and
    dropped (issue #6 F7-R3).

    ONE page reconciliation serves both jobs. The axis comparison needs the pages behind
    a disagreeing quotation; the union needs the pages behind a candidate new event; both
    are reconciled together and, when a page cannot be read at all, escalated together --
    so recovering an event never costs a second paid read of a page this run already
    bought.

    Failure is loud, never silent: anything that goes wrong here propagates to the
    caller's pre-retrieval boundary, which holds the encounter with zero retrieval. A
    second reading that could not be taken must never present as two readings agreeing.
    """
    from . import event_union as _union
    from . import graph_consensus as _gc
    from . import provenance as _prov

    # ---- INDEPENDENCE IS A PRECONDITION, NOT A FOOTNOTE (issue #6 F7-R5) ------------
    # `independent_providers` used to be computed after both readings had been taken and
    # then only recorded, so a deployment whose two readings resolved to ONE vendor
    # produced an artifact asserting an independence the run never had -- and paid twice
    # for it. When this reading is the pipeline's own independence control the identities
    # are compared FIRST, from what each callable declares (`verify.model_profile_of`,
    # the same primitive the relation graph and code corroboration read), and a pair that
    # is not positively different stops the encounter at the caller's pre-retrieval
    # boundary before a token is spent. A caller-supplied second extractor is not this
    # control -- it is a disagreement detector, whose value does not depend on vendor
    # independence -- so for it the fact is recorded, exactly as before.
    primary_provider = str((profiles.get("extraction") or {}).get("provider")
                           or "").strip().lower()
    second_provider = str((profiles.get("second_extraction") or {}).get("provider")
                          or "").strip().lower()
    independent_providers = bool(primary_provider and second_provider
                                 and primary_provider != second_provider)
    if enforce_independence and not independent_providers:
        raise extraction.SecondReadingUnavailable(
            f"the second reading of the note is enabled as an INDEPENDENT control, but "
            f"its provider ({second_provider or 'undeclared'}) is not positively "
            f"different from the primary reading's ({primary_provider or 'undeclared'})"
            f"; two readings by one vendor share training data, tokeniser and failure "
            f"modes, so their agreement is self-confidence rather than confirmation")

    # ---- THE SECOND READING READS THE DOCUMENT, NOT THE FIRST READING'S TRANSCRIPT ---
    # Both extractors used to be handed the SAME string: the primary vision
    # transcription. That makes the second reading a check on what the transcription
    # CONTAINED and nothing at all on what it LEFT OUT — a service the transcription
    # never captured is absent from the only text either extractor ever sees, so both
    # miss it identically and the claim is silently short a line, with every control
    # reporting clean (issue #6 F7-R3, reopened).
    #
    # The compiler already built an independent reading of the ORIGINAL DOCUMENT for
    # every note, and `recall_channel` is its own answer to which channel may be
    # evidence about the primary one — the document's own text layer where it exists
    # (free, deterministic, reproducible from the same bytes forever), a paid page read
    # where it does not. No new document-reading mechanism is introduced here; what
    # changes is that the recall extraction is run over THAT reading.
    #
    # Falling back to `note_text` when no channel could read a single page is not a
    # silent degradation: the second reading is then exactly what it was before (a
    # disagreement detector over one transcript), the fact is recorded on the report,
    # and every quotation on an unreadable page is already held by source
    # reconciliation.
    recall = None
    if source_evidence is not None:
        from app.contracts.source_evidence import recall_reading as _recall_reading
        candidate = _recall_reading(source_evidence)
        recall = candidate if (candidate is not None and candidate.usable) else None

    # ---- PROACTIVELY read pages no independent channel covers, BEFORE extraction runs
    # (issue #6 F7-R3, round-9 re-review, defect A). A paid vision read of an image-only
    # page used to happen only LATER, to verify a quotation a candidate event already
    # rested on -- which means a service the primary TRANSCRIPTION omitted on such a page
    # could never be proposed in the first place, no matter how independent the second
    # reading's model calls were: recall extraction never saw the page's text at all.
    #
    # The SAME reader used for the later, targeted escalation below is reused here. Its
    # channel identity is fixed (derived from the client that will perform the read, not
    # chosen per call -- issue #6 F7-R5), so this is not a second, competing mechanism;
    # `with_channel` now allows a second call for that identity to WIDEN it to pages it has
    # not yet read (never to overwrite one it has), so this proactive read and the later
    # targeted one compose in one document instead of colliding on the same channel id.
    _recall_page_read_pages: tuple[int, ...] = ()
    _recall_page_read_detail = ""
    recall_uncovered = (tuple(recall.uncovered_pages) if recall is not None
                       else tuple(p.page_number for p in source_evidence.pages)
                       if source_evidence is not None else ())
    if source_reader is not None and source_evidence is not None and recall_uncovered:
        from app.contracts.source_evidence import (ChannelIndependenceError,
                                                   require_independent_channel)
        try:
            # Identity first, pages second -- see the identical reasoning at the later
            # escalation call below, which this mirrors exactly.
            channel = source_reader.channel()
            require_independent_channel(source_evidence, channel)
            widened = source_evidence.with_channel(
                channel, source_reader.read_pages(recall_uncovered),
                require_independent=True)
        except ChannelIndependenceError:
            # A control that is not independent is MISCONFIGURED, not unavailable.
            raise
        except Exception as exc:
            _recall_page_read_detail = (
                f"proactive independent read of page(s) {list(recall_uncovered)} "
                f"unavailable ({type(exc).__name__}: {exc}); these pages remain "
                f"recall-uncovered and any service documented only on them is "
                f"unrecoverable this run")
        else:
            source_evidence = widened
            candidate = _recall_reading(source_evidence)
            recall = candidate if (candidate is not None and candidate.usable) else recall
            newly_covered = tuple(p for p in recall_uncovered
                                  if recall is None or p not in recall.uncovered_pages)
            if newly_covered:
                _recall_page_read_pages = newly_covered
                _recall_page_read_detail = (
                    f"proactively read page(s) {list(newly_covered)} with an "
                    f"independent vision channel so recall extraction could see them")
            else:
                _recall_page_read_detail = (
                    f"an independent read of page(s) {list(recall_uncovered)} was "
                    f"obtained but covered none of them; they remain recall-uncovered")

    recall_text = recall.text if recall is not None else note_text
    extracted_b = extraction.extract_note(
        recall_text, extract_llm_b, billing_context,
        run_id=_SECOND_READING_RUN_ID,
        model_profile=profiles.get("second_extraction"))
    # Anchored into the reading it was extracted from, and stamped with WHICH reading
    # that is, so nothing downstream slices the wrong string.
    _prov.anchor_facts(recall_text, extracted_b.facts, document_version=document_version,
                       reading_channel_id=(recall.channel_id if recall is not None
                                           else None))
    # ONE alignment, shared: an event the axis comparison counted as MATCHED must never
    # also be proposed to the union as a new event, and the only way to guarantee that is
    # for both to read the same correspondence rather than recompute it.
    alignment = _gc.align(facts, extracted_b.facts)
    report, primary_by_id, second_by_node = _gc.compare(
        facts, extracted_b.facts, second_origin=extracted_b.origin, alignment=alignment,
        source=source)
    pairs, _unmatched_primary_facts, unmatched_second_facts = alignment
    # Identity first, page reads second. A candidate is only worth paying to prove when
    # it could actually change the claim: it must rest on document text no primary event
    # rests on AND fail to corefer with any primary event. A candidate the record already
    # carries is settled here, for free, before a page is read (issue #6 F7-R3).
    candidates = _union.propose(facts, unmatched_second_facts, source=source)
    report.independent_providers = independent_providers
    report.independence_enforced = bool(enforce_independence)
    # The recall control's own reach, stated in the durable record: which reading it ran
    # over, and which pages of the document no reading other than the transcription
    # covered. On an uncovered page an omitted service remains unrecoverable, and that
    # has to be a visible property of the run rather than an inference from an empty
    # recovery list.
    report.recall_page_read_pages = _recall_page_read_pages
    report.recall_page_read_detail = _recall_page_read_detail
    report.recall_reading_channel_id = (recall.channel_id if recall is not None else "")
    report.recall_uncovered_pages = (tuple(recall.uncovered_pages)
                                     if recall is not None
                                     else tuple(p.page_number
                                                for p in source_evidence.pages)
                                     if source_evidence is not None else ())

    reconciliation = None
    wanted = set(_union.pending_span_ids(candidates))
    if report.disagreements:
        wanted |= _gc.disagreement_span_ids(report.disagreements, primary_by_id,
                                            second_by_node)
    if source_evidence is not None and wanted:
        from app.contracts.source_evidence import (ChannelIndependenceError,
                                                   pages_needing_independent_read,
                                                   require_independent_channel)
        both = list(facts) + list(extracted_b.facts)
        reconciliation = _reconcile_readings(source_evidence, both, recall, only=wanted)
        # TARGETED escalation, exactly as the directive asks: pay for an independent read
        # of only the pages a disagreeing quotation -- or a candidate new event -- sits on
        # that no channel could read.
        if source_reader is not None:
            pages = pages_needing_independent_read(source_evidence, reconciliation,
                                                   wanted)
            if pages:
                try:
                    # Identity first, pages second: `channel()` establishes the reading
                    # vendor from the client that will make the call and refuses if it
                    # is not independent of the primary channel, so a non-independent
                    # reader costs nothing and cannot enter the document. The document
                    # boundary enforces the same property again -- a source reader is a
                    # caller-supplied object (F7-R5).
                    channel = source_reader.channel()
                    # BEFORE the pages are read, not as a side effect of adding them:
                    # `with_channel`'s own check runs only after its arguments have been
                    # evaluated, so relying on it alone would still pay for a reading it
                    # then refuses -- and a caller-supplied reader (unlike this module's
                    # own `IndependentVisionReader`) does not check itself.
                    require_independent_channel(source_evidence, channel)
                    escalated = source_evidence.with_channel(
                        channel, source_reader.read_pages(pages),
                        require_independent=True)
                    # `recall` is deliberately NOT recomputed against the escalated
                    # document: it is the exact string the second extraction was
                    # anchored into, and re-deriving it here (the paid channel may now
                    # cover more pages than the text layer) would move every offset
                    # those spans were verified at.
                    reconciled = _reconcile_readings(escalated, both, recall, only=wanted)
                except ChannelIndependenceError:
                    # A control that is not independent is MISCONFIGURED, not
                    # unavailable. Recording it as "verification unavailable" would turn
                    # a broken safety property into an ordinary hold and hide the reason.
                    raise
                except Exception as exc:
                    # A verification that could not run proves nothing, and must not look
                    # like one: the affected axes stay unresolved and become a query, and
                    # an unproven candidate event is held rather than admitted.
                    report.escalation_detail = (
                        f"targeted page verification unavailable for pages "
                        f"{list(pages)} ({type(exc).__name__}); the disagreeing axes and "
                        f"candidate events remain unsettled by the original document")
                else:
                    source_evidence = escalated
                    reconciliation = reconciled
                    report.escalated_pages = tuple(pages)
                    report.escalation_detail = (
                        "disagreeing axes and candidate events verified against a paid "
                        "independent read of the original pages")
    resolutions = _gc.resolve(list(report.disagreements), primary_by_id, second_by_node,
                              reconciliation)
    _gc.apply_resolutions(primary_by_id, second_by_node, resolutions)
    report.resolutions = tuple(resolutions)
    recovery = _union.admit(
        candidates, reconciliation=reconciliation,
        alignment={str(getattr(right, "fact_id", "") or ""):
                   str(getattr(left, "fact_id", "") or "") for left, right in pairs},
        second_relations=extracted_b.relations,
        taken_ids={str(getattr(f, "fact_id", "") or "") for f in facts},
        id_prefix=f"{_SECOND_READING_RUN_ID}-")
    report.recovered_events = recovery.as_records()
    return report, source_evidence, recovery, recall


def _model_profile_identity(extract_llm, verify_llm, corroborate_llm,
                            extract_llm_b=None) -> dict:
    """Auditable provider/profile identity; no credential values are ever included.

    The verification/corroboration providers are read from what each CALLABLE declares
    (`verify.model_profile_of`), not restated here: a caller-supplied pair used to be
    recorded as openai/claude regardless of who it actually was, which made
    `independent_providers` a statement about this function rather than about the run.
    (Round 5, phase 5.)"""
    from . import verify as _verify

    def callable_name(fn):
        return None if fn is None else f"{getattr(fn, '__module__', '')}.{getattr(fn, '__qualname__', type(fn).__name__)}"

    profiles = {
        # Read from what the CALLABLE declares, for the same reason the verification and
        # corroboration profiles below are: a caller-supplied extractor used to be
        # recorded with the configured provider regardless of who it actually was, which
        # made `independent_providers` a statement about this function's configuration
        # rather than about the run (issue #6 F7-R5, the same defect round 5 fixed one
        # entry lower down).
        "extraction": {"callable": callable_name(extract_llm),
                       **_verify.model_profile_of(extract_llm)},
        "verification": {"callable": callable_name(verify_llm),
                         **_verify.model_profile_of(verify_llm)},
        "corroboration": {"callable": callable_name(corroborate_llm),
                          **_verify.model_profile_of(corroborate_llm)},
        "second_extraction": {"callable": callable_name(extract_llm_b),
                              **_verify.model_profile_of(extract_llm_b)},
    }
    try:
        from app.core import config
        # Configuration identifies the extraction call ONLY when the pipeline itself is
        # making it (`extract_llm is None` -> `extraction._default_llm`, which reads this
        # very setting). For a caller-supplied extractor the configured provider
        # describes a call that is not being made.
        if extract_llm is None:
            profiles["extraction"].update({"provider": config.LLM_PROVIDER,
                                           "model": (config.CLAUDE_MODEL if config.LLM_PROVIDER == "claude"
                                                     else config.OPENAI_MODEL)})
        # Model/effort detail only for the DEFAULT callables — the only ones whose runtime
        # model this configuration actually selects.
        if verify_llm is _verify.default_verify_llm:
            profiles["verification"]["model"] = config.OPENAI_MODEL
        if extract_llm_b is extraction.default_second_extract_llm:
            _second = extraction._SECOND_READING_PROVIDER.get(config.LLM_PROVIDER)
            if _second is not None:
                profiles["second_extraction"].update({
                    "provider": _second,
                    "model": (config.OPENAI_MODEL if _second == "openai"
                              else config.CLAUDE_MODEL)})
        if corroborate_llm is _verify.default_corroborate_llm:
            profiles["corroboration"].update({
                "model": config.CLAUDE_VERIFY_MODEL or config.CLAUDE_MODEL,
                "effort": config.CLAUDE_VERIFY_EFFORT or config.CLAUDE_EFFORT})
    except Exception:
        profiles["identity_status"] = "configuration_unavailable"
    # The DECIDING fact, recorded exactly as resolution computes it: whether an agreement
    # between the two judgement calls may be credited as independent confirmation at all.
    origin = _verify.corroboration_origin(verify_llm, corroborate_llm)
    profiles["corroboration_origin"] = origin
    profiles["independent_providers"] = origin in _verify.INDEPENDENT_CORROBORATION_ORIGINS
    # OBSERVATIONAL, and deliberately not a control input. The assertion the corroborator
    # checks is the entailment SELECTION, so independence is measured against the verifier;
    # but the corroborator sharing a vendor with the EXTRACTOR is a weaker correlation worth
    # seeing in the record (in the current default deployment it does). Promoting it to a
    # control would be a product decision, not a silent code change -- it would make every
    # propose-then-verify line non-independent under that same default.
    _extraction_provider = str(profiles["extraction"].get("provider") or "").strip().lower()
    _corroboration_provider = str(
        profiles["corroboration"].get("provider") or "").strip().lower()
    profiles["corroborator_shares_extraction_provider"] = bool(
        _extraction_provider and _corroboration_provider
        and _extraction_provider == _corroboration_provider)
    return profiles


def _attach_recommendations(result: CodingResult) -> None:
    """Attach each routed item's provider-facing suggested solution to its routing
    entry, so a PROVIDER_QUERY (or any routed item) is self-contained. Joins on the
    STABLE fact_id — a fact's free-text description is not unique — and falls back to
    subject only when no fact_id is present."""
    by_id = {r["fact_id"]: r for r in result.recommendations if r.get("fact_id")}
    by_subject = {}
    for r in result.recommendations:
        by_subject.setdefault(r.get("subject"), r)
    for item in result.routing:
        rec = by_id.get(item.get("fact_id")) or by_subject.get(item.get("subject"))
        if rec:
            item["recommendation"] = rec["recommendation"]


def _occurrence_context(result: CodingResult) -> tuple[dict, set]:
    """What the RECORD says about which events are separate, for occurrence
    reconciliation: each event's service episode, and every explicitly separated pair."""
    from .models import RelationPredicate, RelationState
    episodes: dict[str, str] = {}
    for intent in (result.claim_line_intents or []):
        for event_id in (intent.clinical_event_ids or []):
            if intent.service_episode_id:
                episodes[str(event_id)] = str(intent.service_episode_id)
    separated = {frozenset((str(r.subject_event_id), str(r.object_event_id)))
                 for r in (result.relations or [])
                 if r.predicate is RelationPredicate.SEPARATE_FROM
                 and r.state is RelationState.ASSERTED}
    return episodes, separated


def dedup_lines(result: CodingResult, source: CodeSource | None = None) -> None:
    """Mechanic 4 — OCCURRENCE RECONCILIATION: two documented mentions that resolve to
    the SAME authoritative code are one billable line, and are a second BILLABLE
    OCCURRENCE only when the record says they are.

    This is where procedure identity is finally settled. Two mentions that resolve to
    one authoritative code describe one procedure by the only authority that can say so
    — the descriptor set — however differently they were written. Whether that one
    procedure happened once or twice is a different question, and it is answered here by
    `claude_coder.coreference`, the same test the event union and eligibility use, on the
    record's own axes: anatomy, laterality, performer, approach, an explicitly distinct
    site/session/objective/encounter, an asserted separation, a different episode.

    UNITS ACCUMULATE ONLY FOR AN ESTABLISHED SECOND OCCURRENCE. They used to accumulate
    whenever the duplicate quoted text the first line had not quoted — so one service
    described twice in different words became two units — justified by the
    medically-unlikely-edit ceiling catching anything excessive. A ceiling is a limit on
    what MAY be billed; it is never evidence that a service was performed twice, and a
    service whose documented repeat count is two is inside every ceiling that permits
    two. That reasoning is gone (issue #6 F7-R3, reopened): an unestablished repeat is
    now one line with the units the record itself states, and the merged mention stays in
    the audit trail with the reason it did not add an occurrence.

    Agnostic: a set-merge on the resolved (code, system), never a named code."""
    from . import coreference as _coref
    episodes, separated = _occurrence_context(result)
    keep_by_key: dict[tuple[str, str], ResolvedLine] = {}
    # Every ESTABLISHED occurrence's own representative mention, per resolved code
    # (Codex F7-R3, round-9 re-review, defect B). A later mention used to be tested
    # only against the FIRST line this code was ever seen on, so a third mention that
    # matched the SECOND (already-distinct) occurrence but not the first one was wrongly
    # scored as a THIRD distinct occurrence -- e.g. left, right, right-again overcounted
    # to 3 instead of the documented 2. Testing against every occurrence already
    # recognized, not only the first, is what a genuine multi-occurrence cluster needs.
    reps_by_key: dict[tuple[str, str], list[ResolvedLine]] = {}
    # Cluster-level cardinality, PARALLEL to reps_by_key -- one {"count": int|None,
    # "units": int} per recognized occurrence, independent of which specific mention is
    # "keep" (Codex F7-R3, exact-SHA re-review, defect B). Comparing every new mention
    # only against the representative fact's CURRENT attributes made reconciliation
    # depend on ARRIVAL ORDER: missing-then-2-then-3 compared 3 against a representative
    # that was still "missing" (None), so 3 silently overwrote 2 instead of conflicting
    # with it. Tracking the cluster's own running cardinality -- update on EVERY repeat,
    # not just against whichever fact happens to be `keep` -- makes the result the same
    # for missing->2->3, 2->missing->3, and 2->3->missing alike.
    cluster_state: dict[tuple[str, str], list[dict]] = {}

    def _seed(ln: ResolvedLine) -> dict:
        return {"count": _coref.documented_cardinality(ln.fact.attributes),
               "units": ln.units}

    def _resync(key) -> None:
        keep_by_key[key].units = sum(s["units"] for s in cluster_state[key])

    for ln in result.lines:
        if not (ln.resolved and ln.fact.billable and not ln.excluded_reason):
            continue
        key = (ln.chosen.code, ln.chosen.system)
        keep = keep_by_key.get(key)
        if keep is None:
            keep_by_key[key] = ln
            reps_by_key[key] = [ln]
            cluster_state[key] = [_seed(ln)]
            continue
        verdicts = []
        for i, rep in enumerate(reps_by_key[key]):
            verdict, reason = _coref.event_verdict(
                left_kind=rep.fact.kind, right_kind=ln.fact.kind,
                left_action=rep.fact.description, right_action=ln.fact.description,
                left_attributes=rep.fact.attributes, right_attributes=ln.fact.attributes,
                left_episode=episodes.get(str(rep.fact.fact_id)),
                right_episode=episodes.get(str(ln.fact.fact_id)),
                explicitly_separated=frozenset(
                    (str(rep.fact.fact_id), str(ln.fact.fact_id))) in separated,
                source=source)
            # WHY this pair is undetermined matters (Codex F7-R3-C2, exact-SHA
            # re-review): an AMBIGUOUS AXIS (open vocabulary that is not an exact
            # match and not disjoint either -- a possible synonym OR a possible real
            # distinction) is a genuine, claim-changing uncertainty this resolver may
            # not silently guess through in either direction. Wording alone failing to
            # match with NO axis in question is a different, already-intentional case
            # this module exists to merge (see its own module docstring): two mentions
            # with compatible axes and unrelated action PHRASING are undetermined by
            # `action_identity` alone, and once both have independently resolved to the
            # SAME authoritative code with no documented distinctness, the descriptor
            # set -- not anyone's prose -- has already settled procedure identity.
            axis_ambiguous = bool(_coref.known_known_ambiguous(
                rep.fact.attributes, ln.fact.attributes, source=source))
            verdicts.append((i, verdict, reason, axis_ambiguous))
        same = [v for v in verdicts if v[1] == _coref.SAME_EVENT]
        axis_undetermined = [v for v in verdicts
                             if v[1] == _coref.UNDETERMINED and v[3]]
        wording_undetermined = [v for v in verdicts
                                if v[1] == _coref.UNDETERMINED and not v[3]]
        if same:
            idx, verdict, reason, axis_ambiguous = same[0]
        elif axis_undetermined:
            idx, verdict, reason, axis_ambiguous = axis_undetermined[0]
        elif wording_undetermined:
            idx, verdict, reason, axis_ambiguous = wording_undetermined[0]
        else:
            # DISTINCT from every occurrence recognized so far -- a new occurrence, and
            # this mention becomes ITS OWN representative for any later mention.
            reps_by_key[key].append(ln)
            cluster_state[key].append(_seed(ln))
            idx, verdict, reason, axis_ambiguous = verdicts[-1]
        # The merged mention's evidence is kept either way: it is what the record says
        # about this service, and losing it would make the surviving line rest on less
        # documentation than the encounter actually has.
        keep.fact.evidence = list(keep.fact.evidence) + list(ln.fact.evidence)
        if _coref.is_additional_occurrence(verdict):
            keep.rationale = (f"{keep.rationale}; a SECOND DOCUMENTED OCCURRENCE was "
                              f"folded in — {reason}")
            ln.excluded_reason = (f"second documented occurrence of {ln.chosen.code} — "
                                  f"units folded into the primary line ({reason})")
            _resync(key)
            continue
        if verdict == _coref.UNDETERMINED and axis_ambiguous:
            ln.excluded_reason = (
                f"{ln.chosen.code} is already on the claim, and whether this mention "
                f"is the same documented occurrence or a distinct one is not "
                f"established by the record ({reason}) — held rather than guessed")
            keep.chosen = None
            keep.method = ResolutionMethod.ABSTAINED
            keep.documentation_gap = (
                f"the record does not establish whether this mention of "
                f"{ln.chosen.code} is the same documented occurrence already on the "
                f"claim or a distinct one — please clarify ({reason})")
            keep.rationale = (f"{keep.rationale}; occurrence identity ambiguous — "
                              f"escalate rather than guess ({reason})")
            continue
        # A repeated mention of an occurrence already recognized. Codex F7-R3: a count
        # stated on a LATER mention used to be discarded outright, because only the
        # arrival-order-dependent representative's units ever survived; and comparing
        # against the representative's CURRENT attributes let a third mention silently
        # overwrite an already-established count instead of conflicting with it. Any
        # mention's stated count is honored into ITS CLUSTER regardless of arrival
        # order; two mentions of the SAME occurrence that each state a DIFFERENT
        # explicit count is the record disagreeing with itself, which no guess may
        # resolve, so the line holds instead of picking either one.
        state = cluster_state[key][idx]
        ln_count = _coref.documented_cardinality(ln.fact.attributes)
        if state["count"] is not None and ln_count is not None and state["count"] != ln_count:
            ln.excluded_reason = (
                f"{ln.chosen.code} is already on the claim, and the record states "
                f"conflicting explicit counts for this occurrence "
                f"({state['count']} vs {ln_count}) — held rather than guessed")
            keep.chosen = None
            keep.method = ResolutionMethod.ABSTAINED
            keep.documentation_gap = (
                f"the record states two different counts for this documented "
                f"occurrence ({state['count']} and {ln_count}) — please confirm how "
                f"many times it was performed")
            keep.rationale = (f"{keep.rationale}; conflicting documented counts "
                              f"({state['count']} vs {ln_count}) for one occurrence — "
                              f"escalate rather than guess")
            continue
        if ln_count is not None and state["count"] is None:
            state["count"], state["units"] = ln_count, ln.units
        _resync(key)
        keep.rationale = (f"{keep.rationale}; a second mention of this service was "
                          f"merged without adding units — {reason}")
        ln.excluded_reason = (f"{ln.chosen.code} is already on the claim and the "
                              f"record documents no second occurrence — merged into "
                              f"a single line ({reason})")


def apply_section_applicability(result: CodingResult) -> None:
    """Mechanic 1 — a code whose authoritative descriptor identifies it as a
    different CPT SECTION than the encounter supports is not separately reportable.
    The concrete, agnostic rule: an ANESTHESIA-section service (detected from
    descriptor grammar, not a code range) is billed by the anesthesia provider,
    so on a claim that also carries an operative procedure it is bundled into the
    surgeon's service unless the note documents a separate anesthesia provider.
    Fail-closed: excluded by default, kept in the audit trail."""
    from .ontology import code_section
    from .models import FactKind
    proc_lines = [ln for ln in result.billable_lines
                  if ln.chosen.system in ("cpt", "hcpcs")
                  and ln.fact.kind in (FactKind.PROCEDURE, FactKind.IMAGING)]
    has_operative = any(code_section(ln.chosen.descriptor) != "anesthesia"
                        for ln in proc_lines)
    if not has_operative:
        return                          # e.g. an anesthesia provider's own claim
    from .models import ResolutionMethod
    _SEP = ("anesthesia_provider", "separate_anesthesia_provider",
            "anesthesia_by_separate_provider", "separate_anesthesia")
    _reason = ("anesthesia-section service — not separately reportable by the "
               "operating provider (no separate anesthesia provider documented)")
    for ln in proc_lines:                # a RESOLVED anesthesia-section code
        if code_section(ln.chosen.descriptor) != "anesthesia":
            continue
        if any(ln.fact.attributes.get(k) for k in _SEP):
            continue                    # a separate anesthesia provider is documented
        ln.excluded_reason = _reason
    # An ESCALATED procedure that is itself an ANESTHESIA service is decided
    # DETERMINISTICALLY (exclude), rather than leaving its handling to depend on
    # which specific code the LLM happened to resolve. Signal: the BEST-ranked
    # candidate is the anesthesia section AND that section DOMINATES the candidate
    # set. The 'best + dominant' test matters because an anesthesia-section
    # descriptor ('Anesthesia for procedures on <region>') is a semantic neighbour
    # of any procedure in that region and will appear incidentally among a surgical
    # line's candidates — so mere presence is not enough; it must be the leading match.
    for ln in result.lines:
        if ln.resolved or ln.excluded_reason or not ln.fact.billable:
            continue
        if ln.fact.kind not in (FactKind.PROCEDURE, FactKind.IMAGING):
            continue
        if any(ln.fact.attributes.get(k) for k in _SEP):
            continue
        alts = ln.alternatives
        if not alts or code_section(alts[0].descriptor) != "anesthesia":
            continue                    # the leading match is not anesthesia
        n_anes = sum(1 for c in alts if code_section(c.descriptor) == "anesthesia")
        if n_anes * 2 >= len(alts):      # anesthesia dominates the candidate set
            ln.chosen = alts[0]
            ln.method = ResolutionMethod.DETERMINISTIC
            ln.excluded_reason = _reason


def apply_ncci_bundling(result: CodingResult, source: CodeSource) -> None:
    """Mechanic 3 — turn NCCI PTP edits into a resolution, not a hard block. For
    each pair of billable procedure lines with a PTP edit, if no distinct-service
    modifier is justified (the pair was not bypassed), DEMOTE the column-2
    component code (the authoritative row tells us which side is the bundled
    component) — the claim keeps the payable comprehensive code and drops the
    component, exactly as a coder would, instead of blocking outright. Also honors
    the CPT '(separate procedure)' designation, which bundles a service performed
    alongside another procedure of the same session. All directionality comes from
    the data; no code is named here."""
    from .ontology import is_separate_procedure
    from .models import FactKind
    proc = [ln for ln in result.billable_lines
            if ln.chosen and ln.chosen.system in ("cpt", "hcpcs")]

    # (a) '(separate procedure)' designation — bundled ONLY when billed alongside
    # another actual PROCEDURE this session (a more comprehensive surgical service).
    # A supply/drug/device line (e.g. an implant HCPCS) is NOT a procedure and must
    # not trigger the bundle — otherwise a legitimately separate procedure is dropped
    # just because an implant was also reported.
    def _is_procedure(o) -> bool:
        return o.fact.kind in (FactKind.PROCEDURE, FactKind.IMAGING)

    for ln in proc:
        if ln.excluded_reason or not is_separate_procedure(ln.chosen.descriptor):
            continue
        if any(o is not ln and not o.excluded_reason and _is_procedure(o)
               and o.chosen.code != ln.chosen.code for o in proc):
            ln.excluded_reason = ("'(separate procedure)' designation — bundled "
                                  "when performed with another procedure this session")

    # (b) PTP edits — demote the component of any unbypassed pair.
    by_code = {ln.chosen.code: ln for ln in proc}
    for i in range(len(proc)):
        for j in range(i + 1, len(proc)):
            a, b = proc[i], proc[j]
            if a.excluded_reason or b.excluded_reason:
                continue
            edit = source.ncci_edit(a.chosen.code, b.chosen.code, result.date_of_service)
            if not edit:
                continue
            mod = edit.get("modifier")
            if mod not in ("0", "1"):
                continue                # deleted / non-applicable indicator -> no active edit
            pair = frozenset((a.chosen.code, b.chosen.code))
            if mod == "1" and pair in result.bypassed_ncci:
                continue                # a justified distinct-service modifier keeps both
            comp = by_code.get(edit.get("component"))
            if comp is not None and not comp.excluded_reason:
                comp.excluded_reason = (
                    f"bundled into {edit.get('payable')} per NCCI PTP "
                    f"(no distinct-service modifier justified)")
                result.ncci_suppressed.append((comp.chosen.code, edit.get("payable")))


def apply_integral_bundling(result: CodingResult, source: CodeSource) -> None:
    """Decide the 'integral vs separately billable' gray area authoritatively, so an
    ancillary procedure the resolver could not confidently code is not sent to a
    human when NCCI already answers it. For each ESCALATED (unresolved) procedure
    line, if one of its best candidates is an NCCI 'always-bundled' component
    (modifier indicator 0 — never separately reportable) of a code that IS billed on
    this claim, the ancillary is INTEGRAL to that primary: record it as bundled
    (excluded), not a review item.

    Safe by construction: it ONLY converts an escalation into a NON-billed exclusion
    — it never bills an uncertain code. An indicator-1 (bypassable-with-modifier)
    pair is a genuine judgement (bill-with-modifier vs bundle) and is left escalated.
    Authoritative (NCCI) and agnostic — no code is named here."""
    from .models import FactKind, ResolutionMethod
    billed = {ln.chosen.code for ln in result.billable_lines
              if ln.chosen and ln.chosen.system in ("cpt", "hcpcs")
              and ln.fact.kind in (FactKind.PROCEDURE, FactKind.IMAGING)}
    if not billed:
        return
    for ln in result.lines:
        if ln.resolved or ln.excluded_reason or not ln.fact.billable:
            continue
        if ln.fact.kind not in (FactKind.PROCEDURE, FactKind.IMAGING):
            continue
        done = False
        for cand in ln.alternatives[:4]:
            for primary in billed:
                edit = source.ncci_edit(primary, cand.code, result.date_of_service)
                if (edit and str(edit.get("component")) == cand.code
                        and str(edit.get("payable")) in billed
                        and str(edit.get("modifier")) == "0"):
                    ln.chosen = cand
                    ln.method = ResolutionMethod.DETERMINISTIC
                    ln.excluded_reason = (f"integral to {edit.get('payable')} — always "
                                          f"bundled per NCCI (not separately reportable)")
                    done = True
                    break
            if done:
                break


def apply_global_package(result: CodingResult, source: CodeSource) -> None:
    """Global surgical package (CMS global-period data): an E/M related to a
    same-day procedure that carries a global period (000/010/090) is included in
    that procedure's payment. The E/M is separately billable ONLY if the note
    documents significant, separately identifiable work; otherwise it is bundled
    — dropped from the claim, kept in the audit trail. Fail-closed."""
    from .models import FactKind
    has_global_proc = any(
        source.global_period(ln.chosen.code) in ("000", "010", "090")
        for ln in result.billable_lines
        if ln.fact.kind is not FactKind.EM and ln.chosen.system in ("cpt", "hcpcs"))
    if not has_global_proc:
        return
    for ln in result.lines:
        if (ln.resolved and ln.fact.kind is FactKind.EM and not ln.excluded_reason
                and not ln.fact.attributes.get("separately_identifiable")):
            ln.excluded_reason = ("bundled into the global surgical package "
                                  "(no separately-identifiable E/M documented)")


def render(result: CodingResult) -> str:
    """Human-readable audit trail — the explainability surface."""
    dest = f"  →  {result.destination.value}" if result.destination else ""
    out = [f"Encounter {result.encounter_id}  DOS={result.date_of_service}",
           f"VERDICT: {result.verdict.value}{dest}", ""]
    out.append("LINES:")
    for ln in result.lines:
        f = ln.fact
        if ln.resolved and ln.excluded_reason:
            out.append(f"  ∅ excluded {ln.chosen.system.upper()} {ln.chosen.code}  "
                       f"«{f.description}» — {ln.excluded_reason}")
        elif ln.resolved:
            mods = f"  +{','.join(ln.modifiers)}" if ln.modifiers else ""
            out.append(f"  ✓ {ln.chosen.system.upper()} {ln.chosen.code}{mods}  "
                       f"[{ln.method.value}]  «{f.description}»")
            out.append(f"      descriptor: {ln.chosen.descriptor[:70]}")
            out.append(f"      why: {ln.rationale}")
        else:
            tag = "not billed" if not f.billable else "ESCALATE"
            out.append(f"  ⚠ {tag}  «{f.description}»  — {ln.rationale}")
            cand = list(dict.fromkeys(f"{c.system.upper()} {c.code}"
                                      for c in ln.alternatives if c.code))
            if f.billable and cand:
                out.append(f"      candidates (unconfirmed): {', '.join(cand[:5])}")
        if f.evidence:
            out.append(f"      evidence: «{f.evidence[0].text[:70]}»")
    out.append("")
    out.append("GATES:")
    for g in result.gates:
        out.append(f"  [{g.outcome.value:>14}] {g.name}: {g.detail}  ({g.authority})")
    if result.notes:
        out.append("")
        out.append("DECISION:")
        for n in result.notes:
            out.append(f"  - {n}")
    if result.recommendations:
        out.append("")
        out.append("DOCUMENTATION RECOMMENDATIONS:")
        for r in result.recommendations:
            out.append(f"  • [{r['issue']}] {r['recommendation']}")
    return "\n".join(out)
