import time
import copy
from pathlib import Path
from datetime import datetime

from app.core.logger import get_logger
from app.core import cache as result_cache
from app.core.config import LLM_PROVIDER, RAG_TOP_K
from app.ingestion.pdf_parser import extract_from_pdf
from app.ner.entity_extractor import extract_entities
from app.rag.retriever import CandidateRetriever
from app.rag.vector_store import MedicalCodeVectorStore
from app.rag.code_reference import CodeReferenceDB
from app.coding.code_assigner import assign_codes
from app.validation.validator import CodingValidator
from app.compliance.datastore.store import ComplianceDataStore
from app.compliance.engine import ClaimScrubber, _parse_dos
from app.compliance.agents import build_default_agents
from app.models.schemas import CodingResult
from app.terminology import TerminologyNormalizer
from app.core.model_profiles import active_profile, execution_record
from app.clinical_facts import build_clinical_fact_report

logger = get_logger(__name__)

_VALID_MODIFIER_CLAIM_STATUSES = {"applied", "not_applicable"}


def _sanitize_modifier_claims(raw) -> list[dict]:
    """Defensively filter modifier_reasoning entries before Pydantic
    validation. modifier_reasoning is now a structured list[ModifierClaim]
    (see schemas.py) instead of free-text strings — if the LLM doesn't
    follow the schema (a malformed entry, or a regression to the old
    string format), constructing CPTCode(**data)/HCPCSCode(**data)
    directly would raise a Pydantic ValidationError and crash the whole
    note's processing. Drop anything that isn't a well-formed
    {modifier, status, reason} dict rather than crash — only case-
    normalizes `status`, never guesses an unrecognized value's meaning
    (that guessing is exactly the fragile regex-heuristic problem this
    schema change replaces)."""
    if not isinstance(raw, list):
        return []
    clean = []
    for entry in raw:
        if not isinstance(entry, dict):
            logger.warning(f"    [MODIFIER CLAIM] dropped non-dict modifier_reasoning entry: {entry!r}")
            continue
        # Uppercased to match how real modifier codes are always stored in
        # the "modifiers" array — without this, a lowercase claim (e.g.
        # the LLM writing "rt" instead of "RT") would persist in
        # modifier_reasoning with different casing than modifiers, even
        # though validator.py's own comparison already uppercases before
        # matching (so the auto-correction itself is unaffected — this is
        # about keeping the persisted audit trail internally consistent).
        modifier = str(entry.get("modifier", "")).strip().upper()
        status = str(entry.get("status", "")).strip().lower()
        if not modifier or status not in _VALID_MODIFIER_CLAIM_STATUSES:
            logger.warning(f"    [MODIFIER CLAIM] dropped malformed modifier_reasoning entry: {entry!r}")
            continue
        clean.append({"modifier": modifier, "status": status, "reason": str(entry.get("reason", ""))})
    return clean


class MedicalCodingPipeline:
    """Full Vision → NER → RAG → Multi-Pass LLM → Validation pipeline."""

    def __init__(self):
        self.vector_store = MedicalCodeVectorStore()
        self.ref_db = CodeReferenceDB()
        self.retriever: CandidateRetriever | None = None
        self.validator: CodingValidator | None = None
        self.compliance_store: ComplianceDataStore | None = None
        self.scrubber: ClaimScrubber | None = None
        self.terminology: TerminologyNormalizer | None = None
        self._initialized = False

    def initialize(self, force_rebuild_index: bool = False) -> None:
        logger.info("=" * 70)
        logger.info("INITIALIZING MEDICAL CODING PIPELINE")
        logger.info("=" * 70)

        logger.info("Loading code reference database...")
        self.ref_db.load_all()

        logger.info("Loading governed clinical terminology registry...")
        self.terminology = TerminologyNormalizer()

        logger.info("Building/loading Qdrant hybrid vector store...")
        self.vector_store.build_or_load(force_rebuild=force_rebuild_index)

        self.retriever = CandidateRetriever(self.vector_store)

        logger.info("Building/loading compliance data store + 13-filter scrubber...")
        self.compliance_store = ComplianceDataStore()
        self.compliance_store.build_or_load()
        self.scrubber = ClaimScrubber(
            self.compliance_store, agents=build_default_agents(self.compliance_store)
        )

        # CodingValidator needs the compliance store for authoritative global-period
        # lookups (e.g. distinguishing a diagnostic test from an actual procedure for
        # modifier -25 purposes) — built after compliance_store so it can be passed in.
        self.validator = CodingValidator(self.ref_db, self.compliance_store)

        self._initialized = True
        logger.info("Pipeline initialized successfully")

    def process_note(self, pdf_path: str | Path, use_cache: bool = True) -> CodingResult:
        if not self._initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")

        pdf_path = Path(pdf_path)
        start = time.time()

        logger.info(f"\n{'='*70}")
        logger.info(f"PROCESSING: {pdf_path.name}  [provider={LLM_PROVIDER.upper()}]")
        logger.info(f"{'='*70}")

        # Fix 4 — Response cache: same PDF + same pipeline version always returns same result
        if use_cache:
            cached = result_cache.get_cached(pdf_path)
            if cached is not None:
                try:
                    # note_text rides alongside the CodingResult in the cache
                    # entry (the model itself only keeps truncated sections) so
                    # the re-scrub below sees the full note. Entries written
                    # before this key existed just scrub without it.
                    cached_note_text = cached.pop("note_text", "")
                    r = CodingResult(**cached)
                    r.cached_result = True
                    # Always re-run the (cheap, local, deterministic) compliance
                    # scrubber so cached results still get the clean-claim gate and
                    # reflect the latest compliance data.
                    scrub_payload = r.model_dump()
                    scrub_payload["note_text"] = cached_note_text
                    scrub = self.scrubber.scrub(scrub_payload)
                    self._apply_scrub_verdict(r, scrub)
                    self._refresh_release_artifacts(r)
                    self._print_summary(r)
                    logger.info(f"  [cache] VERDICT: {r.final_disposition} — {r.final_summary}")
                    return r
                except Exception:
                    pass  # corrupt cache entry — reprocess

        # Step 1: Vision-based PDF extraction
        logger.info("[1/5] Extracting from PDF via GPT-4o Vision...")
        extraction = extract_from_pdf(pdf_path)
        metadata = extraction["metadata"]
        sections = extraction["sections"]
        note_category = extraction["note_category"]
        procedures_today = extraction["procedures_performed_today"]
        imaging_today = extraction["imaging_performed_today"]
        supplies_today = extraction["supplies_dispensed_today"]

        prior_surgery_info = extraction.get("prior_surgery_info", {}) or {}
        physician_documented_codes = extraction.get("physician_documented_codes", []) or []
        note_integrity = extraction.get("note_integrity") or {}

        logger.info(f"  Patient: {metadata.get('patient_name')} | DOS: {metadata.get('date_of_service')}")
        logger.info(f"  Category: {note_category}")
        logger.info(f"  Procedures today: {procedures_today}")
        logger.info(f"  Imaging today: {imaging_today}")
        logger.info(f"  Supplies today: {supplies_today}")
        if prior_surgery_info.get("is_post_op_visit"):
            logger.info(
                f"  Post-op visit: day {prior_surgery_info.get('days_post_op')} "
                f"after {prior_surgery_info.get('prior_surgery_description')} "
                f"(CPT {prior_surgery_info.get('prior_surgery_cpt')})"
            )

        # Step 2: NER — extract clinical entities
        logger.info("[2/5] Extracting clinical entities (NER)...")
        entities = extract_entities(sections)
        entities, terminology_report = self.terminology.normalize_entities(
            entities, sections)
        clinical_facts = build_clinical_fact_report(
            entities=entities, sections=sections, procedures=procedures_today,
            imaging=imaging_today, supplies=supplies_today,
            prior_surgery=prior_surgery_info)
        logger.info(f"  Found {len(entities)} entities")
        for e in entities:
            logger.info(f"    [{e.category:>14}] {e.clinical_term} {'['+e.laterality+']' if e.laterality else ''}")

        # Step 3: RAG — retrieve candidate codes
        logger.info("[3/5] Retrieving candidate codes (RAG/Qdrant hybrid)...")
        entity_candidates = self.retriever.retrieve_for_entities(entities)
        note_candidates = self.retriever.retrieve_for_full_note(sections)
        fact_candidates = self.retriever.retrieve_for_clinical_facts(
            clinical_facts)
        note_candidates = {
            system: self.retriever._round_robin(
                [rows for rows in (note_candidates.get(system) or [],
                                   fact_candidates.get(system) or []) if rows],
                RAG_TOP_K)
            for system in ("icd10", "cpt", "hcpcs")
        }

        merged = self._merge_candidates(entity_candidates, note_candidates)
        # Deleted/not-yet-effective codes must never be OFFERED to the coding
        # model in the first place — the vector index carries the full code
        # history, and a discontinued code that reaches the prompt can come
        # back as an assignment (observed live: G0456, deleted 2015, assigned
        # to a 2026 NPWT encounter and only caught post-hoc by validation).
        dos_for_filter = _parse_dos(metadata)
        merged = self._drop_inactive_candidates(merged, dos_for_filter)
        for cs, cands in merged.items():
            logger.info(f"  {cs.upper()}: {len(cands)} candidates retrieved")

        # Step 4: Multi-pass LLM code assignment
        logger.info("[4/5] Assigning codes (4-pass: ICD → CPT → HCPCS → Verify)...")
        entity_dicts = [e.model_dump() for e in entities]

        vision_context = {
            "note_category": note_category,
            "procedures_performed_today": procedures_today,
            "imaging_performed_today": imaging_today,
            "supplies_dispensed_today": supplies_today,
        }

        # Verified-claim exemplars from the finalized-claims registry:
        # shadow mode records what would be injected (calibration), live
        # mode (auto above the registry-size threshold) injects worked
        # examples into the coding prompts. Deterministic per registry
        # state, so it adds no run-to-run variance to the consistency gate.
        from app.coding import exemplars as _exemplars
        exemplar_block, exemplar_info = _exemplars.for_note(
            document_id=pdf_path.stem,
            note_category=note_category,
            note_sections=sections,
        )

        coding_result, usage = assign_codes(
            note_text=sections.get("full_text", ""),
            note_sections=sections,
            patient_metadata=metadata,
            entities=entity_dicts,
            rag_candidates=merged,
            vision_context=vision_context,
            prior_surgery_info=prior_surgery_info,
            db=self.ref_db,
            physician_documented_codes=physician_documented_codes,
            store=self.compliance_store,
            exemplar_block=exemplar_block,
            clinical_facts=clinical_facts,
        )

        # Immutable candidate snapshot: deterministic validation may propose
        # and realize changes, but the pre-validation claim must survive so
        # every mutation can be accounted for at the release boundary.
        from app.release.mutation_ledger import normalize_claim
        candidate_claim = normalize_claim(copy.deepcopy(coding_result))

        # Step 5: Validation
        logger.info("[5/5] Validating codes...")
        plan_text = sections.get("plan", "")
        full_text = sections.get("full_text", "")
        # Payer context for payer-gated deterministic checks (MUE-0
        # suppression): resolved from the note's own insurance field by the
        # same registry the claim scrubber uses, so the validator and the
        # compliance agents always see the same payer identity.
        from app.compliance.payer_registry import parse_insurance_text
        parsed_payer = parse_insurance_text(str(metadata.get("insurance") or ""))
        validation = self.validator.validate(
            coding_result,
            note_plan_text=plan_text,
            note_full_text=full_text,
            physician_documented_codes=physician_documented_codes,
            dos=_parse_dos(metadata),
            note_category=note_category,
            patient_dob=str(metadata.get("date_of_birth") or ""),
            payer_follows_medicare_coverage=parsed_payer.follows_medicare_coverage,
            # Billability anchor for the marginal-secondary demotion: the
            # sections whose contents the ICD prompt itself defines as
            # billable (assessment/diagnoses + imaging findings).
            note_assessment_text=" \n".join(
                s for s in (sections.get("assessment_diagnoses", ""),
                            sections.get("imaging_diagnostics", ""),
                            sections.get("chief_complaint", "")) if s),
            # Completeness invariant: the vision-extracted list of procedures
            # actually performed this encounter — each must be accounted for
            # on the final claim (coded or legitimately excluded).
            procedures_performed=procedures_today,
        )

        # Bind each final line to the exact authoritative database record and
        # edition window used for date-of-service validation.  Evidence is
        # normalized into spans but never invented when the coder omitted it.
        system_rows = (
            ("icd10_codes", "icd10_codes", self.ref_db.validate_icd10),
            ("cpt_codes", "cpt_codes", self.ref_db.validate_cpt),
            ("hcpcs_codes", "hcpcs_codes", self.ref_db.validate_hcpcs),
        )
        for array, source_id, lookup in system_rows:
            for line in coding_result.get(array) or []:
                code = str(line.get("code") or "").strip()
                ref = lookup(code) or {}
                line["source_record_ids"] = [f"{source_id}:{code.upper()}"] \
                    if ref else []
                line["source_effective_from"] = ref.get("effective_from")
                line["source_effective_to"] = ref.get("effective_to")
                line["source_temporal_authority"] = bool(
                    ref.get("temporal_authority", False))
                if not line.get("evidence_spans"):
                    span = line.get("supporting_text") or line.get("source") or ""
                    line["evidence_spans"] = [span] if span else []

        elapsed = time.time() - start

        result = CodingResult(
            document_id=pdf_path.stem,
            timestamp=datetime.now().isoformat(),
            success=True,
            processing_time=round(elapsed, 2),
            patient_metadata=metadata,
            note_sections={k: (v[:200] + "..." if len(v) > 200 else v) if v is not None else ""
                           for k, v in sections.items() if k != "full_text"},
            icd_codes=[self._to_icd(c) for c in coding_result.get("icd10_codes", [])],
            supporting_conditions=[self._to_supporting(c) for c in coding_result.get("supporting_conditions", [])],
            cpt_codes=[self._to_cpt(c) for c in coding_result.get("cpt_codes", [])],
            hcpcs_codes=[self._to_hcpcs(c) for c in coding_result.get("hcpcs_codes", [])],
            snomed_codes=[self._to_snomed(c) for c in coding_result.get("snomed_codes", [])],
            em_level_reasoning=coding_result.get("em_level_reasoning", ""),
            rag_context={
                "entities_extracted": len(entities),
                "candidates_per_system": {cs: len(cands) for cs, cands in merged.items()},
                # The actual candidate CODES offered to the coder, in the
                # rank order it saw them (deduped + similarity-sorted by
                # _merge_candidates, DOS-filtered by _drop_inactive_
                # candidates). Persisted so consistency-run variance can be
                # characterized post hoc: when a minority run lacks a
                # load-bearing code, this decides whether it was ABSENT from
                # that run's candidate set (a retrieval-recall gap) or
                # PRESENT-but-not-chosen (a generation disagreement) — the
                # split that determines whether a fix belongs in RAG or in
                # the coder. Codes only (no descriptors/scores) to keep the
                # per-run artifact small; rank order preserved because "at
                # what rank was 28118 offered" is part of the signal.
                "candidate_codes_per_system": {
                    cs: [str(c.get("code") or "").strip() for c in cands
                         if str(c.get("code") or "").strip()]
                    for cs, cands in merged.items()},
                "corrections_made": coding_result.get("corrections_made", []),
                "audit_notes": coding_result.get("audit_notes", ""),
                "vision_context": vision_context,
                "prior_surgery_info": prior_surgery_info,
                "exemplars": exemplar_info,
                # Full note text rides in every saved artifact (including the
                # per-run consistency dumps) — the flip-actuation pipeline
                # replays the validator against stored runs, and a replay
                # without the note text can't evaluate note-evidence rules.
                "note_full_text": full_text,
            },
            model_source=active_profile().provider,
            model_execution=execution_record(),
            api_usage=usage,
            physician_documented_codes=physician_documented_codes,
            missing_physician_codes=coding_result.get("missing_physician_codes", []),
            ner_entities=entity_dicts,
            terminology_normalization=terminology_report,
            clinical_facts=clinical_facts,
            # Persist the documented procedures so the completeness invariant
            # can re-run when this claim is re-validated on replay/reconcile
            # (the replayer reads it back from the stored payload).
            procedures_performed_today=procedures_today,
            candidate_claim=candidate_claim,
            note_integrity=note_integrity,
            **{k: v for k, v in validation.items()
               if k not in ("auto_coding_review_reasons", "auto_coding_summary")},
            auto_coding_review_reasons=(
                coding_result.get("auto_coding_review_reasons", [])
                + validation.get("auto_coding_review_reasons", [])
            ),
            auto_coding_summary=(
                coding_result.get("auto_coding_summary", "")
                or validation.get("auto_coding_summary", "")
            ),
        )
        # warnings existed on the schema but was never populated — consumers
        # reading result.warnings (instead of walking validation_issues) saw
        # an always-empty list. Mirror WARNING-severity issues into it.
        result.warnings = [
            f"[{i.category}] {i.code}: {i.message}"
            for i in result.validation_issues
            if getattr(i, "severity", "") == "WARNING"
        ]

        # Step 6: 13-filter compliance scrubber — the authoritative clean-claim gate
        logger.info("[6/6] Scrubbing claim through 14 compliance filters...")
        # CodingResult stores note_sections TRUNCATED (200 chars, full_text
        # dropped) for output size, so a bare model_dump() gave the scrubber
        # an empty claim.note_text — DocumentationAgent's distinct-modifier
        # language check could never fire, and letterhead-based state
        # inference (MAC jurisdiction scoping) had nothing to read. Hand the
        # scrubber the real note text explicitly.
        scrub_payload = result.model_dump()
        scrub_payload["note_text"] = full_text
        scrub = self.scrubber.scrub(scrub_payload)
        self._apply_scrub_verdict(result, scrub)

        # Centralized mutation accounting and source manifest are attached
        # after every deterministic layer has run.  Incomplete legacy
        # correction records remain explicitly unresolved and therefore
        # cannot produce AUTO_READY.
        self._refresh_release_artifacts(result)
        logger.info(f"  VERDICT: {result.final_disposition} — {result.final_summary}")

        # Fix 4 — Store to cache (only on success, so failed runs don't get
        # cached). note_text is stored alongside the model dump so cache-hit
        # re-scrubs see the full note (see the cache-read path above).
        if use_cache and result.success:
            cache_payload = result.model_dump()
            cache_payload["note_text"] = full_text
            result_cache.store(pdf_path, cache_payload)

        self._print_summary(result)
        return result

    @staticmethod
    def _refresh_release_artifacts(result: CodingResult) -> None:
        """Attach one internally consistent release snapshot to the model."""
        from app.release.claim_readiness import refresh_release_artifacts
        payload = result.model_dump(mode="json")
        refresh_release_artifacts(payload)
        result.mutation_ledger = payload["mutation_ledger"]
        result.authoritative_source_manifest = payload[
            "authoritative_source_manifest"]
        result.claim_readiness_certificate = payload[
            "claim_readiness_certificate"]

    # Review-reason marker for a scrub-CLEAN claim held back pending the
    # clinical-correctness audit — the audit's uphold verdict removes
    # exactly this marker and promotes the claim to CLEAN.
    AUDIT_PENDING_MARKER = "[clinical_audit/pending]"

    def _apply_scrub_verdict(self, result, scrub) -> None:
        """Make the 13-filter scrubber the single authoritative verdict and
        reconcile the legacy tier so the two can never contradict.

        Per the clean-claim rule: CLEAN passes; anything failing any filter is
        routed to REVIEW with the blocking reasons. The legacy AUTO/REVIEW/REJECT
        tier is derived from this (REJECT collapses into REVIEW).

        One gate sits above the scrubber's CLEAN: NO claim is CLEAN until
        the clinical-correctness review (tools/clinical_auditor.py) upholds
        it — a whole-claim expert review plus a verdict on every
        interpretive layer correction (self-reported and diff-derived).
        The universal hold exists because the absence of recorded
        corrections is exactly what an unreported mutation looks like
        (measured live, routine_00003: a demotion layer moved a coverage
        diagnosis off the claim without reporting it, and the
        corrections-scoped audit vacuously upheld the claim). The claim is
        held at REVIEW with the pending marker; the post-batch audit
        promotes it to CLEAN on an upheld verdict and demotes it to
        genuine REVIEW on a dispute. Fail closed: no audit -> never
        CLEAN."""
        # mode="json": every downstream reader of the record contract
        # (observables' signature(), record_coherence, replay gates) speaks
        # the saved-file string shape — a bare model_dump() would leave
        # Status/DenialRisk enum members in the in-memory record (see the
        # matching fix in tools/replay_reconcile._rebuild_run).
        result.claim_scrub = scrub.model_dump(mode="json")
        result.final_disposition = scrub.disposition.value      # CLEAN | REVIEW
        result.final_summary = scrub.summary

        interpretive = [m for m in (result.material_corrections or [])
                        if isinstance(m, dict) and m.get("interpretive")]
        audit = getattr(result, "clinical_audit", None) or {}
        # A stored "upheld" only releases the hold while it still describes
        # THIS claim — corrections or claim shape changed means the verdict
        # is stale and the claim must be re-reviewed (fail closed).
        audit_current = False
        if audit.get("verdict") == "upheld":
            try:
                from tools.clinical_auditor import corrections_fingerprint
                audit_current = (audit.get("fingerprint")
                                 == corrections_fingerprint(
                                     result.model_dump()))
            except Exception:
                audit_current = False
        if scrub.clean and not audit_current:
            result.final_disposition = "REVIEW"
            result.auto_coding_tier = "REVIEW"
            reason = (f"{self.AUDIT_PENDING_MARKER} claim awaits the "
                      f"clinical-correctness review "
                      f"({len(interpretive)} interpretive layer "
                      f"correction(s) to verdict + whole-claim review) — "
                      f"scrub verdict is CLEAN and will be restored on an "
                      f"upheld audit")
            existing = list(result.auto_coding_review_reasons or [])
            if not any(self.AUDIT_PENDING_MARKER in r for r in existing):
                result.auto_coding_review_reasons = existing + [reason]
            result.final_summary = (
                f"Scrub CLEAN, held for clinical audit: {scrub.summary}")
            result.auto_coding_summary = result.final_summary
            result.auto_coding_confidence = min(
                result.auto_coding_confidence, 0.84)
            return

        if scrub.clean:
            result.auto_coding_tier = "AUTO"
            result.auto_coding_summary = scrub.summary
        else:
            result.auto_coding_tier = "REVIEW"
            result.auto_coding_summary = scrub.summary
            # surface the scrubber's blocking findings as the review reasons
            blocking = [
                f"[{f.filter_id}/{f.denial_risk.value}] {', '.join(f.codes)}: {f.reason}"
                for f in scrub.blocking_findings
            ]
            # keep any pre-existing reasons, then add the authoritative ones
            existing = list(result.auto_coding_review_reasons or [])
            result.auto_coding_review_reasons = existing + blocking

        # auto_coding_confidence is computed by validator.py's _compute_tier
        # BEFORE the scrubber runs, clamped to match validator.py's own
        # LOCAL tier (e.g. AUTO -> floored at 0.85). The scrubber's verdict
        # above can override the tier in either direction — a claim
        # validator.py found no issues with can still be blocked by a
        # 13-filter FAIL (e.g. medical necessity), and vice versa — so the
        # confidence number must be re-clamped to the FINAL tier here too,
        # or a REVIEW-routed claim can carry a stale >=0.85 "AUTO" score
        # (observed live: tier=REVIEW, confidence=0.86 — a scrubber-only
        # MEDICAL_NECESSITY finding with zero validator.py issues), which
        # would mislead anyone scanning a review queue by confidence.
        if scrub.clean:
            result.auto_coding_confidence = max(result.auto_coding_confidence, 0.85)
        else:
            result.auto_coding_confidence = min(result.auto_coding_confidence, 0.84)

    def _merge_candidates(self, entity_cands: dict, note_cands: dict) -> dict:
        """Merge with entity-balanced rank, then deduplicate.

        Similarity scores produced for different queries are not a shared
        ranking.  Globally sorting them let one verbose entity consume the
        entire prompt budget while a different documented diagnosis/service's
        top candidate fell below the selectable cutoff.  Round-robin preserves
        each query's local rank: every entity's first result precedes any
        entity's second result, with full-note retrieval as another source.
        """
        systems = ("icd10", "cpt", "hcpcs")
        sources: dict[str, list[list[dict]]] = {cs: [] for cs in systems}
        for data in entity_cands.values():
            for cs, cands in data.get("candidates", {}).items():
                if cs in sources and cands:
                    # CandidateRetriever already preserves each query form's
                    # local rank via round-robin. Scores from raw/model/
                    # expanded queries are not comparable; re-sorting here
                    # would silently undo that balance.
                    sources[cs].append(list(cands))
        for cs, cands in note_cands.items():
            if cs in sources and cands:
                sources[cs].append(list(cands))

        merged = {cs: [] for cs in systems}
        for cs in systems:
            seen: set[str] = set()
            depth = 0
            while any(depth < len(rows) for rows in sources[cs]):
                for rows in sources[cs]:
                    if depth >= len(rows):
                        continue
                    candidate = rows[depth]
                    code = str(candidate.get("code") or "").strip()
                    if code and code not in seen:
                        seen.add(code)
                        merged[cs].append(candidate)
                depth += 1
        return merged

    def _drop_inactive_candidates(self, merged: dict, dos) -> dict:
        """Filter RAG candidates down to codes effective on the DOS, using the
        reference DB's own per-code effective ranges. Codes unknown to the
        reference DB are kept (the existence check downstream owns that
        question); only positively-dated-out codes are dropped."""
        out = {}
        for cs, cands in merged.items():
            kept = []
            for c in cands:
                code = c.get("code", "")
                entry = {"icd10": self.ref_db.icd10, "cpt": self.ref_db.cpt,
                         "hcpcs": self.ref_db.hcpcs}.get(cs, {}).get(code)
                if entry is not None and not self.ref_db.is_active_for_dos(cs, code, dos):
                    logger.info(f"  [candidate filter] {cs.upper()} {code} dropped — not "
                                f"effective on DOS ({entry['effective_from']}..{entry['effective_to']})")
                    continue
                kept.append(c)
            out[cs] = kept
        return out

    def _to_icd(self, raw: dict):
        from app.models.schemas import ICDCode
        return ICDCode(**{k: v for k, v in raw.items() if k in ICDCode.model_fields})

    def _to_supporting(self, raw: dict):
        from app.models.schemas import SupportingCondition
        return SupportingCondition(**{k: v for k, v in raw.items() if k in SupportingCondition.model_fields})

    def _to_cpt(self, raw: dict):
        from app.models.schemas import CPTCode
        data = {k: v for k, v in raw.items() if k in CPTCode.model_fields}
        if data.get("mdm_details") is None:
            data["mdm_details"] = {}
        data["modifier_reasoning"] = _sanitize_modifier_claims(data.get("modifier_reasoning"))
        return CPTCode(**data)

    def _to_hcpcs(self, raw: dict):
        from app.models.schemas import HCPCSCode
        data = {k: v for k, v in raw.items() if k in HCPCSCode.model_fields}
        data["modifier_reasoning"] = _sanitize_modifier_claims(data.get("modifier_reasoning"))
        return HCPCSCode(**data)

    def _to_snomed(self, raw: dict):
        from app.models.schemas import SNOMEDCode
        return SNOMEDCode(**{k: v for k, v in raw.items() if k in SNOMEDCode.model_fields})

    def _print_summary(self, r: CodingResult):
        cache_tag = " [CACHED]" if r.cached_result else ""
        logger.info(f"\n--- RESULTS: {r.document_id}{cache_tag} ---")
        logger.info(f"  Provider: {r.model_source.upper()} | Tier: {r.auto_coding_tier} | Confidence: {r.auto_coding_confidence} | Audit: {r.pre_submission_audit_score}")
        logger.info(f"  Time: {r.processing_time:.1f}s")
        if r.physician_documented_codes:
            logger.info(f"  Physician-documented codes: {[p.get('code') for p in r.physician_documented_codes]}")
        if r.missing_physician_codes:
            logger.info(f"  ⚠ MISSING physician codes: {[p.get('code') for p in r.missing_physician_codes]}")
        logger.info(f"  ICD-10-CM (billable):")
        for c in r.icd_codes:
            logger.info(f"    [{c.type:>9}] {c.code:>8} — {c.description[:55]}")
        if r.supporting_conditions:
            logger.info(f"  Supporting Conditions (advisory):")
            for c in r.supporting_conditions:
                logger.info(f"    [advisory] {c.code:>8} — {c.description[:55]}")
        logger.info(f"  CPT:")
        for c in r.cpt_codes:
            mods = ",".join(c.modifiers) if c.modifiers else ""
            logger.info(f"    {c.code:>8} {('['+mods+']') if mods else '':>10} — {c.description[:55]}")
        if r.hcpcs_codes:
            logger.info(f"  HCPCS:")
            for c in r.hcpcs_codes:
                mods = ",".join(c.modifiers) if c.modifiers else ""
                logger.info(f"    {c.code:>8} {('['+mods+']') if mods else '':>10} — {c.description[:55]}")
        if r.snomed_codes:
            logger.info(f"  SNOMED:")
            for c in r.snomed_codes:
                logger.info(f"    {c.concept_id:>12} — {c.description[:55]}")
        issues = r.validation_issues
        if issues:
            logger.info(f"  Validation ({len(issues)} issues):")
            for i in issues:
                logger.info(f"    [{i.severity}] {i.message[:65]}")
