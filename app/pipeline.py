import time
from pathlib import Path
from datetime import datetime

from app.core.logger import get_logger
from app.core import cache as result_cache
from app.core.config import LLM_PROVIDER
from app.ingestion.pdf_parser import extract_from_pdf
from app.ner.entity_extractor import extract_entities
from app.rag.retriever import CandidateRetriever
from app.rag.vector_store import MedicalCodeVectorStore
from app.rag.code_reference import CodeReferenceDB
from app.coding.code_assigner import assign_codes
from app.validation.validator import CodingValidator
from app.models.schemas import CodingResult

logger = get_logger(__name__)


class MedicalCodingPipeline:
    """Full Vision → NER → RAG → Multi-Pass LLM → Validation pipeline."""

    def __init__(self):
        self.vector_store = MedicalCodeVectorStore()
        self.ref_db = CodeReferenceDB()
        self.retriever: CandidateRetriever | None = None
        self.validator: CodingValidator | None = None
        self._initialized = False

    def initialize(self, force_rebuild_index: bool = False) -> None:
        logger.info("=" * 70)
        logger.info("INITIALIZING MEDICAL CODING PIPELINE")
        logger.info("=" * 70)

        logger.info("Loading code reference database...")
        self.ref_db.load_all()

        logger.info("Building/loading Qdrant hybrid vector store...")
        self.vector_store.build_or_load(force_rebuild=force_rebuild_index)

        self.retriever = CandidateRetriever(self.vector_store)
        self.validator = CodingValidator(self.ref_db)

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
                    r = CodingResult(**cached)
                    r.cached_result = True
                    self._print_summary(r)
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
        logger.info(f"  Found {len(entities)} entities")
        for e in entities:
            logger.info(f"    [{e.category:>14}] {e.clinical_term} {'['+e.laterality+']' if e.laterality else ''}")

        # Step 3: RAG — retrieve candidate codes
        logger.info("[3/5] Retrieving candidate codes (RAG/Qdrant hybrid)...")
        entity_candidates = self.retriever.retrieve_for_entities(entities)
        note_candidates = self.retriever.retrieve_for_full_note(sections)

        merged = self._merge_candidates(entity_candidates, note_candidates)
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
        )

        # Step 5: Validation
        logger.info("[5/5] Validating codes...")
        plan_text = sections.get("plan", "")
        full_text = sections.get("full_text", "")
        validation = self.validator.validate(
            coding_result,
            prior_surgery_info=prior_surgery_info,
            note_plan_text=plan_text,
            note_full_text=full_text,
            physician_documented_codes=physician_documented_codes,
        )

        elapsed = time.time() - start

        result = CodingResult(
            document_id=pdf_path.stem,
            timestamp=datetime.now().isoformat(),
            success=True,
            processing_time=round(elapsed, 2),
            patient_metadata=metadata,
            note_sections={k: v[:200] + "..." if len(v) > 200 else v
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
                "corrections_made": coding_result.get("corrections_made", []),
                "audit_notes": coding_result.get("audit_notes", ""),
                "vision_context": vision_context,
                "prior_surgery_info": prior_surgery_info,
            },
            model_source=LLM_PROVIDER,
            api_usage=usage,
            physician_documented_codes=physician_documented_codes,
            missing_physician_codes=coding_result.get("missing_physician_codes", []),
            ner_entities=entity_dicts,
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

        # Fix 4 — Store to cache (only on success, so failed runs don't get cached)
        if use_cache and result.success:
            result_cache.store(pdf_path, result.model_dump())

        self._print_summary(result)
        return result

    def _merge_candidates(self, entity_cands: dict, note_cands: dict) -> dict:
        merged = {"icd10": [], "cpt": [], "hcpcs": []}
        seen = {"icd10": set(), "cpt": set(), "hcpcs": set()}

        for key, data in entity_cands.items():
            for cs, cands in data.get("candidates", {}).items():
                for c in cands:
                    code = c.get("code", "")
                    if code not in seen[cs]:
                        seen[cs].add(code)
                        merged[cs].append(c)

        for cs, cands in note_cands.items():
            for c in cands:
                code = c.get("code", "")
                if code not in seen[cs]:
                    seen[cs].add(code)
                    merged[cs].append(c)

        for cs in merged:
            merged[cs].sort(key=lambda x: x.get("similarity_score", 0), reverse=True)

        return merged

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
        return CPTCode(**data)

    def _to_hcpcs(self, raw: dict):
        from app.models.schemas import HCPCSCode
        return HCPCSCode(**{k: v for k, v in raw.items() if k in HCPCSCode.model_fields})

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
