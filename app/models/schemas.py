from pydantic import BaseModel, Field


class ModifierClaim(BaseModel):
    """One structured claim about a single modifier's presence on a code —
    replaces free-text modifier_reasoning strings that had to be parsed
    with regex heuristics (cue words, negation windows) to guess whether a
    sentence meant "this modifier is applied" or "this modifier is not
    applied." That guessing repeatedly failed on new phrasings the
    heuristics hadn't seen (e.g. "removed"/"unnecessary" wrongly read as
    positive; later "retained" wrongly read as negative; an unrelated "no"
    elsewhere in the sentence wrongly vetoing a real claim) — an
    open-ended problem with no ceiling, because natural language can
    express the same polarity in unlimited ways. `status` makes the
    polarity an explicit field instead of something inferred from prose,
    eliminating the entire class of parsing bugs at the source.
    """
    modifier: str = Field(description="The modifier code this claim is about, e.g. '25', 'RT', '59'")
    status: str = Field(description="'applied' (this modifier IS on the code) or 'not_applicable' "
                                     "(this modifier is NOT on the code / was considered and rejected)")
    reason: str = Field(default="", description="Free-text clinical justification — explanatory only, "
                                                 "never used to infer status")


class SupportingCondition(BaseModel):
    """PMH-sourced or advisory ICD codes — informational, not placed on the billable claim."""
    code: str
    description: str = ""
    type: str = "advisory"
    confidence: float = 0.0
    rationale: str = ""
    supporting_text: str = ""
    source_section: str = ""
    billable_tier: str = "advisory"
    needs_review: bool = True
    review_reason: str | None = None


class ClinicalEntity(BaseModel):
    text: str = Field(description="Exact text span from clinical note")
    category: str = Field(description="diagnosis|procedure|medication|supply|finding|body_structure|allergy")
    clinical_term: str = Field(description="Normalized clinical term")
    laterality: str | None = Field(default=None, description="RIGHT|LEFT|BILATERAL")
    specificity: str | None = Field(default=None, description="Additional specificity details")
    source_section: str = Field(description="Section of note: CC|HPI|PE|IMAGING|ASSESSMENT|PLAN|PMH")
    negated: bool = Field(default=False, description="Whether the entity is negated")
    ner_source: str = Field(default="llm", description="gliner_confirmed|llm_only|llm")
    ner_confidence: float = Field(
        default=0.7,
        description="GLiNER score when confirmed; conservative LLM-only default otherwise",
    )
    # Deterministic terminology provenance is populated after NER.  The raw
    # LLM span and clinical_term remain untouched; normalized_text is a
    # retrieval aid whose accepted substitutions are traceable to the
    # versioned terminology registry below.
    source_span: dict = Field(
        default_factory=dict,
        description="Verified section/document offsets for the verbatim entity span",
    )
    terminology_resolutions: list[dict] = Field(
        default_factory=list,
        description="Per-abbreviation candidates, decision, confidence, and provenance",
    )
    normalized_text: str = Field(
        default="", description="Raw entity text with accepted terminology expansions"
    )
    retrieval_terms: list[str] = Field(
        default_factory=list,
        description="Deduplicated raw, model-normalized, and deterministic query forms",
    )
    normalization_status: str = Field(
        default="not_evaluated",
        description="accepted|ambiguous|unresolved|not_applicable|not_evaluated",
    )
    normalization_confidence: float = Field(default=0.0)
    normalization_registry_version: str = Field(default="")
    normalization_registry_sha256: str = Field(default="")


class ICDCode(BaseModel):
    code: str
    description: str = ""
    type: str = "secondary"
    confidence: float = 0.0
    rationale: str = ""
    supporting_text: str = ""
    laterality: str | None = None
    source_section: str = ""
    s3_validated: bool = False
    needs_review: bool = False
    review_reason: str | None = None
    code_source: str = "ai_inferred"  # physician_documented|ai_confirmed|ai_inferred|ai_replaced_physician
    physician_code_note: str | None = None  # original physician code if AI replaced it
    evidence_spans: list[str] = Field(default_factory=list)
    source_record_ids: list[str] = Field(default_factory=list)
    source_effective_from: str | None = None
    source_effective_to: str | None = None
    source_temporal_authority: bool = False


class CPTCode(BaseModel):
    code: str
    description: str = ""
    confidence: float = 0.0
    modifiers: list[str] = Field(default_factory=list)
    modifier_reasoning: list[ModifierClaim] = Field(default_factory=list)
    source: str = ""
    mdm_details: dict = Field(default_factory=dict)
    procedure_status: str = "completed"
    laterality: str | None = None
    linked_diagnoses: list[str] = Field(default_factory=list)
    units: int = 1
    evidence_spans: list[str] = Field(default_factory=list)
    ama_validated: bool = False
    mue_validated: bool = False
    mue_limit: int | None = None
    code_source: str = "ai_inferred"
    physician_code_note: str | None = None
    source_record_ids: list[str] = Field(default_factory=list)
    source_effective_from: str | None = None
    source_effective_to: str | None = None
    source_temporal_authority: bool = False


class HCPCSCode(BaseModel):
    code: str
    description: str = ""
    confidence: float = 0.0
    modifiers: list[str] = Field(default_factory=list)
    # Was missing from this model entirely — _to_hcpcs() filters raw LLM
    # output to only Pydantic-declared fields, so any modifier_reasoning an
    # HCPCS entry carried (e.g. RT/LT laterality justification) was
    # silently dropped from the final persisted output even though
    # validator.py's consistency check already reads it for both cpt and
    # hcpcs entries.
    modifier_reasoning: list[ModifierClaim] = Field(default_factory=list)
    units: int = 1
    linked_diagnoses: list[str] = Field(default_factory=list)
    rationale: str = ""
    supporting_text: str = ""
    needs_review: bool = False
    review_reason: str | None = None
    code_source: str = "ai_inferred"
    physician_code_note: str | None = None
    evidence_spans: list[str] = Field(default_factory=list)
    source_record_ids: list[str] = Field(default_factory=list)
    source_effective_from: str | None = None
    source_effective_to: str | None = None
    source_temporal_authority: bool = False


class SNOMEDCode(BaseModel):
    concept_id: str
    description: str = ""
    entity_text: str = ""
    category: str = ""
    confidence: float = 0.0
    is_root_concept: bool = False


class ValidationIssue(BaseModel):
    severity: str = "INFO"
    code: str = ""
    category: str = ""
    message: str = ""
    recommendation: str = ""
    denial_risk: str | None = None
    # WHICH assertion this issue makes about `code` — the same dimension
    # compliance findings carry (Finding.clause). One category can host
    # several distinct assertions about one code; advisory suppression must
    # be able to retire exactly one of them without retiring its siblings
    # (the routine_00003 inversion). "" means untagged: matched only by an
    # unscoped suppression directive, never by a clause-scoped one. The
    # migration to universal tagging is ratcheted by
    # tests/check_clause_coverage.py.
    clause: str = ""


class DocumentationAudit(BaseModel):
    audit_entries: list[dict] = Field(default_factory=list)
    total_codes: int = 0
    fully_supported: int = 0
    partially_supported: int = 0
    unsupported: int = 0
    codes_with_gaps: int = 0
    documentation_score: float = 1.0


class EncounterIntegrity(BaseModel):
    encounter_issues: list[dict] = Field(default_factory=list)
    error_count: int = 0
    warning_count: int = 0


class CodingResult(BaseModel):
    model_config = {"protected_namespaces": ()}

    document_id: str
    timestamp: str = ""
    success: bool = True
    processing_time: float = 0.0
    patient_metadata: dict = Field(default_factory=dict)
    note_sections: dict = Field(default_factory=dict)
    icd_codes: list[ICDCode] = Field(default_factory=list)
    supporting_conditions: list[SupportingCondition] = Field(default_factory=list)
    cpt_codes: list[CPTCode] = Field(default_factory=list)
    hcpcs_codes: list[HCPCSCode] = Field(default_factory=list)
    snomed_codes: list[SNOMEDCode] = Field(default_factory=list)
    em_level_reasoning: str = ""
    error_message: str | None = None
    validation_issues: list[ValidationIssue] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    encounter_integrity: EncounterIntegrity = Field(default_factory=EncounterIntegrity)
    documentation_audit: DocumentationAudit = Field(default_factory=DocumentationAudit)
    pre_submission_audit_findings: list[ValidationIssue] = Field(default_factory=list)
    pre_submission_audit_score: float = 0.0
    auto_coding_tier: str = "REVIEW"
    auto_coding_confidence: float = 0.0
    auto_coding_review_reasons: list[str] = Field(default_factory=list)
    auto_coding_summary: str = ""
    rag_context: dict = Field(default_factory=dict)
    model_source: str = ""
    model_execution: dict = Field(default_factory=dict)
    api_usage: dict = Field(default_factory=dict)
    physician_documented_codes: list[dict] = Field(default_factory=list)
    missing_physician_codes: list[dict] = Field(default_factory=list)
    cached_result: bool = False
    ner_entities: list[dict] = Field(default_factory=list)
    # Whole-note/entity terminology audit.  This is release-bound evidence:
    # an unresolved affirmed shorthand term blocks autonomy only when the
    # registry marks its context as capable of changing a billed line.
    terminology_normalization: dict = Field(default_factory=dict)
    clinical_facts: dict = Field(default_factory=dict)
    # The operative note's documented procedures (vision extraction's
    # procedures_performed_today). Persisted on the record — not just used
    # transiently for the coding prompts — because the completeness
    # invariant (CodingValidator._check_procedure_completeness) needs it to
    # re-run when the claim is RE-validated on replay/reconciliation. Every
    # replay produces the final shipped claim, so without this the
    # completeness check would no-op on exactly the claim that ships (the
    # replayer reconstructs claim context from the stored payload the same
    # way it already re-reads note_category). Kept as documented (list of
    # strings); never medical codes.
    procedures_performed_today: list = Field(default_factory=list)
    # Every claim-mutating action the deterministic layers took (auto-
    # corrections + suppressions) — the input to the clinical-correctness
    # audit that verifies each layer decision against the note/authorities.
    material_corrections: list[dict] = Field(default_factory=list)
    # Scrub-advisory suppressions recorded by declarative validator rules
    # (CodingValidator.suppress_scrub_advisory): instructions the claim
    # scrubber honors for WARN findings only, each carrying the rule id
    # and authority so the suppressed advisory leaves a rule-decision
    # audit trail instead of silently vanishing.
    scrub_advisory_suppressions: list[dict] = Field(default_factory=list)
    # The clinical-correctness audit's verdict block (tools/
    # clinical_auditor.py). A scrub-CLEAN claim with interpretive
    # corrections stays at REVIEW until this says "upheld".
    clinical_audit: dict = Field(default_factory=dict)
    # 13-filter compliance scrubber result (disposition CLEAN/REVIEW + findings)
    claim_scrub: dict = Field(default_factory=dict)
    # AUTHORITATIVE verdict — driven by the 13-filter scrubber (the clean-claim gate).
    # CLEAN = billable; REVIEW = routed to review with reasons. This is the single
    # source of truth; auto_coding_tier is derived from it for backward compatibility.
    final_disposition: str = "REVIEW"
    final_summary: str = ""
    # Release-boundary artifacts.  CLEAN remains an internal validation
    # result; only an immutable AUTO_READY certificate authorizes autonomous
    # registry ingest/submission.
    candidate_claim: dict = Field(default_factory=dict)
    mutation_ledger: list[dict] = Field(default_factory=list)
    note_integrity: dict = Field(default_factory=dict)
    authoritative_source_manifest: dict = Field(default_factory=dict)
    claim_readiness_certificate: dict = Field(default_factory=dict)
