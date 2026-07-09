from pydantic import BaseModel, Field


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
    ner_confidence: float = Field(default=1.0, description="GLiNER confidence if confirmed")


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


class CPTCode(BaseModel):
    code: str
    description: str = ""
    confidence: float = 0.0
    modifiers: list[str] = Field(default_factory=list)
    modifier_reasoning: list[str] = Field(default_factory=list)
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


class HCPCSCode(BaseModel):
    code: str
    description: str = ""
    confidence: float = 0.0
    modifiers: list[str] = Field(default_factory=list)
    units: int = 1
    linked_diagnoses: list[str] = Field(default_factory=list)
    rationale: str = ""
    supporting_text: str = ""
    needs_review: bool = False
    review_reason: str | None = None
    code_source: str = "ai_inferred"
    physician_code_note: str | None = None


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
    api_usage: dict = Field(default_factory=dict)
    physician_documented_codes: list[dict] = Field(default_factory=list)
    missing_physician_codes: list[dict] = Field(default_factory=list)
    cached_result: bool = False
    ner_entities: list[dict] = Field(default_factory=list)
    # 12-filter compliance scrubber result (disposition CLEAN/REVIEW + findings)
    claim_scrub: dict = Field(default_factory=dict)
    # AUTHORITATIVE verdict — driven by the 12-filter scrubber (the clean-claim gate).
    # CLEAN = billable; REVIEW = routed to review with reasons. This is the single
    # source of truth; auto_coding_tier is derived from it for backward compatibility.
    final_disposition: str = "REVIEW"
    final_summary: str = ""
