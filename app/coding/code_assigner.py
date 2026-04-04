import json
from app.core.llm_client import chat_completion
from app.core.config import CODING_TEMPERATURE, CODING_MAX_TOKENS
from app.core.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# PASS 1 — ICD-10-CM Diagnosis Coding
# ---------------------------------------------------------------------------

ICD_SYSTEM_PROMPT = """You are an expert AAPC/AHIMA-certified medical coder (CPC, CIC, COC) specializing in PODIATRY.
Your ONLY task: assign ICD-10-CM diagnosis codes for this clinical note.

## MANDATORY PROCESS — Scan every section of the note:

### SECTION-BY-SECTION EXTRACTION
1. **CHIEF COMPLAINT** → primary reason for visit
2. **HPI** → conditions described (acute, chronic, history)
3. **PMH** → ALL chronic conditions listed (DM, HTN, hyperlipidemia, obesity, GERD, hypothyroidism, osteoporosis, neuropathy, etc.) — these MUST be coded if they affect management or are actively treated with medication
4. **PHYSICAL EXAM** → any findings that are separately codeable
5. **IMAGING** → findings that may need separate codes
6. **ASSESSMENT/DIAGNOSES** → every diagnosis listed here MUST be coded
7. **PLAN** → post-operative status, aftercare codes if applicable

### ICD-10-CM RULES
1. Code to HIGHEST level of specificity. Always include:
   - Laterality: right (.1), left (.2), bilateral (.3) when documented
   - Episode: initial (A), subsequent (D), sequela (S) when applicable
   - Type: type 1 vs type 2 diabetes, etc.
2. **SEQUENCING** — primary diagnosis = the condition chiefly responsible for the encounter:
   - Office visit → the chief complaint condition is primary
   - Post-op follow-up → aftercare Z-code is primary, underlying condition is secondary
   - Surgical procedure → the condition requiring surgery is primary
3. "Code also" and "Use additional code" instructions must be followed
4. Etiology codes before manifestation codes
5. For DM: use the combination code (E11.65 = T2DM with hyperglycemia), not separate codes
6. If a condition is listed in PMH AND the patient takes medication for it → code it as secondary
7. BMI codes (Z68.x) should accompany obesity codes (E66.x) — always pair them
8. Z-code rules: Z47.x = orthopedic aftercare, Z48.x = other surgical aftercare. Post-op ortho follow-up uses Z47.89.

### COMPLETENESS CHECK
Before finalizing, verify:
- Did I code EVERY diagnosis in the ASSESSMENT section?
- Did I code ALL PMH conditions that have active medications?
- Did I code the correct aftercare Z-code for post-op visits?
- Did I pair BMI + obesity codes when BMI is documented?
- Did I include laterality on EVERY lateralized condition?
- Is my primary diagnosis correctly sequenced?

## OUTPUT — Return valid JSON:
{
  "icd10_codes": [
    {
      "code": "M20.11",
      "description": "Hallux valgus (acquired), right foot",
      "type": "primary",
      "confidence": 0.95,
      "rationale": "Listed in assessment as primary complaint. Laterality: right foot documented.",
      "supporting_text": "exact quote from note that supports this code",
      "laterality": "RIGHT",
      "source_section": "ASSESSMENT"
    }
  ],
  "sequencing_reasoning": "Explanation of why codes are in this order"
}

CRITICAL: Return ONLY valid JSON. No markdown, no code blocks."""


# ---------------------------------------------------------------------------
# PASS 2 — CPT Procedure/E&M Coding
# ---------------------------------------------------------------------------

CPT_SYSTEM_PROMPT = """You are an expert AAPC-certified medical coder (CPC) specializing in PODIATRY CPT coding.
Your ONLY task: assign CPT codes for procedures, E/M services, and imaging performed on this date of service.

## MANDATORY PROCESS — Scan for ALL billable services:

### 1. E/M (Evaluation & Management)
Determine if an E/M code applies:
- New patient visit → 99202-99205
- Established patient visit → 99211-99215
- Calculate MDM level using the 2021+ framework:
  * PROBLEMS: 1=minimal, 2=low (1 chronic stable or 2+ self-limited), 3=moderate (1 chronic worsening or 2+ chronic stable), 4=high
  * DATA: 1=minimal, 2=limited (order/review tests), 3=moderate (independent interpretation of test), 4=extensive
  * RISK: 1=minimal, 2=low (OTC drugs, minor surgery), 3=moderate (prescription drugs, minor surgery with risk factors), 4=high (major surgery, hospitalization)
  * MDM = 2-of-3 highest. Low=2, Moderate=3, High=4.
- If BOTH E/M AND procedure on same day → E/M gets modifier -25 ONLY if there was a separately identifiable evaluation beyond the procedure decision

### 2. SURGICAL PROCEDURES
- Code ONLY procedures that were PERFORMED today (look for: "performed", "excised", "debrided", "avulsed", etc.)
- Do NOT code planned/scheduled/future procedures
- Match the procedure description EXACTLY to the CPT code description:
  * OPEN procedure → use open CPT code
  * ENDOSCOPIC procedure → use endoscopic CPT code
  * Check anatomical site: foot vs ankle vs toe
  * Partial vs complete procedures

### 3. IMAGING / RADIOLOGY
- Code X-rays if performed on this date of service
- Common podiatry imaging CPTs:
  * 73620 = foot X-ray, 2 views
  * 73630 = foot X-ray, complete (3+ views)
  * 73650 = calcaneus X-ray
  * 73660 = toe X-ray
  * 73600 = ankle X-ray, 2 views
  * 73610 = ankle X-ray, complete (3+ views)
- Look for phrases: "X-ray", "radiograph", "imaging performed", "views obtained"
- If the note says "X-ray (3 views)" → code 73630 (complete), NOT 73620 (2 views)

### 4. MODIFIERS — Apply ALL applicable:
- **-RT** (right side), **-LT** (left side) — on ALL lateralized procedures
- **-TA** = right great toe, **-T5** = left great toe
- **-T1** = right 2nd toe, **-T6** = left 2nd toe
- **-T2** = right 3rd toe, **-T7** = left 3rd toe
- **-T3** = right 4th toe, **-T8** = left 4th toe
- **-T4** = right 5th toe, **-T9** = left 5th toe
- **-25** = significant, separately identifiable E/M on same day as procedure
- **-59/XE/XS** = distinct procedural service (for NCCI edit bypass)
- **-50** = bilateral procedure

### COMPLETENESS CHECK
Before finalizing, verify:
- Did I code an E/M visit if the note documents evaluation (history, exam, MDM)?
- Did I code EVERY imaging study mentioned as performed?
- Did I code ALL surgical procedures documented as completed?
- Did I apply laterality modifiers (RT/LT) to every lateralized procedure?
- Did I apply toe modifiers (TA/T1-T9) for toe-specific procedures?
- Did I add modifier -25 to E/M if procedures were also performed?
- Did I match the surgical approach (open vs endoscopic) to the correct CPT?

## OUTPUT — Return valid JSON:
{
  "cpt_codes": [
    {
      "code": "11750",
      "description": "Excision of nail and nail matrix, partial or complete, for permanent removal",
      "confidence": 0.95,
      "modifiers": ["RT", "TA"],
      "modifier_reasoning": ["Right foot — RT", "Great toe — TA"],
      "source": "procedure",
      "mdm_details": {},
      "procedure_status": "completed",
      "laterality": "RT",
      "linked_diagnoses": ["L60.0", "L08.9"],
      "units": 1,
      "evidence_spans": ["exact quote from note"]
    }
  ],
  "em_level_reasoning": "Full MDM calculation if E/M coded"
}

CRITICAL: Return ONLY valid JSON. No markdown, no code blocks."""


# ---------------------------------------------------------------------------
# PASS 3 — HCPCS Level II + SNOMED
# ---------------------------------------------------------------------------

HCPCS_SNOMED_SYSTEM_PROMPT = """You are an expert medical coder. Assign HCPCS Level II codes and SNOMED CT codes.

## HCPCS RULES
1. ONLY code items that were DISPENSED/PROVIDED/APPLIED on this date of service
2. Look for action verbs: "applied", "dispensed", "fitted", "provided", "placed"
3. Do NOT code items that were "prescribed", "ordered", "recommended" but not given in office
4. Common podiatry HCPCS:
   - L3260 = surgical shoe
   - L4361 = walking boot (CAM boot, pneumatic boot)
   - L3000-L3030 = foot orthotics
   - A6021-A6024 = wound dressings (collagen, foam, etc.)
   - Q4100+ = skin substitute grafts
5. "Continue surgical shoe" for existing patient = do NOT code (already dispensed previously)
6. "Applied CAM walker boot" = code L4361 (dispensed on this date)

## SNOMED CT RULES
1. Assign SNOMED codes for ALL clinical findings, disorders, procedures, and body structures
2. Cover: primary diagnosis, secondary diagnoses, procedures performed, anatomical sites
3. Use standard SNOMED CT concept IDs

## OUTPUT — Return valid JSON:
{
  "hcpcs_codes": [
    {
      "code": "L4361",
      "description": "Walking boot, pneumatic",
      "confidence": 0.90,
      "modifiers": [],
      "linked_diagnoses": ["M72.2"],
      "evidence_spans": ["Applied compression dressing and CAM walker boot"],
      "rationale": "CAM walker boot applied in office post-operatively"
    }
  ],
  "snomed_codes": [
    {
      "concept_id": "202735001",
      "description": "Plantar fasciitis",
      "entity_text": "plantar fasciitis",
      "category": "disorder",
      "confidence": 0.90
    }
  ]
}

CRITICAL: Return ONLY valid JSON. No markdown, no code blocks."""


# ---------------------------------------------------------------------------
# PASS 4 — Self-Verification & Correction
# ---------------------------------------------------------------------------

VERIFICATION_SYSTEM_PROMPT = """You are a senior certified medical coding auditor (CCS, CPC-A). Your job is to AUDIT and CORRECT a set of medical codes assigned to a podiatry clinical note.

## ABSOLUTE RULES — READ BEFORE AUDITING

### RULE 1: PROTECTED CODES — NEVER REMOVE THESE
Codes derived from diagnoses EXPLICITLY LISTED in the physician's ASSESSMENT/DIAGNOSES section are PROTECTED. You CANNOT remove them. The physician documented them as distinct diagnoses — that is ground truth. You may CHANGE the specific code (e.g., make it more specific) but you CANNOT remove the diagnosis entirely.

You will receive a list of PROTECTED ANCHORS below. Every anchor MUST have a corresponding code in the final output.

### RULE 2: ASYMMETRIC THRESHOLDS
- To KEEP a code: you need reasonable support from the note
- To REMOVE a code: you need ABSOLUTE CERTAINTY it is unsupported AND it is NOT a protected anchor
- When in doubt: KEEP the code and flag for review rather than removing it

### RULE 3: REMOVAL REQUIRES QUOTED JUSTIFICATION
You may ONLY remove a code if:
1. It is NOT derived from the Assessment section, AND
2. It is NOT a PMH condition with active medication, AND
3. You provide the EXACT quote from the note proving it should not be coded

## AUDIT CHECKLIST

### Completeness Checks (ADD what's missing)
1. Is EVERY diagnosis in the ASSESSMENT section coded? (check each bullet point)
2. Are ALL PMH conditions with active medications coded?
3. Is an E/M code assigned if the note documents HPI + exam + MDM? (history, exam, and plan = E/M visit)
4. Are ALL imaging studies PERFORMED on this date coded?
5. Are ALL surgical procedures PERFORMED on this date coded?
6. Is modifier -25 on E/M when procedures/imaging also billed?
7. Are ALL laterality modifiers applied? (RT/LT)
8. Are toe modifiers applied? (TA for great toe, T1-T9 for specific toes)
9. For post-op visits: is the underlying condition coded alongside the aftercare Z-code?

### Over-Coding Checks (REMOVE only if certain and NOT protected)
1. Do NOT add symptom codes (pain, swelling) when a definitive diagnosis explains them
2. Do NOT add Z47/Z48 aftercare codes unless this is a follow-up for PRIOR surgery
3. Do NOT add BMI codes unless BMI is explicitly documented as a number
4. Do NOT add codes for conditions not documented anywhere in the note

### Quality Checks
1. Does every CPT have at least one linked ICD-10 diagnosis?
2. Is the primary diagnosis correctly sequenced?
3. Are there any NCCI edit conflicts between CPT codes?

## OUTPUT — Return valid JSON:
{
  "corrections_made": [
    {
      "type": "ADDED|REMOVED|CHANGED|RESEQUENCED|FLAGGED",
      "code": "73630",
      "reason": "X-ray (3 views) documented as performed but was not coded",
      "evidence": "exact quote from note"
    }
  ],
  "icd10_codes": [ ... corrected COMPLETE list ... ],
  "cpt_codes": [ ... corrected COMPLETE list ... ],
  "hcpcs_codes": [ ... corrected COMPLETE list ... ],
  "snomed_codes": [ ... corrected COMPLETE list ... ],
  "em_level_reasoning": "...",
  "audit_notes": "Summary of audit findings"
}

CRITICAL: Return ONLY valid JSON. No markdown, no code blocks."""


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def assign_codes(
    note_text: str,
    note_sections: dict,
    patient_metadata: dict,
    entities: list[dict],
    rag_candidates: dict,
    vision_context: dict | None = None,
) -> tuple[dict, dict]:
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    note_context = _build_note_context(note_sections, patient_metadata)
    entity_summary = _format_entities(entities)
    candidate_summary = _format_candidates(rag_candidates)
    vision_block = _format_vision_context(vision_context) if vision_context else ""

    # --- PASS 1: ICD-10-CM ---
    logger.info("  Pass 1/4: ICD-10-CM diagnosis coding...")
    icd_prompt = f"""{note_context}
{vision_block}

## EXTRACTED CLINICAL ENTITIES
{entity_summary}

## ICD-10-CM CANDIDATE CODES (from official FY2026 database via semantic search)
{_format_candidates_for_system(rag_candidates, 'icd10')}

Select the correct ICD-10-CM codes. PREFER candidates from the list above.
Code EVERY diagnosis in the ASSESSMENT section AND every PMH condition with active medication.
Do NOT code symptoms (pain, swelling) when a definitive diagnosis explains them.
Do NOT add Z47/Z48 aftercare codes unless this IS a follow-up visit for a PRIOR surgery."""

    icd_raw, usage = chat_completion(ICD_SYSTEM_PROMPT, icd_prompt, temperature=CODING_TEMPERATURE, max_tokens=2500)
    _add_usage(total_usage, usage)
    icd_result = _safe_parse(icd_raw, "icd10_codes")
    logger.info(f"    → {len(icd_result.get('icd10_codes', []))} ICD-10-CM codes")

    # --- PASS 2: CPT ---
    logger.info("  Pass 2/4: CPT procedure/E&M/imaging coding...")
    icd_summary = _summarize_icd(icd_result.get("icd10_codes", []))
    cpt_prompt = f"""{note_context}
{vision_block}

## ASSIGNED ICD-10-CM CODES (from Pass 1)
{icd_summary}

## EXTRACTED CLINICAL ENTITIES
{entity_summary}

## CPT CANDIDATE CODES (from official database via semantic search)
{_format_candidates_for_system(rag_candidates, 'cpt')}

Select the correct CPT codes. PREFER candidates from the list above.
Link each CPT to supporting ICD-10-CM codes from the list above.
Code ALL performed procedures, ALL imaging, and E/M if applicable."""

    cpt_raw, usage = chat_completion(CPT_SYSTEM_PROMPT, cpt_prompt, temperature=CODING_TEMPERATURE, max_tokens=2500)
    _add_usage(total_usage, usage)
    cpt_result = _safe_parse(cpt_raw, "cpt_codes")
    logger.info(f"    → {len(cpt_result.get('cpt_codes', []))} CPT codes")

    # --- PASS 3: HCPCS + SNOMED ---
    logger.info("  Pass 3/4: HCPCS + SNOMED coding...")
    hcpcs_prompt = f"""{note_context}
{vision_block}

## ASSIGNED ICD-10-CM CODES
{icd_summary}

## EXTRACTED CLINICAL ENTITIES
{entity_summary}

## HCPCS CANDIDATE CODES (from official database via semantic search)
{_format_candidates_for_system(rag_candidates, 'hcpcs')}

Assign HCPCS codes for supplies/DME dispensed today and SNOMED codes for all clinical concepts."""

    hcpcs_raw, usage = chat_completion(HCPCS_SNOMED_SYSTEM_PROMPT, hcpcs_prompt, temperature=CODING_TEMPERATURE, max_tokens=2500)
    _add_usage(total_usage, usage)
    hcpcs_result = _safe_parse(hcpcs_raw, "hcpcs_codes")
    logger.info(f"    → {len(hcpcs_result.get('hcpcs_codes', []))} HCPCS, {len(hcpcs_result.get('snomed_codes', []))} SNOMED")

    # --- PASS 4: Constrained Self-Verification (Anchor-and-Audit) ---
    logger.info("  Pass 4/4: Constrained verification (anchor-and-audit)...")
    combined = {
        "icd10_codes": icd_result.get("icd10_codes", []),
        "cpt_codes": cpt_result.get("cpt_codes", []),
        "hcpcs_codes": hcpcs_result.get("hcpcs_codes", []),
        "snomed_codes": hcpcs_result.get("snomed_codes", []),
        "em_level_reasoning": cpt_result.get("em_level_reasoning", ""),
    }

    # Build protected anchors from Assessment section + PMH
    assessment_text = note_sections.get("assessment_diagnoses", "")
    pmh_text = note_sections.get("pmh_medications_allergies", "")
    anchor_block = _build_anchor_block(assessment_text, pmh_text, vision_context)

    verify_prompt = f"""{note_context}
{vision_block}

{anchor_block}

## EXTRACTED CLINICAL ENTITIES
{entity_summary}

## CURRENTLY ASSIGNED CODES (to audit)
{json.dumps(combined, indent=2)}

## RAG CANDIDATE CODES (for reference — verified to exist in official databases)
### ICD-10-CM Candidates
{_format_candidates_for_system(rag_candidates, 'icd10')}

### CPT Candidates
{_format_candidates_for_system(rag_candidates, 'cpt')}

### HCPCS Candidates
{_format_candidates_for_system(rag_candidates, 'hcpcs')}

## AUDIT INSTRUCTIONS
1. Check every PROTECTED ANCHOR above has a corresponding code. If not, ADD it.
2. Check completeness: E/M visit? Imaging? Procedures? Supplies? All coded?
3. Only REMOVE codes if they have ZERO documentation support AND are NOT protected anchors.
4. When in doubt, KEEP the code and flag with needs_review=true.
5. Return the COMPLETE corrected code set."""

    verify_raw, usage = chat_completion(VERIFICATION_SYSTEM_PROMPT, verify_prompt, temperature=CODING_TEMPERATURE, max_tokens=4096)
    _add_usage(total_usage, usage)
    verified = _safe_parse(verify_raw, "icd10_codes")

    corrections = verified.get("corrections_made", [])
    if corrections:
        logger.info(f"    → {len(corrections)} corrections made:")
        for c in corrections:
            logger.info(f"      [{c.get('type', '?')}] {c.get('code', '?')}: {c.get('reason', '')[:70]}")
    else:
        logger.info("    → No corrections needed")

    final_result = {
        "icd10_codes": verified.get("icd10_codes", combined["icd10_codes"]),
        "cpt_codes": verified.get("cpt_codes", combined["cpt_codes"]),
        "hcpcs_codes": verified.get("hcpcs_codes", combined["hcpcs_codes"]),
        "snomed_codes": verified.get("snomed_codes", combined["snomed_codes"]),
        "em_level_reasoning": verified.get("em_level_reasoning", combined["em_level_reasoning"]),
        "corrections_made": corrections,
        "audit_notes": verified.get("audit_notes", ""),
    }

    logger.info(
        f"  Final: {len(final_result['icd10_codes'])} ICD, "
        f"{len(final_result['cpt_codes'])} CPT, "
        f"{len(final_result['hcpcs_codes'])} HCPCS, "
        f"{len(final_result['snomed_codes'])} SNOMED "
        f"(total tokens: {total_usage['total_tokens']})"
    )

    return final_result, total_usage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_note_context(sections: dict, metadata: dict) -> str:
    return f"""## PATIENT METADATA
- Patient: {metadata.get('patient_name', 'Unknown')}
- DOB: {metadata.get('date_of_birth', 'Unknown')}
- DOS: {metadata.get('date_of_service', 'Unknown')}
- Provider: {metadata.get('provider', 'Unknown')}
- Insurance: {metadata.get('insurance', 'Unknown')}
- Note Type: {metadata.get('note_type', 'Unknown')}

## CLINICAL NOTE

### CHIEF COMPLAINT
{sections.get('chief_complaint', 'N/A')}

### HISTORY OF PRESENT ILLNESS
{sections.get('hpi', 'N/A')}

### PAST MEDICAL HISTORY / MEDICATIONS / ALLERGIES
{sections.get('pmh_medications_allergies', 'N/A')}

### PHYSICAL EXAMINATION
{sections.get('physical_examination', 'N/A')}

### IMAGING / DIAGNOSTICS
{sections.get('imaging_diagnostics', 'N/A')}

### ASSESSMENT / DIAGNOSES
{sections.get('assessment_diagnoses', 'N/A')}

### PLAN
{sections.get('plan', 'N/A')}"""


def _format_entities(entities: list[dict]) -> str:
    if not entities:
        return "No entities extracted."
    lines = []
    for e in entities:
        lat = f" [{e.get('laterality', '')}]" if e.get("laterality") else ""
        spec = f" — {e.get('specificity', '')}" if e.get("specificity") else ""
        lines.append(
            f"- [{e.get('category', '?').upper():>14}] {e.get('clinical_term', '')}{lat}{spec} "
            f"(section: {e.get('source_section', '?')}, text: \"{e.get('text', '')}\")"
        )
    return "\n".join(lines)


def _format_candidates_for_system(rag_candidates: dict, system: str) -> str:
    candidates = rag_candidates.get(system, [])
    if not candidates:
        return "No candidates retrieved."
    lines = []
    for c in candidates[:25]:
        score = c.get("similarity_score", 0)
        code = c.get("code", "")
        desc = c.get("description", "") or c.get("long_description", "") or c.get("short_description", "")
        lines.append(f"  {code} (relevance: {score:.3f}) — {desc[:150]}")
    return "\n".join(lines)


def _format_candidates(rag_candidates: dict) -> str:
    parts = []
    for system_name, candidates in rag_candidates.items():
        if not candidates:
            continue
        label = system_name.upper().replace("ICD10", "ICD-10-CM")
        parts.append(f"\n### {label} Candidates")
        for c in candidates[:20]:
            score = c.get("similarity_score", 0)
            code = c.get("code", "")
            desc = c.get("description", "") or c.get("long_description", "") or c.get("short_description", "")
            parts.append(f"  {code} (score: {score:.3f}) — {desc[:120]}")
    return "\n".join(parts) if parts else "No candidates retrieved."


def _summarize_icd(icd_codes: list[dict]) -> str:
    if not icd_codes:
        return "No ICD-10-CM codes assigned."
    lines = []
    for c in icd_codes:
        lines.append(f"  {c.get('code', '?')} [{c.get('type', '?')}] — {c.get('description', '')[:80]}")
    return "\n".join(lines)


def _safe_parse(raw: str, required_key: str) -> dict:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.error(f"Failed to parse LLM response for {required_key}")
        return {required_key: []}


def _build_anchor_block(assessment_text: str, pmh_text: str, vision_ctx: dict | None) -> str:
    """Build the PROTECTED ANCHORS section from Assessment + PMH + Vision context."""
    lines = [
        "## PROTECTED ANCHORS — These MUST have corresponding codes in the final output",
        "",
        "### Assessment Section Diagnoses (NEVER remove these — physician documented them)",
    ]

    if assessment_text:
        for line in assessment_text.split("\n"):
            cleaned = line.strip().lstrip("•·-–—0123456789.) ").strip()
            if cleaned and len(cleaned) > 3:
                lines.append(f'  - ANCHOR: "{cleaned}"')
    else:
        lines.append("  (no assessment text available)")

    lines.append("")
    lines.append("### PMH Conditions with Active Medications (must be coded as secondary)")

    if pmh_text:
        lines.append(f'  Source text: "{pmh_text[:300]}"')
        lines.append("  Every condition in PMH that has a corresponding medication MUST be coded.")

    if vision_ctx:
        procs = vision_ctx.get("procedures_performed_today", [])
        imgs = vision_ctx.get("imaging_performed_today", [])
        sups = vision_ctx.get("supplies_dispensed_today", [])

        if procs:
            lines.append("")
            lines.append("### Procedures Performed Today (must have CPT codes)")
            for p in procs:
                lines.append(f'  - ANCHOR: "{p}"')

        if imgs:
            lines.append("")
            lines.append("### Imaging Performed Today (must have CPT codes)")
            for i in imgs:
                lines.append(f'  - ANCHOR: "{i}"')

        if sups:
            lines.append("")
            lines.append("### Supplies Dispensed Today (must have HCPCS codes)")
            for s in sups:
                lines.append(f'  - ANCHOR: "{s}"')

        note_cat = vision_ctx.get("note_category", "")
        if note_cat and ("visit" in note_cat or "followup" in note_cat or "follow_up" in note_cat or "post_op" in note_cat):
            lines.append("")
            lines.append("### E/M Visit Detection")
            lines.append(f"  Note category: {note_cat}")
            lines.append("  If this is a patient visit (not surgery-only), an E/M code MUST be assigned")
            lines.append("  if the note documents history + examination + medical decision making.")

    return "\n".join(lines)


def _format_vision_context(ctx: dict) -> str:
    if not ctx:
        return ""
    parts = ["## VISION EXTRACTION CONTEXT (from intelligent PDF reading)"]
    parts.append(f"- Note category: {ctx.get('note_category', 'unknown')}")
    procs = ctx.get("procedures_performed_today", [])
    parts.append(f"- Procedures PERFORMED today: {procs if procs else 'NONE'}")
    imgs = ctx.get("imaging_performed_today", [])
    parts.append(f"- Imaging PERFORMED today: {imgs if imgs else 'NONE'}")
    sups = ctx.get("supplies_dispensed_today", [])
    parts.append(f"- Supplies DISPENSED today: {sups if sups else 'NONE'}")
    parts.append("NOTE: Only code items listed above as performed/dispensed. This is ground truth from reading the PDF.")
    return "\n".join(parts)


def _add_usage(total: dict, new: dict):
    for k in total:
        total[k] += new.get(k, 0)
