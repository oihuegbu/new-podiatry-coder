import json
from app.core.llm_client import chat_completion
from app.core.config import CODING_TEMPERATURE, CODING_MAX_TOKENS
from app.core.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# PASS 1 — ICD-10-CM Diagnosis Coding
# ---------------------------------------------------------------------------

ICD_SYSTEM_PROMPT = """You are an expert AAPC/AHIMA-certified medical coder (CPC, CIC, COC) specializing in PODIATRY.
Your ONLY task: assign ICD-10-CM diagnosis codes for this clinical note using FY2026 guidelines.

## SECTION-BY-SECTION EXTRACTION — scan EVERY section:
1. CHIEF COMPLAINT → primary reason for visit
2. HPI → conditions described (acute, chronic, history)
3. PMH → ALL chronic conditions listed — code if they have active medication OR influenced today's treatment plan
4. PHYSICAL EXAM → separately codeable findings
5. IMAGING → findings needing separate codes
6. ASSESSMENT/DIAGNOSES → every listed diagnosis MUST be coded
7. PLAN → aftercare codes for post-op visits, active conditions managed today

## ICD-10-CM CODING RULES

### Specificity
- Code to HIGHEST specificity: laterality (right=.1, left=.2, bilateral=.3), type, episode
- Diabetes: ALWAYS use combination codes (E11.40, E11.621, E10.40, etc.) — NEVER code DM generic + complication separately
- Do NOT code both E11.9 AND E11.40 for the same patient — E11.40 already captures the diabetes

### Ulcer Anatomic Site — CRITICAL
- L97.1xx = pressure ulcer heel/ankle
- L97.4xx = non-pressure ulcer of HEEL and MIDFOOT (calcaneus, dorsum of foot, midfoot)
- L97.5xx = non-pressure ulcer of OTHER PART of foot (metatarsal heads, ball of foot, toes)
- L97.3xx = non-pressure ulcer of ankle
- The 5th metatarsal head, ball of foot, and lesser metatarsal heads → L97.5xx NOT L97.4xx
- Plantar surface 5th MTH → L97.522 (left), L97.521 (right)

### Sequencing
- Primary = condition chiefly responsible for the encounter
- Post-op follow-up: Z47.xx/Z48.xx aftercare is primary, underlying condition is secondary
- DM with foot ulcer: E10.621/E11.621 FIRST, then ulcer site code (L97.xxx) second
- Manifestation codes NEVER come before etiology codes
- Post-op aftercare Z-codes: Z48.89 = encounter for other specified surgical aftercare (most post-op visits WITHOUT device removal); Z47.2 = ONLY when a device (pin, wire, plate) is ACTUALLY REMOVED at this visit

### PMH Coding Rules
- Code PMH conditions IF they have an active medication AND/OR influenced treatment decisions today
- DO NOT code Z79.84 (long-term hypoglycemic drug), Z79.899 (other long-term drug) for routine outpatient podiatry visits — underlying conditions (E11.x, I10, E78.5) are sufficient
- CKD: code N18.x if it affected today's treatment plan (e.g., drug withheld due to renal function)
- CAD, Hypertension, Hyperlipidemia: code if actively managed with medication listed in the note

### Do NOT code:
- Symptom codes (pain, swelling, erythema) when a definitive diagnosis explains them
- Z47/Z48 aftercare UNLESS this is explicitly a post-operative follow-up visit
- BMI codes unless BMI is documented as a number in the note
- Z79.84 or Z79.899 on outpatient podiatry encounters

## OUTPUT — Return valid JSON only:
{
  "icd10_codes": [
    {
      "code": "M20.11",
      "description": "Hallux valgus (acquired), right foot",
      "type": "primary",
      "confidence": 0.95,
      "rationale": "Listed in assessment. Laterality: right foot documented.",
      "supporting_text": "exact quote from note",
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
Your ONLY task: assign CPT codes for ALL services performed on this date of service.

## 1. E/M (Evaluation & Management)

### New vs Established Patient — CRITICAL
- NEW patient = not seen by this provider OR any provider of the same specialty/group in the PAST 3 YEARS
- Look at the NOTE TYPE field: "NEW PATIENT" → 99202–99205; "ESTABLISHED PATIENT" → 99211–99215
- NEVER assign 99211-99215 to a new patient visit. NEVER assign 99202-99205 to an established patient.

### AMA 2021 MDM Three-Axis Table (2-of-3 rule)
MDM level = whichever level is met by AT LEAST 2 of 3 axes:

**PROBLEMS axis:**
- Minimal (1): 1 self-limited/minor problem
- Low (2): 2+ self-limited problems; OR 1 stable chronic illness; OR 1 acute uncomplicated illness
- Moderate (3): 1+ chronic illness with exacerbation/progression; OR 2+ stable chronic illnesses; OR new problem with uncertain prognosis; OR acute illness with systemic symptoms
- High (4): Chronic illness with severe exacerbation threatening life or bodily function; OR acute or chronic illness requiring hospitalization

**DATA axis:**
- Minimal (1): No data reviewed or ordered
- Limited (2): Ordered and/or reviewed tests, documents from external source; OR independent interpretation of test ordered by another provider
- Moderate (3): Independent interpretation of test performed by another provider; OR discussion of test with ordering provider; OR independent historian; OR independent review of external records
- Extensive (4): Independent interpretation of test AND discussion of management with external provider AND use of independent historian

**RISK axis:**
- Minimal (1): OTC medication management, rest, bandages, superficial dressings
- Low (2): Prescription drug management; OR minor surgery WITHOUT identified patient risk factors
- Moderate (3): Prescription drug management WITH minor surgery; OR minor surgery WITH identified risk factors (DM, obesity, age, anticoagulants); OR new diagnosis requiring further workup; OR diagnostic procedure requiring specialist consultation
- High (4): Major surgery; OR parenteral controlled substances; OR drug therapy requiring intensive monitoring for toxicity; OR decision regarding hospitalization; OR decision not to resuscitate

**E/M Level Assignment:**
- 2-of-3 Minimal → 99202 (new) / 99212 (established)
- 2-of-3 Low → 99202/99203 (new) / 99212/99213 (established)
- 2-of-3 Moderate → 99204 (new) / 99214 (established)
- 2-of-3 High → 99205 (new) / 99215 (established)

**Elective surgery decision = MODERATE risk minimum.** Multiple comorbidities = MODERATE problems minimum.

### Global Surgical Period — CRITICAL
- If this is a post-op follow-up within a prior surgery's global period: use 99024 (post-op follow-up visit, no charge), NOT a billable E/M
- 90-day global period applies to all major surgeries (28xxx bunion/foot reconstruction, 29893 endoscopic plantar fasciotomy, etc.)
- 10-day global period applies to minor procedures (11750 nail matrixectomy, etc.)
- The prior surgery information will be provided in the context if detected

### Modifier -25 Rule — CRITICAL
- Apply modifier -25 ONLY when BOTH conditions are true:
  1. A separately identifiable E/M service was performed beyond the decision to perform a procedure
  2. A BILLABLE PROCEDURE (surgical CPT with a global period > 0) was performed the SAME DAY
- Diagnostic imaging (73xxx), labs (80xxx-89xxx) are NOT procedures — they do NOT trigger modifier -25
- Example: 99204 + 73630 (X-ray only) → NO modifier -25 on E/M
- Example: 99202 + 11750 (matrixectomy) → YES modifier -25 required

### Medicare Routine Foot Care Compliance
- For visits where ONLY routine diabetic foot care is performed (11721 nail debridement, 11057 callus), do NOT add a separate E/M unless a DISTINCT unrelated medical problem was evaluated and documented beyond the foot care scope

## 2. SURGICAL PROCEDURES
- Code ONLY procedures PERFORMED today (verbs: "performed", "excised", "debrided", "avulsed", "applied", "resected")
- Do NOT code planned/scheduled/future procedures
- Match EXACTLY to CPT description: open vs endoscopic, partial vs complete, number of lesions

### Callus/Corn Paring — Count Lesions Exactly
- 11055 = paring of 1 lesion
- 11056 = paring of 2-3 lesions
- 11057 = paring of 4 or MORE lesions
- Count each callus/corn individually: "bilateral 1st and 5th MTH" = 4 lesions → 11057

### NCCI Bundling — CRITICAL
- Before finalizing CPT codes, check each pair: does one procedure's description INCLUDE another service you've coded?
- 28119 (calcaneal spur ostectomy "with or without plantar fascial release") INCLUDES plantar fascial release → do NOT also code 29893 or 28060 for the fascial component
- If CPT A's description says "with or without B" and you coded B separately → REMOVE the separate B code
- When uncertain, check: would billing both codes for the same anatomic site on the same day create a bundling conflict?

## 3. IMAGING / RADIOLOGY
- 73620 = foot X-ray, 2 views; 73630 = foot X-ray, 3+ views (complete)
- 73600 = ankle X-ray, 2 views; 73610 = ankle X-ray, 3+ views
- Apply RT or LT laterality modifier to all imaging

## 4. MODIFIERS
- RT/LT = right/left side on ALL lateralized procedures and imaging
- TA = right great toe; T5 = left great toe
- T1-T4 = right 2nd-5th toes; T6-T9 = left 2nd-5th toes
- 50 = bilateral procedure (same procedure both sides, same session)
- 59/XE/XS/XP/XU = distinct procedural service (NCCI edit bypass — requires documentation of distinct service)

## OUTPUT — Return valid JSON:
{
  "cpt_codes": [
    {
      "code": "99204",
      "description": "...",
      "confidence": 0.95,
      "modifiers": [],
      "modifier_reasoning": ["No modifier -25 — no same-day procedure performed; imaging (73630) is diagnostic not procedural"],
      "source": "E/M",
      "mdm_details": {
        "problems_score": 3,
        "data_score": 2,
        "risk_score": 3,
        "mdm_level": "moderate",
        "problems_rationale": "...",
        "data_rationale": "...",
        "risk_rationale": "..."
      },
      "procedure_status": "completed",
      "laterality": null,
      "linked_diagnoses": ["M20.11", "E11.9"],
      "units": 1,
      "evidence_spans": ["exact quote"]
    }
  ],
  "em_level_reasoning": "Full MDM calculation including all three axes"
}

CRITICAL: Return ONLY valid JSON. No markdown, no code blocks."""


# ---------------------------------------------------------------------------
# PASS 3 — HCPCS Level II + SNOMED
# ---------------------------------------------------------------------------

HCPCS_SNOMED_SYSTEM_PROMPT = """You are an expert medical coder. Assign HCPCS Level II supply/DME codes and SNOMED CT clinical codes.

## HCPCS RULES
1. ONLY code items that were DISPENSED/PROVIDED/APPLIED/FITTED on this date of service
2. Look for action verbs: "applied", "dispensed", "fitted", "provided", "placed", "replaced"
3. Do NOT code items that were "prescribed", "ordered", "recommended", "referred" but not physically given today
4. Do NOT code A5513 (diabetic shoe insert) unless the insert was physically given/replaced at this visit
5. Named wound products have directly assignable codes:
   - Aquacel Ag → A6196 (alginate wound cover)
   - MedHoney → A6248
   - Mepilex → A6212 (foam non-adhesive)
   - Collagen dressings → A6020-A6024 range
6. Cast supplies are separately billable when TCC is applied (Q4038/Q4039 based on material)
7. For supplies where the exact type is unclear, set needs_review=true with reason

## SNOMED CT RULES
1. Assign SNOMED codes for ALL clinical findings, disorders, procedures, and anatomical sites documented
2. Use the MOST SPECIFIC concept available — never use a parent/root concept when a specific child exists
3. AVOID these generic root concepts (too broad, flag if used):
   - 71388002 (Surgical procedure — root of ALL surgery, useless)
   - 404684003 (Clinical finding — root)
   - 64572001 (Disease — root)
   - 123037004 (Body structure — root)
4. For surgical procedures: find the specific procedure concept, not a parent category
   - "Endoscopic plantar fasciotomy" → 239175002 (Plantar fasciotomy), NOT 71388002
   - "Calcaneal spur excision" → 274203009 (Excision of calcaneal spur), NOT 71388002
5. DEDUPLICATION: Do NOT assign the same concept_id to two different clinical terms
   - If the same concept_id appears for two different entity texts, one mapping is wrong — find the correct ID for each
6. Confidence calibration: if you must use a parent concept because no specific one exists, set confidence ≤ 0.4

## OUTPUT — Return valid JSON:
{
  "hcpcs_codes": [
    {
      "code": "A6196",
      "description": "Alginate wound cover, pad size 16 sq in or less",
      "confidence": 0.90,
      "modifiers": [],
      "units": 1,
      "linked_diagnoses": ["L97.522"],
      "rationale": "Aquacel Ag dressing applied and explicitly documented.",
      "supporting_text": "PLAN: Applied Aquacel Ag dressing.",
      "needs_review": false,
      "review_reason": null
    }
  ],
  "snomed_codes": [
    {
      "concept_id": "239175002",
      "description": "Plantar fasciotomy",
      "entity_text": "endoscopic plantar fasciotomy",
      "category": "procedure",
      "confidence": 0.95
    }
  ]
}

CRITICAL: Return ONLY valid JSON. No markdown, no code blocks."""


# ---------------------------------------------------------------------------
# PASS 4 — Self-Verification & Correction (Anchor-and-Audit)
# ---------------------------------------------------------------------------

VERIFICATION_SYSTEM_PROMPT = """You are a senior certified medical coding auditor (CCS, CPC-A, CPMA). Audit and correct a complete code set for a podiatry clinical note.

## ABSOLUTE RULES — MUST READ BEFORE AUDITING

### RULE 1: PROTECTED CODES — NEVER REMOVE
Codes derived from diagnoses EXPLICITLY LISTED in the Assessment/Diagnoses section are PROTECTED.
You CANNOT remove them. You may change the specific code (e.g., make it more specific) but CANNOT eliminate the diagnosis.
Protected anchors will be listed below.

### RULE 2: PROTECT CORRECTLY-INFERRED ENCOUNTER CODES
Post-operative aftercare codes derived from the clinical context are ALSO protected:
- Z48.89 (surgical aftercare) is correct for post-op visits when NO device is being removed
- Z47.2 (removal of internal fixation device) is ONLY correct when a pin, wire, K-wire, or plate is ACTUALLY REMOVED at this visit
- If the note says "hardware removal scheduled for week 4" → the current visit uses Z48.89, NOT Z47.2
- Do NOT change Z48.89 to Z47.2 unless device removal is EXPLICITLY documented as occurring today

### RULE 3: ASYMMETRIC THRESHOLDS
- To KEEP a code: reasonable support from note is sufficient
- To REMOVE a code: you must have ABSOLUTE CERTAINTY it is unsupported AND it is NOT a protected anchor
- When in doubt: KEEP the code and set needs_review=true rather than removing

### RULE 4: REMOVAL REQUIRES QUOTED JUSTIFICATION
You may ONLY remove a code if ALL of these are true:
1. NOT derived from the Assessment section
2. NOT a PMH condition with active medication
3. NOT a correctly-inferred encounter/aftercare code
4. You provide the EXACT quote from the note proving it should not be coded

## AUDIT CHECKLIST

### A. Completeness — ADD what is missing
1. Is every diagnosis in the ASSESSMENT section coded?
2. Are all PMH conditions with active medications coded as secondary?
3. Is an E/M code assigned if the note documents HPI + exam + MDM?
4. Are all imaging studies PERFORMED today coded?
5. Are all surgical procedures PERFORMED today coded?
6. Is modifier -25 on E/M ONLY when a billable PROCEDURE (not imaging) was also performed?
7. Are all laterality modifiers applied (RT/LT)?
8. For post-op visits within global period: was 99024 used instead of a billable E/M?

### B. Over-Coding — REMOVE only if certain and NOT protected
1. Do NOT add symptom codes when a definitive diagnosis explains them
2. Do NOT add Z47/Z48 aftercare unless this is explicitly a follow-up for prior surgery
3. Do NOT add Z79.84, Z79.899 long-term drug codes on outpatient podiatry encounters
4. Do NOT code both E11.9 and E11.40 (or any more-specific DM code) simultaneously
5. Do NOT code A5513/A5512 diabetic shoes unless physically dispensed today (not just referred)

### C. NCCI Bundling — CHECK EVERY CPT PAIR
For each pair of CPT codes, ask: does one code's description include the other service?
- If CPT X description says "with or without Y" and you've coded Y separately → REMOVE Y
- 28119 (calcaneal spur ostectomy "with or without plantar fascial release"):
  * If 28119 is in the code set, REMOVE any separate plantar fascial release code (29893, 28060)
  * If 29893 is in the code set, REMOVE any separate calcaneal spur code that includes fascial release
- When a bundling conflict is identified, keep the more comprehensive code and remove the component

### D. MDM Level Verification
- Surgical decision (elective surgery scheduled) = MODERATE risk, MODERATE problems minimum → 99204/99205
- Post-op follow-up within 90-day global period = 99024 (zero charge), not 99213/99214
- Modifier -25: ONLY when a billable same-day procedure (with global period > 0) was performed
- Imaging (73xxx) does NOT trigger modifier -25

### E. SNOMED Consistency
- Check for duplicate concept_id assigned to different entity texts → flag the incorrect one
- Check for root/parent concepts (71388002, 404684003, 64572001) → flag with confidence ≤ 0.4

## OUTPUT — Return valid JSON:
{
  "corrections_made": [
    {
      "type": "ADDED|REMOVED|CHANGED|RESEQUENCED|FLAGGED|RETAINED|SELECTED|EXCLUDED",
      "code": "73630",
      "reason": "X-ray (3 views) documented as performed but not coded",
      "evidence": "exact quote from note"
    }
  ],
  "icd10_codes": [ ... corrected COMPLETE list ... ],
  "cpt_codes": [ ... corrected COMPLETE list ... ],
  "hcpcs_codes": [ ... corrected COMPLETE list ... ],
  "snomed_codes": [ ... corrected COMPLETE list ... ],
  "em_level_reasoning": "Full reasoning including new/established patient designation and MDM axes",
  "audit_notes": "Summary of all audit actions taken",
  "auto_coding_review_reasons": [
    "Detailed explanation of each correction or flag for human review"
  ],
  "auto_coding_summary": "One-paragraph summary of the coding set and any corrections made"
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
    prior_surgery_info: dict | None = None,
) -> tuple[dict, dict]:
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    note_context = _build_note_context(note_sections, patient_metadata)
    entity_summary = _format_entities(entities)
    vision_block = _format_vision_context(vision_context) if vision_context else ""
    global_block = _format_global_period_context(prior_surgery_info) if prior_surgery_info else ""

    # --- PASS 1: ICD-10-CM ---
    logger.info("  Pass 1/4: ICD-10-CM diagnosis coding...")
    icd_prompt = f"""{note_context}
{vision_block}
{global_block}

## EXTRACTED CLINICAL ENTITIES
{entity_summary}

## ICD-10-CM CANDIDATE CODES (from official FY2026 database via semantic search)
{_format_candidates_for_system(rag_candidates, 'icd10')}

Assign ICD-10-CM codes. PREFER candidates from the list above (all verified in official database).
Code EVERY diagnosis in the ASSESSMENT section AND every PMH condition with active medication that affected care.
Follow the anatomic ulcer site rules exactly (L97.4xx vs L97.5xx).
Do NOT code Z79.84 or Z79.899 on this outpatient podiatry encounter."""

    icd_raw, usage = chat_completion(ICD_SYSTEM_PROMPT, icd_prompt, temperature=CODING_TEMPERATURE, max_tokens=2500)
    _add_usage(total_usage, usage)
    icd_result = _safe_parse(icd_raw, "icd10_codes")
    logger.info(f"    → {len(icd_result.get('icd10_codes', []))} ICD-10-CM codes")

    # --- PASS 2: CPT ---
    logger.info("  Pass 2/4: CPT procedure/E&M/imaging coding...")
    icd_summary = _summarize_icd(icd_result.get("icd10_codes", []))
    note_type = patient_metadata.get("note_type", "").upper()
    is_new_patient = "NEW" in note_type
    is_post_op = (prior_surgery_info or {}).get("is_post_op_visit", False)
    days_post_op = (prior_surgery_info or {}).get("days_post_op")
    prior_cpt = (prior_surgery_info or {}).get("prior_surgery_cpt", "")

    cpt_prompt = f"""{note_context}
{vision_block}
{global_block}

## PATIENT TYPE: {"NEW PATIENT → Use 99202-99205 ONLY" if is_new_patient else "ESTABLISHED PATIENT → Use 99211-99215 ONLY"}
{"## ⚠ POST-OP VISIT: Prior surgery CPT " + prior_cpt + ", Day " + str(days_post_op) + " post-op → Check global period → Use 99024 if within global period, NOT a billable E/M" if is_post_op else ""}

## ASSIGNED ICD-10-CM CODES (from Pass 1)
{icd_summary}

## EXTRACTED CLINICAL ENTITIES
{entity_summary}

## CPT CANDIDATE CODES (from official database via semantic search)
{_format_candidates_for_system(rag_candidates, 'cpt')}

Assign CPT codes. Link each CPT to supporting ICD-10-CM codes.
Check EVERY CPT pair for NCCI bundling before finalizing.
Apply modifier -25 ONLY when a billable procedure (not imaging) was performed same day."""

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

Assign HCPCS codes for supplies dispensed today and SNOMED codes for all clinical concepts.
Only code supplies that were PHYSICALLY given/applied today — not ordered or prescribed."""

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

    assessment_text = note_sections.get("assessment_diagnoses", "")
    pmh_text = note_sections.get("pmh_medications_allergies", "")
    anchor_block = _build_anchor_block(assessment_text, pmh_text, vision_context, prior_surgery_info)

    verify_prompt = f"""{note_context}
{vision_block}
{global_block}

## PATIENT TYPE: {"NEW PATIENT" if is_new_patient else "ESTABLISHED PATIENT"}

{anchor_block}

## EXTRACTED CLINICAL ENTITIES
{entity_summary}

## CURRENTLY ASSIGNED CODES (to audit)
{json.dumps(combined, indent=2)}

## RAG CANDIDATE CODES (verified in official databases)
### ICD-10-CM Candidates
{_format_candidates_for_system(rag_candidates, 'icd10')}

### CPT Candidates
{_format_candidates_for_system(rag_candidates, 'cpt')}

### HCPCS Candidates
{_format_candidates_for_system(rag_candidates, 'hcpcs')}

## AUDIT INSTRUCTIONS
1. Verify every PROTECTED ANCHOR has a code. If missing, ADD it.
2. Check NCCI bundling for every CPT pair. Remove bundled component codes.
3. Verify modifier -25 logic: only on E/M when a billable procedure (not imaging) was performed.
4. Check new/established patient type matches E/M code range.
5. Check global period: if post-op visit within global period, use 99024.
6. Remove Z79.84, Z79.899 if present — not appropriate for outpatient podiatry.
7. Remove duplicate DM codes (e.g., E11.9 when E11.40 already coded).
8. Check SNOMED for duplicate concept IDs and root-concept fallbacks.
9. Return the COMPLETE corrected code set with auto_coding_review_reasons and auto_coding_summary."""

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
        "auto_coding_review_reasons": verified.get("auto_coding_review_reasons", []),
        "auto_coding_summary": verified.get("auto_coding_summary", ""),
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


def _build_anchor_block(
    assessment_text: str,
    pmh_text: str,
    vision_ctx: dict | None,
    prior_surgery_info: dict | None = None,
) -> str:
    """Build the PROTECTED ANCHORS section from Assessment + PMH + Vision + prior surgery context."""
    lines = [
        "## PROTECTED ANCHORS — These MUST have corresponding codes in the final output",
        "",
        "### Assessment Section Diagnoses (NEVER remove — physician documented these)",
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
        lines.append(f'  Source text: "{pmh_text[:400]}"')
        lines.append("  Every PMH condition with an active medication MUST be coded.")

    # Post-op aftercare anchor
    if prior_surgery_info and prior_surgery_info.get("is_post_op_visit"):
        days = prior_surgery_info.get("days_post_op")
        desc = prior_surgery_info.get("prior_surgery_description", "prior surgery")
        lines.append("")
        lines.append("### Post-Op Aftercare Code (PROTECTED — inferred from clinical context)")
        lines.append(f'  - ANCHOR: Aftercare code for post-op follow-up after "{desc}"')
        lines.append(f"  - Days post-op: {days}")
        lines.append("  - Z48.89 is correct UNLESS a device (K-wire, pin, plate) is explicitly removed today")
        lines.append("  - Z47.2 is ONLY correct when hardware removal is documented as performed today")
        lines.append("  - DO NOT change Z48.89 to Z47.2 unless today's note explicitly states device removal")

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
        if note_cat and any(kw in note_cat for kw in ("visit", "followup", "follow_up", "post_op", "urgent")):
            lines.append("")
            lines.append("### E/M Visit Detection")
            lines.append(f"  Note category: {note_cat}")
            lines.append("  If this is a patient visit with HPI + exam + plan, an E/M MUST be coded.")

    return "\n".join(lines)


def _format_vision_context(ctx: dict) -> str:
    if not ctx:
        return ""
    parts = ["## VISION EXTRACTION CONTEXT (from intelligent PDF reading — ground truth)"]
    parts.append(f"- Note category: {ctx.get('note_category', 'unknown')}")
    procs = ctx.get("procedures_performed_today", [])
    parts.append(f"- Procedures PERFORMED today: {procs if procs else 'NONE'}")
    imgs = ctx.get("imaging_performed_today", [])
    parts.append(f"- Imaging PERFORMED today: {imgs if imgs else 'NONE'}")
    sups = ctx.get("supplies_dispensed_today", [])
    parts.append(f"- Supplies DISPENSED today: {sups if sups else 'NONE'}")
    parts.append("NOTE: Only code items listed above as performed/dispensed today.")
    return "\n".join(parts)


def _format_global_period_context(info: dict) -> str:
    if not info or not info.get("is_post_op_visit"):
        return ""
    days = info.get("days_post_op")
    desc = info.get("prior_surgery_description", "prior surgery")
    cpt = info.get("prior_surgery_cpt", "unknown")
    return (
        f"## GLOBAL SURGICAL PERIOD CONTEXT\n"
        f"- This is a POST-OPERATIVE FOLLOW-UP visit\n"
        f"- Prior surgery: {desc} (CPT {cpt})\n"
        f"- Days post-op: {days}\n"
        f"- Major surgeries (28xxx, 29893, etc.) have a 90-day global period\n"
        f"- During the global period: post-op visits use CPT 99024, NOT a billable E/M\n"
        f"- The pre-op day + surgery day + global days are included in the surgical package fee"
    )


def _add_usage(total: dict, new: dict):
    for k in total:
        total[k] += new.get(k, 0)
