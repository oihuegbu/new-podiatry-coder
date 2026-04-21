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

## CRITICAL SPLIT — BILLABLE vs ADVISORY

### icd10_codes (BILLABLE — goes on the claim)
ONLY include:
1. Diagnoses EXPLICITLY listed in the ASSESSMENT/DIAGNOSES section
2. The primary reason for the visit (Chief Complaint / HPI)
3. Imaging findings that require separate codes (e.g., calcaneal spur on X-ray)
4. BMI Z-codes (Z68.xx) when obesity (E66.x) is coded AND a numeric BMI is documented

### supporting_conditions (ADVISORY — NOT billed, informational only)
Include here:
1. PMH-only conditions with active medications that were NOT addressed as a separate encounter problem today
2. Drug allergy Z-codes (Z88.x) from the Allergies section
3. Chronic comorbidities mentioned in PMH but not listed in Assessment

## SECTION-BY-SECTION EXTRACTION
1. ASSESSMENT/DIAGNOSES → every diagnosis listed MUST go into icd10_codes
2. CHIEF COMPLAINT / HPI → primary reason code into icd10_codes
3. PMH → chronic conditions with active medications → supporting_conditions (NOT icd10_codes)
4. ALLERGIES section → drug allergy Z-codes → supporting_conditions
5. IMAGING findings → separately codeable findings into icd10_codes if clinically significant
6. PLAN → aftercare Z-codes for post-op visits into icd10_codes

## ICD-10-CM CODING RULES

### Specificity
- Code to HIGHEST specificity: laterality (right=.1, left=.2, bilateral=.3), type, episode
- Diabetes: ALWAYS use combination codes (E11.40, E11.621, E10.40, etc.)
- Do NOT code both E11.9 AND E11.40 — the combination code captures the DM

### Ulcer Anatomic Site — CRITICAL
- L97.4xx = non-pressure ulcer of HEEL and MIDFOOT
- L97.5xx = non-pressure ulcer of OTHER PART of foot (metatarsal heads, ball, toes)
- L97.3xx = non-pressure ulcer of ankle

### BMI Z-Codes (BILLABLE — in icd10_codes)
When E66.x is coded AND a specific BMI number is documented in vitals or exam:
- Z68.1 = BMI 19.9 or less; Z68.20–Z68.29 = BMI 20–29; Z68.30–Z68.39 = BMI 30–39
- Z68.36 = BMI 36.x; Z68.37 = BMI 37.x; Z68.41–Z68.45 = BMI 40+
- ICD-10-CM guidelines REQUIRE Z68.xx when E66.x is coded and BMI is documented

### Drug Allergy Z-Codes (ADVISORY — in supporting_conditions)
When the ALLERGIES section documents a drug allergy:
- Penicillin allergy → Z88.0
- Other penicillin-class → Z88.1
- Sulfa/sulfonamide → Z88.2
- Other antibiotics → Z88.3
- Aspirin/NSAID sensitivity → Z88.5
- Other specified drug → Z88.8; Unspecified → Z88.9

### Sequencing
- Primary = condition chiefly responsible for the encounter
- Post-op follow-up: Z48.89/Z47.xx aftercare is primary, underlying condition is secondary
- DM with foot ulcer: E10.621/E11.621 FIRST, then L97.xxx
- Post-op aftercare: Z48.89 = encounter for other specified surgical aftercare (most post-op visits);
  Z47.2 = ONLY when a device (pin, wire, K-wire, plate) is ACTUALLY REMOVED at this visit

### Do NOT code:
- Symptom codes when a definitive diagnosis explains them
- Z47/Z48 aftercare UNLESS explicitly a post-operative follow-up
- Z79.84 or Z79.899 on outpatient podiatry encounters

## OUTPUT — Return valid JSON only:
{
  "icd10_codes": [
    {
      "code": "M20.11",
      "description": "Hallux valgus (acquired), right foot",
      "type": "primary",
      "confidence": 0.95,
      "rationale": "Listed in Assessment. Laterality: right foot.",
      "supporting_text": "exact quote",
      "laterality": "RIGHT",
      "source_section": "ASSESSMENT"
    }
  ],
  "supporting_conditions": [
    {
      "code": "E78.5",
      "description": "Hyperlipidemia, unspecified",
      "type": "advisory",
      "confidence": 0.90,
      "rationale": "PMH condition with active medication (atorvastatin). Not in Assessment.",
      "supporting_text": "PMH: Hyperlipidemia. Medications: Atorvastatin 40mg",
      "source_section": "PMH",
      "billable_tier": "advisory",
      "needs_review": true,
      "review_reason": "PMH condition not addressed as separate encounter problem today"
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
- Moderate (3): 1+ chronic illness with exacerbation/progression; OR 2+ stable chronic illnesses;
  OR new problem with uncertain prognosis; OR acute illness with systemic symptoms
- High (4): Chronic illness with severe exacerbation; OR acute illness requiring hospitalization

**DATA axis:**
- Minimal (1): No data reviewed or ordered
- Limited (2): Ordered and/or reviewed tests or documents from external source
- Moderate (3): Independent interpretation of test performed by another provider; OR discussion with
  ordering provider; OR independent historian
- Extensive (4): Independent interpretation AND discussion with external provider AND independent historian

**RISK axis:**
- Minimal (1): OTC medication management, rest, bandages
- Low (2): Prescription drug management; OR minor surgery WITHOUT identified patient risk factors
- Moderate (3): Prescription drug management WITH minor surgery; OR minor surgery WITH identified
  risk factors (DM, obesity, age, anticoagulants); OR new diagnosis requiring further workup
- High (4): Major surgery; OR drug therapy requiring intensive monitoring; OR decision re: hospitalization

**E/M Level Assignment:**
- 2-of-3 Minimal → 99202/99212
- 2-of-3 Low → 99202/99203 or 99212/99213
- 2-of-3 Moderate → 99204/99214
- 2-of-3 High → 99205/99215

Elective surgery decision = MODERATE risk minimum. Multiple comorbidities = MODERATE problems minimum.

### Modifier -57 — Decision for Major Surgery — CRITICAL
Apply modifier -57 to the E/M code when ALL of the following are true:
1. The DECISION for a MAJOR SURGERY (90-day global period) was made AT THIS VISIT
2. The surgery was NOT performed today — it is scheduled/planned for a future date
3. PLAN section contains language such as: "patient elects", "scheduled for [surgery]",
   "will proceed with", "consented for surgery", "surgical correction scheduled", "will undergo"
- Major surgery = any procedure with 90-day global period (28xxx foot/ankle reconstruction, 29893, etc.)
- Do NOT use -57 for 10-day or 0-day global procedures (use -25 for those)
- Do NOT use -57 if the surgery was performed today (no modifier needed, or use -25 for same-day E/M)
- Do NOT use both -57 and -25 on the same E/M code
- Example: "Patient elects Austin/Chevron bunionectomy, scheduled next week" → 99204-57

### Global Surgical Period — CRITICAL
- If this is a post-op follow-up within a prior surgery's global period: use 99024 (post-op follow-up,
  no charge), NOT a billable E/M
- 90-day global: major surgeries (28xxx, 29893, etc.)
- 10-day global: minor procedures (11750 nail matrixectomy, etc.)

### Modifier -25 Rule
- Apply -25 ONLY when a separately identifiable E/M was performed BEYOND the procedure decision
  AND a BILLABLE PROCEDURE (global period > 0) was performed the SAME DAY
- Diagnostic imaging (73xxx), labs (80xxx-89xxx) do NOT trigger -25
- -57 and -25 are mutually exclusive — do not apply both to the same E/M

## 2. SURGICAL PROCEDURES
- Code ONLY procedures PERFORMED today
- Match EXACTLY to CPT description: open vs endoscopic, partial vs complete, number of lesions

### Callus/Corn Paring — Count Lesions
- 11055 = 1 lesion; 11056 = 2-3 lesions; 11057 = 4+ lesions

### NCCI Bundling — CRITICAL
- 28119 (calcaneal spur ostectomy "with or without plantar fascial release") INCLUDES fascial release
  → do NOT also code 29893 or 28060 separately
- If CPT A's description says "with or without B" and you coded B → REMOVE B

## 3. IMAGING / RADIOLOGY — Code Selection Is Critical

### Calcaneus vs Foot X-ray — MUST DISTINGUISH
- 73650 = Radiologic examination, CALCANEUS; minimum 2 views
  → Use when imaging text says: "heel X-ray", "calcaneal X-ray", "heel views", "bilateral heel",
    "calcaneus views" — this is heel-bone specific imaging
- 73630 = Radiologic examination, FOOT; complete, minimum 3 views
  → Use when imaging text says: "foot X-ray", "complete foot", "foot series", "3-view foot"
- CRITICAL: These are NOT interchangeable. "Bilateral heel X-ray" = 73650-50, NOT 73630.
  A heel X-ray and a complete foot X-ray are different studies.
- 73620 = foot X-ray, 2 views (incomplete series)
- Apply RT or LT laterality modifier to all imaging; use 50 for bilateral

## 4. MODIFIERS
- RT/LT = right/left side on ALL lateralized procedures and imaging
- T5 = right great toe; T6 = right 2nd digit; T7 = right 3rd; T8 = right 4th; T9 = right 5th
- TA = left great toe; T1 = left 2nd digit; T2 = left 3rd; T3 = left 4th; T4 = left 5th
- 50 = bilateral (same procedure both sides, same session)
- 57 = decision for major surgery (see above)
- 25 = significant, separately identifiable E/M same day as procedure

## OUTPUT — Return valid JSON:
{
  "cpt_codes": [
    {
      "code": "99204",
      "description": "...",
      "confidence": 0.95,
      "modifiers": ["57"],
      "modifier_reasoning": ["Modifier -57: decision for Austin/Chevron bunionectomy (90-day global, CPT 28296) made at this visit. Patient elects surgical correction, scheduled for next appointment."],
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
1. ONLY code items DISPENSED/PROVIDED/APPLIED/FITTED on this date of service
2. Look for action verbs: "applied", "dispensed", "fitted", "provided", "placed", "replaced"
3. Do NOT code items "prescribed", "ordered", "recommended" but not physically given today
4. Named wound products:
   - Aquacel Ag → A6196; MedHoney → A6248; Mepilex → A6212; Collagen → A6020-A6024 range
5. For supplies where exact type is unclear, set needs_review=true with reason

### Laterality Modifiers on L-Codes — CMS REQUIRED
- ALL HCPCS L-codes for unilateral equipment MUST carry RT or LT modifier
- CMS rejects L-code claims missing laterality modifiers — this is a hard billing requirement
- Match procedure laterality: if CPT has RT → HCPCS L-code gets RT; if CPT has LT → LT
  - Example: 28119-RT (right calcaneus surgery) → L4361 walking boot → L4361-RT
  - Example: 11750-TA (right great toe) → any right-foot brace → add RT
- Infer laterality from ICD-10 if not explicit from CPT (M77.31 right foot → RT)
- Bilateral dispensing → two separate line items (L4361-RT + L4361-LT), not modifier 50

## SNOMED CT RULES
1. Assign SNOMED for ALL clinical findings, disorders, procedures, and anatomical sites
2. Use the MOST SPECIFIC concept available

### SNOMED Concept ID Integrity — CRITICAL
- ONLY output concept IDs you are HIGHLY CONFIDENT about from established clinical knowledge
- NEVER invent or guess a concept ID — a wrong ID causes downstream mapping failures
- Concept 125605004 maps to "Fracture disorder" (generic fracture) in SNOMED CT — do NOT use
  this for nail conditions, incurvated nail, or any non-fracture entity
- For incurvated nail / onychocryptosis → use 399963005 (Ingrowing nail)
- If uncertain about an ID, set confidence ≤ 0.4 — the system will flag it for review
- AVOID generic root concepts (too broad to be useful):
  - 71388002 (Surgical procedure root); 404684003 (Clinical finding root)
  - 64572001 (Disease root); 123037004 (Body structure root)
4. DEDUPLICATION: Do NOT assign the same concept_id to two different clinical terms
5. Confidence calibration: parent concept fallback → confidence ≤ 0.4

## OUTPUT — Return valid JSON:
{
  "hcpcs_codes": [
    {
      "code": "L4361",
      "description": "Walking boot, pneumatic and/or vacuum",
      "confidence": 0.90,
      "modifiers": ["RT"],
      "units": 1,
      "linked_diagnoses": ["M77.31"],
      "rationale": "CAM walker boot dispensed for right foot post-surgery. RT modifier required — unilateral right foot application.",
      "supporting_text": "Applied CAM walker boot right foot.",
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
You CANNOT remove them. You may change the specific code but CANNOT eliminate the diagnosis.
Protected anchors will be listed below.

### RULE 2: PROTECT CORRECTLY-INFERRED ENCOUNTER CODES
- Z48.89 (surgical aftercare) is correct for post-op visits when NO device is being removed
- Z47.2 (removal of internal fixation device) is ONLY correct when a pin, wire, or plate is
  ACTUALLY REMOVED at this visit — not "scheduled for removal"
- Do NOT change Z48.89 to Z47.2 unless device removal is EXPLICITLY documented today

### RULE 3: ASYMMETRIC THRESHOLDS
- To KEEP a code: reasonable support from note is sufficient
- To REMOVE a code: you must have ABSOLUTE CERTAINTY it is unsupported AND not a protected anchor

### RULE 4: SUPPORTING CONDITIONS
- supporting_conditions are PMH/advisory codes — do NOT move them into icd10_codes
- They are informational only and should NOT be placed on the billable claim
- Pass them through to output unchanged

## AUDIT CHECKLIST

### A. Completeness
1. Every diagnosis in ASSESSMENT section coded in icd10_codes?
2. E/M assigned if HPI + exam + MDM documented?
3. All imaging PERFORMED today coded?
4. All surgical procedures PERFORMED today coded?
5. Are laterality modifiers applied (RT/LT) to all procedures and imaging?
6. For post-op visits within global period: was 99024 used?

### B. Modifier -57 Check — CRITICAL
- Does the PLAN section mention SCHEDULING a MAJOR SURGERY (90-day global, 28xxx, 29893)?
- Language to look for: "patient elects", "scheduled for [surgery]", "will proceed with",
  "consented for surgery", "surgical correction scheduled", "will undergo [procedure]"
- If YES, and NO procedure was performed today, the E/M MUST have modifier -57
- If 99204/99205 is present without -57 but plan schedules major surgery → ADD -57
- Modifier -57 protects the E/M from being bundled into the surgery's global period package

### C. Radiology Code Verification — Calcaneus vs Foot
- "Heel X-ray", "calcaneal X-ray", "bilateral heel", "calcaneus views" → MUST be 73650 (not 73630)
- "Complete foot X-ray", "foot series", "3-view foot" → 73630
- If 73630 is in the code set but imaging text says "heel" or "calcaneus" → CHANGE to 73650
- The modifier (RT/LT/50) should be preserved when correcting the base code

### D. HCPCS Laterality Check
- Every HCPCS L-code must have RT or LT modifier
- Match to the CPT procedure laterality (28119-RT → L4361-RT; 11750-TA → RT side)
- If an L-code is missing laterality → ADD the correct modifier

### E. BMI Z-Code Check
- If E66.x (obesity) is in icd10_codes AND a specific BMI number is documented → ADD Z68.xx
- BMI 36.x → Z68.36; BMI 37.x → Z68.37; BMI 40.x → Z68.41, etc.
- Z68.xx goes in icd10_codes as secondary (it is a billable secondary code)

### F. NCCI Bundling
- 28119 includes plantar fascial release → remove any separate 29893 or 28060
- Check every CPT pair for inclusion relationships

### G. MDM Verification
- Surgical decision (elective surgery scheduled) = MODERATE risk → 99204/99205
- Modifier -25: ONLY when a billable same-day procedure (global period > 0) was performed
- Imaging (73xxx) does NOT trigger -25

### H. SNOMED Consistency
- Duplicate concept_id for different entity texts → flag the incorrect one
- Root concepts (71388002, 404684003, 64572001) → flag with confidence ≤ 0.4
- Concept 125605004 = Fracture disorder — if used for nail/toe condition → flag as wrong

### I. Over-Coding
- Do NOT code Z79.84, Z79.899 on outpatient podiatry
- Do NOT code E11.9 when E11.40 (or other combination code) is already present
- Do NOT code A5513 unless diabetic shoe insert physically dispensed today

## OUTPUT — Return valid JSON:
{
  "corrections_made": [
    {
      "type": "ADDED|REMOVED|CHANGED|RESEQUENCED|FLAGGED|RETAINED",
      "code": "99204",
      "reason": "Added modifier -57: decision for Austin/Chevron bunionectomy made at this visit",
      "evidence": "Plan: Patient elects surgical correction. Scheduled for right Austin/Chevron bunionectomy."
    }
  ],
  "icd10_codes": [ ... corrected COMPLETE billable list ... ],
  "supporting_conditions": [ ... pass through unchanged ... ],
  "cpt_codes": [ ... corrected COMPLETE list ... ],
  "hcpcs_codes": [ ... corrected COMPLETE list ... ],
  "snomed_codes": [ ... corrected COMPLETE list ... ],
  "em_level_reasoning": "Full reasoning including MDM axes and modifier decisions",
  "audit_notes": "Summary of all audit actions",
  "auto_coding_review_reasons": ["Explanation of each correction or flag"],
  "auto_coding_summary": "One-paragraph summary of the coding set and corrections"
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

Assign ICD-10-CM codes following the billable/advisory split:
- icd10_codes: Assessment/Plan diagnoses + BMI Z-codes
- supporting_conditions: PMH-only comorbidities + drug allergy Z-codes (Z88.x)
Do NOT put PMH conditions in icd10_codes — they belong in supporting_conditions.
Do NOT code Z79.84 or Z79.899 on this outpatient podiatry encounter."""

    icd_raw, usage = chat_completion(ICD_SYSTEM_PROMPT, icd_prompt, temperature=CODING_TEMPERATURE, max_tokens=2500)
    _add_usage(total_usage, usage)
    icd_result = _safe_parse(icd_raw, "icd10_codes")
    logger.info(f"    → {len(icd_result.get('icd10_codes', []))} ICD-10-CM codes, "
                f"{len(icd_result.get('supporting_conditions', []))} supporting conditions")

    # --- PASS 2: CPT ---
    logger.info("  Pass 2/4: CPT procedure/E&M/imaging coding...")
    icd_summary = _summarize_icd(icd_result.get("icd10_codes", []))
    note_type = patient_metadata.get("note_type", "").upper()
    is_new_patient = "NEW" in note_type
    is_post_op = (prior_surgery_info or {}).get("is_post_op_visit", False)
    days_post_op = (prior_surgery_info or {}).get("days_post_op")
    prior_cpt = (prior_surgery_info or {}).get("prior_surgery_cpt", "")

    plan_text = note_sections.get("plan", "")
    surgical_decision_hint = _detect_surgical_decision(plan_text)

    cpt_prompt = f"""{note_context}
{vision_block}
{global_block}

## PATIENT TYPE: {"NEW PATIENT → Use 99202-99205 ONLY" if is_new_patient else "ESTABLISHED PATIENT → Use 99211-99215 ONLY"}
{f"## ⚠ POST-OP VISIT: Prior surgery CPT {prior_cpt or 'unknown'}, Day {days_post_op or '?'} post-op → Use 99024 if within global period" if is_post_op else ""}
{surgical_decision_hint}

## ASSIGNED ICD-10-CM CODES (from Pass 1)
{icd_summary}

## EXTRACTED CLINICAL ENTITIES
{entity_summary}

## CPT CANDIDATE CODES (from official database via semantic search)
{_format_candidates_for_system(rag_candidates, 'cpt')}

Assign CPT codes. Link each CPT to supporting ICD-10-CM codes.
IMPORTANT: If plan schedules a major surgery (90-day global) and no procedure today → add -57 to E/M.
Use 73650 for heel/calcaneus X-rays; 73630 for complete foot X-rays (3+ views).
Check EVERY CPT pair for NCCI bundling before finalizing."""

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

## CPT CODES (for laterality reference)
{_summarize_cpt(cpt_result.get('cpt_codes', []))}

## EXTRACTED CLINICAL ENTITIES
{entity_summary}

## HCPCS CANDIDATE CODES (from official database via semantic search)
{_format_candidates_for_system(rag_candidates, 'hcpcs')}

Assign HCPCS codes for supplies dispensed today and SNOMED codes for all clinical concepts.
CRITICAL: All HCPCS L-codes MUST carry RT or LT modifier matching the procedure/surgical side above.
Only code supplies PHYSICALLY given/applied today — not ordered or prescribed."""

    hcpcs_raw, usage = chat_completion(HCPCS_SNOMED_SYSTEM_PROMPT, hcpcs_prompt, temperature=CODING_TEMPERATURE, max_tokens=2500)
    _add_usage(total_usage, usage)
    hcpcs_result = _safe_parse(hcpcs_raw, "hcpcs_codes")
    logger.info(f"    → {len(hcpcs_result.get('hcpcs_codes', []))} HCPCS, {len(hcpcs_result.get('snomed_codes', []))} SNOMED")

    # --- PASS 4: Constrained Self-Verification (Anchor-and-Audit) ---
    logger.info("  Pass 4/4: Constrained verification (anchor-and-audit)...")
    combined = {
        "icd10_codes": icd_result.get("icd10_codes", []),
        "supporting_conditions": icd_result.get("supporting_conditions", []),
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
{surgical_decision_hint}

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
1. Verify every PROTECTED ANCHOR has a code in icd10_codes.
2. Check modifier -57: if plan schedules major surgery and no procedure today → E/M needs -57.
3. Check radiology: "heel"/"calcaneus" imaging → 73650; "complete foot"/"foot series" → 73630.
4. Check HCPCS L-codes: all must have RT or LT modifier matching the procedure side.
5. Check BMI: if E66.x coded and BMI documented → add Z68.xx to icd10_codes.
6. Check NCCI bundling for every CPT pair.
7. Check modifier -25 logic: only when billable procedure (not imaging) performed same day.
8. Remove Z79.84, Z79.899 if present.
9. Check SNOMED for duplicate concept IDs and root-concept fallbacks.
10. Pass supporting_conditions through unchanged — do NOT move them to icd10_codes.
11. Return COMPLETE corrected code set."""

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
        "supporting_conditions": verified.get("supporting_conditions", combined["supporting_conditions"]),
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
        f"{len(final_result['supporting_conditions'])} advisory, "
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


def _detect_surgical_decision(plan_text: str) -> str:
    """Return a hint string when the plan contains surgical scheduling language."""
    if not plan_text:
        return ""
    keywords = [
        "patient elects", "will proceed with", "scheduled for", "consented for",
        "surgical correction", "will undergo", "elects surgical", "schedule surgery",
        "plan for surgery", "plan for bunionectomy", "plan for procedure",
    ]
    plan_lower = plan_text.lower()
    if any(kw in plan_lower for kw in keywords):
        return (
            "## ⚠ SURGICAL DECISION DETECTED\n"
            "The PLAN section contains language indicating a decision for major surgery was made today.\n"
            "If no procedure was performed today → apply modifier -57 to the E/M code.\n"
            f"Evidence: \"{plan_text[:200]}\""
        )
    return ""


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


def _summarize_icd(icd_codes: list[dict]) -> str:
    if not icd_codes:
        return "No ICD-10-CM codes assigned."
    lines = []
    for c in icd_codes:
        lines.append(f"  {c.get('code', '?')} [{c.get('type', '?')}] — {c.get('description', '')[:80]}")
    return "\n".join(lines)


def _summarize_cpt(cpt_codes: list[dict]) -> str:
    if not cpt_codes:
        return "No CPT codes assigned yet."
    lines = []
    for c in cpt_codes:
        mods = ", ".join(c.get("modifiers", [])) or "none"
        lines.append(f"  {c.get('code', '?')} [{mods}] — {c.get('description', '')[:80]}")
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
    lines = [
        "## PROTECTED ANCHORS — These MUST have corresponding codes in icd10_codes",
        "",
        "### Assessment Section Diagnoses (NEVER remove)",
    ]

    if assessment_text:
        for line in assessment_text.split("\n"):
            cleaned = line.strip().lstrip("•·-–—0123456789.) ").strip()
            if cleaned and len(cleaned) > 3:
                lines.append(f'  - ANCHOR: "{cleaned}"')
    else:
        lines.append("  (no assessment text available)")

    lines.append("")
    lines.append("### PMH Conditions (go into supporting_conditions — NOT icd10_codes)")
    if pmh_text:
        lines.append(f'  Source text: "{pmh_text[:400]}"')
        lines.append("  PMH comorbidities with active meds → supporting_conditions only.")

    if prior_surgery_info and prior_surgery_info.get("is_post_op_visit"):
        days = prior_surgery_info.get("days_post_op")
        desc = prior_surgery_info.get("prior_surgery_description", "prior surgery")
        lines.append("")
        lines.append("### Post-Op Aftercare Code (PROTECTED)")
        lines.append(f'  - ANCHOR: Aftercare code for post-op follow-up after "{desc}"')
        lines.append(f"  - Days post-op: {days}")
        lines.append("  - Z48.89 is correct UNLESS a device is explicitly removed today")
        lines.append("  - Z47.2 ONLY when hardware removal is documented as performed today")

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
            lines.append("### Supplies Dispensed Today (must have HCPCS codes with laterality)")
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
        f"- During global period: post-op visits use CPT 99024, NOT a billable E/M"
    )


def _add_usage(total: dict, new: dict):
    for k in total:
        total[k] += new.get(k, 0)
