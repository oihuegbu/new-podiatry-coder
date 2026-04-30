import json
from app.core.llm_client import chat_completion
from app.core.config import CODING_TEMPERATURE, CODING_MAX_TOKENS
from app.core.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Database Description Reconciliation
# ---------------------------------------------------------------------------

def _enrich_with_db_descriptions(entries: list[dict], code_system: str, db) -> list[dict]:
    """Lookup each assigned code in the authoritative database and attach its
    real description as ``db_description``.  The verification pass uses this
    to catch LLM hallucinations (e.g. M21.611 assigned for flat foot when the
    database says 'Bunion of right foot')."""
    if db is None:
        return entries
    for entry in entries:
        code = entry.get("code", "").strip()
        if not code:
            continue
        if code_system == "icd10":
            rec = db.validate_icd10(code)
            desc = (rec or {}).get("description", "")
        elif code_system == "cpt":
            rec = db.validate_cpt(code)
            desc = (rec or {}).get("long_description", "") or (rec or {}).get("short_description", "")
        elif code_system == "hcpcs":
            rec = db.validate_hcpcs(code)
            desc = (rec or {}).get("description", "")
        else:
            desc = ""
        entry["db_description"] = desc if desc else "NOT FOUND IN DATABASE"
    return entries


def _build_db_description_block(combined: dict) -> str:
    """Build the authoritative code description block for the verification prompt.

    When the LLM's assigned description or rationale conflicts with the database
    description, the verification pass can catch and correct the wrong code."""
    lines = ["## AUTHORITATIVE DATABASE DESCRIPTIONS (ground truth — use to catch wrong codes)"]
    lines.append("If a code's DB_DESCRIPTION contradicts the clinical context or rationale → that code is WRONG. Fix it.\n")

    icd_lines = []
    for e in combined.get("icd10_codes", []):
        code = e.get("code", "")
        db_desc = e.get("db_description", "NOT FOUND")
        llm_desc = e.get("description", "")
        rationale = e.get("rationale", "")[:60]
        flag = " ⚠ NOT IN DATABASE" if db_desc == "NOT FOUND IN DATABASE" else ""
        icd_lines.append(f"  ICD {code}: DB='{db_desc}'{flag} | LLM='{llm_desc}' | rationale='{rationale}'")

    cpt_lines = []
    for e in combined.get("cpt_codes", []):
        code = e.get("code", "")
        db_desc = e.get("db_description", "NOT FOUND")
        llm_desc = e.get("description", "")[:60]
        flag = " ⚠ NOT IN DATABASE" if db_desc == "NOT FOUND IN DATABASE" else ""
        cpt_lines.append(f"  CPT {code}: DB='{db_desc}'{flag} | LLM='{llm_desc}'")

    hcpcs_lines = []
    for e in combined.get("hcpcs_codes", []):
        code = e.get("code", "")
        db_desc = e.get("db_description", "NOT FOUND")
        llm_desc = e.get("description", "")[:60]
        flag = " ⚠ NOT IN DATABASE" if db_desc == "NOT FOUND IN DATABASE" else ""
        hcpcs_lines.append(f"  HCPCS {code}: DB='{db_desc}'{flag} | LLM='{llm_desc}'")

    if icd_lines:
        lines.append("### ICD-10-CM")
        lines.extend(icd_lines)
    if cpt_lines:
        lines.append("### CPT")
        lines.extend(cpt_lines)
    if hcpcs_lines:
        lines.append("### HCPCS")
        lines.extend(hcpcs_lines)

    lines.append("\nCRITICAL INSTRUCTION: If any DB_DESCRIPTION does NOT match the clinical reason this")
    lines.append("code was assigned, CHANGE that code to the correct one from the RAG candidates above.")
    lines.append("The DB_DESCRIPTION is authoritative ground truth. Trust it over the LLM-assigned description.\n")
    return "\n".join(lines)


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

### Redundant Diabetes Codes — CRITICAL OVERRIDE
When ANY specific DM combination code is assigned (E10.1–E10.8, E11.1–E11.8, E13.1–E13.8):
- DO NOT also code E11.9, E10.9, or E13.9 (unspecified/without complications)
- This applies even if the assessment ALSO lists "T2DM without complications" or "additional coding"
- The combination code (E11.40, E11.621, etc.) captures the DM — the generic code is redundant
- Physician assessment documentation errors ("additional coding") do NOT override ICD-10-CM guidelines

### Code Selection Accuracy
When selecting any ICD-10-CM code, your rationale and description MUST match the actual meaning
of that code. The verification pass will check every code against the official FY2026 database.
If you are uncertain between two similar codes, include both in your rationale and select the most
specific one. Codes marked "NOT FOUND IN DATABASE" in the verification pass must be corrected.

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
- Limited (2): Ordered and/or reviewed tests or documents from external source (prior MRI, X-ray report, labs)
- Moderate (3): Independent interpretation of test performed by another provider; OR discussion with
  ordering provider; OR independent historian
- Extensive (4): Independent interpretation AND discussion with external provider AND independent historian

**RISK axis:**
- Minimal (1): OTC medication management, rest, bandages, splints
- Low (2): Prescription drug management; OR minor surgery WITHOUT identified patient risk factors
- Moderate (3): Minor surgery WITH identified patient risk factors (DM, obesity, anticoagulants, age ≥65);
  OR prescription drug management WITH minor surgery; OR new diagnosis requiring further workup;
  OR corticosteroid injection (steroid injection = moderate risk regardless of other factors)
- High (4): Major surgery with identified risk factors; OR drug therapy requiring intensive monitoring

**E/M Level Assignment:**
- 2-of-3 Minimal → 99202/99212
- 2-of-3 Low → 99203/99213
- 2-of-3 Moderate → 99204/99214
- 2-of-3 High → 99205/99215

Special rules:
- Corticosteroid injection = MODERATE risk → if 2+ problems or limited data also present → 99214
- Elective surgery decision = MODERATE risk minimum → 99204/99214 or higher
- Multiple comorbidities (2+ stable chronic) = MODERATE problems minimum

### CRITICAL — Diabetic Patient Foot Care E/M Level
When a DIABETIC patient (E10.x, E11.x, E13.x) is receiving ANY nail, callus, wound, or skin procedure:
- Risk axis = MODERATE (DM is an identified patient risk factor for minor surgery per 2021 AMA MDM)
- If the patient also has 2+ stable chronic conditions (DM + CKD, DM + CAD, DM + HTN, etc.):
  → Problems = MODERATE (2+ stable chronic illnesses)
  → Risk = MODERATE (minor surgery with DM risk factor)
  → 2-of-3 MODERATE → **99214** (established) or **99205** (new patient), NOT 99213/99204
- 99213 is only correct for a diabetic foot visit if: the patient has only 1 chronic condition (DM alone,
  no other comorbidities) AND the procedures are purely routine without any risk factor documentation

### Modifier -25 — MANDATORY WHEN PROCEDURE PERFORMED SAME DAY
- ALWAYS add -25 to the E/M code when ANY billable procedure (global period > 0) is performed same day
- Billable procedures that trigger -25: injections (64455, 64632, 20600, 20605, 20610, 20550),
  nail procedures (11750, 11055, 11056, 11721), debridement (97597, 97598, 11042), casting (29540),
  any surgical procedure (28xxx, 11xxx, etc.)
- WITHOUT -25: payer bundles the E/M into the procedure and denies the E/M — claim loss
- Does NOT trigger -25: imaging only (73xxx, 76xxx, 77xxx) or labs only (80xxx-89xxx)
- -57 and -25 are mutually exclusive — use -57 for major surgery decision, -25 for same-day procedure

### Modifier -57 — Decision for Major Surgery — CRITICAL
Apply modifier -57 to E/M when ALL are true:
1. Decision for MAJOR SURGERY (90-day global) made AT THIS VISIT
2. Surgery NOT performed today — scheduled/planned for future
3. PLAN contains: "patient elects", "scheduled for", "will proceed with", "consented for surgery",
   "surgical correction scheduled", "will undergo"
- Do NOT use -57 for 10-day or 0-day global procedures
- Do NOT use both -57 and -25 on the same E/M

### Global Surgical Period
- Post-op visit within global period → 99024 (no charge), NOT a billable E/M
- 90-day global: major foot/ankle surgeries (28xxx, 29893)
- 10-day global: minor procedures (11750, 11055-11057)

## 2. INJECTIONS — Code Selection + Image Guidance

### Injection CPT Codes
- 64455 = Injection(s), anesthetic/steroid; plantar COMMON DIGITAL NERVE (Morton's neuroma)
  → Use for interdigital neuroma injections; includes multiple injections same interspace/session
- 64632 = Destruction by neurolytic agent; plantar common digital nerve
  → Use only when alcohol/phenol neurolytic agent is used (NOT corticosteroid)
- 64450 = Injection, anesthetic agent; other peripheral nerve or branch
  → Use for nerve blocks not specifically listed (e.g., sural nerve, common peroneal)
- 20600 = Aspiration and/or injection, small joint (IP joints of toes, MTP joints)
- 20605 = Aspiration and/or injection, intermediate joint (subtalar, midtarsal)
- 20610 = Aspiration and/or injection, large joint (ankle joint proper, tibiotalar)
- 20550 = Injection(s); single tendon sheath or ligament, aponeurosis (plantar fascia)
- 20551 = Injection(s); single tendon origin/insertion (e.g., Achilles origin)
- 64640 = Destruction by neurolytic agent; other peripheral nerve (not plantar digital)

### Image Guidance — ALWAYS CODE SEPARATELY — CRITICAL
When an injection or procedure is performed under image guidance, code the guidance as a SEPARATE CPT:
- FLUOROSCOPIC guidance → 77002 (Fluoroscopic guidance for needle placement)
  Keywords: "under fluoroscopic guidance", "fluoroscopy", "fluoroscopically guided", "C-arm guidance"
- ULTRASOUND guidance → 76942 (Ultrasonic guidance for needle placement, with permanent record)
  Keywords: "under ultrasound guidance", "sonographic guidance", "ultrasound guided", "US-guided"
- 77002 and 76942 are SEPARATELY BILLABLE — never bundled into the injection code
- Code one guidance code per session (RT/LT applies if documented as two separate anatomic sites)
- If note says "fluoroscopic guidance" AND an injection code is in your list → ADD 77002

## 3. NAIL PROCEDURES
- 11719 = Trimming of nondystrophic nails, any number (routine, no pathology)
- 11720 = Debridement of nail(s) by any method; 1–5 nails
- 11721 = Debridement of nail(s) by any method; 6 or more nails
- 11730 = Avulsion of nail plate, partial or complete; simple; single
- 11732 = each additional nail plate (add-on to 11730, list separately)
- 11740 = Evacuation of subungual hematoma
- 11750 = Excision of nail and nail matrix, partial or complete (PERMANENT removal/matrixectomy)
- 11765 = Wedge excision of skin of nail fold (for onychocryptosis without matrix excision)
Use digit modifiers: T5=right great toe, T6=right 2nd, T7=right 3rd, T8=right 4th, T9=right 5th
                     TA=left great toe, T1=left 2nd, T2=left 3rd, T3=left 4th, T4=left 5th

## 4. CALLUS / CORN / SKIN LESIONS
- 11055 = Paring/cutting of benign hyperkeratotic lesion; 1 lesion
- 11056 = 2-3 lesions
- 11057 = 4 or more lesions
- 11300-11313 = Shaving of epidermal or dermal lesion (by size)
- 11400-11446 = Excision benign lesion (by size and location)

## 5. WOUND CARE & DEBRIDEMENT
- 97597 = Debridement, open wound; first 20 sq cm or less (active wound care)
- 97598 = Debridement, open wound; each additional 20 sq cm (add-on to 97597)
- 97602 = Non-selective debridement, without anesthesia (wet-to-dry, enzymatic, autolytic); per session
- 11042 = Debridement, subcutaneous tissue; first 20 sq cm
- 11043 = Debridement, muscle and/or fascia; first 20 sq cm
- 11044 = Debridement, bone; first 20 sq cm
- 97605 = Negative pressure wound therapy (NPWT); ≤50 sq cm
- 97606 = NPWT; >50 sq cm

## 6. CASTING, STRAPPING & IMMOBILIZATION
- 29515 = Application of short leg splint (static, below knee)
- 29540 = Strapping, ankle and/or foot
- 29550 = Strapping, toes
- 29580 = Unna boot application
- 29581 = Application of multi-layer compression system; leg (below knee)
- 29049 = Application of cast, figure-of-eight
Use RT/LT modifier for unilateral; 50 for bilateral same session

## 7. IMAGING / RADIOLOGY

### Calcaneus vs Foot X-ray — MUST DISTINGUISH
- 73650 = Radiologic examination, CALCANEUS; minimum 2 views
  → "heel X-ray", "calcaneal X-ray", "bilateral heel", "calcaneus" — heel-bone specific
- 73630 = Radiologic examination, FOOT; complete, minimum 3 views
  → "foot X-ray", "complete foot", "foot series", "3-view foot"
- 73620 = Foot X-ray, 2 views only
- 73700 = CT ankle; 73701 = CT ankle with contrast
- 73718-73720 = MRI lower extremity (foot/ankle)
- Apply RT/LT to all; use 50 for bilateral

## 8. SURGICAL PROCEDURES — Podiatry
### Bunion/Hallux
- 28285 = Hammertoe correction (PIP arthroplasty/fusion), each toe
- 28296 = Austin/Chevron bunionectomy (distal metatarsal osteotomy)
- 28297 = Lapidus procedure (1st TMT arthrodesis for bunion)
- 28298 = Proximal phalangeal osteotomy for hallux valgus
### Heel/Plantar Fascia
- 28119 = Ostectomy, calcaneus (heel spur), with or without plantar fascial release
- 28060 = Fasciotomy, plantar (partial) — open approach
- 29893 = Endoscopic plantar fascial release
### Foot/Toe
- 28820 = Amputation, toe; metatarsophalangeal joint
- 28810 = Amputation, metatarsal, with toe
- 28308-28312 = Metatarsal osteotomy
### Tendon
- 27680 = Tenolysis, flexor or extensor tendon, leg and/or ankle
- 27685-27686 = Lengthening of tendon, leg
### Ankle
- 27610 = Arthrotomy, ankle, including exploration, drainage
- 27698 = Repair, secondary, disrupted ligament, ankle

### NCCI Bundling — CRITICAL
- 28119 includes plantar fascial release → do NOT add 29893 or 28060 separately
- If CPT description says "with or without B" and you also coded B → REMOVE B

## 9. MODIFIERS
- RT = right side; LT = left side (required on ALL lateralized procedures)
- 50 = bilateral same session
- 25 = significant, separately identifiable E/M same day as billable procedure (MANDATORY — see above)
- 57 = decision for major surgery (see above)
- T5 = right great toe; T6 = right 2nd; T7 = right 3rd; T8 = right 4th; T9 = right 5th
- TA = left great toe; T1 = left 2nd; T2 = left 3rd; T3 = left 4th; T4 = left 5th
- 59 = distinct procedural service (to bypass NCCI bundling when documented as separate)
- 26 = professional component only (when facility bills separately for technical component)

## OUTPUT — Return valid JSON:
{
  "cpt_codes": [
    {
      "code": "99213",
      "description": "...",
      "confidence": 0.95,
      "modifiers": ["25"],
      "modifier_reasoning": ["Modifier -25: separately identifiable E/M performed same day as 64455 injection"],
      "source": "E/M",
      "mdm_details": {
        "problems_score": 2,
        "data_score": 2,
        "risk_score": 3,
        "mdm_level": "low",
        "problems_rationale": "...",
        "data_rationale": "...",
        "risk_rationale": "Corticosteroid injection = moderate risk axis"
      },
      "procedure_status": "completed",
      "laterality": null,
      "linked_diagnoses": ["G57.61", "G57.62"],
      "units": 1,
      "evidence_spans": ["exact quote"]
    }
  ],
  "em_level_reasoning": "Full MDM calculation including all three axes with explicit 2-of-3 determination"
}

CRITICAL: Return ONLY valid JSON. No markdown, no code blocks."""


# ---------------------------------------------------------------------------
# PASS 3 — HCPCS Level II + SNOMED
# ---------------------------------------------------------------------------

HCPCS_SNOMED_SYSTEM_PROMPT = """You are an expert medical coder specializing in podiatry HCPCS Level II DME/supply coding and SNOMED CT clinical coding.

## RULE 1: ONLY CODE ITEMS DISPENSED/APPLIED TODAY
- Look for ACTION VERBS: "applied", "dispensed", "fitted", "provided", "placed", "replaced", "given"
- Do NOT code items merely "prescribed", "ordered", or "recommended" but not physically given today
- "Recommended orthotics" = do NOT code | "Dispensed custom orthotics" = DO code

## RULE 2: CUSTOM FOOT ORTHOTICS — L-CODE DECISION TREE (CRITICAL)

### Step 1: Custom vs Prefabricated?
- Custom (molded/cast/3D-scanned to patient's foot) → L3000–L3031 range
- Prefabricated / off-the-shelf → L3040–L3090 range

### Step 2: For CUSTOM orthotics — What specific design type?
- **L3000** = UCB (University of California Berkeley) shell ONLY
  → Specific heel-cup shell used for neurological/pediatric gait; rarely used in general podiatry
  → DO NOT use L3000 for routine podiatric custom functional orthotics
- **L3010** = Custom foot orthosis, longitudinal arch support only (no metatarsal component)
- **L3020** = Custom foot orthosis, longitudinal arch + metatarsal support
  → Use when: metatarsal pad, Morton's neuroma orthotic, metatarsal support, pressure redistribution
  → **DEFAULT for custom podiatric orthotics** when arch AND metatarsal elements are present
- **L3030** = Custom foot orthosis, full-length (heel to toe, full plantar coverage)
  → Note says "full-length custom orthotic" or "total contact orthotic"
- **L3031** = Custom foot orthosis, bilateral pair dispensed together

### Step 3: Morton's Neuroma orthotics → always L3020
When custom orthotics dispensed for Morton's neuroma/interdigital neuroma:
→ L3020 (arch + metatarsal support) — the metatarsal pad is a defining feature
→ Bilateral: L3020-RT + L3020-LT (two line items)

### Step 4: Plantar Fasciitis / Heel Pain orthotics → L3020 or L3030
→ L3020 = heel cup + arch support (most common custom podiatric orthotic)
→ L3030 = only if explicitly full-length, total-contact design

## RULE 3: WALKING BOOTS / CAM WALKERS
- **L4360** = Walking boot, non-pneumatic
- **L4361** = Walking boot, pneumatic and/or vacuum (CAM walker with air bladder)
  → Keywords: "CAM walker", "pneumatic boot", "air cast", "aircast"
- **L4386** = Walking boot, non-pneumatic, double upright

## RULE 4: DIABETIC SHOES AND INSERTS
- **A5500** = Diabetic shoe, custom-molded (male)
- **A5501** = Diabetic shoe, custom-molded (female)
- **A5507** = Diabetic shoe, depth shoe (male)
- **A5508** = Diabetic shoe, depth shoe (female)
- **A5512** = Full-contact insert for custom-molded diabetic shoe (companion to A5500/A5501)
- **A5513** = Full-contact insert for depth shoe (companion to A5507/A5508)
  → Code A5513 when a diabetic shoe insert is REPLACED or dispensed today, even if original
    shoe was provided at a prior visit — inserts are separately billable per episode
  → Keywords: "replaced diabetic shoe insert", "new insert", "provided insert", "changed insert"
  → "Checked and replaced diabetic shoe insert (worn)" = YES, code A5513

## RULE 5: WOUND CARE SUPPLIES
- **A6196** = Alginate/calcium alginate dressing (Aquacel, Aquacel Ag)
- **A6212** = Foam dressing ≤16 sq in (Mepilex, Allevyn)
- **A6213** = Foam dressing >16 sq in
- **A6248** = Hydrogel dressing (MedHoney, Medihoney, hydrogel gel)
- **A6020** = Collagen dressing ≤16 sq in
- **A6021** = Collagen dressing 17–48 sq in
- **A6022** = Collagen dressing >48 sq in

## RULE 6: BONE STIMULATORS
- **E0747** = Osteogenesis stimulator, electrical, non-invasive (extremity)
  → Keywords: "bone stimulator", "bone stim", "EBI", "OrthoLogic"
- **E0748** = Osteogenesis stimulator, electrical, spinal application
- **E0760** = Osteogenesis stimulator, low intensity ultrasonic (Exogen, LIPUS)

## RULE 7: COMPRESSION
- **A6531** = Gradient compression stocking, below knee, 30–40 mmHg
- **A6532** = Gradient compression stocking, below knee, 40–50 mmHg
- **A6545** = Gradient compression wrap, non-elastic

## RULE 8: INJECTABLE DRUG J-CODES — SEPARATELY BILLABLE (CRITICAL — REVENUE)
When a provider ADMINISTERS an injectable drug in the office, the drug is separately billable.
This is revenue LEFT ON THE TABLE if not coded. Look for the drug name + dose in the note.

### Corticosteroids (most common in podiatry injections)
- **J3301** = Triamcinolone acetonide, per 10mg (Kenalog)
  → 10mg injection → J3301 x 1 unit; 20mg → x 2; 40mg → x 4 (most common for neuroma/joint)
- **J1020** = Methylprednisolone acetate (Depo-Medrol), 20mg
- **J1030** = Methylprednisolone acetate (Depo-Medrol), 40mg
- **J1040** = Methylprednisolone acetate (Depo-Medrol), 80mg
- **J2920** = Methylprednisolone sodium succinate (Solu-Medrol), 20mg/ml
- **J2930** = Methylprednisolone sodium succinate (Solu-Medrol), 40mg/ml
- **J1094** = Dexamethasone sodium phosphate, per 1mg
- **J0702** = Betamethasone acetate + sodium phosphate (Celestone), per 3mg
- **J3490** = Unclassified drug (when specific J-code does not exist)

### When to code J-codes
- Note mentions: "injected [drug]", "administered [drug]", "[drug] injection given"
- Drug examples: "triamcinolone 40mg", "Kenalog 40mg", "Depo-Medrol 40mg", "dexamethasone"
- Bilateral injection same drug → code 2 units (or use modifier 50 on J-code if bilateral same dose)
- DO NOT code J-code for: topical drugs, oral prescriptions, drugs "prescribed" but not given today

### Morton's Neuroma Injection — drug identification (CRITICAL)
- Read the note carefully for the DRUG NAME before assigning a J-code
- "Triamcinolone" or "Kenalog" → J3301 (per 10mg); 40mg injection = J3301 x 4 units
- "Betamethasone" or "Celestone" → J0702 (per 3mg); 6mg injection = J0702 x 2 units
- "Methylprednisolone" or "Depo-Medrol" → J1030 (40mg) or J1020 (20mg)
- Do NOT default to J3301 if the note says betamethasone — that is J0702
- If bilateral neuroma injections → multiply units by 2 (or use modifier 50)

## RULE 9: LATERALITY MODIFIERS ON L-CODES — CMS MANDATORY
- ALL HCPCS L-codes for unilateral equipment MUST carry RT or LT modifier
- CMS REJECTS L-code claims missing laterality — hard billing requirement
- Match to procedure laterality from CPT codes:
  - CPT has RT → L-code gets RT | CPT has LT → L-code gets LT
  - ICD-10 laterality as fallback: G57.61 (right Morton's) → RT; G57.62 (left) → LT
  - Bilateral dispensing → two line items (L3020-RT + L3020-LT), NOT modifier 50
- NEVER use modifier 50 on L-codes for bilateral — always list separately with RT and LT

## SNOMED CT RULES
1. Assign SNOMED for ALL clinical findings, disorders, procedures, and anatomical sites documented
2. Use the MOST SPECIFIC concept available — never a parent root concept

### SNOMED Concept ID Integrity — CRITICAL
- ONLY output concept IDs you are HIGHLY CONFIDENT about from established clinical terminology
- NEVER invent or guess a concept ID — wrong IDs cause downstream mapping failures
- If uncertain about a specific concept ID, set confidence ≤ 0.4 (the verifier will flag it)
- **NEVER use 71388002** (Surgical procedure root) — it is too generic to have clinical value; find the specific procedure concept (arthroscopy, fasciotomy, matrixectomy, etc.)
- AVOID ALL generic root concepts — they are too broad for clinical coding:
  - 71388002 (Surgical procedure root); 404684003 (Clinical finding root)
  - 64572001 (Disease root); 123037004 (Body structure root); 125605004 (Fracture disorder root)
- Use the MOST SPECIFIC concept that matches the clinical term — not a parent category
3. DEDUPLICATION:
   - Do NOT assign the same concept_id to two DIFFERENT clinical conditions
   - If the same condition appears BILATERALLY (right and left), code it ONCE with a bilateral entity_text
   - For POST-OP visits: do NOT create a SNOMED entry for procedures performed in the PAST. Only code the current post-operative diagnosis — not the past procedure itself
4. Confidence calibration: parent concept fallback → confidence ≤ 0.4

## OUTPUT — Return valid JSON:
{
  "hcpcs_codes": [
    {
      "code": "L3020",
      "description": "Custom foot orthosis, longitudinal arch and metatarsal support",
      "confidence": 0.90,
      "modifiers": ["RT"],
      "units": 1,
      "linked_diagnoses": ["G57.61"],
      "rationale": "Custom orthotics with metatarsal pad dispensed for right Morton's neuroma. L3020 = arch + metatarsal (NOT L3000 which is UCB shell only). RT modifier required for unilateral right-foot dispensing.",
      "supporting_text": "Dispensed custom orthotics with metatarsal pad bilaterally.",
      "needs_review": false,
      "review_reason": null
    }
  ],
  "snomed_codes": [
    {
      "concept_id": "57709007",
      "description": "Morton's metatarsalgia",
      "entity_text": "interdigital neuroma bilateral",
      "category": "diagnosis",
      "confidence": 0.9
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
- Imaging (73xxx) does NOT trigger -25

### H. SNOMED Consistency
- Duplicate concept_id for different clinical entities → keep only one; remove the lower-confidence duplicate
- Root concepts (71388002, 404684003, 64572001, 64572001, 125605004) → REMOVE or lower confidence to ≤ 0.4; these are too broad
- Bilateral same condition → ONE entry with bilateral entity_text; not two entries with the same concept_id
- Post-op visits: do NOT create a SNOMED procedure entry for a past surgery if you already have the diagnosis concept for that condition — code diagnosis ONCE
- If any SNOMED concept has confidence ≤ 0.4, flag it with is_root_concept=true for downstream review

### I. Image Guidance — MANDATORY AUDIT (CRITICAL)
When an injection is present in cpt_codes (64455, 64632, 64450, 64640, 20600, 20605, 20610, 20550, 20551):
- Search the CLINICAL NOTE for guidance keywords:
  - Fluoroscopic: "fluoroscopic", "fluoroscopy", "C-arm", "under fluoroscopic guidance"
  - Ultrasound: "ultrasound guided", "ultrasound-guided", "sonographic", "US-guided", "under ultrasound"
- If FLUOROSCOPIC guidance documented AND 77002 is NOT in cpt_codes → ADD 77002
- If ULTRASOUND guidance documented AND 76942 is NOT in cpt_codes → ADD 76942
- 77002 and 76942 are NEVER bundled into injection codes — always separately billable
- Add as correction type "ADDED" with evidence from the note

### J. Modifier -25 — MANDATORY ENFORCEMENT
When ANY billable same-day procedure is performed (global period > 0):
- Billable procedures: 64455, 64632, 20600–20610, 20550, 20551, 11750, 11055–11057, 97597, 29540, any 28xxx
- If E/M (99202–99215) is present AND a billable procedure is present AND E/M lacks modifier -25:
  → MUST ADD modifier -25 to the E/M code
  → This is a CORRECTION (type: CHANGED), not a flag
  → Document: "Mandatory -25 added: E/M performed same day as [procedure code]"
- CRITICAL: Without -25, payer bundles E/M into the procedure's global period → claim denied
- Exception: -25 is NOT added when ONLY diagnostic imaging (73xxx, 76xxx, 77xxx) is performed

### K. Orthotic L-Code Type Verification
- L3000 is ONLY for UCB (University of California Berkeley) shell orthotics — RARELY used in podiatry
- If L3000 is coded for:
  - Morton's neuroma / interdigital neuroma → CHANGE to L3020 (arch + metatarsal support)
  - Plantar fasciitis with full-length orthotic → CHANGE to L3030
  - Generic "custom orthotic" with metatarsal pad → CHANGE to L3020
  - Bilateral custom orthotics dispensed as pair → CHANGE to L3031 or L3020-RT + L3020-LT
- Only keep L3000 if note explicitly documents UCB-type shell design
- **LINKED DIAGNOSES on L-codes**: Link HCPCS codes to ALL supporting ICD codes, not just the primary.
  Any secondary diagnosis that clinically supports the orthotic (e.g., pes planus, neuropathy) should be
  included in linked_diagnoses so every billed ICD has procedure linkage on the claim.

### N. Walking Boot Type Verification — CRITICAL REVENUE ERROR
- **L4360** = Walking boot, NON-PNEUMATIC only
- **L4361** = Walking boot, PNEUMATIC and/or vacuum (CAM walker with air bladder)
- If L4360 is coded BUT the note mentions any of: "CAM walker", "CAM boot", "pneumatic", "air cast",
  "aircast", "air bladder" → CHANGE to L4361. CAM walkers are ALWAYS pneumatic → L4361, never L4360.
- This is a common undercoding error. L4361 is the correct code for the vast majority of walking boots
  dispensed in podiatry practice.

### O. PMH-Only Conditions in icd10_codes — MUST REMOVE
Per ICD-10-CM outpatient coding guidelines: ONLY code conditions that were addressed, evaluated,
or managed at TODAY'S visit. PMH comorbidities with active medications that were NOT listed in
the ASSESSMENT/DIAGNOSES section and NOT addressed as a separate encounter problem today MUST
be in supporting_conditions — NOT in icd10_codes.
- Scan icd10_codes: if a code corresponds to a PMH-only condition (osteoporosis, GERD, hypothyroidism,
  anxiety, allergic rhinitis, hyperlipidemia, etc.) that appears ONLY in PMH/medications and NOT in
  the Assessment section → MOVE it to supporting_conditions (do NOT bill it)
- Exception: DM (E10–E13) is billable as secondary when it influences the podiatric treatment plan
  (e.g., DM patient receiving wound care, diabetic foot procedures, or systemic DM management)
- Exception: HTN (I10) is billable as secondary when it appears in Assessment OR when the provider
  explicitly addresses it at the visit

### L. J-Code Drug Billing Audit
When an injection CPT (64455, 64632, 20600–20610, 20550) is in cpt_codes:
- Check note for drug name and dose: triamcinolone, methylprednisolone, Kenalog, Depo-Medrol, dexamethasone, betamethasone
- If drug administered today and NO J-code in hcpcs_codes → ADD the appropriate J-code
- **Drug identification — read carefully:**
  - "triamcinolone" / "Kenalog" → J3301 (per 10mg); 40mg = x 4 units
  - "betamethasone" / "Celestone" → J0702 (per 3mg); do NOT use J3301 for betamethasone
  - "methylprednisolone" / "Depo-Medrol" → J1030 (40mg) or J1020 (20mg)
- This is revenue left on the table — do NOT omit J-codes for administered injectable drugs
- **MANDATORY OUTPUT RULE**: When you determine a J-code should be added, it MUST appear in BOTH:
  1. `corrections_made` as type="ADDED" (documentation of the correction)
  2. `hcpcs_codes` array (actual billable code in the output)
  - A J-code listed only in corrections_made but absent from hcpcs_codes will NOT be billed — this is a coding failure

### M. Over-Coding — CRITICAL OVERRIDE (supersedes anchor protection)
- Do NOT code Z79.84, Z79.899 on outpatient podiatry
- **GENERIC DM CODE OVERRIDE**: If E11.9, E10.9, or E13.9 is in icd10_codes AND a specific DM
  combination code (E10.1–E10.8, E11.1–E11.8, E13.1–E13.8) is ALSO present → REMOVE the generic
  DM code. This rule OVERRIDES anchor protection — even if the assessment lists "T2DM without
  complications", it MUST be removed when a combination code is present. List as correction:
  type=REMOVED, reason="Redundant DM generic code — combination code already captures diabetes"
- **DIABETIC SHOE INSERT**: "replaced" = dispensed/given today. If plan says "replaced diabetic
  shoe insert" or "new insert" → A5512 or A5513 MUST be coded. Do NOT remove it.
  Keywords that confirm dispensed: "replaced", "new insert", "gave", "provided new", "dispense"

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
    db=None,
    physician_documented_codes: list[dict] | None = None,
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
    # Hard DB gate — remove any hallucinated/invalid ICD codes immediately
    icd_result["icd10_codes"] = _hard_db_gate(icd_result.get("icd10_codes", []), "icd10", db)
    icd_result["supporting_conditions"] = _hard_db_gate(icd_result.get("supporting_conditions", []), "icd10", db)
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
    cpt_result["cpt_codes"] = _hard_db_gate(cpt_result.get("cpt_codes", []), "cpt", db)
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
    hcpcs_result["hcpcs_codes"] = _hard_db_gate(hcpcs_result.get("hcpcs_codes", []), "hcpcs", db)
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

    # Enrich every assigned code with its authoritative database description.
    # This gives the verification LLM ground truth to detect mismatches
    # (e.g. LLM assigned M21.611 for "flat foot" → DB says "Bunion of right foot" → wrong).
    _enrich_with_db_descriptions(combined["icd10_codes"],  "icd10",  db)
    _enrich_with_db_descriptions(combined["cpt_codes"],    "cpt",    db)
    _enrich_with_db_descriptions(combined["hcpcs_codes"],  "hcpcs",  db)
    db_description_block = _build_db_description_block(combined)

    assessment_text = note_sections.get("assessment_diagnoses", "")
    pmh_text = note_sections.get("pmh_medications_allergies", "")
    anchor_block = _build_anchor_block(assessment_text, pmh_text, vision_context, prior_surgery_info)

    physician_block = _format_physician_codes(physician_documented_codes or [])

    verify_prompt = f"""{note_context}
{vision_block}
{global_block}

## PATIENT TYPE: {"NEW PATIENT" if is_new_patient else "ESTABLISHED PATIENT"}
{surgical_decision_hint}

{anchor_block}

{physician_block}

## EXTRACTED CLINICAL ENTITIES
{entity_summary}

{db_description_block}

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
7. **IMAGE GUIDANCE (Section I)**: Search note for fluoroscopic/ultrasound guidance words. If injection present AND guidance documented AND 77002/76942 missing → ADD the guidance code. This is MANDATORY.
8. **MODIFIER -25 (Section J)**: If injection or any billable procedure is present AND E/M lacks -25 → ADD -25 to E/M. This is a MANDATORY correction, not a flag.
9. **ORTHOTIC L-CODES (Section K)**: If L3000 is coded for Morton's neuroma or custom functional orthotic → CHANGE to L3020. If L3000 for full-length → CHANGE to L3030.
10. Remove Z79.84, Z79.899 if present.
11. Check SNOMED for duplicate concept IDs and root-concept fallbacks.
12. Pass supporting_conditions through unchanged — do NOT move them to icd10_codes.
13. **CAM WALKER (Section N)**: If L4360 is coded and note mentions "CAM walker" → CHANGE to L4361.
14. **PMH CONDITIONS (Section O)**: Scan icd10_codes for PMH-only conditions not in Assessment → MOVE to supporting_conditions.
15. Return COMPLETE corrected code set with ALL original codes (corrected as needed)."""

    verify_raw, usage = chat_completion(VERIFICATION_SYSTEM_PROMPT, verify_prompt, temperature=CODING_TEMPERATURE, max_tokens=4096)
    _add_usage(total_usage, usage)
    verified = _safe_parse(verify_raw, "icd10_codes")

    # Fix 7 — J-code enforcement + modifier hygiene
    verified = _enforce_j_codes_from_corrections(verified)
    verified = _strip_invalid_cpt_modifiers(verified)

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

    # Fix 1 — Tag every code with its provenance and detect physician code replacements
    final_result = _tag_code_sources(final_result, physician_documented_codes or [], entities)

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
        ner = e.get("ner_source", "llm")
        # [G] = GLiNER-confirmed (biomedical NER validated), [L] = LLM-only
        ner_tag = "[G]" if ner == "gliner_confirmed" else "[L]"
        lines.append(
            f"- {ner_tag} [{e.get('category', '?').upper():>14}] {e.get('clinical_term', '')}{lat}{spec} "
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


# ---------------------------------------------------------------------------
# Fix 6 — Hard Database Gate
# ---------------------------------------------------------------------------

def _hard_db_gate(entries: list[dict], code_system: str, db) -> list[dict]:
    """Immediately remove codes that are NOT in the reference database.

    This prevents invalid/hallucinated codes from ever reaching the verification
    pass, and ensures every output code is defensible in an audit.
    """
    if db is None:
        return entries
    valid = []
    for entry in entries:
        code = entry.get("code", "").strip()
        if not code:
            continue
        found = False
        if code_system == "icd10":
            found = bool(db.validate_icd10(code))
        elif code_system == "cpt":
            found = bool(db.validate_cpt(code))
        elif code_system == "hcpcs":
            found = bool(db.validate_hcpcs(code))
            if not found:
                # HCPCS codes are sometimes unlisted but valid — keep as INFO, don't remove
                valid.append(entry)
                continue
        if found:
            valid.append(entry)
        else:
            logger.warning(
                f"    [DB GATE] {code_system.upper()} {code!r} NOT FOUND in reference DB — removed"
            )
    return valid


# ---------------------------------------------------------------------------
# Fix 1 — Physician Code Source Tagging + Reconciliation
# ---------------------------------------------------------------------------

def _tag_code_sources(
    result: dict,
    physician_documented_codes: list[dict],
    entities: list[dict] | None = None,
) -> dict:
    """Tag every output code with its provenance and detect physician code replacements.

    Tags:
    - physician_documented  : physician explicitly wrote this code in the note
    - ai_confirmed          : AI assigned same code as physician (agreement), OR entity was
                              validated by GLiNER-BioMed biomedical NER
    - ai_replaced_physician : AI chose a DIFFERENT code in the same category as a physician code
    - ai_inferred           : AI derived this code with no external confirmation
    """
    if not physician_documented_codes:
        # No physician codes — tag everything as ai_inferred first, then upgrade via GLiNER
        for key in ("icd10_codes", "cpt_codes", "hcpcs_codes"):
            for e in result.get(key, []):
                e.setdefault("code_source", "ai_inferred")
        _upgrade_via_gliner(result, entities)
        return result

    # Build maps: exact code → physician entry; 3-char prefix → physician entries
    phys_exact: dict[str, dict] = {}
    phys_prefix: dict[str, list[dict]] = {}
    for p in physician_documented_codes:
        code = p.get("code", "").strip().upper()
        if not code:
            continue
        phys_exact[code] = p
        prefix = code[:3]
        phys_prefix.setdefault(prefix, []).append(p)

    all_ai_codes: set[str] = set()
    for key in ("icd10_codes", "cpt_codes", "hcpcs_codes"):
        for e in result.get(key, []):
            all_ai_codes.add(e.get("code", "").strip().upper())

    # Tag each code
    for key in ("icd10_codes", "cpt_codes", "hcpcs_codes"):
        for e in result.get(key, []):
            code = e.get("code", "").strip().upper()
            if code in phys_exact:
                e["code_source"] = "physician_documented"
            else:
                # Check if a physician code in the same 3-char family was not used
                prefix = code[:3]
                same_family = phys_prefix.get(prefix, [])
                replaced = [p for p in same_family if p.get("code", "").upper() not in all_ai_codes]
                if replaced:
                    e["code_source"] = "ai_replaced_physician"
                    e["physician_code_note"] = (
                        f"Physician wrote {replaced[0].get('code')} "
                        f"({replaced[0].get('description', '')})"
                    )
                    logger.warning(
                        f"    [PHYSICIAN LOCK] AI assigned {code} but physician documented "
                        f"{replaced[0].get('code')} — flagged for review"
                    )
                else:
                    e.setdefault("code_source", "ai_inferred")

    # Detect physician codes completely absent from AI output
    missing = []
    for p in physician_documented_codes:
        code = p.get("code", "").strip().upper()
        if code and code not in all_ai_codes:
            # Check if it wasn't replaced (already caught above)
            prefix = code[:3]
            ai_same_family = [c for c in all_ai_codes if c[:3] == prefix]
            if not ai_same_family:
                # Completely missing — not even a family replacement
                missing.append(p)
                logger.warning(
                    f"    [MISSING PHYSICIAN CODE] {code} ({p.get('description', '')}) "
                    f"was in physician notes but not in AI output"
                )

    result["missing_physician_codes"] = missing
    _upgrade_via_gliner(result, entities)
    return result


_VALID_CPT_MODIFIERS = {
    # Laterality / bilateral
    "RT", "LT", "50",
    # E/M
    "25", "57",
    # Procedural
    "59", "51", "53", "26",
    "54", "55", "56",
    # Distinct encounter subsets (supersede 59)
    "XE", "XS", "XP", "XU",
    # Toe digit modifiers
    "TA", "T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9",
    # Finger digit (rare in podiatry but valid)
    "FA", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9",
}


def _strip_invalid_cpt_modifiers(verified: dict) -> dict:
    """Remove HCPCS/facility modifiers that have no valid meaning on CPT procedure codes.

    Modifiers like Q8, A1-A9, GY, GA, etc. are HCPCS-only and cause claim rejections
    when attached to CPT codes. Strip anything not in the known valid CPT modifier set.
    """
    for entry in verified.get("cpt_codes", []):
        raw = entry.get("modifiers", [])
        if not raw:
            continue
        valid = [m for m in raw if str(m).upper() in _VALID_CPT_MODIFIERS]
        removed = [m for m in raw if str(m).upper() not in _VALID_CPT_MODIFIERS]
        if removed:
            logger.warning(
                f"    [MODIFIER STRIP] CPT {entry.get('code')} — removed invalid modifiers: {removed}"
            )
            entry["modifiers"] = valid
    return verified


def _enforce_j_codes_from_corrections(verified: dict) -> dict:
    """Guarantee J-codes noted as ADDED in corrections actually appear in hcpcs_codes.

    The LLM sometimes writes 'ADDED J0702' in corrections_made but forgets to include
    the code in the hcpcs_codes array. This results in silent billing loss.
    """
    import re
    corrections = verified.get("corrections_made", [])
    hcpcs_list = verified.get("hcpcs_codes", [])
    existing = {h.get("code", "").upper() for h in hcpcs_list}

    for correction in corrections:
        if correction.get("type", "").upper() != "ADDED":
            continue
        code = correction.get("code", "").strip().upper()
        if not re.match(r"^J\d{4}$", code):
            continue
        if code not in existing:
            hcpcs_list.append({
                "code": code,
                "description": correction.get("reason", "")[:100],
                "confidence": 0.85,
                "modifiers": [],
                "units": 1,
                "linked_diagnoses": [],
                "rationale": correction.get("reason", ""),
                "supporting_text": correction.get("evidence", ""),
                "needs_review": False,
                "review_reason": None,
                "code_source": "ai_inferred",
            })
            existing.add(code)
            logger.info(f"    [J-CODE ENFORCER] Rescued {code} from corrections → added to hcpcs_codes")

    verified["hcpcs_codes"] = hcpcs_list
    return verified


def _upgrade_via_gliner(result: dict, entities: list[dict] | None) -> None:
    """Upgrade ai_inferred → ai_confirmed for codes whose driving entity was GLiNER-validated."""
    if not entities:
        return
    confirmed_terms: set[str] = set()
    for e in entities:
        if e.get("ner_source") == "gliner_confirmed":
            for field in ("clinical_term", "text"):
                val = e.get(field, "").lower().strip()
                if len(val) >= 4:
                    confirmed_terms.add(val)
    if not confirmed_terms:
        return
    for key in ("icd10_codes", "cpt_codes", "hcpcs_codes"):
        for code_entry in result.get(key, []):
            if code_entry.get("code_source") != "ai_inferred":
                continue
            evidence = " ".join([
                code_entry.get("rationale", ""),
                code_entry.get("supporting_text", ""),
                code_entry.get("description", ""),
                " ".join(code_entry.get("evidence_spans", [])),
            ]).lower()
            if any(term in evidence for term in confirmed_terms):
                code_entry["code_source"] = "ai_confirmed"


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

    dm_generic_phrases = (
        "without complications", "type 2 dm without", "type 2 diabetes without",
        "e11.9", "e10.9", "e13.9",
    )
    if assessment_text:
        for line in assessment_text.split("\n"):
            cleaned = line.strip().lstrip("•·-–—0123456789.) ").strip()
            if not cleaned or len(cleaned) <= 3:
                continue
            line_lower = cleaned.lower()
            if any(p in line_lower for p in dm_generic_phrases):
                lines.append(
                    f'  - NOT AN ANCHOR (skip): "{cleaned}" — '
                    f"generic DM code cannot coexist with a specific DM combination code"
                )
            else:
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


def _format_physician_codes(physician_codes: list[dict]) -> str:
    """Build a block for the verification prompt listing physician-documented codes."""
    if not physician_codes:
        return ""
    lines = [
        "## PHYSICIAN-DOCUMENTED CODES (explicitly written by the provider in this note)",
        "These codes were literally written by the physician — treat them with highest authority.",
        "If your output differs from any of these, you MUST flag it as needs_review=true and explain why.",
        "If a physician code is correct, include it in the output. If clinically wrong, flag for review — do NOT silently drop.",
        "",
    ]
    for p in physician_codes:
        code = p.get("code", "")
        desc = p.get("description", "")
        section = p.get("section", "")
        lines.append(f"  - {code} ({desc}) [from {section}]")
    return "\n".join(lines)
