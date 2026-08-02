import json
import re

from app.core.llm_client import chat_completion
from app.core.config import (
    CODING_TEMPERATURE,
    CODING_MAX_TOKENS,
    LLM_PROVIDER,
    CLAUDE_VERIFY_MODEL,
    CLAUDE_VERIFY_EFFORT,
    STRUCTURED_OUTPUTS,
)
from app.coding.schemas import (
    ICD_PASS_SCHEMA,
    CPT_PASS_SCHEMA,
    HCPCS_PASS_SCHEMA,
    VERIFY_PASS_SCHEMA,
)
from app.core.logger import get_logger

logger = get_logger(__name__)


def _pass_schema(schema: dict) -> dict | None:
    """The structured-output schema for a pass, honoring the kill switch."""
    return schema if STRUCTURED_OUTPUTS else None


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


def _enforce_real_descriptions(final_result: dict, db) -> None:
    """Overwrite every code's own "description" field with the real
    database text — deterministic, not a prompt instruction. Applied last,
    to the actual returned code arrays (icd10_codes/cpt_codes/hcpcs_codes),
    after Pass 4 has already written whatever it wrote. Only overwrites
    when a real record is found; a code that doesn't validate is left
    alone here (existence is a separate concern, handled by _hard_db_gate
    upstream and validator.py's code-existence check downstream) rather
    than silently blanking a description for a code that turns out to be
    invalid.
    """
    if db is None:
        return
    for entry in final_result.get("icd10_codes", []) + final_result.get("supporting_conditions", []):
        rec = db.validate_icd10(entry.get("code", "").strip())
        if rec and rec.get("description"):
            entry["description"] = rec["description"]
    for entry in final_result.get("cpt_codes", []):
        rec = db.validate_cpt(entry.get("code", "").strip())
        if rec:
            desc = rec.get("long_description") or rec.get("short_description")
            if desc:
                entry["description"] = desc
    for entry in final_result.get("hcpcs_codes", []):
        rec = db.validate_hcpcs(entry.get("code", "").strip())
        if rec and rec.get("description"):
            entry["description"] = rec["description"]


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
- INJURY CODES (chapter S/T — fractures, sprains, dislocations, open wounds) REQUIRE their
  7th character or the code is unbillable: A = initial encounter (active treatment today),
  D = subsequent encounter (routine healing follow-up), S = sequela (late effect). Choose
  from the encounter context: a new injury treated today → A; a post-treatment healing
  check → D. Never emit the 6-character stem (e.g. S93.401) — it is a non-billable header
- If a documented injury has an external cause stated (fall, twisting, sports), add the
  matching external-cause code (V/W/X/Y chapter) as a SECONDARY diagnosis with the same
  7th character — required by ICD-10-CM guidelines for injury claims when the cause is known
- Match the injury SEVERITY level to what the note documents — check the note's own title/
  category (e.g. a note titled "Lisfranc Fracture-Dislocation") and body text together; if
  either uses dislocation-level language, code the dislocation code (e.g. S93.324), not a
  lesser subluxation code (S93.321), unless the body text specifically qualifies it as a
  partial subluxation rather than a complete dislocation. Don't default to the milder code
  when the documentation's own language indicates the more severe one.

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
- A Z79.x long-term-drug-therapy code (Z79.84, Z79.4, Z79.01, etc.) with NO linked condition
  documented on this encounter — but DO code it when the condition it treats is documented and
  managed here (e.g. Z79.84 alongside E11.x diabetes is ICD-10-CM's own "use additional code"
  guidance for that condition, not an over-code; only an ORPHANED Z79.x code with nothing
  documented to justify it should be omitted)

### Redundant Diabetes Codes — CRITICAL OVERRIDE
When ANY specific DM combination code is assigned (E10.1–E10.8, E11.1–E11.8, E13.1–E13.8):
- DO NOT also code E11.9, E10.9, or E13.9 (unspecified/without complications)
- This applies even if the assessment ALSO lists "T2DM without complications" or "additional coding"
- The combination code (E11.40, E11.621, etc.) captures the DM — the generic code is redundant
- Physician assessment documentation errors ("additional coding") do NOT override ICD-10-CM guidelines

### Onset/Temporal Qualifiers (acute / chronic / subacute / congenital) — DETERMINISTIC RULE
- NEVER assign a code whose distinguishing qualifier is acute, chronic, subacute, congenital,
  or hereditary unless the provider DOCUMENTS that word (or its clinical counterpart) for the
  condition. Duration alone ("4-month history") does NOT establish chronicity — that inference
  belongs to the provider, not the coder (ICD-10-CM guideline I.A/I.B)
- Qualifier not documented → assign the Alphabetic Index's bare-term default (usually the
  unspecified code, e.g. documented "osteomyelitis" with no acute/chronic → M86.9), even when
  a qualified sibling looks clinically plausible
- "Congenital" codes (Q-chapter) require the provider to state congenital origin; an acquired
  presentation defaults to the acquired code. Apply identically every run.

### PMH Comorbidity Billability — DETERMINISTIC TIE-BREAK
- A PMH comorbidity goes in icd10_codes ONLY when the Assessment/Plan addresses it as its own
  problem: an assessment line, a medication change, an order, or a referral FOR that condition
- A measurement or screening finding alone (e.g. ABI measured during the foot exam, BP
  recorded) does NOT make the underlying PMH condition a billable encounter diagnosis —
  keep it in supporting_conditions. Apply this test identically every run.

### ONE LESION, ONE CODE — DETERMINISTIC RULE
When the assessment names the definitive condition (e.g. "acute paronychia, right hallux")
and a procedure sentence re-describes the SAME lesion with a generic treatment noun ("I&D of
abscess", "excision of mass", "wound debrided"), code ONLY the assessment's condition. Do NOT
add a second diagnosis for the procedure's descriptive noun at a broader site (a paronychia
IS the nail-fold abscess that was drained — a separate "cutaneous abscess of foot" code
double-codes one lesion). A second code is warranted only when the note documents a SECOND,
anatomically distinct lesion. Apply identically every run.

### BONY OUTGROWTH NAMING — DETERMINISTIC RULE
Code the note's exact term through the Alphabetic Index, never a clinical synonym:
- documented "exostosis" (incl. subungual exostosis) indexes to "Disorder, bone, specified
  type NEC" of the affected site — NOT to the osteophyte code
- documented "osteophyte"/"bone spur of joint margin" takes the osteophyte code
The two are different Index entries; substituting one for the other changes the code family.
Pick by the provider's own word, identically every run.

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

## EVIDENCE GROUNDING — ANTI-HALLUCINATION
Every code's "supporting_text" MUST be a verbatim quote from the clinical note.
- Do NOT assign a code if you cannot find explicit supporting text in this note
- Do NOT infer diagnoses from medical knowledge not documented here
- If a condition is ambiguous, prefer the less-specific code or omit it
- Conditions mentioned only in family history are NOT billable diagnoses

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

### Modifier -25 — MANDATORY WHEN SAME-DAY MINOR/INTERMEDIATE PROCEDURE
- Add -25 to the E/M when a same-day procedure with [global=000] or [global=010] is performed
- Triggers -25: injections (64455, 64632, 20600–20610, 20550), nail procedures (11750, 11055-11057),
  debridement (97597, 97598, 11042), casting (29540), and any other [global=000] or [global=010] CPT
- WITHOUT -25 on minor/intermediate procedures: payer bundles E/M into procedure → claim loss
- Does NOT trigger -25: procedures annotated [global=090] — those require -57 instead (see below)
- -57 and -25 are mutually exclusive on the same E/M line

### WHETHER to bill an E/M AT ALL alongside a same-day minor procedure — DETERMINISTIC RULE
NCCI Policy Manual Ch. 1: the decision to perform a minor ([global=000/010]) procedure is
included in the procedure's payment. The -25 rule above governs the MODIFIER once an E/M is
billed; THIS rule governs whether the E/M line exists. Bill the E/M if and only if at least
one of these is documented in the note:
1. SEPARATE PROBLEM: a condition evaluated or managed today that no procedure addresses —
   chronic disease assessment with plan changes, medication started/adjusted/refilled, a new
   complaint worked up, ordering/reviewing tests for a non-procedure problem
2. NEW EVALUATION CULMINATING IN TODAY'S DECISION: the presenting problem was newly evaluated
   (or had significantly changed/worsened) at THIS visit, with history + exam + MDM documented,
   and the decision to perform today's procedure came out of that evaluation (all new-patient
   visits qualify; an established patient returning as scheduled for a planned/routine
   procedure on a known stable problem does NOT)
If NEITHER applies, output NO E/M code — only the procedure(s). Apply this rule identically
every time you code the same note; when the note is genuinely borderline, the tiebreaker is
criterion 2's literal text: "did the note document an evaluation that produced today's
treatment decision, or did the patient arrive already scheduled for this procedure?"
BORDERLINE TIE-BREAK (apply literally, the same way every run): a note that opens with the
procedure as the visit's purpose (plan/procedure-note format, longstanding known diagnosis,
prior conservative therapy already failed BEFORE today) is a scheduled procedure — NO E/M —
even if a brief history and focused exam are recorded, because that pre-work is included in
the procedure. Only genuinely NEW evaluation content (new differential, new test reviewed
today that changed the plan, new problem) earns the E/M line.
PRECEDENCE (evaluate BEFORE the borderline tie-break, identically every run): criterion 1
evidence outranks the scheduled-procedure shape. A medication INITIATED at today's visit
(new prescription with dose/duration — not a refill and not a dressing/aftercare supply),
or a test ORDERED today to rule out a condition that no procedure on this claim treats
(e.g., a radiograph ordered to rule out osteomyelitis when today's procedure treats the
nail), IS the significant separately identifiable service. When either is documented, bill
the E/M with -25 even though the procedure itself was planned/scheduled. Only when NEITHER
criterion-1 nor criterion-2 evidence exists does the borderline tie-break decide.

### SEPARATELY DOCUMENTED SPECIMEN PROCEDURES — DETERMINISTIC RULE
A biopsy or culture harvest documented as its own act with its own disposition ("bone biopsy
obtained with rongeur — sent for culture and histopathology") is a separately billable
procedure with its own CPT code whenever the code exists and no NCCI edit bundles it into
another same-session procedure. Do NOT fold it into a debridement/excision line, and do not
omit it because it happened through the same incision — bill it every run, identically. (The
validator strips it deterministically if an NCCI edit or same-site rule bundles it.)

### BONE EXCISION OF A PHALANX — FAMILY TIE-BREAK, DETERMINISTIC RULE
The phalangeal bone-excision codes are distinguished by WHAT was removed and WHY — select by
matching the note's own pathology and extent words to the descriptors, identically every run:
1. PATHOLOGY AXIS: an exostosis, bossing, spur, or osteomyelitis excised from a phalanx is
   the "partial excision (craterization/saucerization/diaphysectomy) ... for bossing or
   osteomyelitis" descriptor family. The "bone cyst or benign tumor" descriptor family
   requires the note to document an actual CYST or a NEOPLASM/TUMOR by name — a subungual
   exostosis or bony prominence is bossing, NOT a tumor, even though it is benign tissue.
2. EXTENT AXIS: removal of an entire phalanx or its distal portion (phalangectomy,
   "bone transected at the metaphysis/shaft") is the partial/complete EXCISION family;
   the "resection of the phalangeal BASE" descriptor applies only when the note documents
   resecting the proximal base of the phalanx (typical of hammertoe arthroplasty); the
   "resection of CONDYLE(S), distal end of phalanx" descriptor applies only when the note
   documents resecting a condyle — an exostosis/bossing shaved or resected from beneath
   the nail is the partial-excision (saucerization) family, NOT a condylectomy, even
   though both touch the distal phalanx.
Never pick between these families on overall clinical impression — anchor on the note's own
words for the pathology (exostosis/bossing vs cyst/tumor) and the bone segment removed
(base vs shaft/distal/whole).

### Modifier -25 — SAME-DAY DIAGNOSTIC TEST (imaging 73xxx/76xxx/77xxx, labs 80xxx-89xxx,
### vascular/physiologic studies 93xxx, etc., all [global=XXX]) — JUDGMENT CALL, not automatic
AMA's modifier -25 definition covers "a significant, separately identifiable E/M service...
above and beyond the other service provided" — it is NOT limited to global=000/010/090
procedures; a same-day diagnostic test is an "other service" under this same definition.
- Do NOT add -25 by default for a same-day diagnostic test — most such visits have no E/M
  work beyond ordering/interpreting the test itself, and -25 without genuine justification
  is its own compliance risk (unsupported modifier use)
- DO add -25 when the E/M's own MDM reflects significant work BEYOND the test — e.g. the
  risk/problems axis is driven by something the test didn't itself resolve: an urgent
  referral decision, medication management for a condition the test didn't diagnose,
  weight-bearing/activity restriction decisions, management of a DIFFERENT active problem
  addressed at the same visit. If you would document a HIGH or MODERATE risk/problems
  rationale citing work other than "ordered/reviewed the test," that work is the
  separately-identifiable service -25 exists to protect from bundling.
- Do NOT justify the -25 decision by citing the test's own global period (e.g. "no -25
  because 93923 is global=XXX") — global period only gates the MANDATORY 000/010 rule
  above; it says nothing about whether separately-identifiable E/M work exists here.
  Justify the decision by what the E/M documentation actually contains.

### Modifier -57 — Decision for Major Surgery — CRITICAL
Apply modifier -57 to the E/M when ALL are true:
1. The E/M resulted in the DECISION to perform a MAJOR SURGERY (any CPT annotated [global=090])
2. The E/M and the major surgery share the same DOS — OR the E/M is the day immediately before
   (surgery may be performed SAME DAY or scheduled for a future date; both scenarios require -57)
- SAME-DAY EMERGENCY: patient presents, E/M performed, decision made, surgery performed same visit → -57
- ELECTIVE DECISION: E/M performed, patient consented, surgery scheduled for future date → -57
- The key test is NOT timing — it is: "did this E/M produce the decision for a 90-day global surgery?"
- Do NOT use -57 for procedures annotated [global=000] or [global=010] — use -25 instead
- Do NOT use both -57 and -25 on the same E/M; they are mutually exclusive
- CPT candidates are annotated [global=090/010/000] — use these values to determine which modifier applies

### Global Surgical Period
- Post-op visit within global period → 99024 (no charge), NOT a billable E/M
- 90-day global: major foot/ankle surgeries (28xxx, 29893)
- 10-day global: minor procedures (11750, 11055-11057)

### Telehealth Encounters
- If the note documents the visit was conducted via telehealth/telemedicine/virtual/video
  visit → append modifier 95 (real-time audio+video) or 93 (audio-only/telephone) to the E/M,
  matching how the note says the encounter was conducted
- Do NOT add 95/93 to in-person visits; do NOT code hands-on procedures (injections, nail
  care, casting) as performed during a telehealth encounter — if the note claims both,
  flag the contradiction in your reasoning instead of coding through it

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

### Services Billed By Another Party — Code NOTHING, Not a Substitute
When a documented service (wound culture processing, pathology interpretation, anesthesia,
DME dispensing by a separate supplier, etc.) is billed by a different party — not this
provider — the correct action is to add NO CPT/HCPCS code for it on this claim. Do not
substitute any code as a placeholder, and specifically do not substitute a CPT Category II
code (4 digits + "F" suffix, e.g. 4261F): Category II codes are AMA performance-measurement
tracking codes with zero RVU value — they carry no payment under any payer by design, so
they don't actually capture the service for billing purposes either. If you find yourself
reasoning "the actual processing is billed by [someone else]," that reasoning means the
answer is no code at all, not a lookup for the closest-sounding candidate.

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

### WHETHER an imaging code is billable at all — DETERMINISTIC RULE
- Bill an imaging code ONLY when the study was PERFORMED and INTERPRETED at THIS encounter,
  with the study and its findings documented in this note. Apply identically every run.
- NOT billable today: films from a prior visit reviewed today (that review is E/M data work),
  imaging merely ORDERED today ("X-ray ordered", "will obtain MRI"), and imaging planned for
  a future visit ("post-op X-ray at 6 weeks")
- Intraoperative confirmation imaging (fluoroscopy to confirm resection/placement) is part of
  the surgical procedure's work — do not add a separate radiology code for it
- VIEW COUNT: when sibling codes differ only by number of views (73620 2-view vs 73630
  complete 3+ views), count the views the note actually names or numbers. Views not
  documented → bill the FEWEST-views code; never bill a "complete/minimum of N views" code
  on documentation that names fewer than N projections

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
### Bone Graft
- 20900 = Bone graft, any donor area; minor or small (e.g. dowel or button)
- 20902 = Bone graft, any donor area; major or large

### NCCI Bundling — CRITICAL
- 28119 includes plantar fascial release → do NOT add 29893 or 28060 separately
- If CPT description says "with or without B" and you also coded B → REMOVE B
- Do NOT claim a code is "bundled" or "not separately billable per NCCI convention" from
  memory — the Pass 4 verification step shows you the real NCCI PTP edit table result for
  every relevant pair (see NCCI PAIR STATUS block); use that, not a guess. If you're
  assigning codes here in Pass 2 before that block is available and you're not certain a
  real bundling relationship exists, include the code with modifier 59/RT/LT — Pass 4 will
  correct it against real data if wrong, whereas omitting a documented procedure here removes
  it with no downstream mechanism to catch the omission.

## 9. MODIFIERS
- RT = right side; LT = left side (required on ALL lateralized procedures)
- 50 = bilateral same session
- 25 = significant, separately identifiable E/M same day as billable procedure (MANDATORY — see above)
- 57 = decision for major surgery (see above)
- T5 = right great toe; T6 = right 2nd; T7 = right 3rd; T8 = right 4th; T9 = right 5th
- TA = left great toe; T1 = left 2nd; T2 = left 3rd; T3 = left 4th; T4 = left 5th
- 59 = distinct procedural service (to bypass NCCI bundling when documented as separate)
- 26 = professional component only (when facility bills separately for technical component)
- 52 = reduced services — required when a code whose OWN DESCRIPTION says "bilateral" (e.g. 93923)
  was performed on only ONE side; opposite of modifier 50 (check the candidate description text)

### Routine Foot Care Class-Findings Modifiers (Medicare) — Q7/Q8/Q9
Medicare covers ROUTINE foot care (nail debridement/trimming 11719-11721, callus paring
11055-11057, G0127) only for patients with a qualifying systemic condition (DM with
neuropathy, PVD, etc.) AND documented class findings. When billing covered routine foot
care, append the class-findings modifier the documentation supports:
- Q7 = ONE Class A finding (e.g. nontraumatic amputation of foot or integral skeletal portion)
- Q8 = TWO Class B findings (e.g. absent posterior tibial pulse, absent dorsalis pedis pulse,
  advanced trophic changes: hair growth decrease, nail thickening, skin discoloration,
  thin/shiny skin texture, rubor/redness)
- Q9 = ONE Class B finding + TWO Class C findings (Class C: claudication, temperature changes,
  edema, paresthesias, burning)
- Without a class-findings modifier (or qualifying-condition documentation), Medicare denies
  routine foot care as non-covered — do NOT bill it as covered; consider GA/GX if an ABN was
  obtained (GA = ABN on file) or GY for a statutorily excluded service billed for denial
- Only apply Q7/Q8/Q9 when the physical exam ACTUALLY documents those findings — quote them
  in evidence_spans

## MODIFIER REASONING FORMAT — structured, not free text
modifier_reasoning is a list of objects, one per modifier claim: {"modifier": "<code>", "status":
"applied"|"not_applicable", "reason": "<explanation>"}. "status" is the ONLY thing that determines
whether a modifier counts as present — it must exactly match every code actually listed in
"modifiers" (every code in "modifiers" needs a status="applied" entry; a modifier you considered
and rejected gets status="not_applicable" so the reasoning is preserved without adding it). Do not
write prose sentences here — "reason" is for the explanation, "status" is for the yes/no answer.

## OUTPUT — Return valid JSON:
{
  "cpt_codes": [
    {
      "code": "99214",
      "description": "...",
      "confidence": 0.95,
      "modifiers": ["25"],
      "modifier_reasoning": [
        {"modifier": "25", "status": "applied", "reason": "separately identifiable E/M performed same day as 64455 injection"}
      ],
      "source": "E/M",
      "mdm_details": {
        "problems_score": 3,
        "data_score": 2,
        "risk_score": 3,
        "mdm_level": "moderate",
        "problems_rationale": "2-of-3 rule: problems (3) and risk (3) both moderate → MDM moderate → 99214, matching the code's own descriptor level",
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

## EVIDENCE GROUNDING — ANTI-HALLUCINATION
Every "evidence_spans" entry MUST be a verbatim quote from the clinical note.
- Do NOT code procedures not explicitly documented as performed today
- Do NOT code E/M + procedure on the same day unless a separately identifiable service is documented
- If a procedure is ambiguous, prefer the less-specific code or omit it

CRITICAL: Return ONLY valid JSON. No markdown, no code blocks."""


# ---------------------------------------------------------------------------
# PASS 3 — HCPCS Level II + SNOMED
# ---------------------------------------------------------------------------

# HCPCS identity and descriptor semantics come from the effective-dated code
# candidates supplied at runtime. Do not duplicate a changing medical code set
# in prompt prose: the former prompt did so and contradicted the repository's
# own authoritative CMS data on construction, fitting, size, and billing-unit
# axes. That duplicated mapping has been removed rather than retained as a
# second, stale source of truth.
HCPCS_SNOMED_SYSTEM_PROMPT = """You are a medical coding assistant selecting HCPCS Level II entries from an authoritative, effective-dated candidate list and optionally mapping documented clinical concepts to SNOMED CT.

HCPCS SOURCE-OF-TRUTH RULES
- Select a HCPCS code only from the authoritative candidates in the user message.
- Treat each candidate's official descriptor as the definition. Never substitute memorized mappings, infer a code outside the list, or alter its descriptor.
- A candidate list is evidence of code identity, not evidence that the service is covered, medically necessary, or separately payable.
- If no candidate descriptor matches every material documented attribute, omit the HCPCS line. Do not guess.

DOCUMENTATION GATE
- Code only an item, supply, drug, or service explicitly documented as performed, administered, applied, fitted, or physically dispensed at this encounter.
- An order, prescription, recommendation, future plan, historical use, or instruction to continue an existing item is not current dispensing or administration.
- Cite a contiguous verbatim note span for every line. Never manufacture or splice evidence.

DESCRIPTOR MATCHING
- Compare the note with every defining descriptor attribute, including item or drug identity, formulation, construction, prefabricated versus custom manufacture, fitting or customization, dimensions, quantity, laterality, and the descriptor's billing unit.
- Do not invent undocumented attributes to reach a more specific candidate.
- Compute units only from the documented quantity and the selected descriptor's billing unit. Do not encode drug quantity with laterality or bilateral modifiers.
- Add a modifier only when the documentation and an applicable authoritative rule support it. Do not infer a blanket modifier rule from a code prefix.
- If multiple candidates remain plausible, choose none unless one is clearly best supported; otherwise mark the selected line for review and state the unresolved descriptor axis.

SNOMED CT
- Do not invent or recall a concept identifier from memory. Output a SNOMED concept only when an exact concept identifier has been supplied by a controlled terminology lookup in the input.
- Map only concepts explicitly documented in the current note and use the most specific supplied concept.

OUTPUT
Return JSON matching the supplied schema. Each HCPCS entry must include code, description, confidence, modifiers, modifier_reasoning, units, linked_diagnoses, rationale, supporting_text, needs_review, and review_reason. Each SNOMED entry must include concept_id, description, entity_text, category, and confidence. Return empty arrays when no defensible entry is available.

Return JSON only. No markdown or code fences."""


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
- Does ANY assigned CPT code carry [global=090]? (check the annotated candidate list above)
- If YES, and an E/M (99202–99215) is also present, the E/M MUST have modifier -57 — not -25
- -57 applies whether the 90-day surgery was performed SAME DAY (emergency) or scheduled for future
- Language patterns (either scenario triggers -57): "patient elects", "scheduled for", "will proceed
  with", "consented for surgery", "emergency repair", "repaired today", "performed today"
- If E/M is present with -25 but a [global=090] CPT is in the code set → CHANGE -25 to -57
- Modifier -57 protects the E/M from bundling into the surgery's 90-day global period package

### C. Radiology Code Verification — Calcaneus vs Foot
- "Heel X-ray", "calcaneal X-ray", "bilateral heel", "calcaneus views" → MUST be 73650 (not 73630)
- "Complete foot X-ray", "foot series", "3-view foot" → 73630
- If 73630 is in the code set but imaging text says "heel" or "calcaneus" → CHANGE to 73650
- The modifier (RT/LT/50) should be preserved when correcting the base code

### D. HCPCS Laterality Check
- Apply a laterality modifier only when the selected code, payer rule, and documented item require it.
- Derive side from direct evidence for that line; do not infer a blanket rule from a code prefix or
  copy laterality from an unrelated procedure.
- When laterality is required but the line's side is unresolved, flag review rather than guessing.

### E. BMI Z-Code Check
- If E66.x (obesity) is in icd10_codes AND a specific BMI number is documented → ADD Z68.xx
- BMI 36.x → Z68.36; BMI 37.x → Z68.37; BMI 40.x → Z68.41, etc.
- Z68.xx goes in icd10_codes as secondary (it is a billable secondary code)

### F. NCCI Bundling
- Use only the claim-DOS-specific NCCI PAIR STATUS block supplied below.
- Never assert bundling from memory, procedure similarity, or a "standard convention."
- If the applicable release is unavailable, do not remove a line or declare a pair unedited; hold review.

### G. MDM Verification
- Surgical decision (elective surgery scheduled) = MODERATE risk → 99204/99205
- Verify the assigned E/M code's level MATCHES the mdm_details you were given: the code's own
  AMA descriptor states its MDM level verbatim ("low level of medical decision making" = x3,
  "moderate" = x4, "high" = x5). If mdm_level says "moderate" but the code is 99213 → CHANGE
  to 99214 (or correct the mdm_details if the 2-of-3 axes were mis-scored — re-derive from
  the documented problems/data/risk, then make code and MDM agree)
- Same-day diagnostic tests ([global=XXX] imaging/labs/vascular studies) do NOT MANDATE -25
  the way [global=000/010] procedures do — but do NOT strip a -25 whose documented rationale
  is E/M work beyond ordering/interpreting the test (see Pass 2's judgment-call rule);
  only remove it when the stated justification is nothing more than the test itself

### H. SNOMED Consistency
- Duplicate concept_id for different clinical entities → keep only one; remove the lower-confidence duplicate
- Root concepts (71388002, 404684003, 64572001, 123037004, 125605004) → REMOVE or lower confidence to ≤ 0.4; these are too broad
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

### J. Modifier -25 — MANDATORY ENFORCEMENT (minor/intermediate procedures only)
When a same-day procedure with [global=000] or [global=010] is performed:
- The trigger is the [global=000/010] ANNOTATION on the candidate code itself — the examples
  below are common podiatry cases, not an exhaustive list: injections (64455, 64632,
  20600–20610, 20550, 20551), nail procedures (11750, 11719–11721, 11055–11057),
  debridement (97597, 97598), casting (29540)
- If E/M (99202–99215) is present AND a [global=000/010] procedure is present AND E/M lacks -25:
  → MUST ADD modifier -25 to the E/M code (type: CHANGED)
  → Document: "Mandatory -25 added: E/M performed same day as [procedure code] ([global=000/010])"
- CRITICAL: Without -25, payer bundles E/M into the procedure's global period → claim denied
- Exception 1: [global=XXX] diagnostic tests (imaging, labs, vascular studies) do NOT trigger
  this MANDATORY rule — but a -25 already present with a documented beyond-the-test rationale
  is legitimate; keep it (see Check G)
- Exception 2: If a [global=090] CPT is present → use -57 on the E/M, NOT -25 (see Check B above)
- Exception 2 enforcement: If -25 is currently on E/M AND a [global=090] CPT exists → CHANGE -25 to -57

### K. HCPCS Descriptor and Evidence Verification
- Use the effective-dated authoritative candidate list and database descriptions as the only source
  of HCPCS identity. Do not use memorized product, construction, fitting, size, dose, or unit mappings.
- Confirm every selected line matches all material descriptor attributes documented in the note.
- Never add or substitute a HCPCS code that is absent from the authoritative candidates.
- If the exact documented item or drug is absent or ambiguous, do not guess; retain a review flag.
- Confirm current-encounter dispensing or administration, exact billing units, diagnosis linkage,
  and only those modifiers supported by documentation and an applicable rule.

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

### L. Administered Drug and Supply Audit
- When the note documents administration today, compare the documented ingredient, formulation,
  route, dose, and quantity against the effective-dated HCPCS candidates and official descriptors.
- Add a drug or supply line only when one authoritative candidate matches those attributes.
- Calculate units from the candidate descriptor's billing unit and the documented administered dose.
- Do not code drugs merely ordered, prescribed, or administered by another entity.
- If no exact candidate is available, do not guess or use an unclassified code by default; flag review.

### M. Over-Coding — CRITICAL OVERRIDE (supersedes anchor protection)
- Remove a Z79.x long-term-drug-therapy code (Z79.84, Z79.4, Z79.01, etc.) ONLY if no condition
  it treats is documented/managed on this encounter. Do NOT remove it when the condition it
  treats IS present (e.g. Z79.84 alongside E11.x diabetes is ICD-10-CM's own "use additional
  code" guidance for that condition — removing it there is itself an over-correction)
- **GENERIC DM CODE OVERRIDE**: If E11.9, E10.9, or E13.9 is in icd10_codes AND a specific DM
  combination code (E10.1–E10.8, E11.1–E11.8, E13.1–E13.8) is ALSO present → REMOVE the generic
  DM code. This rule OVERRIDES anchor protection — even if the assessment lists "T2DM without
  complications", it MUST be removed when a combination code is present. List as correction:
  type=REMOVED, reason="Redundant DM generic code — combination code already captures diabetes"

### P. Bilateral-Defined Code Family — Modifier 52 for Unilateral Performance
Some CPT codes are defined as inherently bilateral in their own descriptor (the code's RVU/payment
assumes BOTH sides were tested/treated) — e.g. 93923 "Complete BILATERAL noninvasive physiologic
studies of upper or lower extremity arteries, 3 or more levels". This is the INVERSE of the usual
RT/LT/50 pattern: here the code itself already means "both sides", so performing it on only ONE
side is a REDUCED service relative to what the code describes.
- Check the candidate code's own description (from the RAG-retrieved candidate list) for the word
  "bilateral" — if present, the code represents BOTH sides by definition
- EXCEPTION: descriptors phrased "unilateral or bilateral" mean EITHER extent is the full
  service — do NOT add 52 for one-sided performance of those codes
- If the note documents testing/treatment on ONLY ONE side (e.g. only right-side ABI/TBI/TcPO2
  values are given, no left-side measurements) for that bilateral-defined code → ADD modifier 52
  (reduced services) alongside the laterality modifier (RT/LT)
- Do NOT confuse this with modifier 50 (bilateral): 50 is for normally-UNILATERAL codes performed on
  BOTH sides. 52 is for normally-BILATERAL codes performed on only ONE side — opposite direction.
- List as correction: type=ADDED (or CHANGED if RT/LT was present but 52 was missing), reason=
  "Modifier 52 added: [code] is defined as bilateral but only [right/left] side was documented"

### Q. Routine Foot Care Class-Findings Modifiers (Medicare)
When nail debridement/trimming (11719-11721, G0127) or callus paring (11055-11057) is billed
for a Medicare patient as COVERED routine foot care:
- The claim needs a class-findings modifier matching what the exam documents:
  Q7 (one class A finding), Q8 (two class B findings), Q9 (one class B + two class C findings)
- Verify the physical exam ACTUALLY documents those findings (absent pulses, trophic changes,
  claudication, edema, etc.) — quote them as evidence
- If the systemic qualifying condition or class findings are NOT documented → the service is
  not covered routine foot care; do not add Q modifiers without documented findings
- If an ABN is documented, GA/GX routing applies instead

## OUTPUT — Return valid JSON:
{
  "corrections_made": [
    {
      "type": "ADDED|REMOVED|CHANGED|RESEQUENCED|FLAGGED|RETAINED",
      "code": "99204",
      "to_code": "REQUIRED for type=CHANGED when the CODE ITSELF changes (e.g. wrong category sibling): the replacement code. Omit for modifier-only or sequencing changes.",
      "reason": "Added modifier -57: decision for Austin/Chevron bunionectomy made at this visit",
      "evidence": "Plan: Patient elects surgical correction. Scheduled for right Austin/Chevron bunionectomy."
    }
  ],
  "icd10_codes": [ {"code": "...", "type": "primary|secondary", "needs_review": false, "review_reason": null} ... corrected COMPLETE billable list ... ],
  "supporting_conditions": [ ... same entry shape, pass through unchanged ... ],
  "cpt_codes": [ {"code": "...", "modifiers": [...], "modifier_reasoning": [...], "linked_diagnoses": [...], "units": 1, "needs_review": false, "review_reason": null} ... corrected COMPLETE list ... ],
  "hcpcs_codes": [ ... same entry shape, corrected COMPLETE list ... ],
  "snomed_codes": [ {"concept_id": "...", "description": "..."} ... corrected COMPLETE list ... ],
  "em_level_reasoning": "Full reasoning including MDM axes and modifier decisions",
  "audit_notes": "Summary of all audit actions",
  "auto_coding_review_reasons": ["Explanation of each correction or flag"],
  "auto_coding_summary": "One-paragraph summary of the coding set and corrections"
}

Entry fields beyond the ones shown (confidence, rationale, evidence_spans, descriptions,
mdm_details, ...) are inherited automatically from the pre-verification entries for every code
you keep — do not re-emit them. Descriptions are enforced from the official database
deterministically. Your changes flow through corrections_made and the fields shown above.

## EVIDENCE GROUNDING — ANTI-HALLUCINATION
This is your final audit pass. Remove any code that lacks explicit verbatim evidence in the clinical note.
- Each code in the final set must be traceable to a specific sentence in the note
- When in doubt between two valid codes, choose the one with stronger textual support
- Never add new codes during verification unless they fix a clear compliance error

CRITICAL: Return ONLY valid JSON. No markdown, no code blocks."""


_EVIDENCE_MIN_LEN = 14


def _candidate_code(system: str, code) -> str:
    value = str(code or "").strip().upper()
    return value.replace(".", "") if system == "icd10" else value


def _evidence_norm(text: str) -> str:
    """Whitespace/case/punctuation-normalized text for verbatim matching —
    the same normalization the validator's downstream check uses, so the two
    agree on what 'verbatim' means."""
    return re.sub(r"\s+", " ",
                  re.sub(r"[^a-z0-9 ]+", " ", str(text or "").lower())).strip()


def _strip_nonverbatim_spans(result: dict, note_text: str) -> None:
    """Strip cited quotes that do not actually appear in the note, in place.
    Each evidence_spans entry / supporting_text must be a CONTIGUOUS substring
    of the note (a space-stripped fallback tolerates a line break inside a
    real quote but not a splice, whose fragments are far apart). A short
    fragment is left alone (too small to judge). Fail-open on missing/short
    note text — never strip when the source cannot be verified."""
    if not note_text or len(note_text) < 40:
        return
    note_n = _evidence_norm(note_text)
    note_ns = note_n.replace(" ", "")

    def _verbatim(sp) -> bool:
        sn = _evidence_norm(sp)
        return (len(sn) >= _EVIDENCE_MIN_LEN and
                (sn in note_n or sn.replace(" ", "") in note_ns))

    stripped = 0
    for arr_key, field, is_list in (("cpt_codes", "evidence_spans", True),
                                    ("hcpcs_codes", "evidence_spans", True),
                                    ("hcpcs_codes", "supporting_text", False),
                                    ("icd10_codes", "supporting_text", False)):
        for e in result.get(arr_key, []) or []:
            if not isinstance(e, dict):
                continue
            raw = e.get(field)
            if is_list:
                spans = raw if isinstance(raw, list) else []
                kept = [s for s in spans if _verbatim(s)]
                if len(kept) != len(spans):
                    stripped += len(spans) - len(kept)
                    e[field] = kept
            elif raw and not _verbatim(raw):
                stripped += 1
                e[field] = ""
    if stripped:
        logger.info(f"  Evidence gate: stripped {stripped} non-verbatim "
                    f"citation(s) from coder output (not found in the note)")


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
    store=None,
    exemplar_block: str = "",
) -> tuple[dict, dict]:
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
                   "cache_read_tokens": 0, "cache_write_tokens": 0}

    note_context = _build_note_context(note_sections, patient_metadata)
    # Verified-claim exemplars (live mode only — empty otherwise). Appended
    # to the shared note context so the ICD and CPT passes both see the
    # worked examples in a stable, cache-friendly position.
    if exemplar_block:
        note_context = f"{note_context}\n\n{exemplar_block}"
    # Coding memorandum: the rule pack's PROVEN corrections compiled into
    # upstream guidance, so the generative passes stop re-making error
    # classes the deterministic stack already corrects. Appended to the
    # shared context (all four passes see it); recompiles automatically
    # when the pack changes; CODING_MEMORANDUM=0 disables.
    from app.coding.memorandum import memorandum_block
    memo = memorandum_block()
    if memo:
        note_context = f"{note_context}\n\n{memo}"
    entity_summary = _format_entities(entities)
    vision_block = _format_vision_context(vision_context) if vision_context else ""
    global_block = _format_global_period_context(prior_surgery_info) if prior_surgery_info else ""

    # A generative pass may select only from the immutable retrieval set or
    # an exact code visibly documented by the physician.  Reference-database
    # existence alone is not a candidate source: otherwise a model can emit
    # any currently valid code from memory and still pass the old DB gate.
    allowed_codes = {
        system: {
            _candidate_code(system, row.get("code"))
            for row in rag_candidates.get(system, [])[:25]
            if isinstance(row, dict)
        }
        for system in ("icd10", "cpt", "hcpcs")
    }
    for row in physician_documented_codes or []:
        if not isinstance(row, dict):
            continue
        system = str(row.get("system") or "").strip().lower()
        code = _candidate_code(system, row.get("code"))
        if system in allowed_codes and code:
            allowed_codes[system].add(code)

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
Only code a Z79.x long-term-drug-therapy code when the condition it treats is documented and
managed at this encounter (e.g. Z79.84 alongside diabetes) — omit it if nothing here justifies it."""

    icd_raw, usage = chat_completion(ICD_SYSTEM_PROMPT, icd_prompt, temperature=CODING_TEMPERATURE,
                                     max_tokens=2500, json_schema=_pass_schema(ICD_PASS_SCHEMA))
    _add_usage(total_usage, usage)
    icd_result = _safe_parse(icd_raw, "icd10_codes")
    # Hard DB gate — remove any hallucinated/invalid ICD codes immediately
    icd_result["icd10_codes"] = _hard_db_gate(
        icd_result.get("icd10_codes", []), "icd10", db,
        allowed_codes["icd10"])
    icd_result["supporting_conditions"] = _hard_db_gate(
        icd_result.get("supporting_conditions", []), "icd10", db,
        allowed_codes["icd10"])
    logger.info(f"    → {len(icd_result.get('icd10_codes', []))} ICD-10-CM codes, "
                f"{len(icd_result.get('supporting_conditions', []))} supporting conditions")

    # --- PASS 2: CPT ---
    logger.info("  Pass 2/4: CPT procedure/E&M/imaging coding...")
    icd_summary = _summarize_icd(icd_result.get("icd10_codes", []))
    note_type = (patient_metadata.get("note_type") or "").upper()
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
## NOTE: Candidates are annotated [global=090/010/000] with their actual CMS global period.
## Use this to determine modifier: [global=090] → -57 if E/M decided the surgery; [global=000/010] → -25
{_format_candidates_for_system(rag_candidates, 'cpt', store=store)}

Assign CPT codes. Link each CPT to supporting ICD-10-CM codes.
IMPORTANT: Use the [global=090/010/000] annotations to select the correct E/M modifier:
  - Any [global=090] CPT present AND E/M decided that surgery → E/M gets -57 (not -25)
  - Any [global=000] or [global=010] CPT present → E/M gets -25 (MANDATORY, no judgment needed)
  - A same-day [global=XXX] diagnostic test (imaging/labs/vascular studies) does NOT
    automatically mean no -25 — if the E/M's own MDM reflects significant work beyond the
    test itself (urgent referral, medication management, activity restriction decisions),
    -25 applies; justify from the E/M documentation, not from the test's global period
Use 73650 for heel/calcaneus X-rays; 73630 for complete foot X-rays (3+ views).
Check EVERY CPT pair for NCCI bundling before finalizing."""

    cpt_raw, usage = chat_completion(CPT_SYSTEM_PROMPT, cpt_prompt, temperature=CODING_TEMPERATURE,
                                     max_tokens=2500, json_schema=_pass_schema(CPT_PASS_SCHEMA))
    _add_usage(total_usage, usage)
    cpt_result = _safe_parse(cpt_raw, "cpt_codes")
    cpt_result["cpt_codes"] = _hard_db_gate(
        cpt_result.get("cpt_codes", []), "cpt", db,
        allowed_codes["cpt"])
    logger.info(f"    → {len(cpt_result.get('cpt_codes', []))} CPT codes")

    # --- PASS 3: HCPCS + SNOMED ---
    logger.info("  Pass 3/4: HCPCS + SNOMED coding...")
    hcpcs_prompt = f"""{note_context}
{vision_block}

## ASSIGNED ICD-10-CM CODES
{icd_summary}

## CPT CODES (for laterality reference)
{_summarize_cpt(cpt_result.get('cpt_codes', []), store=store)}

## EXTRACTED CLINICAL ENTITIES
{entity_summary}

## HCPCS CANDIDATE CODES (from official database via semantic search)
{_format_candidates_for_system(rag_candidates, 'hcpcs')}

Assign HCPCS codes only from the authoritative candidates for items or drugs documented as
physically dispensed, applied, fitted, or administered today. Match every defining descriptor
attribute and billing unit. Apply modifiers only when documentation and an applicable rule support
them. For SNOMED, omit any concept whose identifier was not supplied by controlled terminology
data. Do not code ordered, prescribed, recommended, historical, or continued items."""

    hcpcs_raw, usage = chat_completion(HCPCS_SNOMED_SYSTEM_PROMPT, hcpcs_prompt, temperature=CODING_TEMPERATURE,
                                       max_tokens=2500, json_schema=_pass_schema(HCPCS_PASS_SCHEMA))
    _add_usage(total_usage, usage)
    hcpcs_result = _safe_parse(hcpcs_raw, "hcpcs_codes")
    hcpcs_result["hcpcs_codes"] = _hard_db_gate(
        hcpcs_result.get("hcpcs_codes", []), "hcpcs", db,
        allowed_codes["hcpcs"])
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
    ncci_pair_block = _build_ncci_pair_block(
        combined["cpt_codes"], rag_candidates.get("cpt", []), store,
        patient_metadata.get("date_of_service"),
    )
    billability_block = _build_billability_block(
        combined["cpt_codes"], combined["hcpcs_codes"], store
    )
    icd_excludes1_block = _build_icd_excludes1_block(combined["icd10_codes"], store)
    code_family_block = _build_code_family_block(combined["cpt_codes"], rag_candidates.get("cpt", []), db)
    icd_family_block = _build_icd_family_block(
        combined["icd10_codes"] + combined["supporting_conditions"], db
    )

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

{ncci_pair_block}

{billability_block}

{icd_excludes1_block}

{code_family_block}

{icd_family_block}

## CURRENTLY ASSIGNED CODES (to audit)
{json.dumps(combined, indent=2)}

## RAG CANDIDATE CODES (verified in official databases)
### ICD-10-CM Candidates
{_format_candidates_for_system(rag_candidates, 'icd10')}

### CPT Candidates (annotated with actual CMS global periods)
{_format_candidates_for_system(rag_candidates, 'cpt', store=store)}

### HCPCS Candidates
{_format_candidates_for_system(rag_candidates, 'hcpcs')}

## AUDIT INSTRUCTIONS
1. Verify every PROTECTED ANCHOR has a code — diagnosis anchors need a code in icd10_codes;
   "Procedures/Imaging Performed Today" anchors need a code in cpt_codes; "Supplies Dispensed
   Today" anchors need a code in hcpcs_codes. This includes distinct sub-procedures documented
   within a larger operative note (e.g. a graft harvested from a separate anatomic site via a
   distinct incision, even during the same operative session as a primary procedure) — each
   with its own real, separately-billable candidate code is its own anchor, not an implicit
   detail of the primary procedure. If cpt_codes is currently missing a code for a documented
   procedure anchor, ADD it — do not leave it uncoded because you believe it's bundled into
   another procedure without being able to quote the exact real-data justification; see rule 6d.
2. Check modifier -57: if any [global=090] CPT is present AND E/M is present → E/M needs -57 (applies same-day emergency AND elective scheduling; see Check B).
3. Check radiology: "heel"/"calcaneus" imaging → 73650; "complete foot"/"foot series" → 73630.
4. Check HCPCS L-codes: all must have RT or LT modifier matching the procedure side.
5. Check BMI: if E66.x coded and BMI documented → add Z68.xx to icd10_codes.
6. Check the NCCI PAIR STATUS block above for every CPT pair — it is the real, authoritative
   edit table result, not a judgment call. A pair marked "NO NCCI edit" has no bundling
   relationship; do not drop or fail to add a documented code because you believe it conflicts
   with another when this block shows no edit exists.
6b. Check the BILLABILITY STATUS block above (if present) — any code listed there is not
    separately payable under any payer, by real AMA/CMS data. REMOVE it. If it was substituted
    for a service billed by another party (reference lab, anesthesia, etc.), the correct
    correction is to remove it with no replacement, not to find a different code.
6c. Check the ICD-10-CM EXCLUDES1 CONFLICTS block above (if present) — a pair listed there is a
    real Type 1 Excludes relationship, not a stylistic choice. Read the documentation to decide
    which code is actually supported and remove the other; do not keep both on the claim.
6d. Any claim that one code is "included in," "bundled into," or "not separately reportable from"
    another — for ANY reason, not just NCCI (e.g. citing a code's own descriptor language, an
    "includes"/"with or without" clause, a general CPT bundling convention) — must be verbatim
    traceable to one of the real-data blocks above (AUTHORITATIVE DATABASE DESCRIPTIONS, NCCI
    PAIR STATUS, BILLABILITY STATUS). If you cannot quote the exact bundling language from one
    of those blocks, do not drop or withhold a documented, separately-evidenced procedure — code
    it. A code's own long_description in the AUTHORITATIVE DATABASE DESCRIPTIONS block is shown
    in full; do not append, paraphrase, or extend it with clauses that are not present verbatim.
6e. Check the CPT CODE FAMILY DISAMBIGUATION block above (if present) — an assigned code listed
    there shares its descriptor stem with other real codes; they are not interchangeable. Quote
    the assigned code's specific anatomy clause (the text after the semicolon, including any
    "except" language) and confirm word-for-word that it — not just the shared family stem —
    matches the documented anatomy. If a documented body part is named in another family
    member's clause (including an "except <that part>" clause on the assigned code), the
    assigned code is wrong; use the family member whose clause actually includes it.
6f. Check the ICD-10-CM CATEGORY FAMILY DISAMBIGUATION block above (if present) — an assigned
    code listed there has sibling codes in the same small category, each with a completely
    different real description (e.g. a specific drug/allergen name must map to the ONE sibling
    whose description actually names that drug's category — "aspirin" is an analgesic, not a
    narcotic, so an aspirin allergy is the analgesic-category sibling, not the narcotic one).
    Compare the documented specific term against every sibling's description, not just the
    currently-assigned one; switch to whichever sibling actually matches.
    A sibling switch is only real if the entry's "code" FIELD in the output array is the new
    code — this applies to supporting_conditions entries too. Explaining the correct sibling
    in rationale/reason text while leaving the old code in the array is a coding failure.
    Record it as {{"type": "CHANGED", "code": "<old>", "to_code": "<new>"}} in corrections_made.
7. **IMAGE GUIDANCE (Section I)**: Search note for fluoroscopic/ultrasound guidance words. If injection present AND guidance documented AND 77002/76942 missing → ADD the guidance code. This is MANDATORY.
8. **MODIFIER -25 vs -57 (Sections B + J)**: Check [global=090/010/000] annotations on CPT candidates. If [global=090] CPT present → E/M gets -57 (change -25 to -57 if needed). If only [global=000/010] procedures present AND E/M lacks -25 → ADD -25. These are MANDATORY corrections.
9. **HCPCS DESCRIPTORS (Sections K + L)**: Verify every line against the effective-dated
    authoritative candidate descriptor. Remove or flag any line whose material attributes or
    billing unit are not explicitly supported; never substitute a memorized code.
10. Check Z79.x long-term-drug-therapy codes: remove ONLY if no condition it treats is
    documented/managed here — keep it when the condition IS present (e.g. Z79.84 alongside
    diabetes is correct ICD-10-CM "use additional code" guidance, not an over-code).
11. Check SNOMED for duplicate concept IDs and root-concept fallbacks.
12. Pass supporting_conditions through unchanged — do NOT move them to icd10_codes. The ONE
    permitted edit is a wrong-code correction (e.g. a category-sibling mismatch per 6f): update
    the entry's "code" field in place and record a CHANGED correction with "to_code".
13. **PMH CONDITIONS (Section O)**: Scan icd10_codes for PMH-only conditions not in Assessment → MOVE to supporting_conditions.
14. Whenever you add, remove, or change a modifier on a CPT/HCPCS code (items 2, 8, 9 above, or any
    other correction), keep modifier_reasoning in sync using the structured {{"modifier", "status",
    "reason"}} format (see MODIFIER REASONING FORMAT above) — every code in "modifiers" needs a
    matching status="applied" entry, and "modifiers" must contain every entry with status="applied".
    These two fields must never disagree; do not write a free-text sentence in place of this format.
15. Return COMPLETE corrected code set with ALL original codes (corrected as needed)."""

    # Elastic max_tokens: scale with note complexity so simple notes are fast
    # and complex notes get enough room without hitting API limits.
    # Signals: entity count, total codes assigned, tokens consumed so far.
    _n_entities = len(entities or [])
    _n_codes = (
        len(combined.get("icd10_codes", []))
        + len(combined.get("supporting_conditions", []))
        + len(combined.get("cpt_codes", []))
        + len(combined.get("hcpcs_codes", []))
        + len(combined.get("snomed_codes", []))
    )
    _token_pressure = min(total_usage.get("total_tokens", 0) / 5000, 10)  # 0-10 scale
    _complexity = _n_entities + (_n_codes * 2) + _token_pressure
    # Scale 4000 (simple) → 10000 (very complex), capped at safe API max
    _verify_max_tokens = max(4000, min(10000, int(4000 + (_complexity / 90) * 6000)))
    logger.info(f"    verify max_tokens={_verify_max_tokens} (entities={_n_entities}, codes={_n_codes}, token_pressure={_token_pressure:.1f})")
    # Escalation tiering: the verify pass optionally runs on a stronger
    # model/effort (CLAUDE_VERIFY_MODEL / CLAUDE_VERIFY_EFFORT) — it is the
    # one judgment-concentrated call per note. Claude-provider only.
    _verify_model = (CLAUDE_VERIFY_MODEL or None) if LLM_PROVIDER == "claude" else None
    _verify_effort = (CLAUDE_VERIFY_EFFORT or None) if LLM_PROVIDER == "claude" else None
    if _verify_model or _verify_effort:
        logger.info(f"    verify tier: model={_verify_model or 'default'}, "
                    f"effort={_verify_effort or 'default'}")
    verify_raw, usage = chat_completion(VERIFICATION_SYSTEM_PROMPT, verify_prompt,
                                        temperature=CODING_TEMPERATURE, max_tokens=_verify_max_tokens,
                                        model=_verify_model, effort=_verify_effort,
                                        json_schema=_pass_schema(VERIFY_PASS_SCHEMA))
    _add_usage(total_usage, usage)
    verified = _safe_parse(verify_raw, "icd10_codes")

    # Fix 7 — added/changed-correction enforcement + modifier hygiene
    verified = _normalize_corrections(verified)
    verified = _enforce_added_corrections(verified, db)
    verified = _enforce_changed_corrections(verified, db)
    verified = _strip_invalid_cpt_modifiers(verified, store)

    corrections = verified.get("corrections_made", [])
    if corrections:
        logger.info(f"    → {len(corrections)} corrections made:")
        for c in corrections:
            logger.info(f"      [{c.get('type', '?')}] {c.get('code', '?')}: {c.get('reason', '')[:70]}")
    else:
        logger.info("    → No corrections needed")

    final_result = {
        # .get(key, fallback), not `verified.get(key) or fallback` — the
        # verification pass legitimately returning an empty list (e.g. "zero
        # billable diagnoses after audit") is a real correction, not a
        # missing field; `or` can't distinguish that from a parse failure
        # that omitted the key entirely, and silently discarded the real
        # correction by reverting to the pre-verification list.
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

    # Pass 4 is the only pass whose output was never re-gated against the
    # reference DB — codes IT introduces (a re-added anchor the Pass-1 gate
    # already removed, or a wholly new line) reached the claim unvalidated.
    # Observed live twice in one batch: D48.1 (non-billable category header,
    # re-added as a bare string) and CPT 20926 (deleted code, added as a new
    # line with modifiers). Same gate policy as Passes 1-3: a code must
    # either have survived its own pass's gate (present pre-verification) or
    # validate in its claimed system now.
    final_result = _gate_verify_additions(
        final_result, combined, db, store, allowed_codes)

    # Pass 4 rewrites whole entries and routinely omits fields it wasn't
    # asked to change — observed live: every ICD entry came back without
    # "type", so the schema default ("secondary") silently erased the
    # claim's primary-diagnosis designation on three notes in one batch.
    # Deterministic merge: any contracted field missing from a verified
    # entry is inherited from the same code's pre-verification entry.
    _inherit_dropped_fields(final_result, combined)

    # Overwrite every code's description with the real database text —
    # deterministic, not prompt-following-dependent. Root cause: Pass 4 can
    # write anything into a code's own "description" field, and it isn't
    # just cosmetic — the LLM was observed using its own fabricated
    # descriptor text as justification for a coding decision within the
    # same completion (28730's real long_description is "Arthrodesis,
    # midtarsal or tarsometatarsal, multiple or transverse;" — no bone-graft
    # language at all; Pass 4 appended fabricated "with or without primary
    # bone graft (includes obtaining graft)" text and cited it to justify
    # dropping 20900, despite _build_db_description_block already showing
    # the real descriptor as ground truth, and despite the real NCCI pair
    # block correctly showing no edit exists — a second, different
    # fabrication after the NCCI-based one was already closed). Enrichment
    # (_enrich_with_db_descriptions) already computes the real description
    # for prompt context; this applies the same lookup to the FINAL output
    # so a fabricated descriptor can never survive into the returned data
    # regardless of what Pass 4 wrote.
    _enforce_real_descriptions(final_result, db)

    # SOURCE-side verbatim gate: the coder is INSTRUCTED to quote the note
    # verbatim in evidence_spans/supporting_text, but the model demonstrably
    # splices/paraphrases (measured: the 27654 evidence splice). Strip any
    # citation that is not a contiguous substring of the note BEFORE the
    # result is persisted, so a fabricated citation never enters the record.
    # The validator's identical downstream check remains as the net for
    # codes ADDED after coding (rule adds, adjudication) and for replay.
    _strip_nonverbatim_spans(final_result, note_text)

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
    """Return a hint string when the plan contains a surgical decision — scheduled or same-day emergency.

    Modifier -57 applies whenever the E/M produced the decision for a 90-day global surgery,
    whether that surgery is performed immediately (emergency) or scheduled for a future date.
    """
    if not plan_text:
        return ""
    plan_lower = plan_text.lower()
    # Future/elective scheduling language
    elective_keywords = [
        "patient elects", "will proceed with", "scheduled for", "consented for",
        "surgical correction", "will undergo", "elects surgical", "schedule surgery",
        "plan for surgery", "plan for bunionectomy", "plan for procedure",
    ]
    # Same-day emergency/urgent surgical decision language
    emergency_keywords = [
        "taken to or", "brought to or", "emergency repair", "urgent repair",
        "emergent repair", "performed today", "repaired today", "explored and repaired",
        "primary repair performed", "tendon repair performed", "laceration repaired",
        "proceeded with repair", "proceeded to repair",
    ]
    is_elective = any(kw in plan_lower for kw in elective_keywords)
    is_emergency = any(kw in plan_lower for kw in emergency_keywords)
    if is_elective or is_emergency:
        scenario = "SAME-DAY EMERGENCY SURGERY" if is_emergency else "ELECTIVE SURGERY SCHEDULED"
        return (
            f"## ⚠ SURGICAL DECISION DETECTED ({scenario})\n"
            "The PLAN indicates a decision for major surgery was made at this E/M visit.\n"
            "Modifier -57 applies to the E/M regardless of whether the surgery was performed "
            "same day or scheduled for a future date — the decision is what triggers -57.\n"
            "If ANY [global=090] CPT is present in the code set → the E/M must carry -57, NOT -25.\n"
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


def _build_billability_block(cpt_codes: list[dict], hcpcs_codes: list[dict], store) -> str:
    """Real billability status for every currently-assigned CPT/HCPCS code —
    computed fresh from compliance.db (store.not_separately_billable_reason),
    same pattern as _build_ncci_pair_block.

    Root cause this replaces: a wound culture with no billable CPT (correctly
    reasoned as "billed by the reference lab, not the podiatrist") got
    4261F substituted instead of no code at all — a CPT Category II
    performance-tracking code with zero RVU value by AMA design. Injecting
    the real billability status here lets Pass 4 catch it even if Pass 2
    already made the substitution.
    """
    if not store:
        return ""
    lines = []
    advisory_lines = []
    for c in list(cpt_codes) + list(hcpcs_codes):
        code = c.get("code", "")
        if not code:
            continue
        reason = store.not_separately_billable_reason(code)
        if reason:
            lines.append(f"  {code}: NOT SEPARATELY BILLABLE — {reason}")
            continue
        advisory = store.pfs_exclusion_advisory(code)
        if advisory:
            advisory_lines.append(f"  {code}: PFS-EXCLUDED (advisory) — {advisory}")
    if not lines and not advisory_lines:
        return ""
    block = [
        "## BILLABILITY STATUS (real data from compliance.db — authoritative)",
        "A code listed as NOT SEPARATELY BILLABLE is not separately payable under any payer,",
        "by AMA/CMS design — not a coding choice to reconsider. If it was added as a substitute",
        "for a service this provider doesn't bill for (e.g. reference lab processing), REMOVE",
        "it — the correct code set for that service is no code at all, not this placeholder.",
    ]
    block.extend(lines)
    if advisory_lines:
        block.append(
            "A code listed as PFS-EXCLUDED is real and payable, but under a DIFFERENT fee\n"
            "schedule (labs → CLFS, DME/supplies → DMEPOS) — keep it ONLY if THIS provider\n"
            "performs and bills that service (e.g. in-office CLIA lab, DMEPOS-enrolled\n"
            "supplier); if an outside lab/supplier bills it, remove it from this claim."
        )
        block.extend(advisory_lines)
    return "\n".join(block)


def _build_code_family_block(cpt_codes: list[dict], cpt_candidates: list[dict], db) -> str:
    """Real CPT code-family disambiguation — when an assigned CPT code
    shares its descriptor stem (the text before the first semicolon) with
    another code among the current candidates, show every family member's
    FULL real descriptor side by side, computed fresh from cpt_codes.json.

    Root cause this addresses: 28120/28122/28124 all share the stem
    "Partial excision ... bone (eg, osteomyelitis or bossing)", differing
    only in the anatomy clause after the semicolon (28120 = talus or
    calcaneus; 28122 = tarsal or metatarsal bone, EXCEPT talus or
    calcaneus; 28124 = phalanx of toe). Observed live: a navicular excision
    (a tarsal bone, not talus/calcaneus) was coded 28120 with reasoning
    that literally quoted 28120's real "talus or calcaneus" descriptor and
    then concluded it was a "verbatim match" for the navicular anyway — a
    reasoning failure, not a data-fabrication one (the real descriptor was
    already shown correctly). Unlike NCCI/billability/Excludes1 conflicts,
    which one CPT family member applies depends on which anatomy is
    documented — not decidable from structured data alone — so this can
    only be a real-data disambiguation aid, not an auto-correcting
    deterministic backstop; the actual match against documented anatomy is
    still the LLM's job, just with the full, real family laid out so a
    single member's descriptor can't be skimmed and misapplied.
    """
    if not db or not cpt_codes:
        return ""
    assigned_codes = {c.get("code", "") for c in cpt_codes if c.get("code")}
    candidate_codes = {c.get("code", "") for c in cpt_candidates if c.get("code")}
    relevant = assigned_codes | candidate_codes

    families: dict[str, list[str]] = {}
    for code in assigned_codes:
        rec = db.validate_cpt(code)
        desc = (rec or {}).get("long_description", "")
        if ";" not in desc:
            continue
        stem = desc.split(";")[0].strip()
        families.setdefault(stem, [])

    if not families:
        return ""

    # Second pass: find every relevant code (assigned or candidate) sharing
    # each stem found among assigned codes.
    lines: list[str] = []
    seen_stems: set[str] = set()
    for code in assigned_codes:
        rec = db.validate_cpt(code)
        desc = (rec or {}).get("long_description", "")
        if ";" not in desc:
            continue
        stem = desc.split(";")[0].strip()
        if stem in seen_stems:
            continue
        members = []
        for other in relevant:
            other_rec = db.validate_cpt(other)
            other_desc = (other_rec or {}).get("long_description", "")
            if other_desc.split(";")[0].strip() == stem:
                members.append((other, other_desc))
        if len(members) < 2:
            continue
        seen_stems.add(stem)
        lines.append(f'  Family "{stem}":')
        for m_code, m_desc in sorted(members):
            marker = " <- ASSIGNED" if m_code in assigned_codes else ""
            lines.append(f"    {m_code}: {m_desc}{marker}")

    if not lines:
        return ""
    return (
        "## CPT CODE FAMILY DISAMBIGUATION (real data from cpt_codes.json — authoritative)\n"
        "An assigned code below shares its descriptor stem with other real codes — they are\n"
        "NOT interchangeable. Read every family member's FULL descriptor, including any\n"
        '"except"/anatomy-restricting clause after the semicolon, and confirm the assigned\n'
        "code's specific clause — not just the shared stem — actually matches the documented\n"
        "anatomy. Do not assume the first-considered or most general-sounding family member is\n"
        "correct without checking whether a more specific member's clause fits better.\n"
        + "\n".join(lines)
    )


# ICD-10 category families above this size are excluded — real-world sizes
# range from single-digit (Z88.x: 10 drug-allergy-category codes) to
# thousands (max observed: 3096; median 9). A small family like Z88.x is a
# genuinely enumerable "pick one of N real options" choice worth showing in
# full; dumping dozens-to-thousands of siblings from a large category (e.g.
# M19: 52, E11: 87) would be noise, not disambiguation help, for a category
# that's usually distinguished by laterality/episode/severity rather than a
# fixed small enumeration.
_ICD_FAMILY_SIZE_CAP = 15


def _build_icd_family_block(icd_codes: list[dict], db) -> str:
    """Real ICD-10-CM category-family disambiguation — the ICD-10 analog of
    _build_code_family_block. Unlike CPT (siblings share a literal
    descriptor stem), ICD-10 siblings share only a category prefix (e.g.
    "Z88"); each sibling's full description differs entirely.

    Root cause this addresses: aspirin (an analgesic/NSAID) was coded
    Z88.5 ("Allergy status to narcotic agent") instead of Z88.6 ("...
    analgesic agent") — the assigned code's own real description was shown
    and stored correctly (not fabricated), but the LLM mapped the
    documented allergen to the wrong sibling in a 10-member category it
    never saw laid out side by side. Shown for EVERY assigned code's
    category, not filtered to RAG-retrieved candidates — the retrieval
    step itself can be the thing that anchored on the wrong sibling (e.g.
    if it only ever retrieved Z88.5 for an "aspirin allergy" query), so
    filtering to candidates would show nothing for exactly the cases where
    this matters most.
    """
    if not db:
        return ""
    lines: list[str] = []
    seen_prefixes: set[str] = set()
    for entry in icd_codes:
        code = entry.get("code", "")
        if not code:
            continue
        prefix = code.replace(".", "").strip().upper()[:3]
        if not prefix or prefix in seen_prefixes:
            continue
        siblings = db.icd10_siblings(prefix)
        if len(siblings) < 2 or len(siblings) > _ICD_FAMILY_SIZE_CAP:
            continue
        seen_prefixes.add(prefix)
        assigned_norm = code.replace(".", "").strip().upper()
        lines.append(f'  Category "{prefix}":')
        for s_code, s_desc in siblings:
            marker = " <- ASSIGNED" if s_code == assigned_norm else ""
            lines.append(f"    {s_code}: {s_desc}{marker}")

    if not lines:
        return ""
    return (
        "## ICD-10-CM CATEGORY FAMILY DISAMBIGUATION (real data — authoritative)\n"
        "An assigned code below belongs to a small category with other real, distinct sibling\n"
        "codes — they are NOT interchangeable, and each has its own complete, different\n"
        "description. Compare the documented specific term (e.g. a specific drug/allergen name)\n"
        "against every sibling's actual description, not just the assigned code's — a documented\n"
        "term can belong to a different sibling's category than the one currently assigned.\n"
        + "\n".join(lines)
    )


def _build_icd_excludes1_block(icd_codes: list[dict], store) -> str:
    """Real ICD-10-CM Type 1 Excludes ("not coded here") conflicts among the
    currently-assigned ICD-10 codes — computed fresh from compliance.db,
    same pattern as _build_ncci_pair_block/_build_billability_block.

    Root cause this replaces: M12.571 (Traumatic arthropathy) and M19.171
    (Post-traumatic osteoarthritis) coded together — M12.5's own Tabular
    List entry carries an explicit excludes1 note referencing M19.1;
    structurally mutually exclusive, not two similar codes to pick between.
    Which one is correct is a documentation-reading judgment call (per AHA
    Coding Clinic guidance on this exact pair: default to the
    osteoarthritis code unless documentation specifies a non-osteoarthritis
    traumatic arthropathy) — this block surfaces the real conflict so that
    judgment gets applied deliberately instead of both codes silently
    coexisting.
    """
    if not store or not icd_codes:
        return ""
    codes = [c.get("code", "") for c in icd_codes if c.get("code")]
    lines: list[str] = []
    seen: set[tuple[str, str]] = set()
    for i in range(len(codes)):
        for j in range(i + 1, len(codes)):
            key = tuple(sorted((codes[i], codes[j])))
            if key in seen:
                continue
            seen.add(key)
            if store.excludes1_conflict(codes[i], codes[j]):
                lines.append(
                    f"  {codes[i]} + {codes[j]}: TYPE 1 EXCLUDES CONFLICT — these are structurally "
                    f"mutually exclusive per the ICD-10-CM Tabular List; cannot both be coded on "
                    f"this claim"
                )
    if not lines:
        return ""
    return (
        "## ICD-10-CM EXCLUDES1 CONFLICTS (real data from compliance.db — authoritative)\n"
        'A pair listed here is a real Type 1 Excludes ("not coded here") relationship — CMS\'s\n'
        "own Tabular List says these two conditions cannot be coded together, not a stylistic\n"
        "choice between similar codes. Read the documentation to determine which condition is\n"
        "actually supported and remove the other; do not keep both.\n"
        + "\n".join(lines)
    )


def _build_ncci_pair_block(
    cpt_codes: list[dict], cpt_candidates: list[dict], store, dos=None,
) -> str:
    """Real, authoritative NCCI PTP pairwise status for every pair among the
    currently-assigned CPT codes, plus each assigned code against the top
    not-yet-assigned candidates — computed fresh from compliance.db every
    time, the same way global_period is already injected as [global=XXX] on
    CPT candidates (see _format_candidates_for_system), rather than a hand-
    written bundling rule per scenario.

    Root cause this replaces: a documented bone graft (20900) was silently
    dropped with the stated reason "bundled into the arthrodesis per
    standard NCCI convention" — store.ncci_pair('28730', '20900') returns
    None; no such edit exists. The LLM had no way to check this at
    generation time and pattern-matched a plausible-sounding but false
    claim. Injecting the real answer here removes the need to guess.
    """
    if not store or not cpt_codes:
        return ""
    if not dos or not store.ncci_data_available(dos):
        return (
            "## NCCI PAIR STATUS\n"
            "The applicable NCCI release is unavailable for this date of service. "
            "Do not infer that any pair is unedited; downstream release must remain on hold."
        )
    assigned = [c.get("code", "") for c in cpt_codes if c.get("code")]
    candidate_codes = [
        c.get("code", "") for c in cpt_candidates[:15]
        if c.get("code") and c.get("code") not in assigned
    ]

    seen: set[tuple[str, str]] = set()
    lines: list[str] = []

    def _check(c1: str, c2: str) -> None:
        if c1 == c2:
            return
        key = tuple(sorted((c1, c2)))
        if key in seen:
            return
        seen.add(key)
        edit = store.ncci_pair(c1, c2, dos)
        if edit:
            indicator = str(edit.get("modifier_indicator", ""))
            meaning = {
                "0": "hard bundle — NEVER separately billable together, regardless of modifier",
                "1": "bundled by default, but separately billable with modifier 59/XE/XS/XP/XU/25/57 if documented as distinct",
                "9": "edit does not restrict this pair",
            }.get(indicator, f"indicator={indicator}")
            lines.append(f"  {c1} + {c2}: NCCI EDIT EXISTS — {meaning}")
        else:
            lines.append(f"  {c1} + {c2}: NO NCCI edit in the real edit table — separately billable by default")

    for i in range(len(assigned)):
        for j in range(i + 1, len(assigned)):
            _check(assigned[i], assigned[j])
    for a in assigned:
        for c in candidate_codes:
            _check(a, c)

    if not lines:
        return ""
    return (
        "## NCCI PAIR STATUS (real data from compliance.db — authoritative, not a guess)\n"
        "This is the actual NCCI PTP edit table result for these code pairs. Use it directly.\n"
        'Do NOT describe a pair as "bundled," "not separately billable," or "per NCCI\n'
        'convention" unless it is listed here as having an edit — a pair listed as "NO NCCI\n'
        "edit\" has no bundling relationship in the real data and should be coded separately\n"
        "if clinically documented, with modifier 59/RT/LT as appropriate.\n"
        + "\n".join(lines)
    )


def _format_candidates_for_system(rag_candidates: dict, system: str, store=None) -> str:
    candidates = rag_candidates.get(system, [])
    if not candidates:
        return "No candidates retrieved."
    lines = []
    for c in candidates[:25]:
        score = c.get("similarity_score", 0)
        code = c.get("code", "")
        desc = c.get("description", "") or c.get("long_description", "") or c.get("short_description", "")
        # Annotate CPT candidates with their actual global period from the data store.
        # This lets the LLM distinguish major (090) from minor (010/000) procedures at
        # coding time — the data-driven foundation for correct -25 vs -57 selection.
        global_tag = ""
        if system == "cpt" and store:
            glob = store.global_period(code)
            if glob:
                global_tag = f" [global={glob}]"
        max_desc = 500 if system == "hcpcs" else 150
        lines.append(f"  {code}{global_tag} (relevance: {score:.3f}) — {desc[:max_desc]}")
    return "\n".join(lines)


def _summarize_icd(icd_codes: list[dict]) -> str:
    if not icd_codes:
        return "No ICD-10-CM codes assigned."
    lines = []
    for c in icd_codes:
        lines.append(f"  {c.get('code', '?')} [{c.get('type', '?')}] — {c.get('description', '')[:80]}")
    return "\n".join(lines)


def _summarize_cpt(cpt_codes: list[dict], store=None) -> str:
    if not cpt_codes:
        return "No CPT codes assigned yet."
    lines = []
    for c in cpt_codes:
        mods = ", ".join(c.get("modifiers", [])) or "none"
        code = c.get("code", "?")
        global_tag = ""
        if store:
            glob = store.global_period(code)
            if glob:
                global_tag = f" [global={glob}]"
        lines.append(f"  {code}{global_tag} [{mods}] — {c.get('description', '')[:80]}")
    return "\n".join(lines)


def _safe_parse(raw: str, required_key: str) -> dict:
    try:
        # Every consumer expects list-of-dicts code arrays — normalize once
        # here so bare-string entries can't crash any pass (1-3 feed
        # _hard_db_gate, the verify pass feeds the enforcement helpers).
        return _normalize_code_arrays(json.loads(raw))
    except json.JSONDecodeError:
        logger.error(f"Failed to parse LLM response for {required_key}")
        # Empty dict, not {required_key: []} — every caller falls back via
        # .get(key, <pre-verification value>), and a failure dict that
        # already contains required_key: [] would make that fallback treat
        # a genuine parse FAILURE as if verification had legitimately
        # returned an empty list for that one field, skipping the fallback
        # for it while every other field correctly falls back.
        return {}


# ---------------------------------------------------------------------------
# Fix 6 — Hard Database Gate
# ---------------------------------------------------------------------------

def _hard_db_gate(entries: list[dict], code_system: str, db,
                  allowed_codes: set[str] | None = None) -> list[dict]:
    """Immediately remove codes that are NOT in the reference database.

    This prevents invalid/hallucinated codes from ever reaching the verification
    pass, and ensures every output code is defensible in an audit.
    """
    valid = []
    if db is None:
        logger.error(f"    [DB GATE] {code_system.upper()} reference database "
                     "is unavailable — removed every proposed code")
        return valid
    for entry in entries:
        code = entry.get("code", "").strip()
        if not code:
            continue
        if (allowed_codes is not None and
                _candidate_code(code_system, code) not in allowed_codes):
            logger.warning(
                f"    [CANDIDATE GATE] {code_system.upper()} {code!r} was "
                "not retrieved or physician-documented — removed")
            continue
        found = False
        if code_system == "icd10":
            found = bool(db.validate_icd10(code))
        elif code_system == "cpt":
            found = bool(db.validate_cpt(code))
        elif code_system == "hcpcs":
            found = bool(db.validate_hcpcs(code))
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

    # Every match below is keyed by (code_system, code) / (code_system, prefix),
    # never bare code/prefix — ICD-10 (stored undotted, e.g. "E1100") and HCPCS
    # ("E1100" = a real wheelchair code) can be byte-identical strings that
    # mean completely different things, so a bare-code match risks tagging an
    # AI-assigned code from one system as "physician_documented" because a
    # physician happened to write the same string in a different system, or
    # suppressing a genuinely missing physician code via a false cross-system
    # family match. physician_documented_codes carries its own "system" field
    # (icd10/cpt/hcpcs, from the extraction schema); AI-assigned codes get
    # their system from which result[key] list they're actually in.
    _KEY_TO_SYSTEM = {"icd10_codes": "icd10", "cpt_codes": "cpt", "hcpcs_codes": "hcpcs"}

    # Build maps: (system, exact code) → physician entry; (system, 3-char prefix) → physician entries
    phys_exact: dict[tuple[str, str], dict] = {}
    phys_prefix: dict[tuple[str, str], list[dict]] = {}
    for p in physician_documented_codes:
        code = p.get("code", "").strip().upper()
        system = str(p.get("system", "")).strip().lower()
        if not code or not system:
            continue
        phys_exact[(system, code)] = p
        prefix = code[:3]
        phys_prefix.setdefault((system, prefix), []).append(p)

    all_ai_codes: set[tuple[str, str]] = set()
    for key, system in _KEY_TO_SYSTEM.items():
        for e in result.get(key, []):
            all_ai_codes.add((system, e.get("code", "").strip().upper()))

    # Tag each code
    for key, system in _KEY_TO_SYSTEM.items():
        for e in result.get(key, []):
            code = e.get("code", "").strip().upper()
            if (system, code) in phys_exact:
                e["code_source"] = "physician_documented"
            else:
                # Check if a physician code in the same system + 3-char family was not used
                prefix = code[:3]
                same_family = phys_prefix.get((system, prefix), [])
                replaced = [p for p in same_family if (system, p.get("code", "").upper()) not in all_ai_codes]
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
        system = str(p.get("system", "")).strip().lower()
        if code and system and (system, code) not in all_ai_codes:
            # Check if it wasn't replaced (already caught above)
            prefix = code[:3]
            ai_same_family = [c for s, c in all_ai_codes if s == system and c[:3] == prefix]
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


def _strip_invalid_cpt_modifiers(verified: dict, store=None) -> dict:
    """Remove modifiers that are not recognized modifiers AT ALL, per the
    merged AMA CPT Appendix A + CMS HCPCS Level II modifier reference
    (modifiers.json) — a hallucination gate, same role as _hard_db_gate for
    codes.

    Two deliberate design points, both learned from real failures:

    1. Gate on modifier_valid (recognized in either source), NOT
       modifier_valid_for_cpt (AMA's CPT-book cross-listing). CMS Level II
       modifiers absent from AMA's list — Q7/Q8/Q9 (routine-foot-care class
       findings), KX, GA/GX/GY/GZ, QW — are legitimately appended to CPT
       lines on CMS-1500 claims per CMS billing rules; AMA-book scope is the
       wrong authority for claim-form validity. (An earlier version gated on
       the CPT cross-listing and would have deleted a required Q8 from a
       covered 11720 routine-foot-care line.)

    2. Fail safe when the reference data is absent: if the modifier table has
       zero rows, every modifier looks "unrecognized" and a strip would
       delete ALL modifiers (-25/-57/RT/LT/TA…) from every claim — observed
       live when an ingestion bug left the table's systems column empty.
       Missing data means "cannot check", never "strip everything".
    """
    if store is None:
        return verified
    if store.modifier_count() == 0:
        logger.warning(
            "    [MODIFIER STRIP] modifier reference table is empty — "
            "skipping strip entirely (missing data must never remove modifiers)"
        )
        return verified
    for entry in verified.get("cpt_codes", []):
        raw = entry.get("modifiers", [])
        if not raw:
            continue
        valid = [m for m in raw if store.modifier_valid(str(m))]
        removed = [m for m in raw if not store.modifier_valid(str(m))]
        if removed:
            logger.warning(
                f"    [MODIFIER STRIP] CPT {entry.get('code')} — removed unrecognized modifiers: {removed}"
            )
            entry["modifiers"] = valid
    return verified


# First plausible code token in a free-text correction line — used only to
# recover the subject code when the LLM emits corrections_made as bare
# strings instead of the contracted dict shape.
_CORRECTION_CODE_RE = re.compile(r"\b([A-Z]\d{2}(?:\.[0-9A-Z]{1,4})?|\d{5}|[A-Z]\d{4})\b")


def _gate_verify_additions(final_result: dict, combined: dict, db, store=None,
                           allowed_codes: dict[str, set[str]] | None = None) -> dict:
    """Reference-DB gate for codes INTRODUCED by the verification pass.

    Passes 1-3 are hard-gated (_hard_db_gate), but Pass 4's output never
    was: a code the earlier gate removed could be re-added by the audit
    ("protected anchor" reasoning), or a wholly new line invented — both
    observed live in one batch (D48.1, a non-billable ICD category header,
    and CPT 20926, deleted from the code set years ago).

    Policy per entry not present before verification:
      * validates in its claimed system → keep (legitimate audit addition);
      * a real Tabular/code-set entry at the WRONG level or date (category
        header with billable children, or a code active on some other
        date) → keep — the specificity filter FAILs it with an actionable
        message ("assign the billable child" / "not active for DOS"),
        which serves reviewers better than silent disappearance;
      * known nowhere in any reference data → remove (hallucination).
    """
    if db is None:
        for key in ("icd10_codes", "supporting_conditions", "cpt_codes",
                    "hcpcs_codes"):
            final_result[key] = []
        return final_result
    checks = {
        "icd10_codes": ("ICD10", db.validate_icd10),
        "supporting_conditions": ("ICD10", db.validate_icd10),
        "cpt_codes": ("CPT", db.validate_cpt),
        "hcpcs_codes": ("HCPCS", db.validate_hcpcs),
    }
    allowed_by_key = {
        "icd10_codes": (allowed_codes or {}).get("icd10"),
        "supporting_conditions": (allowed_codes or {}).get("icd10"),
        "cpt_codes": (allowed_codes or {}).get("cpt"),
        "hcpcs_codes": (allowed_codes or {}).get("hcpcs"),
    }
    for key, (system, validate) in checks.items():
        pre = {
            (e.get("code") or "").upper()
            for e in combined.get(key, []) if isinstance(e, dict)
        }
        entries = final_result.get(key, [])
        kept = []
        for e in entries:
            code = (e.get("code") or "").strip()
            if not code:
                continue
            allowed = allowed_by_key.get(key)
            normalized_system = "icd10" if system == "ICD10" else system.lower()
            if (allowed is not None and
                    _candidate_code(normalized_system, code) not in allowed):
                logger.warning(
                    f"    [VERIFY CANDIDATE GATE] {key}: '{code}' was not "
                    "retrieved or physician-documented — removed")
                continue
            if code.upper() in pre or validate(code):
                kept.append(e)
                continue
            known_elsewhere = store is not None and (
                store.code_active_any_date(system, code)
                or store.children_exist(system, code)
            )
            if known_elsewhere:
                kept.append(e)  # real entry, wrong level/date — specificity flags it
                continue
            logger.warning(
                f"    [VERIFY GATE] {key}: '{code}' was introduced by the verification "
                f"pass but exists nowhere in the {system} reference data — removed "
                f"(likely hallucinated)"
            )
        if len(kept) != len(entries):
            final_result[key] = kept
    return final_result


def _inherit_dropped_fields(final_result: dict, combined: dict) -> None:
    """Backfill fields Pass 4 dropped from entries it re-emitted. The verify
    contract says 'return every entry in full', but the model often returns
    only the fields it thought about; anything omitted then collapses to the
    schema default (e.g. ICD type -> 'secondary', units -> 1), silently
    mutating claim data no correction ever mentioned. For each code present
    both pre- and post-verification, copy over any key the verified entry is
    missing. Keys the model explicitly returned (even falsy) are respected —
    an intentional change always survives; only omissions are repaired."""
    for key in ("icd10_codes", "supporting_conditions", "cpt_codes",
                "hcpcs_codes", "snomed_codes"):
        before = {e.get("code"): e for e in combined.get(key, [])
                  if isinstance(e, dict) and e.get("code")}
        for entry in final_result.get(key, []):
            if not isinstance(entry, dict):
                continue
            # A CHANGED-correction entry carries its pre-verification code so
            # inheritance can find the original: the slim verify entry (or an
            # in-place enforcement rewrite) holds the NEW code, which has no
            # pre-verification counterpart under that key.
            pre = entry.pop("_pre_verify_code", None)
            prior = before.get(entry.get("code")) or (before.get(pre) if pre else None)
            if not prior:
                continue
            for field, val in prior.items():
                if field not in entry and field != "code":
                    entry[field] = val


def _normalize_code_arrays(verified: dict) -> dict:
    """Coerce the verify pass's code arrays into the contracted
    list-of-dicts shape. Observed live: the model occasionally compacts an
    array to bare code strings ('icd10_codes': ['E11.42', ...]), crashing
    the first downstream .get() and aborting the note. Strings that look
    like codes are wrapped as minimal entries (descriptions/rationales are
    refreshed downstream by _enforce_real_descriptions); other non-dict
    garbage is dropped. A key that exists but isn't a list at all is
    removed so the caller's pre-verification fallback kicks in."""
    for key in ("icd10_codes", "supporting_conditions", "cpt_codes",
                "hcpcs_codes", "snomed_codes"):
        if key not in verified:
            continue  # missing key = fall back to pre-verification value
        raw = verified[key]
        if not isinstance(raw, list):
            logger.warning(f"    [ARRAY NORMALIZE] {key} is {type(raw).__name__}, "
                           f"not a list — dropped (pre-verification value will be used)")
            del verified[key]
            continue
        out = []
        for e in raw:
            if isinstance(e, dict):
                out.append(e)
            elif isinstance(e, str) and e.strip():
                out.append({"code": e.strip()})
                logger.warning(f"    [ARRAY NORMALIZE] {key}: bare string entry "
                               f"'{e.strip()}' wrapped as a code entry")
        verified[key] = out
    return verified


def _normalize_corrections(verified: dict) -> dict:
    """Coerce corrections_made into the contracted list-of-dicts shape.

    The verify pass occasionally emits corrections as bare strings
    ('Removed 11719 — not documented') instead of dicts — observed live
    crashing every downstream consumer (`'str' object has no attribute
    'get'`) and aborting the whole note. One malformed narrative entry must
    never kill a claim, so normalization happens once, here, before any
    consumer: dicts pass through, strings are wrapped as best-effort dicts
    (type inferred from the leading verb, code from the first code-shaped
    token), anything else is dropped."""
    raw = verified.get("corrections_made", [])
    if not isinstance(raw, list):
        verified["corrections_made"] = []
        return verified
    out = []
    for c in raw:
        if isinstance(c, dict):
            out.append(c)
            continue
        if isinstance(c, str) and c.strip():
            low = c.lower()
            kind = ("ADDED" if "add" in low.split(chr(32))[0] else
                    "REMOVED" if low.startswith(("remov", "delet", "drop")) else
                    "CHANGED" if low.startswith(("chang", "correct", "switch", "remap", "updat")) else
                    "OTHER")
            m = _CORRECTION_CODE_RE.search(c)
            entry = {"type": kind, "code": m.group(1) if m else "", "reason": c.strip()}
            out.append(entry)
            logger.warning(
                f"    [CORRECTIONS NORMALIZE] string entry coerced to dict: {entry['type']} "
                f"{entry['code'] or '(no code found)'}"
            )
    verified["corrections_made"] = out
    return verified


def _enforce_added_corrections(verified: dict, db=None) -> dict:
    """Guarantee every code noted as ADDED in corrections_made actually
    appears in its code array. The LLM sometimes writes 'ADDED J0702' (or
    any other code) in corrections_made but forgets to include the code in
    the actual array — silent billing/coding loss.

    Generalizes a prior J-code-only version of this rescuer (HCPCS drug
    codes specifically): the same "claimed but not applied" gap isn't
    J-code-shaped, it can happen to any code in any system, so scoping the
    fix to one regex pattern only rescued a fraction of real occurrences.

    System is resolved by validating the code against the real reference DB
    for each candidate system, not by code shape — CPT is unambiguous (5
    digits), but ICD-10 (stored undotted, e.g. "E1100") and HCPCS ("E1100"
    is a real wheelchair code) can be byte-identical strings across systems,
    so a shape-only guess (letter + 4 digits) is genuinely ambiguous. A code
    that validates as real in more than one system is left alone rather
    than guessed into the wrong array; a code that validates in neither is
    left alone too (likely hallucinated — the correction claims a code that
    doesn't exist, which is a separate problem this function shouldn't try
    to paper over).
    """
    import re
    if db is None:
        return verified
    corrections = verified.get("corrections_made", [])
    if not corrections:
        return verified

    icd_list = verified.get("icd10_codes", [])
    cpt_list = verified.get("cpt_codes", [])
    hcpcs_list = verified.get("hcpcs_codes", [])
    icd_existing = {c.get("code", "").upper() for c in icd_list}
    cpt_existing = {c.get("code", "").upper() for c in cpt_list}
    hcpcs_existing = {c.get("code", "").upper() for c in hcpcs_list}

    for correction in corrections:
        if correction.get("type", "").upper() != "ADDED":
            continue
        code = correction.get("code", "").strip().upper()
        if not code or code in icd_existing or code in cpt_existing or code in hcpcs_existing:
            continue

        is_cpt_shaped = bool(re.match(r"^\d{5}$", code))
        icd_rec = None if is_cpt_shaped else db.validate_icd10(code)
        hcpcs_rec = None if is_cpt_shaped else db.validate_hcpcs(code)
        cpt_rec = db.validate_cpt(code) if is_cpt_shaped else None

        base_entry = {
            "code": code,
            "description": correction.get("reason", "")[:100],
            "confidence": 0.85,
            "rationale": correction.get("reason", ""),
            "supporting_text": correction.get("evidence", ""),
            "code_source": "ai_inferred",
        }

        if icd_rec and hcpcs_rec:
            logger.warning(
                f"    [CORRECTION RESCUE] {code}: claimed ADDED but validates as a real code in "
                f"BOTH ICD-10 and HCPCS — genuinely ambiguous, not rescued (needs manual review)"
            )
        elif cpt_rec:
            cpt_list.append({**base_entry, "modifiers": [], "units": 1, "linked_diagnoses": []})
            cpt_existing.add(code)
            logger.info(f"    [CORRECTION RESCUE] {code} added to cpt_codes (was in corrections_made but missing)")
        elif hcpcs_rec:
            hcpcs_list.append({
                **base_entry, "modifiers": [], "units": 1, "linked_diagnoses": [],
                "needs_review": False, "review_reason": None,
            })
            hcpcs_existing.add(code)
            logger.info(f"    [CORRECTION RESCUE] {code} added to hcpcs_codes (was in corrections_made but missing)")
        elif icd_rec:
            icd_list.append({**base_entry, "type": "secondary"})
            icd_existing.add(code)
            logger.info(f"    [CORRECTION RESCUE] {code} added to icd10_codes (was in corrections_made but missing)")
        else:
            logger.warning(
                f"    [CORRECTION RESCUE] {code}: claimed ADDED but not found in any code system's "
                f"reference DB — not rescued (likely hallucinated)"
            )

    verified["icd10_codes"] = icd_list
    verified["cpt_codes"] = cpt_list
    verified["hcpcs_codes"] = hcpcs_list
    return verified


# Correction language that signals the CODE VALUE itself was replaced —
# used only to recover a missing to_code from the correction's own reason
# text (grammar of the audit narrative, not any medical-code list).
_CHANGED_TO_RE = re.compile(
    r"(?:changed|corrected|switched|remapped|updated)\s*(?:\w+\s+){0,4}?to\s+([A-Z][0-9][0-9A-Z]{1,5}(?:\.[0-9A-Z]{1,4})?)",
    re.IGNORECASE,
)


def _enforce_changed_corrections(verified: dict, db=None) -> dict:
    """Guarantee every code-value CHANGED correction is actually applied to
    its code array. The narrative-claims-fix-but-data-unchanged failure is
    not ADDED-shaped only: observed live, the audit pass wrote 'Z88.5
    changed to reflect analgesic-agent category (Z88.6)' in its narrative
    fields while the array entry still carried code=Z88.5 — the reasoning
    described the fix, the data didn't contain it (same failure class as
    the modifier-reasoning divergence fixed with structured ModifierClaim).

    Enforcement is deterministic: for each CHANGED correction carrying the
    old code and a replacement (structured `to_code`, else recovered from
    the correction's own 'changed/corrected ... to <code>' language), find
    the old code across ALL code arrays (including supporting_conditions —
    where sibling miscodes live), validate the replacement exists in the
    SAME code system's reference DB (a replacement that doesn't validate is
    likely hallucinated — left alone, existence checks flag it), then write
    the new code into the entry's code field. Descriptions refresh via
    _enforce_real_descriptions downstream.
    """
    if db is None:
        return verified

    arrays = {
        "icd10_codes": lambda c: db.validate_icd10(c),
        "supporting_conditions": lambda c: db.validate_icd10(c),
        "cpt_codes": lambda c: db.validate_cpt(c),
        "hcpcs_codes": lambda c: db.validate_hcpcs(c),
    }
    # Layer 1: explicit CHANGED entries in corrections_made. Layer 2 below
    # runs unconditionally — the observed live failure had NO correction
    # entry at all, only the code entry's own narrative.
    for correction in verified.get("corrections_made", []):
        if correction.get("type", "").upper() != "CHANGED":
            continue
        old = (correction.get("code") or "").strip().upper()
        new = (correction.get("to_code") or "").strip().upper()
        if not new:
            m = _CHANGED_TO_RE.search(correction.get("reason", "") or "")
            new = m.group(1).upper() if m else ""
        if not old or not new or old == new:
            continue
        for key, validate in arrays.items():
            for entry in verified.get(key, []):
                if (entry.get("code") or "").strip().upper() != old:
                    continue
                if not validate(new):
                    logger.warning(
                        f"    [CORRECTION ENFORCE] {old}→{new} claimed in corrections_made but "
                        f"{new} doesn't validate in the same code system — not applied"
                    )
                    continue
                entry.setdefault("_pre_verify_code", old)
                entry["code"] = new
                logger.info(
                    f"    [CORRECTION ENFORCE] {key}: {old} → {new} "
                    f"(CHANGED correction applied to the code field, not just the narrative)"
                )

    # Second layer: sibling switches declared only in an entry's OWN narrative
    # (no corrections_made entry at all — the observed Z88.5 case, where the
    # rationale argued 'aspirin allergy maps to the analgesic-agent sibling
    # (Z88.6)' and review_reason said 'corrected', while the code field still
    # read Z88.5). Scoped tightly to stay deterministic and safe:
    #   - only TARGET codes count: the code a directional phrase points AT
    #     ('from X to Y' → Y; 'maps to (Y)' → the nearest code after the
    #     verb). Direction matters both ways — when the entry's current code
    #     IS the target, the correction was already applied and the OLD code
    #     named in the same sentence must not flip it back (observed live:
    #     'corrected from Z88.5 to Z88.6' on an entry already carrying Z88.6
    #     was reverted to Z88.5 by a direction-blind version of this);
    #   - correction verb and candidate must share a SENTENCE with no refusal
    #     phrasing ('rather than', 'retained', ...), so 'retained X rather
    #     than switching to Y' never triggers;
    #   - the target must be a same-category sibling (shared 3-char category
    #     — the miscoded-sibling class the disambiguation block corrects)
    #     that validates in the entry's own code system;
    #   - exactly ONE distinct target, else leave it to existence/review.
    code_pat = r"[A-Z]\d{2,4}(?:\.[0-9A-Z]{1,4})?"
    pair_re = re.compile(
        rf"from\s+\(?({code_pat})\)?[^.;]{{0,60}}?\bto\s+\(?({code_pat})\)?", re.IGNORECASE)
    verb_re = re.compile(
        rf"\b(?:corrected|changed|switch(?:ed)?|remap(?:ped)?|should be|maps? to|use)\b"
        rf"[^.;]{{0,60}}?\b({code_pat})\b", re.IGNORECASE)
    neg_re = re.compile(r"\b(rather than|instead of|avoid(?:ed)?|declined|rejected|kept|retain(?:ed)?)\b",
                        re.IGNORECASE)
    for key, validate in arrays.items():
        for entry in verified.get(key, []):
            cur = (entry.get("code") or "").strip().upper()
            text = " ".join(filter(None, [entry.get("rationale"), entry.get("review_reason")]))
            if not cur or not text:
                continue
            targets, sources = set(), set()
            for sentence in re.split(r"[.;](?:\s|$)", text):
                if neg_re.search(sentence):
                    continue
                for m in pair_re.finditer(sentence):
                    sources.add(m.group(1).upper())
                    targets.add(m.group(2).upper())
                for m in verb_re.finditer(sentence):
                    c = m.group(1).upper()
                    if c not in sources:
                        targets.add(c)
            if cur in targets:
                continue  # correction already applied — never flip back to the old code
            candidates = {c for c in targets if c != cur and c[:3] == cur[:3] and validate(c)}
            if len(candidates) != 1:
                continue  # none, or ambiguous — leave to existence/review checks
            new = candidates.pop()
            entry.setdefault("_pre_verify_code", cur)
            entry["code"] = new
            logger.info(
                f"    [CORRECTION ENFORCE] {key}: {cur} → {new} (entry's own narrative declared "
                f"a sibling correction that was never applied to the code field)"
            )
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

    # Prose phrases only — no hardcoded code literals (e.g. "e11.9"/"e10.9"/
    # "e13.9") here. That was the exact same anti-pattern already found and
    # fixed in validator.py's _check_redundant_dm_codes this session (a
    # hardcoded 3-code tuple that also silently missed E11.A, the 4th real
    # "without complications" DM code): a fixed code list goes stale the
    # moment CMS revises the DM code family, and this function only has raw
    # assessment TEXT, not parsed code entries with real DB descriptions, to
    # check anyway. Real enforcement already happens downstream, against the
    # actual final assigned codes with real data, in
    # validator.py::_check_redundant_dm_codes — this block is prompt
    # guidance, not the enforcement point, and every real "E11.9 written
    # with zero descriptive prose" case still reaches that backstop.
    dm_generic_phrases = (
        "without complications", "type 2 dm without", "type 2 diabetes without",
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
