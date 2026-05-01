import base64
import json
import re
from pathlib import Path
from pdf2image import convert_from_path
from io import BytesIO
import anthropic

from app.core.logger import get_logger

logger = get_logger(__name__)


EXTRACTION_SYSTEM_PROMPT = """You are an expert medical document parser specializing in podiatry clinical notes.

You will receive one or more page images of a clinical note PDF. Extract ALL information into structured sections.

## ACCURACY REQUIREMENT — READ THIS FIRST
This extracted data feeds a medical billing pipeline where a single misread character causes coding errors and claim denials.

- ACCURACY over completeness: if you cannot read text with confidence, write [UNCLEAR] rather than guessing
- Medical codes have ZERO tolerance for character substitution:
  - The letter O and the digit 0 are different — re-read every code
  - The letter I and the digit 1 are different — e.g., "I10" (hypertension) vs "110"
  - The letter S and the digit 5, the letter B and the digit 8 are easily confused in print
- Re-read every code character by character before writing it
- Laterality is critical — RT, LT, bilateral, right, left must be read exactly as written; never assume or infer laterality
- Drug names, dosages, injection amounts, and measurements must be verbatim — do NOT round or estimate
- Digit/toe numbers (1st, 2nd, 3rd, 4th, 5th) must be exact — these determine which CPT code applies

## INSTRUCTIONS
1. Use your thinking to carefully analyze the entire document before extracting — take your time
2. Read the ENTIRE document — every word matters for medical coding
3. If multiple page images are provided, read ALL pages and combine information coherently
4. Extract patient metadata from the header table exactly as printed
5. Extract each clinical section EXACTLY as written — do not summarize, paraphrase, or improve phrasing
6. Preserve ALL medical details: measurements, dosages, laterality, specific digits/toes
7. Preserve bullet points and numbered lists in the Assessment/Diagnoses section
8. If imaging mentions specific views or measurements, include the FULL detail
9. For the Plan section, capture EVERY action item including medications, procedures performed, injections given, follow-up instructions

## UNCLEAR TEXT PROTOCOL
- Handwritten or poor-quality print: write the best reading followed by a question mark, e.g., "metformin 500mg?"
- Completely unreadable word: write [UNCLEAR]
- For physician_documented_codes: if a code is partially unclear, skip it entirely — do NOT guess a code
- Never fabricate clinical findings, drug names, or codes you cannot clearly see

## OUTPUT — Return valid JSON with no markdown fences:
{
  "patient_metadata": {
    "patient_name": "full name",
    "date_of_birth": "MM/DD/YYYY",
    "date_of_service": "Month DD, YYYY",
    "provider": "Dr. Name, Credentials",
    "npi": "number or null",
    "mrn": "number or null",
    "insurance": "full insurance line or null",
    "note_type": "e.g., NEW PATIENT – OFFICE VISIT"
  },
  "sections": {
    "chief_complaint": "exact verbatim text",
    "hpi": "exact verbatim text — full history of present illness",
    "pmh_medications_allergies": "exact verbatim text — include PMH, Medications with doses, Allergies",
    "physical_examination": "exact verbatim text — all findings, measurements, vitals",
    "imaging_diagnostics": "exact verbatim text — study type, views, findings, measurements",
    "assessment_diagnoses": "exact verbatim text — every diagnosis listed, preserve numbering/bullets",
    "plan": "exact verbatim text — every action item, procedures performed, medications, follow-up"
  },
  "note_category": "new_patient_visit|established_patient_visit|post_op_followup|surgical_procedure|urgent_visit",
  "procedures_performed_today": ["list of procedures actually done on this date, not planned for future"],
  "imaging_performed_today": ["list of imaging studies done on this date"],
  "supplies_dispensed_today": ["list of DME/supplies/orthotics given to patient today"],
  "prior_surgery_info": {
    "is_post_op_visit": true,
    "days_post_op": 14,
    "prior_surgery_description": "Austin/Chevron bunionectomy right foot",
    "prior_surgery_cpt": "28296"
  },
  "physician_documented_codes": [
    {
      "code": "E11.42",
      "system": "icd10",
      "description": "Type 2 DM with polyneuropathy",
      "section": "ASSESSMENT",
      "raw_text": "E11.42 - Type 2 DM with polyneuropathy"
    }
  ]
}

## IMPORTANT RULES FOR physician_documented_codes:
- Extract ONLY codes explicitly written by the physician in the document — no inferred codes
- Look in: Assessment/Diagnoses section, Plan section, header fields, anywhere codes appear
- ICD-10-CM codes: one capital letter followed by digits and optional decimal (e.g., E11.42, M20.11, L84, B35.1, I10)
- CPT codes: exactly 5 digits (e.g., 11721, 99213, 28296) — if you count 4 or 6 digits, re-read
- HCPCS codes: exactly one capital letter followed by exactly 4 digits (e.g., A5513, L3020, J3301, J0702)
- CHARACTER ACCURACY CHECK before including any code:
  - Verify letter vs digit in every position (I10 not 110, L84 not 184, O not 0)
  - Verify decimal placement in ICD-10 codes (E11.42 not E114.2)
  - Verify digit count in CPT codes (exactly 5)
- If the physician wrote NO explicit codes, return an empty array []
- If a code is partially obscured or unclear, skip it — do NOT guess

## IMPORTANT RULES FOR prior_surgery_info:
- Set is_post_op_visit=true ONLY if the note explicitly documents a follow-up after a prior surgery
- Look for: "post-op day X", "s/p [procedure]", "post-operative visit", "follow-up after surgery", "surgical follow-up"
- days_post_op: exact number if stated (e.g., "Day 14 post-op" → 14), else null
- prior_surgery_cpt: best estimate CPT for the prior surgery, or null if unknown
- If NOT a post-op visit: {"is_post_op_visit": false, "days_post_op": null, "prior_surgery_description": null, "prior_surgery_cpt": null}

CRITICAL: Return ONLY valid JSON with no markdown code fences. Every character in every medical code must be verified before output. When uncertain about any character, re-read it; if still uncertain, use [UNCLEAR] or skip the code."""


def extract_from_pdf(pdf_path: str | Path) -> dict:
    """Use Claude Opus 4.7 Vision to intelligently extract structured data from a clinical note PDF."""
    pdf_path = Path(pdf_path)
    logger.info(f"Converting {pdf_path.name} to image for Claude Opus 4.7 Vision...")

    images = convert_from_path(str(pdf_path), dpi=300, first_page=1, last_page=2)

    image_blocks = []
    for img in images:
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        image_blocks.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": b64},
        })

    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=8192,
        thinking={"type": "adaptive"},
        output_config={"effort": "xhigh"},
        system=[{
            "type": "text",
            "text": EXTRACTION_SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        }],
        messages=[{
            "role": "user",
            "content": [
                # Images first — improves OCR accuracy on Opus 4.7
                *image_blocks,
                {"type": "text", "text": "Extract all information from this clinical note into the required JSON structure. Take your time to read every character carefully, especially medical codes and laterality."},
            ],
        }],
    )

    raw = next(block.text for block in response.content if block.type == "text")
    # Strip markdown code fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    raw = re.sub(r"\s*```$", "", raw.strip())
    usage = {
        "prompt_tokens": response.usage.input_tokens,
        "completion_tokens": response.usage.output_tokens,
        "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
    }

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        logger.error("Failed to parse vision extraction response")
        result = {"patient_metadata": {}, "sections": {}}

    metadata = result.get("patient_metadata", {})
    sections = result.get("sections", {})

    # Build full_text from all sections for downstream compatibility
    full_text_parts = []
    for key in ["chief_complaint", "hpi", "pmh_medications_allergies",
                "physical_examination", "imaging_diagnostics",
                "assessment_diagnoses", "plan"]:
        text = sections.get(key, "")
        if text:
            full_text_parts.append(text)
    sections["full_text"] = "\n\n".join(full_text_parts)

    logger.info(
        f"  Vision extraction complete: "
        f"category={result.get('note_category', '?')}, "
        f"procedures={result.get('procedures_performed_today', [])}, "
        f"imaging={result.get('imaging_performed_today', [])}, "
        f"supplies={result.get('supplies_dispensed_today', [])}, "
        f"tokens={usage.get('total_tokens', 0)}"
    )

    prior_surgery_info = result.get("prior_surgery_info", {}) or {}
    if not isinstance(prior_surgery_info, dict):
        prior_surgery_info = {}

    physician_codes = result.get("physician_documented_codes", []) or []
    if not isinstance(physician_codes, list):
        physician_codes = []
    if physician_codes:
        logger.info(f"  Physician-documented codes found: {[p.get('code') for p in physician_codes]}")

    return {
        "metadata": metadata,
        "sections": sections,
        "note_category": result.get("note_category", ""),
        "procedures_performed_today": result.get("procedures_performed_today", []),
        "imaging_performed_today": result.get("imaging_performed_today", []),
        "supplies_dispensed_today": result.get("supplies_dispensed_today", []),
        "prior_surgery_info": prior_surgery_info,
        "physician_documented_codes": physician_codes,
        "extraction_usage": usage,
    }
