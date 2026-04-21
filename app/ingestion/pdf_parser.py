import base64
import json
from pathlib import Path
from pdf2image import convert_from_path
from io import BytesIO

from app.core.llm_client import chat_completion, get_openai_client
from app.core.config import OPENAI_MODEL
from app.core.logger import get_logger

logger = get_logger(__name__)


EXTRACTION_SYSTEM_PROMPT = """You are an expert medical document parser specializing in podiatry clinical notes.

You will receive an image of a clinical note PDF. Extract ALL information into structured sections.

## INSTRUCTIONS
1. Read the ENTIRE document carefully — every word matters for medical coding
2. Extract patient metadata from the header table
3. Extract each clinical section EXACTLY as written — do not summarize or paraphrase
4. Preserve ALL medical details: measurements, dosages, laterality, specific digits/toes
5. Preserve bullet points and list items in the Assessment/Diagnoses section
6. If imaging mentions specific views or measurements, include the FULL detail
7. For the Plan section, capture EVERY action item including medications, procedures performed, follow-up

## OUTPUT — Return valid JSON:
{
  "patient_metadata": {
    "patient_name": "full name",
    "date_of_birth": "MM/DD/YYYY",
    "date_of_service": "Month DD, YYYY",
    "provider": "Dr. Name, Credentials",
    "npi": "number",
    "mrn": "number",
    "insurance": "full insurance line",
    "note_type": "e.g., NEW PATIENT – OFFICE VISIT"
  },
  "sections": {
    "chief_complaint": "exact text",
    "hpi": "exact text — full history of present illness",
    "pmh_medications_allergies": "exact text — include PMH, Medications with doses, Allergies",
    "physical_examination": "exact text — all findings, measurements, vitals",
    "imaging_diagnostics": "exact text — study type, views, findings, measurements",
    "assessment_diagnoses": "exact text — every diagnosis listed, preserve numbering/bullets",
    "plan": "exact text — every action item, procedures performed, medications, follow-up"
  },
  "note_category": "new_patient_visit|established_patient_visit|post_op_followup|surgical_procedure|urgent_visit",
  "procedures_performed_today": ["list of procedures actually done on this date, not planned"],
  "imaging_performed_today": ["list of imaging studies done on this date"],
  "supplies_dispensed_today": ["list of DME/supplies given to patient today"],
  "prior_surgery_info": {
    "is_post_op_visit": true,
    "days_post_op": 14,
    "prior_surgery_description": "Austin/Chevron bunionectomy right foot",
    "prior_surgery_cpt": "28296"
  }
}

## IMPORTANT RULES FOR prior_surgery_info:
- Set is_post_op_visit=true ONLY if the note explicitly documents a follow-up visit after a prior surgery
- Look for language like: "post-op day X", "s/p [procedure]", "post-operative visit", "follow-up after surgery", "surgical follow-up"
- days_post_op: the exact number if stated (e.g., "Day 14 post-op" → 14), else null
- prior_surgery_cpt: your best estimate of the CPT code for the prior surgery, or null if unknown
- If NOT a post-op visit, return: {"is_post_op_visit": false, "days_post_op": null, "prior_surgery_description": null, "prior_surgery_cpt": null}

CRITICAL: Return ONLY valid JSON. Extract VERBATIM text from the document — do not interpret or summarize."""


def extract_from_pdf(pdf_path: str | Path) -> dict:
    """Use GPT-4o Vision to intelligently extract structured data from a clinical note PDF."""
    pdf_path = Path(pdf_path)
    logger.info(f"Converting {pdf_path.name} to image for GPT-4o Vision...")

    images = convert_from_path(str(pdf_path), dpi=300, first_page=1, last_page=2)

    image_payloads = []
    for img in images:
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        image_payloads.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"},
        })

    client = get_openai_client()
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract all information from this clinical note:"},
                    *image_payloads,
                ],
            },
        ],
        temperature=0.0,
        max_tokens=3000,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content
    usage = {
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
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

    return {
        "metadata": metadata,
        "sections": sections,
        "note_category": result.get("note_category", ""),
        "procedures_performed_today": result.get("procedures_performed_today", []),
        "imaging_performed_today": result.get("imaging_performed_today", []),
        "supplies_dispensed_today": result.get("supplies_dispensed_today", []),
        "prior_surgery_info": prior_surgery_info,
        "extraction_usage": usage,
    }
