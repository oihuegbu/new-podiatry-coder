import base64
import json
import re
import time
from pathlib import Path
from pdf2image import convert_from_path
from io import BytesIO

from app.core.logger import get_logger
from app.core.config import CLAUDE_MODEL, CLAUDE_EFFORT

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
5. Extract the LETTERHEAD and FOOTER bands too — the practice/facility name and its full street address (street, city, state, ZIP) and phone. These are usually at the very top or bottom edge of the page, outside the clinical sections. The servicing address determines which Medicare contractor's coverage policies govern this claim, so copy it exactly; if no address is printed anywhere, use null — never infer one
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
    "note_type": "e.g., NEW PATIENT – OFFICE VISIT",
    "place_of_service": "2-digit CMS POS code if stated or clearly inferable from the note's own setting language (office/clinic letterhead → 11, hospital → 21/22, ASC → 24, patient's home → 12, SNF → 31), else null — never guess between settings",
    "service_facility": {
      "name": "facility/practice name from the letterhead or footer, else null",
      "address": "street address exactly as printed, else null",
      "city": "city, else null",
      "state": "2-letter USPS state abbreviation exactly as printed on the letterhead/footer, else null — never infer from area codes or context",
      "zip": "ZIP code, else null",
      "phone": "phone number, else null"
    },
    "signature_block": "verbatim closing signature block (provider name, credentials, and any title/NPI printed with the signature), else null"
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

    # Shared client from llm_client: hard request timeout + our own retry
    # loop below. A bare anthropic.Anthropic() here had NO timeout and NO
    # retry — a single wedged socket on this one call froze an entire
    # consistency batch for hours (all workers asleep on the same read),
    # and a single transient overload aborted the whole note.
    #
    # Parse/truncation failures are retried inside the same loop, and the
    # FINAL failure raises instead of returning empty dicts: an empty
    # extraction used to flow silently through the whole pipeline and
    # produce a garbage claim, which is far worse than one loudly FAILED
    # note (observed live before this fix).
    from app.core.llm_client import (
        get_anthropic_client, _claude_message_via_batch, _RETRYABLE_MARKERS)
    from app.core.config import ANTHROPIC_USE_BATCH
    client = get_anthropic_client()
    max_tokens = 8192
    result = None
    response = None
    last_err = "unknown"
    for attempt in range(1, 4):
        try:
            body = {
                "model": CLAUDE_MODEL,
                "max_tokens": max_tokens,
                "thinking": {"type": "adaptive"},
                "output_config": {"effort": CLAUDE_EFFORT},
                "system": [{
                    "type": "text",
                    "text": EXTRACTION_SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }],
                "messages": [{
                    "role": "user",
                    "content": [
                        # Images first — improves OCR accuracy on Opus 4.7
                        *image_blocks,
                        # cache_control on the final block caches the whole
                        # prefix including the images — the 3 consistency
                        # runs of one note send identical pages, so runs 2/3
                        # read the OCR-heavy prefix at 10% of input price.
                        {"type": "text",
                         "text": "Extract all information from this clinical note into the required JSON structure. Take your time to read every character carefully, especially medical codes and laterality.",
                         "cache_control": {"type": "ephemeral"}},
                    ],
                }],
            }
            if ANTHROPIC_USE_BATCH:
                response = _claude_message_via_batch(client, body)
            else:
                response = client.messages.create(**body)
        except Exception as exc:
            msg = f"{type(exc).__name__} {exc}".lower()
            if not any(m in msg for m in _RETRYABLE_MARKERS) or attempt == 3:
                raise
            delay = 10.0 * attempt
            logger.warning(f"  Vision extraction attempt {attempt} failed "
                           f"({str(exc)[:120]}) — retrying in {delay:.0f}s")
            time.sleep(delay)
            continue

        # Truncation: a response cut off at max_tokens is a broken JSON
        # prefix (or, cut mid-thinking, has no text block at all). Retry
        # with a doubled budget, mirroring chat_completion.
        if response.stop_reason == "max_tokens":
            last_err = f"truncated at {max_tokens} tokens"
            max_tokens *= 2
            logger.warning(f"  Vision extraction {last_err} — retrying "
                           f"with doubled budget")
            continue

        raw = next((b.text for b in response.content if b.type == "text"), None)
        if raw is None:
            last_err = "response contained no text block"
            logger.warning(f"  Vision extraction attempt {attempt}: {last_err} — retrying")
            continue

        # Strip markdown code fences if present
        raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        raw = re.sub(r"\s*```$", "", raw.strip())
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            last_err = f"invalid JSON ({exc})"
            logger.warning(f"  Vision extraction attempt {attempt}: {last_err} — retrying")
            continue

        # Valid-but-empty JSON is the same failure as unparseable JSON: a
        # note with no metadata and no sections produces a garbage claim
        # downstream (observed live: 'category=?, procedures=[]' run).
        if not parsed.get("patient_metadata") and not parsed.get("sections"):
            last_err = "valid JSON but empty extraction (no metadata, no sections)"
            logger.warning(f"  Vision extraction attempt {attempt}: {last_err} — retrying")
            continue

        result = parsed
        break

    if result is None:
        raise RuntimeError(
            f"Vision extraction failed for {pdf_path.name} after 3 attempts: {last_err}")

    usage = {
        "prompt_tokens": response.usage.input_tokens,
        "completion_tokens": response.usage.output_tokens,
        "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
        "cache_read_tokens": getattr(
            response.usage, "cache_read_input_tokens", 0) or 0,
        "cache_write_tokens": getattr(
            response.usage, "cache_creation_input_tokens", 0) or 0,
    }

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
