import base64
import hashlib
import json
import re
import time
from pathlib import Path
from pdf2image import convert_from_path
from io import BytesIO

from app.core.logger import get_logger
from app.core.config import CLAUDE_MODEL, CLAUDE_EFFORT

logger = get_logger(__name__)

#: Identity of the request/response shape this transcriber uses. It is declared here,
#: next to the call it describes, and read by `app.ingestion.source_evidence` -- the
#: compiler must never restate a channel's identity from its own constants.
VISION_CHANNEL_SCHEMA_VERSION = "pdf_parser/vision-1"


EXTRACTION_SYSTEM_PROMPT = """You are an expert medical document parser specializing in podiatry clinical notes.

You will receive one or more page images of a clinical note PDF. Extract ALL information into structured sections.

## ACCURACY REQUIREMENT — READ THIS FIRST
This extracted data feeds a medical billing pipeline where a single misread character causes coding errors and claim denials.

- ACCURACY over completeness: if you cannot read text with confidence, write [UNCLEAR] rather than guessing
- Medical codes have ZERO tolerance for character substitution:
  - The letter O and the digit 0 are different — re-read every code
  - The letter I and the digit 1 are different
  - The letter S and the digit 5, the letter B and the digit 8 are easily confused in print
- Re-read every code character by character before writing it
- Laterality is critical — side and bilateral wording must be read exactly as written; never assume or infer laterality
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
  "page_texts": [
    {
      "page_number": 1,
      "status": "extracted",
      "text": "complete verbatim transcription of this page, including header and footer"
    }
  ],
  "patient_metadata": {
    "patient_name": "full name",
    "date_of_birth": "MM/DD/YYYY",
    "date_of_service": "Month DD, YYYY",
    "provider": "Dr. Name, Credentials",
    "npi": "number or null",
    "mrn": "number or null",
    "gender": "gender/sex value exactly as printed, else null",
    "insurance": "full insurance line or null",
    "insurance_plan": "plan/product name exactly as printed, else null",
    "member_id": "member/policy identifier exactly as printed, else null",
    "group_number": "group identifier exactly as printed, else null",
    "authorization_number": "prior-authorization identifier exactly as printed, else null",
    "note_type": "e.g., NEW PATIENT – OFFICE VISIT",
    "place_of_service": "CMS POS code only when explicitly printed, else null",
    "care_setting": "setting language exactly as printed (office, hospital, facility, home, etc.), else null; do not translate it to a code",
    "service_facility": {
      "name": "facility/practice name from the letterhead or footer, else null",
      "address": "street address exactly as printed, else null",
      "city": "city, else null",
      "state": "2-letter USPS state abbreviation exactly as printed on the letterhead/footer, else null — never infer from area codes or context",
      "zip": "ZIP code, else null",
      "phone": "phone number, else null"
    },
    "signature_block": "verbatim closing signature block (provider name, credentials, and any title/NPI printed with the signature), else null",
    "provider_specialty": "specialty explicitly printed in the document, or podiatry when DPM credentials are printed, else null"
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
    "prior_surgery_cpt": "verbatim code if explicitly printed, else null"
  },
  "physician_documented_codes": [
    {
      "code": "verbatim code",
      "system": "icd10",
      "description": "verbatim description",
      "section": "ASSESSMENT",
      "raw_text": "complete verbatim line"
    }
  ]
}

## IMPORTANT RULES FOR physician_documented_codes:
- Extract ONLY codes explicitly written by the physician in the document — no inferred codes
- Look in: Assessment/Diagnoses section, Plan section, header fields, anywhere codes appear
- ICD-10-CM codes: one capital letter followed by digits and an optional decimal
- CPT codes: exactly 5 digits — if you count 4 or 6 digits, re-read
- HCPCS codes: exactly one capital letter followed by exactly 4 digits
- CHARACTER ACCURACY CHECK before including any code:
  - Compare every letter, digit, and decimal with the source glyph by glyph
  - Never normalize an uncertain letter into a digit or an uncertain digit into a letter
  - Preserve the source's ICD-10-CM decimal placement exactly
  - Verify digit count in CPT codes (exactly 5)
- If the physician wrote NO explicit codes, return an empty array []
- If a code is partially obscured or unclear, skip it — do NOT guess

## IMPORTANT RULES FOR prior_surgery_info:
- Set is_post_op_visit=true ONLY if the note explicitly documents a follow-up after a prior surgery
- Look for: "post-op day X", "s/p [procedure]", "post-operative visit", "follow-up after surgery", "surgical follow-up"
- days_post_op: exact number if stated (e.g., "Day 14 post-op" → 14), else null
- prior_surgery_cpt: copy the code only when it is explicitly printed; otherwise null
- If NOT a post-op visit: {"is_post_op_visit": false, "days_post_op": null, "prior_surgery_description": null, "prior_surgery_cpt": null}

CRITICAL: Return ONLY valid JSON with no markdown code fences. Every character in every medical code must be verified before output. When uncertain about any character, re-read it; if still uncertain, use [UNCLEAR] or skip the code."""


def extract_from_pdf(pdf_path: str | Path) -> dict:
    """Use Claude Opus 4.7 Vision to intelligently extract structured data from a clinical note PDF."""
    pdf_path = Path(pdf_path)
    logger.info(f"Converting {pdf_path.name} to image for Claude Opus 4.7 Vision...")

    # Convert the complete document.  The previous first_page/last_page cap
    # silently discarded page 3 onward while still returning a successful
    # extraction — an unacceptable release ambiguity for a medical claim.
    images = convert_from_path(str(pdf_path), dpi=300)
    if not images:
        raise RuntimeError(f"PDF conversion produced no pages: {pdf_path.name}")
    source_digest = hashlib.sha256(pdf_path.read_bytes()).hexdigest()

    # The rendered page images ARE the thing the vision channel read, so their
    # identity is recorded here rather than re-derived later: a compiler that
    # re-rendered the PDF to hash it would be attesting to bytes that are merely
    # LIKELY the same as the ones the model saw (dpi, poppler version, colour
    # profile all move the digest). Issue #6 F6-R6-A / directive §1.
    image_blocks = []
    page_images = []
    # The rendered bytes themselves, kept for the compilation step that immediately
    # follows this call. Without them the Source Evidence Compiler can only RESTATE the
    # digests below, which makes source identity an upstream assertion rather than a
    # fact the compiler established (issue #6 F7-R5). They are deliberately NOT part of
    # `note_integrity`: that record is persisted and compared, and must stay JSON.
    page_image_bytes = {}
    for index, img in enumerate(images, start=1):
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        raw = buffer.getvalue()
        page_image_bytes[index] = raw
        b64 = base64.b64encode(raw).decode("utf-8")
        image_blocks.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": b64},
        })
        page_images.append({
            "page_number": index,
            "sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
            "width": img.size[0],
            "height": img.size[1],
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
        get_anthropic_client, client_identity, provider_of_client,
        _claude_message_via_batch, _RETRYABLE_MARKERS)
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

        # A complete per-page transcription is the extraction contract. The
        # old implementation equated "image sent" with "page extracted" and
        # could certify a page the model silently omitted.
        page_texts = parsed.get("page_texts") or []
        page_numbers = sorted(
            int(p.get("page_number")) for p in page_texts
            if isinstance(p, dict) and
            str(p.get("page_number") or "").isdigit())
        page_valid = (
            len(page_texts) == len(images) and
            page_numbers == list(range(1, len(images) + 1)) and
            all(isinstance(p, dict) and
                p.get("status") in {"extracted", "blank"} and
                (str(p.get("text") or "").strip()
                 if p.get("status") == "extracted" else
                 not str(p.get("text") or "").strip())
                for p in page_texts)
        )
        if not page_valid:
            last_err = "per-page transcription coverage is absent or incomplete"
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

    # WHICH VENDOR ACTUALLY READ THE PAGES (issue #6 F7-R5). Derived from the client
    # object that answered and the model that was actually sent -- never from a generic
    # configuration setting. This function calls Anthropic unconditionally, so a
    # deployment whose `LLM_PROVIDER` names another vendor would otherwise have the
    # Source Evidence Compiler record a primary-channel provider that is simply false;
    # a genuinely independent second-vendor page read would then be rejected as
    # same-provider, and in the mirror case a same-provider read would be accepted as
    # independent. Same pattern as `claude_coder.extraction.ExtractionOrigin` and
    # `claude_coder.verify.declare_model_profile`: identity travels with what ran.
    vision_channel = {
        "provider": provider_of_client(client),
        "profile": str(body.get("model") or ""),
        "prompt_sha256": "sha256:" + hashlib.sha256(
            EXTRACTION_SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "schema_version": VISION_CHANNEL_SCHEMA_VERSION,
        "client": client_identity(client),
    }
    if not vision_channel["provider"]:
        # Loud, and fail-closed downstream: an unestablished provider makes every
        # same-kind channel non-independent of this one, so quotations hold rather
        # than being proven by a reading that might share this one's vendor.
        logger.warning(
            f"  Vision transcription client {vision_channel['client']} has no "
            f"recognised provider identity; no second vision channel can be credited "
            f"as independent of this reading")

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

    # The complete per-page transcription, not a reconstruction from selected
    # clinical sections, is the evidence corpus used by every downstream gate.
    page_texts = sorted(result["page_texts"], key=lambda p: int(p["page_number"]))
    sections["full_text"] = "\n\n".join(
        str(p.get("text") or "") for p in page_texts)
    text_digest = hashlib.sha256(sections["full_text"].encode()).hexdigest()

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
        # The per-page transcription, kept rather than only concatenated. It is the
        # PRIMARY read channel of the Source Evidence Compiler, and the page boundaries
        # are what map an evidence span's character offsets onto a page of the original
        # document. `sections["full_text"]` remains the exact PAGE_SEPARATOR join of
        # these texts, so every existing offset is unchanged.
        "page_texts": [{"page_number": int(p["page_number"]),
                        "status": str(p.get("status") or ""),
                        "text": str(p.get("text") or "")}
                       for p in page_texts],
        "note_category": result.get("note_category", ""),
        "procedures_performed_today": result.get("procedures_performed_today", []),
        "imaging_performed_today": result.get("imaging_performed_today", []),
        "supplies_dispensed_today": result.get("supplies_dispensed_today", []),
        "prior_surgery_info": prior_surgery_info,
        "physician_documented_codes": physician_codes,
        "extraction_usage": usage,
        # The exact rendered bytes each page image digest below was taken from, so the
        # compiler recomputes rather than trusts. Keyed by page number; consumed by
        # `app.ingestion.source_evidence.compile_source_evidence` and not persisted.
        "page_image_bytes": page_image_bytes,
        "note_integrity": {
            "complete": True,
            "page_count": len(images),
            "extracted_page_count": len(page_texts),
            "source_pdf_sha256": f"sha256:{source_digest}",
            "extracted_text_sha256": f"sha256:{text_digest}",
            # Identity of the exact rendered images the vision channel was shown.
            "page_images": page_images,
            # WHO read them: the provider/profile/prompt/schema of the call that was
            # actually made, which is what channel independence is decided on.
            "vision_channel": vision_channel,
            "page_coverage": [
                {
                    "page_number": int(p["page_number"]),
                    "status": p["status"],
                    "text_sha256": ("sha256:" + hashlib.sha256(
                        str(p.get("text") or "").encode()).hexdigest()
                        if p["status"] == "extracted" else ""),
                }
                for p in page_texts
            ],
        },
    }
