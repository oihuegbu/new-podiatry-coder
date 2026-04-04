import json
from app.core.llm_client import chat_completion
from app.models.schemas import ClinicalEntity
from app.core.logger import get_logger

logger = get_logger(__name__)

NER_SYSTEM_PROMPT = """You are an expert clinical NLP system for PODIATRY medical coding.

Extract ALL clinically relevant entities from the podiatry clinical note.

## Entity Categories
- **diagnosis**: Conditions, disorders, diseases documented in assessment or mentioned in HPI/PMH
- **procedure**: Any procedure PERFORMED (not planned) on this date of service
- **finding**: Physical exam findings, test results, imaging findings
- **medication**: Medications prescribed or currently taken
- **supply**: DME, braces, boots, shoes, orthotics DISPENSED/PROVIDED today
- **body_structure**: Specific anatomical sites referenced

## Rules
1. Extract from ALL sections of the note (CC, HPI, PMH, PE, Imaging, Assessment, Plan)
2. Include PMH conditions (diabetes, HTN, etc.) as they affect coding
3. Mark laterality (RIGHT/LEFT/BILATERAL) when documented
4. Mark negated findings (e.g., "no erythema" → negated=true)
5. For procedures: ONLY extract if PERFORMED today, not scheduled/planned
6. For supplies: ONLY extract if DISPENSED today (e.g., "applied CAM walker" = yes, "prescribed surgical shoe" = maybe)
7. Include specificity details (e.g., "2nd digit", "great toe", "medial calcaneal tubercle")
8. Use the EXACT text from the note in the "text" field

## Output Format
Return valid JSON:
{
  "entities": [
    {
      "text": "exact text from note",
      "category": "diagnosis|procedure|finding|medication|supply|body_structure",
      "clinical_term": "normalized clinical term",
      "laterality": "RIGHT|LEFT|BILATERAL|null",
      "specificity": "additional detail or null",
      "source_section": "CC|HPI|PMH|PE|IMAGING|ASSESSMENT|PLAN",
      "negated": false
    }
  ]
}"""


def extract_entities(note_sections: dict) -> list[ClinicalEntity]:
    user_prompt = f"""Extract all clinical entities from this podiatry note:

### CHIEF COMPLAINT
{note_sections.get('chief_complaint', 'N/A')}

### HISTORY OF PRESENT ILLNESS
{note_sections.get('hpi', 'N/A')}

### PAST MEDICAL HISTORY / MEDICATIONS / ALLERGIES
{note_sections.get('pmh_medications_allergies', 'N/A')}

### PHYSICAL EXAMINATION
{note_sections.get('physical_examination', 'N/A')}

### IMAGING / DIAGNOSTICS
{note_sections.get('imaging_diagnostics', 'N/A')}

### ASSESSMENT / DIAGNOSES
{note_sections.get('assessment_diagnoses', 'N/A')}

### PLAN
{note_sections.get('plan', 'N/A')}"""

    raw, usage = chat_completion(
        system_prompt=NER_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=0.05,
        max_tokens=3000,
    )

    try:
        data = json.loads(raw)
        entities_data = data.get("entities", [])
    except json.JSONDecodeError:
        logger.error("Failed to parse NER response")
        return []

    entities = []
    for item in entities_data:
        try:
            entity = ClinicalEntity(**item)
            if not entity.negated:
                entities.append(entity)
        except Exception as e:
            logger.warning(f"Skipping malformed entity: {e}")

    logger.info(f"Extracted {len(entities)} entities (usage: {usage.get('total_tokens', 0)} tokens)")
    return entities
