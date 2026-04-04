# Podiatry Medical Coding System

Automated medical coding system for podiatry clinical notes. Translates doctor's notes into ICD-10-CM, CPT, HCPCS, and SNOMED CT codes using a multi-stage AI pipeline.

## Architecture

```
Doctor's Note (PDF)
       │
       ▼
  [1] GPT-4o Vision ── Intelligent PDF extraction (sections, metadata, procedures, imaging, supplies)
       │
       ▼
  [2] Clinical NER ── Extract 15-25 clinical entities per note (diagnoses, procedures, findings, meds)
       │
       ▼
  [3] RAG / FAISS ── Semantic search across 94K+ embedded medical codes to retrieve candidates
       │                 ├── 74,719 ICD-10-CM codes
       │                 ├── 11,574 CPT codes
       │                 └── 8,928 HCPCS codes
       ▼
  [4] Multi-Pass Code Assignment (GPT-4o)
       │    ├── Pass 1: ICD-10-CM diagnosis coding
       │    ├── Pass 2: CPT procedure / E&M / imaging coding
       │    ├── Pass 3: HCPCS + SNOMED CT coding
       │    └── Pass 4: Anchor-and-Audit self-verification
       ▼
  [5] Validation Engine
            ├── NCCI edit conflict detection
            ├── MUE unit limit checks
            ├── LCD qualifying diagnosis checks (L36199)
            ├── ICD-10 sequencing rules
            ├── Modifier validation (RT/LT, TA/T1-T9, -25, -59)
            ├── CPT-ICD linkage verification
            └── Documentation audit scoring
```

## Key Features

- **GPT-4o Vision** reads PDFs as images for accurate section extraction
- **FAISS vector store** with OpenAI embeddings for semantic code retrieval
- **Multi-pass coding** separates ICD/CPT/HCPCS into focused passes for higher accuracy
- **Anchor-and-Audit verification** protects Assessment-listed diagnoses from being removed by the self-correction pass
- **Full validation engine** with NCCI, MUE, LCD, sequencing, and modifier checks
- **Auto-coding tiers** (AUTO / REVIEW / REJECT) with confidence scores
- **Pydantic models** for structured, typed output
- **Structured logging** to console and file

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-4o
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSIONS=1536
FAISS_INDEX_PATH=data/vector_store
LOG_LEVEL=INFO
```

### 3. Place data files

The system expects the following JSON files in the `kachi203-attachments` directory (sibling to the project folder):

| File | Description |
|---|---|
| `icd10cm_codes.json` | ICD-10-CM code set (FY2026) |
| `cpt_codes.json` | CPT codes with descriptions |
| `hcpcs_codes_20260330_211727.json` | HCPCS Level II codes |
| `latest_ncci_data.json` | NCCI edit pairs |
| `latest_mue_practitioner.json` | MUE unit limits |
| `podiatry_routine_foot_care_qualifying_dx.json` | LCD L36199 qualifying diagnoses |
| `NOTE_*.pdf` | Clinical note PDFs |

## Usage

```bash
# Process all notes
python run.py

# Process a single note
python run.py --note NOTE_01_Marcus_Thornton_Hallux_Valgus_New_Patient.pdf

# Force rebuild FAISS indices
python run.py --rebuild-index
```

The first run will embed all 94K+ codes into FAISS (takes ~5 minutes). Subsequent runs load cached indices in ~1 second.

## Output Format

Each note produces a JSON file with:

```json
{
  "document_id": "NOTE_01_...",
  "success": true,
  "processing_time": 52.3,
  "patient_metadata": { "patient_name": "...", "dos": "...", ... },
  "icd_codes": [
    {
      "code": "M20.11",
      "description": "Hallux valgus (acquired), right foot",
      "type": "primary",
      "confidence": 0.95,
      "rationale": "...",
      "supporting_text": "...",
      "s3_validated": true
    }
  ],
  "cpt_codes": [
    {
      "code": "99203",
      "modifiers": ["25"],
      "linked_diagnoses": ["M20.11", "E11.9"],
      "ama_validated": true,
      "mue_validated": true
    }
  ],
  "hcpcs_codes": [...],
  "snomed_codes": [...],
  "validation_issues": [...],
  "auto_coding_tier": "AUTO",
  "auto_coding_confidence": 0.93,
  "pre_submission_audit_score": 0.97
}
```

## Validation Checks

| Check | Description |
|---|---|
| Code existence | Every code verified against official FY2026 databases |
| NCCI edits | CPT code-pair conflict detection |
| MUE limits | Maximum units per service per day |
| LCD L36199 | Medicare routine foot care qualifying diagnoses |
| ICD sequencing | Primary diagnosis, etiology/manifestation ordering |
| Modifier validation | RT/LT, TA/T1-T9, -25, -59, -50 |
| CPT-ICD linkage | Every procedure has supporting diagnoses |
| Documentation audit | Every code has supporting note text |

## Tech Stack

| Component | Technology |
|---|---|
| LLM | OpenAI GPT-4o |
| Embeddings | OpenAI text-embedding-3-small |
| Vector store | FAISS (faiss-cpu) |
| PDF extraction | GPT-4o Vision + pdf2image |
| Data models | Pydantic v2 |
| Code databases | ICD-10-CM, CPT, HCPCS, NCCI, MUE (JSON) |

## Requirements

- Python 3.11+
- OpenAI API key with GPT-4o access
- poppler (for pdf2image): `brew install poppler` (macOS) or `apt install poppler-utils` (Linux)
