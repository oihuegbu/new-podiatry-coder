import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(BASE_DIR / ".env")

# --- OpenAI ---
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o")
OPENAI_EMBEDDING_MODEL: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIMENSIONS: int = int(os.getenv("EMBEDDING_DIMENSIONS", "1536"))

# --- Paths ---
DATA_DIR = BASE_DIR / "data"
VECTOR_STORE_DIR = BASE_DIR / Path(os.getenv("FAISS_INDEX_PATH", "data/vector_store"))
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
OUTPUT_DIR = BASE_DIR / "output" / "results"
LOGS_DIR = BASE_DIR / "logs"

ATTACHMENTS_DIR = BASE_DIR.parent / "kachi203-attachments"

# --- Source data files ---
ICD10_FILE = ATTACHMENTS_DIR / "icd10cm_codes.json"
CPT_FILE = ATTACHMENTS_DIR / "cpt_codes.json"
HCPCS_FILE = ATTACHMENTS_DIR / "hcpcs_codes_20260330_211727.json"
NCCI_FILE = ATTACHMENTS_DIR / "latest_ncci_data.json"
MUE_FILE = ATTACHMENTS_DIR / "latest_mue_practitioner.json"
LCD_FILE = ATTACHMENTS_DIR / "podiatry_routine_foot_care_qualifying_dx.json"
NOTES_DIR = ATTACHMENTS_DIR

# --- Supplementary rule tables (bundled with system) ---
GLOBAL_PERIODS_FILE = DATA_DIR / "global_periods.json"
SNOMED_ROOTS_FILE = DATA_DIR / "snomed_root_concepts.json"

# --- RAG settings ---
RAG_TOP_K: int = 15
RAG_SIMILARITY_THRESHOLD: float = 0.35

# --- Coding engine ---
CODING_TEMPERATURE: float = 0.0
CODING_MAX_TOKENS: int = 4096

# --- Logging ---
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
