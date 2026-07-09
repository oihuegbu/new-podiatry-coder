import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(BASE_DIR / ".env")

# --- LLM Provider ("openai" or "claude") ---
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai").lower()

# --- OpenAI ---
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o")

# --- Anthropic / Claude ---
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL: str = os.getenv("CLAUDE_MODEL", "claude-opus-4-7")
# Reasoning effort. "xhigh" = deepest (Opus); Sonnet supports high/low/max/medium.
# Default "high" works on every current model; override per model via env.
CLAUDE_EFFORT: str = os.getenv("CLAUDE_EFFORT", "high")

# --- Paths ---
DATA_DIR = BASE_DIR / "data"
QDRANT_DIR = BASE_DIR / Path(os.getenv("QDRANT_PATH", "data/qdrant_store"))
QDRANT_URL: str = os.getenv("QDRANT_URL", "")  # if set, connect to Qdrant server; else use local path
OUTPUT_DIR = BASE_DIR / "output" / "results"
LOGS_DIR = BASE_DIR / "logs"

ATTACHMENTS_DIR = BASE_DIR / "doctors_notes"

# --- Code reference files — stored in data/codes/ inside the repo
#     Override any filename via .env without touching code
CODES_DIR = DATA_DIR / os.getenv("CODES_DIR", "codes")

ICD10_FILE  = CODES_DIR / os.getenv("ICD10_FILENAME",  "icd10cm_codes.json")
CPT_FILE    = CODES_DIR / os.getenv("CPT_FILENAME",    "cpt_codes.json")
HCPCS_FILE  = CODES_DIR / os.getenv("HCPCS_FILENAME",  "hcpcs_codes.json")
NCCI_FILE   = CODES_DIR / os.getenv("NCCI_FILENAME",   "ncci_data.json")
MUE_FILE    = CODES_DIR / os.getenv("MUE_FILENAME",    "mue_practitioner.json")
LCD_FILE    = CODES_DIR / os.getenv("LCD_FILENAME",    "podiatry_lcd.json")

# Notes directory (PDFs to process) — can be any folder
NOTES_DIR = Path(os.getenv("NOTES_DIR", str(ATTACHMENTS_DIR)))

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
