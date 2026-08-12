import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(BASE_DIR / ".env")

# --- LLM Provider ("openai" or "claude") ---
# Default "claude", matching .env.example's recommendation — the previous
# "openai" default meant a missing/partial .env silently ran coding on a
# different provider than every documented setup path configures. Note the
# Anthropic key is required EITHER WAY: PDF extraction always uses Claude
# Vision regardless of which provider does the coding passes.
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "claude").lower()

# --- OpenAI ---
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o")

# --- Anthropic / Claude ---
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
# Default matches .env.example's verified model — the previous default
# (an Opus slug) diverged from every documented configuration.
CLAUDE_MODEL: str = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")
# Reasoning effort. "xhigh" = deepest (Opus); Sonnet supports high/low/max/medium.
# Default "high" works on every current model; override per model via env.
CLAUDE_EFFORT: str = os.getenv("CLAUDE_EFFORT", "high")

# --- Verification-pass tiering (escalation) ---
# Pass 4 re-audits everything with the full rule context injected — it's where
# judgment errors concentrate, and it's one call per note. Optionally run it
# on a stronger model/effort than passes 1-3 (e.g. claude-opus-4-8 / xhigh).
# Empty = same model/effort as the other passes. Claude-provider only.
CLAUDE_VERIFY_MODEL: str = os.getenv("CLAUDE_VERIFY_MODEL", "")
CLAUDE_VERIFY_EFFORT: str = os.getenv("CLAUDE_VERIFY_EFFORT", "")

# --- Anthropic Message Batches API (50% discount) ---
# When enabled, every Claude call is submitted through the Batches API —
# identical model, identical prompts, identical output distribution; the
# discount buys Anthropic scheduling flexibility, so the only trade is
# latency (calls typically complete in minutes, worst case longer). Set to
# "0" for latency-sensitive interactive runs.
ANTHROPIC_USE_BATCH: bool = os.getenv("ANTHROPIC_USE_BATCH", "1") != "0"
# Give up on a batch request after this long and let the caller's retry
# logic resubmit. Batches can queue up to 24h in the worst case; 2h is a
# pragmatic ceiling for a pipeline that runs on a schedule.
ANTHROPIC_BATCH_MAX_WAIT_S: float = float(
    os.getenv("ANTHROPIC_BATCH_MAX_WAIT_S", "7200"))

# --- Structured outputs ---
# When enabled, the coding passes send a strict JSON Schema via the provider's
# structured-output API (Anthropic output_config.format / OpenAI json_schema
# response_format), eliminating the malformed-JSON error class (bare strings
# in code arrays, string corrections, dropped required keys) at the source.
# The downstream normalizers stay as a backstop. Set to "0" to disable.
STRUCTURED_OUTPUTS: bool = os.getenv("STRUCTURED_OUTPUTS", "1") != "0"

# --- Paths ---
DATA_DIR = BASE_DIR / "data"
QDRANT_DIR = BASE_DIR / Path(os.getenv("QDRANT_PATH", "data/qdrant_store"))
QDRANT_URL: str = os.getenv("QDRANT_URL", "")  # if set, connect to Qdrant server; else use local path
OUTPUT_DIR = BASE_DIR / "output" / "results"
# Phase-3 durable provenance store (SQLite): a DEDICATED database, separate from the
# authoritative compliance.db (different lifecycle + retention). Append-only, WAL.
PROVENANCE_DB = BASE_DIR / Path(os.getenv("PROVENANCE_DB", "output/provenance.db"))
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
# Parsed MCD bulk-export cache (written by the weekly mcd_articles refresh;
# see app/compliance/refresh/runner._write_coverage_cache). Carries the
# covered-ICD GROUP roles the flat seed file lacks; _ingest_lcd overlays it
# over the seed so compliance.db rebuilds keep the claim-composition grammar.
MCD_COVERAGE_CACHE_FILE = CODES_DIR / "mcd_coverage_cache.json"

# Notes directory (PDFs to process) — can be any folder
NOTES_DIR = Path(os.getenv("NOTES_DIR", str(ATTACHMENTS_DIR)))

# --- Supplementary rule tables (bundled with system) ---
GLOBAL_PERIODS_FILE = DATA_DIR / "global_periods.json"
# CMS PFS payment-policy indicator extract (GLOB DAYS + BILAT SURG) the CODER reads for
# global-surgical-period and bilateral-surgery decisions, built by
# tools/build_global_period.py.  This is a DIFFERENT extract from GLOBAL_PERIODS_FILE
# above (which the compliance datastore ingests): both are release-bearing, both are
# declared in app/release/source_manifest, and the path lives here -- not as a literal in
# the reader -- so the bytes that are certified are provably the bytes that are read.
PFS_INDICATOR_FILE = CODES_DIR / os.getenv("PFS_INDICATOR_FILENAME", "global_period.json")
# Authoritative modifier definitions the coder's modifier engine resolves against.
MODIFIER_FILE = CODES_DIR / os.getenv("MODIFIER_FILENAME", "modifiers.json")
# ICD-10-CM Tabular instructional (Excludes1/Excludes2) notes -- gate-bearing: the
# Excludes1 conflict check degrades to NOT_APPLICABLE without them.
INSTRUCTIONAL_NOTES_FILE = CODES_DIR / os.getenv(
    "INSTRUCTIONAL_NOTES_FILENAME", "icd10cm_instructional_notes.json")
VALIDATOR_RULES_FILE = DATA_DIR / "rules" / "validator_rules.json"
SNOMED_ROOTS_FILE = DATA_DIR / "snomed_root_concepts.json"
TERMINOLOGY_REGISTRY_FILE = (
    DATA_DIR / "terminology" / "clinical_abbreviations.json"
)
# Reviewed, versioned claim-affecting control configuration shipped with the coder.
# Registered here so the release manifest content-addresses the exact bytes the
# fail-closed controls load at decision time.
CODER_CONTROLS_DIR = BASE_DIR / "claude_coder" / "controls"
NECESSITY_RELATION_CONTROL_FILE = CODER_CONTROLS_DIR / "necessity_relation_control.json"
RELATION_EVIDENCE_GRAMMAR_FILE = CODER_CONTROLS_DIR / "relation_evidence_grammar.json"

# --- RAG settings ---
# top_k feeds the coder's candidate list; 20 gives recall headroom for a
# borderline-ranked-but-correct code (e.g. a semicolon-parent) without
# flooding the prompt. The dense prefetch keeps a cosine floor, but the
# enriched multi-descriptor / index-synonym embedding text (see
# vector_store record loaders) is what actually lifts vocabulary-mismatched
# codes into range on BOTH dense and sparse.
RAG_TOP_K: int = 20
RAG_SIMILARITY_THRESHOLD: float = 0.35

# --- Coding engine ---
CODING_TEMPERATURE: float = 0.0
CODING_MAX_TOKENS: int = 4096

# --- Verified-claim exemplars (few-shot from the finalized-claims registry) ---
# Mode:
#   auto    (default) shadow until the registry holds more than
#           EXEMPLAR_LIVE_THRESHOLD verified claims, then live
#   shadow  retrieve similar verified encounters and record what WOULD be
#           injected (rag_context.exemplars) — prompts unchanged
#   live    inject the rendered exemplar block into the coding prompts
#   off     disabled entirely
EXEMPLAR_MODE: str = os.getenv("EXEMPLAR_MODE", "auto").lower()
# auto flips shadow → live above this many verified claims: enough registry
# coverage that a same-scenario neighbor usually exists, so exemplars anchor
# rather than mislead.
EXEMPLAR_LIVE_THRESHOLD: int = int(os.getenv("EXEMPLAR_LIVE_THRESHOLD", "500"))
# How many exemplars to retrieve, and the minimum similarity for a verified
# claim to qualify as a neighbor at all (Jaccard over distinctive terms).
EXEMPLAR_TOP_K: int = int(os.getenv("EXEMPLAR_TOP_K", "3"))
EXEMPLAR_MIN_SIM: float = float(os.getenv("EXEMPLAR_MIN_SIM", "0.2"))

# --- Logging ---
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
