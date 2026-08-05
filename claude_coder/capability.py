"""Capability manifest — make authoritative-source availability LOUD, not silent.

The retrieval/decision layers degrade quietly when an optional source (a synonym
layer, the index-terms file, the SNOMED crosswalk) is absent, and a REQUIRED source
going missing (a code table, an edit-policy file) would silently weaken the claim.
This module inspects each source by its CONFIGURED path and reports what loaded, what
is missing, and whether the deployment is degraded — surfaced on the release
certificate and enforced by a fail-closed gate (a missing REQUIRED source BLOCKS).

Agnostic: source identities come from the filename stem, and paths come from central
config — no medical code, code family, or domain list is written here.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


def _probe(path: Any, required: bool, role: str) -> dict[str, Any]:
    p = Path(path)
    present = p.exists() and p.is_file() and p.stat().st_size > 0
    return {
        "source": p.stem,
        "required": bool(required),
        "present": present,
        "status": "loaded" if present else ("MISSING" if required else "absent-optional"),
        "role": role,
        "path": str(p),
    }


def build_manifest() -> dict[str, Any]:
    """Probe every authoritative source the coder depends on. Required sources gate
    release (fail closed); optional sources are recorded so their absence is visible
    rather than silently degrading retrieval."""
    from app.core import config

    codes = config.CODES_DIR
    # (path, required, role). Required = a source whose absence makes the claim
    # unsafe/underspecified; optional = a recall/precision aid that degrades quality
    # but not correctness. Paths come from config or the configured codes directory.
    checks: list[tuple[Any, bool, str]] = [
        (config.ICD10_FILE, True, "diagnosis code table"),
        (config.CPT_FILE, True, "procedure code table"),
        (config.HCPCS_FILE, True, "supply/device code table"),
        (config.NCCI_FILE, True, "procedure-pair edit policy"),
        (config.MUE_FILE, True, "unit-limit policy"),
        (config.GLOBAL_PERIODS_FILE, True, "global-period / fee-schedule policy"),
        (config.LCD_FILE, False, "local coverage policy"),
        (codes / "icd10cm_index_terms.json", False, "official alphabetic index terms"),
        (codes / "icd10cm_instructional_notes.json", False, "tabular exclusion notes"),
        (codes / "cpt_synonyms.json", False, "synonym recall aid"),
        (codes / "hcpcs_synonyms.json", False, "synonym recall aid"),
        (codes / "icd10_synonyms.json", False, "synonym recall aid"),
        (codes / "snomed_icd10_map.json", False, "concept crosswalk"),
        (codes / "cpt_index_terms.json", False, "procedure descriptor index"),
        (codes / "learned_cpt_index.json", False, "learned resolution index"),
        (codes / "hcpcs_drug_table.json", False, "drug dosing table"),
    ]

    sources = [_probe(p, req, role) for p, req, role in checks]
    missing_required = [s["source"] for s in sources if s["required"] and not s["present"]]
    degraded_optional = [s["source"] for s in sources
                         if not s["required"] and not s["present"]]
    return {
        "sources": sources,
        "missing_required": missing_required,
        "degraded_optional": degraded_optional,
        "status": "BLOCKED" if missing_required else "OK",
    }
