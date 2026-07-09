"""ClaimScrubber — orchestrates the 12 compliance agents and applies the gate.

Flow:  CodingResult dict  ──build_claim──▶  Claim
       Claim  ──[agent.check() for each agent]──▶  Findings
       Findings  ──gate──▶  ScrubResult (CLEAN only if zero FAIL)
"""
from __future__ import annotations

import re
from datetime import date, datetime

from app.compliance.agents.base import ComplianceAgent
from app.compliance.datastore.store import ComplianceDataStore
from app.compliance.models import (
    Claim, ClaimLine, Diagnosis, Payer, Provider, ScrubResult, Subscriber,
)
from app.core.logger import get_logger

logger = get_logger(__name__)


def _parse_dos(meta: dict) -> date | None:
    raw = (meta or {}).get("date_of_service") or (meta or {}).get("dos") or ""
    raw = str(raw).strip()
    if not raw:
        return None
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m-%d-%Y", "%B %d, %Y", "%b %d, %Y", "%m/%d/%y"):
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            continue
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", raw)
    if m:
        try:
            return date(int(m[1]), int(m[2]), int(m[3]))
        except ValueError:
            pass
    return None


def build_claim(result: dict) -> Claim:
    """Normalize a CodingResult-shaped dict into the canonical Claim."""
    meta = result.get("patient_metadata", {}) or {}
    sections = result.get("note_sections", {}) or {}
    rag = result.get("rag_context", {}) or {}

    diagnoses = [
        Diagnosis(
            code=c.get("code", ""),
            description=c.get("description", ""),
            is_primary=(c.get("type") == "primary"),
            laterality=c.get("laterality"),
            supporting_text=c.get("supporting_text", ""),
            source_section=c.get("source_section", ""),
        )
        for c in result.get("icd_codes", []) if c.get("code")
    ]

    pos = meta.get("place_of_service") or meta.get("pos")

    lines: list[ClaimLine] = []
    for c in result.get("cpt_codes", []):
        if not c.get("code"):
            continue
        lines.append(ClaimLine(
            code=c["code"], code_system="CPT",
            description=c.get("description", ""),
            units=int(c.get("units", 1) or 1),
            modifiers=list(c.get("modifiers", []) or []),
            place_of_service=c.get("place_of_service") or pos,
            linked_diagnoses=list(c.get("linked_diagnoses", []) or []),
            supporting_text=" ".join(c.get("evidence_spans", []) or []) or c.get("supporting_text", ""),
            evidence_spans=list(c.get("evidence_spans", []) or []),
        ))
    for c in result.get("hcpcs_codes", []):
        if not c.get("code"):
            continue
        lines.append(ClaimLine(
            code=c["code"], code_system="HCPCS",
            description=c.get("description", ""),
            units=int(c.get("units", 1) or 1),
            modifiers=list(c.get("modifiers", []) or []),
            place_of_service=c.get("place_of_service") or pos,
            linked_diagnoses=list(c.get("linked_diagnoses", []) or []),
            supporting_text=c.get("supporting_text", ""),
        ))

    return Claim(
        document_id=result.get("document_id", ""),
        date_of_service=_parse_dos(meta),
        place_of_service=pos,
        provider=Provider(
            npi=meta.get("provider_npi") or meta.get("npi"),
            specialty=meta.get("provider_specialty") or "podiatry",
            organization_name=meta.get("facility") or meta.get("organization_name"),
        ),
        payer=Payer(
            name=meta.get("payer", "Medicare"),
            payer_id=meta.get("payer_id"),
            is_medicare=str(meta.get("payer", "Medicare")).lower().startswith("medicare"),
        ),
        subscriber=Subscriber(
            member_id=meta.get("member_id") or meta.get("insurance_id"),
            first_name=meta.get("patient_first_name"),
            last_name=meta.get("patient_last_name"),
            date_of_birth=meta.get("date_of_birth") or meta.get("dob"),
            authorization_number=meta.get("authorization_number") or meta.get("prior_auth_number"),
        ),
        diagnoses=diagnoses,
        lines=lines,
        note_text=sections.get("full_text", "") or result.get("note_text", ""),
        note_sections=sections,
        prior_surgery_info=rag.get("prior_surgery_info", {}) or result.get("prior_surgery_info", {}) or {},
    )


class ClaimScrubber:
    def __init__(self, store: ComplianceDataStore, agents: list[ComplianceAgent] | None = None):
        self.store = store
        self.agents: list[ComplianceAgent] = agents or []

    def scrub(self, result: dict) -> ScrubResult:
        claim = build_claim(result)
        out = ScrubResult(document_id=claim.document_id)
        for agent in self.agents:
            try:
                findings = agent.check(claim) or []
            except Exception as exc:  # an agent crash must never sink the whole scrub
                logger.error(f"agent {agent.filter_id} crashed: {exc}", exc_info=True)
                continue
            out.findings.extend(findings)
        return out.finalize()
