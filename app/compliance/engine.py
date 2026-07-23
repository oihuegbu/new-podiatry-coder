"""ClaimScrubber — orchestrates the 14 compliance agents and applies the gate.

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
    Claim, ClaimLine, DenialRisk, Diagnosis, Finding, Payer, Provider,
    ScrubResult, Status, Subscriber,
)
from app.compliance.geo import state_from_text
from app.compliance.payer_registry import parse_insurance_text
from app.core.logger import get_logger

logger = get_logger(__name__)


def _parse_date(raw: str) -> date | None:
    raw = str(raw or "").strip()
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


def _parse_dos(meta: dict) -> date | None:
    raw = (meta or {}).get("date_of_service") or (meta or {}).get("dos") or ""
    return _parse_date(raw)


def _split_name(meta: dict) -> tuple[str | None, str | None]:
    """The extraction schema only produces a single "patient_name" field
    (e.g. "Margaret Holloway") — there is no separate first/last field, so
    Stedi's 270 subscriber.firstName/lastName must be derived here rather
    than read from keys that never exist in patient_metadata."""
    raw = str((meta or {}).get("patient_name") or "").strip()
    if not raw:
        return None, None
    parts = raw.split()
    if len(parts) == 1:
        return parts[0], None
    return " ".join(parts[:-1]), parts[-1]


def _parse_dob(meta: dict) -> str | None:
    """Subscriber.date_of_birth is documented as YYYYMMDD (X12) — the note's
    extracted DOB comes in whatever prose format the source PDF used (e.g.
    "03/14/1978"), so it must be normalized here, once, rather than left for
    each clearinghouse adapter to reinterpret."""
    raw = (meta or {}).get("date_of_birth") or (meta or {}).get("dob") or ""
    parsed = _parse_date(raw)
    return parsed.strftime("%Y%m%d") if parsed else None


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

    # patient_metadata has no "payer"/"member_id" keys — the LLM extracts a
    # single free-text "insurance" field (format varies note to note). Parse
    # it here so payer/member ID are derived from what the note actually
    # says, not defaulted to Medicare for every claim regardless of insurer.
    ins = parse_insurance_text(meta.get("insurance", ""))
    first_name, last_name = _split_name(meta)

    # Claim state (for MAC-jurisdiction scoping of LCDs/Articles): the notes
    # carry no structured address, so infer from the insurance line ("Medicaid
    # Florida", "BCBS of Florida") and the note's own letterhead ("City, FL
    # 33134"). Unknown stays None — jurisdiction scoping then does nothing.
    note_text = sections.get("full_text", "") or result.get("note_text", "")
    state = state_from_text(
        meta.get("insurance", "") or "",
        meta.get("facility", "") or "",
        note_text,
    )

    return Claim(
        document_id=result.get("document_id", ""),
        date_of_service=_parse_dos(meta),
        place_of_service=pos,
        state=state,
        provider=Provider(
            npi=meta.get("provider_npi") or meta.get("npi"),
            specialty=meta.get("provider_specialty") or "podiatry",
            organization_name=meta.get("facility") or meta.get("organization_name"),
        ),
        payer=Payer(
            # "Unknown", not "Medicare" — claiming Medicare when the note has
            # no insurance text at all would repeat the exact bug this fixes.
            name=ins.payer_name or "Unknown",
            payer_id=ins.payer_id,
            stedi_trading_partner_id=ins.stedi_trading_partner_id,
            is_medicare=ins.is_medicare,
            kind=ins.kind,
            follows_medicare_coverage=ins.follows_medicare_coverage,
        ),
        subscriber=Subscriber(
            member_id=meta.get("member_id") or meta.get("insurance_id") or ins.member_id,
            first_name=first_name,
            last_name=last_name,
            date_of_birth=_parse_dob(meta),
            authorization_number=meta.get("authorization_number") or meta.get("prior_auth_number"),
        ),
        diagnoses=diagnoses,
        lines=lines,
        note_text=sections.get("full_text", "") or result.get("note_text", ""),
        note_sections=sections,
        prior_surgery_info=rag.get("prior_surgery_info", {}) or result.get("prior_surgery_info", {}) or {},
    )


def _apply_advisory_suppressions(findings: list, suppressions: list[dict],
                                 filter_id: str) -> list:
    """Honor validator-rule advisory suppressions for one agent's findings.

    A suppression names a (filter_id, code); it removes matching findings
    of Status.WARN ONLY — a FAIL is the clean-claim gate and can never be
    config-suppressed — and replaces each with a PASS finding recording
    the rule id and authority, so the record shows a rule decision
    ("checked, and the advisory yields to a documented pathway"), never a
    silently vanished check. Grown for advisory-shaped audit disputes
    (the deterministic realization of an adjudicated 'this advisory is
    wrong for this fact pattern' verdict)."""
    if not suppressions:
        return findings
    by_code: dict[str, dict] = {}
    for s in suppressions:
        if not isinstance(s, dict):
            continue
        if str(s.get("filter_id") or "").strip() != filter_id:
            continue
        code = str(s.get("code") or "").strip().upper()
        if code:
            by_code[code] = s
    if not by_code:
        return findings
    out = []
    for f in findings:
        s = None
        if f.status == Status.WARN:
            for c in f.codes:
                s = by_code.get(str(c).strip().upper())
                if s:
                    break
        if s is None:
            out.append(f)
            continue
        why = str(s.get("note") or "a documented alternative pathway "
                                    "satisfies the governing policy")
        out.append(Finding(
            filter_id=f.filter_id, filter_name=f.filter_name,
            status=Status.PASS, codes=list(f.codes),
            denial_risk=DenialRisk.NONE,
            reason=(f"Advisory suppressed by validator rule "
                    f"{s.get('rule_id') or '(unnamed)'}: {why} "
                    f"[suppressed advisory: {f.reason[:200]}]"),
            recommendation="No action needed — the suppressing rule's "
                           "authority citation governs this fact pattern.",
            source_rule=str(s.get("authority") or "")[:300]
            or f.source_rule,
        ))
        logger.info(f"  [scrub] {filter_id} advisory on "
                    f"{','.join(f.codes)} suppressed by rule "
                    f"{s.get('rule_id')!r}")
    return out


class ClaimScrubber:
    def __init__(self, store: ComplianceDataStore, agents: list[ComplianceAgent] | None = None):
        self.store = store
        self.agents: list[ComplianceAgent] = agents or []

    def scrub(self, result: dict) -> ScrubResult:
        claim = build_claim(result)
        suppressions = [s for s in
                        (result.get("scrub_advisory_suppressions") or [])
                        if isinstance(s, dict)]
        out = ScrubResult(document_id=claim.document_id)
        for agent in self.agents:
            try:
                findings = agent.check(claim) or []
            except Exception as exc:  # an agent crash must never sink the whole scrub
                logger.error(f"agent {agent.filter_id} crashed: {exc}", exc_info=True)
                # A crashed filter is NOT a passed filter — record it so the
                # output never presents an unchecked claim as fully checked.
                out.filter_results.append({
                    "filter_id": agent.filter_id, "filter_name": agent.filter_name,
                    "status": "ERROR", "findings": 0,
                })
                continue
            findings = _apply_advisory_suppressions(
                findings, suppressions, agent.filter_id)
            out.findings.extend(findings)
            statuses = {f.status for f in findings}
            status = ("FAIL" if Status.FAIL in statuses
                      else "WARN" if Status.WARN in statuses else "PASS")
            out.filter_results.append({
                "filter_id": agent.filter_id, "filter_name": agent.filter_name,
                "status": status, "findings": len(findings),
            })
        return out.finalize(filter_count=len(self.agents))
