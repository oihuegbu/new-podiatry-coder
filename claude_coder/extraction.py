"""Stage 1 — Clinical Language Understanding (fact extraction).

The model reads the note and emits STRUCTURED CLINICAL FACTS with verbatim
evidence — and nothing else. It is never asked for, and must never output, a
medical code. This is the deliberate inversion: the LLM does the genuinely
LLM-shaped job (understanding messy prose, negation, laterality, whether a thing
was performed vs merely discussed), and the deterministic layer downstream does
the code assignment from authoritative data.

Because the prompt carries no codes, it cannot go stale when the code sets
change, and the hardcoding guard has nothing to catch here.
"""
from __future__ import annotations

import json
import re
from typing import Callable

from .models import ClinicalFact, Disposition, EvidenceSpan, FactKind

# A callable (system_prompt, user_prompt) -> JSON string. Injectable for tests.
LLMFn = Callable[[str, str], str]

_SYSTEM = """You are a clinical language understanding engine for medical coding.
Read the clinical note and extract every DISTINCT billable clinical event as a
structured fact. You describe WHAT HAPPENED in plain clinical language — you do
NOT assign or output any billing codes of any kind.

For each fact return an object with:
  - "kind": one of procedure | diagnosis | supply | drug | imaging |
            evaluation_management
  - "description": a precise clinical phrase for the event (no codes)
  - "attributes": the axes that determine specificity, when documented —
        anatomy, laterality (left/right/bilateral), count/quantity, depth,
        area/size, product/material, drug + dose + wasted amount, approach,
        contrast, technical_vs_professional. Omit what the note does not state;
        never infer laterality, count, or site that is not written.
        For an evaluation_management fact, also give the medical-decision-making
        elements when documented: "problems", "data", "risk" each as one of
        straightforward | low | moderate | high, plus "new_patient" (true/false),
        "setting" (office | emergency | inpatient | observation | nursing | home —
        from the place of service / note header, default office for a clinic),
        "total_time_minutes" if the note records visit time, and
        "separately_identifiable" (true only if the note documents E/M work
        significant and separate from any procedure done the same day).
  - "disposition": performed_today | ordered | planned | discussed |
        historical | unclear  — ONLY performed_today / dispensed work is billable.
        For a PROCEDURE/supply/drug this is whether it was actually done today.
        For a DIAGNOSIS, use performed_today for a CURRENT/active condition
        addressed at this encounter (this is the default for anything in the
        assessment/impression); use historical ONLY when the note frames it as
        past — "history of", "resolved", "status post", or listed under past
        medical history.
  - "negated": true if the note denies/rules out this finding, else false
  - "certainty": confirmed | suspected | ruled_out — a probable/possible/likely/
        working/rule-out/differential condition is "suspected" and, per outpatient
        coding rules, must NOT be coded as if confirmed; "confirmed" for a
        definitively documented condition/finding; "ruled_out" for one the note
        excludes. Default confirmed only when the note states the condition plainly.
  - "experiencer": patient | family | other — whose condition/finding this is; a
        family-history or other-person mention is NOT the patient's coded condition.
  - "evidence": a list of VERBATIM quotes copied exactly from the note that
        support this fact (never paraphrased)
  - "confidence": 0.0-1.0, your certainty this event is documented as stated

Rules: quote evidence verbatim; separate a planned/ordered service from a
performed one; capture negation; do not merge distinct events; do not invent
facts the note does not support. For a DIAGNOSIS, the "description" must be the
concise clinical name of ONE condition — when a note phrase lists several
conditions together, emit a SEPARATE diagnosis fact for each, and keep severity
prose, counts, and functional-limitation wording OUT of the description (put
them in attributes or omit). Return JSON only: {"facts": [ ... ]}."""


def _coerce_kind(value: str) -> FactKind | None:
    try:
        return FactKind(str(value).strip().lower())
    except ValueError:
        return None


def _coerce_disposition(value) -> Disposition:
    # Fail-closed: a missing (None) or unrecognized disposition is UNCLEAR, never
    # assumed performed. Only an explicit, valid disposition is trusted.
    try:
        return Disposition(str(value).strip().lower())
    except (ValueError, AttributeError):
        return Disposition.UNCLEAR


def _extract_json(text: str) -> dict:
    text = text.strip()
    if not text.startswith("{"):
        m = re.search(r"\{.*\}", text, re.DOTALL)
        text = m.group(0) if m else "{}"
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def _default_llm(system: str, user: str) -> str:
    from app.core.llm_client import chat_completion
    out, _ = chat_completion(system, user, temperature=0.0, json_mode=True)
    return out


def extract_facts(note_text: str, llm: LLMFn | None = None) -> list[ClinicalFact]:
    llm = llm or _default_llm
    raw = _extract_json(llm(_SYSTEM, note_text))
    facts: list[ClinicalFact] = []
    for i, item in enumerate(raw.get("facts", []) or []):
        if not isinstance(item, dict):
            continue
        kind = _coerce_kind(item.get("kind", ""))
        desc = str(item.get("description", "")).strip()
        if kind is None or not desc:
            continue
        # A negated finding, or one the note RULES OUT, is documentation of ABSENCE
        # — never billed. An OMITTED certainty defaults to confirmed (a plainly
        # documented condition, per the prompt); an explicit value is taken as-is.
        raw_cert = item.get("certainty")
        certainty = str(raw_cert).strip().lower() if raw_cert is not None else "confirmed"
        if item.get("negated") is True or certainty == "ruled_out":
            continue
        # Fail-closed on both assertion axes: a condition is coded as present ONLY when
        # it is explicitly CONFIRMED — suspected/probable/possible, or any unrecognized
        # certainty, is not coded as confirmed; and it is the PATIENT's condition only
        # when the experiencer is explicitly the patient — family/other, or any
        # unrecognized experiencer, is not the patient's coded condition.
        certain = certainty == "confirmed"
        experiencer = str(item.get("experiencer", "patient")).strip().lower() or "patient"
        spans = [EvidenceSpan(text=str(q)) for q in (item.get("evidence") or [])
                 if str(q).strip()]
        facts.append(ClinicalFact(
            kind=kind,
            description=desc,
            attributes=item.get("attributes") or {},
            disposition=_coerce_disposition(item.get("disposition")),
            certain=certain,
            experiencer=experiencer,
            evidence=spans,
            confidence=float(item.get("confidence") or 0.0),
            fact_id=f"f{i+1}",
        ))
    return facts
