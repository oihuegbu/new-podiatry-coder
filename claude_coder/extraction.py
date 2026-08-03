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
        straightforward | low | moderate | high, plus "new_patient" (true/false)
        and "total_time_minutes" if the note records visit time.
  - "disposition": performed_today | ordered | planned | discussed |
        historical | unclear  — ONLY performed_today / dispensed work is billable
  - "negated": true if the note denies/rules out this finding, else false
  - "evidence": a list of VERBATIM quotes copied exactly from the note that
        support this fact (never paraphrased)
  - "confidence": 0.0-1.0, your certainty this event is documented as stated

Rules: quote evidence verbatim; separate a planned/ordered service from a
performed one; capture negation; do not merge distinct events; do not invent
facts the note does not support. Return JSON only: {"facts": [ ... ]}."""


def _coerce_kind(value: str) -> FactKind | None:
    try:
        return FactKind(str(value).strip().lower())
    except ValueError:
        return None


def _coerce_disposition(value: str) -> Disposition:
    try:
        return Disposition(str(value).strip().lower())
    except ValueError:
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
        # A negated finding is documentation of ABSENCE — never billed.
        if item.get("negated") is True:
            continue
        spans = [EvidenceSpan(text=str(q)) for q in (item.get("evidence") or [])
                 if str(q).strip()]
        facts.append(ClinicalFact(
            kind=kind,
            description=desc,
            attributes=item.get("attributes") or {},
            disposition=_coerce_disposition(item.get("disposition", "performed_today")),
            evidence=spans,
            confidence=float(item.get("confidence") or 0.0),
            fact_id=f"f{i+1}",
        ))
    return facts
