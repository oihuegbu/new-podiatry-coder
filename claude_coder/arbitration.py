"""Stage 3 — Bounded arbitration (LLM, only for residual ambiguity).

Called only when deterministic resolution found several comparable candidates.
The model is shown the clinical fact, its evidence, and the SHORTLIST of
candidate descriptors that were RETRIEVED from the authoritative source — and is
asked to pick the option whose descriptor the note best supports, by NUMBER. It
therefore selects among data-provided codes; it can never recall or invent a
code from memory, so a wrong answer costs a review, never a fabricated bill.

Below a confidence floor the line stays abstained and escalates to a human.
"""
from __future__ import annotations

import json
import re
from typing import Callable

from .models import CandidateCode, ResolutionMethod, ResolvedLine

LLMFn = Callable[[str, str], str]
_MIN_CONFIDENCE = 0.6

_SYSTEM = """You are disambiguating a medical-coding candidate. You are given a
documented clinical fact and a numbered list of candidate code DESCRIPTORS that
were retrieved from the authoritative code set. Choose the single option whose
descriptor is ENTAILED by the fact and its evidence, applying these principles
generally (they hold for any specialty, service, or code set — reason from the
words, not from examples):
  - every clinically distinguishing element the descriptor states must be supported
    by the documentation (the specific act/service, the structure/site, laterality,
    count/quantity, approach or technique, and any qualifiers);
  - near-synonyms that denote clinically DIFFERENT acts or entities are not a match
    unless the documentation supports that exact meaning;
  - a documented specific value satisfies a descriptor that is unspecified or that
    is defined as "other than" a DIFFERENT specific value; a descriptor that names a
    value the documentation contradicts, or omits, is not entailed.
Do not use any knowledge of code numbers; judge only the descriptor text against
the fact. If no option is clearly entailed, choose 0. Return JSON only:
{"choice": <int>, "confidence": 0.0-1.0, "reason": "<short>"}"""


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


def _prompt(line: ResolvedLine) -> str:
    fact = line.fact
    ev = " | ".join(s.text for s in fact.evidence)
    opts = "\n".join(f"{i+1}. {c.descriptor}" for i, c in enumerate(line.alternatives))
    return (f"FACT: {fact.description}\n"
            f"ATTRIBUTES: {json.dumps(fact.attributes)}\n"
            f"EVIDENCE: {ev}\n\n"
            f"CANDIDATE DESCRIPTORS:\n{opts}\n\n"
            f"Pick the best option number (or 0 if none fits).")


def arbitrate(line: ResolvedLine, llm: LLMFn | None = None) -> ResolvedLine:
    if line.resolved or not line.alternatives:
        return line
    llm = llm or _default_llm
    ans = _extract_json(llm(_SYSTEM, _prompt(line)))

    choice = ans.get("choice")
    confidence = float(ans.get("confidence") or 0.0)
    reason = str(ans.get("reason") or "").strip()

    valid = isinstance(choice, int) and 1 <= choice <= len(line.alternatives)
    if valid and confidence >= _MIN_CONFIDENCE:
        chosen: CandidateCode = line.alternatives[choice - 1]
        return ResolvedLine(
            fact=line.fact,
            chosen=chosen,
            alternatives=[c for c in line.alternatives if c is not chosen][:4],
            method=ResolutionMethod.ARBITRATED,
            rationale=f"arbitrated (confidence {confidence:.2f}): {reason}")

    # Not confidently resolvable -> keep abstained for human review.
    return ResolvedLine(
        fact=line.fact, chosen=None, alternatives=line.alternatives,
        method=ResolutionMethod.ABSTAINED,
        rationale=f"arbitration inconclusive (confidence {confidence:.2f}) — escalate")
