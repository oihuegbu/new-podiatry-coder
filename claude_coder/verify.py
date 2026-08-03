"""Propose-then-verify — recall as a candidate GENERATOR, authoritative data as TRUTH.

This is the license-clean substitute for the AMA CPT Alphabetic Index: it needs no
new data, only the authoritative descriptors already loaded. Two bounded LLM steps,
neither of which is ever trusted to emit a billable code from memory:

  PROPOSE  — the model names candidate code NUMBERS it thinks fit the documented
             procedure. Every proposal is then VALIDATED against the authoritative
             registry: a code that does not exist is dropped, and the descriptor is
             read from the record (never from the model). So the model only widens
             the candidate pool; it cannot invent a code or author a descriptor.

  VERIFY   — the model judges whether a candidate code's AUTHORITATIVE descriptor is
             ENTAILED by the documented facts, by GENERAL principles (any specialty
             / code set): every distinguishing element the descriptor states — the
             specific act/service, the structure/site, laterality, count, approach,
             and qualifiers — must be supported; near-synonyms that denote different
             acts are distinguished; and a documented specific correctly satisfies an
             unspecified or "other than …" descriptor. The descriptor is the
             citation; a code is accepted only when the documentation entails it,
             else the line escalates. Nothing bills on the model's say-so alone.

The caller runs these over a ranked candidate pool (retrieval + proposals) and takes
the first code whose descriptor is verified — grounding every accepted procedure in
an authoritative descriptor the documentation supports.
"""
from __future__ import annotations

import json
import re
from typing import Callable

from .data_access import CodeSource
from .models import CandidateCode, ClinicalFact

LLMFn = Callable[[str, str], str]


def default_verify_llm(system: str, user: str) -> str:
    # use_batch=False: propose/verify are interactive, latency-sensitive calls; the
    # Batches API (~minutes/call) would make the loop unusable.
    from app.core.llm_client import chat_completion
    out, _ = chat_completion(system, user, temperature=0.0, json_mode=True,
                             use_batch=False)
    return out


def _json(text: str) -> dict:
    text = (text or "").strip()
    if not text.startswith("{"):
        m = re.search(r"\{.*\}", text, re.DOTALL)
        text = m.group(0) if m else "{}"
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


_PROPOSE_SYSTEM = """You propose CANDIDATE procedure/supply code NUMBERS for a
documented service. List the codes most likely to represent EXACTLY what was
documented — these are only candidates that will be verified against the official
descriptor, so include close alternatives. Do not explain, do not invent codes you
are unsure of. Return JSON only: {"codes": ["<code>", "<code>", ...]} (max 6)."""


def propose_codes(fact: ClinicalFact, source: CodeSource, llm: LLMFn,
                  max_codes: int = 6) -> list[CandidateCode]:
    """LLM-proposed candidates, each VALIDATED against the authoritative registry
    (nonexistent codes dropped; descriptor read from the record). The model widens
    the pool; it never supplies truth."""
    ev = " | ".join(s.text for s in fact.evidence)
    user = (f"SYSTEM (code set): {fact.system.upper()}\n"
            f"PROCEDURE: {fact.description}\n"
            f"ATTRIBUTES: {json.dumps(fact.attributes)}\n"
            f"EVIDENCE: {ev}\n\nList candidate {fact.system.upper()} codes.")
    ans = _json(llm(_PROPOSE_SYSTEM, user))
    out: list[CandidateCode] = []
    seen: set[str] = set()
    for raw in (ans.get("codes") or [])[:max_codes]:
        code = re.sub(r"[^A-Za-z0-9]", "", str(raw)).upper()
        if not code or code in seen:
            continue
        rec = source.lookup(code, fact.system)
        if not rec:                       # fabricated or wrong system -> drop
            continue
        seen.add(code)
        desc = (rec.get("long_description") or rec.get("description")
                or rec.get("short_description") or "")
        out.append(CandidateCode(
            code=code, system=fact.system, descriptor=str(desc), score=0.0,
            source="llm-proposed-validated",
            authority={"source": "LLM proposal validated against authoritative record"}))
    return out


_SELECT_SYSTEM = """You verify medical codes against clinical documentation. You are
given a documented clinical fact and a NUMBERED list of candidate codes' OFFICIAL
descriptors. Choose the ONE option whose descriptor is fully ENTAILED by the
documentation, applying these principles GENERALLY (they hold for any specialty,
procedure, diagnosis, or code set — reason from the words, not from examples):
  - Every clinically distinguishing element the descriptor states must be supported
    by the documentation: the specific act/service performed, the
    structure/site/organ, laterality, count or quantity, approach or technique, and
    any qualifiers (with/without a feature, material or substance, stage, acuity,
    encounter type).
  - Distinguish near-synonyms that denote clinically DIFFERENT acts or entities:
    similar or overlapping wording is not a match unless the documentation supports
    that exact meaning.
  - Honor generality correctly. A documented specific value satisfies a descriptor
    that is unspecified, or that is defined as "other than" a DIFFERENT specific
    value. But a descriptor that names a specific value the documentation
    contradicts — or that requires an element the documentation does not state — is
    NOT entailed.
If NO option's descriptor is entailed by the documentation, choose 0. Judge ONLY
the descriptor text against the documentation; use no knowledge of what a code
number 'usually' means. Return JSON only: {"choice": <int>, "reason": "<short>"}"""


def _best_descriptor(source: CodeSource, cand: CandidateCode) -> str:
    try:
        tiers = source.descriptions(cand.code, cand.system)
    except Exception:
        tiers = []
    return tiers[0] if tiers else cand.descriptor


def select_entailed(fact: ClinicalFact, candidates: list[CandidateCode],
                    source: CodeSource, llm: LLMFn) -> tuple[CandidateCode | None, str]:
    """ONE call: among the candidates, the single one whose OFFICIAL descriptor the
    documentation entails (or None). Judged on the authoritative descriptor text —
    the citation — not on the code number. A single call over the shortlist keeps
    the loop cheap (propose + select = 2 LLM calls per procedure)."""
    if not candidates:
        return None, "no candidates"
    opts = "\n".join(f"{i + 1}. {_best_descriptor(source, c)}"
                     for i, c in enumerate(candidates))
    ev = " | ".join(s.text for s in fact.evidence)
    user = (f"DOCUMENTED FACT: {fact.description}\n"
            f"ATTRIBUTES: {json.dumps(fact.attributes)}\n"
            f"EVIDENCE: {ev}\n\n"
            f"CANDIDATE OFFICIAL DESCRIPTORS:\n{opts}\n\n"
            f"Which option's descriptor is entailed by the documentation? (0 if none)")
    ans = _json(llm(_SELECT_SYSTEM, user))
    choice = ans.get("choice")
    reason = str(ans.get("reason") or "").strip()
    if isinstance(choice, int) and 1 <= choice <= len(candidates):
        return candidates[choice - 1], reason
    return None, reason
