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
    from app.core.config import OPENAI_MODEL
    out, _ = chat_completion(system, user, model=OPENAI_MODEL, provider="openai",
                             temperature=0.0, json_mode=True, use_batch=False)
    return out


def default_corroborate_llm(system: str, user: str) -> str:
    """The INDEPENDENT second opinion — a different/stronger model (CLAUDE_VERIFY_
    MODEL/EFFORT, typically the Opus verification tier) so agreement is genuine
    cross-provider corroboration, not the same provider re-confirming itself. The
    verifier is pinned to OpenAI and this corroborator to Anthropic; an optional
    CLAUDE_VERIFY_MODEL selects the Anthropic tier."""
    from app.core.llm_client import chat_completion
    model = effort = None
    try:
        from app.core import config
        model = getattr(config, "CLAUDE_VERIFY_MODEL", "") or None
        effort = getattr(config, "CLAUDE_VERIFY_EFFORT", "") or None
    except Exception:
        pass
    out, _ = chat_completion(system, user, model=model, effort=effort, provider="claude",
                             temperature=0.0, json_mode=True, use_batch=False)
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


_PROPOSE_SYSTEM = """You propose CANDIDATE code NUMBERS for a documented clinical
fact — a procedure, service, supply, drug, or diagnosis. List the codes most likely
to represent EXACTLY what was documented — these are only candidates that will be
verified against the official descriptor, so include close alternatives. Do not
explain, do not invent codes you are unsure of. Return JSON only:
{"codes": ["<code>", "<code>", ...]} (max 6)."""


def propose_codes(fact: ClinicalFact, source: CodeSource, llm: LLMFn,
                  max_codes: int = 6) -> list[CandidateCode]:
    """LLM-proposed candidates, each VALIDATED against the authoritative registry
    (nonexistent codes dropped; descriptor read from the record). The model widens
    the pool; it never supplies truth."""
    ev = " | ".join(s.text for s in fact.evidence)
    user = (f"SYSTEM (code set): {fact.system.upper()}\n"
            f"DOCUMENTED FACT ({fact.kind.value}): {fact.description}\n"
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
descriptors. Choose the ONE option whose descriptor the documentation FULLY entails
AND which most accurately represents the documented fact — applying these
principles GENERALLY (any specialty; a procedure, service, supply, or DIAGNOSIS;
any code set; reason from the words, not from examples):
  - EVERY clinically distinguishing element the descriptor states must be supported
    by the documentation: the specific act, service, or CONDITION; the
    structure/site/organ; laterality; count/quantity; approach/technique; acuity,
    stage, or encounter type (e.g. acute vs chronic, initial vs subsequent); and any
    qualifiers or BUNDLED COMPONENTS (with/without a feature, material, type/status).
  - Do NOT choose a code that ASSERTS MORE than the documentation supports — e.g. a
    descriptor that bundles additional components, steps, or findings the note does
    not document. Over-assertion is not entailment.
  - Do NOT choose a code that OMITS a documented, clinically significant part of the
    service when a more complete, fully-supported option exists — that under-
    represents the service.
  - Distinguish near-synonyms that denote clinically DIFFERENT acts or entities:
    overlapping wording is not a match unless the documentation supports that exact
    meaning.
  - A documented specific value satisfies a descriptor that is unspecified or
    defined as "other than" a DIFFERENT value; a descriptor naming a value the
    documentation contradicts, or requiring an element it does not state, is NOT
    entailed.
If NO option is BOTH fully supported AND an accurate representation, choose 0. Judge
ONLY the descriptor text; use no knowledge of what a code number 'usually' means.
Return JSON only: {"choice": <int>, "reason": "<short>"}"""


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


_CORROBORATE_SYSTEM = """You INDEPENDENTLY check whether a candidate code is correct
for documented care. You are given a documented clinical fact and ONE candidate
code's OFFICIAL descriptor. Decide, skeptically and on the descriptor text alone,
whether the documentation FULLY ENTAILS that exact descriptor, applying general
principles (any specialty; a procedure, service, supply, or DIAGNOSIS; any code
set): every clinically distinguishing element the descriptor states — the specific
act, service, or CONDITION; the structure/site; laterality; count; approach; acuity,
stage, or encounter type (e.g. acute vs chronic, initial vs subsequent); and any
qualifiers or BUNDLED COMPONENTS — must be supported; a near-synonym
that denotes a DIFFERENT act, condition, or entity does not qualify; a documented
specific value satisfies an unspecified or "other than …" descriptor.
If it is NOT entailed, also classify WHY:
  - "missing_element": true when the code is the RIGHT KIND of service/condition for
    what was documented, but its descriptor REQUIRES an element/qualifier/finding
    the documentation does not state (a documentation gap — the note may simply be
    incomplete);
  - "missing_element": false when the descriptor denotes a DIFFERENT act, site,
    condition, or concept than what was documented (a genuinely wrong code).
If you are not confident the documentation entails it, answer entailed=false. Judge
only the descriptor text; use no knowledge of what a code number 'usually' means.
Return JSON only:
{"entailed": true|false, "missing_element": true|false, "reason": "<short>"}"""


def corroborate(fact: ClinicalFact, candidate: CandidateCode,
                source: CodeSource, llm: LLMFn) -> tuple[bool, str, bool]:
    """An INDEPENDENT second-model entailment check on the already-selected code.
    Returns (entailed, reason, missing_element). `entailed` is True only when this
    model also finds the documentation fully entails the descriptor — so a code
    bills only when TWO independent judgements agree. On disagreement,
    `missing_element` distinguishes a DOCUMENTATION GAP (right kind of code, an
    element undocumented -> a provider query) from a WRONG CODE (re-select)."""
    ev = " | ".join(s.text for s in fact.evidence)
    user = (f"DOCUMENTED FACT: {fact.description}\n"
            f"ATTRIBUTES: {json.dumps(fact.attributes)}\n"
            f"EVIDENCE: {ev}\n\n"
            f"CANDIDATE OFFICIAL DESCRIPTOR: {_best_descriptor(source, candidate)}\n\n"
            f"Is this descriptor fully entailed by the documentation?")
    ans = _json(llm(_CORROBORATE_SYSTEM, user))
    return (ans.get("entailed") is True, str(ans.get("reason") or "").strip(),
            ans.get("missing_element") is True)
