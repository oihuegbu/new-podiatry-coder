"""Propose-then-verify — recall as a candidate GENERATOR, authoritative data as TRUTH.

This is the license-clean substitute for the AMA CPT Alphabetic Index: it needs no
new data, only the authoritative descriptors already loaded. Two bounded LLM steps,
neither of which is ever trusted to emit a billable code from memory:

  PROPOSE  — the model names candidate code NUMBERS it thinks fit the documented
             procedure. Every proposal is then VALIDATED against the authoritative
             registry: a code that does not exist is dropped, and the descriptor is
             read from the record (never from the model). So the model only widens
             the candidate pool; it cannot invent a code or author a descriptor.

  VERIFY   — the model judges, for EVERY candidate on the shortlist, whether that
             candidate's AUTHORITATIVE descriptor is ENTAILED by the documented facts,
             by GENERAL principles (any specialty / code set): every distinguishing
             element the descriptor states — the specific act/service, the
             structure/site, laterality, count, approach, and qualifiers — must be
             supported; near-synonyms that denote different acts are distinguished;
             and a documented specific correctly satisfies an unspecified or "other
             than …" descriptor. The descriptor is the citation; a code is accepted
             only when the documentation entails it, else the line escalates. Nothing
             bills on the model's say-so alone.

Both judging calls return a `Judgement` over the WHOLE shortlist — what still stands and
the NAMED reason every other candidate is out — never just a pick. The caller may release
a code only when exactly ONE candidate is left standing; several standing candidates are a
TIE, and a tie belongs to the document (`tiebreak`), not to a model's preference. Two
models agreeing about one candidate says that candidate is defensible; it says nothing
about the ones neither of them was asked about (Codex F8-R1).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable

from .data_access import CodeSource
from .models import CandidateCode, ClinicalFact
from .requirement import DescriptorRequirement, RequirementJudgement, RequirementStatus

LLMFn = Callable[[str, str], str]

# The providers the two default judgement calls are PINNED to. They are named once, here,
# and both the call itself and its declared identity below read them — so the identity can
# never drift away from the provider actually contacted (which is what a downstream
# independence check is entitled to rely on).
VERIFY_PROVIDER = "openai"
CORROBORATE_PROVIDER = "claude"


def default_verify_llm(system: str, user: str) -> str:
    # use_batch=False: propose/verify are interactive, latency-sensitive calls; the
    # Batches API (~minutes/call) would make the loop unusable.
    from app.core.llm_client import chat_completion
    from app.core.config import OPENAI_MODEL
    out, _ = chat_completion(system, user, model=OPENAI_MODEL, provider=VERIFY_PROVIDER,
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
    out, _ = chat_completion(system, user, model=model, effort=effort,
                             provider=CORROBORATE_PROVIDER,
                             temperature=0.0, json_mode=True, use_batch=False)
    return out


# ---- assertion-origin independence of the corroborating call ----------------------------
# `select_entailed` and `corroborate` make two judgements about the SAME shortlist.
# Their AGREEMENT is only worth something when the two judgements come from DISTINCT
# ORIGINS. Two calls into one vendor's model family are one opinion sampled twice — they
# share training data, tokeniser, alignment and failure modes — so their agreeing is model
# self-confidence, not confirmation, and a `profile_id` (or a second model tier from the
# same vendor) is a LABEL, not an origin.
#
# The identity therefore travels ON the callable as the reviewed provider/model profile,
# and is read through the SAME provider-identity primitive the relation graph uses
# (`extraction.profile_identity`, established with `ExtractionOrigin` in round 4) rather
# than a second, parallel id scheme that could disagree with it.
#
# FAIL-CLOSED: only a POSITIVELY ESTABLISHED difference of provider counts as independent.
# An absent corroborator, the same callable object, and an undeclared identity are all
# "not independent" — "we cannot tell" must never be read as "confirmed".
DISTINCT_ORIGIN = "distinct_origin"        # different declared providers — genuinely independent
SHARED_ORIGIN = "shared_origin"            # one vendor answering twice
UNDECLARED_ORIGIN = "undeclared_origin"    # no declared identity — independence unestablished
NO_CORROBORATION = "no_corroboration"      # no second judgement was made at all
CORROBORATION_ORIGINS = frozenset({DISTINCT_ORIGIN, SHARED_ORIGIN, UNDECLARED_ORIGIN,
                                   NO_CORROBORATION})
# The single place that answers "may this agreement be credited as independent
# confirmation?". A control reads THIS set instead of listing safe values itself.
INDEPENDENT_CORROBORATION_ORIGINS = frozenset({DISTINCT_ORIGIN})


def declare_model_profile(fn: LLMFn, **profile) -> LLMFn:
    """Stamp an LLM callable with the reviewed provider/model identity of the calls it
    makes, so an independence check reads a DECLARED fact rather than guessing from the
    callable's name. Returns the callable, so it can wrap a definition."""
    fn.model_profile = dict(profile)
    return fn


def model_profile_of(fn) -> dict:
    """The declared provider/model identity of an LLM callable ({} when undeclared)."""
    profile = getattr(fn, "model_profile", None) if fn is not None else None
    return dict(profile) if isinstance(profile, dict) else {}


def corroboration_origin(primary: LLMFn | None, corroborator: LLMFn | None) -> str:
    """Which `CORROBORATION_ORIGINS` value describes `corroborator` relative to `primary`.

    `primary` is the call that made the assertion being checked (the entailment selection);
    `corroborator` is the second opinion on it. Only a different DECLARED provider yields
    `DISTINCT_ORIGIN`."""
    if corroborator is None:
        return NO_CORROBORATION
    if corroborator is primary:
        return SHARED_ORIGIN
    from .extraction import profile_identity
    primary_provider, _ = profile_identity(model_profile_of(primary))
    second_provider, _ = profile_identity(model_profile_of(corroborator))
    primary_provider = primary_provider.strip().lower()
    second_provider = second_provider.strip().lower()
    if not primary_provider or not second_provider:
        return UNDECLARED_ORIGIN
    return SHARED_ORIGIN if primary_provider == second_provider else DISTINCT_ORIGIN


declare_model_profile(default_verify_llm, provider=VERIFY_PROVIDER)
declare_model_profile(default_corroborate_llm, provider=CORROBORATE_PROVIDER)


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


# ---- one verdict shape, answered about the WHOLE shortlist -----------------------------
# Codex F8-R1: the selector chose ONE candidate, the corroborator was asked about THAT
# candidate alone, and the two agreeing released it -- while another candidate the same
# documentation entailed just as well was never evaluated by either model. Agreement on a
# candidate establishes that it is DEFENSIBLE. It cannot establish that it is the ONLY
# defensible one, and only the second is a reason to bill without a human.
#
# So both judging calls answer the SAME contract about EVERY shortlisted candidate: which
# ones the documentation still entails, and -- for each of the others -- the NAMED reason
# it is out (a documented fact its descriptor contradicts, an element the documentation
# does not state, or the coding rule another option satisfies better). Silence about a
# candidate is recorded AS silence (`unaccounted`), never read as an elimination.

_SHORTLIST_CONTRACT = """Answer about EVERY numbered option, not only the one you prefer:
  - "entailed": the numbers of ALL options the documentation FULLY entails and that you
    cannot eliminate — each one an equally defensible representation of the documented
    fact. Include an option you did not choose whenever the documentation supports it just
    as well; listing only your preferred option when another is equally supported is wrong.
  - "eliminated": one entry for every OTHER option, each NAMING why it is out: the
    documented fact its descriptor contradicts, the element its descriptor requires that
    the documentation does not state, or the coding rule that rules it out (for example,
    another listed option represents the same documented service more completely or more
    specifically). Set "missing_element": true when the option is the RIGHT KIND of
    service/condition but its descriptor requires an element/qualifier/finding the
    documentation does not state (a documentation gap); false when its descriptor denotes
    a DIFFERENT act, site, condition, or concept (a wrong option).
  - "choice": the single option you would code, which MUST appear in "entailed"; 0 when no
    option is both fully supported and an accurate representation.
Every option number must appear exactly once, in "entailed" or in "eliminated".
Return JSON only:
{"choice": <int>, "entailed": [<int>, ...], "reason": "<short>",
 "eliminated": [{"option": <int>, "reason": "<short>", "missing_element": true|false}, ...]}"""


_SELECT_SYSTEM = """You verify medical codes against clinical documentation. You are
given a documented clinical fact and a NUMBERED list of candidate codes' OFFICIAL
descriptors. Judge which options' descriptors the documentation FULLY entails, and which
one most accurately represents the documented fact — applying these principles
GENERALLY (any specialty; a procedure, service, supply, or DIAGNOSIS; any code set;
reason from the words, not from examples):
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
Judge ONLY the descriptor text; use no knowledge of what a code number 'usually' means.
""" + _SHORTLIST_CONTRACT


_CORROBORATE_SYSTEM = """You INDEPENDENTLY re-judge a shortlist of candidate codes for
documented care. You are given a documented clinical fact and a NUMBERED list of
candidate codes' OFFICIAL descriptors. Decide, skeptically and on the descriptor text
alone, which of them the documentation FULLY ENTAILS, applying general principles (any
specialty; a procedure, service, supply, or DIAGNOSIS; any code set): every clinically
distinguishing element the descriptor states — the specific act, service, or CONDITION;
the structure/site; laterality; count; approach; acuity, stage, or encounter type (e.g.
acute vs chronic, initial vs subsequent); and any qualifiers or BUNDLED COMPONENTS —
must be supported; a near-synonym that denotes a DIFFERENT act, condition, or entity
does not qualify; a documented specific value satisfies an unspecified or "other than …"
descriptor. If you are not confident the documentation entails an option, do not list it
as entailed. Judge only the descriptor text; use no knowledge of what a code number
'usually' means. No other judgement has been shown to you: reach your own.
""" + _SHORTLIST_CONTRACT


# ---- per-requirement judging (issue #6 F9-R6) --------------------------------------
# Additive to the shortlist contract above, never a replacement: when `compile_
# requirements` finds nothing typed for a shortlist, none of this renders and the
# prompt is byte-identical to before this existed. `requirement_id` is always ECHOED
# from the compiled list the caller supplies, never invented by the model -- the same
# discipline `_option_code` already applies to candidate numbers. `authority_offset`/
# `source_identity` are validation-only and never shown to the model: exposing them
# would let a model echo back a plausible-looking offset without truly having found
# the clause, defeating the point of independently re-checking it.
_REQUIREMENTS_CONTRACT = """

Additionally, for EACH numbered REQUIREMENT listed below (each tagged with the
option number it belongs to and its own requirement_id), judge it independently
from the documentation:
  - "supported": the documentation states this requirement's expected value.
  - "contradicted": the documentation states something incompatible with it.
  - "not_documented": the documentation is simply silent on it.
  - "unresolved": you cannot tell from the documentation.
Cite the EXACT bracketed evidence id(s) (e.g. "[e2]") whose text supports a
supported/contradicted verdict; leave span ids empty for not_documented/unresolved.
Never invent a requirement_id not listed below, and never cite an evidence id not
shown above. Add to your JSON:
"requirements": [{"requirement_id": "<id, copied exactly>",
 "status": "supported"|"contradicted"|"not_documented"|"unresolved",
 "span_ids": ["<id>", ...], "quote": "<verbatim quoted text, or empty>"}]"""


def _requirement_options(requirements: tuple[DescriptorRequirement, ...],
                         candidates: list[CandidateCode]) -> str:
    """Per-option compiled requirements, rendered for the shortlist prompt -- axis
    and expected value only (never `authority_offset`/`source_identity`, which are
    validation-only)."""
    if not requirements:
        return ""
    by_code = {c.code: i + 1 for i, c in enumerate(candidates)}
    lines = []
    for req in requirements:
        opt = by_code.get(req.candidate_code)
        if opt is None:
            continue
        tag = "required" if req.required else "optional"
        lines.append(f"option {opt}, {req.requirement_id}: axis={req.axis!r} "
                     f"expected={list(req.expected)!r} ({tag})")
    return "\n".join(lines)


def _evidence_options(fact: ClinicalFact) -> tuple[str, dict[str, str]]:
    """Evidence rendered with a stable bracketed id per quote, and the id->real
    span_id map used to validate a model's cited ids afterward. An unanchored span
    (no span_id yet) still renders -- so the model can still read it -- but is never
    a valid citation target (its id maps to "").

    issue #6 F9-R6-R4: also renders `fact.attribute_evidence`'s spans -- without
    this, a correctly inherited, independently validated attribute (e.g. a
    laterality/site value stated once on a parent fact and inherited from there)
    could settle graph consensus but could never be cited by the requirement
    verifier at all, since it never appears in `fact.evidence`. Uses the SAME
    usability filter `graph_consensus._attribute_span_support` already applies --
    a `"local"` entry is always usable; an `"inherited"` entry needs
    `scope_validated=True` (never a bare `scope_validated=True` filter on its
    own, which would wrongly suppress every local entry too, since those default
    `scope_validated=False`). A span already shown via `fact.evidence` is never
    duplicated under a second tag.
    """
    lines = []
    id_to_span: dict[str, str] = {}
    seen_spans: set[str] = set()
    n = 0
    for s in fact.evidence:
        n += 1
        tag = f"e{n}"
        lines.append(f"[{tag}] {s.text}")
        sid = str(getattr(s, "span_id", "") or "")
        id_to_span[tag] = sid
        if sid:
            seen_spans.add(sid)
    for attr_name, entries in sorted((fact.attribute_evidence or {}).items()):
        for e in entries:
            if not (e.scope == "local" or e.scope_validated):
                continue
            sid = str(getattr(e.span, "span_id", "") or "")
            if not sid or sid in seen_spans:
                continue
            seen_spans.add(sid)
            n += 1
            tag = f"e{n}"
            lines.append(f"[{tag}] {e.span.text}")
            id_to_span[tag] = sid
    return " | ".join(lines), id_to_span


def _best_descriptor(source: CodeSource, cand: CandidateCode) -> str:
    try:
        tiers = source.descriptions(cand.code, cand.system)
    except Exception:
        tiers = []
    return tiers[0] if tiers else cand.descriptor


@dataclass(frozen=True)
class Judgement:
    """ONE model's verdict over a WHOLE shortlist.

    `entailed` is what still stands after this model applied every elimination it could;
    `eliminated` maps each remaining candidate to the reason this model NAMED for ruling
    it out. `unaccounted` are the candidates it simply did not mention — kept separate on
    purpose, because "did not say" is not "eliminated", and a release that requires every
    alternative to carry a named elimination has to be able to tell the two apart.

    `declared` records whether the model answered the shortlist contract at all. An
    undeclared verdict leaves every other candidate unaccounted, so it cannot establish
    uniqueness — the fail-closed direction, reached without a special case.
    """

    chosen: CandidateCode | None = None
    reason: str = ""
    entailed: tuple[str, ...] = ()
    eliminated: dict[str, str] = field(default_factory=dict)
    missing_element: dict[str, bool] = field(default_factory=dict)
    declared: bool = False
    unaccounted: tuple[str, ...] = ()
    #: This model's verdict on each compiled `DescriptorRequirement` (issue #6
    #: F9-R6) -- empty for any shortlist `compile_requirements` found nothing typed
    #: for, or for any judgement made before this field existed. Never trusted on
    #: its own; see `requirement.validated_requirement`.
    requirement_judgements: tuple[RequirementJudgement, ...] = ()

    def entails(self, code: str) -> bool:
        return code in self.entailed

    def elimination_of(self, code: str) -> str:
        """The reason this model NAMED for eliminating `code`, or "" if it did not
        eliminate it. A candidate this model still entails is never eliminated by it, even
        if it also appeared in the eliminated list — a self-contradicting verdict leaves
        the candidate STANDING, which blocks a release rather than allowing one."""
        if code in self.entailed:
            return ""
        return self.eliminated.get(code, "")

    def as_record(self) -> dict:
        return {"chosen": (self.chosen.code if self.chosen else ""),
                "declared_shortlist_verdict": self.declared,
                "entailed": list(self.entailed),
                "eliminated": dict(sorted(self.eliminated.items())),
                "missing_element": sorted(k for k, v in self.missing_element.items() if v),
                "unaccounted": list(self.unaccounted),
                "reason": self.reason,
                "requirements": [rj.as_record() for rj in self.requirement_judgements]}


def _option_code(raw, codes: list[str]) -> str:
    """The candidate code an OPTION NUMBER refers to ("" when it refers to none).

    Booleans are refused explicitly: in Python `True` is an `int` equal to 1, so a model
    answering `{"choice": true}` would otherwise silently select the first candidate."""
    if isinstance(raw, bool):
        return ""
    try:
        i = int(raw)
    except (TypeError, ValueError):
        return ""
    return codes[i - 1] if 1 <= i <= len(codes) else ""


def _requirement_judgements(ans: dict, requirements: tuple[DescriptorRequirement, ...],
                            id_to_span: dict[str, str], evaluator_origin: dict
                            ) -> tuple[RequirementJudgement, ...]:
    """Parse the `"requirements"` field of a model answer, fail-closed exactly like
    `_judgement`'s own candidate parsing: an unknown `requirement_id`, a malformed
    status, or a cited evidence id this shortlist never showed the model are all
    dropped rather than trusted. A cited id that WAS shown but is unanchored (maps
    to "" in `id_to_span`) is dropped too -- an unanchored quote is not a real span
    to validate against."""
    if not requirements:
        return ()
    known_ids = {r.requirement_id for r in requirements}
    valid_statuses = {s.value for s in RequirementStatus}
    raw = ans.get("requirements")
    if not isinstance(raw, list):
        return ()
    out: list[RequirementJudgement] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        rid = str(item.get("requirement_id") or "").strip()
        if not rid or rid not in known_ids or rid in seen:
            continue
        status_raw = str(item.get("status") or "").strip().lower()
        if status_raw not in valid_statuses:
            continue
        raw_spans = item.get("span_ids")
        span_ids: list[str] = []
        if isinstance(raw_spans, list):
            for tag in raw_spans:
                real = id_to_span.get(str(tag).strip())
                if real and real not in span_ids:
                    span_ids.append(real)
        seen.add(rid)
        out.append(RequirementJudgement(
            requirement_id=rid, status=RequirementStatus(status_raw),
            evidence_span_ids=tuple(span_ids),
            quoted_text=str(item.get("quote") or "").strip(),
            evaluator_origin=dict(evaluator_origin)))
    return tuple(out)


def _judgement(ans: dict, candidates: list[CandidateCode],
               requirements: tuple[DescriptorRequirement, ...] = (),
               id_to_span: dict[str, str] | None = None,
               evaluator_origin: dict | None = None) -> Judgement:
    """Parse one model answer into a `Judgement`, fail-closed at every branch: an
    out-of-range option number, a non-list verdict, and an elimination with no stated
    reason are all read as saying NOTHING about that candidate."""
    codes = [c.code for c in candidates]
    by_code = {c.code: c for c in candidates}
    raw_entailed = ans.get("entailed")
    declared = isinstance(raw_entailed, list)
    entailed: list[str] = []
    if declared:
        for opt in raw_entailed:
            code = _option_code(opt, codes)
            if code and code not in entailed:
                entailed.append(code)
    chosen_code = _option_code(ans.get("choice"), codes)
    if chosen_code and chosen_code not in entailed:
        entailed.insert(0, chosen_code)      # choosing an option asserts its entailment
    eliminated: dict[str, str] = {}
    missing: dict[str, bool] = {}
    raw_eliminated = ans.get("eliminated")
    if isinstance(raw_eliminated, list):
        for item in raw_eliminated:
            if not isinstance(item, dict):
                continue
            code = _option_code(item.get("option"), codes)
            why = str(item.get("reason") or "").strip()
            if not code or code in entailed or not why:
                continue                     # an UNNAMED elimination is not an elimination
            eliminated[code] = why
            missing[code] = item.get("missing_element") is True
    return Judgement(
        chosen=by_code.get(chosen_code),
        reason=str(ans.get("reason") or "").strip(),
        entailed=tuple(entailed), eliminated=eliminated, missing_element=missing,
        declared=declared,
        unaccounted=tuple(c for c in codes if c not in entailed and c not in eliminated),
        requirement_judgements=_requirement_judgements(
            ans, requirements, id_to_span or {}, evaluator_origin or {}))


def _shortlist_prompt(fact: ClinicalFact, candidates: list[CandidateCode],
                      source: CodeSource,
                      requirements: tuple[DescriptorRequirement, ...] = ()
                      ) -> tuple[str, dict[str, str]]:
    """(prompt, id_to_span) -- the id_to_span map is needed by the caller to
    validate a model's cited requirement evidence ids afterward.

    When `requirements` is empty, renders BYTE-IDENTICAL to before this field
    existed (plain-text evidence, no REQUIREMENTS section) -- zero format-
    regression risk for the vast majority of shortlists this phase doesn't touch."""
    opts = "\n".join(f"{i + 1}. {_best_descriptor(source, c)}"
                     for i, c in enumerate(candidates))
    if requirements:
        ev, id_to_span = _evidence_options(fact)
    else:
        ev, id_to_span = " | ".join(s.text for s in fact.evidence), {}
    req_block = _requirement_options(requirements, candidates)
    req_section = f"\n\nREQUIREMENTS:\n{req_block}" if req_block else ""
    prompt = (f"DOCUMENTED FACT: {fact.description}\n"
             f"ATTRIBUTES: {json.dumps(fact.attributes)}\n"
             f"EVIDENCE: {ev}\n\n"
             f"CANDIDATE OFFICIAL DESCRIPTORS:\n{opts}"
             f"{req_section}\n\n"
             f"Which options' descriptors does the documentation entail, which single "
             f"option would you code (0 if none), and why is each other option out?")
    return prompt, id_to_span


def select_entailed(fact: ClinicalFact, candidates: list[CandidateCode],
                    source: CodeSource, llm: LLMFn,
                    requirements: tuple[DescriptorRequirement, ...] = ()) -> Judgement:
    """ONE call over the whole shortlist: which candidates' OFFICIAL descriptors the
    documentation entails, which single one this model would code, and the named reason
    every other candidate is out. Judged on the authoritative descriptor text — the
    citation — not on the code number.

    Returns the full verdict rather than only the pick, because the caller has to be able
    to ask whether anything ELSE is still entailed. It is still ONE call over the whole
    shortlist, so the loop stays at two model calls per fact.

    `requirements` (issue #6 F9-R6) is compiled ONCE by the caller and passed
    identically to `select_entailed`/`corroborate`, so "both evaluators judged the
    same requirement" is structural (identical `requirement_id`s), not coincidental.
    When empty, the system/user prompt render byte-identical to before this field
    existed."""
    if not candidates:
        return Judgement(reason="no candidates")
    system = _SELECT_SYSTEM + (_REQUIREMENTS_CONTRACT if requirements else "")
    prompt, id_to_span = _shortlist_prompt(fact, candidates, source, requirements)
    return _judgement(
        _json(llm(system, prompt)), candidates, requirements, id_to_span,
        {"provider": VERIFY_PROVIDER})


def corroborate(fact: ClinicalFact, candidates: list[CandidateCode],
                source: CodeSource, llm: LLMFn,
                requirements: tuple[DescriptorRequirement, ...] = ()) -> Judgement:
    """The INDEPENDENT second judgement, over the SAME shortlist and the SAME contract.

    It is deliberately NOT told which candidate the first model picked: a corroborator that
    only re-confirms a supplied answer cannot notice that a DIFFERENT candidate is also
    entailed, which is exactly the gap this closes. A code therefore bills only when two
    independent judgements agree BOTH that it is entailed AND that nothing else on the
    shortlist survives."""
    if not candidates:
        return Judgement(reason="no candidates")
    system = _CORROBORATE_SYSTEM + (_REQUIREMENTS_CONTRACT if requirements else "")
    prompt, id_to_span = _shortlist_prompt(fact, candidates, source, requirements)
    return _judgement(
        _json(llm(system, prompt)), candidates, requirements, id_to_span,
        {"provider": CORROBORATE_PROVIDER})
