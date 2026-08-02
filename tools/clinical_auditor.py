#!/usr/bin/env python3
"""Clinical-correctness review of the WHOLE final claim, plus every layer
decision that changed it.

The engineering gates (structural validation, replay, registry protection,
pack audits) verify that deterministic layers work as SPECIFIED. This module
verifies that the claim they produced is clinically RIGHT — the review that
was missing when a sibling-arbitration layer rewrote a heel deformity into a
thigh deformity off a tourniquet-placement sentence (routine_00001), and
when a demotion layer moved a load-bearing coverage diagnosis off a claim
without reporting the action, so the corrections-only audit never saw it
(routine_00003). Deterministic-layer errors are guaranteed unanimous across
consistency runs (the layers apply identically every run), so unanimity can
never catch them; this review is the check that can.

How it works:
  - The validator records every claim-mutating action in
    `material_corrections` — self-reported corrections PLUS entries derived
    by diffing the claim state before/after validation, so no layer can act
    unseen. Each is tagged `interpretive` (grounded in note-text
    interpretation) or data-grounded (table lookups); derived entries are
    always interpretive (unknown provenance must be audited).
  - The reviewer is shown the COMPLETE saved record (`full_record`), not a
    curated excerpt: consistency run votes, adjudication decisions, the
    correction ledger, scrubber findings, disposition history — everything
    the pipeline wrote. This is the structural fix for the redaction
    failure measured live on routine_00008: an outside reviewer handed the
    raw output JSON + the doctor's note caught contradictions (an
    adjudication block saying modifiers=[] beside a claim line carrying
    RT; three REVIEW votes beside a CLEAN disposition) that the
    in-pipeline review could not see, because the case file it was built
    excluded exactly those fields. The mechanical cross-field checks live
    in tools/record_coherence.py (zero LLM); this review reads the same
    full record for the CLINICAL defects only judgment can catch.
  - For EVERY result, an expert-coder model (Fable 5, same grounding
    discipline as tools/coder_adjudicator.py) does two jobs: (1) verdict
    each interpretive correction (uphold/overturn/uncertain), and (2)
    review the whole final claim as a payer would — code selection,
    primary designation, missing documented codes, modifiers, linkage,
    coverage logic, and the system's own advisory findings (a HIGH-risk
    recommendation that is authoritatively wrong for the fact pattern is
    itself a reportable defect). Every verdict/finding must cite an
    authority AND quote note evidence; quotes are mechanically verified
    against the note. Ungrounded verdicts degrade to uncertain; ungrounded
    billing-material findings degrade to uncertain materiality (still
    routed); ungrounded advisory findings are dropped.
  - Enforcement is conservative and mechanical: the audit NEVER rewrites
    billing content. All upheld and no material findings -> the claim is
    promoted to CLEAN and may auto-verify into the claims registry. Any
    overturn/uncertain/material finding -> forced to REVIEW with the item
    named, and registry auto-recording is blocked (enforced independently
    in claims_registry.eligible_for_auto). Disputed items and findings are
    enqueued by the triage scan as audit_dispute flip classes — the growth
    loop that turns confirmed review findings into deterministic rules,
    templates, or layers through the actuation acceptance gates (never a
    hardcoded patch).
  - Idempotent: the audit block records a fingerprint of the corrections
    AND the claim shape it judged; a re-run with both unchanged is a no-op.

CLI (inside the app container):
  python tools/clinical_auditor.py [results_dir] [--docs stem1,stem2]
      [--force]

Env:
  CLINICAL_AUDIT         "1" (default) run automatically post-batch
  CLINICAL_AUDITOR_MODEL default claude-fable-5
  CLINICAL_AUDIT_PASSES  independent passes that must agree (default 1;
                         2+ treats any disagreement as uncertain)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from loguru import logger  # noqa: E402

DEFAULT_RESULTS = ROOT / "output" / "results"

AUDITOR_MODEL = os.getenv("CLINICAL_AUDITOR_MODEL", "claude-fable-5")
AUDIT_PASSES = max(1, int(os.getenv("CLINICAL_AUDIT_PASSES", "1")))

_AUDITOR_PROMPT = """\
You are an expert certified professional coder (CPC) performing the final
clinical review of a podiatry claim (private practice, professional
claims) produced by an automated system. You have TWO jobs:

JOB 1 — AUDIT THE SYSTEM'S CORRECTIONS. The rule layers changed the claim
after the initial coding pass; each change is listed in
corrections_under_audit with the layer's stated reason (entries marked
DERIVED were detected by diffing the claim — no layer reported them, so
judge the mutation itself). Judge whether each change is clinically
CORRECT — not whether the rule executed as designed.

JOB 2 — REVIEW THE WHOLE RECORD AS A PAYER WOULD. The case file's
full_record field is the COMPLETE saved output for this note — every
field the system wrote, including the consistency run votes, the
adjudication block's decisions, the correction ledger, the scrubber's
findings, and the disposition history. Read it end to end against the
doctor's note (note_text) and the authoritative reference data. Examine
the final claim: code selection and specificity, primary/secondary
designation, missing documented diagnoses or services, modifiers, units,
diagnosis linkage, and coverage logic. Also examine the record's own
history: if an earlier decision recorded in the record (an adjudicated
verdict, a correction, a run vote) is contradicted by the final claim, or
any recorded reasoning misreads what the note actually documents, report
that as a finding. The case file includes the system's own advisory
findings (system_advisories) — if an advisory's recommendation is
authoritatively WRONG for this fact pattern (e.g. it demands a modifier
this coverage pathway does not use), report that as a finding too. Report
every defect you can ground in an authority; report NOTHING you cannot
ground.

BINDING RULES OF THIS ROLE:
1. AUTHORITY, NOT INTUITION. Every verdict and finding must be derived
   from an authoritative source: the code descriptors and reference data
   in the case file, ICD-10-CM Official Guidelines and Tabular/Index
   conventions, NCCI Policy Manual Chapter 1, the AMA CPT guidelines,
   CMS coverage and documentation policy. Name the source and principle
   in "authority".
2. THE NOTE IS THE ONLY CLINICAL EVIDENCE. Quote the note sentence(s)
   that ground each verdict/finding in "note_evidence" VERBATIM — quotes
   are mechanically checked against the note text, and a quote that does
   not appear (beyond minor formatting differences) is discarded and the
   verdict/finding degraded. If your grounding is the ABSENCE of
   documentation, say so explicitly ("no rupture is documented anywhere
   in the note") rather than inventing a quote. Beware INCIDENTAL
   anatomy: a body part mentioned in tourniquet placement, positioning,
   prep/drape, or anesthesia language is NOT documentation of a
   condition there.
3. JUDGE, CHANGE NOTHING. You have no authority to rewrite the claim.
   Correction verdicts:
   - "uphold": the correction is clinically right.
   - "overturn": clinically wrong — state exactly why, citing the
     authority it violates.
   - "uncertain": the authorities genuinely do not decide it. A note
     routed to a human coder on an uncertain verdict is a correct
     outcome, not a failure.
4. WATCH FOR CONSEQUENTIAL DAMAGE. A correction can be right in
   isolation and wrong in effect (a justified code removal that leaves
   documented work uncoded; a diagnosis swap that corrupted the primary
   designation downstream). Judge the claim-level effect.
5. MATERIALITY DISCIPLINE for findings. "billing_material" means the
   claim's billed content or its coverage outcome is wrong (wrong/missing
   /extra code, wrong primary, wrong modifier/units/linkage, a coverage
   requirement the claim fails). "advisory" means a defect worth fixing
   that does not change what would be billed (a wrong system advisory, a
   style issue, a documentation nicety). Do not inflate advisory issues
   into billing_material.
6. DISPUTED SYSTEM ADVISORIES get their own kind. When the defect is that
   one of the record's own compliance-scrubber ADVISORIES (a WARN entry
   in claim_scrub.findings) is wrong for this note's documented fact
   pattern — e.g. a coverage advisory demanding one pathway's evidence
   when the authority recognizes a distinct pathway the note documents —
   report kind "advisory_defect" with the advisory's code, the governing
   authority, and the note evidence, EVEN IF the advisory's subject is
   coverage/modifiers/units. The claim itself is correct as billed
   (materiality "advisory"); the advisory's emission is what you are
   disputing, and it is adjudicated and mechanized like any other
   dispute. Use the billing kinds only when a BILLED LINE is wrong.

Respond with JSON only:
{"items": [
   {"index": <correction index as given>,
    "verdict": "uphold" | "overturn" | "uncertain",
    "authority": "<source + principle>",
    "note_evidence": "<verbatim quote, or the absence relied on>"}
 ],
 "claim_findings": [
   {"kind": "wrong_code" | "missing_code" | "primary_designation" |
            "modifier" | "units" | "linkage" | "coverage" |
            "advisory_defect" | "other",
    "array": "icd_codes" | "cpt_codes" | "hcpcs_codes" | "claim",
    "code": "<the code at issue, or empty for claim-level>",
    "materiality": "billing_material" | "advisory",
    "finding": "<what is clinically wrong, one or two sentences>",
    "authority": "<source + principle>",
    "note_evidence": "<verbatim quote, or the absence relied on>"}
 ],
 "claim_level_concerns": "<anything clinically wrong that neither a
correction verdict nor a finding captures, or empty string>",
 "overall_rationale": "<2-4 sentences>"}

An empty claim_findings list is the CORRECT output for a clean claim —
do not manufacture findings."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def material_corrections_of(result: dict) -> list[dict]:
    """The result's recorded corrections; falls back to deriving them from
    validation_issues for results saved before the field existed."""
    mats = result.get("material_corrections")
    if isinstance(mats, list) and mats:
        return mats
    out = []
    for i in result.get("validation_issues") or []:
        if not isinstance(i, dict):
            continue
        if str(i.get("message", "")).startswith("AUTO-CORRECTED"):
            out.append({
                "category": i.get("category", ""),
                "code": i.get("code", ""),
                "action": "auto_correction",
                # pre-field results can't distinguish; audit them all
                "interpretive": True,
                "message": i.get("message", ""),
            })
    return out


# Review-protocol version, salted into the fingerprint: a stored verdict
# identifies not just WHAT claim state it judged but what the reviewer was
# SHOWN when judging it. v2 = the full-record case file (the reviewer sees
# the complete saved output, not a curated excerpt). v3 = disputed system
# advisories are first-class: the reviewer is instructed to report a wrong
# scrubber advisory as kind "advisory_defect" (not to shoehorn it into a
# billing kind), and such findings now DISPUTE the verdict regardless of
# materiality so they reach adjudication and record as verified emission
# targets. Bumping this makes every verdict rendered under the older
# protocol stale — those claims fail closed back into the pending hold and
# get one fresh review under the current standards.
_AUDIT_PROTOCOL_VERSION = 3


def corrections_fingerprint(result: dict) -> str:
    mats = material_corrections_of(result)
    sig = {
        "protocol": _AUDIT_PROTOCOL_VERSION,
        # The measurement vocabulary is part of what the reviewer was
        # SHOWN (synthesized observables add finding kinds to the prompt)
        # — when an observable installs, every verdict rendered without
        # it goes stale and the notes are re-reviewed under the grown
        # vocabulary. This is how "a gate grew, re-run the note against
        # the new system" reaches the review layer mechanically.
        "measurement_vocabulary": sorted(_observable_kinds()),
        "corrections": [(m.get("category"), m.get("code"), m.get("message"))
                        for m in mats],
        "claim": [
            [(e.get("code"), e.get("type"), tuple(e.get("modifiers") or []))
             for e in (result.get(arr) or []) if isinstance(e, dict)]
            for arr in ("icd_codes", "cpt_codes", "hcpcs_codes")],
        # Adjudicated decisions are part of what the review judges (the
        # full record shows them) — a changed decision must stale the
        # verdict even when the claim arrays happen to look the same.
        "adjudication": [
            (i.get("array"), i.get("code"), i.get("kind"),
             i.get("decision"), i.get("decision_code"), i.get("fields"))
            for i in ((result.get("adjudication") or {}).get("items") or [])
            if isinstance(i, dict)],
    }
    return hashlib.sha256(
        json.dumps(sig, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _full_record_view(result: dict) -> dict:
    """The complete saved record, as the reviewer's primary exhibit — the
    same artifact a human reviewer reads when handed the output JSON.
    Only two redactions, both non-informational: the PRIOR clinical_audit
    block (a stale verdict must not anchor the fresh one) and the full
    note text duplicated inside rag_context (provided once, un-truncated,
    as the case's note_text)."""
    view = json.loads(json.dumps(result, default=str))
    view.pop("clinical_audit", None)
    rag = view.get("rag_context")
    if isinstance(rag, dict) and rag.get("note_full_text"):
        rag["note_full_text"] = "(provided separately as note_text)"
    return view


def assemble_case(doc: str, result: dict, note: str, rep) -> dict:
    from tools.auto_actuate import _authoritative_evidence

    mats = material_corrections_of(result)
    interp = [dict(m, index=i) for i, m in enumerate(mats)
              if m.get("interpretive")]
    def _array_of(c: str) -> str:
        # dotted or long alpha codes are ICD; exactly letter+4-digits is
        # HCPCS; everything else (5 digits / 4 digits+letter) is CPT.
        # A misclassification only means an empty reference lookup.
        if "." in c or (c[:1].isalpha() and not
                        (len(c) == 5 and c[1:].isdigit())):
            return "icd_codes"
        if c[:1].isalpha():
            return "hcpcs_codes"
        return "cpt_codes"

    # Reference data for EVERY code the review can touch: the whole final
    # claim, the corrections' codes, and the demoted supporting codes —
    # the whole-claim review is only as grounded as the data it is shown.
    codes_by_array: dict[str, set] = {}
    for m in interp:
        c = str(m.get("code") or "").upper()
        if c:
            codes_by_array.setdefault(_array_of(c), set()).add(c)
    for arr in ("icd_codes", "cpt_codes", "hcpcs_codes"):
        for e in (result.get(arr) or []):
            if isinstance(e, dict) and e.get("code"):
                codes_by_array.setdefault(arr, set()).add(
                    str(e["code"]).upper())
    for e in (result.get("supporting_conditions") or []):
        if isinstance(e, dict) and e.get("code"):
            codes_by_array.setdefault("icd_codes", set()).add(
                str(e["code"]).upper())
    # Adjudicated codes too: the full record shows every adjudication
    # decision, including ones that excluded a code from the final claim —
    # the reviewer needs that code's reference data to judge the decision.
    for i in ((result.get("adjudication") or {}).get("items") or []):
        if not isinstance(i, dict):
            continue
        for c in ([i.get("code")] + list(i.get("codes") or [])):
            c = str(c or "").upper()
            if c:
                arr = str(i.get("array") or "") or _array_of(c)
                if arr not in ("icd_codes", "cpt_codes", "hcpcs_codes"):
                    arr = _array_of(c)
                codes_by_array.setdefault(arr, set()).add(c)
    evidence = {}
    for arr, codes in codes_by_array.items():
        try:
            evidence[arr] = _authoritative_evidence(rep, arr, sorted(codes))
        except Exception:
            evidence[arr] = []

    # The system's own advisory conclusions, offered for the contradiction
    # check: a HIGH-risk recommendation that is authoritatively wrong for
    # this fact pattern is itself a reportable defect (measured live,
    # routine_00003: the scrubber demanded Q-modifiers on the coverage
    # pathway that uses none, contradicting the RAG layer in the same
    # result — and nothing reconciled them).
    advisories = []
    scrub = result.get("claim_scrub") or {}
    for fnd in (scrub.get("findings") or []):
        if isinstance(fnd, dict):
            advisories.append({
                "source": f"scrubber/{fnd.get('filter_id')}",
                "codes": fnd.get("codes"),
                "denial_risk": fnd.get("denial_risk"),
                "reason": str(fnd.get("reason"))[:400],
                "recommendation": str(fnd.get("recommendation"))[:300],
            })
    for w in (result.get("validation_issues") or []):
        if isinstance(w, dict) and w.get("severity") == "WARNING" \
                and not str(w.get("message", "")).startswith(
                    ("AUTO-CORRECTED", "AUTO-ADDED")):
            advisories.append({
                "source": f"validator/{w.get('category')}",
                "codes": [w.get("code")],
                "denial_risk": w.get("denial_risk"),
                "reason": str(w.get("message"))[:400],
                "recommendation": str(w.get("recommendation"))[:300],
            })

    return {
        "document_id": doc,
        "note_text": note[:12000],
        "payer": str((result.get("patient_metadata") or {})
                     .get("insurance") or ""),
        # The COMPLETE saved record — consistency votes, adjudication
        # decisions, correction ledger, scrubber findings, disposition
        # history. The curated fields below remain as the review's index
        # into the record (correction indices, per-code reference data),
        # but the record itself is what the reviewer reads end to end.
        "full_record": _full_record_view(result),
        "final_claim": {arr: [
            {k: e.get(k) for k in ("code", "description", "modifiers",
                                   "units", "type", "linked_diagnoses")
             if e.get(k) is not None}
            for e in (result.get(arr) or []) if isinstance(e, dict)]
            for arr in ("icd_codes", "cpt_codes", "hcpcs_codes")},
        # Codes the layers left OFF the claim, with the stated reasons —
        # the whole-claim review must see what was demoted to judge
        # whether the claim is missing something documented.
        "supporting_conditions_not_billed": [
            {k: e.get(k) for k in ("code", "description", "type",
                                   "review_reason") if e.get(k)}
            for e in (result.get("supporting_conditions") or [])
            if isinstance(e, dict)],
        "corrections_under_audit": interp,
        "system_advisories": advisories[:20],
        "authoritative_reference_data": evidence,
    }


def _vocabulary_supplement() -> str:
    """Synthesized measurement observables extend the reviewer's finding
    vocabulary: each contributes finding kinds (beyond the static schema)
    with its own measurement doc. Built-ins are already covered by the
    static prompt's advisory_defect instruction."""
    try:
        from tools.observables import all_observables
        lines = []
        for name, e in sorted(all_observables().items()):
            if e.get("builtin"):
                continue
            kinds = ", ".join(f'"{k}"' for k in e["finding_kinds"])
            lines.append(f"- kinds {kinds} (measured by {name}): "
                         f"{str(e['schema_doc'])[:400]}")
        if not lines:
            return ""
        return ("\n\nADDITIONAL FINDING KINDS (synthesized measurement "
                "observables — use these kinds, with the phenomenon's "
                "code, when the defect they describe is what you are "
                "disputing; like advisory_defect they dispute the record "
                "regardless of materiality):\n" + "\n".join(lines))
    except Exception:
        return ""


def _audit_once(case: dict, pass_idx: int = 0) -> dict:
    from app.core.config import LLM_PROVIDER
    from app.core.llm_client import chat_completion
    user = (f"CASE FILE:\n{json.dumps(case, indent=1, default=str)}\n\n"
            f"Audit every correction listed in corrections_under_audit, "
            f"then read full_record end to end against the doctor's note "
            f"(note_text) and the authoritative reference data, and review "
            f"the whole final claim (including the system advisories and "
            f"the not-billed supporting conditions). If "
            f"preliminary_reviewer_notes is present, it is an earlier "
            f"open-ended read — treat each point as a LEAD to check, not a "
            f"finding: act on it only when you can independently ground it "
            f"in an authority and a verbatim note quote, and ignore any "
            f"lead you cannot. Report your grounded findings.")
    model = AUDITOR_MODEL if LLM_PROVIDER == "claude" else None
    temperature = 0.05 if pass_idx == 0 else 0.4
    system = _AUDITOR_PROMPT + _vocabulary_supplement()
    # The final whole-claim review is the highest-stakes judgment in the
    # pipeline (it gates CLEAN), so it runs at the maximum deliberation
    # budget — matching the coding verify pass, not the cheaper default.
    try:
        text, usage = chat_completion(
            system_prompt=system, user_prompt=user,
            model=model, temperature=temperature, max_tokens=8192,
            json_mode=True, effort="xhigh")
    except Exception as exc:
        if model is None:
            raise
        logger.warning(f"Auditor model {model!r} failed ({exc}) — "
                       f"falling back to the pipeline default")
        model = None
        text, usage = chat_completion(
            system_prompt=system, user_prompt=user,
            temperature=temperature, max_tokens=8192,
            json_mode=True, effort="xhigh")
    verdict = json.loads(text)
    verdict["_model"] = model or "pipeline-default"
    verdict["_usage"] = usage
    return verdict


_EXPLORATORY_PROMPT = """You are a senior medical-coding auditor giving a \
claim a FIRST, OPEN-ENDED read before any scoring. This is the "fresh eyes on \
the finished artifact" pass: reason aloud, follow your instincts, and do NOT \
fill in any schema or scores yet.

You are handed the COMPLETE saved output for one note (full_record), the \
doctor's note (note_text), and the authoritative reference data. Read the \
whole claim end to end against the note and the authorities and think about \
what a careful human expert would notice on a first pass:

- Does every procedure the note documents appear on the claim, coded or \
explicitly accounted for? Is anything the surgeon did missing, or is anything \
billed that the note does not support?
- Is each code the RIGHT code for what the note describes (not just a \
plausible sibling in the same family)? Do the descriptors actually match the \
documented work?
- Do the diagnoses, laterality, sequencing, and modifiers fit the note and \
the coding rules?
- Does anything look internally inconsistent, or like an artifact of the \
automated pipeline rather than the clinical reality?

Write a plain-prose list of everything that looks wrong, risky, or worth a \
second look — each with the note detail and the coding/authority reason it \
concerns you. Raising a suspicion you are unsure about is fine here; this pass \
generates LEADS, and every lead is independently re-verified against the \
authorities in the scored pass that follows. Do not invent facts about the \
note. If the claim looks clean, say so and say why."""


def _exploratory_scan(case: dict) -> str:
    """Free-form 'what's wrong with this claim?' pass — open-ended expert
    reasoning over the FINISHED claim BEFORE the structured verdict schema
    constrains it. Its prose is fed into the scored passes as leads to
    check, never as findings: every lead the scored pass acts on must still
    independently cite an authority + a mechanically-verified note quote, so
    an unfounded suspicion here cannot become a verdict. Fail-open — this is
    an enhancement to the scored review, not a gate; if it errors the audit
    proceeds without preliminary notes (the scored passes are unchanged)."""
    from app.core.config import LLM_PROVIDER
    from app.core.llm_client import chat_completion
    model = AUDITOR_MODEL if LLM_PROVIDER == "claude" else None
    user = (f"CASE FILE:\n{json.dumps(case, indent=1, default=str)}\n\n"
            f"Give this finished claim your open-ended first read. What looks "
            f"wrong, risky, or incomplete against the note and the "
            f"authoritative data?")
    try:
        text, _ = chat_completion(
            system_prompt=_EXPLORATORY_PROMPT, user_prompt=user,
            model=model, temperature=0.5, max_tokens=4096,
            json_mode=False, effort="xhigh")
        return (text or "").strip()
    except Exception as exc:  # fail-open: enhancement, not a gate
        logger.warning(f"  exploratory audit pass failed ({exc}) — "
                       f"proceeding to scored review without preliminary notes")
        return ""


def _grounded(item: dict) -> bool:
    return bool(str(item.get("authority") or "").strip()) and \
        bool(str(item.get("note_evidence") or "").strip())


# Absence-shaped evidence ("no rupture documented", "denies trauma") is a
# legitimate grounding that has no sentence to quote — it is carved out of
# the verbatim check. A negation/absence marker anywhere in the evidence
# string is the signal. These are lexicon words, never medical codes.
_ABSENCE_RE = re.compile(
    r"\b(no|not|non|without|absent|absence|none|nowhere|never|neither|nor|"
    r"denies|denied|lack|lacks|lacking|negative|unremarkable|undocumented|"
    r"unspecified|fails?\s+to|no\s+mention|not\s+documented)\b", re.I)


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(s or "").lower()).strip()


def _evidence_supported(item: dict, note: str) -> bool:
    """A quote-shaped note_evidence must actually occur in the note —
    verbatim after normalization, or (tolerating light paraphrase: dropped
    articles, reordering, case/punctuation) with >=80% of its content
    tokens present. Absence-grounded evidence is carved out (nothing to
    quote). No note text to check against -> not this gate's failure
    (returns True; other gates still apply). This closes the one channel a
    false UPHOLD could rest on — a fabricated quote — and its only failure
    direction is degrading a real verdict to 'uncertain' (human review)."""
    ev = str(item.get("note_evidence") or "").strip()
    if not ev or not note:
        return bool(ev)
    if _ABSENCE_RE.search(ev):
        return True
    nnote = _norm(note)
    ntoks = set(nnote.split())
    frags = [f for f in re.split(r"\.\.\.|\u2026", ev) if _norm(f)]
    if not frags:
        return True
    for frag in frags:
        if _norm(frag) in nnote:
            continue
        ftoks = set(_norm(frag).split())
        if not ftoks:
            continue
        overlap = sum(1 for t in ftoks if t in ntoks) / len(ftoks)
        if overlap < 0.8:
            return False
    return True


def _verdict_map(verdict: dict, wanted: set[int],
                 note: str = "") -> dict[int, str] | None:
    """{correction index -> uphold|overturn|uncertain}; None when the
    verdict is malformed. Unknown/ungrounded verdicts and verdicts whose
    note quote cannot be found in the note degrade to 'uncertain' (fail
    closed, never fail open)."""
    out: dict[int, str] = {}
    for item in (verdict.get("items") or []):
        try:
            idx = int(item.get("index"))
        except (TypeError, ValueError):
            return None
        if idx not in wanted or idx in out:
            return None
        v = str(item.get("verdict") or "").lower()
        if (v not in ("uphold", "overturn", "uncertain")
                or not _grounded(item)
                or not _evidence_supported(item, note)):
            v = "uncertain"
        out[idx] = v
    return out if set(out) == wanted else None


_FINDING_KINDS = {"wrong_code", "missing_code", "primary_designation",
                  "modifier", "units", "linkage", "coverage",
                  "advisory_defect", "other"}


def _observable_kinds() -> set[str]:
    """Finding kinds claimed by measurement observables
    (tools/observables.py) — advisory_defect plus whatever synthesized
    observables register. These kinds route to adjudication regardless of
    materiality (the claim is correct as billed; the phenomenon's
    emission is what's disputed), and they stay first-class kinds in
    _findings_of instead of degrading to 'other'."""
    try:
        from tools.observables import all_observables
        return {k for e in all_observables().values()
                for k in e["finding_kinds"]}
    except Exception:
        return {"advisory_defect"}


def _findings_of(verdict: dict, note: str) -> list[dict]:
    """Normalize and gate the whole-claim findings of one audit pass.
    The gates mirror the correction-verdict discipline, with the failure
    direction chosen per materiality: a billing_material finding that is
    ungrounded or whose quote fails verification DEGRADES TO uncertain
    materiality (still routes to a human — fail closed); an advisory
    finding that fails the gates is DROPPED (it changes no billing, so
    an ungrounded one is pure noise)."""
    out = []
    allowed = _FINDING_KINDS | _observable_kinds()
    for f in (verdict.get("claim_findings") or []):
        if not isinstance(f, dict):
            continue
        kind = str(f.get("kind") or "other").lower()
        if kind not in allowed:
            kind = "other"
        arr = str(f.get("array") or "claim").lower()
        if arr not in ("icd_codes", "cpt_codes", "hcpcs_codes", "claim"):
            arr = "claim"
        materiality = str(f.get("materiality") or "").lower()
        grounded = _grounded(f) and _evidence_supported(f, note)
        if materiality == "billing_material":
            if not grounded:
                materiality = "uncertain"
        elif materiality != "advisory":
            materiality = "uncertain" if grounded else ""
        elif not grounded:
            continue
        if not materiality:
            continue
        if not str(f.get("finding") or "").strip():
            continue
        out.append({
            "kind": kind, "array": arr,
            "code": str(f.get("code") or "").upper(),
            "materiality": materiality,
            "finding": str(f.get("finding"))[:500],
            "authority": str(f.get("authority") or "")[:400],
            "note_evidence": str(f.get("note_evidence") or "")[:400],
        })
    return out


def _corroborate_findings(all_findings: list[list[dict]]) -> list[dict]:
    """Multi-pass hardening for findings (passes >= 2): a routing-grade
    finding (billing_material/uncertain materiality, or the
    advisory_defect kind — which routes regardless of materiality because
    it feeds adjudication) must be corroborated — some finding on the
    same (array, code) in every pass — or it loses its routing power
    (material findings degrade to advisory materiality; an advisory_defect
    degrades to the non-routing 'other' kind), still logged and
    growth-queued. With a single pass this is the identity."""
    primary = all_findings[0]
    if len(all_findings) < 2:
        return primary
    keysets = [{(f["array"], f["code"]) for f in fl}
               for fl in all_findings[1:]]
    obs_kinds = _observable_kinds()
    out = []
    for f in primary:
        routing = (f["materiality"] in ("billing_material", "uncertain")
                   or f["kind"] in obs_kinds)
        if routing and \
                not all((f["array"], f["code"]) in ks for ks in keysets):
            f = dict(f, materiality=("advisory"
                                     if f["materiality"] != "advisory"
                                     else f["materiality"]),
                     corroboration="not corroborated by all passes — "
                                   "degraded to advisory")
            if f["kind"] in obs_kinds:
                f["kind"] = "other"
        out.append(f)
    return out


def audit_result(doc: str, result: dict, note: str, rep,
                 passes: int = AUDIT_PASSES) -> dict:
    """Audit one result in place: verify every interpretive correction AND
    review the whole final claim (the browser-expert review, run through
    the deterministic gates). ALWAYS runs — a claim with no interpretive
    corrections still gets the whole-claim review, because the absence of
    self-reported corrections is exactly what an unreported mutation looks
    like (measured live, routine_00003). Returns the clinical_audit block
    (also mutates result routing on dispute)."""
    mats = material_corrections_of(result)
    interp_idx = {i for i, m in enumerate(mats) if m.get("interpretive")}
    fingerprint = corrections_fingerprint(result)

    case = assemble_case(doc, result, note, rep)
    # Open-ended "what's wrong with this claim?" pass FIRST, so unconstrained
    # expert reasoning happens before the verdict schema narrows attention.
    # Its prose rides along in the case as LEADS the scored passes must still
    # independently ground; it never bypasses the authority+quote gate.
    prelim = _exploratory_scan(case)
    if prelim:
        case["preliminary_reviewer_notes"] = prelim
    maps, verdicts, findings_per_pass = [], [], []
    for i in range(passes):
        try:
            v = _audit_once(case, pass_idx=i)
        except Exception as exc:
            logger.warning(f"  audit pass {i + 1} failed: {exc}")
            maps.append(None)
            continue
        verdicts.append(v)
        maps.append(_verdict_map(v, interp_idx, note))
        findings_per_pass.append(_findings_of(v, note))

    if not verdicts or any(m is None for m in maps):
        final = {i: "uncertain" for i in interp_idx}
        basis = "audit incomplete — treated as uncertain (fail closed)"
    elif any(m != maps[0] for m in maps[1:]):
        final = {i: (maps[0][i] if all(m[i] == maps[0][i] for m in maps)
                     else "uncertain") for i in interp_idx}
        basis = f"{passes} independent passes; disagreements -> uncertain"
    else:
        final = maps[0]
        basis = f"{passes} independent grounded pass(es), unanimous"

    # Whole-claim findings: gated per pass, corroborated across passes.
    # An incomplete audit (no verdicts at all) fails closed via the empty
    # findings + uncertain corrections path; when there are no interpretive
    # corrections either, the explicit incomplete marker below disputes.
    findings = (_corroborate_findings(findings_per_pass)
                if findings_per_pass else [])
    # Observable-kind findings (advisory_defect + synthesized observables'
    # kinds) dispute REGARDLESS of materiality: the billed lines are
    # right, but the record carries a measured phenomenon the authorities
    # contradict — that is a mechanizable dispute (the adjudicator rules
    # on its emission and the verdict records as a verified emission
    # target), and an "upheld" verdict here would ship the record CLEAN
    # with the defect unadjudicated, deadlocking its flip class at
    # awaiting_verification forever.
    obs_kinds = _observable_kinds()
    material = [f for f in findings
                if f["materiality"] in ("billing_material", "uncertain")
                or f["kind"] in obs_kinds]

    disputed = {i: v for i, v in final.items() if v != "uphold"}
    concerns = str((verdicts[0].get("claim_level_concerns") if verdicts
                    else "") or "").strip()
    incomplete = not verdicts
    block = {
        "at": _now(),
        "model": verdicts[0].get("_model") if verdicts else None,
        "passes": passes,
        "verdict": ("upheld" if not disputed and not material
                    and not concerns and not incomplete else "disputed"),
        "fingerprint": fingerprint,
        "basis": basis,
        "items": (verdicts[0].get("items") if verdicts else
                  [{"index": i, "verdict": "uncertain",
                    "authority": "", "note_evidence": ""}
                   for i in sorted(interp_idx)]),
        "claim_findings": findings,
        "claim_level_concerns": concerns or (
            "audit incomplete — no pass returned a verdict (fail closed)"
            if incomplete else ""),
        "overall_rationale": (verdicts[0].get("overall_rationale", "")
                              if verdicts else ""),
        "protocol": ("authority-grounded expert-coder review of the whole "
                     "final claim plus every interpretive layer correction "
                     "(self-reported and diff-derived); ungrounded/split "
                     "verdicts degrade to uncertain; any non-uphold or "
                     "billing-material finding forces REVIEW and blocks "
                     "registry auto-recording; advisory findings feed the "
                     "actuation growth queue without routing"),
    }
    result["clinical_audit"] = block
    _enforce_verdict(result, block, mats, disputed,
                     block["claim_level_concerns"])
    from app.release.claim_readiness import refresh_release_artifacts
    refresh_release_artifacts(result)
    return block


AUDIT_PENDING_MARKER = "[clinical_audit/pending]"


def _scrub_clean(result: dict) -> bool:
    scrub = result.get("claim_scrub") or {}
    return bool(scrub.get("clean")) or \
        str(scrub.get("disposition", "")).upper() == "CLEAN"


def _enforce_verdict(result: dict, block: dict, mats: list[dict],
                     disputed: dict, concerns: str) -> None:
    """Realize the audit verdict on the result's routing. The audit stage
    now sits INSIDE the CLEAN path: the pipeline holds a scrub-CLEAN claim
    with interpretive corrections at REVIEW under the pending marker, and
    this is the only code that releases it — upheld promotes to CLEAN,
    anything else replaces the pending hold with the named dispute.

    Prior [clinical_audit/...] entries are stripped along with the pending
    marker: this verdict SUPERSEDES the previous review's, and leaving the
    old entries in place duplicated every finding once per re-review
    (measured live on routine_00001: the convergence loop's three review
    passes left three copies of each finding in the review reasons)."""
    reasons = [r for r in (result.get("auto_coding_review_reasons") or [])
               if AUDIT_PENDING_MARKER not in str(r)
               and "[clinical_audit/" not in str(r)]

    # DETERMINISTIC OVERRIDE GUARD: an upheld review never promotes a
    # claim whose replay overrode an adjudicated decision (measured live
    # on routine_00008: a layer re-added the modifier the adjudicator had
    # removed, and the review graded the recurrence advisory). The review
    # judges clinical substance; verdict fidelity is code's job — the
    # conflict must be resolved (rule fixed or human decision), not
    # graded.
    overridden = (result.get("adjudication") or {}).get(
        "overridden_by_replay")
    if block["verdict"] == "upheld" and overridden:
        marker = "[adjudication/overridden]"
        kept = [r for r in reasons if marker not in str(r)]
        result["auto_coding_review_reasons"] = kept + [
            f"{marker} {c.get('array')}/{c.get('code')}: replay produced "
            f"{c.get('observed')} where the adjudicated verdict required "
            f"{c.get('decision')} — layer-vs-adjudicator conflict, human "
            f"decision required" for c in overridden]
        if str(result.get("final_disposition", "")).upper() == "CLEAN":
            result["final_disposition"] = "REVIEW"
            result["auto_coding_tier"] = "REVIEW"
            result["auto_coding_confidence"] = min(
                float(result.get("auto_coding_confidence") or 0.0), 0.84)
        return

    if block["verdict"] == "upheld":
        # RECORD COHERENCE GUARD (deterministic): before an upheld review
        # may release the claim, the record must agree with itself —
        # adjudicated decisions realized, correction ledger matching the
        # claim state, linkage intact, one first-listed diagnosis. An
        # outside reviewer caught the 00008 defects by reading the saved
        # record for exactly these contradictions; this is that reading,
        # mechanized at the promotion gate.
        try:
            from tools.record_coherence import (COHERENCE_MARKER,
                                                coherence_violations)
            violations = coherence_violations(
                result, require_audit_release=False)
        except Exception as exc:  # fail closed: unverifiable ≠ coherent
            violations = [f"coherence gate could not run ({exc})"]
            COHERENCE_MARKER = "[record_coherence]"
        if violations:
            kept = [r for r in reasons if COHERENCE_MARKER not in str(r)]
            result["auto_coding_review_reasons"] = kept + [
                f"{COHERENCE_MARKER} {viol}" for viol in violations]
            if str(result.get("final_disposition", "")).upper() == "CLEAN":
                result["final_disposition"] = "REVIEW"
                result["auto_coding_tier"] = "REVIEW"
                result["auto_coding_confidence"] = min(
                    float(result.get("auto_coding_confidence") or 0.0),
                    0.84)
            return
        result["auto_coding_review_reasons"] = reasons
        # An upheld audit releases ONLY its own hold — a note routed to
        # REVIEW for non-unanimity (or any other verdict) stays routed.
        cons = result.get("consistency") or {}
        routed = (result.get("review_routing") == "routed"
                  or (cons and not cons.get("unanimous")))
        if not routed and _scrub_clean(result) and \
                str(result.get("final_disposition", "")).upper() != "CLEAN":
            scrub = result.get("claim_scrub") or {}
            result["final_disposition"] = "CLEAN"
            result["auto_coding_tier"] = "AUTO"
            result["final_summary"] = scrub.get(
                "summary", result.get("final_summary", ""))
            result["auto_coding_summary"] = result["final_summary"]
            result["auto_coding_confidence"] = max(
                float(result.get("auto_coding_confidence") or 0.0), 0.85)
        return

    for i in sorted(disputed):
        m = mats[i]
        reasons.append(
            f"[clinical_audit/{disputed[i]}] {m.get('category')} on "
            f"{m.get('code')}: {str(m.get('message'))[:160]}")
    obs_kinds = _observable_kinds()
    for f in (block.get("claim_findings") or []):
        if f.get("materiality") in ("billing_material", "uncertain") \
                or f.get("kind") in obs_kinds:
            reasons.append(
                f"[clinical_audit/finding/{f.get('kind')}] "
                f"{f.get('code') or 'claim'}: "
                f"{str(f.get('finding'))[:200]}")
    if concerns:
        reasons.append(f"[clinical_audit/claim] {concerns[:300]}")
    if str(result.get("final_disposition", "")).upper() == "CLEAN":
        result["final_disposition"] = "REVIEW"
        result["auto_coding_tier"] = "REVIEW"
        result["auto_coding_confidence"] = min(
            float(result.get("auto_coding_confidence") or 0.0), 0.84)
    result["auto_coding_review_reasons"] = reasons


def audit_batch(results_dir: Path, docs: list[str] | None = None,
                rep=None, force: bool = False) -> dict:
    stats = {"audited": 0, "upheld": 0, "disputed": 0, "skipped": 0,
             "docs": {}}
    targets = []
    for f in sorted(results_dir.glob("*_results.json")):
        if f.name == "all_results.json":
            continue
        doc = f.stem.removesuffix("_results")
        if docs is not None and doc not in docs:
            continue
        try:
            result = json.loads(f.read_text())
        except Exception:
            continue
        if not isinstance(result, dict) or not result.get("success"):
            continue
        targets.append((doc, f, result))
    if not targets:
        return stats

    if rep is None:
        from tools.auto_actuate import Replayer
        rep = Replayer()
    from tools.auto_actuate import _load_runs, _note_text_for

    for doc, f, result in targets:
        prior = result.get("clinical_audit") or {}
        if not force and prior.get("fingerprint") == \
                corrections_fingerprint(result):
            # The stored verdict already covers these exact corrections —
            # no LLM re-spend, but its ROUTING effect must still be
            # realized (a re-scrub can re-place the pending hold after
            # the audit ran; the uphold must release it again).
            before = json.dumps(result, sort_keys=True, default=str)
            mats = material_corrections_of(result)
            disputed = {int(i["index"]): str(i.get("verdict") or "")
                        for i in (prior.get("items") or [])
                        if str(i.get("verdict") or "").lower() != "uphold"
                        and isinstance(i.get("index"), int)
                        and 0 <= i["index"] < len(mats)}
            _enforce_verdict(result, prior, mats, disputed,
                             str(prior.get("claim_level_concerns") or ""))
            from app.release.claim_readiness import refresh_release_artifacts
            refresh_release_artifacts(result)
            if json.dumps(result, sort_keys=True, default=str) != before:
                f.write_text(json.dumps(result, indent=2, default=str))
            stats["skipped"] += 1
            stats["docs"][doc] = f"unchanged since last audit ({prior.get('verdict')})"
            continue
        note = _note_text_for(doc, results_dir,
                              _load_runs(doc, results_dir), result)
        if not note:
            stats["skipped"] += 1
            stats["docs"][doc] = "no note text available — not audited"
            continue
        block = audit_result(doc, result, note, rep)
        f.write_text(json.dumps(result, indent=2, default=str))
        stats["audited"] += 1
        stats[block["verdict"]] += 1
        n_interp = len([m for m in material_corrections_of(result)
                        if m.get("interpretive")])
        stats["docs"][doc] = f"{block['verdict']} ({n_interp} correction(s))"
        logger.info(f"Clinical audit {doc}: {block['verdict']} "
                    f"({n_interp} interpretive correction(s))")
    return stats


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results_dir", nargs="?", default=str(DEFAULT_RESULTS))
    p.add_argument("--docs", default="")
    p.add_argument("--force", action="store_true",
                   help="re-audit even when the corrections are unchanged")
    args = p.parse_args()
    docs = [s.strip() for s in args.docs.split(",") if s.strip()] or None
    stats = audit_batch(Path(args.results_dir), docs=docs, force=args.force)
    print(json.dumps(stats, indent=2, default=str))


if __name__ == "__main__":
    main()
