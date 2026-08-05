"""End-to-end orchestration.

    note ─► extract facts (CLU, code-free)
         ─► resolve each fact -> code (deterministic, from authoritative data)
         ─► arbitrate only the ambiguous ones (bounded LLM over retrieved codes)
         ─► positive release gates (fail-closed, data-backed)
         ─► autonomy controller (AUTO_READY | REVIEW_REQUIRED | BLOCKED)

Every step is pluggable: pass a `MockSource` and stub LLMs to run the whole
pipeline deterministically in a test, or the real `AuthoritativeSource` in
production. The result carries its own audit trail (evidence -> fact -> code ->
method -> authority) so any decision can be explained.
"""
from __future__ import annotations

from . import arbitration, certificate, em, extraction, gates, ontology, resolution
from .arbitration import LLMFn
from .autonomy import decide
from .data_access import AuthoritativeSource, CodeSource
from .models import CodingResult, ResolutionMethod, ResolvedLine


def code_encounter(
    encounter_id: str,
    note_text: str,
    date_of_service: str | None,
    source: CodeSource | None = None,
    extract_llm: LLMFn | None = None,
    arbitrate_llm: LLMFn | None = None,
    verify_llm: LLMFn | None = None,
    corroborate_llm: LLMFn | None = None,
    modifier_engine: "ModifierEngine | None" = None,
) -> CodingResult:
    from .models import Outcome
    from .modifiers import ModifierEngine
    source = source or AuthoritativeSource()
    modifier_engine = modifier_engine or ModifierEngine()

    # Propose-then-verify is enabled in real mode (no stubbed LLMs). It grounds every
    # procedure code in an authoritative descriptor the documentation entails — the
    # license-clean substitute for the CPT Index. In real mode it is also corroborated
    # by an INDEPENDENT second model, so a procedure bills only when two independent
    # judgements agree. Tests pass stub LLMs and leave these None -> deterministic
    # path unchanged, no corroboration.
    if verify_llm is None and arbitrate_llm is None:
        from .verify import default_corroborate_llm, default_verify_llm
        verify_llm = default_verify_llm
        if corroborate_llm is None:
            corroborate_llm = default_corroborate_llm

    from .models import FactKind
    facts = extraction.extract_facts(note_text, extract_llm)
    # Phase-0 (SHADOW): anchor every evidence quote to a verified source offset+hash and
    # record a shadow audit artifact. Fail-safe — anchoring only adds offsets to spans
    # (billing/gates read span text, unchanged) and a write failure must not affect
    # release. Retrieval still consumes ClinicalFacts today; ClaimLineIntent is Phase 1.
    _elig_state: dict = {}
    try:
        from . import provenance as _prov
        _prov.anchor_facts(note_text, facts)
        from app.core.config import OUTPUT_DIR
        _repo = _prov.JsonlAuditRepository(OUTPUT_DIR / "audit")
        _repo.append(encounter_id, "evidence_anchoring", _prov.anchoring_report(facts))
        # Phase-1a (SHADOW): run the code-free eligibility engine and record which events
        # WOULD become claim-line intents vs are non-claim/held -- for audit + diffing
        # against today's performed==billable behavior. Relations are empty until Phase 2,
        # so nothing is demoted yet; this establishes the seam, not a release change.
        from . import eligibility as _elig
        _intents = _elig.evaluate(facts, [], encounter_id, date_of_service)
        _repo.append(encounter_id, "eligibility_shadow", _elig.summary(_intents))
        _repo.append(encounter_id, "eligibility_diff", _elig.shadow_diff(facts, _intents))
        _elig_state = {e: it for it in _intents for e in it.clinical_event_ids}
    except Exception:
        _elig_state = {}

    # Phase 1c: retrieval consumes the eligibility decision. A billable event only reaches
    # code retrieval via an ELIGIBLE ClaimLineIntent; one explicitly demoted as an integral
    # component (NON_CLAIM_EVIDENCE) is diverted BEFORE retrieval -- a performed event is not
    # automatically searched for a code. ENABLED ONE GATE AT A TIME: only the part_of gate
    # diverts for now (a billable fact turns NON_CLAIM only via an explicit PART_OF -- an
    # occurrence-based NON_CLAIM is already non-billable). Evidence/ownership/conflict stay
    # SHADOW until validated on more notes. Relations are empty until Phase 2, so the divert
    # set is currently empty (shadow-diff verified zero divergence) -- behavior-identical now.
    from .eligibility import EligibilityState as _ES
    lines = []
    for fact in facts:
        _it = _elig_state.get(fact.fact_id)
        if fact.kind is FactKind.EM:
            line = em.resolve_em(fact, source)      # MDM-driven leveling
        elif (fact.billable and _it is not None
              and _it.state is _ES.NON_CLAIM_EVIDENCE):
            _r = "; ".join(f"{d.gate}: {d.detail}" for d in _it.decisions
                           if d.outcome is not Outcome.PASS) or _it.state.value
            line = ResolvedLine(
                fact=fact, chosen=None, method=ResolutionMethod.ABSTAINED,
                rationale=f"diverted before retrieval — documented integral component, "
                          f"not an independent claim line ({_r})")
        else:
            line = resolution.resolve(fact, source, llm=verify_llm,
                                      corroborate=corroborate_llm,
                                      dos=date_of_service)
        # A fact that went through propose-then-verify is already resolved-or-
        # escalated on authoritative entailment; don't second-guess it with the
        # weaker arbitration fallback. (Diagnoses verify too when they reach the
        # embedding fallback.) Other kinds still arbitrate residual ambiguity.
        went_through_pv = (verify_llm is not None
                           and fact.kind in (FactKind.PROCEDURE, FactKind.IMAGING,
                                             FactKind.DIAGNOSIS))
        if (not line.resolved) and line.alternatives and fact.billable and not went_through_pv:
            line = arbitration.arbitrate(line, arbitrate_llm)
        # OBSERVE: feed a propose-then-verify success into the learned index so that,
        # once the same phrase->code is confirmed across enough distinct encounters,
        # it resolves deterministically next time. Real mode only; fail-safe.
        if (verify_llm is not None and line.resolved
                and line.method is ResolutionMethod.VERIFIED
                and fact.kind in (FactKind.PROCEDURE, FactKind.IMAGING)):
            from . import learned
            learned.observe(encounter_id, fact.description, line.chosen.code,
                            line.chosen.system, line.chosen.descriptor,
                            [s.text for s in fact.evidence])
        # ICD-10-CM 'highest documented specificity': sharpen an unspecified/NOS
        # diagnosis to the most-specific code the documentation entails — a
        # structural laterality upgrade, then (in real mode) a verified upgrade past
        # a broad catch-all to a specific on-concept relative. Authoritative and
        # entailment-checked; escalates rather than billing an unspecified code when
        # the record supports a specific one but verification is split.
        if line.resolved and fact.kind is FactKind.DIAGNOSIS:
            line = resolution.refine_diagnosis_specificity(
                line, source, verify_llm, corroborate_llm)
        if line.resolved and line.fact.billable:
            # Data-driven bundling filter: a resolved code the source declares
            # NOT separately reportable (bundled / non-covered / MUE 0) is kept
            # for the audit trail but dropped from the claim. Agnostic.
            if source.separately_billable(
                    line.chosen.code, line.chosen.system, date_of_service) is Outcome.BLOCKED:
                line.excluded_reason = "not separately reportable per authoritative data"
            elif line.chosen.system in ("cpt", "hcpcs"):
                # Laterality/bilateral modifiers and billing UNITS belong to
                # procedure/supply codes only. An ICD-10 DIAGNOSIS encodes laterality
                # IN the code (right vs left vs unspecified) and never takes an RT/LT
                # modifier or a unit count — so this whole block is skipped for it.
                # Data-driven per-line modifiers (laterality) + billing units
                # (descriptor-driven, so a "2-4 items" code bills as one unit).
                line.modifiers = modifier_engine.assign(
                    line.fact, line.chosen.descriptor,
                    bilat=source.bilat_indicator(line.chosen.code))
                cnt = line.fact.attributes.get("count") or line.fact.attributes.get("quantity") or 1
                try:
                    cnt = int(cnt)
                except (TypeError, ValueError):
                    cnt = 1
                line.units = ontology.billing_units(cnt, line.chosen.descriptor)
                # A dosed drug bills by dose, not count: documented total dose /
                # the code's authoritative per-unit dose (e.g. 30 mg / 'per 15 mg'
                # = 2 units). Falls back to the count-based units above when the
                # dose or per-unit is unavailable.
                if line.fact.kind is FactKind.DRUG:
                    documented = " ".join(
                        [str(v) for v in line.fact.attributes.values()]
                        + [s.text for s in line.fact.evidence] + [line.fact.description])
                    du = ontology.drug_billing_units(
                        documented, source.drug_unit(line.chosen.code))
                    if du is not None:
                        line.units = du
        lines.append(line)

    result = CodingResult(
        encounter_id=encounter_id,
        date_of_service=date_of_service,
        lines=lines,
    )
    # Mechanic 4 — collapse duplicate resolved codes into one line before anything
    # downstream reasons about the claim as a set.
    dedup_lines(result)
    # Mechanic 1 — code-type/section applicability (e.g. an anesthesia-section code
    # is not separately reportable by the operating provider).
    apply_section_applicability(result)
    # Claim-level modifiers (E/M-25, distinct-service 59/X) once all lines exist —
    # this records which PTP pairs a justified modifier bypasses.
    modifier_engine.assign_claim(result, source)
    # Mechanic 3 — resolve NCCI PTP conflicts by DEMOTING the bundled component
    # (not blocking the claim) whenever no distinct-service modifier is justified.
    apply_ncci_bundling(result, source)
    # An ancillary procedure that ESCALATED but is an NCCI 'always-bundled' component
    # of a billed primary is INTEGRAL — decide it (bundle), don't send it to review.
    apply_integral_bundling(result, source)

    apply_global_package(result, source)
    result.gates = gates.run_gates(result, note_text, source)
    decide(result, source=source)
    # Actionable documentation guidance for whatever could not be coded confidently.
    from . import recommendations as _recs
    result.recommendations = _recs.build_recommendations(result)
    # Make each routed item self-contained: attach its provider-facing suggested
    # solution to the routing entry (joined by stable fact_id) so a PROVIDER_QUERY
    # carries the exact question to send — no fragile description-based join needed.
    _attach_recommendations(result)
    try:
        fingerprint = source.data_fingerprint()
    except Exception:
        fingerprint = {}
    result.certificate = certificate.build_certificate(
        result, note_text,
        source_identity={"source": type(source).__name__, "data": fingerprint})
    return result


def _attach_recommendations(result: CodingResult) -> None:
    """Attach each routed item's provider-facing suggested solution to its routing
    entry, so a PROVIDER_QUERY (or any routed item) is self-contained. Joins on the
    STABLE fact_id — a fact's free-text description is not unique — and falls back to
    subject only when no fact_id is present."""
    by_id = {r["fact_id"]: r for r in result.recommendations if r.get("fact_id")}
    by_subject = {}
    for r in result.recommendations:
        by_subject.setdefault(r.get("subject"), r)
    for item in result.routing:
        rec = by_id.get(item.get("fact_id")) or by_subject.get(item.get("subject"))
        if rec:
            item["recommendation"] = rec["recommendation"]


def dedup_lines(result: CodingResult) -> None:
    """Mechanic 4 — two documented phrases that resolve to the SAME code are one
    billable line, not two. Keep the first occurrence (union its evidence) and
    exclude the rest from the claim, keeping them in the audit trail. Agnostic: a
    set-merge on the resolved (code, system), never a named code. Genuinely
    repeated services are expressed through units/modifiers, not a second
    identical line, so collapsing here prevents accidental double-billing while
    the MUE gate still governs unit counts."""
    seen: dict[tuple[str, str], ResolvedLine] = {}
    for ln in result.lines:
        if not (ln.resolved and ln.fact.billable and not ln.excluded_reason):
            continue
        key = (ln.chosen.code, ln.chosen.system)
        keep = seen.get(key)
        if keep is None:
            seen[key] = ln
            continue
        # A duplicate that carries NEW evidence is a separately-documented instance,
        # not a re-mention: accumulate its units so a repeated service is not
        # silently dropped (underbilled). The MUE gate caps the total — an
        # accumulation past the unit limit BLOCKS release, so this can never silently
        # overbill either. A duplicate whose evidence is already present is a
        # re-mention of the same event and is simply merged.
        keep_texts = {s.text for s in keep.fact.evidence}
        new_spans = [s for s in ln.fact.evidence if s.text not in keep_texts]
        keep.fact.evidence = list(keep.fact.evidence) + list(ln.fact.evidence)
        if new_spans:
            keep.units += ln.units
            keep.rationale = (f"{keep.rationale}; merged a separately-documented "
                              f"repeat — units accumulated (MUE-capped)")
            ln.excluded_reason = (f"repeat of {ln.chosen.code} — units folded into "
                                  f"the primary line (MUE governs the total)")
        else:
            ln.excluded_reason = (f"duplicate of {ln.chosen.code} already on the "
                                  f"claim — merged into a single line")


def apply_section_applicability(result: CodingResult) -> None:
    """Mechanic 1 — a code whose authoritative descriptor identifies it as a
    different CPT SECTION than the encounter supports is not separately reportable.
    The concrete, agnostic rule: an ANESTHESIA-section service (detected from
    descriptor grammar, not a code range) is billed by the anesthesia provider,
    so on a claim that also carries an operative procedure it is bundled into the
    surgeon's service unless the note documents a separate anesthesia provider.
    Fail-closed: excluded by default, kept in the audit trail."""
    from .ontology import code_section
    from .models import FactKind
    proc_lines = [ln for ln in result.billable_lines
                  if ln.chosen.system in ("cpt", "hcpcs")
                  and ln.fact.kind in (FactKind.PROCEDURE, FactKind.IMAGING)]
    has_operative = any(code_section(ln.chosen.descriptor) != "anesthesia"
                        for ln in proc_lines)
    if not has_operative:
        return                          # e.g. an anesthesia provider's own claim
    from .models import ResolutionMethod
    _SEP = ("anesthesia_provider", "separate_anesthesia_provider",
            "anesthesia_by_separate_provider", "separate_anesthesia")
    _reason = ("anesthesia-section service — not separately reportable by the "
               "operating provider (no separate anesthesia provider documented)")
    for ln in proc_lines:                # a RESOLVED anesthesia-section code
        if code_section(ln.chosen.descriptor) != "anesthesia":
            continue
        if any(ln.fact.attributes.get(k) for k in _SEP):
            continue                    # a separate anesthesia provider is documented
        ln.excluded_reason = _reason
    # An ESCALATED procedure that is itself an ANESTHESIA service is decided
    # DETERMINISTICALLY (exclude), rather than leaving its handling to depend on
    # which specific code the LLM happened to resolve. Signal: the BEST-ranked
    # candidate is the anesthesia section AND that section DOMINATES the candidate
    # set. The 'best + dominant' test matters because an anesthesia-section
    # descriptor ('Anesthesia for procedures on <region>') is a semantic neighbour
    # of any procedure in that region and will appear incidentally among a surgical
    # line's candidates — so mere presence is not enough; it must be the leading match.
    for ln in result.lines:
        if ln.resolved or ln.excluded_reason or not ln.fact.billable:
            continue
        if ln.fact.kind not in (FactKind.PROCEDURE, FactKind.IMAGING):
            continue
        if any(ln.fact.attributes.get(k) for k in _SEP):
            continue
        alts = ln.alternatives
        if not alts or code_section(alts[0].descriptor) != "anesthesia":
            continue                    # the leading match is not anesthesia
        n_anes = sum(1 for c in alts if code_section(c.descriptor) == "anesthesia")
        if n_anes * 2 >= len(alts):      # anesthesia dominates the candidate set
            ln.chosen = alts[0]
            ln.method = ResolutionMethod.DETERMINISTIC
            ln.excluded_reason = _reason


def apply_ncci_bundling(result: CodingResult, source: CodeSource) -> None:
    """Mechanic 3 — turn NCCI PTP edits into a resolution, not a hard block. For
    each pair of billable procedure lines with a PTP edit, if no distinct-service
    modifier is justified (the pair was not bypassed), DEMOTE the column-2
    component code (the authoritative row tells us which side is the bundled
    component) — the claim keeps the payable comprehensive code and drops the
    component, exactly as a coder would, instead of blocking outright. Also honors
    the CPT '(separate procedure)' designation, which bundles a service performed
    alongside another procedure of the same session. All directionality comes from
    the data; no code is named here."""
    from .ontology import is_separate_procedure
    from .models import FactKind
    proc = [ln for ln in result.billable_lines
            if ln.chosen and ln.chosen.system in ("cpt", "hcpcs")]

    # (a) '(separate procedure)' designation — bundled ONLY when billed alongside
    # another actual PROCEDURE this session (a more comprehensive surgical service).
    # A supply/drug/device line (e.g. an implant HCPCS) is NOT a procedure and must
    # not trigger the bundle — otherwise a legitimately separate procedure is dropped
    # just because an implant was also reported.
    def _is_procedure(o) -> bool:
        return o.fact.kind in (FactKind.PROCEDURE, FactKind.IMAGING)

    for ln in proc:
        if ln.excluded_reason or not is_separate_procedure(ln.chosen.descriptor):
            continue
        if any(o is not ln and not o.excluded_reason and _is_procedure(o)
               and o.chosen.code != ln.chosen.code for o in proc):
            ln.excluded_reason = ("'(separate procedure)' designation — bundled "
                                  "when performed with another procedure this session")

    # (b) PTP edits — demote the component of any unbypassed pair.
    by_code = {ln.chosen.code: ln for ln in proc}
    for i in range(len(proc)):
        for j in range(i + 1, len(proc)):
            a, b = proc[i], proc[j]
            if a.excluded_reason or b.excluded_reason:
                continue
            edit = source.ncci_edit(a.chosen.code, b.chosen.code, result.date_of_service)
            if not edit:
                continue
            mod = edit.get("modifier")
            if mod not in ("0", "1"):
                continue                # deleted / non-applicable indicator -> no active edit
            pair = frozenset((a.chosen.code, b.chosen.code))
            if mod == "1" and pair in result.bypassed_ncci:
                continue                # a justified distinct-service modifier keeps both
            comp = by_code.get(edit.get("component"))
            if comp is not None and not comp.excluded_reason:
                comp.excluded_reason = (
                    f"bundled into {edit.get('payable')} per NCCI PTP "
                    f"(no distinct-service modifier justified)")
                result.ncci_suppressed.append((comp.chosen.code, edit.get("payable")))


def apply_integral_bundling(result: CodingResult, source: CodeSource) -> None:
    """Decide the 'integral vs separately billable' gray area authoritatively, so an
    ancillary procedure the resolver could not confidently code is not sent to a
    human when NCCI already answers it. For each ESCALATED (unresolved) procedure
    line, if one of its best candidates is an NCCI 'always-bundled' component
    (modifier indicator 0 — never separately reportable) of a code that IS billed on
    this claim, the ancillary is INTEGRAL to that primary: record it as bundled
    (excluded), not a review item.

    Safe by construction: it ONLY converts an escalation into a NON-billed exclusion
    — it never bills an uncertain code. An indicator-1 (bypassable-with-modifier)
    pair is a genuine judgement (bill-with-modifier vs bundle) and is left escalated.
    Authoritative (NCCI) and agnostic — no code is named here."""
    from .models import FactKind, ResolutionMethod
    billed = {ln.chosen.code for ln in result.billable_lines
              if ln.chosen and ln.chosen.system in ("cpt", "hcpcs")
              and ln.fact.kind in (FactKind.PROCEDURE, FactKind.IMAGING)}
    if not billed:
        return
    for ln in result.lines:
        if ln.resolved or ln.excluded_reason or not ln.fact.billable:
            continue
        if ln.fact.kind not in (FactKind.PROCEDURE, FactKind.IMAGING):
            continue
        done = False
        for cand in ln.alternatives[:4]:
            for primary in billed:
                edit = source.ncci_edit(primary, cand.code, result.date_of_service)
                if (edit and str(edit.get("component")) == cand.code
                        and str(edit.get("payable")) in billed
                        and str(edit.get("modifier")) == "0"):
                    ln.chosen = cand
                    ln.method = ResolutionMethod.DETERMINISTIC
                    ln.excluded_reason = (f"integral to {edit.get('payable')} — always "
                                          f"bundled per NCCI (not separately reportable)")
                    done = True
                    break
            if done:
                break


def apply_global_package(result: CodingResult, source: CodeSource) -> None:
    """Global surgical package (CMS global-period data): an E/M related to a
    same-day procedure that carries a global period (000/010/090) is included in
    that procedure's payment. The E/M is separately billable ONLY if the note
    documents significant, separately identifiable work; otherwise it is bundled
    — dropped from the claim, kept in the audit trail. Fail-closed."""
    from .models import FactKind
    has_global_proc = any(
        source.global_period(ln.chosen.code) in ("000", "010", "090")
        for ln in result.billable_lines
        if ln.fact.kind is not FactKind.EM and ln.chosen.system in ("cpt", "hcpcs"))
    if not has_global_proc:
        return
    for ln in result.lines:
        if (ln.resolved and ln.fact.kind is FactKind.EM and not ln.excluded_reason
                and not ln.fact.attributes.get("separately_identifiable")):
            ln.excluded_reason = ("bundled into the global surgical package "
                                  "(no separately-identifiable E/M documented)")


def render(result: CodingResult) -> str:
    """Human-readable audit trail — the explainability surface."""
    dest = f"  →  {result.destination.value}" if result.destination else ""
    out = [f"Encounter {result.encounter_id}  DOS={result.date_of_service}",
           f"VERDICT: {result.verdict.value}{dest}", ""]
    out.append("LINES:")
    for ln in result.lines:
        f = ln.fact
        if ln.resolved and ln.excluded_reason:
            out.append(f"  ∅ excluded {ln.chosen.system.upper()} {ln.chosen.code}  "
                       f"«{f.description}» — {ln.excluded_reason}")
        elif ln.resolved:
            mods = f"  +{','.join(ln.modifiers)}" if ln.modifiers else ""
            out.append(f"  ✓ {ln.chosen.system.upper()} {ln.chosen.code}{mods}  "
                       f"[{ln.method.value}]  «{f.description}»")
            out.append(f"      descriptor: {ln.chosen.descriptor[:70]}")
            out.append(f"      why: {ln.rationale}")
        else:
            tag = "not billed" if not f.billable else "ESCALATE"
            out.append(f"  ⚠ {tag}  «{f.description}»  — {ln.rationale}")
            cand = list(dict.fromkeys(f"{c.system.upper()} {c.code}"
                                      for c in ln.alternatives if c.code))
            if f.billable and cand:
                out.append(f"      candidates (unconfirmed): {', '.join(cand[:5])}")
        if f.evidence:
            out.append(f"      evidence: «{f.evidence[0].text[:70]}»")
    out.append("")
    out.append("GATES:")
    for g in result.gates:
        out.append(f"  [{g.outcome.value:>14}] {g.name}: {g.detail}  ({g.authority})")
    if result.notes:
        out.append("")
        out.append("DECISION:")
        for n in result.notes:
            out.append(f"  - {n}")
    if result.recommendations:
        out.append("")
        out.append("DOCUMENTATION RECOMMENDATIONS:")
        for r in result.recommendations:
            out.append(f"  • [{r['issue']}] {r['recommendation']}")
    return "\n".join(out)
