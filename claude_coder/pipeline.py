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
    modifier_engine: "ModifierEngine | None" = None,
) -> CodingResult:
    from .models import Outcome
    from .modifiers import ModifierEngine
    source = source or AuthoritativeSource()
    modifier_engine = modifier_engine or ModifierEngine()

    # Propose-then-verify is enabled in real mode (no stubbed LLMs). It grounds every
    # procedure code in an authoritative descriptor the documentation entails — the
    # license-clean substitute for the CPT Index. Tests pass stub LLMs and leave
    # verify_llm None, so they keep the deterministic path unchanged.
    if verify_llm is None and arbitrate_llm is None:
        from .verify import default_verify_llm
        verify_llm = default_verify_llm

    from .models import FactKind
    facts = extraction.extract_facts(note_text, extract_llm)

    lines = []
    for fact in facts:
        if fact.kind is FactKind.EM:
            line = em.resolve_em(fact, source)      # MDM-driven leveling
        else:
            line = resolution.resolve(fact, source, llm=verify_llm)
        # A procedure that went through propose-then-verify is already resolved-or-
        # escalated on authoritative entailment; don't second-guess it with the
        # weaker arbitration fallback. Other kinds still arbitrate residual ambiguity.
        went_through_pv = (verify_llm is not None
                           and fact.kind in (FactKind.PROCEDURE, FactKind.IMAGING))
        if (not line.resolved) and line.alternatives and fact.billable and not went_through_pv:
            line = arbitration.arbitrate(line, arbitrate_llm)
        if line.resolved and line.fact.billable:
            # Data-driven bundling filter: a resolved code the source declares
            # NOT separately reportable (bundled / non-covered / MUE 0) is kept
            # for the audit trail but dropped from the claim. Agnostic.
            if source.separately_billable(
                    line.chosen.code, line.chosen.system, date_of_service) is Outcome.BLOCKED:
                line.excluded_reason = "not separately reportable per authoritative data"
            else:
                # Data-driven per-line modifiers (laterality) + billing units
                # (descriptor-driven, so a "2-4 lesions" code bills as one unit).
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

    apply_global_package(result, source)
    result.gates = gates.run_gates(result, note_text, source)
    decide(result)
    result.certificate = certificate.build_certificate(
        result, note_text, source_identity={"source": type(source).__name__})
    return result


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
        # merge evidence for the audit trail, then drop the duplicate line
        keep.fact.evidence = list(keep.fact.evidence) + list(ln.fact.evidence)
        ln.excluded_reason = (f"duplicate of {ln.chosen.code} already on the claim "
                              f"— merged into a single line")


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
    _SEP = ("anesthesia_provider", "separate_anesthesia_provider",
            "anesthesia_by_separate_provider", "separate_anesthesia")
    for ln in proc_lines:
        if code_section(ln.chosen.descriptor) != "anesthesia":
            continue
        if any(ln.fact.attributes.get(k) for k in _SEP):
            continue                    # a separate anesthesia provider is documented
        ln.excluded_reason = ("anesthesia-section service — not separately "
                              "reportable by the operating provider "
                              "(no separate anesthesia provider documented)")


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
    proc = [ln for ln in result.billable_lines
            if ln.chosen and ln.chosen.system in ("cpt", "hcpcs")]

    # (a) '(separate procedure)' designation — bundled when billed with another
    # distinct procedure line this session.
    for ln in proc:
        if ln.excluded_reason or not is_separate_procedure(ln.chosen.descriptor):
            continue
        if any(o is not ln and not o.excluded_reason
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
    out = [f"Encounter {result.encounter_id}  DOS={result.date_of_service}",
           f"VERDICT: {result.verdict.value}", ""]
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
    return "\n".join(out)
