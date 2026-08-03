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
from .models import CodingResult, ResolutionMethod


def code_encounter(
    encounter_id: str,
    note_text: str,
    date_of_service: str | None,
    source: CodeSource | None = None,
    extract_llm: LLMFn | None = None,
    arbitrate_llm: LLMFn | None = None,
    modifier_engine: "ModifierEngine | None" = None,
) -> CodingResult:
    from .models import Outcome
    from .modifiers import ModifierEngine
    source = source or AuthoritativeSource()
    modifier_engine = modifier_engine or ModifierEngine()

    from .models import FactKind
    facts = extraction.extract_facts(note_text, extract_llm)

    lines = []
    for fact in facts:
        if fact.kind is FactKind.EM:
            line = em.resolve_em(fact, source)      # MDM-driven leveling
        else:
            line = resolution.resolve(fact, source)
        if (not line.resolved) and line.alternatives and fact.billable:
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
        lines.append(line)

    result = CodingResult(
        encounter_id=encounter_id,
        date_of_service=date_of_service,
        lines=lines,
    )
    # Claim-level modifiers (E/M-25, distinct-service 59/X) once all lines exist.
    modifier_engine.assign_claim(result, source)

    apply_global_package(result, source)
    result.gates = gates.run_gates(result, note_text, source)
    decide(result)
    result.certificate = certificate.build_certificate(
        result, note_text, source_identity={"source": type(source).__name__})
    return result


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
