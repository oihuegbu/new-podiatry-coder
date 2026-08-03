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

from . import arbitration, certificate, extraction, gates, resolution
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
) -> CodingResult:
    source = source or AuthoritativeSource()

    facts = extraction.extract_facts(note_text, extract_llm)

    lines = []
    for fact in facts:
        line = resolution.resolve(fact, source)
        if (not line.resolved) and line.alternatives and fact.billable:
            line = arbitration.arbitrate(line, arbitrate_llm)
        lines.append(line)

    result = CodingResult(
        encounter_id=encounter_id,
        date_of_service=date_of_service,
        lines=lines,
    )
    result.gates = gates.run_gates(result, note_text, source)
    decide(result)
    result.certificate = certificate.build_certificate(
        result, note_text, source_identity={"source": type(source).__name__})
    return result


def render(result: CodingResult) -> str:
    """Human-readable audit trail — the explainability surface."""
    out = [f"Encounter {result.encounter_id}  DOS={result.date_of_service}",
           f"VERDICT: {result.verdict.value}", ""]
    out.append("LINES:")
    for ln in result.lines:
        f = ln.fact
        if ln.resolved:
            out.append(f"  ✓ {ln.chosen.system.upper()} {ln.chosen.code}  "
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
