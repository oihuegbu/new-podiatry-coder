"""claude-medical-coder — facts-first, deterministic, autonomous medical coding.

The LLM understands the note and extracts evidence-linked clinical FACTS; a
deterministic layer maps facts to codes from the authoritative data; the model
is consulted only for residual ambiguity; positive gates decide billability; and
an autonomy controller releases only what it can defend. No medical code is
hardcoded anywhere in this package.
"""
from . import em
from .data_access import AuthoritativeSource, CodeSource, MockSource
from .modifiers import ModifierEngine
from .models import (
    CandidateCode,
    ClinicalFact,
    CodingResult,
    Disposition,
    EvidenceSpan,
    FactKind,
    GateResult,
    Outcome,
    ResolutionMethod,
    ResolvedLine,
    Verdict,
)
from .pipeline import code_encounter, render

__all__ = [
    "code_encounter", "render", "em", "ModifierEngine",
    "AuthoritativeSource", "MockSource", "CodeSource",
    "ClinicalFact", "FactKind", "Disposition", "EvidenceSpan",
    "CandidateCode", "ResolvedLine", "ResolutionMethod",
    "GateResult", "Outcome", "Verdict", "CodingResult",
]
