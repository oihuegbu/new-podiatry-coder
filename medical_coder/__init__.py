"""Domain-agnostic medical coding kernel."""

from .models import ClaimContext, EvidenceFact, EvidenceGraph, SourceSpan

__all__ = ["ClaimContext", "EvidenceFact", "EvidenceGraph", "SourceSpan"]
__version__ = "0.1.0"

