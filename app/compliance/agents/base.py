"""Base class for all compliance agents.

An agent encapsulates ONE of the 12 filters. It is pure logic: it reads rules
from the ComplianceDataStore and inspects the canonical Claim — it must never
contain hardcoded code lists. Each agent returns a list of Findings.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from app.compliance.datastore.store import ComplianceDataStore
from app.compliance.models import Claim, Finding


class ComplianceAgent(ABC):
    #: stable id used in findings + the review queue
    filter_id: str = "BASE"
    filter_name: str = "Base"

    def __init__(self, store: ComplianceDataStore):
        self.store = store

    @abstractmethod
    def check(self, claim: Claim) -> list[Finding]:
        """Inspect the claim and return findings (PASS/WARN/FAIL)."""
        raise NotImplementedError

    # convenience builder so agents stay terse
    def finding(self, **kwargs) -> Finding:
        kwargs.setdefault("filter_id", self.filter_id)
        kwargs.setdefault("filter_name", self.filter_name)
        return Finding(**kwargs)
