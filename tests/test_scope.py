from __future__ import annotations

import unittest
from datetime import date
from pathlib import Path

from medical_coder.models import ClaimContext
from medical_coder.scope import AutonomyScope


class ScopeTestCase(unittest.TestCase):
    def context(self, **changes: str) -> ClaimContext:
        values = {
            "date_of_service": date(2026, 1, 1),
            "payer_identifier": "payer",
            "payer_type": "type",
            "claim_type": "professional",
            "billing_entity_role": "billing",
            "performing_entity_role": "performing",
            "place_of_service": "place",
            "jurisdiction": "jurisdiction",
            "contract_profile": "contract",
            "authorization_status": "not-required",
            "organizational_profile": "medium-private-practice",
            "facility_type": "office",
            "practice_profile": "surgical",
        }
        values.update(changes)
        return ClaimContext(**values)

    def test_private_surgical_scope_accepts_declared_profile(self) -> None:
        scope = AutonomyScope(Path("source-packs/scopes/medium-private-surgical-practice.json"))
        self.assertTrue(scope.evaluate(self.context()).eligible)

    def test_facility_claim_is_outside_professional_scope(self) -> None:
        scope = AutonomyScope(Path("source-packs/scopes/medium-private-surgical-practice.json"))
        result = scope.evaluate(self.context(claim_type="institutional"))
        self.assertFalse(result.eligible)
        self.assertIn("separate autonomy scope", result.reason)


if __name__ == "__main__":
    unittest.main()

