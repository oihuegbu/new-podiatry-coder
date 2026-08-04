from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .models import ClaimContext


@dataclass(frozen=True)
class ScopeResult:
    eligible: bool
    reason: str
    required_capabilities: frozenset[str]
    scope_id: str
    version: str


class AutonomyScope:
    def __init__(self, path: Path) -> None:
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("schema_version") != 1:
            raise ValueError("unsupported autonomy-scope schema")
        self.value = value

    @staticmethod
    def _evaluate(actual: Any, operator: str, expected: Any) -> bool:
        if operator == "equals":
            return actual == expected
        if operator == "in":
            return actual in expected
        if operator == "present":
            return actual not in (None, "")
        raise ValueError(f"unsupported scope operator: {operator}")

    def evaluate(self, context: ClaimContext) -> ScopeResult:
        values = asdict(context)
        failures = []
        for requirement in self.value["context_requirements"]:
            field = requirement["field"]
            if field not in values:
                raise ValueError(f"scope references unknown claim-context field: {field}")
            if not self._evaluate(values[field], requirement["operator"], requirement.get("value")):
                failures.append(requirement.get("reason") or f"{field} is outside scope")
        return ScopeResult(
            eligible=not failures,
            reason="eligible for configured autonomy scope" if not failures else "; ".join(failures),
            required_capabilities=frozenset(self.value["required_capabilities"]),
            scope_id=self.value["scope_id"],
            version=self.value["version"],
        )

