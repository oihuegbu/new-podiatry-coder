from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .models import EvidenceFact


class UnsupportedPredicate(RuntimeError):
    pass


def resolve_path(fact: EvidenceFact, path: str) -> Any:
    if not path.startswith("fact."):
        raise UnsupportedPredicate("requirement path must start with fact")
    current: Any = asdict(fact)
    for part in path.split(".")[1:]:
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def evaluate_requirement(fact: EvidenceFact, requirement: dict[str, Any]) -> bool:
    actual = resolve_path(fact, str(requirement["path"]))
    operator = requirement["operator"]
    expected = requirement.get("value")
    if operator == "equals":
        return actual == expected
    if operator == "in":
        return actual in expected
    if operator == "present":
        return actual not in (None, "", [], {})
    if operator == "absent":
        return actual in (None, "", [], {})
    if operator == "at_least":
        return actual is not None and actual >= expected
    if operator == "at_most":
        return actual is not None and actual <= expected
    if operator == "contains":
        return actual is not None and expected in actual
    raise UnsupportedPredicate(f"unsupported requirement operator: {operator}")


def entails(facts: list[EvidenceFact], requirements: list[dict[str, Any]]) -> bool:
    return all(any(evaluate_requirement(fact, requirement) for fact in facts) for requirement in requirements)

