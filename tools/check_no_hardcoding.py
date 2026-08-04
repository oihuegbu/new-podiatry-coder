from __future__ import annotations

import ast
import re
import sys
from pathlib import Path


CODE_SHAPED = re.compile(r"^(?:[A-Z][0-9A-Z]{3,6}|[0-9]{5})$")
ALLOWED_NAMES = {
    "SUPPORTED_REPORTABLE",
    "SUPPORTED_DIFFERENT_CLAIM",
    "SUPPORTED_PACKAGED",
    "PERFORMED_INTEGRAL",
    "DOCUMENTED_NOT_PERFORMED",
    "HISTORICAL",
    "ORDERED_ONLY",
    "CANDIDATE_NOT_ENTAILED",
    "CANDIDATE_INACTIVE",
    "MORE_SPECIFIC_CANDIDATE_AVAILABLE",
    "SOURCE_CONFLICT",
    "POLICY_REVIEW",
    "PROVIDER_QUERY",
    "CLAIM_CONTEXT_REQUIRED",
    "SOURCE_UNAVAILABLE",
    "COMPILER_ERROR",
}
NON_MEDICAL_PROTOCOL_LITERALS = {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"}


def violations(root: Path) -> list[str]:
    found = []
    for path in sorted((root / "medical_coder").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                value = node.value.strip()
                if value not in ALLOWED_NAMES and value not in NON_MEDICAL_PROTOCOL_LITERALS and CODE_SHAPED.fullmatch(value):
                    found.append(f"{path}:{node.lineno}: code-shaped literal {value!r}")
    return found


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    found = violations(root)
    if found:
        print("\n".join(found))
        return 1
    print("PASS: kernel contains no code-shaped medical identifier literals")
    return 0


if __name__ == "__main__":
    sys.exit(main())

