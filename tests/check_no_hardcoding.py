"""Fail CI when production coding logic embeds medical-code knowledge.

Python may implement generic mechanics.  CPT, HCPCS, ICD-10-CM, modifier
membership, code-family classification, and applicability must come from the
versioned authoritative datastore.  Tests and data files may name expected
codes; production decision modules may not.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGETS = [
    *sorted((ROOT / "app").rglob("*.py")),
    ROOT / "tools" / "calibration_dataset.py",
    ROOT / "tools" / "audit_results.py",
    ROOT / "tools" / "claim_submitter.py",
]

_MEDICAL_CODE = re.compile(
    r"^(?:\d{5}|[A-V]\d{4}|[A-TV-Z]\d{2}(?:\.[A-Z0-9-]{1,4})?)$"
)
_MODIFIER = re.compile(r"^(?:\d{2}|[A-Z][A-Z0-9])$")
# ISO/IEC 7812 issuer identifier used by the CMS NPI check-digit algorithm;
# it is an administrative checksum constant, not a medical code.
_NON_MEDICAL_STANDARD_LITERALS = {"80840"}


def _docstring_nodes(tree: ast.AST) -> set[ast.Constant]:
    out = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if (isinstance(body, list) and body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            out.add(body[0].value)
    return out


def _literal_strings(node: ast.AST) -> list[str]:
    return [item.value for item in ast.walk(node)
            if isinstance(item, ast.Constant) and isinstance(item.value, str)]


def scan_file(path: Path) -> list[str]:
    source = path.read_text()
    tree = ast.parse(source, filename=str(path))
    docs = _docstring_nodes(tree)
    parents = {child: parent for parent in ast.walk(tree)
               for child in ast.iter_child_nodes(parent)}
    violations = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Constant) and node not in docs
                and isinstance(node.value, str)
                and node.value not in _NON_MEDICAL_STANDARD_LITERALS
                and _MEDICAL_CODE.fullmatch(node.value)):
            violations.append(f"line {node.lineno}: medical code literal {node.value!r}")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id == "range":
            values = [arg.value for arg in node.args
                      if isinstance(arg, ast.Constant) and isinstance(arg.value, int)]
            if any(abs(value) >= 100 for value in values):
                violations.append(
                    f"line {node.lineno}: numeric code-family range {values!r}")
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr in {"startswith", "endswith"} and node.args):
            prefixes = _literal_strings(node.args[0])
            receiver = ast.unparse(node.func.value).casefold()
            if ("code" in receiver and any(
                    prefix and prefix.isalnum() for prefix in prefixes)):
                violations.append(
                    f"line {node.lineno}: code-family string classification "
                    f"{node.func.attr}{prefixes!r}")
        if isinstance(node, (ast.Set, ast.List, ast.Tuple)):
            values = [elt.value for elt in node.elts
                      if isinstance(elt, ast.Constant) and isinstance(elt.value, str)]
            modifiers = [value for value in values if _MODIFIER.fullmatch(value)]
            if len(modifiers) >= 2:
                violations.append(
                    f"line {node.lineno}: hardcoded modifier/code family {modifiers!r}")
            chapter_letters = [value for value in values
                               if len(value) == 1 and value.isalpha()
                               and value.isupper()]
            parent = parents.get(node)
            if (len(chapter_letters) >= 2 and isinstance(parent, ast.Compare)
                    and "code" in ast.unparse(parent).casefold()):
                violations.append(
                    f"line {node.lineno}: hardcoded alphabetic code family "
                    f"{chapter_letters!r}")
    return violations


def main() -> int:
    violations = []
    for path in TARGETS:
        for finding in scan_file(path):
            violations.append(f"{path.relative_to(ROOT)}:{finding}")
    if violations:
        print("❌ HARDCODING GUARD FAILED — resolve from authoritative data:")
        for violation in violations:
            print(f"   - {violation}")
        return 1
    print(f"✅ No medical-code literals, families, or modifier lists in "
          f"{len(TARGETS)} production decision modules")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
