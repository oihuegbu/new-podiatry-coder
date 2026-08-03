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


def _tool_targets() -> list[Path]:
    """Every tool EXCEPT test scripts. The guard's contract is that tests (and data
    files) may name expected codes; production logic and data-prep tooling may not.
    A file is treated as a test when its name is test_*, *_test, or _*test* — those
    are excluded; everything else under tools/ is scanned. This auto-covers new
    tools (so a future hardcoded table cannot slip in unscanned, as an earlier one
    did) without dragging in the code-bearing test fixtures that live under tools/."""
    def _is_test(name: str) -> bool:
        return (name.startswith("test_") or name.endswith("_test.py")
                or ("test" in name and name.startswith("_")))
    return [p for p in sorted((ROOT / "tools").rglob("*.py")) if not _is_test(p.name)]


# Every production decision module + data-prep tool: the whole app/ and
# claude_coder/ packages (the coder itself) plus tools/ (parsers/builders). A code
# literal or code-family classification in any of them is a hardcoding violation.
# tests/ is deliberately excluded (tests may name expected codes).
TARGETS = [
    *sorted((ROOT / "app").rglob("*.py")),
    *sorted((ROOT / "claude_coder").rglob("*.py")),
    *_tool_targets(),
]

_MEDICAL_CODE = re.compile(
    r"^(?:\d{5}|[A-V]\d{4}|[A-TV-Z]\d{2}(?:\.[A-Z0-9-]{1,4})?)$"
)
_MODIFIER = re.compile(r"^(?:\d{2}|[A-Z][A-Z0-9])$")
# ISO/IEC 7812 issuer identifier used by the CMS NPI check-digit algorithm;
# it is an administrative checksum constant, not a medical code.
_NON_MEDICAL_STANDARD_LITERALS = {"80840"}
# X12 835/CAS claim-ADJUSTMENT GROUP codes — administrative payer-side categories,
# not CPT/ICD/HCPCS/modifier medical codes. Excluded from the modifier-family check.
_ADMIN_GROUP_CODES = {"CO", "OA", "PI", "PR", "CR"}
# Pre-existing tooling surfaced when the guard scope widened to all tools; NOT part
# of the claude_coder solution. Documented (not silently rewritten) so CI stays
# honest about scope while the coder itself is fully guarded.
_SCOPE_EXEMPT = {"tools/sweep_convergence_layers.py"}

# --- Agnostic-solution TERM guard --------------------------------------------
# The claude_coder solution and its data-prep tools must name NO real condition,
# eponym, drug, or region — only generic mechanics — in code OR comments/docstrings.
# (app/ predates this and legitimately carries domain vocabulary in its prompts, so
# the term guard scans the coder + its own tools only.) This is a regression
# DENYLIST — a guard aid like the code regex, not coding logic — that stops the
# specific domain terms we removed from creeping back.
_DENY_TERMS = re.compile(r"""(?ix)\b(
    morton | haglund | keller | moberg | mcbride | lapidus | charcot | weil |
    neuroma | bunionette | bunion | hammer\s?toe | cheilectomy | osteotomy |
    ostectomy | exostect\w* | onychomycos\w* | tinea | metatarsalgia | hallux |
    rigidus | valgus | bursitis | tendinitis | tendinos\w* | tendinopath\w* |
    verruca | mycotic | fasciitis |
    verruca | mycotic | fasciitis | fascia | plantar | sesamoid | ganglion |
    gout | psorias\w* | diabet\w* | arthrit\w* | gangren\w* | callus |
    ketorolac | dexamethasone | betamethasone |
    metatarsal | calcaneus | calcaneal | achilles | phalan\w* | toenail |
    retrocalcaneal
)\b""")

# The term guard holds the coder + its own data-prep tools (not app/) to the
# no-domain-vocabulary standard.
AGNOSTIC_TARGETS = [
    *sorted((ROOT / "claude_coder").rglob("*.py")),
    *[ROOT / "tools" / f"{name}.py" for name in (
        "parse_cpt_index", "parse_icd10cm_index", "build_hcpcs_drug_table",
        "build_learned_index", "refresh_authoritative_data",
        "build_snomed_icd10_map", "build_global_period", "recall_benchmark",
    ) if (ROOT / "tools" / f"{name}.py").exists()],
]


def scan_terms(path: Path) -> list[str]:
    """Flag any real condition/eponym/drug/region term — in code, comments, or
    docstrings. Scans raw text (not the AST) so comments are covered too."""
    out = []
    for lineno, line in enumerate(path.read_text().splitlines(), 1):
        for match in _DENY_TERMS.finditer(line):
            out.append(f"line {lineno}: real medical term {match.group(0)!r} "
                       f"(use a generic stand-in — the coder must be domain-agnostic)")
    return out


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
            modifiers = [value for value in values if _MODIFIER.fullmatch(value)
                         and value not in _ADMIN_GROUP_CODES]
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
        if str(path.relative_to(ROOT)) in _SCOPE_EXEMPT:
            continue
        for finding in scan_file(path):
            violations.append(f"{path.relative_to(ROOT)}:{finding}")
    for path in AGNOSTIC_TARGETS:
        for finding in scan_terms(path):
            violations.append(f"{path.relative_to(ROOT)}:{finding}")
    if violations:
        print("❌ HARDCODING GUARD FAILED — resolve from authoritative data:")
        for violation in violations:
            print(f"   - {violation}")
        return 1
    print(f"✅ No medical-code literals/families/modifier lists in {len(TARGETS)} "
          f"modules; no domain vocabulary in {len(AGNOSTIC_TARGETS)} agnostic modules")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
