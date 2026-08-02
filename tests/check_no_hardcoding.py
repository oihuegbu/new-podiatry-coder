"""Guard against hardcoded medical code tables anywhere in the codebase.

CPT/ICD-10-CM/HCPCS knowledge must come from the data store — never from code
literals, whether in production logic, tools, prompts, or test fixtures. A
hardcoded code list silently goes stale the moment the authoritative source
changes (quarterly NCCI/MUE, annual CPT/HCPCS), so it is banned everywhere.

Catches the two shapes that leak, by DENSITY (several codes in one place), not
by counting scattered references:
  1. a COLLECTION literal holding several code strings — e.g.
     `code in {"11055", "11056", ...}`, a mapping dict, or a benchmark table;
  2. a STRING / PROMPT that embeds a cluster of codes — e.g. a coding prompt
     that maps codes to meanings.
A single code named in a comment, one assertion, or one structural constant is
fine and is NOT flagged — only a cluster (>= THRESHOLD distinct codes) inside a
single collection or a single string.

RATCHET: the codebase already carries pre-existing clusters. Those are recorded
in `hardcoding_baseline.txt` (a visible debt ledger to burn down) and do NOT
fail the build. Any NEW cluster — a signature not in the baseline — fails. This
prevents re-introducing the pattern (the reason the guard exists) without
requiring a full legacy refactor first.

  python tests/check_no_hardcoding.py                    # CI: fail on NEW only
  python tests/check_no_hardcoding.py --update-baseline  # accept current set
  python tests/check_no_hardcoding.py --all              # show every cluster

Escape hatch for a genuinely legitimate cluster: put `# codes-ok: <reason>` on
the line where the collection/string starts.
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCAN_DIRS = ("app", "tools", "tests")
BASELINE = Path(__file__).resolve().parent / "hardcoding_baseline.txt"
THRESHOLD = 4          # distinct codes in ONE collection/string to count as a table
ALLOW_MARK = "codes-ok"

# CPT (5 digits) | HCPCS/ICD-10-CM (letter + 2 digits, optional dotted/undotted
# detail and trailing char). Matches "11055", "A5500", "E11.621", "M7731".
_CODE = re.compile(r"(?:\d{5}|[A-Z]\d{2}(?:\.?\d{1,4})?[A-Z]?)")
_CODE_FULL = re.compile(rf"^{_CODE.pattern}$")
_CODE_TOKEN = re.compile(rf"\b{_CODE.pattern}\b")


def _is_code(s: str) -> bool:
    s = s.strip()
    # "00001"-style note/record IDs are not billable codes (CPT never starts 0000)
    return bool(_CODE_FULL.match(s)) and not s.startswith("0000")


def _flatten(elts):
    for e in elts:
        if isinstance(e, (ast.Tuple, ast.List, ast.Set)):
            yield from _flatten(e.elts)
        else:
            yield e


class _Guard(ast.NodeVisitor):
    def __init__(self, allow_lines: set[int]):
        self.allow = allow_lines
        self.viol: list[tuple[int, str, frozenset[str]]] = []

    def _record(self, lineno: int, kind: str, codes: set[str]) -> None:
        if len(codes) >= THRESHOLD and lineno not in self.allow:
            self.viol.append((lineno, kind, frozenset(codes)))

    def _collection(self, node) -> None:
        codes = {e.value for e in _flatten(getattr(node, "elts", []))
                 if isinstance(e, ast.Constant) and isinstance(e.value, str)
                 and _is_code(e.value)}
        self._record(node.lineno, "code collection", codes)

    def visit_List(self, node):
        self._collection(node)
        self.generic_visit(node)

    def visit_Set(self, node):
        self._collection(node)
        self.generic_visit(node)

    def visit_Tuple(self, node):
        self._collection(node)
        self.generic_visit(node)

    def visit_Dict(self, node):
        codes = {k.value for k in list(node.keys) + list(node.values)
                 if isinstance(k, ast.Constant) and isinstance(k.value, str)
                 and _is_code(k.value)}
        self._record(node.lineno, "code mapping", codes)
        self.generic_visit(node)

    def visit_Constant(self, node):
        if isinstance(node.value, str) and len(node.value) > 40:
            self._record(node.lineno, "prompt/string",
                         {t for t in _CODE_TOKEN.findall(node.value) if _is_code(t)})

    def visit_JoinedStr(self, node):
        text = "".join(v.value for v in ast.walk(node)
                       if isinstance(v, ast.Constant) and isinstance(v.value, str))
        self._record(node.lineno, "prompt/f-string",
                     {t for t in _CODE_TOKEN.findall(text) if _is_code(t)})
        self.generic_visit(node)


def _allow_lines(text: str) -> set[int]:
    return {i for i, ln in enumerate(text.splitlines(), 1) if ALLOW_MARK in ln}


def _signature(relpath: str, kind: str, codes: frozenset[str]) -> str:
    """Line-independent identity: file + kind + the exact code set. Moving a
    cluster keeps its signature; changing which codes it hardcodes makes it new."""
    return f"{relpath}\t{kind}\t{','.join(sorted(codes))}"


def _scan() -> list[tuple[str, str, frozenset[str], int]]:
    out = []
    files = []
    for d in SCAN_DIRS:
        files.extend(sorted((ROOT / d).rglob("*.py")))
    for py in files:
        if py.name == Path(__file__).name:
            continue
        text = py.read_text()
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        g = _Guard(_allow_lines(text))
        g.visit(tree)
        rel = str(py.relative_to(ROOT))
        for lineno, kind, codes in g.viol:
            out.append((rel, kind, codes, lineno))
    return out


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    clusters = _scan()

    if "--update-baseline" in argv:
        sigs = sorted({_signature(r, k, c) for r, k, c, _ in clusters})
        BASELINE.write_text("\n".join(sigs) + "\n")
        print(f"✅ baseline updated: {len(sigs)} accepted cluster(s) -> "
              f"{BASELINE.name}")
        return 0

    baseline = set()
    if BASELINE.exists():
        baseline = {ln for ln in BASELINE.read_text().splitlines() if ln.strip()}

    show_all = "--all" in argv
    new = [(r, k, c, ln) for r, k, c, ln in clusters
           if show_all or _signature(r, k, c) not in baseline]

    if new:
        header = ("every hardcoded cluster" if show_all
                  else "NEW hardcoded code cluster(s) — not in baseline")
        print(f"❌ HARDCODING GUARD: {header}. Codes belong in the data store "
              f"(or mark `# {ALLOW_MARK}: reason`, or --update-baseline if "
              f"intentional):")
        for r, k, c, ln in sorted(new):
            print(f"   - {r}:{ln}  {k} of {len(c)} codes e.g. {sorted(c)[:6]}")
        return 1

    print(f"✅ No NEW hardcoded code tables in {', '.join(SCAN_DIRS)}/ "
          f"({len(baseline)} pre-existing cluster(s) grandfathered in "
          f"{BASELINE.name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
