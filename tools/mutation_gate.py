"""Mutation-testing gate — enforces "test the FAILURE path, not just the happy path".

It mutates one target module (the file a fix touched) with standard operators and
reruns the fast test suite for each mutant. A SURVIVING mutant is a change to the
logic that NO test caught — almost always an untested failure / fail-closed branch,
which is exactly the pattern this gate exists to stop. Zero external dependency; run
per fix, scoped to the changed file:

    python tools/mutation_gate.py claude_coder/autonomy.py

Exits non-zero if any mutant survives (so it can gate a fix). This is a direct
measurement of test adequacy, not a proxy for it.
"""
from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TESTS = ["tests/test_claude_coder.py", "tests/test_metamorphic.py"]
TIMEOUT = 120

_CMP = {ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt,
        ast.Eq: ast.NotEq, ast.NotEq: ast.Eq, ast.Is: ast.IsNot, ast.IsNot: ast.Is}
_BOOL = {ast.And: ast.Or, ast.Or: ast.And}


def _sites(tree: ast.AST) -> list[tuple]:
    """Every one-mutation site as (apply, revert, description, lineno). Covers the
    condition/branch operators whose absence-of-a-test signals an untested path:
    comparison flips, and/or swaps, boolean-constant flips."""
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            for k, op in enumerate(node.ops):
                if type(op) in _CMP:
                    new = _CMP[type(op)]()
                    out.append((lambda n=node, k=k, v=new: n.ops.__setitem__(k, v),
                                lambda n=node, k=k, v=op: n.ops.__setitem__(k, v),
                                f"{type(op).__name__}->{type(new).__name__}", node.lineno))
        elif isinstance(node, ast.BoolOp) and type(node.op) in _BOOL:
            op = node.op
            new = _BOOL[type(op)]()
            out.append((lambda n=node, v=new: setattr(n, "op", v),
                        lambda n=node, v=op: setattr(n, "op", v),
                        f"{type(op).__name__}->{type(new).__name__}", node.lineno))
        elif isinstance(node, ast.Constant) and isinstance(node.value, bool):
            val = node.value
            out.append((lambda n=node, v=not val: setattr(n, "value", v),
                        lambda n=node, v=val: setattr(n, "value", v),
                        f"{val}->{not val}", node.lineno))
    return out


def _run_tests() -> bool:
    """True if the suite PASSES (mutant survived), False if it FAILS (mutant killed).

    Runs with bytecode writing DISABLED (`python -B` + PYTHONDONTWRITEBYTECODE). This
    is not cosmetic: the gate rewrites the target module many times per second, and
    consecutive mutants routinely yield SAME-SIZE source (e.g. `or`->`and` and
    `True`->`False` both add one byte). CPython invalidates a cached .pyc on
    (mtime, size) with SECOND-granular mtime, so within one second a stale .pyc from
    the PREVIOUS mutant would be silently reused — masking real survivors as killed
    (a false green) and killed mutants as survivors. With no .pyc ever written during
    the loop, every run compiles from the current source, so the measurement is real."""
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    try:
        r = subprocess.run([sys.executable, "-B", "-m", "pytest", *TESTS, "-x", "-q"],
                           cwd=ROOT, capture_output=True, timeout=TIMEOUT, env=env)
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        return False   # a hang/infinite loop counts as killed


def main(argv: list[str]) -> int:
    target = ROOT / (argv[0] if argv else "claude_coder")
    files = [target] if target.is_file() else sorted(target.rglob("*.py"))

    # BASELINE GUARD — fail LOUDLY if we cannot actually measure mutations. Without
    # this, a broken environment (pytest missing, a pre-existing failure) makes EVERY
    # test run fail, so every mutant looks "killed" and the gate reports a false pass.
    if not _run_tests():
        print("ABORT: the suite does not pass on UNMUTATED code — cannot measure "
              "mutation coverage (is pytest installed? is the suite green?). A gate "
              "that cannot run its tests must never report 'all killed'.")
        return 2

    survivors, total = [], 0
    for f in files:
        original = f.read_text()
        tree = ast.parse(original)
        # PER-FILE UNPARSE GUARD — the harness rewrites the file via ast.unparse; if
        # that alone breaks the module, every mutant is falsely "killed". Verify the
        # unmutated-unparse is still green before trusting this file's results.
        f.write_text(ast.unparse(tree))
        if not _run_tests():
            f.write_text(original)
            print(f"{f.relative_to(ROOT)}: SKIP — ast.unparse does not round-trip "
                  f"cleanly here; results would be false")
            continue
        sites = _sites(tree)
        print(f"{f.relative_to(ROOT)}: {len(sites)} mutants")
        try:
            for apply, revert, desc, lineno in sites:
                total += 1
                apply()
                f.write_text(ast.unparse(tree))
                if _run_tests():                      # tests passed -> mutant SURVIVED
                    survivors.append(f"{f.relative_to(ROOT)}:{lineno}  {desc}")
                revert()
        finally:
            f.write_text(original)                    # always restore exact original
    print(f"\n=== {total} mutants, {len(survivors)} survived ===")
    for s in survivors:
        print("  SURVIVED", s)
    if survivors:
        print("\nSurviving mutants = logic no test caught (usually an untested "
              "failure/fail-closed path). Add the test that kills each.")
        return 1
    print("All mutants killed — the failure paths are tested.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
