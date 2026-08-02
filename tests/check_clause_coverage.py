"""Architectural guard: ratchet the number of untagged validator-issue
emission sites down to zero, and fail if it ever goes back up.

WHY THIS EXISTS. Advisory suppression is clause-scoped on the compliance-
engine side (`engine._apply_advisory_suppressions` keys `(code, clause)`)
but was category-scoped on the validator side (`(category, code)`), because
`ValidationIssue` carried no clause. A filter can emit several distinct
assertions about the same code — MEDICAL_NECESSITY's class-findings-modifier
advisory versus its claim-composition gate — so blunt matching let a rule
verified against one clause silently retire its siblings. That inversion was
observed live on routine_00003, where a suppression grounded on the
Q-modifier pathway flipped the whole filter to PASS while the claim was
missing the policy's required secondary diagnosis.

The fix adds `clause` to `ValidationIssue` and to `_add()`, and matches
`(category, code, clause)` with a MIGRATION FALLBACK: a suppression
directive that carries NO clause still matches on `(category, code)`, as
today. That makes the change a no-op on day one — and it means the fallback
is load-bearing until every emission site is tagged.

This guard is that migration's end condition, made countable. It walks every
tree that can construct a validator issue and counts `_add(...)` calls with
no `clause=` keyword. The count may only go DOWN. When it reaches zero the
fallback can be deleted and the validator flipped to both-null-exact
matching, matching the engine's semantics exactly (see
`_apply_validator_advisory_suppressions`).

Without a counter, "document the divergence at both sites" decays into two
permanent comments describing behavior that no longer converges.

Run:  python -m tests.check_clause_coverage
Also gated by tests/test_clause_coverage.py so it fails the suite, not just
a command someone remembers to run.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Every tree whose code can reach CodingValidator._add. Order is the
# migration order recommended by review: the auto-template and rule-engine
# sites carry the highest collision risk (one template emits several
# distinct assertions about the same code), so they drain first.
SCAN_TREES: tuple[tuple[str, Path, str], ...] = (
    ("auto-templates (sandboxed, minted)",
     ROOT / "data" / "rules" / "auto_templates", "*.py"),
    ("graduated templates (promoted out of the sandbox)",
     ROOT / "app" / "validation" / "graduated", "*.py"),
    ("rule engine (built-in template mechanics)",
     ROOT / "app" / "validation", "rule_engine.py"),
    ("validator checks",
     ROOT / "app" / "validation", "validator.py"),
)

# The ratchet. Lower this — never raise it — as sites are tagged.
# 0 means the migration is complete: delete the fallback in
# _apply_validator_advisory_suppressions and drop this guard to a
# simple "must stay 0" assertion.
BASELINE_UNTAGGED = 139


def _untagged_sites(src: str, filename: str) -> list[tuple[int, str]]:
    """(lineno, receiver) for every `<recv>._add(...)` call that does not
    pass a `clause=` keyword.

    A `**kwargs` spread counts as UNTAGGED: the guard cannot prove a clause
    is present, and a ratchet that assumes the favorable case would let the
    real count drift up while reading clean.
    """
    try:
        tree = ast.parse(src)
    except SyntaxError as exc:
        raise SystemExit(f"{filename}: syntax error, cannot scan: {exc}")

    out: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if not (isinstance(fn, ast.Attribute) and fn.attr == "_add"):
            continue
        if any(kw.arg == "clause" for kw in node.keywords):
            continue
        recv = fn.value
        name = (recv.id if isinstance(recv, ast.Name)
                else ast.unparse(recv) if hasattr(ast, "unparse")
                else "?")
        out.append((node.lineno, f"{name}._add"))
    return out


def scan() -> tuple[int, list[str]]:
    total = 0
    lines: list[str] = []
    for label, directory, pattern in SCAN_TREES:
        if not directory.exists():
            lines.append(f"   {label}: (tree absent)")
            continue
        subtotal = 0
        files = 0
        for py in sorted(directory.glob(pattern)):
            if py.name == "__init__.py":
                continue
            files += 1
            hits = _untagged_sites(py.read_text(encoding="utf-8"), py.name)
            subtotal += len(hits)
        total += subtotal
        lines.append(f"   {label}: {subtotal} untagged "
                     f"across {files} file(s)")
    return total, lines


def main() -> int:
    total, breakdown = scan()
    print("Clause-tagging coverage of validator-issue emission sites")
    for line in breakdown:
        print(line)
    print(f"   TOTAL untagged: {total} (ratchet: {BASELINE_UNTAGGED})")

    if total > BASELINE_UNTAGGED:
        print(f"\n❌ CLAUSE RATCHET FAILED — untagged emission sites rose "
              f"from {BASELINE_UNTAGGED} to {total}.")
        print("   A new _add(...) call was added without clause=. Every "
              "issue emitted untagged can only be suppressed by an "
              "unscoped directive, which is the blunt (category, code) "
              "matching this migration exists to retire.")
        print("   Fix: pass clause=\"<the assertion this issue makes>\" at "
              "the new site. Do not raise BASELINE_UNTAGGED.")
        return 1

    if total < BASELINE_UNTAGGED:
        print(f"\n❌ CLAUSE RATCHET NOT TIGHTENED — {total} untagged sites "
              f"remain but BASELINE_UNTAGGED is still "
              f"{BASELINE_UNTAGGED}.")
        print(f"   Progress must be locked in or it silently reverses. "
              f"Set BASELINE_UNTAGGED = {total}.")
        if total == 0:
            print("   At zero: also delete the unscoped fallback in "
                  "CodingValidator._apply_validator_advisory_suppressions "
                  "so the validator matches (category, code, clause) "
                  "both-null-exact, like engine._apply_advisory_"
                  "suppressions.")
        return 1

    if total == 0:
        print("\n✅ Every emission site is clause-tagged.")
    else:
        print(f"\n✅ Held at {total} untagged site(s) — no new untagged "
              f"emissions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
