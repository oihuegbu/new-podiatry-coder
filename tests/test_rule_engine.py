"""Declarative rule engine plumbing tests.

Behavior of each migrated rule is covered by tests/test_validator_checks.py
(the delegating _check_* methods are exercised there with real data). This
suite covers the engine machinery itself: pack loading, rule gating
(disabled/unknown rules are silent no-ops), config completeness, and the
no-hardcoding contract for the pack file.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.validation.rule_engine import RULES_FILE, load_rule_pack  # noqa: E402

PASS = FAIL = 0


def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  \u2705 {name}")
    else:
        FAIL += 1
        print(f"  \u274c {name}")


KNOWN_TEMPLATES = {"context_gate", "tiered_family_arbitration",
                   "companion_completion", "residual_secondary_demotion",
                   "icd_tiered_axis"}

# Codes named in message/authority text are allowed to be cited as EXAMPLES
# inside prose; what the guard forbids is codes in OPERATIVE fields (selectors,
# lexicons, targets) — a rule must derive its codes from data, never list them.
OPERATIVE_FIELDS = ("applies_to", "family", "tiers", "carrier",
                    "companion_trigger", "index_link", "contexts",
                    "mention_terms", "residual_markers", "anchor_evidence",
                    "evidence", "claim_surgery_range")
CODE_PATTERN = re.compile(r"(?<![\w^$\\{])[A-Z]\d{2}(?:\.\d+)?(?![\w}])")


def main():
    print("\n[rule pack loading]")
    pack = load_rule_pack()
    rules = pack.get("rules", [])
    check("pack file exists", RULES_FILE.exists())
    check("pack loads with a version", bool(pack.get("version")))
    check("pack has rules", len(rules) >= 4)
    check("load_rule_pack is cached (same object)",
          load_rule_pack() is pack)

    print("\n[rule schema completeness]")
    ids = [r.get("id") for r in rules]
    check("rule ids unique", len(ids) == len(set(ids)))
    for r in rules:
        rid = r.get("id", "?")
        check(f"{rid}: known template",
              r.get("template") in KNOWN_TEMPLATES)
        check(f"{rid}: cites an authority", bool(r.get("authority")))
        act = r.get("action", {})
        check(f"{rid}: action has severity/category/message",
              all(act.get(k) for k in ("severity", "category", "message")))

    print("\n[no operative hardcoded codes]")
    for r in rules:
        rid = r.get("id", "?")
        operative = {k: v for k, v in r.items() if k in OPERATIVE_FIELDS}
        blob = json.dumps(operative)
        # ICD-like tokens (letter + digits) in operative config would be a
        # hardcoded code list; regex character classes are stripped first.
        hits = [m for m in CODE_PATTERN.findall(blob)]
        check(f"{rid}: operative fields contain no literal ICD codes",
              not hits)
        # CPT-like: any bare 5-digit literal outside claim_surgery_range
        blob_no_range = json.dumps(
            {k: v for k, v in operative.items() if k != "claim_surgery_range"})
        cpt_hits = re.findall(r"(?<![\d\\])\d{5}(?!\d)", blob_no_range)
        check(f"{rid}: operative fields contain no literal CPT codes",
              not cpt_hits)

    print("\n[engine gating]")
    from app.validation.rule_engine import RuleEngine

    class _StubValidator:
        db = None
        store = None

    eng = RuleEngine(_StubValidator())
    check("unknown rule id is a silent no-op",
          eng.rule("no-such-rule") is None
          or eng.context_gate("no-such-rule", [], "note") is None)
    # disable a rule in-memory and confirm the gate respects it
    some_id = rules[0]["id"]
    eng.rules[some_id] = dict(eng.rules[some_id], enabled=False)
    check("disabled rule is skipped", eng.rule(some_id) is None)

    print("\n" + "=" * 50)
    print(f"RESULT: {PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
