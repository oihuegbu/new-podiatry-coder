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


# Templates the engine can actually dispatch: the built-ins implemented in
# rule_engine.py's dispatch table, plus GRADUATED (promoted, static) and the
# sandboxed auto-templates (data/rules/auto_templates/, re-gated on load).
# Derived from the engine's own registries, never a frozen literal set — the
# system synthesizes new templates autonomously, so a hardcoded list here
# would fail every legitimately-minted rule that followed it.
from app.validation.auto_templates import load_auto_templates  # noqa: E402
from app.validation.graduated import GRADUATED  # noqa: E402

BUILTIN_TEMPLATES = {"context_gate", "tiered_family_arbitration",
                     "companion_completion", "residual_secondary_demotion",
                     "icd_tiered_axis", "documented_service_completion",
                     "documented_diagnosis_completion"}
KNOWN_TEMPLATES = (BUILTIN_TEMPLATES | set(GRADUATED)
                   | set(load_auto_templates()))

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
        # DISABLED rules are inert history (the engine's rule() gate skips
        # them before template dispatch) — they may legitimately reference a
        # template that was since retired/deleted, so the known-template
        # contract applies only to rules the engine can actually run.
        if r.get("enabled", True):
            check(f"{rid}: known template",
                  r.get("template") in KNOWN_TEMPLATES)
        check(f"{rid}: cites an authority", bool(r.get("authority")))
        act = r.get("action", {})
        # Finding-emitting actions carry severity + category + a message
        # field (whose name varies by template: built-ins use "message",
        # auto templates define their own, e.g. message_added/
        # message_undocumented — see each template's SCHEMA_DOC).
        # SUPPRESSION actions emit no finding — they retire an advisory —
        # so they carry only "note", the audit-trail text recorded on the
        # correction; severity/category would be meaningless there. The
        # note lives under action.note (coverage_evidence_advisory_
        # suppression) or at the rule top level (advisory_identity_
        # evidence_suppression) — both templates' documented schemas.
        emits_finding = any(act.get(k) for k in ("severity", "category"))
        check(f"{rid}: action has severity/category/message (or is a "
              f"note-only suppression)",
              (emits_finding
               and all(act.get(k) for k in ("severity", "category"))
               and any(v for k, v in act.items() if k.startswith("message")))
              or (not emits_finding
                  and bool(act.get("note") or r.get("note"))))

    print("\n[no operative hardcoded codes]")
    for r in rules:
        rid = r.get("id", "?")
        operative = {k: v for k, v in r.items() if k in OPERATIVE_FIELDS}
        # Underscore-prefixed keys (_family_why, _strip_why, ...) are
        # embedded provenance PROSE, not operative config — codes cited
        # there as examples are as legitimate as codes in message text.
        def _strip_prose(v):
            if isinstance(v, dict):
                return {k: _strip_prose(x) for k, x in v.items()
                        if not k.startswith("_")}
            if isinstance(v, list):
                return [_strip_prose(x) for x in v]
            return v
        operative = _strip_prose(operative)
        # A family block may anchor itself to an ICD CATEGORY via
        # code_regex ONLY when a descriptor gate in the same block actually
        # selects the members by querying the authoritative descriptors
        # (descriptor_requires_any/all, category_desc_contains_any) — the
        # prefix is then a structural index into the published ICD chapter
        # layout, and membership still tracks the real code set as it
        # changes. A code_regex WITHOUT a descriptor gate is a hardcoded
        # code-family list and stays forbidden.
        fam = operative.get("family") or {}
        if isinstance(fam, dict) and any(
                fam.get(k) for k in ("descriptor_requires_any",
                                     "descriptor_requires_all",
                                     "category_desc_contains_any")):
            fam = {k: ("" if k == "code_regex" else v) for k, v in fam.items()}
            operative = dict(operative, family=fam)
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
