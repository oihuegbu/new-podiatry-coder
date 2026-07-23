#!/usr/bin/env python3
"""Graduate proven self-authored templates into the app tree.

Synthesized templates live sandboxed in data/rules/auto_templates/,
re-validated on every load. This tool closes their lifecycle: a template
that has PROVEN itself in production is promoted verbatim into
app/validation/graduated/ — static, trusted application code, the same
standing as the hand-written mechanics — and the sandbox copy retires.
Rules referencing the template keep working unchanged: graduation moves
the code's trust category, never its behavior, and the template
vocabulary (tools/auto_actuate.all_templates) keeps the name throughout,
so no escalation record churns.

PROVEN means every deterministic criterion holds (env-tunable):

  age         live >= GRADUATE_MIN_DAYS (default 14) since the earliest
              enabled rule referencing it was actuated (file mtime as
              fallback)
  exposure    >= GRADUATE_MIN_DOCS (default 25) distinct documents have
              result files newer than that go-live moment — the template
              executed against real, fresh traffic
  rules       at least one enabled rule references it, and NO rule
              referencing it was ever disabled (a rollback is disproof)
  held        no flip class actuated by one of its rules has reopened
              (tools/flip_triage marks recurrence) — its fixes stuck
  static      the source still passes the full safety gate (sandbox
              constraints + no hardcoded medical codes)

Promotion is transactional: write to the graduated package, import-check
the new module, and only then retire the sandbox file; any failure rolls
the promotion back and leaves the sandbox copy authoritative.

Runs inside the app container (wired into run.py's post-batch sequence;
also standalone):
  docker compose run --rm app python tools/graduate_templates.py
  --dry-run    report eligibility, promote nothing
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from loguru import logger  # noqa: E402

RULES_PATH = ROOT / "data" / "rules" / "validator_rules.json"
GRADUATED_DIR = ROOT / "app" / "validation" / "graduated"
DEFAULT_RESULTS = ROOT / "output" / "results"

MIN_DAYS = float(os.getenv("GRADUATE_MIN_DAYS", "14"))
MIN_DOCS = int(os.getenv("GRADUATE_MIN_DOCS", "25"))


def _rules_of(template: str) -> tuple[list[dict], list[dict]]:
    """(enabled, disabled) pack rules referencing the template."""
    pack = json.loads(RULES_PATH.read_text())
    ref = [r for r in pack.get("rules", [])
           if r.get("template") == template]
    return ([r for r in ref if r.get("enabled", True)],
            [r for r in ref if not r.get("enabled", True)])


def _live_since(template: str, path: Path,
                enabled_rules: list[dict]) -> datetime:
    """The moment the template went live: its earliest enabled rule's
    actuation timestamp, else the module file's mtime."""
    stamps = []
    for r in enabled_rules:
        t = (r.get("provenance") or {}).get("actuated_at")
        if t:
            try:
                dt = datetime.fromisoformat(str(t))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                stamps.append(dt)
            except ValueError:
                pass
    if stamps:
        return min(stamps)
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)


def _docs_processed_since(since: datetime, results_dir: Path) -> int:
    n = 0
    for f in results_dir.glob("*_results.json"):
        if f.name == "all_results.json":
            continue
        if datetime.fromtimestamp(f.stat().st_mtime,
                                  tz=timezone.utc) > since:
            n += 1
    return n


def _reopened_classes(rule_ids: set[str]) -> list[str]:
    """Flip classes actuated by one of the template's rules that later
    REOPENED (the deployed fix did not hold) — including ones re-actuated
    since: a reopen on record is disproof regardless of current status."""
    try:
        from tools import flip_triage
        queue = flip_triage.load_queue(flip_triage.QUEUE_PATH)
    except Exception:
        return []
    out = []
    for c in queue:
        act = c.get("actuation") or {}
        if act.get("rule_id") in rule_ids and c.get("reopened"):
            out.append(c["class_key"])
    return out


def eligibility(template: str, path: Path,
                results_dir: Path = DEFAULT_RESULTS) -> dict:
    """Every criterion with its measured value; 'eligible' iff all hold."""
    from app.validation.auto_templates import validate_template_source

    enabled, disabled = _rules_of(template)
    since = _live_since(template, path, enabled)
    age_days = (datetime.now(timezone.utc) - since).total_seconds() / 86400
    docs = _docs_processed_since(since, results_dir)
    reopened = _reopened_classes({r.get("id") for r in enabled + disabled})
    static_problems = validate_template_source(
        path.read_text(encoding="utf-8"))

    checks = {
        "age": {"ok": age_days >= MIN_DAYS,
                "value": round(age_days, 2), "min": MIN_DAYS},
        "exposure": {"ok": docs >= MIN_DOCS, "value": docs,
                     "min": MIN_DOCS},
        "rules": {"ok": bool(enabled) and not disabled,
                  "enabled": [r.get("id") for r in enabled],
                  "disabled": [r.get("id") for r in disabled]},
        "held": {"ok": not reopened, "reopened_classes": reopened},
        "static": {"ok": not static_problems,
                   "problems": static_problems[:4]},
    }
    return {"template": template, "live_since": since.isoformat(),
            "eligible": all(c["ok"] for c in checks.values()),
            "checks": checks}


def promote(template: str, path: Path, report: dict) -> str:
    """Transactional promotion; returns '' on success, else the reason
    the promotion was rolled back."""
    import importlib.util

    dest = GRADUATED_DIR / f"{template}.py"
    if dest.exists():
        return f"{dest.name} already exists in the graduated package"
    src = path.read_text(encoding="utf-8")
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    criteria = json.dumps(report["checks"], default=str)[:600]
    header = (
        f"# GRADUATED self-authored template — promoted verbatim from\n"
        f"# data/rules/auto_templates/{path.name} by "
        f"tools/graduate_templates.py.\n"
        f"# graduated_at: {stamp}\n"
        f"# criteria: {criteria}\n\n")
    # The sandbox loader injects `re` into every template's namespace, so
    # sandboxed code may use re.* without importing it. Plain Python has
    # no such courtesy — materialize the import the sandbox implied.
    if re.search(r"\bre\s*\.", src) and not re.search(
            r"^\s*import\s+re\b", src, re.MULTILINE):
        header += "import re\n\n"
    dest.write_text(header + src, encoding="utf-8")
    try:
        # Path-based import: verifies the promoted file loads as ordinary
        # (unsandboxed) Python and still honors the template contract.
        spec = importlib.util.spec_from_file_location(
            f"_graduation_check_{template}", dest)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert getattr(mod, "TEMPLATE_NAME", None) == template, \
            "TEMPLATE_NAME mismatch"
        assert callable(getattr(mod, "execute", None)), "execute missing"
    except Exception as exc:
        dest.unlink(missing_ok=True)
        return f"import check failed after promotion: {exc!r}"
    # Import verified: retire the sandbox copy and refresh the in-process
    # registry so the vocabulary never shrinks between the two steps.
    path.unlink()
    from app.validation import graduated
    graduated.refresh()
    from app.validation.auto_templates import _cache
    _cache.pop(str(path), None)
    return ""


def graduate(results_dir: Path = DEFAULT_RESULTS,
             dry_run: bool = False) -> dict:
    from app.validation.auto_templates import (AUTO_TEMPLATES_DIR,
                                               load_auto_templates)
    summary = {"considered": 0, "promoted": [], "not_yet": [],
               "failed": []}
    for name, entry in sorted(load_auto_templates().items()):
        summary["considered"] += 1
        path = Path(entry["path"])
        if not str(path).startswith(str(AUTO_TEMPLATES_DIR)):
            continue
        report = eligibility(name, path, results_dir)
        if not report["eligible"]:
            why = {k: v for k, v in report["checks"].items()
                   if not v["ok"]}
            summary["not_yet"].append({"template": name, "blocking": why})
            logger.info(f"Graduation: {name} not yet eligible "
                        f"({', '.join(why)})")
            continue
        if dry_run:
            summary["promoted"].append({"template": name,
                                        "dry_run": True})
            logger.info(f"Graduation DRY RUN: {name} is eligible")
            continue
        err = promote(name, path, report)
        if err:
            summary["failed"].append({"template": name, "reason": err})
            logger.error(f"Graduation of {name} rolled back: {err}")
        else:
            summary["promoted"].append({"template": name,
                                        "report": report})
            logger.info(f"GRADUATED: {name} -> app/validation/graduated/ "
                        f"(trusted app code; sandbox copy retired)")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", default=str(DEFAULT_RESULTS))
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    print(json.dumps(graduate(Path(args.results_dir), args.dry_run),
                     indent=2, default=str))


if __name__ == "__main__":
    main()
