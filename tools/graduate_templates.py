#!/usr/bin/env python3
"""Propose graduation of proven self-authored templates.

Synthesized templates live sandboxed in data/rules/auto_templates/ and are
re-validated on every load. This tool evaluates maturity and writes an inert
graduation proposal. It never writes app/validation/graduated/, retires the
sandbox copy, or changes executable policy at runtime.

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

Eligibility produces an inert, fingerprinted proposal.  Production code
is never written by this runtime tool; a separately reviewed, signed build
must perform deployment.

Runs inside the app container (wired into run.py's post-batch sequence;
also standalone):
  docker compose run --rm app python tools/graduate_templates.py
  --dry-run    report eligibility, promote nothing
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from loguru import logger  # noqa: E402

RULES_PATH = ROOT / "data" / "rules" / "validator_rules.json"
GRADUATED_DIR = ROOT / "app" / "validation" / "graduated"
PROPOSALS_DIR = ROOT / "data" / "rules" / "proposals"
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
    from app.validation.auto_templates import (
        validate_template_clause_tagging, validate_template_source)

    enabled, disabled = _rules_of(template)
    since = _live_since(template, path, enabled)
    age_days = (datetime.now(timezone.utc) - since).total_seconds() / 86400
    docs = _docs_processed_since(since, results_dir)
    reopened = _reopened_classes({r.get("id") for r in enabled + disabled})
    src = path.read_text(encoding="utf-8")
    static_problems = validate_template_source(src)
    # Graduation is where a sandboxed template becomes permanent ordinary
    # Python. An untagged emission site promoted here would outlive the
    # migration, so clause tagging gates promotion — never loading. A
    # template that fails this stays installed and keeps running; it just
    # waits in the sandbox until its _add sites name their clauses.
    clause_problems = validate_template_clause_tagging(src)

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
        "clause_tagging": {"ok": not clause_problems,
                           "problems": clause_problems[:4]},
    }
    return {"template": template, "live_since": since.isoformat(),
            "eligible": all(c["ok"] for c in checks.values()),
            "checks": checks}


def promote(template: str, path: Path, report: dict) -> str:
    """Create an inert graduation proposal; never write executable app code."""
    dest = GRADUATED_DIR / f"{template}.py"
    if dest.exists():
        return f"{dest.name} already exists in the graduated package"
    try:
        source_path = str(path.relative_to(ROOT))
    except ValueError:
        source_path = str(path)
    body = {
        "proposal_version": 1, "status": "draft",
        "proposal_type": "graduate_template", "template": template,
        "source_path": source_path,
        "source_sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
        "template_source": path.read_text(encoding="utf-8"),
        "eligibility": report,
        "required_lifecycle": ["independent_human_code_review", "signed_build",
                               "sandbox_replay", "shadow_deployment",
                               "rollback_rehearsal"],
    }
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":"),
                         default=str).encode()
    body["proposal_fingerprint"] = "sha256:" + hashlib.sha256(encoded).hexdigest()
    PROPOSALS_DIR.mkdir(parents=True, exist_ok=True)
    proposal = PROPOSALS_DIR / (
        f"graduate-{template}-{body['proposal_fingerprint'][7:19]}.json")
    if not proposal.exists():
        proposal.write_text(json.dumps(body, indent=2, sort_keys=True,
                                       default=str))
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
            logger.info(f"GRADUATION PROPOSED: {name} (sandbox and live app "
                        f"code unchanged)")
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
