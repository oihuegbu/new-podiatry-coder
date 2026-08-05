#!/usr/bin/env python3
"""Validate the repository-wide Claude-Codex collaboration contract."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import sys


REQUIRED_FILES = {
    "COLLABORATION.md": "SHA-bound pull-request approval",
    "AGENTS.md": "COLLABORATION_PROTOCOL: independent-reviewer",
    "CLAUDE.md": "COLLABORATION_PROTOCOL: primary-implementer",
    ".collaboration/HANDOFF_TEMPLATE.md": "READY_FOR_REVIEW handoff",
    ".collaboration/REVIEW_TEMPLATE.md": "Independent review",
    ".github/PULL_REQUEST_TEMPLATE.md": "Review target SHA",
    ".github/ISSUE_TEMPLATE/work-item.yml": "Claude-Codex work item",
    ".github/workflows/collaboration-governance.yml": (
        "Require SHA-bound independent verification"
    ),
}

REQUIRED_HEADINGS = (
    "## Work item",
    "## Risk and control mode",
    "## Invariants and authorities",
    "## Verification",
    "## Self-review and handoff",
    "## Independent review",
)

IMPLEMENTATION_FIELDS = (
    "Objective",
    "Non-goals",
    "Risk class",
    "Claim-affecting",
    "Implementer",
    "Independent reviewer",
    "Base SHA",
    "Control mode",
    "Rollback/disable strategy",
    "Affected runtime/deployment boundaries",
    "Invariant",
    "Authoritative sources",
    "Missing/stale/ambiguous-data behavior",
    "No-hardcoded-medical-code evidence",
    "Focused tests",
    "Negative/failure tests",
    "Repository guards",
    "Full affected suite",
    "Clean build/deploy",
    "Handoff status",
    "Target SHA",
    "Full-path re-read",
    "Failure and boundary review",
    "Adjacent defect-class review",
    "Known limitations",
)

REVIEW_FIELDS = (
    "Review status",
    "Review target SHA",
    "Open P0-P2 findings",
    "Review evidence",
)

FIELD_PATTERN = re.compile(
    r"^\s*-\s*\*\*([^*]+):\*\*\s*(.*?)\s*$", re.MULTILINE
)
HTML_COMMENT_PATTERN = re.compile(r"<!--.*?-->", re.DOTALL)
FULL_SHA_PATTERN = re.compile(r"^[0-9a-fA-F]{40}$")
PLACEHOLDERS = {"", "tbd", "todo", "required", "<required>"}


def _clean(value: str) -> str:
    return HTML_COMMENT_PATTERN.sub("", value).strip()


def parse_fields(body: str) -> dict[str, str]:
    """Return normalized bold-list fields from a pull-request body."""
    return {
        name.strip(): _clean(value)
        for name, value in FIELD_PATTERN.findall(body or "")
    }


def validate_repository(root: Path) -> list[str]:
    """Validate that the durable collaboration files and role markers exist."""
    errors: list[str] = []
    for relative_path, marker in REQUIRED_FILES.items():
        path = root / relative_path
        if not path.is_file():
            errors.append(f"missing required collaboration file: {relative_path}")
            continue
        content = path.read_text(encoding="utf-8")
        if marker not in content:
            errors.append(f"{relative_path} is missing required marker: {marker}")
    return errors


def validate_pr_body(
    body: str,
    *,
    expected_head: str | None = None,
    require_reviewed: bool = False,
) -> list[str]:
    """Validate a PR work-item/handoff and its SHA-bound review state."""
    errors: list[str] = []
    for heading in REQUIRED_HEADINGS:
        if heading not in body:
            errors.append(f"missing required PR heading: {heading}")

    fields = parse_fields(body)
    for field in IMPLEMENTATION_FIELDS + REVIEW_FIELDS:
        if field not in fields:
            errors.append(f"missing required PR field: {field}")

    for field in IMPLEMENTATION_FIELDS:
        value = fields.get(field, "")
        if value.lower() in PLACEHOLDERS:
            errors.append(f"PR field must be completed: {field}")

    risk = fields.get("Risk class", "").upper()
    if risk not in {"A", "B", "C"}:
        errors.append("Risk class must be exactly A, B, or C")

    claim_affecting = fields.get("Claim-affecting", "").lower()
    if claim_affecting not in {"yes", "no"}:
        errors.append("Claim-affecting must be Yes or No")
    if risk == "A" and claim_affecting != "yes":
        errors.append("Risk class A must declare Claim-affecting: Yes")

    mode = fields.get("Control mode", "")
    allowed_modes = {"OBSERVATIONAL", "ENFORCED_FAIL_CLOSED", "DISABLED"}
    if mode not in allowed_modes:
        errors.append(
            "Control mode must be OBSERVATIONAL, ENFORCED_FAIL_CLOSED, or DISABLED"
        )

    implementer = fields.get("Implementer", "").casefold()
    reviewer = fields.get("Independent reviewer", "").casefold()
    if implementer and reviewer and implementer == reviewer:
        errors.append("Implementer and Independent reviewer must differ")

    for field in ("Base SHA", "Target SHA"):
        value = fields.get(field, "")
        if value and not FULL_SHA_PATTERN.fullmatch(value):
            errors.append(f"{field} must be an exact 40-character Git SHA")

    if expected_head:
        if not FULL_SHA_PATTERN.fullmatch(expected_head):
            errors.append("expected pull-request head is not a full Git SHA")
        elif fields.get("Target SHA", "").lower() != expected_head.lower():
            errors.append("Target SHA does not match the current pull-request head")

    handoff_status = fields.get("Handoff status", "")
    if handoff_status != "READY_FOR_REVIEW":
        errors.append("Handoff status must be READY_FOR_REVIEW")

    review_status = fields.get("Review status", "")
    if review_status not in {"PENDING", "VERIFIED"}:
        errors.append("Review status must be PENDING or VERIFIED")

    review_target = fields.get("Review target SHA", "")
    if review_status == "PENDING":
        if review_target != "PENDING":
            errors.append("PENDING review must use Review target SHA: PENDING")
    elif review_status == "VERIFIED":
        if not FULL_SHA_PATTERN.fullmatch(review_target):
            errors.append("VERIFIED review requires an exact Review target SHA")
        if expected_head and review_target.lower() != expected_head.lower():
            errors.append(
                "Review target SHA does not match the current pull-request head"
            )
        for field in ("Open P0-P2 findings", "Review evidence"):
            if fields.get(field, "").lower() in PLACEHOLDERS:
                errors.append(f"VERIFIED review must complete: {field}")

    if require_reviewed and review_status != "VERIFIED":
        errors.append("non-draft pull request requires Review status: VERIFIED")

    return errors


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository root",
    )
    parser.add_argument(
        "--pr-body-env",
        help="environment variable containing the pull-request body",
    )
    parser.add_argument(
        "--expected-head-env",
        help="environment variable containing the pull-request head SHA",
    )
    parser.add_argument(
        "--require-reviewed",
        action="store_true",
        help="require VERIFIED review state (for non-draft pull requests)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    errors = validate_repository(args.root)

    if args.require_reviewed and not args.pr_body_env:
        errors.append("--require-reviewed requires --pr-body-env")

    if args.pr_body_env:
        body = os.environ.get(args.pr_body_env, "")
        if not body:
            errors.append(f"empty PR body environment variable: {args.pr_body_env}")
        else:
            expected_head = (
                os.environ.get(args.expected_head_env, "")
                if args.expected_head_env
                else None
            )
            errors.extend(
                validate_pr_body(
                    body,
                    expected_head=expected_head or None,
                    require_reviewed=args.require_reviewed,
                )
            )

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print("Collaboration contract valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
