from pathlib import Path

from tools.check_collaboration_contract import (
    parse_fields,
    validate_pr_body,
    validate_repository,
)


HEAD = "a" * 40
BASE = "b" * 40


def _body(
    *,
    status: str = "PENDING",
    review_target: str = "PENDING",
    target: str = HEAD,
    implementer: str = "Claude",
    reviewer: str = "Codex",
    risk: str = "A",
    claim_affecting: str = "Yes",
) -> str:
    return f"""## Work item

- **Objective:** Improve the service graph.
- **Non-goals:** No unrelated model changes.
- **Risk class:** {risk}
- **Claim-affecting:** {claim_affecting}
- **Implementer:** {implementer}
- **Independent reviewer:** {reviewer}
- **Base SHA:** {BASE}

## Risk and control mode

- **Control mode:** OBSERVATIONAL
- **Rollback/disable strategy:** Set the source configuration to DISABLED.
- **Affected runtime/deployment boundaries:** Pipeline and audit writer.

## Invariants and authorities

- **Invariant:** Ineligible services cannot reach retrieval.
- **Authoritative sources:** Versioned source manifest and cited policy.
- **Missing/stale/ambiguous-data behavior:** Fail closed and emit an audit reason.
- **No-hardcoded-medical-code evidence:** Guard script passes.

## Verification

- **Focused tests:** pytest focused tests passed.
- **Negative/failure tests:** missing-data cases passed.
- **Repository guards:** all repository guards passed.
- **Full affected suite:** pytest passed.
- **Clean build/deploy:** clean image build passed.

## Self-review and handoff

- **Handoff status:** READY_FOR_REVIEW
- **Target SHA:** {target}
- **Full-path re-read:** completed.
- **Failure and boundary review:** completed.
- **Adjacent defect-class review:** completed.
- **Known limitations:** None after deliberate review.

## Independent review

- **Review status:** {status}
- **Review target SHA:** {review_target}
- **Open P0-P2 findings:** None.
- **Review evidence:** Independent report recorded.
"""


def test_repository_contract_is_installed():
    root = Path(__file__).resolve().parents[1]
    assert validate_repository(root) == []


def test_parse_fields_removes_template_comments():
    fields = parse_fields("- **Objective:** <!-- required -->\n")
    assert fields["Objective"] == ""


def test_draft_handoff_contract_passes():
    assert validate_pr_body(_body(), expected_head=HEAD) == []


def test_verified_contract_passes_for_exact_head():
    body = _body(status="VERIFIED", review_target=HEAD)
    assert validate_pr_body(
        body, expected_head=HEAD, require_reviewed=True
    ) == []


def test_new_commit_invalidates_existing_review():
    old_head = "c" * 40
    body = _body(status="VERIFIED", review_target=old_head, target=old_head)
    errors = validate_pr_body(
        body, expected_head=HEAD, require_reviewed=True
    )
    assert "Target SHA does not match the current pull-request head" in errors
    assert (
        "Review target SHA does not match the current pull-request head"
        in errors
    )


def test_non_draft_requires_independent_verification():
    errors = validate_pr_body(
        _body(), expected_head=HEAD, require_reviewed=True
    )
    assert "non-draft pull request requires Review status: VERIFIED" in errors


def test_same_implementer_and_reviewer_is_rejected():
    errors = validate_pr_body(
        _body(implementer="Claude", reviewer="claude"), expected_head=HEAD
    )
    assert "Implementer and Independent reviewer must differ" in errors


def test_class_a_must_be_claim_affecting():
    errors = validate_pr_body(
        _body(risk="A", claim_affecting="No"), expected_head=HEAD
    )
    assert "Risk class A must declare Claim-affecting: Yes" in errors


def test_placeholders_do_not_satisfy_implementation_contract():
    body = _body().replace(
        "- **Objective:** Improve the service graph.",
        "- **Objective:** <!-- required -->",
    )
    errors = validate_pr_body(body, expected_head=HEAD)
    assert "PR field must be completed: Objective" in errors
