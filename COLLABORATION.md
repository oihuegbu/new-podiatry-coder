# Claude–Codex collaboration protocol

This document is the project-wide operating contract for every change in this
repository. It applies to application code, medical knowledge ingestion,
retrieval, eligibility, validation, infrastructure, deployment, documentation,
and research. It supplements `CLAUDE.md`; stricter safety requirements win.

## Outcome

The process exists to produce changes that are:

- traceable to an explicit objective, invariant, and source of authority;
- independently reviewed at the exact commit that will be deployed;
- tested on happy paths, failure paths, boundaries, and adjacent bug classes;
- implemented in source so a clean build or deployment reproduces the result;
- unable to silently weaken a medical-coding safety control.

Agreement between models is not the goal. Evidence-backed convergence is. A
finding remains open until its implementation is independently verified or the
reviewer explicitly narrows or rescinds it.

## Roles and separation of duties

The default roles are:

- **Product owner:** the user. Sets product intent and decides genuine product
  tradeoffs.
- **Primary implementer:** Claude. Discovers, specifies, implements, tests,
  self-reviews, and prepares the handoff.
- **Independent reviewer:** Codex. Reviews read-only first, reproduces claims,
  challenges assumptions and failure paths, and verifies remediation.

Roles may switch only when the user explicitly assigns them for a work item.
The implementer must never be the sole independent reviewer. Only one role
writes during a review cycle:

1. The implementer writes through `READY_FOR_REVIEW`.
2. The reviewer does not edit that implementation while reviewing.
3. The implementer remediates accepted findings.
4. The reviewer re-reviews the resulting commit.

This avoids two models silently co-authoring the same mistake and preserves
clear accountability.

## Source of truth

The Git commit is the source of truth for code and configuration. A work item
or pull-request body is the source of truth for scope and review state.
Generated audit artifacts are evidence, not source. A live-system hand edit is
never a completed fix.

Medical decisions must derive from versioned authoritative data and provenance.
No work item may introduce hardcoded medical codes, code families, or
descriptor proxies. Deterministic rule instances belong in versioned rule
configuration and cite their authority.

## Risk classification

Every work item declares one class:

- **A — claim-affecting:** can alter extracted evidence, eligibility,
  candidate retrieval/ranking, code or modifier selection, units, bundling,
  coverage, claim output, or autonomy/blocking decisions.
- **B — operational:** can alter availability, data refresh, privacy,
  credentials, observability, deployment, queues, retries, or persistence
  without intentionally changing claim semantics.
- **C — non-behavioral:** documentation, comments, or tooling proven not to
  affect runtime or authoritative data.

Uncertain classification defaults upward. Class A requires domain-authority
citations, claim-level negative tests, and independent review. Class B requires
failure/recovery and clean-deployment tests. Class C still requires an exact
diff review and applicable checks.

## Control mode

Every new claim-affecting control declares exactly one mode:

- `OBSERVATIONAL`: records decisions and shadow diffs but cannot change the
  emitted claim or autonomy outcome.
- `ENFORCED_FAIL_CLOSED`: may block, defer, or change a claim only when its
  documented invariant is violated.
- `DISABLED`: unavailable at runtime except through a new reviewed change.

The mode must be explicit in configuration and audit output. Shadow and
enforced results must never be mixed implicitly.

## Work-item contract

Before implementation, record:

- objective and non-goals;
- risk class and control mode;
- claim-affecting yes/no;
- base commit and intended branch;
- invariants that must remain true;
- authoritative sources and effective dates;
- acceptance criteria and negative cases;
- affected data, interfaces, callers, deployment path, and audit records;
- rollback or disable strategy;
- known uncertainty and expected review evidence.

Use the repository issue template, pull-request template, or an equivalent
tracked work item. Material scope changes update this contract before code.

## Implementation cycle

The implementer performs these stages:

1. **Discover:** trace the full current path and inspect adjacent instances of
   the same bug or design class.
2. **Specify:** write the work-item contract and identify authority.
3. **Implement:** make the smallest complete architectural change in source.
4. **Verify:** run focused tests, broad guards, failure tests, and clean-build
   checks proportionate to risk.
5. **Self-review:** re-read the full path, not only the diff. Inspect the fix's
   own failure modes, retry/process/deploy boundaries, observability, and
   rollback.
6. **Commit and hand off:** publish an exact commit and stop writing until the
   independent review is returned.

The handoff follows `.collaboration/HANDOFF_TEMPLATE.md`. Tests are reported
with exact commands and results; “tests pass” is not sufficient.

## Independent review cycle

The reviewer follows `.collaboration/REVIEW_TEMPLATE.md` and performs:

1. verify base, target, and diff scope;
2. restate the intended invariant independently;
3. trace the full execution path and authoritative-data path;
4. reproduce the implementer's tests;
5. add adversarial tests for omissions, ambiguity, stale data, empty data,
   conflicting evidence, malformed input, unavailable dependencies, and
   restart/retry boundaries as applicable;
6. inspect adjacent instances of the same defect class;
7. review auditability, reversibility, and deployment reproducibility;
8. publish findings with severity, evidence, reproduction, and acceptance
   criterion.

Finding severity:

- **P0:** immediate patient, claim, security, or irreversible integrity risk.
- **P1:** likely incorrect claim/safety outcome, material data loss, or a
  control that can silently fail.
- **P2:** bounded correctness, resilience, auditability, or maintainability
  gap.
- **P3:** low-risk improvement that does not block the current objective.

Each finding is `OPEN`, `ACCEPTED`, `DISPUTED`, `SUPERSEDED`, or
`VERIFIED`. P0–P2 findings block final verification unless the reviewer
explicitly rescinds them or the product owner accepts a documented product
tradeoff.

## Remediation and disagreement

The implementer answers every finding:

- `ACCEPTED`: implement it and provide a new target commit plus focused
  regression evidence.
- `DISPUTED`: provide executable evidence, authoritative sources, or a
  falsifiable counterexample.
- `SUPERSEDED`: identify the later finding or design decision that replaces
  it.

The reviewer then independently re-runs relevant tests and marks the finding
`VERIFIED`, `REOPENED`, `PARTIALLY_FIXED`, `SUPERSEDED`, or
`RESCINDED`. Acceptance by the implementer never closes a finding.

When Claude disproves a Codex finding, Codex must narrow or rescind it. When
Codex establishes a gap and Claude accepts it, Claude implements it and Codex
reviews that implementation for new or remaining issues. Neither model should
manufacture consensus. Escalate only unresolved product intent, competing
authorities, or risk tolerance to the product owner.

## Universal gates

Before a target is `VERIFIED`, all applicable gates must pass:

- scope and base/target identity are exact;
- no secrets, PHI, runtime output, or unrelated files are committed;
- compilation, repository guard scripts, focused tests, and full tests pass;
- no medical code or code-family logic is hardcoded;
- authoritative inputs have provenance, version, effective date, integrity
  checks, and explicit missing/stale behavior;
- failures are loud, typed, observable, and do not degrade into empty success;
- callers handle new returns and exceptions across process/retry boundaries;
- audit output distinguishes observation, enforcement, model input/output,
  deterministic decisions, data versions, and final claim lineage;
- the change is reproducible from a clean build/deployment;
- rollback or disable behavior is tested;
- the unconditional post-fix review in `CLAUDE.md` is complete.

Specialized Class A gates also require evidence anchoring, eligibility before
retrieval, version-aware retrieval, tabular/coverage/NCCI/MUE validation as
applicable, deterministic ownership/deduplication, and regression cases where
plausible alternatives must not become billable output.

LLM changes additionally record provider, model/profile identity, prompt and
schema version, independence assumptions, timeout/retry behavior, and
deterministic tie/arbitration behavior. Model agreement alone is never proof.

## SHA-bound pull-request approval

Independent review is valid only for one exact 40-character target commit.
The pull-request contract records:

- `Review status: PENDING` while work or review is incomplete;
- `Review status: VERIFIED` only after independent verification;
- `Review target SHA: <exact head commit>`.

Any new commit invalidates the previous verification automatically because the
recorded SHA no longer matches the pull-request head. A non-draft pull request
must pass the collaboration-governance check, which requires `VERIFIED` and
an exact SHA match. Drafts may remain `PENDING`.

## Deployment and observation

Deploy only the reviewed commit, using checked-in infrastructure and deployment
source. Record commit, image digest, data-manifest versions, rule-pack version,
model profiles, configuration mode, and deployment time. Smoke tests confirm
service health and audit identity; claim-affecting changes also compare
expected and actual claim/audit behavior.

If production observation reveals a defect, open a new work item and fix the
source. Never treat an SSH edit or manual data repair as the durable solution.

## Completion

A work item is complete only when:

- all acceptance criteria and applicable gates pass;
- all P0–P2 findings are verified, rescinded, superseded, or explicitly
  resolved by the product owner;
- the independent reviewer verifies the exact final commit;
- clean deployment evidence exists when deployment is in scope;
- known limitations and follow-up measurements are recorded.
