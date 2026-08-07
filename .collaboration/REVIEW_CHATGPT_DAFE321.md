# Independent review

## Target integrity

- Reviewer: ChatGPT (independent reviewer lane requested by product owner)
- Implementer: Claude
- Work item: Current Claude remediation cycle on `claude-medical-coder`; existing checked-in handoff is `HANDOFF_F4F5.md`
- Base SHA in existing handoff: `0a74a3ced3c27450589a56fc03c3516d6ab62580`
- Reviewed target SHA in existing handoff: `5186c189315fac851cf321796caa0d2a050e6538`
- Current implementation head SHA observed: `dafe32151b2b55ef9b634f7cee6c835ccc1417dc`
- Scope matches contract: No
- Reviewer remained read-only with respect to implementation: Yes

**STOP CONDITION TRIGGERED.** `.collaboration/REVIEW_TEMPLATE.md` requires the reviewer to stop when the reviewed target and current head differ. The implementation branch is 10 commits ahead of the target named in `HANDOFF_F4F5.md`, and those later commits include broad Class-A claim-affecting changes. Substantive verification of `5186c189...` would therefore be stale and cannot produce a valid `VERIFIED` disposition.

## Independent interpretation

- Intended objective represented by the current head: architectural remediation following Codex re-review, including extraction fail-closed behavior, actor identity/ownership, per-service necessity, release/source attestation, eligibility snapshot binding, production NCCI preparation, and audit-chain truncation detection.
- Claim/safety invariant: no claim-affecting change may be accepted or deployed unless the exact current commit is independently reviewed with its complete scope, authorities, tests, failure modes, and rollback behavior.
- Risk class and control mode assessment: Class A; expected `ENFORCED_FAIL_CLOSED` for the claim-affecting controls described in the current commit.
- Controlling repository authority: `COLLABORATION.md`, `CLAUDE.md`, and `.collaboration/REVIEW_TEMPLATE.md` exact-SHA rules.

## Review passes

- [x] Exact target/head identity check
- [x] Stale-scope check
- [ ] Exact diff and unrelated-file check — intentionally stopped pending current handoff
- [ ] Full execution and caller path — intentionally stopped pending current handoff
- [ ] Authoritative-data provenance/effective-date path — intentionally stopped pending current handoff
- [ ] Missing, stale, empty, ambiguous, and conflicting data — intentionally stopped pending current handoff
- [ ] Negative claim and plausible-alternative cases — intentionally stopped pending current handoff
- [ ] Exception, timeout, retry, process, restart, and rollback boundaries — intentionally stopped pending current handoff
- [ ] Audit lineage and observation/enforcement separation — intentionally stopped pending current handoff
- [ ] Adjacent instances of the same defect class — intentionally stopped pending current handoff
- [ ] Reproduction of implementer commands — no current handoff commands supplied for `dafe321...`
- [ ] Additional adversarial tests — intentionally stopped pending current handoff
- [ ] Clean build/deployment path when applicable — intentionally stopped pending current handoff

## Findings

### CHATGPT-R1 — P1 — Current claim-affecting head has no SHA-matching READY_FOR_REVIEW handoff

- Status: OPEN
- Boundary: collaboration governance / review identity
- Evidence: checked-in `HANDOFF_F4F5.md` names target `5186c189315fac851cf321796caa0d2a050e6538`; current `claude-medical-coder` head resolves to `dafe32151b2b55ef9b634f7cee6c835ccc1417dc`; comparison shows 10 commits after the stale target. The current head itself contains Class-A remediation touching extraction, eligibility, gates, pipeline, provenance, authoritative data preparation, and tests.
- Reproduction: compare `5186c189315fac851cf321796caa0d2a050e6538...claude-medical-coder`; resolve `claude-medical-coder` head.
- Impact: any approval tied to `5186c189...` would not cover the code that would actually be deployed. This can allow unreviewed claim-affecting behavior through the collaboration boundary.
- Controlling invariant: independent review is valid only for the exact current 40-character target SHA; any later commit returns review status to `PENDING`.
- Acceptance criterion: Claude publishes a new `READY_FOR_REVIEW` handoff for the exact current implementation head (or a newer final head), with the correct base SHA, complete changed scope, risk/control mode, authoritative sources, exact test commands/results, unconditional self-review, known limitations, rollback/disable strategy, and reviewer-focus section; then Claude stops writing until review completes.

## Remediation verification

### CHATGPT-R1

- Implementer response: pending
- Remediation target SHA: pending
- Reviewer disposition: pending
- Commands rerun: pending
- Evidence: pending
- New or remaining gaps: substantive technical review has not started because the repository-mandated target-integrity stop condition is active.

## Final disposition

- Review status: PENDING
- Verified target SHA: none
- Open P0–P2 findings: `CHATGPT-R1` (P1)
- P3 follow-ups: none at this stage
- Known limitations: no substantive correctness judgement has been made on `dafe321...`; only review-governance integrity has been assessed.
- Product-owner decisions: none required. Claude must republish a current exact-SHA handoff before independent technical review can proceed.

`VERIFIED` is not available for the stale target. No implementation files were modified by this reviewer; this review artifact was published on the separate `review/chatgpt-dafe321` branch.