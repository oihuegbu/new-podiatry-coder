# Independent review

## Target integrity

- Reviewer:
- Implementer:
- Work item:
- Base SHA:
- Reviewed target SHA:
- Current head SHA:
- Scope matches contract: Yes / No
- Reviewer remained read-only: Yes / No

Stop if the reviewed target and current head differ.

## Independent interpretation

- Intended objective:
- Claim/safety invariant:
- Risk class and control mode assessment:
- Controlling authoritative sources:

## Review passes

- [ ] Exact diff and unrelated-file check
- [ ] Full execution and caller path
- [ ] Authoritative-data provenance/effective-date path
- [ ] Missing, stale, empty, ambiguous, and conflicting data
- [ ] Negative claim and plausible-alternative cases
- [ ] Exception, timeout, retry, process, restart, and rollback boundaries
- [ ] Audit lineage and observation/enforcement separation
- [ ] Adjacent instances of the same defect class
- [ ] Reproduction of implementer commands
- [ ] Additional adversarial tests
- [ ] Clean build/deployment path when applicable

## Findings

Repeat this block for each finding.

### FINDING-ID — P0 / P1 / P2 / P3 — title

- Status: OPEN
- File/line or data boundary:
- Evidence:
- Reproduction or counterexample:
- Claim, safety, operational, or audit impact:
- Controlling authority or invariant:
- Acceptance criterion:

If there are no findings, state “No actionable findings” and list the residual
risks that were examined.

## Remediation verification

Repeat after the implementer responds.

### FINDING-ID

- Implementer response: ACCEPTED / DISPUTED / SUPERSEDED
- Remediation target SHA:
- Reviewer disposition: VERIFIED / REOPENED / PARTIALLY_FIXED / SUPERSEDED / RESCINDED
- Commands rerun:
- Evidence:
- New or remaining gaps:

## Final disposition

- Review status: PENDING / VERIFIED / ESCALATED
- Verified target SHA:
- Open P0–P2 findings:
- P3 follow-ups:
- Known limitations:
- Product-owner decisions:

`VERIFIED` applies only when the target SHA is the exact current head, all
applicable gates pass, and no blocking finding remains. A new commit returns
the review to `PENDING`.
