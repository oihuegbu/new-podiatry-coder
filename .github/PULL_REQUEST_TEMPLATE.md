## Work item

- **Objective:** <!-- required -->
- **Non-goals:** <!-- required -->
- **Risk class:** <!-- A, B, or C -->
- **Claim-affecting:** <!-- Yes or No -->
- **Implementer:** <!-- required -->
- **Independent reviewer:** <!-- must differ from implementer -->
- **Base SHA:** <!-- exact 40-character SHA -->

## Risk and control mode

- **Control mode:** <!-- OBSERVATIONAL, ENFORCED_FAIL_CLOSED, or DISABLED -->
- **Rollback/disable strategy:** <!-- required -->
- **Affected runtime/deployment boundaries:** <!-- required -->

## Invariants and authorities

- **Invariant:** <!-- required -->
- **Authoritative sources:** <!-- source, version/effective date, citation -->
- **Missing/stale/ambiguous-data behavior:** <!-- required -->
- **No-hardcoded-medical-code evidence:** <!-- required -->

## Verification

<!-- Give exact commands and results. -->

- **Focused tests:** <!-- required -->
- **Negative/failure tests:** <!-- required -->
- **Repository guards:** <!-- required -->
- **Full affected suite:** <!-- required -->
- **Clean build/deploy:** <!-- required or explain not applicable -->

## Self-review and handoff

- **Handoff status:** READY_FOR_REVIEW
- **Target SHA:** <!-- exact 40-character SHA -->
- **Full-path re-read:** <!-- required -->
- **Failure and boundary review:** <!-- required -->
- **Adjacent defect-class review:** <!-- required -->
- **Known limitations:** <!-- required; use None only after deliberate review -->

## Independent review

<!--
Keep PENDING while the PR is a draft or after any new commit.
Only the independent reviewer changes these fields to VERIFIED and the exact
current head SHA after completing .collaboration/REVIEW_TEMPLATE.md.
-->

- **Review status:** PENDING
- **Review target SHA:** PENDING
- **Open P0-P2 findings:** <!-- required; use None only when independently verified -->
- **Review evidence:** <!-- review comment/report link or concise result -->

## Final checklist

- [ ] Scope matches the work-item contract
- [ ] No unrelated files, secrets, PHI, or runtime artifacts are committed
- [ ] Applicable checks pass with exact results above
- [ ] The unconditional post-fix review is complete
- [ ] The independent reviewer reviewed the exact current head
- [ ] This PR remains draft unless Review status is VERIFIED
