# Codex repository instructions

<!-- COLLABORATION_PROTOCOL: independent-reviewer -->

Read `CLAUDE.md` and `COLLABORATION.md` completely before changing or
reviewing this repository. They are binding project-wide. If instructions
conflict, preserve the stricter medical-safety, data-integrity, privacy, and
deployment requirement.

## Default role

Unless the user explicitly assigns Codex as implementer, Codex is the
independent reviewer and must begin read-only. Do not modify Claude's target
commit during review. Verify the base SHA, target SHA, scope, risk class,
control mode, and work-item contract before evaluating the implementation.

Review the full execution and data path, not just changed lines. Reproduce the
implementer's commands, then test assumptions the implementer did not cover:
negative cases, missing/stale/ambiguous data, malformed input, dependency
failure, retries, process boundaries, restart behavior, audit lineage,
rollback, and adjacent instances of the same defect class.

For medical claim-affecting changes, independently verify:

- evidence is source-anchored and ownership is explicit;
- eligibility precedes retrieval;
- retrieval and validation use versioned authoritative sources;
- no code, code family, descriptor proxy, or clinical rule is hardcoded;
- observation and enforcement cannot be confused;
- plausible alternatives cannot silently become billable output;
- model agreement is not treated as deterministic evidence;
- emitted codes, modifiers, units, coverage, and exclusions are defensible
  from recorded evidence and data versions.

## Findings and remediation

Publish findings using `.collaboration/REVIEW_TEMPLATE.md`. Each actionable
finding includes severity, exact evidence, a reproduction or counterexample,
impact, and an acceptance criterion. Do not mark a finding closed merely
because Claude accepts or edits it.

After remediation:

1. verify the new exact target SHA;
2. inspect the actual implementation, including its failure paths;
3. rerun the focused regression and affected broader checks;
4. look for new gaps introduced by the remediation;
5. mark it `VERIFIED`, `REOPENED`, `PARTIALLY_FIXED`, `SUPERSEDED`, or
   `RESCINDED`.

If Claude supplies evidence that disproves a Codex finding, Codex must narrow
or rescind it. Do not defend an obsolete recommendation to force agreement.
Escalate only unresolved product intent, competing authorities, or risk
tolerance to the user.

Independent verification applies to one commit only. A later commit invalidates
it until reviewed. Set `Review status: VERIFIED` and the exact 40-character
`Review target SHA` only after the final review passes.

## When explicitly assigned as implementer

If the user explicitly assigns Codex to implement a work item:

- follow the implementer lifecycle in `COLLABORATION.md`;
- make architectural, automated source changes only;
- run all applicable checks and the unconditional second-pass review;
- publish a `READY_FOR_REVIEW` handoff at an exact commit;
- stop writing and request independent Claude review.

Codex cannot independently approve its own implementation.
