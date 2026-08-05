# Engineering conventions

## No patches or quick fixes

All fixes must be **architectural, automated, and complete** — not manual
workarounds applied by hand to a running system.

Concretely:

- If a bug is found in deployed infrastructure (e.g. a broken `user_data`
  script, a missing package, a misconfigured service), fix the **source**
  (Terraform, Dockerfile, docker-compose.yml, provisioning scripts) and
  redeploy from that source. Do not SSH in and hand-patch a live instance
  and call it done — a hand-patch that isn't reflected in the source will
  silently vanish the next time infrastructure is recreated, and leaves no
  record of what was actually fixed.
- Verification is allowed to be manual (SSH in, tail a log, run a command
  to check status) — the *fix* itself must not be.
- "It works now" is not the bar. "It works from a clean `terraform apply` /
  fresh build with zero manual intervention" is the bar.
- If a live-system test reveals a fix works, port that fix back into the
  source of truth and prove it again from scratch before considering the
  work done.

This applies to infrastructure, deployment scripts, and application code
alike.

## No hardcoded medical codes in fixes

Fixes must **never** encode specific ICD-10/CPT/HCPCS/modifier values (or
prefix ranges, or code-family lists) directly in Python. Medical code sets
change — quarterly NCCI/MUE updates, annual CPT/HCPCS revisions, LCD
updates — and a hardcoded list silently goes stale the moment the
authoritative source changes, with no error, no signal, just quietly wrong
claims.

Concretely:

- Every code-dependent decision must be resolved by **querying the
  authoritative data** already loaded into `compliance.db` /
  `CodeReferenceDB` (from `data/codes/*.json`, sourced from real AMA/CMS
  files with provenance) — never by writing `if code in {"...", "..."}` or
  `if code.startswith(("70", "71", ...))` for anything code-family-shaped.
- If the data needed to make a decision correctly isn't in the authoritative
  source yet (e.g. a CPT-vs-HCPCS applicability flag, a bilateral-vs-
  unilateral descriptor), that's a signal to either pull in a real source
  file with that field, or query an existing field that already carries the
  distinction (e.g. `global_period` to distinguish a procedure from a
  diagnostic test) — not to approximate it with a hardcoded list that
  happens to match today's code set.
- Two real examples from this codebase's history: `_VALID_CPT_MODIFIERS`
  (a hardcoded 25-modifier allowlist) and `IMAGING_PREFIXES` (a hardcoded
  CPT-section prefix tuple used as a proxy for "not a billable procedure")
  were both replaced with queries against real data (`modifiers.json`'s
  `systems` field; `global_period`'s `XXX`/`000`/`010`/`090` values) —
  the hardcoded versions were already wrong for codes outside the narrow
  set the original author tested against (global-period modifiers 24/58/
  78/79; the 93xxx vascular-study code family).
- This complements — doesn't replace — the existing `check_no_hardcoding`
  CI guard; that guard catches literal code lists, this rule covers the
  broader pattern (prefix ranges, section heuristics, anything that
  approximates a real, queryable field with a fixed proxy).

## New deterministic rules are config, not code

Validation rules are instances of a small set of generic mechanics
("templates") implemented once in `app/validation/rule_engine.py`:
`context_gate`, `tiered_family_arbitration`, `companion_completion`,
`residual_secondary_demotion`. Individual rules live as versioned config in
`data/rules/validator_rules.json` — descriptor grammar, language lexicons,
context regexes, message text, authority citations. Never medical codes
(guarded by `tests/test_rule_engine.py`).

Concretely:

- When a review cycle exposes a new error class, first ask which existing
  template it instantiates. If one fits, the fix is a new rule-pack entry
  plus regression tests — no new Python.
- Only add a new `_check_*` method when the mechanic itself is genuinely
  new; then consider whether it generalizes into a template so the NEXT
  rule of its kind is config.
- Every rule must cite its authority (CPT/ICD-10-CM guideline, CMS manual
  chapter, NCCI policy) in the pack — a rule that can't name its source is
  a guess, not a rule.
- Rule behavior is regression-tested through the validator suite
  (`tests/test_validator_checks.py`); the engine plumbing through
  `tests/test_rule_engine.py`; pack consumption through the rule-coverage
  guard.

## Always review new fixes for bugs or gaps

A fix is not done when it compiles and deploys — it is done after a
deliberate second pass looking for the bugs and gaps in the fix itself.

**The deliberate pass is unconditional and self-initiated.** It runs after
EVERY fix, patch, refactor, config change, or deployment — no exceptions,
and never only because someone asked "did you check for bugs?". Syntax
checks and a green test suite are not the pass: tests prove the code does
what the tests say, the pass hunts for what the fix's author didn't think
to test. Declaring work complete without having done the pass is the
defect. Before declaring a fix complete:

- **Re-read the full path the fix depends on**, not just the lines changed.
  Verify each assumption against the actual code: does the retry loop
  classify this new exception as retryable? Does the caller handle what the
  fix now raises? Does the log line the monitor greps for actually exist?
- **Check the failure paths of the fix, not just its happy path.** What
  happens when the fix's own action fails — does it fail loudly, or fall
  back to something silent and worse (empty result, hung wait, stale data)?
- **Look for adjacent instances of the same bug class.** A missing timeout
  on one client usually means a missing timeout on every client; a silent
  fallback in one parser usually has siblings.
- **Check boundary interactions**: process boundaries (is what you raise
  picklable across a multiprocessing Pool?), retry boundaries (who owns
  retries when both the SDK and the caller have them?), deploy boundaries
  (do already-running workers see the new code, and is that safe?).

Three real examples from this codebase's history, all found only because a
post-fix review was (belatedly) done: the vision-extraction retry fix
initially covered API errors but left the silent-empty-JSON fallback intact
(which fired live the same night); the consistency-worker parallelization
raised SDK exceptions that cannot be unpickled across the Pool boundary —
crashing the result-handler thread and hanging the entire batch for hours;
and the dynamic convergence-cycle change shipped with its patience guard
keyed on the CLEAN count — which a single-note scope can never raise until
fully done, silently reinstating the exact fixed cycle cap the change
existed to remove (plus a `--resume` seeding bug that could re-run every
already-CLEAN note). The third was caught only when the user asked whether
the review had been done — which is precisely the failure the
"unconditional and self-initiated" requirement above exists to prevent.

## Claude–Codex collaboration

<!-- COLLABORATION_PROTOCOL: primary-implementer -->

`COLLABORATION.md` is the project-wide collaboration contract. Read and apply
it to every work item, not only the evidence/service graph phases. Unless the
user explicitly switches roles, Claude is the primary implementer and Codex is
the independent reviewer.

Before writing code, Claude must record the objective, non-goals, risk class,
control mode, claim-affecting status, base commit, invariants, authoritative
sources, acceptance/negative cases, affected boundaries, and rollback plan.
Use the issue or pull-request contract defined by the repository.

Claude's implementation cycle is:

1. trace the complete current execution and data path;
2. implement the smallest complete architectural source change;
3. add focused positive, negative, failure, boundary, and adjacent-class tests;
4. run repository guards and affected broad tests;
5. perform the unconditional post-fix review required above;
6. commit and push an exact target SHA;
7. publish `.collaboration/HANDOFF_TEMPLATE.md` as `READY_FOR_REVIEW`;
8. stop writing until Codex returns its independent review.

Claude must respond to every review finding as `ACCEPTED`, `DISPUTED`, or
`SUPERSEDED`, with evidence. If accepted, implement the remediation, rerun
checks and the post-fix review, publish a new exact SHA, and return it to Codex.
The finding remains open until Codex reviews the implementation and verifies
it. If Codex establishes a new issue during remediation review, repeat the
cycle.

If Claude can disprove a Codex recommendation with executable evidence or a
controlling authoritative source, dispute it explicitly; Codex is expected to
narrow or rescind it. Do not implement a known-worse change merely to create
agreement. Escalate genuine product choices or unresolved competing authority
to the user.

New claim-affecting controls must declare one explicit mode:
`OBSERVATIONAL`, `ENFORCED_FAIL_CLOSED`, or `DISABLED`. Observation must
not affect claim output. Enforcement must be tied to a documented invariant,
auditable, reversible, and independently reviewed.

Independent review is valid only for the exact reviewed commit. Claude must not
mark a pull request ready unless the body has `Review status: VERIFIED`, its
`Review target SHA` equals the current head, and all applicable gates pass.
Any new commit resets review to `PENDING`.
