# READY_FOR_REVIEW handoff

## Identity

- Work item: Remediate Codex review findings F4 (measurement dimension-safety) and F5 (distinctness-aware dedup)
- Implementer: Claude
- Independent reviewer: Codex
- Branch: claude-medical-coder
- Base SHA: 0a74a3ced3c27450589a56fc03c3516d6ab62580
- Target SHA: 5186c189315fac851cf321796caa0d2a050e6538  (F4/F5 code+tests; this docs-only line correction is a trailing commit)
- Risk class: A (claim-affecting: candidate selection + claim-line count)
- Control mode: ENFORCED_FAIL_CLOSED (corrections to already-enforced selection/dedup paths; the comparison now fails closed on incompatible/ambiguous input)
- Claim-affecting: Yes

## Objective and scope

- Objective: (F4) the candidate-elimination AND specificity logic must compare a documented measurement to a descriptor interval ONLY when they share a dimension unambiguously; an incompatible/unitless/ambiguous measurement must neither eliminate nor add specificity. (F5) duplicate-intent merging must not suppress an explicitly distinct service.
- Non-goals: the rule-coverage guard, validator, datastore, and fail-closed-control work currently uncommitted in the tree (Codex/collaborator lane) — NOT touched. The hard retrieval boundary (F1) and certificate binding (F7) are separate follow-up work items.
- Changed paths: claude_coder/measurement.py, claude_coder/resolution.py, claude_coder/eligibility.py, tests/test_measurement.py, tests/test_eligibility.py
- Runtime/deployment boundaries: pure in-process logic in the `app-app` image; no data, deploy, or infra change.

## Invariants and authority

- Invariants: a measurement comparison is dimension- and role-safe (never unit-blind); a documented distinct service is never merged away; grouping/dedup never fabricates or drops a claim line silently.
- Authoritative sources: descriptor grammar (ontology intervals) + generic physical unit conversions (measurement.py); no code/family/descriptor proxy introduced. No-hardcoded-medical-code: `check_no_hardcoding` passes (137 modules).
- Missing/ambiguous-data behavior: `measurement_for_dimension` returns None for zero OR multiple same-dimension matches → no comparison (fail-closed). Incompatible unit/dimension → no comparison.

## Implementation

- F4: `measurement.py` gains `measurements_of` (all typed measurements) and `measurement_for_dimension` (the UNIQUE measurement of a dimension, else None). `resolution._ranked` replaces the bare-`measure` elimination + specificity with one dimension-guarded `_in_range` tri-value used for BOTH: None→no action, True→specificity, False→eliminate.
- F5: `eligibility.merge_duplicate_intents` rewritten with `_distinct_intents` guard — any distinctness fact (SEPARATE_FROM / distinct site/session/objective) or differing performer/approach/site PROHIBITS the merge; a true merge now unions distinctness facts + decisions (not just the first intent's).
- Alternatives rejected: keeping specificity unit-blind (the original defect); dict-keyed merge (cannot keep two same-key-but-distinct intents).
- Audit/provenance: none changed. Rollback: revert this commit (pure logic).

## Verification

| Command | Result | Evidence |
| --- | --- | --- |
| focused tests | PASS | `pytest tests/test_measurement.py tests/test_eligibility.py -q` → 36 passed |
| negative/failure tests | PASS | F4: length-vs-area gives no specificity/elimination; role-ambiguity → None. F5: SEPARATE_FROM / differing performer prohibit merge |
| repository guards | PASS (no-hardcoding) | `check_no_hardcoding.py` → clean (137 modules) |
| full affected suite | PASS | `pytest tests/ -q` → 723 passed, 3 skipped |
| clean build/deploy | n/a | pure logic; no deploy change |

Note: `check_rule_coverage.py` is RED in the tree, but for pre-existing data/manifest dispositions (statute field, stale NCCI manifest keys) that the collaborator is actively fixing in uncommitted work — orthogonal to F4/F5, which touch no data manifest.

## Unconditional self-review

- Full path re-read: `_ranked` elimination+specificity now share `_in_range`; `merge_duplicate_intents` scan + `_distinct_intents`.
- Failure paths: unknown unit / unitless / ambiguous role → `_in_range` None → no eliminate, no specificity (fail-closed). convert() None → no action.
- Adjacent instances: `resolution.py:85` presence-check still uses bare `measurement_of` — intentionally out of scope (verification trigger, not a selection comparison); flagged for a possible follow-up.
- Boundaries: pure logic, no new exceptions/IO; live retrocalcaneal note unchanged (facts carry no size/area measurement).
- New gaps found and resolved: initial F5 anchor mismatch (stray `support` line) corrected before applying.

## Known limitations

- F4 does not yet parse a descriptor's specific role axis (width vs depth): two same-dimension measurements are treated as ambiguous (no comparison) rather than role-matched — safe, but a future descriptor-role parser could disambiguate.
- Does NOT address F1 (hard retrieval boundary), F3 (ownership model / UNKNOWN→hold), F7 (certificate binding) — separate work items; F1/F8 overlap the collaborator's in-progress fail-closed-controls lane.

Review status: PENDING
