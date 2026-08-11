# Claude Code operating model (new-podiatry-coder)

Purpose: keep Opus-level reasoning at architectural and safety-critical decision points
while making **git, GitHub issues, and repository files -- not the Claude conversation --
the durable memory** of this project. Claude sessions are disposable working memory.

Repo facts this doc assumes:
- repository: `oihuegbu/new-podiatry-coder`
- **Claude Code itself runs locally** (your laptop) -- it reaches the EC2 box only over SSH to
  run shell/docker/git commands there. It does not launch `claude` on the box for normal work.
- remote working dir (reached over SSH, never logged into directly): `~/work` on the box (the
  deployed app lives at `/opt/app`)
- active branch: `claude-medical-coder`
- engineering contract: `CLAUDE.md`; collaboration contract: `COLLABORATION.md`

## Memory layers

```
CLAUDE.md              permanent engineering rules (always loaded, keep lean)
COLLABORATION.md       Claude/Codex review + risk contract
docs/engineering/*     detailed durable docs (loaded only when relevant)
GitHub issue           current work-item state (the durable task memory)
Claude session         temporary working context (disposable)
commit + handoff       the artifact that outlives the session
```

## The fifteen non-negotiables

1. No Claude session survives between unrelated work items.
2. No normal development past ~150K context (finish the phase, persist to GitHub, restart).
3. No 8+ hour interactive session.
4. One GitHub issue = durable state for one coherent work item.
5. Every Claude commit gets an exact-SHA `READY_FOR_REVIEW` handoff.
6. Claude **exits while independent review happens** (do not idle in-session).
7. Every remediation begins in a **fresh** session (reconstruct state from the issue).
8. Opus makes architectural / claim-affecting decisions and the final adversarial review.
9. Sonnet is for genuinely mechanical work (formatting, docs, boilerplate). For
   claim-affecting code, staying on Opus through implementation is acceptable and often
   preferable -- do not dogmatically downgrade safety-critical work to save tokens.
10. Full-suite testing is never the inner loop -- run the focused file(s) first.
11. Large medical datasets / long logs never enter context wholesale -- query them.
12. Subagents are bounded (<= 2) and read-only for investigation.
13. Three speculative fix/test iterations maximum, then stop and report the blocker.
14. Permanent knowledge belongs in the repo, not the chat.
15. Don't continue development on new commits while a prior SHA is under review.

## Model policy

| Activity | Model |
| --- | --- |
| System / coding architecture, root-cause of a hard bug | Opus |
| Any claim-affecting reasoning or implementation | Opus |
| Initial plan; final adversarial self-review | Opus |
| Formatting, docs, boilerplate tests, mechanical refactor | Sonnet |

> The distinction is "reduce the *amount* of expensive reasoning applied to mechanical
> work", not "reduce reasoning quality on the parts that matter."

## Session lifecycle

1. **Start clean**, from your local Claude Code session, over SSH:
   ```bash
   ssh -i terraform/podiatry-coder-key.pem ec2-user@<box-ip> \
     'cd ~/work && git fetch origin -q && git status --short --branch && git rev-parse HEAD'
   ```
   Confirm branch + HEAD match what the issue's latest handoff/review expects before touching
   anything. If the box is stopped, start it (`aws ec2 start-instances` /
   `aws ec2 wait instance-running`) and re-query its public IP -- it changes on stop/start.
2. **Bootstrap prompt** (small -- do NOT paste prior Claude/Codex history):
   ```
   Work only on GitHub issue #<N> in oihuegbu/new-podiatry-coder (branch claude-medical-coder).
   Treat CLAUDE.md as the engineering contract, COLLABORATION.md as the collaboration
   contract, and issue #<N> as the durable work-item state.
   First: confirm branch+HEAD; read the issue + latest handoff/review; restate the exact
   remediation state; trace ONLY the paths this work item needs; restate invariants +
   acceptance criteria; produce a plan. Do not modify files yet. Do not scan unrelated
   packages or load large datasets. Do not start background loops.
   ```
3. **Plan (Opus)** -> **implement (per model policy)** -> **focused tests** ->
   **Opus adversarial self-review** -> **commit on the box** -> **push from local** ->
   **exact-SHA handoff** -> **exit**.
4. **Codex reviews out of session.** Findings -> **new fresh session** for remediation.

Push happens from local, not the box: the box has no git push credentials and none should be
installed there. From local: `git remote add ec2 ec2-user@<box-ip>:work`, then
`GIT_SSH_COMMAND="ssh -i terraform/podiatry-coder-key.pem" git fetch ec2 claude-medical-coder`,
verify `git merge-base --is-ancestor origin/claude-medical-coder ec2/claude-medical-coder`
(expect fast-forward -- stop and report if not), then
`git push origin ec2/claude-medical-coder:refs/heads/claude-medical-coder`, then
`git remote remove ec2`. This pushes the box's exact fetched objects, not the local working
tree -- the local checkout is not kept in sync and must never be used as a development copy.

## Testing discipline

- Inner loop: `docker exec <test-container> python -m pytest -q tests/test_<target>.py`
- Then the adjacent files; only then the full suite.
- One failure at a time: `... tests/test_x.py::test_name -q`.
- Logs: `rg -n "ERROR|Traceback|SYSTEM_HOLD|UNKNOWN|BLOCKED" <log> | tail -n 100` -- never
  paste a 30k-line log.
- Datasets: print schema + a few representative rows via a tiny script; never read a full
  `data/codes/*.json` (tens of thousands of rows) into context.

## Context hygiene

- 60-90K: check `/context`; continue if on the same task.
- 90-120K: `/compact` with a directed instruction (see below).
- >120K: finish the current logical phase; start no new refactor.
- >150K: stop normal development; persist state to the issue; `/clear` and restart.

Directed `/compact` -- preserve only: issue #, objective, non-goals, base + HEAD SHA, risk
class, control mode, safety invariants, accepted design decisions, authoritative sources,
files changed, exact tests run + exact failures, open reviewer findings, blockers, rollback.
Discard: exploratory discussion, rejected designs, repeated explanations, old tool output,
superseded diffs, verbose logs.

## Three-strike rule

After three unsuccessful implementation/test cycles, STOP and report: failing assertion;
expected vs actual; root-cause hypothesis + supporting evidence; unverified assumptions;
files involved; recommended next diagnostic step. Do not make another speculative change.

## Credential safety (specific to this repo)

Normal work (section above) never launches `claude` on the box at all -- Claude Code runs
locally and only reaches the box over SSH to run shell/docker/git commands, so the box's own
credentials are never in play and this section does not apply to it.

It matters only for the **manual fallback**: a human SSHing directly into the box and running
`claude` interactively there (e.g. exploratory debugging without a laptop on hand). In that
case: the app `.env` (fetched from Secrets Manager) contains `ANTHROPIC_API_KEY` and
`OPENAI_API_KEY`, and `~/.bashrc` sources `~/.claude-code-env` into every login shell on the
box. An interactive Claude launched from such a shell without unsetting those would bill the
app's API account instead of a subscription. Always launch via `~/bin/claude-session`
(`scripts/claude-session.sh`), which unsets those before exec-ing `claude`. Never delete the
app's `.env`.

Note: as of 2026-08-11 the box's `claude` has no subscription login (`claude auth status` ->
`loggedIn: false`) -- unsetting the API keys with no other credential configured means
`claude-session` would need an interactive login (device-code OAuth) completed by a human at
the terminal before first use; it cannot be driven headless/unattended in that state.
