# Claim Scrubber — Implementation Plan & Progress Tracker

**Goal:** Extend the podiatry medical coder into a full **clean-claim scrubber** that runs every
claim line through the 12-filter compliance gauntlet. Only fully clean claims pass; anything that
fails any filter is routed to a review queue with the specific reason + denial risk.

**Hard requirements**
- 100% **dynamic / data-driven** — NO hardcoded code lists in logic. All medical knowledge lives in
  versioned, effective-dated data tables. (Client will test with many other documents.)
- Every datastore lookup is **date-of-service aware** (active-for-DOS).
- Engine behavior: **CLEAN → pass**, otherwise **REVIEW** with findings.
- Medicare/CMS-first; payer field kept extensible for commercial payers later.

**Status legend:** ⬜ todo · 🔄 in progress · ✅ done · ⏸️ blocked

---

## Architecture (target)

```
app/compliance/
├── datastore/        ComplianceDataStore (SQLite, effective-dated) + source adapters
├── models.py         Claim, ClaimLine, Finding, ScrubResult
├── agents/           one ComplianceAgent per filter (pure logic, zero hardcoded codes)
├── adapters/         Stedi eligibility(270/271) + prior-auth(278/FHIR)
└── engine.py         ClaimScrubber: build Claim → run agents → gate → ScrubResult
```

Testing: validate agents against the 12 existing `output/results/*.json` fixtures (free, deterministic),
plus a CI grep guard that fails if hardcoded code-list literals reappear in `agents/`.

---

## The 12 filters → agents

| # | Filter | Agent | Phase | Status |
|---|--------|-------|-------|--------|
| 1 | Code validity & specificity | `specificity` | 1 | ✅ |
| 2 | NCCI PTP edits (0/1/9 modifier indicator) | `ncci_ptp` | 2 | ✅ |
| 3 | MUE + MAI (1/2/3) | `mue_mai` | 1 | ✅ |
| 4 | Modifiers (validity, X{EPSU}, conflicts) | `modifiers` | 2 | ✅ |
| 5 | Medical necessity LCD/NCD ICD↔CPT | `medical_necessity` | 3 | ⬜ |
| 6 | Global surgical period (+24/58/78/79) | `global_period` | 1 | ✅ |
| 7 | Frequency / lifetime / duplicate | `frequency` | 2 | ✅ |
| 8 | Add-on code rules | `addon` | 2 | ✅ |
| 9 | Place of Service & provider eligibility | `pos_eligibility` | 2 | ✅ |
| 10 | Prior authorization (278/FHIR) | `prior_auth` | 4 | ✅ |
| 11 | Eligibility & benefit coverage (270/271) | `benefits` | 4 | ✅ |
| 12 | Documentation support | `documentation` | 5 | ✅ |

---

## Phase 0 — Foundation ✅ DONE
- [x] ✅ `compliance/` package skeleton
- [x] ✅ `models.py`: Claim, ClaimLine, Diagnosis, Finding, ScrubResult (Pydantic)
- [x] ✅ SQLite schema (effective-dated) + `ComplianceDataStore` (`data/compliance.db`)
- [x] ✅ Ingestion: ICD-10 (74,719), CPT (11,601), HCPCS (8,928; 820 dirty rows skipped)
- [x] ✅ Ingestion: NCCI PTP (6,714 pairs; 59 junk rows filtered; 0/1/9 indicator parsed)
- [x] ✅ Ingestion: MUE (15,095; **MAI 1/2/3 parsed for 100% of rows** — the bug Kachi flagged)
- [x] ✅ Ingestion: global periods (365) + LCD L36199 (201 dx)
- [x] ✅ `engine.py`: ClaimScrubber + `build_claim()`; pass-through gate
- [x] ✅ Test harness `tests/scrub_fixtures.py` — 11/11 fixtures normalize & run, 0 crashes
- [ ] ⬜ CI grep guard against hardcoded code lists  _(do at Phase 5 hardening)_
- [ ] ⬜ Wire ScrubResult into `pipeline.py` / output JSON  _(deferred to Phase 5 — keep working pipeline untouched until agents proven on fixtures)_

## Phase 1 — High-yield filters ✅ DONE
- [x] ✅ Agent #3 MUE/MAI (MAI 2 = hard wall, 1 = line bypass w/ modifier, 3 = appealable→review)
- [x] ✅ Agent #1 specificity (existence + active-for-DOS + non-billable-header/unspecified detection)
- [x] ✅ Agent #6 global period (E/M in prior surgery's window, +24/58/78/79 bypass)
- [x] ✅ `tests/test_agents.py` — 12/12 adversarial violation tests pass
- [x] ✅ Fixture regression — 11/11 real client docs stay CLEAN (zero false positives)
- [x] ✅ Datastore `_asof()` effective-dated lookup with single-snapshot fallback

## Phase 2 — Structural filters ✅ DONE
- [x] ✅ Agent #2 NCCI PTP — indicator 0=hard edit, 1=modifier-bypassable, 9=skip
- [x] ✅ Agent #8 add-on codes — add-on status derived from CPT descriptor phrasing (data-driven);
        flags add-on billed with no primary. _Precise primary-matching pending CMS Add-On Edit file._
- [x] ✅ Agent #7 frequency / duplicate — duplicate-line detection (site/repeat-modifier aware).
        _Annual/lifetime frequency limits pending MCD/NCD frequency-policy table._
- [x] ✅ Agent #9 POS — POS code validity + facility flag (`pos_codes.json` ref data).
        _Per-code POS payability + provider credentialing pending PFS payability + enrollment source._
- [x] ✅ Agent #4 modifier engine — validity (`modifiers.json`), X{EPSU}-over-59, RT/LT & 50 conflicts
- [x] ✅ Tests — 26/26 adversarial pass; 11/11 fixtures CLEAN (1 helpful non-blocking advisory)

## Phase 3 — Medical necessity coverage ✅ DONE
- [x] ✅ Generalized coverage engine (`coverage_cpt`/`coverage_icd` tables) replacing hardcoded L36199
- [x] ✅ `store.load_coverage_articles()` — MCD Article ingestion entry point (same schema, no logic change)
- [x] ✅ Agent #5 medical necessity (ICD↔CPT coverage check, data-driven)
- [x] ✅ Tested (FAIL on non-qualifying dx, pass on qualifying); fixtures CLEAN
- [ ] ⬜ _Load full MCD Billing & Coding Article bulk file (needs the CMS download)_

## Phase 4 — External (Stedi) ✅ DONE
- [x] ✅ `adapters/stedi.py` — `ClearinghouseAdapter` interface + `StediAdapter` (270/271), config-gated
- [x] ✅ Live sandbox call verified (parses X12 271, returns structured result)
- [x] ✅ Agent #11 eligibility/benefits — graceful degradation when not configured / no member id
- [x] ✅ Agent #10 prior authorization — data-driven `prior_auth_required` table, FHIR-ready
- [ ] ⬜ _Load payer Required-PA list + wire Stedi member data from real claims_

## Phase 5 — Gate, routing, documentation ✅ DONE
- [x] ✅ Agent #12 documentation support (code support + modifier-justification audit)
- [x] ✅ Clean-claim gate (`ScrubResult.finalize()` — CLEAN iff zero FAIL) + review routing w/ reasons
- [x] ✅ Wired ScrubResult into `pipeline.py` (step 6) + `CodingResult.claim_scrub` output field
- [x] ✅ CI guard `tests/check_no_hardcoding.py` — no hardcoded code lists (14 files scanned)
- [x] ✅ Pipeline integration smoke-tested on fixture (CLEAN disposition, JSON serializes)

## ALL 12 FILTERS LIVE ✅ — 40/40 adversarial tests pass, 11/11 fixtures CLEAN, 0 hardcoded lists

### Bundling refinements (per Kachi's 2026-06-17 bundling brief)
- [x] ✅ E/M + same-day procedure → **modifier 25** check (modifier agent): E/M in the CPT surgery
        section's company without 25/57 → FAIL (bundles into the procedure). Radiology/lab excluded.
- [x] ✅ NCCI agent now accepts **25/57** as the valid unbundler for E/M↔procedure pairs (59/X{EPSU}
        for all others); recommendation text is E/M-aware.
- [x] ✅ Both under-billing (missing valid modifier → flag) and over-billing (modifier without
        documented distinct service → doc-audit flag) failure modes covered, working in tandem.

## QA pass (manual, senior-QA) ✅ DONE — 2026-06-22
- Audited: data integrity, build_claim normalization, per-agent boundaries, gate semantics, refresh robustness.
- Engine/agent **logic verified correct** at all boundaries (MUE cap off-by-one, NCCI indicator 9 +
  reversed order, global-period exact window, WARN-only stays CLEAN, multi-FAIL→REVIEW, crashing-agent
  isolation, empty/garbage inputs, multi-format dates).
- **Defect D1 (med):** tests wrote rows into the shared `compliance.db` (MUE drifted 15095→15097).
  Fixed: tests now clean up their inserts; verified stable across repeated runs.
- **Defect D2 (low):** HCPCS ingestion stored duplicate codes (source has many rows/code).
  Fixed: dedup on ingest (8928→8763 unique; code_set 95248→95083; 0 dups remain).
- Post-fix: 40/40 agent tests · 10/10 refresh · 0 hardcoded · 11/11 fixtures CLEAN · DB counts stable.

## Cross-cutting (hardening) ✅ DONE
- [x] ✅ **Refresh layer** `app/compliance/refresh/` — source registry (mirrors the Excel),
        real-format parsers (NCCI/MUE/PFS/POS/MCD, header-driven + junk-filtered), runner with
        cadence + offline ingest, `run_refresh.py` CLI.
- [x] ✅ **History retention** — `ingest_snapshot()` additive + effective-dated; `_asof()` picks the
        snapshot in force on the DOS; idempotent re-ingest; `data_source_version` provenance table.
- [x] ✅ Live CMS download verified from this environment (221 KB fetched); `tests/test_refresh.py` 10/10.
- [x] ✅ Idempotent DB migration so existing stores gain new tables without rebuild.
- [x] ✅ End-to-end pipeline wired (step 6) + cache-hit path also runs the scrubber.

- [x] ✅ **Scheduling shipped** — `deploy/refresh.crontab` (Excel cadence), `deploy/compliance-refresh.{service,timer}`
        (systemd), and `deploy/README.md` (initial load + scheduling + Qdrant/license notes).
- [x] ✅ `load_coverage_articles()` made idempotent (per-policy replace) so repeated refreshes don't duplicate.

### Operational notes (data loading is now a config/run task, not a code task)
- Loaders are built + tested; point `run_refresh.py --source <id>` at the current CMS quarterly file
  (or `--file` a local download) to replace seed data with full datasets:
  full MCD Articles, CMS Add-On Edit (→ exact add-on↔primary), PFS payability/status, Required-PA list.
- Schedule via cron per Excel cadence: MCD weekly · NCCI/PFS quarterly · ICD annual (Oct) / CPT annual (Jan).

---

## Progress log
- _2026-06-17_ — Plan created. Verified env (Py 3.11.9, pydantic 2.5.3, sqlite3, all heavy deps).
  Confirmed data quirks: MUE MAI = first char of `description`; NCCI modifier indicator in
  `description` field; garbage rows filterable by 5-char code validation. Stedi sandbox key verified
  (HTTP 200, valid X12 271). 11 result fixtures available for free deterministic testing.
- _2026-06-17_ — **Phase 0 complete.** Built `compliance/` package, canonical models, SQLite
  ComplianceDataStore (95,248 codes + 6,714 NCCI + 15,095 MUE w/ MAI + 365 global + 201 LCD),
  ClaimScrubber engine + `build_claim()`, fixture harness. 11/11 fixtures normalize & run clean.
- _2026-06-17_ — **Phase 1 complete.** Agents #1 specificity, #3 MUE/MAI, #6 global period.
  Added `_asof()` effective-dated lookup (single-snapshot fallback). 12/12 adversarial tests pass;
  11/11 fixtures stay CLEAN (no false positives).
- _2026-06-17_ — **Phase 2 complete.** Agents #2 NCCI, #4 modifiers, #7 frequency, #8 add-on,
  #9 POS. Added `pos_codes.json` + `modifiers.json` reference data; add-on status derived from CPT
  descriptors; new `addon`/`pos`/`modifier` tables. 26/26 adversarial tests pass; 11/11 fixtures CLEAN.
  **8 of 12 filters now live** (#1,2,3,4,6,7,8,9). Deferred sub-items: add-on primary-matching,
  frequency limits, POS payability/credentialing — all behind data adapters, no logic rewrite needed.
- _2026-06-17_ — **Phase 3 complete.** Coverage engine generalized; agent #5 medical necessity.
  29/29 adversarial tests; fixtures CLEAN.
- _2026-06-17_ — **Phase 4 complete.** Stedi adapter (live sandbox verified); agents #10 prior auth,
  #11 eligibility with graceful degradation.
- _2026-06-17_ — **Phase 5 complete.** Agent #12 documentation; clean-claim gate + review routing;
  wired into pipeline step 6 + `claim_scrub` output; CI hardcoding guard added.
- _2026-06-17_ — **ALL 12 FILTERS LIVE.** 37/37 adversarial tests pass · 11/11 real client docs CLEAN
  (1 advisory) · 0 hardcoded code lists · pipeline integration smoke-tested.
- _2026-06-17_ — **Hardening complete.** Built refresh layer (registry + parsers + runner + CLI),
  history retention with effective-dated snapshots, idempotent migration. `tests/test_refresh.py`
  10/10; live CMS fetch verified (221 KB). Wired scrubber into `pipeline.py` step 6 + cache path.
- _2026-06-17_ — **Single authoritative verdict.** Scrubber is now the source of truth:
  `final_disposition` (CLEAN/REVIEW) + `final_summary`; legacy `auto_coding_tier` derived from it
  (AUTO if CLEAN else REVIEW) so they can't contradict; blocking findings surfaced as review reasons.
  All regression green (37/37 · 10/10 · 0 hardcoded · 11/11 fixtures CLEAN).
- _2026-06-17_ — **LIVE END-TO-END VERIFIED (Sonnet).** Full pipeline run on NOTE_01:
  PDF→Vision→NER(28 entities)→RAG→Sonnet 4-pass coding→validation→12-filter scrubber. Output JSON
  carries `claim_scrub` = {disposition: CLEAN, 0 findings}. Made model+effort config-driven
  (`CLAUDE_MODEL`/`CLAUDE_EFFORT`) — Sonnet rejects the Opus-only "xhigh" effort; removed the two
  hardcoded `claude-opus-4-7`/`xhigh` literals. (Original keys were expired; Kachi supplied a fresh one.)
