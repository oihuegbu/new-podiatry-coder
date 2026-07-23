# Cost Analysis — Per-Note Unit Economics

Snapshot date: **Jul 17, 2026**. All figures are for the production operating
model: every note runs 3 independent times (`CONSISTENCY_RUNS=3`) through
Claude Opus generation + verification, and unanimity across billing arrays
gates auto-submission. Token profile measured on the 10-note convergence
campaign: **~30k prompt + ~25k completion tokens per run** (completion
includes thinking tokens).

---

## 1. Before — interactive API, no caching discounts

Straight Opus rates ($15/M input, $75/M output), every call streamed
interactively, no cache reuse across the 3 consistency runs.

| Component | Calculation | Cost |
|---|---|---|
| Output tokens (3 runs) | 3 × 25k × $75/M | $5.63 |
| Input tokens (3 runs) | 3 × 30k × $15/M | $1.35 |
| EC2 + EBS share (~10 min wall clock on `r6i.4xlarge` @ ~$1.03/hr, ÷ notes in flight) | | ~$0.02 |
| **Total marginal cost per note** | | **~$7.00** |

Wall clock: **~7–12 min per note** (3 runs in parallel).

## 2. Current — Batch API + prompt caching (implemented Jul 17, 2026)

Two of the three cost levers are now live in the pipeline:

- **Anthropic Message Batches API** (`ANTHROPIC_USE_BATCH=1`, default on):
  every call is a single-request batch — **50% off all tokens**, identical
  model and output distribution.
- **Prompt caching**: cache breakpoints on the system prompt (shared across
  every note in a batch) and the user turn (note + RAG context + rendered
  page images — identical across the 3 consistency runs of one note). Cache
  reads bill at 10% of the input rate; the batch discount stacks on top.

| Component | Calculation | Cost |
|---|---|---|
| Output tokens (3 runs) | 3 × 25k × $37.50/M (batch) — caching cannot reduce output/thinking tokens | $2.81 |
| Input tokens, run 1 | ~30k mostly cache-write @ 125% × batch rate | ~$0.28 |
| Input tokens, runs 2–3 | ~80% of prefix read from cache @ 10% × batch rate; verify pass's user turn differs per run so only its system prompt caches | ~$0.13 |
| EC2 + EBS share | | ~$0.02 |
| **Total marginal cost per note** | | **~$3.20** (range $2.90–3.50 with note complexity and cache-hit variance) |

Wall clock: **~30–90 min per note typical** (batch queue latency; each call
bounded by a 2-hour retryable ceiling). Still far inside the 24–72 h
turnaround of human coding services. `ANTHROPIC_USE_BATCH=0` restores
~7-minute interactive runs at full price.

> These are projections from the measured token profile. Validate by reading
> the `cache_read_tokens` / `cache_write_tokens` now surfaced in every
> result's `api_usage` after the next batch run.

## 3. Recommended customer pricing

**$4–6 per successfully coded note** (tiered by volume), or a hybrid
**$299–499/mo platform fee + $3/note**. At a typical 300–600
encounters/month podiatry practice this lands at $1,500–3,500/mo — priced
like per-chart coding outsourcing ($3–12/chart) while also delivering
deterministic NCCI/LCD/MUE scrubbing, the 3-run repeatability gate, per-code
audit rationale, and the denial feedback loop that those services don't.

## 4. Profit (or loss) per note at each price point

Gross profit = price − marginal COGS. Fixed costs (support, compliance,
sales, idle infra) are excluded.

| Price per note | Before (COGS $7.00) | Current (COGS $3.20) |
|---|---|---|
| $4 | **−$3.00 loss** (−75% margin) | **+$0.80 profit** (20% margin) |
| $5 | **−$2.00 loss** (−40% margin) | **+$1.80 profit** (36% margin) |
| $6 | **−$1.00 loss** (−17% margin) | **+$2.80 profit** (47% margin) |

**Bottom line:** before the levers, every note sold at the recommended price
lost $1–3 — charging per note was underwater at any market-acceptable price.
After Batch API + caching, every tier is gross-margin positive, but margins
are thin relative to the **60–70%** a sustainable software business needs
once fixed costs are loaded in.

## 5. Remaining lever to reach target margin

**Sonnet-class generation with Opus verification** (not yet applied): tier
the generation passes down to a Sonnet-class model and keep Opus only for
the verify pass. Projected COGS: **~$1.50–2.00/note**, giving **~60–70%
gross margin at $5/note**. Requires an accuracy regression run (the frozen
gold-standard benchmark + consistency gate) before switching production
traffic.

| Scenario | COGS/note | Margin @ $5/note |
|---|---|---|
| Before (interactive Opus) | ~$7.00 | −40% |
| Current (Batch + caching) | ~$3.20 | 36% |
| + Sonnet/Opus tiering | ~$1.50–2.00 | 60–70% |

## 6. One-time capital context (not part of marginal COGS)

Reaching the 90% auto-submission threshold on this corpus took ~10–11 h of
productive iteration (~$35 API + ~$2 EC2 + ~$4 agent/IDE time per note ≈
**~$41/note all-in on the 10-note campaign**). This is one-time layer-building
capital, not a recurring cost: every deterministic layer generalizes, so
future notes that hit only covered error classes cost just the marginal
figures above.
