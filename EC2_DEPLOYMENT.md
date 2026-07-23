# EC2 Deployment

> For actual deployment instructions (Terraform, commands, cost model), see
> [terraform/README.md](terraform/README.md) — that's the source of truth.
> This document covers the cold-start analysis and what's been fixed vs. what
> remains open as future scaling work.

---

## Status: fixes 1–2 done, fixes 3–6 not built

This started as a pre-deployment bottleneck analysis. Since then, the fixes
below were implemented and a real deployment now exists (single EC2 instance,
Terraform-managed — see `terraform/`). Two of the six originally-proposed
fixes are done; the other four describe a materially different, more
elaborate architecture (FastAPI service, ECS, SQS, S3-backed I/O, ALB) that
was never built and isn't currently planned. Treat those as options for a
future scaling phase, not a description of what's deployed.

---

## What was actually the bottleneck (measured on real infra, not estimated)

The original analysis (from local `pipeline.log`) identified NCCI JSON
parsing and `compliance.db` rebuilding as the dominant cold-start cost
(~9 min combined). Once deployed to EC2 and measured end-to-end, that
turned out to be **wrong about which part dominates**:

| Component | Original estimate | Actually measured on EC2 |
|---|---|---|
| NCCI JSON parse (`CodeReferenceDB`) | 6m 11s | **Eliminated** — now queries `compliance.db` directly (Fix 2) |
| `ComplianceDataStore` build (SQLite) | 2m 47s | **~12 sec**, even from scratch — the local 15-20 min figure was Docker Desktop/macOS VM I/O overhead, not inherent to the workload |
| Qdrant collection build (~94K ICD10/CPT/HCPCS codes, CPU embedding) | not separately measured | **~60-90 min** — this is the real dominant cost, at ~30 codes/sec on `r6i.xlarge` CPU |
| FastEmbed + GLiNER model load | 49s (repeated every run) | ~1-5 sec once cached (`hf_cache` volume) |

The practical upshot: the original 6-fix roadmap under-weighted Qdrant
embedding entirely (a >1-hour cost) while over-weighting `compliance.db`
(a 12-second cost once measured on real infrastructure instead of a
resource-constrained local Docker Desktop VM). This is why "fix the thing
you measured on your laptop" and "fix the thing that's actually slow in
production" aren't always the same list.

---

## Fixes implemented

### Fix 1 — Persist `compliance.db` across container restarts — done, differently than proposed

Originally proposed as a single bind mount:
```yaml
- ./data/compliance.db:/app/data/compliance.db
```
This has a real Docker gotcha: bind-mounting a path that doesn't exist yet
on the host creates a **directory**, not a file, breaking `sqlite3.connect()`
on first boot. Implemented instead as a named volume over the whole
`/app/data` directory:

```yaml
volumes:
  - app_data:/app/data
```

Named volumes auto-populate from the image's baked-in content on first use
(so `data/codes/*.json` etc. are still there), and everything written back —
`compliance.db`, `result_cache/` — persists across container recreation,
restarts, and instance stop/start. See `docker-compose.yml`.

### Fix 2 — Read NCCI from SQLite, not JSON — done

`CodeReferenceDB` (`app/rag/code_reference.py`) previously parsed
`ncci_data.json` (475MB, 1.7M pairs) into an in-memory Python dict —
measured at **710MB RSS** — duplicating data already indexed in
`compliance.db`. `check_ncci()` now queries `compliance.db` directly via a
short-lived read-only SQLite connection per call, matching the original
non-date-aware match semantics exactly (deliberately not switched to
`ComplianceDataStore.ncci_pair()`'s date-aware lookup, to avoid silently
changing compliance-relevant matching behavior).

---

## What's actually deployed (not FastAPI/ECS — see terraform/)

The real deployment is a **single EC2 instance** (`r6i.xlarge`) running the
existing `docker-compose.yml` stack via `user_data`, not the FastAPI/ECS
target architecture originally sketched below. Two-phase, not
request/response:

1. **Setup** (`--setup-only`) — loads everything expensive once: GLiNER,
   embeddings, Qdrant collections, `compliance.db`. ~60-90 min from scratch,
   persisted in named Docker volumes (`app_data`, `hf_cache`, `qdrant_data`)
   on the EBS root volume — a one-time cost per instance, not per note.
2. **Process** (`process-notes.sh`) — reuses Phase 1's state, no rebuild.
   ~3-4 min/note (LLM-call-bound — Vision extraction + NER + 4-pass coding),
   run manually or auto-triggered by `note-watcher.service` (systemd +
   `inotify`) watching `doctors_notes/` for new PDF uploads, debounced 15s.

Also running on the instance: Claude Code CLI (`claude`), authenticated via
the same `ANTHROPIC_API_KEY` pulled from Secrets Manager.

Full details, commands, and cost model: [terraform/README.md](terraform/README.md).

---

## Not built — future scaling options

The rest of the original roadmap describes a genuinely different, more
horizontally-scalable architecture. None of this is built or currently
planned; it's here as a reference for if/when single-instance, serial
processing stops being enough (e.g. need to process many notes concurrently,
need sub-10-second cold starts, need to survive instance loss without a
~60-90 min rebuild).

### Option A — Bake `compliance.db` + model weights into the Docker image at build time

Instead of building on first `--setup-only` run and relying on volume
persistence, run the build during `docker build` (CI pipeline) so any fresh
instance — new EC2, auto-scaling event, blue/green swap — starts warm with
no rebuild and no volume dependency:

```dockerfile
RUN python -c "\
from app.compliance.datastore.store import ComplianceDataStore; \
ComplianceDataStore().build_or_load(); \
print('compliance.db baked into image layer')"
```

Would still need Qdrant embeddings solved separately (Option C) since that's
the actual dominant cost, not `compliance.db`.

### Option B — Long-running FastAPI service instead of batch script

`pipeline.initialize()` runs once on container start; notes served as
`POST /code-note` HTTP requests instead of a filesystem-watch-triggered
batch script. Useful if you need synchronous request/response semantics
(e.g. a UI waiting on a result) rather than the current fire-and-forget
upload → auto-process → JSON-on-disk model.

### Option C — Persistent Qdrant with pre-built collections, decoupled from app instances

Since Qdrant embedding (~60-90 min for ~94K codes) is the actual dominant
cold-start cost, the highest-leverage future fix is ensuring any new app
instance can attach to an **already-built** Qdrant index rather than
rebuilding it — either a persistent Qdrant service with its own EBS volume
(shared across horizontally-scaled app instances), or a pre-built Qdrant
snapshot restored on boot.

### Option D — S3-backed notes/results + SQS job queue + horizontal workers

Replace local `doctors_notes/`/`output/results/` directories with S3, add
an SQS queue for job distribution, and run N app instances behind it for
parallel note processing. Per-note latency (~3-4 min) is LLM-call-bound and
irreducible per note, so the only way to increase aggregate throughput is
horizontal scaling — this is the mechanism for that, not needed until
serial processing is provably the bottleneck.

**Not needed at current scale:** RDS, DynamoDB, ElastiCache, EFS. No
relational data — results are self-contained JSON documents.
