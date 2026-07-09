# Compliance data — deployment & refresh

The 12-filter scrubber reads all its rules from `ComplianceDataStore` (SQLite,
`data/compliance.db`). This directory automates keeping that data current from
the authoritative CMS/AMA sources mapped in `Claimly_Compliance_Data_Sources.xlsx`.

## 1. Initial full data load

The repo ships with seed/current snapshots so the system runs out of the box.
To load the **full** authoritative datasets, point the (already-tested) loaders
at each source's current file:

```bash
# online (downloads the current quarterly/weekly file from CMS):
python run_refresh.py --source ncci_ptp
python run_refresh.py --source mue
python run_refresh.py --source pfs_global
python run_refresh.py --source mcd_articles      # → medical-necessity ICD↔CPT
python run_refresh.py --source hcpcs

# offline / air-gapped (ingest a file you downloaded yourself):
python run_refresh.py --source mue --file ./MCR_MUE_Practitioner.csv --effective-from 2026-04-01

# preview without writing:
python run_refresh.py --all --dry-run

# provenance of every snapshot ever ingested (history is retained):
python run_refresh.py --history
```

Each ingest is **idempotent** (re-running a quarter is a no-op) and
**history-retentive** (old quarters are kept as effective-dated snapshots, so a
claim is always scrubbed against the rules in force on its date of service).

## 2. Scheduling

Cadence matches the Excel refresh schedule:

| Source | Cadence |
|---|---|
| MCD Articles (medical necessity) | weekly |
| NCCI PTP, MUE, PFS, HCPCS | quarterly (Jan/Apr/Jul/Oct) |
| ICD-10-CM | annual (Oct) |
| CPT (AMA-licensed) | annual (Jan) |

**Option A — cron:** `crontab deploy/refresh.crontab` (edit `APP_DIR`/`PYTHON`).

**Option B — systemd:** install `deploy/compliance-refresh.service` +
`deploy/compliance-refresh.timer`, then `systemctl enable --now compliance-refresh.timer`.

**Option C — Terraform/cloud:** schedule `run_refresh.py --all` as a daily job;
`--all` self-selects only the sources due in the current month.

## 3. Notes
- **Qdrant:** set `QDRANT_URL` to the running Qdrant server in production (the
  local-path store is for dev only and rebuilds slowly on CPU).
- **AMA CPT license:** required to embed CPT descriptors (client holds it).
- **Stedi:** `STEDI_API_KEY` in `.env`; eligibility/prior-auth degrade gracefully
  if absent.
