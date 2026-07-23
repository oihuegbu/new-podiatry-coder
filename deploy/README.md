# Compliance data — deployment & refresh

The 15-filter scrubber reads all its rules from `ComplianceDataStore` (SQLite,
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
| MCE diagnosis edits | annual (Oct, with the IPPS final rule) |
| ICD-10-CM instructional notes | annual (Oct, with the CDC Tabular XML) |

Two annual sources regenerate their JSON via in-repo parsers rather than the
refresh runner (each fiscal year's file layout is stable; both parsers refuse
to write on incomplete parses):

```bash
# MCE (age conflicts, manifestation/unacceptable principal dx): download the
# "Definition of Medicare Code Edits vXX" zip from CMS's MS-DRG page, then:
python tools/parse_mce_definitions.py "Definitions of Medicare Code Edits_vXX.txt"

# ICD-10-CM instructional notes (Excludes1/Includes/codeFirst/useAdditionalCode/
# codeAlso): download the CDC Tabular XML zip, then:
python tools/parse_icd10cm_tabular.py icd10cm-tabular-YYYY.xml
```

The compliance store detects the changed JSONs by checksum on next start and
re-ingests all dependent tables automatically.

One more manual-tier file: `data/codes/mac_jurisdictions.json` maps each MAC
contractor (by name aliases) to the states it adjudicates — this is what
scopes LCDs/Billing & Coding Articles to their jurisdiction instead of
applying every MAC's policies nationwide. Source: CMS "Who are the MACs".
MAC contract awards change rarely (years apart); re-verify the state lists
when CMS announces a jurisdiction transition. The weekly MCD article refresh
carries each article's contractor automatically; only the name→states map
is maintained here.

**Option A — cron:** `crontab deploy/refresh.crontab` (edit `APP_DIR`/`PYTHON`).

**Option B — systemd (native install):** install `deploy/compliance-refresh.service` +
`deploy/compliance-refresh.timer`, then `systemctl enable --now compliance-refresh.timer`.
(Assumes a host venv at `/opt/podiatry-medical-coder` and a `claimly` user.)

**Option B2 — systemd (Docker deployment, e.g. the EC2 instance):** the app has
no host venv there, so the refresh runs inside the container instead:

```bash
sudo cp deploy/compliance-refresh-docker.{service,timer} /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now compliance-refresh-docker.timer
# verify: sudo systemctl list-timers compliance-refresh-docker.timer
# manual run: sudo systemctl start compliance-refresh-docker.service
```

**Option C — Terraform/cloud:** schedule `run_refresh.py --all` as a daily job;
`--all` self-selects only the sources due in the current month.

## 3. Notes
- **Qdrant:** set `QDRANT_URL` to the running Qdrant server in production (the
  local-path store is for dev only and rebuilds slowly on CPU).
- **AMA CPT license:** required to embed CPT descriptors (client holds it).
- **Stedi:** `STEDI_API_KEY` in `.env`; eligibility/prior-auth degrade gracefully
  if absent.
