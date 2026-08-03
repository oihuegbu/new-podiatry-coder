# AWS deployment

Single EC2 instance (`r6i.xlarge`, 4 vCPU / 32GB — ~2x headroom over the
~8-9GB measured peak) running the `docker-compose.yml` stack (qdrant + app).
No SQS/ECS — `run.py` loads models once and loops over notes within one
process, so a single right-sized box covers it.

Deployment is two-part, matching how the pipeline actually spends time:

1. **Setup** (`--setup-only`, runs automatically via `user_data`): loads
   everything expensive — downloads GLiNER-BioMed, builds the dense/sparse
   embedding models, embeds all ~94K ICD10/CPT/HCPCS codes into Qdrant, builds
   `compliance.db` from the raw ICD/CPT/NCCI JSON. **~60-90 min from scratch**
   on a fresh instance (the Qdrant embedding step dominates — CPU-bound,
   ~30 codes/sec). This state persists in the `app_data`, `hf_cache`, and
   `qdrant_data` named Docker volumes (survives container recreation,
   container restarts, and instance stop/start — it's on the EBS root
   volume), so it's a one-time cost per instance, not per deploy.
2. **Process notes** (`/opt/app/process-notes.sh`, run on demand, or
   automatically — see below): reuses everything Phase 1 built. No
   re-download, no re-build — just loads the already-cached state into
   memory and processes whatever's in `doctors_notes/`. Takes a few minutes
   per note (dominated by the multi-pass LLM coding calls), not hours.

### Auto-trigger on upload

`note-watcher.service` (systemd, installed by `user_data`) watches
`doctors_notes/` via `inotify` and automatically runs Phase 2 when a new PDF
lands — debounced 15s so a multi-file `scp`/`rsync` triggers one run, not
one per file. It **only ever calls `process-notes.sh`**, never Phase 1 — so
dropping files never triggers a rebuild. Upload with:

```bash
scp -i podiatry-coder-key.pem note.pdf ec2-user@<public_ip>:/opt/app/doctors_notes/
```

Watch it pick the file up: `ssh ... tail -f /var/log/process-notes-auto.log`

### Claude Code

The instance also has the Claude Code CLI installed (`user_data`), pre-authed
via the same `ANTHROPIC_API_KEY` from Secrets Manager — just `ssh` in and run
`claude`.

## Command reference

| | Local (repo root) | EC2 |
|---|---|---|
| **Phase 1 — setup** | `docker compose run --rm app python run.py --setup-only` | `docker compose run --rm app python run.py --setup-only` (auto-runs via `user_data` on first deploy) |
| Phase 1 + force Qdrant rebuild | `docker compose run --rm app python run.py --setup-only --rebuild-index` | same, run manually over SSH after `cd /opt/app` |
| **Phase 2 — process all notes** | `docker compose run --rm app python run.py` | `/opt/app/process-notes.sh` |
| Phase 2 — single note | `docker compose run --rm app python run.py --note X.pdf` | `/opt/app/process-notes.sh --note X.pdf` |
| Phase 2 — skip result cache | `docker compose run --rm app python run.py --no-cache` | `/opt/app/process-notes.sh --no-cache` |

Same underlying flags either way — `process-notes.sh` is just
`docker compose run --rm app python run.py "$@"`. Root README also covers the
non-Docker (`python run.py` directly) form.

## First deploy

```bash
cd terraform
terraform init
cp terraform.tfvars.example terraform.tfvars   # fill in ssh_allowed_cidr + API keys
terraform apply
```

`user_data` runs Phase 1 automatically — Docker install, S3 artifact pull,
secrets fetch, image build, `python run.py --setup-only`. Takes ~10-15 min
after the instance reaches "running".

Watch progress: `ssh -i podiatry-coder-key.pem ec2-user@<public_ip> tail -f /var/log/app-bootstrap.log`

## Processing notes (Phase 2 — repeatable, fast)

Upload new notes to `doctors_notes/` on the instance (or sync via S3/scp),
then:

```bash
ssh -i podiatry-coder-key.pem ec2-user@<public_ip>
/opt/app/process-notes.sh                  # process everything in doctors_notes/
/opt/app/process-notes.sh --note X.pdf     # single note
/opt/app/process-notes.sh --no-cache       # force reprocessing, skip result cache
```

This does **not** re-run Phase 1 — `pipeline.initialize()` still executes
(it has to, to load the models into a fresh process's memory) but hits the
fast path: reads the already-built `compliance.db` and Qdrant collections
instead of rebuilding them, and models load from `hf_cache` instead of
re-downloading.

## Redeploying code changes

`user_data` only runs once at first boot. To ship a code change:

```bash
terraform apply   # repackages + uploads the new source zip to S3
ssh -i podiatry-coder-key.pem ec2-user@<public_ip>
sudo bash -c 'aws s3 cp s3://<bucket>/<new-key> /tmp/app.zip && unzip -o /tmp/app.zip -d /opt/app'
cd /opt/app && docker compose build app
```

If the change touches reference data (`data/codes/*`) or compliance logic,
re-run Phase 1 (`docker compose run --rm app python run.py --setup-only
--rebuild-index`) to rebuild the affected caches; otherwise just use
`process-notes.sh` as normal — it'll pick up the new image.

For an isolated release directory, use the checked-in transactional installer
from the published artifact. It refuses to deploy over an active app batch,
builds before swapping directories, preserves bind-mounted runtime state,
keeps a timestamped rollback tree, refreshes secrets from Secrets Manager,
and leaves the named volumes attached to the explicit Compose project:

```bash
sudo deploy/install_ec2_release.sh \
  s3://<bucket>/<content-addressed-key> \
  /opt/podiatry-autonomy-safety \
  podiatry-autonomy-safety \
  6335 \
  <secret-arn> \
  us-east-1
```

## Rotating secrets (API keys, etc.)

Same `user_data`-only-runs-once problem applies to `aws_secretsmanager_secret_version.app_env` — updating `terraform.tfvars` and running `terraform apply` writes a new secret version, but nothing pushes it onto an already-running instance. After `terraform apply`:

```bash
ssh -i podiatry-coder-key.pem ec2-user@<public_ip>
sudo /opt/app/refresh-secrets.sh
```

This overwrites `/opt/app/.env` from the latest Secrets Manager value. `process-notes.sh` and any new `docker compose run` pick it up automatically. A long-running container already started (e.g. a batch launched with `docker compose run -d`) has its env baked in at creation time — stop/recreate it to pick up the new secret.

## Cost model (stop between runs)

The root EBS volume is not deleted on stop, only compute billing pauses.

```bash
aws ec2 stop-instances  --instance-ids <id> --region us-east-1   # ~$8/mo storage only
aws ec2 start-instances --instance-ids <id> --region us-east-1   # resumes, ~$0.25/hr while running
```

A stopped-then-started instance keeps all Phase 1 state (it's in Docker
volumes on the EBS root volume). `qdrant` has `restart: unless-stopped` so
it comes back automatically; `app` is on-demand only (`process-notes.sh`),
nothing to restart there.

### Idle watchdog (automatic stop)

The instance stops itself after **2.5 hours with nothing processing**.
`idle-watchdog.timer` (installed by `user_data`, runs every 5 min) counts
the box as active while any of these holds:

- a docker container other than the long-lived `qdrant` service is running
  (all pipeline work — `run.py`, convergence loops, note processing — runs
  as containers);
- an SSH connection or SSM session is established (someone is working on
  the box);
- the 5-min load average is ≥ 1.0 (backstop for host-side work that isn't
  a container, e.g. an in-daemon image build).

When none holds for 150 consecutive minutes it logs to
`/var/log/idle-watchdog.log` and powers off — which *stops* (not
terminates) the instance per `instance_initiated_shutdown_behavior`, so
only EBS storage bills until the next `start-instances`. The idle clock
resets on every boot and on any activity. To change or test the threshold
on a live box: `echo IDLE_MINUTES=5 | sudo tee /etc/default/idle-watchdog`.

## Tearing down

```bash
terraform destroy
```
