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

## Running terraform from the EC2 box

Operator-driven infra work (this repo's own terraform) is meant to run from
the box over SSH, like everything else — but the box's own always-attached
IAM role (`podiatry-coder-ec2-role`) is deliberately narrow: it's the same
identity the running application uses, and it must never be able to widen
its own permissions or touch the checkpoint-anchor bucket's delete-Deny
(issue #6, F6-R4-A — a prior version of this file attached PowerUserAccess
directly to that role, which let it delete/rewrite its own restricting
policy; that was a real hole, not a theoretical one).

Instead, a separate `podiatry-coder-terraform-operator` role holds
PowerUserAccess + IAM management scoped to this project's own roles. It is
assumable only by a dedicated `podiatry-coder-terraform-operator-user` IAM
user whose own permissions are just `sts:AssumeRole` on that one role,
nothing else — a leaked bootstrap key can start a session, not act directly.
**Not** assumable by the account root: AWS unconditionally refuses to let the
root user call `AssumeRole` on anything (a hard, non-configurable
restriction — a prior version of this doc documented a root-based workflow
that had never actually been verified to work and, when checked live, does
not). Not assumable by the box's own instance-profile credentials either,
which are not a trusted principal in the role's trust policy. To run
terraform from the box:

```bash
# from LOCAL, ONE TIME: fetch the bootstrap user's key and store it in a
# password manager -- never in this repo, never on the box:
terraform output -raw terraform_operator_access_key_id
terraform output -raw terraform_operator_secret_access_key

# each time you need to run terraform, from LOCAL, using that stored key:
AWS_ACCESS_KEY_ID=<stored id> AWS_SECRET_ACCESS_KEY=<stored secret> \
  aws sts assume-role --role-arn arn:aws:iam::<account-id>:role/podiatry-coder-terraform-operator \
  --role-session-name terraform-work --duration-seconds 3600
# copy the resulting AccessKeyId/SecretAccessKey/SessionToken into the SSH
# session's environment (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY /
# AWS_SESSION_TOKEN) -- never write them to a file on the box, never persist
# them past that one terraform session. They expire in <= 1 hour regardless.
cd ~/work/terraform && terraform plan / apply
```

After applying, sync `terraform.tfstate` back to local (`scp`) so both
copies stay consistent — there's no shared remote backend, so this is a
manual step, not automatic.
