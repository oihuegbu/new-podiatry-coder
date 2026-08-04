#!/usr/bin/env bash
set -euo pipefail

repo_root=/opt/gpt-medical-coder
snapshot_root="$repo_root/build/snapshots"
cd "$repo_root"

source_args=(
  --source professional-ptp
  --source practitioner-mue
  --source pfs-procedure-attributes
  --source mcd-coverage-articles
)
if [[ -n "${UMLS_API_KEY:-}" ]]; then
  source_args+=(--source snomed-ct-us-edition --source umls-snomed-cpt-candidate-terms)
fi

result="$(sudo -u ec2-user $repo_root/.venv/bin/medical-coder refresh-sources \
  --repository-root "$repo_root" \
  --registry "$repo_root/source-packs/refresh/sources.json" \
  "${source_args[@]}" \
  --pack "$repo_root/source-packs/authoritative/pack.json" \
  --output "$snapshot_root")"
snapshot_id="$(sudo -u ec2-user $repo_root/.venv/bin/python -c 'import json,sys; value=json.load(sys.stdin); print(value[0]["snapshot_id"])' <<<"$result")"
sudo -u ec2-user $repo_root/.venv/bin/medical-coder activate-snapshot \
  --snapshot "$snapshot_root/$snapshot_id" \
  --link "$snapshot_root/current"
systemctl try-restart gpt-medical-coder.service
