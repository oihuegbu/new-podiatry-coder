#!/usr/bin/env bash
# Install a published source artifact into an isolated EC2 release directory.
# The source tree is built and validated before the active directory is
# replaced. Mutable bind-mounted state is copied forward; named Docker volumes
# remain attached to the explicitly supplied Compose project.
set -euo pipefail

if [[ $# -ne 6 ]]; then
  echo "usage: $0 S3_URI TARGET_DIR COMPOSE_PROJECT QDRANT_HOST_PORT SECRET_ARN AWS_REGION" >&2
  exit 64
fi

artifact_uri=$1
target_dir=$2
compose_project=$3
qdrant_host_port=$4
secret_arn=$5
aws_region=$6

if [[ $target_dir != /opt/* || $target_dir == *".."* || $target_dir == /opt/ ]]; then
  echo "TARGET_DIR must be a specific directory below /opt" >&2
  exit 64
fi
if [[ ! $compose_project =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]*$ ]]; then
  echo "COMPOSE_PROJECT contains unsupported characters" >&2
  exit 64
fi
if [[ ! $qdrant_host_port =~ ^[0-9]+$ ]] \
    || (( qdrant_host_port < 1 || qdrant_host_port > 65535 )); then
  echo "QDRANT_HOST_PORT must be an integer from 1 through 65535" >&2
  exit 64
fi
if [[ $(id -u) -ne 0 ]]; then
  exec sudo -- "$0" "$@"
fi

target_parent=$(dirname -- "$target_dir")
mkdir -p -- "$target_parent"
stage_dir=$(mktemp -d "${target_parent}/.podiatry-release.XXXXXX")
archive_path=$(mktemp /tmp/podiatry-source.XXXXXX.zip)
cleanup() {
  rm -f -- "$archive_path"
  if [[ -n ${stage_dir:-} && -d $stage_dir ]]; then
    rm -rf -- "$stage_dir"
  fi
}
trap cleanup EXIT

aws s3 cp "$artifact_uri" "$archive_path" --region "$aws_region"
unzip -q "$archive_path" -d "$stage_dir"
test -f "$stage_dir/docker-compose.yml"
test -f "$stage_dir/run.py"

# These directories are mutable source-of-truth or runtime state exposed by
# bind mounts. Copy them into the candidate release without mutating the
# active release; the old tree remains a rollback snapshot after the swap.
runtime_paths=(
  doctors_notes
  output
  logs
  data/feedback
  data/registry
  data/policy
  data/rules
  benchmark
)
if [[ -d $target_dir ]]; then
  for relative_path in "${runtime_paths[@]}"; do
    source_path="$target_dir/$relative_path"
    destination_path="$stage_dir/$relative_path"
    if [[ -e $source_path ]]; then
      rm -rf -- "$destination_path"
      mkdir -p -- "$(dirname -- "$destination_path")"
      cp -a -- "$source_path" "$destination_path"
    fi
  done
fi

aws secretsmanager get-secret-value \
  --secret-id "$secret_arn" \
  --region "$aws_region" \
  --query SecretString --output text \
  | jq -r 'to_entries[] | "\(.key)=\(.value)"' > "$stage_dir/.env"
printf 'QDRANT_HOST_PORT=%s\n' "$qdrant_host_port" >> "$stage_dir/.env"
chmod 600 "$stage_dir/.env"

active_app=$(docker ps -q \
  --filter "label=com.docker.compose.project=$compose_project" \
  --filter "label=com.docker.compose.service=app")
if [[ -n $active_app ]]; then
  echo "refusing to deploy while an app container is active for $compose_project" >&2
  exit 75
fi

docker compose -p "$compose_project" -f "$stage_dir/docker-compose.yml" config -q
docker compose -p "$compose_project" -f "$stage_dir/docker-compose.yml" build app

backup_dir=""
if [[ -d $target_dir ]]; then
  backup_dir="${target_dir}.previous.$(date -u +%Y%m%dT%H%M%SZ)"
  mv -- "$target_dir" "$backup_dir"
fi
mv -- "$stage_dir" "$target_dir"
stage_dir=""

if ! docker compose -p "$compose_project" \
    -f "$target_dir/docker-compose.yml" up -d qdrant; then
  failed_dir="${target_dir}.failed.$(date -u +%Y%m%dT%H%M%SZ)"
  mv -- "$target_dir" "$failed_dir"
  if [[ -n $backup_dir ]]; then
    mv -- "$backup_dir" "$target_dir"
  fi
  echo "qdrant refresh failed; restored the previous release" >&2
  exit 1
fi

echo "installed=$target_dir"
echo "rollback=${backup_dir:-none}"
echo "compose_project=$compose_project"
