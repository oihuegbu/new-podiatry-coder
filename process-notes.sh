#!/usr/bin/env bash
# Phase 2: process notes against the already-loaded dependencies from Phase 1.
# Usage: ./process-notes.sh [--note FILE.pdf] [--no-cache]
set -euo pipefail

# A checkpoint identifies one durable audit store, not "whatever database happens to
# be called provenance.db".  The old filename fallback made every EC2 deployment that
# shared the checkpoint bucket contend for the same checkpoint key.  A replacement host
# then saw the previous host's later sequence and correctly (but permanently) refused
# every claim.  Bind both the database path and the external store identity to one
# persistent installation id. Stop/start preserves the id with its EBS data; a replacement
# host or a second checkout gets a fresh store automatically.
APP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$APP_ROOT"

resolve_ec2_instance_id() {
  local token instance_id
  token="$(curl --noproxy '*' --silent --show-error --fail --max-time 2 \
    -X PUT -H 'X-aws-ec2-metadata-token-ttl-seconds: 60' \
    http://169.254.169.254/latest/api/token 2>/dev/null || true)"
  [ -n "$token" ] || return 1
  instance_id="$(curl --noproxy '*' --silent --show-error --fail --max-time 2 \
    -H "X-aws-ec2-metadata-token: $token" \
    http://169.254.169.254/latest/meta-data/instance-id 2>/dev/null || true)"
  case "$instance_id" in
    i-[0-9a-f]*) printf '%s\n' "$instance_id" ;;
    *) return 1 ;;
  esac
}

if [ -z "${PROVENANCE_STORE_ID:-}" ]; then
  identity_file="$APP_ROOT/output/.provenance-store-id"
  mkdir -p "$APP_ROOT/output"
  # Serialize first creation so two simultaneous runs cannot fork identities.  The
  # installation UUID is required even on EC2: two checked-out deployments can live
  # on one host and must not bind their different databases to the same S3 key.
  exec 9>"${identity_file}.lock"
  flock 9
  if [ ! -s "$identity_file" ]; then
    if instance_id="$(resolve_ec2_instance_id)"; then
      identity_prefix="ec2-${instance_id}"
    else
      identity_prefix="install"
    fi
    if [ -r /proc/sys/kernel/random/uuid ]; then
      installation_uuid="$(tr -d '\n' < /proc/sys/kernel/random/uuid)"
    elif command -v uuidgen >/dev/null 2>&1; then
      installation_uuid="$(uuidgen | tr '[:upper:]' '[:lower:]')"
    else
      installation_uuid="$(python3 -c 'import uuid; print(uuid.uuid4())')"
    fi
    umask 077
    printf '%s-%s\n' "$identity_prefix" "$installation_uuid" > "$identity_file"
  fi
  PROVENANCE_STORE_ID="$(tr -cd 'A-Za-z0-9._-' < "$identity_file")"
fi

case "$PROVENANCE_STORE_ID" in
  *[!A-Za-z0-9._-]*|'')
    echo "Invalid PROVENANCE_STORE_ID; expected a non-empty filesystem-safe identifier" >&2
    exit 2
    ;;
esac

export PROVENANCE_STORE_ID
export PROVENANCE_DB="${PROVENANCE_DB:-/app/output/provenance-${PROVENANCE_STORE_ID}.db}"

exec docker compose run --rm app python run.py "$@"
