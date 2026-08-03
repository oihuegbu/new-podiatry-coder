#!/usr/bin/env bash
# Atomically materialize the current JSON deployment secret as a Compose env.
# Usage: refresh_runtime_env.sh RELEASE_DIR SECRET_ID AWS_REGION [QDRANT_PORT]
set -euo pipefail

release_dir=${1:?release directory is required}
secret_id=${2:?Secrets Manager secret ID is required}
aws_region=${3:?AWS region is required}
qdrant_port=${4:-}

case "$release_dir" in
  /*) ;;
  *) echo "release directory must be absolute" >&2; exit 2 ;;
esac
test -d "$release_dir"
command -v aws >/dev/null
command -v jq >/dev/null
if [[ -n "$qdrant_port" && ! "$qdrant_port" =~ ^[0-9]+$ ]]; then
  echo "Qdrant port must be numeric" >&2
  exit 2
fi

env_file=$(mktemp "$release_dir/.env.XXXXXX")
secret_file=$(mktemp "$release_dir/.secret.XXXXXX")
trap 'rm -f -- "$env_file" "$secret_file"' EXIT
chmod 0600 "$env_file" "$secret_file"

aws secretsmanager get-secret-value \
  --secret-id "$secret_id" \
  --region "$aws_region" \
  --query SecretString \
  --output text > "$secret_file"

jq -e 'type == "object" and
  all(keys[]; test("^[A-Z][A-Z0-9_]*$")) and
  all(.[]; type != "string" or ((contains("\n") or contains("\r")) | not))' \
  "$secret_file" >/dev/null
if [[ -n "$qdrant_port" ]] && jq -e 'has("QDRANT_HOST_PORT")' \
    "$secret_file" >/dev/null; then
  echo "deployment secret must not override QDRANT_HOST_PORT" >&2
  exit 2
fi
jq -r 'to_entries[] | select(.value != null) |
  if (.value | type) == "string" then
    "\(.key)=\(.value)"
  else
    "\(.key)=\(.value | tojson)"
  end' "$secret_file" > "$env_file"

if [[ -n "$qdrant_port" ]]; then
  printf 'QDRANT_HOST_PORT=%s\n' "$qdrant_port" >> "$env_file"
fi

chmod 0600 "$env_file"
mv -f -- "$env_file" "$release_dir/.env"
rm -f -- "$secret_file"
trap - EXIT
