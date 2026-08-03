#!/usr/bin/env bash
# Write an atomic deployment receipt for an extracted release directory.
# Usage: write_deployment_receipt.sh RELEASE_DIR COMMIT_SHA ARTIFACT [INSTALLED_AT]
set -euo pipefail

release_dir=${1:?release directory is required}
commit_sha=${2:?40-character commit SHA is required}
artifact=${3:?artifact identity is required}
installed_at=${4:-$(date -u +%Y-%m-%dT%H:%M:%SZ)}

case "$release_dir" in
  /*) ;;
  *) echo "release directory must be absolute" >&2; exit 2 ;;
esac
test -d "$release_dir"
[[ "$commit_sha" =~ ^[0-9a-f]{40}$ ]] || {
  echo "commit SHA must contain exactly 40 lowercase hexadecimal characters" >&2
  exit 2
}
[[ "$artifact" =~ ^[A-Za-z0-9._-]+$ ]] || {
  echo "artifact identity contains unsafe characters" >&2
  exit 2
}
[[ "$installed_at" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$ ]] || {
  echo "installed timestamp must be UTC ISO-8601" >&2
  exit 2
}

receipt=$(mktemp "$release_dir/.deployment-receipt.XXXXXX")
trap 'test ! -e "$receipt" || rm -f -- "$receipt"' EXIT
printf 'git_commit=%s\nartifact=%s\ninstalled_at=%s\n' \
  "$commit_sha" "$artifact" "$installed_at" > "$receipt"
chmod 0644 "$receipt"
mv -f -- "$receipt" "$release_dir/DEPLOYMENT_RECEIPT"
trap - EXIT
