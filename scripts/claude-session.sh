#!/usr/bin/env bash
# Credential-safe Claude launcher for new-podiatry-coder.
# Unsets the application's Anthropic/OpenAI credentials so an interactive Claude uses your
# subscription, not the app's API account. Does NOT touch the app's .env.
set -euo pipefail
cd "${CLAIMLY_WORKDIR:-$HOME/work}"
echo "=== new-podiatry-coder Claude session ==="
echo "branch: $(git branch --show-current)"
echo "HEAD:   $(git rev-parse HEAD)"
git status --short
echo "launching credential-safe claude..."
exec env -u ANTHROPIC_API_KEY -u ANTHROPIC_AUTH_TOKEN -u ANTHROPIC_BASE_URL \
         -u OPENAI_API_KEY claude "$@"
