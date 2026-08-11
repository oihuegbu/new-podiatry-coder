#!/usr/bin/env bash
# Pre-exit hygiene check: repo state + any lingering Claude/tmux/screen sessions.
set -euo pipefail
cd "${CLAIMLY_WORKDIR:-$HOME/work}"
echo "branch: $(git branch --show-current)"
echo "HEAD:   $(git rev-parse HEAD)"
echo "--- working tree ---"; git status --short
echo "--- claude procs ---"; pgrep -af 'claude' || true
echo "--- tmux ---";        tmux list-sessions 2>/dev/null || true
echo "--- screen ---";      screen -ls 2>/dev/null || true
