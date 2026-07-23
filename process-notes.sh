#!/usr/bin/env bash
# Phase 2: process notes against the already-loaded dependencies from Phase 1.
# Usage: ./process-notes.sh [--note FILE.pdf] [--no-cache]
cd /opt/app
docker compose run --rm app python run.py "$@"
