#!/usr/bin/env bash
# Watches doctors_notes/ for new PDFs and runs process-notes.sh once uploads
# go quiet (debounced, so an scp of several files triggers one run, not one
# per file). Never runs Phase 1 — only ever calls process-notes.sh.
set -uo pipefail
WATCH_DIR=/opt/app/doctors_notes
LOCK_FILE=/tmp/process-notes.lock
DEBOUNCE_SECONDS=15
EVENT_STAMP=/tmp/.last-note-event
LOG=/var/log/process-notes-auto.log

inotifywait -m -e close_write -e moved_to --format '%f' "$WATCH_DIR" 2>>"$LOG" |
while read -r file; do
  case "$file" in
    *.pdf|*.PDF)
      echo "$(date '+%F %T') detected new note: $file" >> "$LOG"
      date +%s > "$EVENT_STAMP"
      ;;
  esac
done &

while true; do
  sleep 5
  if [ -f "$EVENT_STAMP" ]; then
    last=$(cat "$EVENT_STAMP")
    now=$(date +%s)
    if [ $((now - last)) -ge $DEBOUNCE_SECONDS ]; then
      rm -f "$EVENT_STAMP"
      echo "$(date '+%F %T') debounce elapsed — running Phase 2" >> "$LOG"
      flock "$LOCK_FILE" /opt/app/process-notes.sh >> "$LOG" 2>&1
      echo "$(date '+%F %T') Phase 2 run finished" >> "$LOG"
    fi
  fi
done
