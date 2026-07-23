#!/usr/bin/env bash
# Stops the instance after IDLE_MINUTES with no work. "Work" is any of:
#   1. a docker container other than the long-lived qdrant service (pipeline
#      runs, convergence loops, and note processing all run as containers);
#   2. an established SSH connection or an active SSM session (someone is
#      working on the box interactively);
#   3. 5-min load average >= 1.0 (backstop for host-side work that isn't a
#      container in `docker ps`, e.g. an in-daemon `docker compose build`).
# Runs every 5 min from idle-watchdog.timer. The last-active stamp lives on
# tmpfs, so every boot restarts the idle clock from zero. The instance has
# instance_initiated_shutdown_behavior=stop, so poweroff stops (not
# terminates) it and only EBS storage bills until the next start.
set -uo pipefail
IDLE_MINUTES=60
# Test/override hook: `echo IDLE_MINUTES=5 > /etc/default/idle-watchdog`
[ -f /etc/default/idle-watchdog ] && . /etc/default/idle-watchdog
STAMP=/run/idle-watchdog.stamp
LOG=/var/log/idle-watchdog.log

active=""
containers=$(docker ps --format '{{.Names}}' 2>/dev/null | grep -vi qdrant || true)
[ -n "$containers" ] && active="containers: $(echo "$containers" | tr '\n' ' ')"

if [ -z "$active" ]; then
  ssh_count=$(ss -tn state established '( sport = :22 )' 2>/dev/null | tail -n +2 | wc -l)
  pgrep -f ssm-session-worker >/dev/null 2>&1 && ssh_count=$((ssh_count + 1))
  [ "$ssh_count" -gt 0 ] && active="ssh/ssm sessions: $ssh_count"
fi

if [ -z "$active" ]; then
  load5=$(cut -d' ' -f2 /proc/loadavg)
  awk -v l="$load5" 'BEGIN{exit !(l >= 1.0)}' && active="load5: $load5"
fi

now=$(date +%s)
if [ -n "$active" ]; then
  echo "$now" > "$STAMP"
  exit 0
fi

# First idle observation after boot (or after activity) starts the clock.
[ -f "$STAMP" ] || { echo "$now" > "$STAMP"; exit 0; }

idle_min=$(( (now - $(cat "$STAMP")) / 60 ))
if [ "$idle_min" -ge "$IDLE_MINUTES" ]; then
  echo "$(date '+%F %T') idle for $idle_min min (limit $IDLE_MINUTES) — stopping instance" >> "$LOG"
  systemctl poweroff
fi
