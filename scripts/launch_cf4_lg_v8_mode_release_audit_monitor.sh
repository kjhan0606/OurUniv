#!/usr/bin/env bash
set -Eeuo pipefail

readonly session=cf4-v8-mode-audit-monitor
readonly monitor=/home/kjhan/BACKUP/CF4/scripts/monitor_cf4_lg_v8_mode_release_audit.sh
if tmux has-session -t "$session" 2>/dev/null; then
    echo "tmux session already exists: $session" >&2
    exit 75
fi
if [[ ! -x "$monitor" ]]; then
    echo "monitor is not executable: $monitor" >&2
    exit 66
fi
tmux new-session -d -s "$session" "exec '$monitor'"
echo "launched $session"
