#!/usr/bin/env bash
set -Eeuo pipefail

readonly host=lageunha
readonly session=cf4-nyquist-control
readonly runner=/home/kjhan/BACKUP/CF4/scripts/run_cf4_projection_nyquist_control_lageunha.sh
readonly state=/gpfs/kjhan/CF4/recon/linear_cr/projection_nyquist_control

if [[ -e "$state/result.json" || -e "$state/RUNNING" \
      || -e "$state/COMPLETE" || -e "$state/FAILED" \
      || -e "$state/run.log" || -e "$state/environment.txt" ]]; then
    echo "Nyquist-control output or lifecycle file already exists under $state" >&2
    exit 73
fi
if ssh -o BatchMode=yes "$host" "tmux has-session -t '$session' 2>/dev/null"; then
    echo "remote tmux session already exists: $host:$session" >&2
    exit 75
fi
ssh -o BatchMode=yes "$host" \
    "tmux new-session -d -s '$session' \"exec '$runner'\""
echo "launched $host:$session"
