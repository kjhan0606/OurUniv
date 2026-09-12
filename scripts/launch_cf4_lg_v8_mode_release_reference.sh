#!/usr/bin/env bash
set -Eeuo pipefail

readonly session=cf4-v8-ref
readonly runner=/home/kjhan/BACKUP/CF4/scripts/run_cf4_lg_v8_mode_release_reference.sh
readonly state=/gpfs/kjhan/CF4/recon/linear_cr/v8_cf4_mode_release_reference

if tmux has-session -t "$session" 2>/dev/null; then
    echo "tmux session already exists: $session" >&2
    exit 75
fi
if [[ -e "$state/calibration.json" || -e "$state/RUNNING" \
      || -e "$state/COMPLETE" || -e "$state/FAILED" \
      || -e "$state/run.log" || -e "$state/environment.txt" ]]; then
    echo "reference output or lifecycle marker already exists under $state" >&2
    exit 73
fi
if [[ ! -x "$runner" ]]; then
    echo "runner is not executable: $runner" >&2
    exit 66
fi

tmux new-session -d -s "$session" "exec env CUDA_VISIBLE_DEVICES=0 '$runner'"
echo "launched $session; inspect $state/RUNNING and $state/run.log"
