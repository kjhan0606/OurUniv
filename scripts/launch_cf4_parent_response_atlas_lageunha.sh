#!/usr/bin/env bash
set -Eeuo pipefail

readonly host=lageunha
readonly session=cf4-parent-response-atlas-v1
readonly runner=/home/kjhan/BACKUP/CF4/scripts/run_cf4_parent_response_atlas_lageunha.sh
readonly data=/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_parent_response_atlas_v1
readonly state=/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_parent_response_atlas_v1_run

if [[ -e "$data" || -e "$state" ]]; then
    echo "response-atlas data or lifecycle state already exists" >&2; exit 73
fi
if ssh -o BatchMode=yes "$host" "tmux has-session -t '$session' 2>/dev/null"; then
    echo "remote tmux session already exists: $host:$session" >&2; exit 75
fi
ssh -o BatchMode=yes "$host" \
    "tmux new-session -d -s '$session' \"exec '$runner'\""
echo "launched $host:$session"
