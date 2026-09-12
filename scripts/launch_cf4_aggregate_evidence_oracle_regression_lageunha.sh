#!/usr/bin/env bash
set -Eeuo pipefail

readonly host=lageunha
readonly session=cf4-aggregate-oracle-regression-v1
readonly runner=/home/kjhan/BACKUP/CF4/scripts/run_cf4_aggregate_evidence_oracle_regression_lageunha.sh
readonly data=/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_oracle_regression_v1
readonly state=/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_oracle_regression_v1_run

if [[ -e "$data" || -e "$state" ]]; then
    echo "oracle-regression data or lifecycle state already exists" >&2; exit 73
fi
if ssh -o BatchMode=yes "$host" "tmux has-session -t '$session' 2>/dev/null"; then
    echo "remote tmux session already exists: $host:$session" >&2; exit 75
fi
ssh -o BatchMode=yes "$host" \
    "tmux new-session -d -s '$session' \"exec '$runner'\""
echo "launched $host:$session"
