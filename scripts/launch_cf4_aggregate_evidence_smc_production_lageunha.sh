#!/usr/bin/env bash
set -Eeuo pipefail

readonly host=lageunha
readonly session=cf4-aggregate-evidence-smc-v1
readonly repo=/home/kjhan/BACKUP/CF4
readonly python=/home/kjhan/miniconda3/envs/circle/bin/python
readonly program="$repo/config/cf4_aggregate_evidence_smc_production_program.json"
readonly runner=/home/kjhan/BACKUP/CF4/scripts/run_cf4_aggregate_evidence_smc_production_lageunha.sh
readonly data=/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_v1
readonly state=/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_v1_run

env PYTHONPATH="$repo/src" "$python" - "$program" <<'PY'
import json
from pathlib import Path
import sys

program = json.loads(Path(sys.argv[1]).read_text())
if program.get("authorization", {}).get("production_execution_authorized") is not True:
    raise SystemExit("production SMC execution remains unauthorized")
PY

if [[ -e "$data" || -e "$state" ]]; then
    echo "production SMC data or lifecycle state already exists" >&2; exit 73
fi
if ssh -o BatchMode=yes "$host" "test -e '$data' -o -e '$state'"; then
    echo "remote production SMC data or lifecycle state already exists" >&2; exit 73
fi
if ssh -o BatchMode=yes "$host" "tmux has-session -t '$session' 2>/dev/null"; then
    echo "remote tmux session already exists: $host:$session" >&2; exit 75
fi
ssh -o BatchMode=yes "$host" \
    "tmux new-session -d -s '$session' \"exec '$runner'\""
echo "launched $host:$session"
