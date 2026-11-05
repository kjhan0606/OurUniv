#!/usr/bin/env bash
# v5 launcher; it refuses locally until the future one-shot authorization exists.
set -Eeuo pipefail

readonly host=lageunha
readonly session=cf4-aggregate-evidence-smc-authorized-v5
readonly repo=/home/kjhan/BACKUP/CF4
readonly python=/home/kjhan/miniconda3/envs/circle/bin/python
readonly program="$repo/config/cf4_aggregate_evidence_smc_execution_authorization_program_v5.json"
readonly runner="$repo/scripts/run_cf4_aggregate_evidence_smc_authorized_v5_lageunha.sh"
readonly data=/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_v5
readonly state=/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_v5_run
readonly receipt=/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_v5_receipts/one-shot-receipt

# No ssh, tmux, receipt, state, data, or cache action can occur before this gate.
env PYTHONPATH="$repo/src" "$python" - "$program" <<'PY'
from pathlib import Path
import sys
from cf4_aggregate_evidence_smc_execution_authorized_v5 import (
    CANONICAL_PROGRAM, load_canonical_authorization_program,
    require_execution_authorization,
)
if Path(sys.argv[1]).resolve() != CANONICAL_PROGRAM.resolve():
    raise SystemExit("v5 program path is not canonical")
require_execution_authorization(load_canonical_authorization_program())
PY
if [[ -e "$data" || -e "$state" || -e "$receipt" ]]; then
    echo "v5 data, state, or receipt already exists" >&2; exit 73
fi
if ssh -o BatchMode=yes "$host" "test -e '$data' -o -e '$state' -o -e '$receipt'"; then
    echo "remote v5 data, state, or receipt already exists" >&2; exit 73
fi
if ssh -o BatchMode=yes "$host" "tmux has-session -t '$session' 2>/dev/null"; then
    echo "remote tmux session already exists" >&2; exit 75
fi
ssh -o BatchMode=yes "$host" "tmux new-session -d -s '$session' \"exec '$runner'\""
