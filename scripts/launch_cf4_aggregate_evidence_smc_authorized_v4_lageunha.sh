#!/usr/bin/env bash
set -Eeuo pipefail

readonly host=lageunha
readonly session=cf4-aggregate-evidence-smc-authorized-v4
readonly repo=/home/kjhan/BACKUP/CF4
readonly python=/home/kjhan/miniconda3/envs/circle/bin/python
readonly program="$repo/config/cf4_aggregate_evidence_smc_execution_authorization_program_v4.json"
readonly runner="$repo/scripts/run_cf4_aggregate_evidence_smc_authorized_v4_lageunha.sh"
readonly data=/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_v4
readonly state=/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_v4_run

# Refuse locally before any filesystem reservation, remote query, or tmux action.
env PYTHONPATH="$repo/src" "$python" - "$program" <<'PY'
from pathlib import Path
import sys
from cf4_aggregate_evidence_smc_execution_authorized_v4 import (
    CANONICAL_PROGRAM,
    load_canonical_authorization_program,
    require_execution_authorization,
)

if Path(sys.argv[1]).resolve() != CANONICAL_PROGRAM.resolve():
    raise SystemExit("authorization program path is not canonical")
program = load_canonical_authorization_program(verify_file_hashes=False)
require_execution_authorization(program)
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


