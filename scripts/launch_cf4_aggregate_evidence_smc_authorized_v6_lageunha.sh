#!/usr/bin/env bash
# v6 launcher: local fail-closed authorization precedes every remote action.
set -Eeuo pipefail

readonly repo=/home/kjhan/BACKUP/CF4
readonly python=/home/kjhan/miniconda3/envs/circle/bin/python
readonly host=lageunha
readonly session=cf4-aggregate-evidence-smc-authorized-v6
readonly program="$repo/config/cf4_aggregate_evidence_smc_execution_authorization_program_v6.json"
readonly runner="$repo/scripts/run_cf4_aggregate_evidence_smc_authorized_v6_lageunha.sh"
readonly data=/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_v6
readonly state=/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_v6_run

env PYTHONPATH="$repo/src" "$python" - "$program" <<'PY'
from pathlib import Path
import sys
from cf4_aggregate_evidence_smc_execution_authorized_v6 import (
    CANONICAL_PROGRAM, load_canonical_authorization_program,
    require_execution_authorization,
)
if Path(sys.argv[1]).resolve() != CANONICAL_PROGRAM.resolve():
    raise SystemExit("v6 program path is not canonical")
require_execution_authorization(load_canonical_authorization_program())
PY

# A future runner implementation may add remote launch logic only after a new
# authorization change. This shipped launcher deliberately has no remote call.
echo "unreachable v6 launch path" >&2
exit 65
