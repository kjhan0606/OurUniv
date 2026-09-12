#!/usr/bin/env bash
set -Eeuo pipefail
readonly repo=/home/kjhan/BACKUP/CF4
readonly python=/home/kjhan/miniconda3/envs/circle/bin/python
readonly program="$repo/config/cf4_aggregate_evidence_smc_execution_authorization_program_v6_open.json"
readonly runner="$repo/scripts/run_cf4_aggregate_evidence_smc_authorized_v6_open_lageunha.sh"
env PYTHONPATH="$repo/src" "$python" - "$program" <<'PY'
from pathlib import Path
import sys
from cf4_aggregate_evidence_smc_execution_authorized_v6_open import PROGRAM,load_program,require_execution_authorization
if Path(sys.argv[1]).resolve()!=PROGRAM.resolve(): raise SystemExit('noncanonical program')
require_execution_authorization(load_program(),'pilot')
PY
exit 65
