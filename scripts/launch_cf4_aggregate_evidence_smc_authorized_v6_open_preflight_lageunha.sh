#!/usr/bin/env bash
set -Eeuo pipefail
readonly repo=/home/kjhan/BACKUP/CF4
readonly python=/home/kjhan/miniconda3/envs/circle/bin/python
readonly program="$repo/config/cf4_aggregate_evidence_smc_execution_authorization_program_v6_open_preflight.json"
readonly host=$(hostname)
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$repo/src" "$python" - "$program" "$host" <<'PY'
from pathlib import Path
import sys
from cf4_aggregate_evidence_smc_execution_authorized_v6_open_preflight import PROGRAM, run_preflight_v6_open
if Path(sys.argv[1]).resolve() != PROGRAM.resolve():
    raise SystemExit("noncanonical preflight program")
print(run_preflight_v6_open(PROGRAM, "pilot", sys.argv[2]))
PY
exit 65
