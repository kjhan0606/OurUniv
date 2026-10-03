#!/usr/bin/env bash
set -Eeuo pipefail

readonly repo=/home/kjhan/BACKUP/CF4
readonly python=/home/kjhan/miniconda3/envs/circle/bin/python
readonly program="$repo/config/cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_execution_authorized_program_v1.json"
readonly runner="$repo/scripts/run_cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_authorized_v1_lageunha.sh"

if [[ ! -x "$runner" || ! -x "$python" ]]; then
    echo "authorized local launcher files are absent" >&2
    exit 69
fi

# Authorization is proven once read-only before exec; the runner and wrapper
# repeat it.  This launcher neither backgrounds nor submits the workload.
env PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 \
    PYTHONPATH="$repo/src" "$python" - "$program" <<'PY'
from pathlib import Path
import sys
from cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_execution_authorized_v1 import (
    PROGRAM, load_program, validate_authorization,
)
if Path(sys.argv[1]).resolve() != PROGRAM.resolve():
    raise SystemExit("wrapper program path is not canonical")
validate_authorization(load_program())
PY
host=$(hostname)
host_short=${host%%.*}
host_normalized=$(LC_ALL=C tr '[:upper:]' '[:lower:]' <<<"$host_short")
readonly host host_short host_normalized
if [[ "$host_normalized" != lageunha ]]; then
    echo "authorized local launcher host gate failed" >&2
    exit 69
fi
exec "$runner"
