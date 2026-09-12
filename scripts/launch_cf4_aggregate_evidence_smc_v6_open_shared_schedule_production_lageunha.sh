#!/usr/bin/env bash
set -Eeuo pipefail

readonly repo=/home/kjhan/BACKUP/CF4
readonly python=/home/kjhan/miniconda3/envs/circle/bin/python
readonly program="$repo/config/cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_program.json"
readonly runner="$repo/scripts/run_cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_lageunha.sh"
readonly session=cf4_v6_open_shared_schedule_production

if [[ ! -x "$python" || ! -x "$runner" || ! -f "$program" ]]; then
    echo "missing canonical launcher input" >&2
    exit 66
fi

# Refuse locally before SSH, tmux, timeout, or any remote/process action.
if env PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 \
    PYTHONPATH="$repo/src" "$python" - <<'PY'
from cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_execution import (
    load_canonical_program,
)

program = load_canonical_program(verify_file_hashes=True)
if any(program["authorization"].values()):
    raise SystemExit("unexpected open runtime authorization")
raise SystemExit("production launcher remains intentionally unauthorized")
PY
then
    echo "unauthorized program unexpectedly passed the launcher gate" >&2
fi

echo "production launcher is fail-closed; session not created: $session" >&2
exit 65
