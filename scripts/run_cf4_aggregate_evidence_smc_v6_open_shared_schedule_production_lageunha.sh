#!/usr/bin/env bash
set -Eeuo pipefail

umask 077
readonly repo=/home/kjhan/BACKUP/CF4
readonly python=/home/kjhan/miniconda3/envs/circle/bin/python
readonly program="$repo/config/cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_program.json"
readonly expected_program_sha=54ffb61a9053a6e7935a7355a6d5a948184c4ded8dc585cf85b45d209a9b2dbc
readonly data=/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_v6_open_shared_schedule_production_v1
readonly state=/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_v6_open_shared_schedule_production_v1_run
readonly receipt=/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_v6_open_shared_schedule_production_v1_receipts
readonly cache=/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_v6_open_shared_schedule_production_v1_cache
readonly -a future_timeout_command=(
    /usr/bin/timeout --foreground --signal=TERM --kill-after=300s 12h
)

# This runner is intentionally pre-execution only.  A separately audited
# authorization wrapper must replace this refusal boundary in the future.
if [[ ! -x "$python" || ! -f "$program" ]]; then
    echo "missing canonical Python or production program" >&2
    exit 66
fi
if [[ "$(sha256sum "$program" | awk '{print $1}')" != "$expected_program_sha" ]]; then
    echo "canonical production program hash changed" >&2
    exit 65
fi

host=$(hostname)
host_short=${host%%.*}
host_normalized=$(LC_ALL=C tr '[:upper:]' '[:lower:]' <<<"$host_short")
readonly host host_short host_normalized
if [[ "$host_normalized" != lageunha ]]; then
    echo "host gate failed: expected lageunha, found $host" >&2
    exit 69
fi

if [[ -e "$data" || -e "$state" || -e "$receipt" || -e "$cache" ]]; then
    echo "prospective production namespace is not absent" >&2
    exit 73
fi

if env PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 \
    PYTHONPATH="$repo/src" "$python" - <<'PY'
from cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_execution import (
    load_canonical_program,
)

program = load_canonical_program(verify_file_hashes=True)
if any(program["authorization"].values()):
    raise SystemExit("unexpected open runtime authorization")
raise SystemExit("production execution remains intentionally unauthorized")
PY
then
    echo "unauthorized program unexpectedly passed the runtime gate" >&2
fi

# The preceding Python gate always exits nonzero for this immutable program.
# No resource check, reservation, timeout child, or marker may be reached.
exit 65
