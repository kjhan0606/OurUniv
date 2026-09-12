#!/usr/bin/env bash
set -Eeuo pipefail

umask 077
readonly repo=/home/kjhan/BACKUP/CF4
readonly python=/home/kjhan/miniconda3/envs/circle/bin/python
readonly program="$repo/config/cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_execution_authorized_program_v1.json"
readonly data=/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_v6_open_shared_schedule_production_v1
readonly state=/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_v6_open_shared_schedule_production_v1_run
readonly receipts=/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_v6_open_shared_schedule_production_v1_receipts
readonly cache=/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_v6_open_shared_schedule_production_v1_cache

if [[ ! -x "$python" || ! -f "$program" ]]; then
    echo "canonical Python or wrapper program is absent" >&2
    exit 66
fi

# This is the complete read-only authorization gate.  With the committed
# execution-false program and absent grant/pair it exits before flock,
# resources, receipt, cache, data, state, or evaluator creation.
grant_id=$(env PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 \
    PYTHONPATH="$repo/src" "$python" - "$program" <<'PY'
from pathlib import Path
import sys
from cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_execution_authorized_v1 import (
    PROGRAM, load_program, validate_authorization,
)
if Path(sys.argv[1]).resolve() != PROGRAM.resolve():
    raise SystemExit("wrapper program path is not canonical")
print(validate_authorization(load_program())["grant"]["grant_id"])
PY
)
readonly grant_id
if [[ ! "$grant_id" =~ ^[0-9a-f]{64}$ ]]; then
    echo "authorization did not return a full grant ID" >&2
    exit 65
fi

runtime_pid=
seal_supervisor_failed() {
    local checkpoint=$1
    env PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 \
        PYTHONPATH="$repo/src" "$python" - "$grant_id" "$checkpoint" <<'PY'
import sys
from cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_execution_authorized_v1 import (
    _supervisor_force_failed,
)
_supervisor_force_failed(sys.argv[1], sys.argv[2])
PY
}
handle_runner_signal() {
    local signal_name=$1
    local exit_code=$2
    trap - INT TERM HUP
    if [[ -n "$runtime_pid" ]]; then
        kill -TERM "$runtime_pid" 2>/dev/null || true
        wait "$runtime_pid" 2>/dev/null || true
    fi
    seal_supervisor_failed "runner_signal_${signal_name}"
    exit "$exit_code"
}
trap 'handle_runner_signal INT 130' INT
trap 'handle_runner_signal TERM 143' TERM
trap 'handle_runner_signal HUP 129' HUP

host=$(hostname)
host_short=${host%%.*}
host_normalized=$(LC_ALL=C tr '[:upper:]' '[:lower:]' <<<"$host_short")
readonly host host_short host_normalized
if [[ "$host_normalized" != lageunha ]]; then
    echo "host gate failed: expected lageunha, found $host" >&2
    exit 69
fi

if [[ -e "$data" || -L "$data" || -e "$state" || -L "$state" \
      || -e "$receipts" || -L "$receipts" || -e "$cache" || -L "$cache" ]]; then
    echo "one-shot production namespace is not absent" >&2
    exit 73
fi

mem_available_kib=$(awk '$1=="MemAvailable:" {print $2}' /proc/meminfo)
readonly mem_available_kib
if [[ ! "$mem_available_kib" =~ ^[0-9]+$ ]] || (( mem_available_kib < 80 * 1024 * 1024 )); then
    echo "MemAvailable is below the fixed 80 GiB gate" >&2
    exit 70
fi
free_kib=$(df -Pk "$(dirname "$data")" | awk 'NR==2 {print $4}')
readonly free_kib
if [[ ! "$free_kib" =~ ^[0-9]+$ ]] || (( free_kib < 40 * 1024 * 1024 )); then
    echo "free disk is below the fixed 40 GiB gate" >&2
    exit 70
fi
available_cpus=$(nproc)
readonly available_cpus
if [[ ! "$available_cpus" =~ ^[0-9]+$ ]] || (( available_cpus < 8 )); then
    echo "CPU affinity exposes fewer than eight CPUs" >&2
    exit 70
fi

export CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export MALLOC_ARENA_MAX=2

# The wrapper owns receipt/state terminal transitions.  timeout sends TERM,
# allows five minutes for the wrapper's fail-closed exception path, and then
# returns nonzero.  There is no retry, resume, loop, or automatic follow-on.
set +e
/usr/bin/timeout --foreground --signal=TERM --kill-after=300s 12h \
    env PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 \
    PYTHONPATH="$repo/src" "$python" - "$program" <<'PY' &
from pathlib import Path
import sys
from cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_execution_authorized_v1 import (
    run_authorized_production,
)
run_authorized_production(Path(sys.argv[1]))
PY
runtime_pid=$!
readonly runtime_pid
wait "$runtime_pid"
runtime_rc=$?
set -e
readonly runtime_rc
trap - INT TERM HUP
if (( runtime_rc != 0 )); then
    seal_supervisor_failed "supervisor_timeout_or_child_exit_${runtime_rc}"
    exit "$runtime_rc"
fi
