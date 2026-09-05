#!/bin/bash
#SBATCH --job-name=cf4_smc_v6_gdebug
#SBATCH --partition=debug
#SBATCH --nodelist=grammar-debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --signal=B:TERM@300
#SBATCH --no-requeue
#SBATCH --export=NONE
#SBATCH --chdir=/home/kjhan/BACKUP/CF4
#SBATCH --output=/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_v6_open_shared_schedule_production_grammar_debug_v2_slurm-%j.log
#SBATCH --error=/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_v6_open_shared_schedule_production_grammar_debug_v2_slurm-%j.log
#SBATCH --open-mode=truncate
set -Eeuo pipefail

umask 077
unset BASH_ENV ENV PYTHONSTARTUP LD_PRELOAD LD_LIBRARY_PATH
export PATH=/usr/bin:/bin
readonly repo=/home/kjhan/BACKUP/CF4
readonly python=/home/kjhan/miniconda3/envs/circle/bin/python
readonly module=cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_grammar_debug_execution_authorized_v2

export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONSAFEPATH=1
export PYTHONPATH="$repo/src"
export CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export MALLOC_ARENA_MAX=2

dirty=$(
    /usr/bin/git -C "$repo" status --porcelain=v1 --untracked-files=all -- \
        config src scripts tests ':(exclude)scripts/tripwire/**'
)
readonly dirty
if [[ -n "$dirty" ]]; then
    exit 65
fi

grant_id=
runtime_pid=
seal_supervisor_failed() {
    local checkpoint=$1
    if [[ "$grant_id" =~ ^[0-9a-f]{64}$ ]]; then
        /usr/bin/env PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 \
            PYTHONPATH="$repo/src" "$python" -c \
            'import sys; from cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_grammar_debug_execution_authorized_v2 import _supervisor_force_failed; _supervisor_force_failed(sys.argv[1], sys.argv[2])' \
            "$grant_id" "$checkpoint"
    fi
}
handle_batch_signal() {
    local signal_name=$1
    local exit_code=$2
    trap - INT TERM HUP
    if [[ -n "$runtime_pid" ]]; then
        kill -TERM "$runtime_pid" 2>/dev/null || true
        wait "$runtime_pid" 2>/dev/null || true
    fi
    seal_supervisor_failed "slurm_batch_signal_${signal_name}"
    exit "$exit_code"
}
trap 'handle_batch_signal INT 130' INT
trap 'handle_batch_signal TERM 143' TERM
trap 'handle_batch_signal HUP 129' HUP

if [[ ! -x "$python" ]]; then
    exit 66
fi

grant_id=$(
    /usr/bin/env PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 \
        PYTHONPATH="$repo/src" CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 \
        OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
        VECLIB_MAXIMUM_THREADS=1 MALLOC_ARENA_MAX=2 "$python" -c \
        'from cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_grammar_debug_execution_authorized_v2 import load_program, validate_authorization, _read_scheduler_context, _require_resources, _require_runtime_environment; a=validate_authorization(load_program()); s=_read_scheduler_context(); _require_resources(s); _require_runtime_environment(); print(a["grant"]["grant_id"])'
)
readonly grant_id
if [[ ! "$grant_id" =~ ^[0-9a-f]{64}$ ]]; then
    exit 65
fi

set +e
/usr/bin/timeout --foreground --signal=TERM --kill-after=240s 12h \
    /usr/bin/env PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 \
    PYTHONPATH=/home/kjhan/BACKUP/CF4/src CUDA_VISIBLE_DEVICES= \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MALLOC_ARENA_MAX=2 \
    /home/kjhan/miniconda3/envs/circle/bin/python -m \
    cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_grammar_debug_execution_authorized_v2 &
runtime_pid=$!
readonly runtime_pid
wait "$runtime_pid"
runtime_rc=$?
set -e
readonly runtime_rc
trap - INT TERM HUP

case "$runtime_rc" in
    0) exit 0 ;;
    124) seal_supervisor_failed supervisor_timeout_124 ;;
    137) seal_supervisor_failed supervisor_timeout_kill_137 ;;
    143) seal_supervisor_failed supervisor_term_143 ;;
    *) seal_supervisor_failed "supervisor_invalid_child_exit_${runtime_rc}" ;;
esac
exit "$runtime_rc"
