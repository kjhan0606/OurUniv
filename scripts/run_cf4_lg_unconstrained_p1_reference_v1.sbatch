#!/bin/bash
#SBATCH --job-name=cf4_unconstrained_p1_ref
#SBATCH --partition=a10,a40,h100,h200,a100,a100_pcie
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=20G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --no-requeue
#SBATCH --export=NONE
#SBATCH --chdir=/home/kjhan/BACKUP/CF4
#SBATCH --output=/home/kjhan/BACKUP/CF4/logs/unconstrained_p1_reference_v1-%j.log
#SBATCH --error=/home/kjhan/BACKUP/CF4/logs/unconstrained_p1_reference_v1-%j.log
#SBATCH --open-mode=append

# Host peak estimate is 16 GiB for the sequential N=192 member forward,
# transient float64 draw, ephemeral host field handoffs, and summary rows.
# --mem=20G supplies exactly 25% headroom, exceeding the frozen >=20% requirement.
set -Eeuo pipefail
umask 077

readonly repo=/home/kjhan/BACKUP/CF4
readonly python=/home/kjhan/miniconda3/envs/circle/bin/python3.11
readonly program="$repo/config/cf4_lg_unconstrained_p1_reference_program_v1.json"
readonly runner="$repo/src/cf4_lg_unconstrained_p1_reference.py"
readonly checker="$repo/scripts/check_cf4_lg_unconstrained_p1_reference_v1.py"
readonly grant="$repo/config/cf4_lg_unconstrained_p1_reference_execution_grant_v1.json"
readonly output=/gpfs/kjhan/CF4/recon/linear_cr/lg_unconstrained_p1_reference_v1
readonly activation_timeout_seconds=3600

phase=pregrant
abort_technical() {
    trap - ERR HUP INT TERM
    if [[ "$phase" == pregrant ]]; then
        echo 'TECHNICAL_PRE_EXECUTION_ABORT_NO_SCIENCE' >&2
    else
        echo 'TECHNICAL_POST_GATE_INCOMPLETE_NO_RETRY_OR_CLEANUP' >&2
    fi
    exit 70
}
trap abort_technical ERR HUP INT TERM
abort_preexecution() { abort_technical; }

[[ "${SLURM_CLUSTER_NAME:-}" == syntax ]] || abort_preexecution
[[ -n "${SLURM_JOB_ID:-}" && -n "${SLURM_JOB_NODELIST:-}" ]] || abort_preexecution
[[ "$(hostname)" != syntax ]] || abort_preexecution
[[ "${SLURM_NTASKS:-}" == 1 && "${SLURM_CPUS_PER_TASK:-}" == 16 ]] || abort_preexecution
[[ -x "$python" && "$(readlink -f "$python")" == "$python" ]] || abort_preexecution

export PYTHONPATH="$repo/src"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONSAFEPATH=1
export JAX_ENABLE_X64=1
export JAX_PLATFORM_NAME=gpu
export TF_CPP_MIN_LOG_LEVEL=2
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export NUMEXPR_NUM_THREADS=16

# Freeze the implementation and exact upstream remote-tracking ref before the
# receipt.  The worker never switches allocations or branches under this grant.
readonly implementation_commit="$(git -C "$repo" rev-parse HEAD)"
readonly upstream_ref="$(git -C "$repo" rev-parse --symbolic-full-name '@{upstream}')"
[[ "$upstream_ref" == refs/remotes/* ]] || abort_preexecution
readonly expected_old_commit="$(git -C "$repo" rev-parse "$upstream_ref")"
[[ "$implementation_commit" == "$expected_old_commit" ]] || abort_preexecution
readonly ref="$repo/.git/$upstream_ref"
[[ -f "$ref" && -d "$(dirname "$ref")" ]] || abort_preexecution

# Receipt generation imports no JAX/PMWD and creates no scientific array.
"$python" -P "$runner" --config "$program" --emit-runtime-receipt
echo 'HELD_ALLOCATION_BLOCKED_NO_SCIENCE_WAITING_EXACT_GRANT'

# One Linux inotify event wait on the exact upstream remote-tracking ref; this
# event follows a successful controller push, avoiding a local-commit/push race.
# There is no filesystem scan, polling loop, or external tool dependency.
"$python" -P "$runner" --config "$program" --wait-ref "$ref" \
    --expected-old-commit "$expected_old_commit" \
    --wait-timeout "$activation_timeout_seconds" || abort_preexecution
[[ -f "$grant" ]] || abort_preexecution

# Exact upstream grant, lineage, implementation bytes, allocation receipt, and
# runtime pins are verified inside the same allocation before science imports.
"$python" -P "$runner" --config "$program" --grant "$grant" --lineage-preflight
phase=postgate
echo 'HELD_ALLOCATION_GRANT_ACTIVE_SCIENCE_GATE_OPEN'
[[ ! -e "$output" ]] || abort_preexecution
"$python" -P "$runner" --config "$program" --grant "$grant"
exec "$python" -P "$checker" --config "$program" --grant "$grant" --output "$output"
