#!/usr/bin/env bash
#SBATCH --job-name=cf4-z0-phasec-v3
#SBATCH --partition=a40,a100,h200,a100_pcie
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=10240M
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --array=0-7
#SBATCH --no-requeue
#SBATCH --export=NONE
#SBATCH --chdir=/home/kjhan/BACKUP/CF4
#SBATCH --output=/gpfs/kjhan/CF4/z0_density/slurm-%x-%A_%a.out
#SBATCH --error=/gpfs/kjhan/CF4/z0_density/slurm-%x-%A_%a.err
#SBATCH --open-mode=truncate

set -euo pipefail
readonly repo=/home/kjhan/BACKUP/CF4
readonly python=/home/kjhan/miniconda3/envs/circle/bin/python3.11
readonly program="$repo/config/cf4_datum_bearing_z0_phasec_execution_amendment_v3.json"
readonly implementation="$repo/src/cf4_datum_bearing_z0_phasec_pilot.py"
readonly preflight=/gpfs/kjhan/CF4/z0_density/datum_bearing_phasec_preflight_v3
readonly output_root=/gpfs/kjhan/CF4/z0_density/datum_bearing_phasec_pilot_v3
readonly program_sha=f2c52c834e133ec4e8702275f63214d37fdeb0ab1f53a83e2b6e6a99d83bfb50
readonly implementation_sha=657f3353e9105ba7a10babb82c6821ae0c8def2a2e1bb12047fffd1bf7a3c1de

: "${SLURM_JOB_ID:?active Slurm job required}"
: "${SLURM_ARRAY_TASK_ID:?Slurm array task required}"
: "${SLURM_SUBMIT_HOST:?Slurm submit host required}"
: "${PREFLIGHT_OUTPUT:?PREFLIGHT_OUTPUT must be explicitly exported}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT must be explicitly exported}"
: "${EXPECTED_COMMIT:?EXPECTED_COMMIT must be explicitly exported}"
: "${EXPECTED_UPSTREAM_COMMIT:?EXPECTED_UPSTREAM_COMMIT must be explicitly exported}"
: "${SUBMISSION_CONTROLLER:?SUBMISSION_CONTROLLER must be explicitly exported}"
: "${CONTROLLER_ACTION:?CONTROLLER_ACTION must be explicitly exported}"
[[ "$PREFLIGHT_OUTPUT" == "$preflight" ]]
[[ "$OUTPUT_ROOT" == "$output_root" ]]
[[ "$SUBMISSION_CONTROLLER" == syntax ]]
[[ "$CONTROLLER_ACTION" == submit ]]
[[ "$SLURM_SUBMIT_HOST" == syntax || "$SLURM_SUBMIT_HOST" == syntax.* ]]
[[ "$SLURM_ARRAY_TASK_ID" =~ ^[0-7]$ ]]
readonly cluster_name=$(scontrol show config | awk '$1 == "ClusterName" {print $3}')
[[ "$cluster_name" == syntax ]]
readonly host_name=$(hostname -s)
[[ "$host_name" != syntax && "$host_name" != syntax.* && "$host_name" != syn101 && "$host_name" != syn101.* ]]
readonly head_commit=$(git -C "$repo" rev-parse HEAD)
readonly upstream_commit=$(git -C "$repo" rev-parse '@{upstream}')
[[ "$head_commit" == "$EXPECTED_COMMIT" ]]
[[ "$upstream_commit" == "$EXPECTED_UPSTREAM_COMMIT" ]]
[[ "$head_commit" == "$upstream_commit" ]]
[[ -z "$(git -C "$repo" status --porcelain --untracked-files=all -- . ':(exclude)scripts/tripwire/**')" ]]
[[ "$(sha256sum "$program" | awk '{print $1}')" == "$program_sha" ]]
[[ "$(sha256sum "$implementation" | awk '{print $1}')" == "$implementation_sha" ]]
[[ -f "$preflight/COMPLETE" ]]
"$python" -I -P "$implementation" validate-preflight --directory "$preflight"
[[ "${SLURM_CPUS_PER_TASK:-0}" -eq 8 ]]
[[ "${SLURM_MEM_PER_NODE:-0}" -eq 10240 ]]
command -v nvidia-smi >/dev/null
nvidia-smi -L | grep -q '^GPU '

export JAX_ENABLE_X64=True
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS=--xla_gpu_autotune_level=0
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 PYTHONHASHSEED=0
"$python" -I -P "$implementation" run \
  --program "$program" --output-root "$OUTPUT_ROOT" \
  --task-index "$SLURM_ARRAY_TASK_ID" --implementation-commit "$EXPECTED_COMMIT"
readonly seed=$((2026083000 + SLURM_ARRAY_TASK_ID))
readonly arms=AABBCCDD
readonly arm=${arms:SLURM_ARRAY_TASK_ID:1}
readonly task_dir=$(printf '%s/mock_%02d_seed_%d_arm_%s' "$OUTPUT_ROOT" "$SLURM_ARRAY_TASK_ID" "$seed" "$arm")
"$python" -I -P "$implementation" validate-task --directory "$task_dir"
