#!/usr/bin/env bash
#SBATCH --job-name=cf4-pmwd-scan1
#SBATCH --partition=a40,a100,h200,a100_pcie
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=9216M
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --array=0-1
#SBATCH --no-requeue
#SBATCH --export=NONE
#SBATCH --chdir=/home/kjhan/BACKUP/CF4
#SBATCH --output=/gpfs/kjhan/CF4/z0_density/slurm-%x-%A_%a.out
#SBATCH --error=/gpfs/kjhan/CF4/z0_density/slurm-%x-%A_%a.err
#SBATCH --open-mode=truncate
set -euo pipefail

readonly repo=/home/kjhan/BACKUP/CF4
readonly python=/home/kjhan/miniconda3/envs/circle/bin/python3.11
readonly program="$repo/config/cf4_phasec_pmwd_scan_diagnostic_v1.json"
readonly implementation="$repo/src/cf4_phasec_pmwd_scan_diagnostic.py"
readonly output_root=/gpfs/kjhan/CF4/z0_density/phasec_pmwd_scan_diagnostic_v1
readonly program_sha=fa99690bac24282bce9275439297d287dda141fb0740177d417d416db82233cc
readonly implementation_sha=ed180607f32b0009061139333d1039c9cfaceb7e71a9f681e63b5b1d01362669

: "${SLURM_JOB_ID:?}" "${SLURM_ARRAY_TASK_ID:?}" "${SLURM_SUBMIT_HOST:?}"
: "${OUTPUT_ROOT:?}" "${EXPECTED_COMMIT:?}" "${EXPECTED_UPSTREAM_COMMIT:?}"
: "${SUBMISSION_CONTROLLER:?}" "${CONTROLLER_ACTION:?}"
[[ "$OUTPUT_ROOT" == "$output_root" ]]
[[ "$SUBMISSION_CONTROLLER" == syntax && "$CONTROLLER_ACTION" == pmwd_scan_diagnostic ]]
[[ "$SLURM_SUBMIT_HOST" == syntax || "$SLURM_SUBMIT_HOST" == syntax.* ]]
[[ "$SLURM_ARRAY_TASK_ID" =~ ^[0-1]$ ]]
[[ "$(scontrol show config | awk '$1 == "ClusterName" {print $3}')" == syntax ]]
readonly host_name=$(hostname -s)
[[ "$host_name" != syntax && "$host_name" != syntax.* ]]
[[ "$host_name" != syn101 && "$host_name" != syn101.* ]]
readonly head_commit=$(git -C "$repo" rev-parse HEAD)
readonly upstream_commit=$(git -C "$repo" rev-parse '@{upstream}')
[[ "$head_commit" == "$EXPECTED_COMMIT" ]]
[[ "$upstream_commit" == "$EXPECTED_UPSTREAM_COMMIT" ]]
[[ "$head_commit" == "$upstream_commit" ]]
[[ -z "$(git -C "$repo" status --porcelain --untracked-files=all -- . ':(exclude)scripts/tripwire/**')" ]]
[[ "$(sha256sum "$program" | awk '{print $1}')" == "$program_sha" ]]
[[ "$(sha256sum "$implementation" | awk '{print $1}')" == "$implementation_sha" ]]
[[ "${SLURM_CPUS_PER_TASK:-0}" -eq 8 && "${SLURM_MEM_PER_NODE:-0}" -eq 9216 ]]
command -v nvidia-smi >/dev/null
nvidia-smi -L | grep -q '^GPU '

export JAX_ENABLE_X64=True
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS=--xla_gpu_autotune_level=0
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONSAFEPATH=1
export PYTHONHASHSEED=0

"$python" -I -P "$implementation" run \
  --program "$program" \
  --output-root "$OUTPUT_ROOT" \
  --task-index "$SLURM_ARRAY_TASK_ID" \
  --implementation-commit "$EXPECTED_COMMIT"
readonly seed=$((2026083002 + 5 * SLURM_ARRAY_TASK_ID))
readonly task_dir=$(printf '%s/seed_%02d_%d' "$OUTPUT_ROOT" "$SLURM_ARRAY_TASK_ID" "$seed")
"$python" -I -P "$implementation" validate-task --directory "$task_dir"
