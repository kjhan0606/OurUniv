#!/bin/bash
#SBATCH --job-name=cf4_highk_pm_pilot
#SBATCH --partition=h200,h100,a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --export=NONE
#SBATCH --chdir=/home/kjhan/BACKUP/CF4
#SBATCH --output=/gpfs/kjhan/CF4/recon/linear_cr/lg_highk_streaming_pm_pilot_v1_slurm-%j.log
#SBATCH --error=/gpfs/kjhan/CF4/recon/linear_cr/lg_highk_streaming_pm_pilot_v1_slurm-%j.log
#SBATCH --open-mode=truncate

set -Eeuo pipefail

readonly repo=/home/kjhan/BACKUP/CF4
readonly program="$repo/config/cf4_lg_highk_streaming_forward_program_v1.json"
readonly output=/gpfs/kjhan/CF4/recon/linear_cr/lg_highk_streaming_pm_pilot_v1
readonly expected_program_sha=34a8e8976be7b5b8690568b620898910d534913ca8326cb5f86120f63cfcd8b1
readonly expected_runner_sha=0494bdf3440e85906e4ded45ce9b1284fc40533b9a7c8f12aab1dac617201814
readonly expected_field_sha=d2c1ee27596f2f4af9b0328f2d3142e4eca02c6ffad3ba2bcd8eb1334431e317
readonly expected_p2_sha=e6287e81500148baf245833468d04fb483ef68b632378981e2b9c66f520d9799
readonly expected_likelihood_sha=0fa110b3165de723896666b9249e81d81400d2b9a4a5968d108fecf6cb9d4f22

declare -A expected_sha=(
    ["$program"]="$expected_program_sha"
    ["$repo/src/cf4_lg_highk_streaming_forward.py"]="$expected_runner_sha"
    ["$repo/src/cf4_lg_highk_conditional_field.py"]="$expected_field_sha"
    ["$repo/src/cf4_p2_screen.py"]="$expected_p2_sha"
    ["$repo/src/cf4_lg_z0_likelihood.py"]="$expected_likelihood_sha"
)
for path in "${!expected_sha[@]}"; do
    [[ "$(sha256sum "$path" | awk '{print $1}')" == "${expected_sha[$path]}" ]]
done

export PYTHONPATH="$repo/src"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONSAFEPATH=1
export JAX_ENABLE_X64=True
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"

nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
exec /home/kjhan/miniconda3/envs/circle/bin/python \
    -m cf4_lg_highk_streaming_forward \
    --program "$program" \
    --mode pilot \
    --output-root "$output"
