#!/bin/bash
#SBATCH --job-name=cf4_highk_pm_recover
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=21G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --array=1,2,4-15%1
#SBATCH --no-requeue
#SBATCH --export=NONE
#SBATCH --chdir=/home/kjhan/BACKUP/CF4
#SBATCH --output=/gpfs/kjhan/CF4/recon/linear_cr/lg_highk_streaming_pm_production_v1_recovery_slurm-%A_%a.log
#SBATCH --error=/gpfs/kjhan/CF4/recon/linear_cr/lg_highk_streaming_pm_production_v1_recovery_slurm-%A_%a.log
#SBATCH --open-mode=truncate

set -Eeuo pipefail

readonly repo=/home/kjhan/BACKUP/CF4
readonly program="$repo/config/cf4_lg_highk_streaming_forward_program_v1.json"
readonly cache=/gpfs/kjhan/CF4/recon/linear_cr/lg_highk_covariance_cache_v1/cache.npz
readonly output=/gpfs/kjhan/CF4/recon/linear_cr/lg_highk_streaming_pm_production_v1
readonly gpu_guard="$repo/scripts/cf4_lg_highk_recovery_gpu_guard.py"
readonly expected_program_sha=856b758457dc69f3d9e6ee19ad8b88d68da96bc3d801e6b5a9f01b87a98dd135
readonly expected_cache_sha=c15747dd7afdfb6d2b0013c726bdaa99b8fc1a3b3a1c75581086eebe35470b1a
readonly expected_gpu_guard_sha=98467d11e12d43485297b2290d86773cbdab690114e7e6836048c632681fa1a7

declare -A expected_sha=(
    ["$repo/src/cf4_lg_highk_streaming_forward.py"]=0494bdf3440e85906e4ded45ce9b1284fc40533b9a7c8f12aab1dac617201814
    ["$repo/src/cf4_lg_highk_conditional_field.py"]=d2c1ee27596f2f4af9b0328f2d3142e4eca02c6ffad3ba2bcd8eb1334431e317
    ["$repo/src/cf4_lg_highk_covariance_cache.py"]=6da87bb7ef73ada54269424c82e949d16059a87eb209bc79235144681979040d
    ["$repo/src/cf4_peak_evidence_phase_cache.py"]=6359497a141aa0814c0dd663353ae6623f01e3676c0c2fdfea5a1b272e9d7106
    ["$repo/src/cf4_aggregate_evidence_oracle.py"]=3da4bd598f381e8a6fccd1dc2ae179cdd01e14d80ad0ad6c25dd1b3a93631d7f
    ["$repo/src/cf4_p2_screen.py"]=e6287e81500148baf245833468d04fb483ef68b632378981e2b9c66f520d9799
    ["$repo/src/cf4_lg_z0_likelihood.py"]=0fa110b3165de723896666b9249e81d81400d2b9a4a5968d108fecf6cb9d4f22
    ["$repo/src/mock_pipeline.py"]=74971ac119a69611854e0ed716bde23da3a50d34b555a556c5f06c092dd7b975
    ["$repo/src/fof.py"]=9352f0fe344790eacfdde41cfce68caacf73f97dbcfd0b4186587cfd30b0b5bb
)
[[ "$(sha256sum "$program" | awk '{print $1}')" == "$expected_program_sha" ]]
[[ "$(sha256sum "$cache" | awk '{print $1}')" == "$expected_cache_sha" ]]
[[ "$(sha256sum "$gpu_guard" | awk '{print $1}')" == "$expected_gpu_guard_sha" ]]
for path in "${!expected_sha[@]}"; do
    [[ "$(sha256sum "$path" | awk '{print $1}')" == "${expected_sha[$path]}" ]]
done

# Job 324676 established that a 95,830 MiB H100 does not reliably fit the
# compiled N576 forward, while the 143,771 MiB H200 completed the same batch.
# Require exactly one visible GPU and at least 20% headroom above the observed
# H100 capacity before JAX is imported.  A partition/GRES mismatch fails before
# the runner can inspect or create a canonical row.
/home/kjhan/miniconda3/bin/python "$gpu_guard" --minimum-mib 115000
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

export PYTHONPATH="$repo/src"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONSAFEPATH=1
export JAX_ENABLE_X64=True
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"

exec /home/kjhan/miniconda3/envs/circle/bin/python \
    -m cf4_lg_highk_streaming_forward \
    --program "$program" \
    --mode production \
    --batch-index "$SLURM_ARRAY_TASK_ID" \
    --covariance-cache "$cache" \
    --output-root "$output"
