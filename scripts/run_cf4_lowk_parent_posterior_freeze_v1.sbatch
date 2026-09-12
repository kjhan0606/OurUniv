#!/bin/bash
#SBATCH --job-name=cf4_lowk_freeze
#SBATCH --partition=a40,a100,h100,h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:20:00
#SBATCH --no-requeue
#SBATCH --export=NONE
#SBATCH --chdir=/home/kjhan/BACKUP/CF4
#SBATCH --output=/gpfs/kjhan/CF4/recon/linear_cr/lowk_parent_posterior_bridge_v1_slurm-%j.log
#SBATCH --error=/gpfs/kjhan/CF4/recon/linear_cr/lowk_parent_posterior_bridge_v1_slurm-%j.log
#SBATCH --open-mode=truncate

set -Eeuo pipefail

readonly repo=/home/kjhan/BACKUP/CF4
readonly python=/home/kjhan/miniconda3/envs/circle/bin/python
readonly bridge=/gpfs/kjhan/CF4/recon/linear_cr/lowk_cross_mode_bridge_2048_v3
readonly source=/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_v6_open_shared_schedule_production_grammar_debug_v2/terminal_parent_frozen.npz
readonly output=/gpfs/kjhan/CF4/recon/linear_cr/lowk_parent_posterior_bridge_v1
readonly base=/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_v6_open_shared_schedule_production_grammar_debug_v2_cache/shard_000000.npz
readonly terminal=/gpfs/kjhan/CF4/recon/linear_cr/lowk_terminal_rejuvenation_pilot_v1
readonly pilot128=/gpfs/kjhan/CF4/recon/linear_cr/lowk_cross_mode_bridge_pilot_v1/new_evidence_cache.npz
readonly pilot512=/gpfs/kjhan/CF4/recon/linear_cr/lowk_cross_mode_bridge_512_v2/new_evidence_cache.npz
readonly pilot2048=/gpfs/kjhan/CF4/recon/linear_cr/lowk_cross_mode_bridge_2048_v3/new_evidence_cache.npz

if [[ -e "$output" ]]; then
    echo "refusing to overwrite existing output: $output" >&2
    exit 65
fi

export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONSAFEPATH=1
export PYTHONPATH="$repo/src"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

exec "$python" -m cf4_lowk_parent_posterior_freeze \
    --artifact-root "$bridge" \
    --source-terminal "$source" \
    --output-root "$output" \
    --cache-shard "$base" \
    --cache-shard "$terminal/new_evidence_cache_replicate_0.npz" \
    --cache-shard "$terminal/new_evidence_cache_replicate_1.npz" \
    --cache-shard "$terminal/new_evidence_cache_replicate_2.npz" \
    --cache-shard "$terminal/new_evidence_cache_replicate_3.npz" \
    --cache-shard "$pilot128" \
    --cache-shard "$pilot512" \
    --cache-shard "$pilot2048"
