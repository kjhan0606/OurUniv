#!/usr/bin/env bash
#SBATCH --job-name=cf4_lg_s30113_env
#SBATCH --partition=a100_pcie,h200,h100,a100
#SBATCH --exclude=syn06
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output=/gpfs/kjhan/CF4/logs/%x_%A_%a.out
#SBATCH --error=/gpfs/kjhan/CF4/logs/%x_%A_%a.err
#SBATCH --array=0-1
set -Eeuo pipefail
cd /home/kjhan/BACKUP/CF4
export JAX_ENABLE_X64=True
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
readonly field=/gpfs/kjhan/CF4/recon/linear_cr/parent_b3292_beta025_lg_selected_s30113_v1/passing_field_000.npz
if [[ "$SLURM_ARRAY_TASK_ID" == 0 ]]; then
  exec /home/kjhan/miniconda3/envs/circle/bin/python3.11 scripts/cf4_parent_candidate_n256_stability.py \
    --candidate "$field" --field-key s_conditioned --p1-config config/p1_targets_v2_observer.json \
    --N 256 --seeds 30113 \
    --output /gpfs/kjhan/CF4/diagnostics/cf4_lg_selected_s30113_n256_environment_v1.json
fi
exec /home/kjhan/miniconda3/envs/circle/bin/python3.11 scripts/cf4_parent_candidate_n256_stability.py \
  --candidate "$field" --field-key s_conditioned --p1-config config/p1_targets_v2_observer.json \
  --N 512 --cluster-only --seeds 30113 \
  --output /gpfs/kjhan/CF4/diagnostics/cf4_lg_selected_s30113_n512_clusters_v1.json
