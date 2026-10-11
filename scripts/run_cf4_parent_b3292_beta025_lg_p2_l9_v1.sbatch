#!/usr/bin/env bash
#SBATCH --job-name=cf4_b3292_lg_p2
#SBATCH --partition=h200,h100,a100
#SBATCH --exclude=syn06
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=/gpfs/kjhan/CF4/logs/%x_%j.out
#SBATCH --error=/gpfs/kjhan/CF4/logs/%x_%j.err
set -Eeuo pipefail
cd /home/kjhan/BACKUP/CF4
export JAX_ENABLE_X64=True
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
readonly output=/gpfs/kjhan/CF4/recon/linear_cr/parent_b3292_beta025_lg_p2_l9_v1
if [[ -e "$output" ]]; then
  echo "refusing to overwrite existing output: $output" >&2
  exit 3
fi
exec /home/kjhan/miniconda3/envs/circle/bin/python3.11 src/cf4_p2_screen.py \
  --p1-result config/cf4_parent_b3292_beta025_p1_bridge_v1.json \
  --config config/p2_lg_targets_v11_parent_b3292_beta025_l9.json \
  --outdir "$output"
