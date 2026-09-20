#!/usr/bin/env bash
#SBATCH --job-name=cf4_lg_zoom_ic
#SBATCH --partition=a100_pcie,a40,a100,h100,h200
#SBATCH --exclude=syn06
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=/gpfs/kjhan/CF4/logs/%x_%j.out
#SBATCH --error=/gpfs/kjhan/CF4/logs/%x_%j.err
set -Eeuo pipefail
: "${SLURM_JOB_ID:?}"
cd /home/kjhan/BACKUP/CF4
export PYTHONPATH="$PWD/src:$PWD/scripts"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
exec /home/kjhan/miniconda3/envs/circle/bin/python3.11 src/cf4_zoom_ic2.py \
  --parent /gpfs/kjhan/CF4/kf_design/production_ic_grafic_v1 \
  --parent-level 8 --parent-grid-size 256 \
  --transfer /gpfs/kjhan/CF4/zoom/transfer_parent_n256_v1.npz \
  --out /gpfs/kjhan/CF4/zoom/cf4_lg_zoom_pilot_v1 \
  --resolution-config config/ic_resolution_v1.json \
  --tier pilot --levelmin 8 --levelmax-ic 12 --runtime-levelmax 19 \
  --box-hmpc 384.0 \
  --mask-npz /gpfs/kjhan/CF4/zoom/cf4_lg_trace_v1/lg_mask_l8.npz \
  --seed 5108 --fft-workers 8
