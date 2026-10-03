#!/usr/bin/env bash
#SBATCH --job-name=cf4_R2_selquad
#SBATCH --partition=h100
#SBATCH --gres=gpu:H100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --time=00:30:00
#SBATCH --output=/gpfs/kjhan/CF4/logs/%x_%j.out
#SBATCH --error=/gpfs/kjhan/CF4/logs/%x_%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?}" "${CF4_EXPECTED_COMMIT:?}" "${CF4_R2_OUT_DIR:?}"
module purge
cd /home/kjhan/BACKUP/CF4
actual_commit=$(git rev-parse HEAD)
if [[ "$actual_commit" != "$CF4_EXPECTED_COMMIT" ]]; then
  printf 'source commit mismatch: expected %s, found %s\n' "$CF4_EXPECTED_COMMIT" "$actual_commit" >&2
  exit 2
fi
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
exec /home/kjhan/miniconda3/bin/python3.13 scripts/cf4_r2_selection_quadrature_audit.py
