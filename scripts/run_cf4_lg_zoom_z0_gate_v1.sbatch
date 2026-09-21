#!/usr/bin/env bash
#SBATCH --job-name=cf4_lg_zoom_gate
#SBATCH --partition=a100_pcie,a40,a100,h100,h200
#SBATCH --exclude=syn06
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=/gpfs/kjhan/CF4/logs/%x_%j.out
#SBATCH --error=/gpfs/kjhan/CF4/logs/%x_%j.err
set -Eeuo pipefail
output=/gpfs/kjhan/CF4/ramses/cf4_lg_zoom_z0_production_v1/job_388664/output_00002
work=/gpfs/kjhan/CF4/diagnostics/cf4_lg_zoom_z0_gate_v1
mkdir -p "$work"
export PYTHONPATH=/home/kjhan/BACKUP/CF4/src
python /home/kjhan/BACKUP/CF4/src/cf4_zoom_z0_gate.py \
  --output "$output" --work "$work" --box 384.0 \
  --screen /gpfs/kjhan/CF4/diagnostics/no_observational_screen_for_zoom_gate.npz \
  --halo-finder hop
printf 'LG_ZOOM_GATE_PASS job=%s work=%s\n' "${SLURM_JOB_ID:?}" "$work"
