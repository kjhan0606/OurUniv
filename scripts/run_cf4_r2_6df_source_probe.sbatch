#!/usr/bin/env bash
#SBATCH --job-name=cf4-r2-6df-source
#SBATCH --partition=a10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128M
#SBATCH --time=00:02:00
#SBATCH --no-requeue
#SBATCH --chdir=/home/kjhan/BACKUP/CF4
#SBATCH --export=NONE
#SBATCH --output=/gpfs/kjhan/CF4/logs/cf4_r2_6df_source_%j.out
#SBATCH --error=/gpfs/kjhan/CF4/logs/cf4_r2_6df_source_%j.err
set -euo pipefail
repo=/home/kjhan/BACKUP/CF4
: "${SLURM_JOB_ID:?}" "${CF4_EXPECTED_COMMIT:?}"
[[ "$(git -C "$repo" rev-parse HEAD)" == "$CF4_EXPECTED_COMMIT" ]]
[[ "$(hostname -s)" != syntax && "$(hostname -s)" != syn101 ]]
[[ -z "${SLURM_JOB_GPUS:-}" ]]
git -C "$repo" diff --exit-code "$CF4_EXPECTED_COMMIT" -- scripts/cf4_r2_6df_source_probe.py
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1
out_root=/gpfs/kjhan/CF4/z0_density/r2_6df_source_probe_20261107
mkdir -p "$out_root"
/home/kjhan/miniconda3/envs/circle/bin/python3.11 -P \
    scripts/cf4_r2_6df_source_probe.py --output-dir "$out_root/$SLURM_JOB_ID"
