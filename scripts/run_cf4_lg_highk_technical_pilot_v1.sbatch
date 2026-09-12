#!/bin/bash
#SBATCH --job-name=cf4_highk_tech4
#SBATCH --partition=a40,a100,h100,h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=36G
#SBATCH --time=08:00:00
#SBATCH --no-requeue
#SBATCH --export=NONE
#SBATCH --chdir=/home/kjhan/BACKUP/CF4
#SBATCH --output=/gpfs/kjhan/CF4/recon/linear_cr/lg_highk_technical_pilot_v1_slurm-%j.log
#SBATCH --error=/gpfs/kjhan/CF4/recon/linear_cr/lg_highk_technical_pilot_v1_slurm-%j.log
#SBATCH --open-mode=truncate

set -Eeuo pipefail

readonly repo=/home/kjhan/BACKUP/CF4
readonly python=/home/kjhan/miniconda3/bin/python
readonly schedule=/gpfs/kjhan/CF4/recon/linear_cr/lg_highk_joint_schedule_v1/schedule.npz
readonly filter=/gpfs/kjhan/CF4/recon/linear_cr/peak_evidence_phase_control_v2/density_filter_rfft.npy
readonly manifest=/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_parent_response_atlas_v1/manifest.json
readonly output=/gpfs/kjhan/CF4/recon/linear_cr/lg_highk_technical_pilot_v1

if [[ -e "$output" ]]; then
    echo "refusing to overwrite existing technical pilot: $output" >&2
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

exec "$python" -m cf4_lg_highk_technical_pilot \
    --schedule "$schedule" \
    --schedule-sha256 10c8526f3508a908a14981636ca9f730480838b65ea4bbcd3186c0730f330ccd \
    --filter "$filter" \
    --filter-sha256 1e1d2ce4b022c908b8a3a64257e82fb8621c40cf8f77b1fd206f1347cdd2f59a \
    --parent-manifest "$manifest" \
    --parent-manifest-sha256 47049d0047aa626912652c82ac34757f01ebe4adc0654d07f674d6b943db4211 \
    --output-root "$output"
