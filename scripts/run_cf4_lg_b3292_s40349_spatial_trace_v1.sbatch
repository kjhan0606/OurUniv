#!/usr/bin/env bash
#SBATCH --job-name=cf4_s40349_spatial_trace
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --no-requeue
#SBATCH --output=/gpfs/kjhan/CF4/logs/grammar_%x_%j.out
#SBATCH --error=/gpfs/kjhan/CF4/logs/grammar_%x_%j.err
set -Eeuo pipefail

root=/home/kjhan/BACKUP/CF4
z0=/gpfs/kjhan/CF4/ramses/sv1_grammar/i248_s40349/job_1104302/output_00001
initial=/gpfs/kjhan/CF4/ramses/s40349_traceonly/job_1108290/output_00001
evidence=/gpfs/kjhan/CF4/diagnostics/cf4_lg_b3292_i248_s40349_survival_v1.json
members=/gpfs/kjhan/CF4/masks/cf4_lg_b3292_i248_s40349_survival_members_v1.npz
positions=/gpfs/kjhan/CF4/masks/cf4_lg_b3292_s40349_spatial_lagrangian_positions_v1.npz
mask=/gpfs/kjhan/CF4/masks/cf4_lg_b3292_s40349_spatial_mask_l9_v1.npz
report=/gpfs/kjhan/CF4/diagnostics/cf4_lg_b3292_s40349_spatial_trace_v1.json

test -d "$z0"
test -d "$initial"
test -s "$evidence"
test -s "$members"
test ! -e "$positions"
test ! -e "$mask"
test ! -e "$report"
cd "$root"
export PYTHONPATH="$root/src:$root/scripts"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/home/kjhan/miniconda3/envs/circle/bin/python3.11 \
  scripts/cf4_trace_ramses_spatial_region.py \
  --z0-snapshot "$z0" --initial-snapshot "$initial" \
  --selection-evidence "$evidence" --members "$members" \
  --positions-output "$positions" --mask-output "$mask" --report-output "$report" \
  --box 384.0 --radius 5.0 --base-level 9 --buffer 1.5 \
  --subbox-pad-base-cells 8

/home/kjhan/miniconda3/envs/circle/bin/python3.11 - "$report" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1]))
assert report["selection_status"] == "trace-only-zoom-candidate"
assert report["parent_promoted"] is False
assert report["m33_resolved"] is False
assert report["z0_selection"]["selected_pair_members_retained"] == report["z0_selection"]["selected_pair_member_count"]
assert report["z0_selection"]["particle_count"] >= 100
assert report["mask"]["occupied_base_cells"] > 0
PY
printf 'SPATIAL_TRACE_ONLY_MASK_PASS job=%s report=%s mask=%s\n' \
  "$SLURM_JOB_ID" "$report" "$mask"
