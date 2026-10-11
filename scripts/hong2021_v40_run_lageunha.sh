#!/usr/bin/env bash
set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
program=$repo/config/hong2021_v40_object_structure_sufficiency_program.json
sequence=/gpfs/kjhan/IllustrisTNG/TNG100-1/evaluation/tng100_simba_swift_v40_object_structure_sufficiency_sequence
output=/gpfs/kjhan/IllustrisTNG/TNG100-1/evaluation/tng100_simba_swift_v40_object_structure_sufficiency/audit.json

cd "$repo"
export PYTHONPATH=$repo/src
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16

[[ $(hostname -s | tr '[:upper:]' '[:lower:]') == lageunha ]] || { echo "V40 requires Lageunha" >&2; exit 1; }
[[ -z $(git status --porcelain) ]] || { echo "V40 requires clean worktree" >&2; exit 1; }
[[ ! -e $sequence ]] || { echo "V40 refuses existing sequence: $sequence" >&2; exit 1; }
[[ ! -e ${output%/*} ]] || { echo "V40 refuses existing output directory: ${output%/*}" >&2; exit 1; }

mkdir -p "$sequence"
pytest -q >"$sequence/pytest.log" 2>&1
python -u src/hong2021_v40_object_structure_sufficiency.py \
  --program "$program" \
  --repo "$repo" \
  --out "$output" \
  >"$sequence/audit.log" 2>&1
