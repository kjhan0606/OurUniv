#!/usr/bin/env bash
set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
tng=/gpfs/kjhan/IllustrisTNG/TNG100-1
out=$tng/evaluation/tng100_simba_swift_v27_sequence/preflight.json
training=$tng/training/tng100_simba_swift_v27_e15_parent_aligned_haar_flow
evaluation=$tng/evaluation/tng100_simba_swift_v27_e15_parent_aligned_haar_flow
cd "$repo"
export PYTHONPATH=$repo/src

[[ ${HOSTNAME,,} == lageunha ]] || { echo "V27 preflight requires Lageunha" >&2; exit 1; }
[[ -z $(git status --porcelain) ]] || { echo "V27 preflight requires a clean committed worktree" >&2; exit 1; }
for forbidden in "$training" "$evaluation" "$out"; do
  [[ ! -e $forbidden ]] || { echo "V27 preflight refuses pre-existing output: $forbidden" >&2; exit 1; }
done
mkdir -p "$(dirname "$out")"
pytest -q
python -u scripts/hong2021_v27_preflight.py \
  --repo "$repo" \
  --registry "$repo/config/hong2021_v27_development_program.json" \
  --out "$out"
