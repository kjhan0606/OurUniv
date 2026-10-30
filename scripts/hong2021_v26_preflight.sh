#!/usr/bin/env bash
set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
out=/gpfs/kjhan/IllustrisTNG/TNG100-1/evaluation/tng100_simba_swift_v26_sequence/preflight.json
training=/gpfs/kjhan/IllustrisTNG/TNG100-1/training/tng100_simba_swift_v26_e14_conditional_haar_flow
evaluation=/gpfs/kjhan/IllustrisTNG/TNG100-1/evaluation/tng100_simba_swift_v26_e14_conditional_haar_flow
cd "$repo"
export PYTHONPATH=$repo/src

[[ ${HOSTNAME,,} == lageunha ]] || { echo "V26 preflight requires Lageunha" >&2; exit 1; }
[[ -z $(git status --porcelain) ]] || { echo "V26 preflight requires a clean committed worktree" >&2; exit 1; }
for forbidden in "$training" "$evaluation" "$out"; do
  [[ ! -e $forbidden ]] || { echo "V26 preflight refuses pre-existing output: $forbidden" >&2; exit 1; }
done
mkdir -p "$(dirname "$out")"
pytest -q
python -u scripts/hong2021_v26_preflight.py \
  --repo "$repo" \
  --registry "$repo/config/hong2021_v26_development_program.json" \
  --out "$out"
