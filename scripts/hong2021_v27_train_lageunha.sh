#!/usr/bin/env bash
set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
tng=/gpfs/kjhan/IllustrisTNG/TNG100-1
out=$tng/training/tng100_simba_swift_v27_e15_parent_aligned_haar_flow
preflight=$tng/evaluation/tng100_simba_swift_v27_sequence/preflight.json
lock=/gpfs/kjhan/.hong2021_locks/v27_e15_training.lock
cd "$repo"
export PYTHONPATH=$repo/src

[[ ${HOSTNAME,,} == lageunha ]] || { echo "V27 training requires Lageunha" >&2; exit 1; }
mkdir -p "$(dirname "$lock")"
exec 8>"$lock"
flock -n 8 || { echo "another V27 training process holds the lock" >&2; exit 2; }
exec python -u src/hong2021_v27.py train \
  --registry "$repo/config/hong2021_v27_development_program.json" \
  --repo "$repo" --out "$out" --preflight "$preflight" --device cuda
