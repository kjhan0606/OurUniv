#!/usr/bin/env bash
set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
tng=/gpfs/kjhan/IllustrisTNG/TNG100-1
sequence=$tng/evaluation/tng100_simba_swift_v28_empirical_sequence
preflight=$sequence/preflight.json

[[ ${HOSTNAME,,} == lageunha ]] || {
  echo "V28 hard preflight requires Lageunha" >&2
  exit 1
}
cd "$repo"
[[ -z $(git status --porcelain) ]] || {
  echo "V28 hard preflight requires a clean committed worktree" >&2
  exit 1
}
[[ ! -e $preflight && ! -e ${preflight}.partial ]] || {
  echo "V28 hard preflight output already exists" >&2
  exit 1
}
mkdir -p "$sequence"
export PYTHONPATH=$repo/src
python -m pytest -q
python -u scripts/hong2021_v28_preflight.py \
  --registry "$repo/config/hong2021_v28_development_program.json" \
  --repo "$repo" --out "$preflight" --device cuda
