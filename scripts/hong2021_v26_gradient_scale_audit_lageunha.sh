#!/usr/bin/env bash
set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
tng=/gpfs/kjhan/IllustrisTNG/TNG100-1
output=$tng/evaluation/tng100_simba_swift_v26_e14_conditional_haar_flow/gradient_scale_audit_v2.json
lock=/gpfs/kjhan/.hong2021_locks/v26_gradient_scale_audit.lock

[[ ${HOSTNAME,,} == lageunha ]] || {
  echo "V26 gradient-scale audit requires Lageunha" >&2
  exit 1
}
cd "$repo"
[[ -z $(git status --porcelain) ]] || {
  echo "V26 gradient-scale audit requires a clean committed worktree" >&2
  exit 1
}
mkdir -p "$(dirname "$lock")"
exec 8>"$lock"
flock -n 8 || {
  echo "another V26 gradient-scale audit holds the lock" >&2
  exit 2
}
export PYTHONPATH=$repo/src
exec python -u scripts/hong2021_v26_gradient_scale_audit.py \
  --registry "$repo/config/hong2021_v26_development_program.json" \
  --repo "$repo" \
  --out "$output" \
  --device cuda
