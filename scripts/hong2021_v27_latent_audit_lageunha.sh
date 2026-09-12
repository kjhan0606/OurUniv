#!/usr/bin/env bash
set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
tng=/gpfs/kjhan/IllustrisTNG/TNG100-1
evaluation=$tng/evaluation/tng100_simba_swift_v27_e15_parent_aligned_haar_flow
output=$evaluation/trained_flow_latent_audit.json
log=$evaluation/trained_flow_latent_audit.log
lock=/gpfs/kjhan/.hong2021_locks/v27_trained_flow_latent_audit.lock

[[ ${HOSTNAME,,} == lageunha ]] || {
  echo "V27 latent audit requires Lageunha" >&2
  exit 1
}
cd "$repo"
[[ -z $(git status --porcelain) ]] || {
  echo "V27 latent audit requires a clean committed worktree" >&2
  exit 1
}
mkdir -p "$(dirname "$lock")" "$evaluation"
exec 8>"$lock"
flock -n 8 || {
  echo "another V27 latent audit holds the lock" >&2
  exit 2
}
export PYTHONPATH=$repo/src
python -u scripts/hong2021_v27_latent_audit.py \
  --audit-program "$repo/config/hong2021_v27_latent_audit_program.json" \
  --repo "$repo" \
  --out "$output" \
  --device cuda >"$log" 2>&1
