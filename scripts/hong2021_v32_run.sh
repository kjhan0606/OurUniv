#!/usr/bin/env bash
set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
program=$repo/config/hong2021_v32_context_audit_program.json
root=/gpfs/kjhan/IllustrisTNG/TNG100-1/evaluation/tng100_simba_swift_v32_context_audit
out=$root/audit.json

cd "$repo"
[[ -z $(git status --porcelain) ]] || { echo "V32 requires clean worktree" >&2; exit 1; }
[[ ! -e $root ]] || { echo "V32 output already exists: $root" >&2; exit 1; }
mkdir -p "$root"
export PYTHONPATH=$repo/src
python -m pytest -q
python -u src/hong2021_v32_context_audit.py \
  --program "$program" --repo "$repo" --out "$out" \
  >"$root/audit.log" 2>&1
