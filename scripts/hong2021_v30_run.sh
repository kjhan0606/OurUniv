#!/usr/bin/env bash
set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
program=$repo/config/hong2021_v30_backbone_condition_audit.json
out=/gpfs/kjhan/IllustrisTNG/TNG100-1/evaluation/tng100_simba_swift_v30_backbone_condition_audit/audit.json
log=${out%.json}.log

cd "$repo"
test -z "$(git status --porcelain)"
PYTHONPATH=src pytest -q
PYTHONPATH=src python -u src/hong2021_v30_backbone_audit.py \
  --program "$program" \
  --repo "$repo" \
  --out "$out" >"$log" 2>&1
