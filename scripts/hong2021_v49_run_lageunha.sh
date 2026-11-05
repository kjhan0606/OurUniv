#!/usr/bin/env bash
set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
program="$repo/config/hong2021_v49_gaussian_extreme_calibration_audit_program.json"
root=/gpfs/kjhan/IllustrisTNG/TNG100-1/evaluation/tng100_simba_swift_v49_gaussian_extreme_calibration_audit
output="$root/audit.json"

cd "$repo"
test -z "$(git status --porcelain)"
test "$(sha256sum "$program" | cut -d' ' -f1)" = "7595330346f0a1fcc8a195f99ca8df805b47879b1d6cad1f9e3a9e85559b6c5a"
test ! -e "$output"
mkdir -p "$root"
python -m pytest -q 2>&1 | tee "$root/pytest.log"
python -u src/hong2021_v49_gaussian_extreme_audit.py \
  --program "$program" \
  --repo "$repo" \
  --out "$output" 2>&1 | tee "$root/audit.log"
