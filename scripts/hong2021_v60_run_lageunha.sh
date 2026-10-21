#!/usr/bin/env bash
set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
program="$repo/config/hong2021_v60_reachable_support_grid_program.json"
sequence=/gpfs/kjhan/IllustrisTNG/TNG100-1/evaluation/tng100_simba_swift_v60_reachable_support_grid_sequence
grid_root=/gpfs/kjhan/IllustrisTNG/TNG100-1/evaluation/tng100_simba_swift_v60_reachable_support_grid
grid="$grid_root/grid.json"

cd "$repo"
if [[ "$(hostname -s | tr '[:upper:]' '[:lower:]')" != "lageunha" ]]; then
  echo "V60 requires Lageunha" >&2
  exit 1
fi
if [[ -e "$sequence" || -e "$grid_root" ]]; then
  echo "V60 refuses existing output" >&2
  exit 1
fi
mkdir -p "$sequence"
printf 'testing\n' > "$sequence/status"
pytest -q > "$sequence/pytest.log" 2>&1
printf 'materializing\n' > "$sequence/status"
python -u src/hong2021_v60_reachable_support_grid.py \
  --program "$program" \
  --repo "$repo" \
  --out "$grid" \
  > "$sequence/grid.log" 2>&1
jq -e '.status == "complete_reachable_support_grid" and .existing_thresholds_byte_equal == true and .final_threshold_equals_global_reachable_upper == true and .independent_gate_locked == true and .development_accessed == false' "$grid" >/dev/null
printf 'complete\n' > "$sequence/status"
