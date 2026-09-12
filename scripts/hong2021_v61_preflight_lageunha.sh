#!/usr/bin/env bash
set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
program="$repo/config/hong2021_v61_reachable_support_model_program.json"
sequence=/gpfs/kjhan/IllustrisTNG/TNG100-1/evaluation/tng100_simba_swift_v61_reachable_support_sequence
preflight="$sequence/preflight.json"

cd "$repo"
if [[ "$(hostname -s | tr '[:upper:]' '[:lower:]')" != "lageunha" ]]; then
  echo "V61 requires Lageunha" >&2
  exit 1
fi
if [[ -e "$sequence" ]]; then
  echo "V61 refuses existing output" >&2
  exit 1
fi
mkdir -p "$sequence"
trap 'printf "failed_preflight_execution\n" > "$sequence/status"' ERR
printf 'testing\n' > "$sequence/status"
pytest -q > "$sequence/pytest.log" 2>&1
printf 'preflighting\n' > "$sequence/status"
python -u src/hong2021_v61_preflight.py \
  --program "$program" \
  --repo "$repo" \
  --out "$preflight" \
  > "$sequence/preflight.log" 2>&1
jq -e '.status == "pass" and .grid_cells == 134 and .training_performed == false and .independent_gate_locked == true and .development_accessed == false' "$preflight" >/dev/null
trap - ERR
printf 'complete_preflight_pass\n' > "$sequence/status"
