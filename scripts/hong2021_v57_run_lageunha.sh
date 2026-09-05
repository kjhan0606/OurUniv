#!/usr/bin/env bash
set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
program="$repo/config/hong2021_v57_grid_tail_component_audit_program.json"
sequence=/gpfs/kjhan/IllustrisTNG/TNG100-1/evaluation/tng100_simba_swift_v57_grid_tail_component_audit_sequence
audit_root=/gpfs/kjhan/IllustrisTNG/TNG100-1/evaluation/tng100_simba_swift_v57_grid_tail_component_audit
audit="$audit_root/audit.json"

cd "$repo"
if [[ "$(hostname -s | tr '[:upper:]' '[:lower:]')" != "lageunha" ]]; then
  echo "V57 requires Lageunha" >&2
  exit 1
fi
if [[ -e "$sequence" || -e "$audit_root" ]]; then
  echo "V57 refuses existing output" >&2
  exit 1
fi
mkdir -p "$sequence"
printf 'testing\n' > "$sequence/status"
pytest -q > "$sequence/pytest.log" 2>&1
printf 'auditing\n' > "$sequence/status"
python -u src/hong2021_v57_grid_tail_component_audit.py \
  --program "$program" \
  --repo "$repo" \
  --out "$audit" \
  > "$sequence/audit.log" 2>&1
jq -e '.status == "complete_train_only_grid_tail_component_audit" and .independent_gate_locked == true and .development_accessed == false' "$audit" >/dev/null
printf 'complete\n' > "$sequence/status"
