#!/usr/bin/env bash
set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
tng=/gpfs/kjhan/IllustrisTNG/TNG100-1
root=$tng/evaluation/tng100_simba_swift_v15_sequence
status=$root/status.json
e2_training=$tng/training/tng100_simba_swift_v15_e2_relative_noise_edm
e2_evaluation=$tng/evaluation/tng100_simba_swift_v15_e2_relative_noise_edm
e3_training=$tng/training/tng100_simba_swift_v15_e3_relative_noise_tail_quarter_edm
e3_evaluation=$tng/evaluation/tng100_simba_swift_v15_e3_relative_noise_tail_quarter_edm
mkdir -p "$root"
cd "$repo"

write_status() {
  python - "$status" "$1" "$2" <<'PY'
import json, os, socket, sys
from datetime import datetime, timezone
from pathlib import Path
path, state, detail = sys.argv[1:]
temporary = Path(path).with_suffix(".json.partial")
temporary.write_text(json.dumps({
    "schema": "hong2021-v15-development-supervisor-status-v1",
    "state": state,
    "detail": detail,
    "host": socket.gethostname(),
    "updated_utc": datetime.now(timezone.utc).isoformat(),
}, indent=2) + "\n")
os.replace(temporary, path)
PY
}

failure() {
  write_status failed "line=$1 command=$2"
}
trap 'failure "$LINENO" "$BASH_COMMAND"' ERR

write_status waiting_for_e2_training "$e2_training/run.json"
while true; do
  if [[ -s $e2_training/run.json ]]; then
    state=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$e2_training/run.json")
    if [[ $state == complete ]]; then
      break
    fi
  fi
  if ! tmux has-session -t hong_v15_e2 2>/dev/null; then
    write_status failed_e2_training "E2 tmux ended without complete run.json"
    exit 1
  fi
  sleep 30
done

write_status running_e2_gate "$e2_evaluation"
bash scripts/run_hong2021_v15_variant_gate_lageunha.sh e2
e2_pass=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["development_pass"])' "$e2_evaluation/development_decision.json")
if [[ $e2_pass == True ]]; then
  write_status complete_e2_passed_astrid_still_unopened "$e2_evaluation/development_decision.json"
  exit 0
fi

if [[ -e $e3_training/last.pt || -e $e3_training/run.json ]]; then
  write_status failed_e3_preexisting "refusing to overwrite $e3_training"
  exit 1
fi
write_status running_predeclared_e3_training "$e3_training"
bash scripts/run_hong2021_v15_e3_lageunha.sh
write_status running_e3_gate "$e3_evaluation"
bash scripts/run_hong2021_v15_variant_gate_lageunha.sh e3
e3_pass=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["development_pass"])' "$e3_evaluation/development_decision.json")
if [[ $e3_pass == True ]]; then
  write_status complete_e3_passed_astrid_still_unopened "$e3_evaluation/development_decision.json"
else
  write_status complete_e3_failed_astrid_unopened "$e3_evaluation/development_decision.json"
fi
