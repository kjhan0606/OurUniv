#!/usr/bin/env bash
set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
tng=/gpfs/kjhan/IllustrisTNG/TNG100-1
training=$tng/training/tng100_simba_swift_v17_e5_band_balanced_edm
evaluation=$tng/evaluation/tng100_simba_swift_v17_e5_band_balanced_edm
sequence=$tng/evaluation/tng100_simba_swift_v17_sequence
status=$sequence/status.json
mkdir -p "$sequence"
cd "$repo"

write_status() {
  python - "$status" "$1" "$2" <<'PY'
import json, os, socket, sys
from datetime import datetime, timezone
from pathlib import Path
path, state, detail = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
temporary = path.with_suffix(".json.partial")
temporary.write_text(json.dumps({
    "schema": "hong2021-v17-development-supervisor-status-v1",
    "state": state, "detail": detail, "host": socket.gethostname(),
    "updated_utc": datetime.now(timezone.utc).isoformat(),
}, indent=2) + "\n")
os.replace(temporary, path)
PY
}

failure() { write_status failed "line=$1 command=$2"; }
trap 'failure "$LINENO" "$BASH_COMMAND"' ERR

if [[ -e $training/last.pt || -e $training/run.json ]]; then
  write_status failed_preexisting_training "refusing to overwrite $training"
  exit 1
fi
write_status running_predeclared_e5_training "$training"
bash scripts/run_hong2021_v17_e5_lageunha.sh >"$sequence/training.log" 2>&1
write_status running_e5_three_domain_gate "$evaluation"
bash scripts/run_hong2021_v17_e5_gate_lageunha.sh >"$sequence/gate.log" 2>&1

decision=$evaluation/development_decision.json
passed=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["development_pass"])' "$decision")
next=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["next"])' "$decision")
if [[ $passed == True ]]; then
  write_status complete_e5_passed_astrid_still_unopened "$decision"
else
  write_status complete_e5_failed_astrid_unopened "$next"
fi
