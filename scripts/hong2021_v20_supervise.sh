#!/usr/bin/env bash
set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
tng=/gpfs/kjhan/IllustrisTNG/TNG100-1
training=$tng/training/tng100_simba_swift_v20_e8_gaussianized_marginal_edm
evaluation=$tng/evaluation/tng100_simba_swift_v20_e8_gaussianized_marginal
sequence=$tng/evaluation/tng100_simba_swift_v20_sequence
status=$sequence/status.json
seal=$repo/config/hong2021_v20_astrid_one_shot_seal.json
if [[ -e $sequence || -e $seal ]]; then
  echo "V20-E8 refuses a pre-existing sequence or seal" >&2
  exit 1
fi
mkdir -p "$sequence"
cd "$repo"

mkdir -p /gpfs/kjhan/.hong2021_locks
exec 9>/gpfs/kjhan/.hong2021_locks/v20_e8_development.lock
if ! flock -n 9; then
  echo "Another V20-E8 development supervisor holds the lock." >&2
  exit 2
fi

write_status() {
  python - "$status" "$1" "$2" <<'PY'
import json, os, socket, sys
from datetime import datetime, timezone
from pathlib import Path
path, state, detail = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
temporary = path.with_suffix(".json.partial")
temporary.write_text(json.dumps({
    "schema": "hong2021-v20-development-supervisor-status-v1",
    "state": state, "detail": detail, "host": socket.gethostname(),
    "independent_data_paths_accessed": False,
    "updated_utc": datetime.now(timezone.utc).isoformat(),
}, indent=2) + "\n")
os.replace(temporary, path)
PY
}

failure() { write_status failed "line=$1 command=$2"; }
trap 'failure "$LINENO" "$BASH_COMMAND"' ERR

if [[ -e $training || -e $evaluation ]]; then
  echo "V20-E8 refuses pre-existing training or evaluation output" >&2
  exit 1
fi
write_status running_hard_preflight "$sequence/preflight.json"
bash scripts/hong2021_v20_preflight.sh "$sequence/preflight.json" \
  >"$sequence/preflight.log" 2>&1
write_status running_predeclared_e8_training "$training"
bash scripts/hong2021_v20_train_lageunha.sh >"$sequence/training.log" 2>&1
write_status running_predeclared_e8_sampling_gate "$evaluation"
bash scripts/hong2021_v20_gate.sh >"$sequence/gate.log" 2>&1

decision=$evaluation/development_decision.json
passed=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["development_pass"])' "$decision")
next=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["next"])' "$decision")
if [[ $passed == True ]]; then
  write_status complete_e8_passed_astrid_still_unopened "$decision"
  exec scripts/hong2021_v20_finalize.sh
else
  write_status complete_e8_failed_astrid_unopened "$next"
fi
