#!/usr/bin/env bash
set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
tng=/gpfs/kjhan/IllustrisTNG/TNG100-1
evaluation=$tng/evaluation/tng100_simba_swift_v18_e6_prior_matched_init
sequence=$tng/evaluation/tng100_simba_swift_v18_sequence
status=$sequence/status.json
mkdir -p "$sequence"
cd "$repo"

mkdir -p /gpfs/kjhan/.hong2021_locks
exec 9>/gpfs/kjhan/.hong2021_locks/v18_e6_development.lock
if ! flock -n 9; then
  echo "Another V18-E6 development supervisor holds the lock." >&2
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
    "schema": "hong2021-v18-development-supervisor-status-v1",
    "state": state, "detail": detail, "host": socket.gethostname(),
    "updated_utc": datetime.now(timezone.utc).isoformat(),
}, indent=2) + "\n")
os.replace(temporary, path)
PY
}

failure() { write_status failed "line=$1 command=$2"; }
trap 'failure "$LINENO" "$BASH_COMMAND"' ERR

if [[ -e $sequence/status.json ]]; then
  echo "V18-E6 refuses a pre-existing sequence status: $sequence/status.json" >&2
  exit 1
fi
write_status running_predeclared_e6_sampling_gate "$evaluation"
bash scripts/run_hong2021_v18_e6_gate_lageunha.sh >"$sequence/gate.log" 2>&1

decision=$evaluation/development_decision.json
passed=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["development_pass"])' "$decision")
next=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["next"])' "$decision")
if [[ $passed == True ]]; then
  write_status complete_e6_passed_astrid_still_unopened "$decision"
else
  write_status complete_e6_failed_astrid_unopened "$next"
fi
