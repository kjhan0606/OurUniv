#!/usr/bin/env bash
set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
tng=/gpfs/kjhan/IllustrisTNG/TNG100-1
training=$tng/training/tng100_simba_swift_v27_e15_parent_aligned_haar_flow
evaluation=$tng/evaluation/tng100_simba_swift_v27_e15_parent_aligned_haar_flow
sequence=$tng/evaluation/tng100_simba_swift_v27_sequence
status=$sequence/status.json
mkdir -p "$sequence" /gpfs/kjhan/.hong2021_locks
cd "$repo"
exec 9>/gpfs/kjhan/.hong2021_locks/v27_e15_development.lock
flock -n 9 || { echo "another V27 supervisor holds the lock" >&2; exit 2; }

write_status() {
  python - "$status" "$1" "$2" <<'PY'
import json, os, socket, sys
from datetime import datetime, timezone
from pathlib import Path
path, state, detail = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
partial = path.with_suffix(".json.partial")
partial.write_text(json.dumps({
 "schema":"hong2021-v27-development-supervisor-status-v1", "state":state,
 "detail":detail, "host":socket.gethostname(), "independent_data_paths_accessed":False,
 "updated_utc":datetime.now(timezone.utc).isoformat()}, indent=2)+"\n")
os.replace(partial,path)
PY
}
trap 'write_status failed "line=$LINENO command=$BASH_COMMAND"' ERR
write_status awaiting_or_monitoring_training "$training"
for _ in $(seq 1 30); do
  if pgrep -f "src/hong2021_v27.py train.*$training" >/dev/null || [[ -e $training/run.json ]]; then
    break
  fi
  sleep 2
done
if ! pgrep -f "src/hong2021_v27.py train.*$training" >/dev/null && [[ ! -e $training/run.json ]]; then
  write_status training_start_not_observed "$training"
  exit 1
fi
while pgrep -f "src/hong2021_v27.py train.*$training" >/dev/null; do
  detail=$(python - "$training/history.json" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
if not path.is_file():
    print("step=0/30000")
else:
    rows = json.loads(path.read_text())
    print(f"step={rows[-1]['step']}/30000 balanced_validation_nll={rows[-1]['balanced_validation_nll']:.8f}")
PY
)
  write_status training_running "$detail"
  sleep 60
done
state=$(python -c 'import json,sys; print(json.load(open(sys.argv[1])).get("status"))' "$training/run.json")
if [[ $state != complete ]]; then
  write_status training_failed "$state"
  exit 1
fi
write_status running_development_sampling_gate "$evaluation"
bash scripts/hong2021_v27_gate.sh >"$sequence/gate.log" 2>&1
decision=$evaluation/development_decision.json
passed=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["development_pass"])' "$decision")
if [[ $passed == True ]]; then
  write_status complete_passed_awaiting_user_approval "$decision"
else
  audit=$evaluation/automatic_failure_audit.json
  write_status running_automatic_failure_audit "$audit"
  python scripts/hong2021_v27_failure_audit.py --decision "$decision" \
    --out "$audit" >"$evaluation/automatic_failure_audit.log" 2>&1
  write_status complete_failed_audited_astrid_unopened "$audit"
fi
