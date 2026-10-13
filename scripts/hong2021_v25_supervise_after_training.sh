#!/usr/bin/env bash
set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
tng=/gpfs/kjhan/IllustrisTNG/TNG100-1
training=$tng/training/tng100_simba_swift_v25_e13_unweighted_edm
evaluation=$tng/evaluation/tng100_simba_swift_v25_e13_unweighted
sequence=$tng/evaluation/tng100_simba_swift_v25_sequence
status=$sequence/status.json
mkdir -p "$sequence" /gpfs/kjhan/.hong2021_locks
cd "$repo"
exec 9>/gpfs/kjhan/.hong2021_locks/v25_e13_development.lock
flock -n 9 || { echo "another V25 supervisor holds the lock" >&2; exit 2; }

write_status() {
  python - "$status" "$1" "$2" <<'PY'
import json, os, socket, sys
from datetime import datetime, timezone
from pathlib import Path
path, state, detail = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
partial = path.with_suffix(".json.partial")
partial.write_text(json.dumps({
 "schema":"hong2021-v25-development-supervisor-status-v1", "state":state,
 "detail":detail, "host":socket.gethostname(), "independent_data_paths_accessed":False,
 "updated_utc":datetime.now(timezone.utc).isoformat()}, indent=2)+"\n")
os.replace(partial,path)
PY
}
trap 'write_status failed "line=$LINENO command=$BASH_COMMAND"' ERR
write_status awaiting_or_monitoring_training "$training"
for _ in $(seq 1 30); do
  if pgrep -f "src/hong2021_v25_edm.py.*$training" >/dev/null || [[ -e $training/run.json ]]; then
    break
  fi
  sleep 2
done
if ! pgrep -f "src/hong2021_v25_edm.py.*$training" >/dev/null && [[ ! -e $training/run.json ]]; then
  write_status training_start_not_observed "$training"
  exit 1
fi
while pgrep -f "src/hong2021_v25_edm.py.*$training" >/dev/null; do sleep 60; done
state=$(python -c 'import json,sys; print(json.load(open(sys.argv[1])).get("status"))' "$training/run.json")
if [[ $state != complete ]]; then
  write_status training_failed "$state"
  exit 1
fi
write_status running_development_sampling_gate "$evaluation"
bash scripts/hong2021_v25_gate.sh >"$sequence/gate.log" 2>&1
decision=$evaluation/development_decision.json
passed=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["development_pass"])' "$decision")
if [[ $passed == True ]]; then
  write_status complete_passed_awaiting_user_approval "$decision"
else
  audit=$evaluation/automatic_failure_audit.json
  write_status running_automatic_failure_audit "$audit"
  python scripts/hong2021_v25_failure_audit.py --decision "$decision" \
    --out "$audit" >"$evaluation/automatic_failure_audit.log" 2>&1
  write_status complete_failed_audited_astrid_unopened "$audit"
fi
