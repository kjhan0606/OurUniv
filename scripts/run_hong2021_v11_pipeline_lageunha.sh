#!/usr/bin/env bash
set -euo pipefail
repo=/home/kjhan/BACKUP/CF4
tng=/gpfs/kjhan/IllustrisTNG/TNG100-1
v10_eval=$tng/evaluation/tng100_simba_v10_twocomponent
v11_eval=$tng/evaluation/tng100_simba_v11_recentered
log=$tng/training/hong2021_v11_pipeline.log
cd "$repo"
export PYTHONPATH=$repo/src
mkdir -p "$v11_eval"
exec > >(tee -a "$log") 2>&1
printf '[v11-supervisor] start %s host=%s\n' "$(date --iso-8601=seconds)" "$(hostname)"

while tmux has-session -t hong2021_v10_gate 2>/dev/null; do
    sleep 30
done
v10_state=$(python -c 'import json,sys;print(json.load(open(sys.argv[1]))["state"])' "$v10_eval/sequence_status.json")
if [[ "$v10_state" != complete_failed_development ]]; then
    printf '[v11-supervisor] not entered because V10 ended at %s\n' "$v10_state"
    exit 0
fi

python - "$v11_eval/sequence_status.json" <<'PY'
import json,socket,sys
from datetime import datetime,timezone
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({
  "schema":"hong2021-v11-automatic-sequence-status-v1",
  "state":"preparing_recentered_development_caches",
  "detail":"TNG and SIMBA development only",
  "host":socket.gethostname(),"updated_utc":datetime.now(timezone.utc).isoformat()
},indent=2)+"\n")
PY
bash scripts/run_hong2021_v11_prepare_lageunha.sh
bash scripts/run_hong2021_v11_smoke_lageunha.sh
bash scripts/run_hong2021_v11_train_lageunha.sh
bash scripts/run_hong2021_v11_gate_sequence_lageunha.sh
printf '[v11-supervisor] end %s host=%s\n' "$(date --iso-8601=seconds)" "$(hostname)"
