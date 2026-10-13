#!/usr/bin/env bash
set -euo pipefail
repo=/home/kjhan/BACKUP/CF4
tng=/gpfs/kjhan/IllustrisTNG/TNG100-1
evaluation=$tng/evaluation/tng100_simba_v12_gaussianized
log=$tng/training/hong2021_v12_pipeline.log
cd "$repo"
export PYTHONPATH=$repo/src
mkdir -p "$evaluation"
exec > >(tee -a "$log") 2>&1
while tmux has-session -t hong2021_v12_prepare 2>/dev/null; do sleep 20; done
test -s "$tng/training/tng100_simba_v12_gaussianized_smoke/smoke_ensemble.h5"
python - "$evaluation/sequence_status.json" <<'PY'
import json,socket,sys
from datetime import datetime,timezone
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({
 "schema":"hong2021-v12-automatic-sequence-status-v1","state":"training",
 "detail":"10000 steps; validation every 500","host":socket.gethostname(),
 "updated_utc":datetime.now(timezone.utc).isoformat()
},indent=2)+"\n")
PY
bash scripts/run_hong2021_v12_train_lageunha.sh
bash scripts/run_hong2021_v12_gate_sequence_lageunha.sh
