#!/usr/bin/env bash
set -euo pipefail
tng=/gpfs/kjhan/IllustrisTNG/TNG100-1
sequence=$tng/evaluation/tng100_simba_swift_v70_latent_spatial_sequence
training=$tng/training/tng100_simba_swift_v70_latent_spatial
history=$training/history.json
status=$sequence/status
monitor=$sequence/train_supervisor.log
[[ $(hostname -s | tr '[:upper:]' '[:lower:]') == lageunha ]] || exit 1
tmux has-session -t hong_v70_train 2>/dev/null || exit 1
while tmux has-session -t hong_v70_train 2>/dev/null; do
  python - "$history" >>"$monitor" 2>&1 <<'PY'
import datetime, json, os, sys
stamp=datetime.datetime.now().astimezone().isoformat(timespec="seconds")
if not os.path.exists(sys.argv[1]):
    print(stamp, "history_absent", flush=True)
else:
    rows=json.load(open(sys.argv[1]))
    row=rows[-1]
    print(stamp, "step", row["step"], "loss", row["source_balanced_EDM_loss"], "scale", row["AMP_scale_after_update"], "peak", row["peak_allocated_bytes"], flush=True)
PY
  sleep 60
done
current=$(cat "$status" 2>/dev/null || true)
printf "%s training_tmux_exited status=%s\n" "$(date --iso-8601=seconds)" "$current" >>"$monitor"
[[ $current == complete_V70_fixed_training_pending_train_only_gate ]] || exit 2
exec bash /home/kjhan/BACKUP/CF4/scripts/hong2021_v70_train_gate_lageunha.sh
