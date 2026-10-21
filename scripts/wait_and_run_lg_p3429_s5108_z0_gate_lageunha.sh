#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname | tr '[:upper:]' '[:lower:]')" != lageunha ]]; then
    echo "This gate watcher must execute on Lageunha." >&2
    exit 2
fi

repo=/home/kjhan/BACKUP/CF4
python=/home/kjhan/miniconda3/envs/circle/bin/python
work=/gpfs/kjhan/CF4/recon/lg_p3429_s5108_z0_gate_v1
hop_work="$work/hop_work"
watch_log="$work/gate_watch.log"

mkdir -p "$work"
rm -f "$work/GATE_COMPLETE" "$work/GATE_FAILED"

fail() {
    printf '%s %s\n' "$(date --iso-8601=seconds)" "$1" \
        | tee -a "$watch_log" >"$work/GATE_FAILED"
    exit 1
}

echo "$(date --iso-8601=seconds) waiting for HOP products" | tee -a "$watch_log"
while [[ ! -s "$hop_work/HOP_COMPLETE" ]]; do
    if [[ -s "$hop_work/HOP_FAILED" ]]; then
        fail "HOP_FAILED marker detected"
    fi
    if ! tmux has-session -t cf4_hop_5108_z0 2>/dev/null; then
        fail "HOP tmux exited without a completion marker"
    fi
    sleep 60
done

echo "$(date --iso-8601=seconds) starting HOP/M200c gate v2" | tee -a "$watch_log"
if ! "$python" "$repo/src/cf4_zoom_z0_gate_v2.py" --reuse-catalog \
    >"$work/gate_v2.log" 2>&1; then
    fail "HOP/M200c gate v2 failed; inspect gate_v2.log"
fi

echo "$(date --iso-8601=seconds) starting recentered P1 environment gate" \
    | tee -a "$watch_log"
if ! "$python" "$repo/src/cf4_zoom_recenter_p1.py" \
    >"$work/environment_v2.log" 2>&1; then
    fail "recentered P1 gate failed; inspect environment_v2.log"
fi

{
    echo "$(date --iso-8601=seconds) complete"
    "$python" - <<'PY'
import json
path = "/gpfs/kjhan/CF4/recon/lg_p3429_s5108_z0_gate_v1/gate_result_v2.json"
result = json.load(open(path))
print(json.dumps(result["verdict"], sort_keys=True))
PY
} | tee -a "$watch_log" >"$work/GATE_COMPLETE"
