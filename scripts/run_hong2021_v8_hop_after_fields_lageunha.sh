#!/usr/bin/env bash
set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
tng=/gpfs/kjhan/IllustrisTNG/TNG100-1
simba=/gpfs/kjhan/CAMELS/SIMBA/L25n256
eagle=/gpfs/kjhan/EAGLE/RefL0100N1504
evaluation=$tng/evaluation/tng100_simba_v8_observable_context
full=$evaluation/development_confirmation
stress=$simba/evaluation/hong2021_v8_observable_context_historical_stress
eagle_eval=$eagle/evaluation/hong2021_v8_observable_context_confirmation32
status=$evaluation/sequence_status.json
hop_adapter=$tng/tools/hop_grid_adapter

cd "$repo"
export PYTHONPATH=$repo/src

while tmux has-session -t hong2021_v8_gate 2>/dev/null; do
    sleep 30
done
state=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["state"])' "$status")
if [[ "$state" != field_gates_pass_ready_for_grid_hop ]]; then
    printf 'V8 grid-HOP not entered: field sequence ended at %s\n' "$state"
    exit 0
fi

python src/hong2021_hop_grid_gate.py \
    --edm "$full/tng/ensemble16_steps40.h5" \
    --out "$full/tng/grid_hop_gate.json" --work "$full/tng/grid_hop_work" \
    --hop-dir "$hop_adapter" --members 16 --objects 16 --workers 8 \
    --voxel-mpc-h 0.3125 --omega-m 0.3
python src/hong2021_hop_grid_gate.py \
    --edm "$stress/ensemble16_steps40.h5" \
    --out "$stress/grid_hop_gate.json" --work "$stress/grid_hop_work" \
    --hop-dir "$hop_adapter" --members 16 --objects 16 --workers 8 \
    --voxel-mpc-h 0.3125 --omega-m 0.3
python src/hong2021_hop_grid_gate.py \
    --edm "$eagle_eval/ensemble16_steps40.h5" \
    --out "$eagle_eval/grid_hop_gate.json" --work "$eagle_eval/grid_hop_work" \
    --hop-dir "$hop_adapter" --members 16 --objects 32 --workers 8 \
    --voxel-mpc-h 0.3125 --omega-m 0.307

python src/hong2021_independent_gate.py \
    --ensemble-metrics "$full/tng/ensemble_evaluation/metrics.json" \
    --hop "$full/tng/grid_hop_gate.json" --out "$full/tng/final_decision.json" \
    --simulation TNG100-DEVELOPMENT --bootstrap 50000 --seed 2021
python src/hong2021_independent_gate.py \
    --ensemble-metrics "$stress/ensemble_evaluation/metrics.json" \
    --hop "$stress/grid_hop_gate.json" --out "$stress/final_decision.json" \
    --simulation CAMELS-SIMBA-CV-0-15-HISTORICAL-STRESS --bootstrap 50000 --seed 2022
python src/hong2021_independent_gate.py \
    --ensemble-metrics "$eagle_eval/ensemble_evaluation/metrics.json" \
    --hop "$eagle_eval/grid_hop_gate.json" \
    --out "$eagle_eval/final_decision.json" \
    --simulation EAGLE-RefL0100N1504-CONFIRMATION32 \
    --bootstrap 50000 --seed 2023

python - "$evaluation/final_gate_decision.json" \
    "$full/tng/final_decision.json" "$stress/final_decision.json" \
    "$eagle_eval/final_decision.json" <<'PY'
import json, sys
from pathlib import Path

out = Path(sys.argv[1])
labels = ("tng_development", "historical_simba_stress", "eagle_confirmation32")
paths = [Path(value) for value in sys.argv[2:]]
domains = {}
for label, path in zip(labels, paths, strict=True):
    report = json.loads(path.read_text())
    result = report["candidates"]["edm"]
    domains[label] = {
        "decision": str(path.resolve()),
        "field_pass": result["field_gate"]["pass"],
        "grid_hop_pass": result["grid_hop_gate"]["pass"],
        "overall_pass": result["overall_pass"],
    }
accepted = all(value["overall_pass"] for value in domains.values())
result = {
    "schema": "hong2021-v8-final-three-domain-gate-v1",
    "domains": domains,
    "accepted": accepted,
    "advance_to_forward_particle_dynamics": accepted,
}
out.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
PY
