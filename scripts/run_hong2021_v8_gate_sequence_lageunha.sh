#!/usr/bin/env bash
set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
tng=/gpfs/kjhan/IllustrisTNG/TNG100-1
simba=/gpfs/kjhan/CAMELS/SIMBA/L25n256
eagle=/gpfs/kjhan/EAGLE/RefL0100N1504
training=$tng/training/tng100_simba_v8_observable_context
evaluation=$tng/evaluation/tng100_simba_v8_observable_context
screening=$evaluation/development_screening
tng_data=$tng/derived/hong2021_v2/split00_l0_paper/tng100_validation.h5
tng_cache=$tng/derived/hong2021_v6/tng100_validation_laplacian_sigma2.h5
simba_dev_data=$simba/derived/hong2021_v1/simba_cv24_26_validation_all_observers.h5
simba_dev_cache=$simba/derived/hong2021_v1/simba_cv24_26_validation_laplacian_sigma2.h5
tng_indices=49,8,63,0,15,4,32,21,12,74,79,53,76,29,57,62
status=$evaluation/sequence_status.json

cd "$repo"
export PYTHONPATH=$repo/src
mkdir -p "$evaluation"

write_status() {
    local state=$1
    local detail=$2
    python - "$status" "$state" "$detail" <<'PY'
import json, socket, sys
from datetime import datetime, timezone
from pathlib import Path
path, state, detail = sys.argv[1:]
Path(path).write_text(json.dumps({
    "schema": "hong2021-v8-automatic-sequence-status-v1",
    "state": state,
    "detail": detail,
    "host": socket.gethostname(),
    "updated_utc": datetime.now(timezone.utc).isoformat(),
}, indent=2) + "\n")
PY
}

write_status waiting_for_training "$training"
while tmux has-session -t hong2021_v8_train 2>/dev/null; do
    sleep 30
done
python - "$training/run.json" <<'PY'
import json, sys
report = json.load(open(sys.argv[1]))
if report.get("status") != "complete":
    raise SystemExit(f"V8 training did not complete: {report.get('status')}")
PY

simba_dev_indices=$(python - <<'PY'
import json
print(",".join(map(str, json.load(open(
    "config/hong2021_simba_dev_representative16_v1.json"
))["indices"])))
PY
)

write_status screening_development "steps 500,2000,5000,10000"
for step in 000500 002000 005000 010000; do
    checkpoint=$training/validation_checkpoints/step_${step}.pt
    root=$screening/step_${step}
    mkdir -p "$root/tng" "$root/simba_dev"
    python src/hong2021_residual_v8_context.py sample \
        --data "$tng_data" --cache "$tng_cache" --checkpoint "$checkpoint" \
        --out "$root/tng/ensemble8_steps20.h5" --indices "$tng_indices" \
        --ensemble 8 --sampling-steps 20 --seed 5777 --device cuda
    python src/hong2021_residual_evaluate.py \
        --candidate "edm=$root/tng/ensemble8_steps20.h5" \
        --out "$root/tng/ensemble_evaluation" --voxel-mpc-h 0.3125
    python src/hong2021_residual_v8_context.py sample \
        --data "$simba_dev_data" --cache "$simba_dev_cache" \
        --checkpoint "$checkpoint" \
        --out "$root/simba_dev/ensemble8_steps20.h5" \
        --indices "$simba_dev_indices" \
        --ensemble 8 --sampling-steps 20 --seed 6777 --device cuda
    python src/hong2021_residual_evaluate.py \
        --candidate "edm=$root/simba_dev/ensemble8_steps20.h5" \
        --out "$root/simba_dev/ensemble_evaluation" --voxel-mpc-h 0.3125
done

python src/hong2021_v8_development_gate.py select \
    --root "$screening" --training "$training" \
    --steps 500 2000 5000 10000 --out "$evaluation/checkpoint_selection.json"
selected_checkpoint=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_checkpoint"])' "$evaluation/checkpoint_selection.json")
selected_step=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_step"])' "$evaluation/checkpoint_selection.json")

write_status confirming_development "selected step $selected_step"
full=$evaluation/development_confirmation
mkdir -p "$full/tng" "$full/simba_dev"
python src/hong2021_residual_v8_context.py sample \
    --data "$tng_data" --cache "$tng_cache" --checkpoint "$selected_checkpoint" \
    --out "$full/tng/ensemble16_steps40.h5" --indices "$tng_indices" \
    --ensemble 16 --sampling-steps 40 --seed 777 --device cuda
python src/hong2021_residual_evaluate.py \
    --candidate "edm=$full/tng/ensemble16_steps40.h5" \
    --out "$full/tng/ensemble_evaluation" --voxel-mpc-h 0.3125
python src/hong2021_residual_v8_context.py sample \
    --data "$simba_dev_data" --cache "$simba_dev_cache" \
    --checkpoint "$selected_checkpoint" \
    --out "$full/simba_dev/ensemble16_steps40.h5" \
    --indices "$simba_dev_indices" \
    --ensemble 16 --sampling-steps 40 --seed 1777 --device cuda
python src/hong2021_residual_evaluate.py \
    --candidate "edm=$full/simba_dev/ensemble16_steps40.h5" \
    --out "$full/simba_dev/ensemble_evaluation" --voxel-mpc-h 0.3125
python src/hong2021_v8_development_gate.py confirm \
    --checkpoint "$selected_checkpoint" \
    --tng-metrics "$full/tng/ensemble_evaluation/metrics.json" \
    --simba-metrics "$full/simba_dev/ensemble_evaluation/metrics.json" \
    --out "$full/decision.json"
if ! python - "$full/decision.json" <<'PY'
import json, sys
raise SystemExit(0 if json.load(open(sys.argv[1]))["both_field_gates_pass"] else 1)
PY
then
    write_status complete_failed_development "$full/decision.json"
    exit 0
fi

# CV0-15 was already inspected after V7.  It is a conservative stress test,
# not a new independent V8 test and never participates in model selection.
write_status testing_historical_simba_stress "$selected_checkpoint"
stress=$simba/evaluation/hong2021_v8_observable_context_historical_stress
mkdir -p "$stress"
python src/hong2021_residual_v8_context.py sample \
    --data "$simba/derived/hong2021_v1/simba_cv16.h5" \
    --cache "$simba/derived/hong2021_v1/simba_cv16_laplacian_sigma2.h5" \
    --checkpoint "$selected_checkpoint" \
    --out "$stress/ensemble16_steps40.h5" \
    --indices 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
    --ensemble 16 --sampling-steps 40 --seed 2777 --device cuda
python src/hong2021_residual_evaluate.py \
    --candidate "edm=$stress/ensemble16_steps40.h5" \
    --out "$stress/ensemble_evaluation" --voxel-mpc-h 0.3125
python src/hong2021_independent_gate.py \
    --ensemble-metrics "$stress/ensemble_evaluation/metrics.json" \
    --out "$stress/decision.json" --simulation CAMELS-SIMBA-CV-0-15-HISTORICAL-STRESS
if ! python - "$stress/decision.json" <<'PY'
import json, sys
result = json.load(open(sys.argv[1]))["candidates"]["edm"]["field_gate"]["pass"]
raise SystemExit(0 if result else 1)
PY
then
    write_status complete_failed_historical_simba_stress "$stress/decision.json"
    exit 0
fi

# This is the one-time opening of the predeclared 32-object EAGLE reserve.
write_status testing_eagle_confirmation_once "$selected_checkpoint"
eagle_eval=$eagle/evaluation/hong2021_v8_observable_context_confirmation32
mkdir -p "$eagle_eval"
eagle_indices=$(python - <<'PY'
import json
print(",".join(map(str, json.load(open(
    "config/hong2021_eagle_confirmation32_v1.json"
))["indices"])))
PY
)
python src/hong2021_residual_v8_context.py sample \
    --data "$eagle/derived/hong2021_v1/eagle_ref100_z0_test.h5" \
    --cache "$eagle/derived/hong2021_v1/eagle_ref100_z0_laplacian_sigma2.h5" \
    --checkpoint "$selected_checkpoint" \
    --out "$eagle_eval/ensemble16_steps40.h5" \
    --indices "$eagle_indices" --ensemble 16 --sampling-steps 40 \
    --seed 3777 --device cuda
python src/hong2021_residual_evaluate.py \
    --candidate "edm=$eagle_eval/ensemble16_steps40.h5" \
    --out "$eagle_eval/ensemble_evaluation" --voxel-mpc-h 0.3125
python src/hong2021_independent_gate.py \
    --ensemble-metrics "$eagle_eval/ensemble_evaluation/metrics.json" \
    --out "$eagle_eval/decision.json" --simulation EAGLE-RefL0100N1504-CONFIRMATION32
if ! python - "$eagle_eval/decision.json" <<'PY'
import json, sys
result = json.load(open(sys.argv[1]))["candidates"]["edm"]["field_gate"]["pass"]
raise SystemExit(0 if result else 1)
PY
then
    write_status complete_failed_eagle_confirmation "$eagle_eval/decision.json"
    exit 0
fi

write_status field_gates_pass_ready_for_grid_hop "$selected_checkpoint"
