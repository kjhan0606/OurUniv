#!/usr/bin/env bash
set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
tng=/gpfs/kjhan/IllustrisTNG/TNG100-1
simba=/gpfs/kjhan/CAMELS/SIMBA/L25n256
eagle=/gpfs/kjhan/EAGLE/RefL0100N1504
training=$tng/training/tng100_simba_v9_tail_balanced
evaluation=$tng/evaluation/tng100_simba_v9_tail_balanced
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
    python - "$status" "$1" "$2" <<'PY'
import json, socket, sys
from datetime import datetime, timezone
from pathlib import Path
path, state, detail = sys.argv[1:]
Path(path).write_text(json.dumps({
    "schema": "hong2021-v9-automatic-sequence-status-v1",
    "state": state, "detail": detail, "host": socket.gethostname(),
    "updated_utc": datetime.now(timezone.utc).isoformat(),
}, indent=2) + "\n")
PY
}

write_status waiting_for_training "$training"
while tmux has-session -t hong2021_v9_train 2>/dev/null; do sleep 30; done
python - "$training/run.json" <<'PY'
import json, sys
state = json.load(open(sys.argv[1])).get("status")
if state != "complete": raise SystemExit(f"V9 training ended with {state}")
PY
simba_indices=$(python -c 'import json; print(",".join(map(str,json.load(open("config/hong2021_simba_dev_representative16_v1.json"))["indices"])))')

write_status full_fidelity_development_gate "steps 1000,3000,5000"
for step in 001000 003000 005000; do
    checkpoint=$training/validation_checkpoints/step_${step}.pt
    root=$evaluation/development_candidates/step_${step}
    mkdir -p "$root/tng" "$root/simba_dev"
    python src/hong2021_residual_v9_tail.py sample \
        --data "$tng_data" --cache "$tng_cache" --checkpoint "$checkpoint" \
        --out "$root/tng/ensemble16_steps40.h5" --indices "$tng_indices" \
        --ensemble 16 --sampling-steps 40 --seed 8777 --device cuda
    python src/hong2021_residual_evaluate.py \
        --candidate "edm=$root/tng/ensemble16_steps40.h5" \
        --out "$root/tng/ensemble_evaluation" --voxel-mpc-h 0.3125
    python src/hong2021_residual_v9_tail.py sample \
        --data "$simba_dev_data" --cache "$simba_dev_cache" \
        --checkpoint "$checkpoint" \
        --out "$root/simba_dev/ensemble16_steps40.h5" \
        --indices "$simba_indices" --ensemble 16 --sampling-steps 40 \
        --seed 9777 --device cuda
    python src/hong2021_residual_evaluate.py \
        --candidate "edm=$root/simba_dev/ensemble16_steps40.h5" \
        --out "$root/simba_dev/ensemble_evaluation" --voxel-mpc-h 0.3125
done
python src/hong2021_v9_development_gate.py \
    --root "$evaluation/development_candidates" --training "$training" \
    --steps 1000 3000 5000 --out "$evaluation/development_decision.json"
if ! python - "$evaluation/development_decision.json" <<'PY'
import json, sys
raise SystemExit(0 if json.load(open(sys.argv[1]))["development_pass"] else 1)
PY
then
    write_status complete_failed_development "$evaluation/development_decision.json"
    exit 0
fi
checkpoint=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_checkpoint"])' "$evaluation/development_decision.json")

write_status testing_historical_simba_stress "$checkpoint"
stress=$simba/evaluation/hong2021_v9_tail_balanced_historical_stress
mkdir -p "$stress"
python src/hong2021_residual_v9_tail.py sample \
    --data "$simba/derived/hong2021_v1/simba_cv16.h5" \
    --cache "$simba/derived/hong2021_v1/simba_cv16_laplacian_sigma2.h5" \
    --checkpoint "$checkpoint" --out "$stress/ensemble16_steps40.h5" \
    --indices 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
    --ensemble 16 --sampling-steps 40 --seed 10777 --device cuda
python src/hong2021_residual_evaluate.py \
    --candidate "edm=$stress/ensemble16_steps40.h5" \
    --out "$stress/ensemble_evaluation" --voxel-mpc-h 0.3125
python src/hong2021_independent_gate.py \
    --ensemble-metrics "$stress/ensemble_evaluation/metrics.json" \
    --out "$stress/decision.json" --simulation CAMELS-SIMBA-CV-0-15-HISTORICAL-STRESS
if ! python - "$stress/decision.json" <<'PY'
import json, sys
passed=json.load(open(sys.argv[1]))["candidates"]["edm"]["field_gate"]["pass"]
raise SystemExit(0 if passed else 1)
PY
then
    write_status complete_failed_historical_simba_stress "$stress/decision.json"
    exit 0
fi

write_status testing_sealed_eagle_confirmation_once "$checkpoint"
eagle_eval=$eagle/evaluation/hong2021_v9_tail_balanced_confirmation32
mkdir -p "$eagle_eval"
eagle_indices=$(python -c 'import json; print(",".join(map(str,json.load(open("config/hong2021_eagle_confirmation32_v1.json"))["indices"])))')
python src/hong2021_residual_v9_tail.py sample \
    --data "$eagle/derived/hong2021_v1/eagle_ref100_z0_test.h5" \
    --cache "$eagle/derived/hong2021_v1/eagle_ref100_z0_laplacian_sigma2.h5" \
    --checkpoint "$checkpoint" --out "$eagle_eval/ensemble16_steps40.h5" \
    --indices "$eagle_indices" --ensemble 16 --sampling-steps 40 \
    --seed 11777 --device cuda
python src/hong2021_residual_evaluate.py \
    --candidate "edm=$eagle_eval/ensemble16_steps40.h5" \
    --out "$eagle_eval/ensemble_evaluation" --voxel-mpc-h 0.3125
python src/hong2021_independent_gate.py \
    --ensemble-metrics "$eagle_eval/ensemble_evaluation/metrics.json" \
    --out "$eagle_eval/decision.json" \
    --simulation EAGLE-RefL0100N1504-CONFIRMATION32
if ! python - "$eagle_eval/decision.json" <<'PY'
import json, sys
passed=json.load(open(sys.argv[1]))["candidates"]["edm"]["field_gate"]["pass"]
raise SystemExit(0 if passed else 1)
PY
then
    write_status complete_failed_eagle_confirmation "$eagle_eval/decision.json"
    exit 0
fi
write_status field_gates_pass_ready_for_grid_hop "$checkpoint"
