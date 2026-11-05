#!/usr/bin/env bash
set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
tng=/gpfs/kjhan/IllustrisTNG/TNG100-1
simba=/gpfs/kjhan/CAMELS/SIMBA/L25n256
eagle=/gpfs/kjhan/EAGLE/RefL0100N1504
training=$tng/training/tng100_simba_v12_gaussianized
v12_evaluation=$tng/evaluation/tng100_simba_v12_gaussianized
evaluation=$tng/evaluation/tng100_simba_v13_dc_corrected
model=$tng/derived/hong2021_v13/observable_dc_ridge.json
correction=$tng/training/tng100_simba_v10_twocomponent/validation_checkpoints/step_005000.pt
status=$evaluation/sequence_status.json
hop_adapter=$tng/tools/hop_grid_adapter

cd "$repo"
export PYTHONPATH=$repo/src
mkdir -p "$evaluation"

write_status() {
    python - "$status" "$1" "$2" <<'PY'
import json,socket,sys
from datetime import datetime,timezone
from pathlib import Path
p,s,d=sys.argv[1:]
Path(p).write_text(json.dumps({
 "schema":"hong2021-v13-automatic-sequence-status-v1","state":s,
 "detail":d,"host":socket.gethostname(),
 "updated_utc":datetime.now(timezone.utc).isoformat()
},indent=2)+"\n")
PY
}

while tmux has-session -t hong2021_v12_pipeline 2>/dev/null; do sleep 20; done
while tmux has-session -t hong2021_v13_fit 2>/dev/null; do sleep 20; done
test -s "$model"
v12_state=$(python -c 'import json,sys;print(json.load(open(sys.argv[1]))["state"])' \
    "$v12_evaluation/sequence_status.json")
if [[ "$v12_state" != complete_failed_development ]]; then
    write_status stopped_parent_v12_did_not_fail_development "$v12_state"
    exit 0
fi

apply_one() {
    local parent=$1 data=$2 cache=$3 output=$4
    python src/hong2021_dc_correction.py apply \
        --ensemble "$parent" --data "$data" --preprocessing-cache "$cache" \
        --model "$model" --out "$output"
}

evaluate_one() {
    local ensemble=$1 output=$2
    python src/hong2021_residual_evaluate.py \
        --candidate "edm=$ensemble" --out "$output" --voxel-mpc-h 0.3125
}

write_status applying_frozen_dc_model_to_development "steps 2000,5000,10000"
for step in 002000 005000 010000; do
    parent=$v12_evaluation/development_candidates/step_${step}
    root=$evaluation/development_candidates/step_${step}
    mkdir -p "$root/tng" "$root/simba_dev"
    apply_one "$parent/tng/ensemble16_steps40.h5" \
        "$tng/derived/hong2021_v2/split00_l0_paper/tng100_validation.h5" \
        "$tng/derived/hong2021_v11/tng100_validation_corrected_fullband.h5" \
        "$root/tng/ensemble16_steps40.h5"
    evaluate_one "$root/tng/ensemble16_steps40.h5" \
        "$root/tng/ensemble_evaluation"
    apply_one "$parent/simba_dev/ensemble16_steps40.h5" \
        "$simba/derived/hong2021_v1/simba_cv24_26_validation_all_observers.h5" \
        "$simba/derived/hong2021_v11/simba_cv24_26_validation_corrected_fullband.h5" \
        "$root/simba_dev/ensemble16_steps40.h5"
    evaluate_one "$root/simba_dev/ensemble16_steps40.h5" \
        "$root/simba_dev/ensemble_evaluation"
done

python src/hong2021_v9_development_gate.py \
    --root "$evaluation/development_candidates" --training "$training" \
    --steps 2000 5000 10000 \
    --report-schema hong2021-v13-full-fidelity-development-selection-v1 \
    --out "$evaluation/development_decision.json"
if ! python - "$evaluation/development_decision.json" <<'PY'
import json,sys
raise SystemExit(0 if json.load(open(sys.argv[1]))["development_pass"] else 1)
PY
then
    write_status complete_failed_development "$evaluation/development_decision.json"
    exit 0
fi

checkpoint=$(python -c 'import json,sys;print(json.load(open(sys.argv[1]))["selected_checkpoint"])' \
    "$evaluation/development_decision.json")
selected_step=$(python -c 'import json,sys;print(json.load(open(sys.argv[1]))["selected_step"])' \
    "$evaluation/development_decision.json")
selected_root=$evaluation/development_candidates/step_$(printf '%06d' "$selected_step")

write_status testing_historical_simba_stress "$checkpoint"
stress=$simba/evaluation/hong2021_v13_dc_corrected_historical_stress
mkdir -p "$stress"
python src/hong2021_residual_v12_gaussianized.py sample \
    --data "$simba/derived/hong2021_v1/simba_cv16.h5" \
    --original-cache "$simba/derived/hong2021_v1/simba_cv16_laplacian_sigma2.h5" \
    --correction-checkpoint "$correction" --checkpoint "$checkpoint" \
    --out "$stress/parent_v12_ensemble16_steps40.h5" \
    --indices 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
    --ensemble 16 --sampling-steps 40 --seed 26777 --device cuda
apply_one "$stress/parent_v12_ensemble16_steps40.h5" \
    "$simba/derived/hong2021_v1/simba_cv16.h5" \
    "$simba/derived/hong2021_v1/simba_cv16_laplacian_sigma2.h5" \
    "$stress/ensemble16_steps40.h5"
evaluate_one "$stress/ensemble16_steps40.h5" "$stress/ensemble_evaluation"
python src/hong2021_independent_gate.py \
    --ensemble-metrics "$stress/ensemble_evaluation/metrics.json" \
    --out "$stress/decision.json" \
    --simulation CAMELS-SIMBA-CV-0-15-HISTORICAL-STRESS
if ! python - "$stress/decision.json" <<'PY'
import json,sys
raise SystemExit(0 if json.load(open(sys.argv[1]))["candidates"]["edm"]["field_gate"]["pass"] else 1)
PY
then
    write_status complete_failed_historical_simba_stress "$stress/decision.json"
    exit 0
fi

write_status testing_sealed_eagle_confirmation_once "$checkpoint"
eagle_eval=$eagle/evaluation/hong2021_v13_dc_corrected_confirmation32
mkdir -p "$eagle_eval"
eagle_indices=$(python -c 'import json;print(",".join(map(str,json.load(open("config/hong2021_eagle_confirmation32_v1.json"))["indices"])))')
python src/hong2021_residual_v12_gaussianized.py sample \
    --data "$eagle/derived/hong2021_v1/eagle_ref100_z0_test.h5" \
    --original-cache "$eagle/derived/hong2021_v1/eagle_ref100_z0_laplacian_sigma2.h5" \
    --correction-checkpoint "$correction" --checkpoint "$checkpoint" \
    --out "$eagle_eval/parent_v12_ensemble16_steps40.h5" \
    --indices "$eagle_indices" --ensemble 16 --sampling-steps 40 \
    --seed 27777 --device cuda
apply_one "$eagle_eval/parent_v12_ensemble16_steps40.h5" \
    "$eagle/derived/hong2021_v1/eagle_ref100_z0_test.h5" \
    "$eagle/derived/hong2021_v1/eagle_ref100_z0_laplacian_sigma2.h5" \
    "$eagle_eval/ensemble16_steps40.h5"
evaluate_one "$eagle_eval/ensemble16_steps40.h5" "$eagle_eval/ensemble_evaluation"
python src/hong2021_independent_gate.py \
    --ensemble-metrics "$eagle_eval/ensemble_evaluation/metrics.json" \
    --out "$eagle_eval/decision.json" \
    --simulation EAGLE-RefL0100N1504-CONFIRMATION32
if ! python - "$eagle_eval/decision.json" <<'PY'
import json,sys
raise SystemExit(0 if json.load(open(sys.argv[1]))["candidates"]["edm"]["field_gate"]["pass"] else 1)
PY
then
    write_status complete_failed_eagle_confirmation "$eagle_eval/decision.json"
    exit 0
fi

write_status running_grid_hop "TNG, historical SIMBA, EAGLE confirmation32"
run_hop() {
    local ensemble=$1 output=$2 work=$3 objects=$4 omega=$5
    python src/hong2021_hop_grid_gate.py --edm "$ensemble" --out "$output" \
        --work "$work" --hop-dir "$hop_adapter" --members 16 \
        --objects "$objects" --workers 8 --voxel-mpc-h 0.3125 --omega-m "$omega"
}
run_hop "$selected_root/tng/ensemble16_steps40.h5" \
    "$selected_root/tng/grid_hop_gate.json" "$selected_root/tng/grid_hop_work" 16 0.3
run_hop "$stress/ensemble16_steps40.h5" \
    "$stress/grid_hop_gate.json" "$stress/grid_hop_work" 16 0.3
run_hop "$eagle_eval/ensemble16_steps40.h5" \
    "$eagle_eval/grid_hop_gate.json" "$eagle_eval/grid_hop_work" 32 0.307

python src/hong2021_independent_gate.py \
    --ensemble-metrics "$selected_root/tng/ensemble_evaluation/metrics.json" \
    --hop "$selected_root/tng/grid_hop_gate.json" \
    --out "$selected_root/tng/final_decision.json" \
    --simulation TNG100-DEVELOPMENT --bootstrap 50000 --seed 2021
python src/hong2021_independent_gate.py \
    --ensemble-metrics "$stress/ensemble_evaluation/metrics.json" \
    --hop "$stress/grid_hop_gate.json" --out "$stress/final_decision.json" \
    --simulation CAMELS-SIMBA-CV-0-15-HISTORICAL-STRESS \
    --bootstrap 50000 --seed 2022
python src/hong2021_independent_gate.py \
    --ensemble-metrics "$eagle_eval/ensemble_evaluation/metrics.json" \
    --hop "$eagle_eval/grid_hop_gate.json" \
    --out "$eagle_eval/final_decision.json" \
    --simulation EAGLE-RefL0100N1504-CONFIRMATION32 \
    --bootstrap 50000 --seed 2023
python - "$evaluation/final_gate_decision.json" \
    "$selected_root/tng/final_decision.json" "$stress/final_decision.json" \
    "$eagle_eval/final_decision.json" <<'PY'
import json,sys
from pathlib import Path
out=Path(sys.argv[1]); labels=("tng_development","historical_simba_stress","eagle_confirmation32")
domains={}
for label,value in zip(labels,sys.argv[2:],strict=True):
    path=Path(value); result=json.loads(path.read_text())["candidates"]["edm"]
    domains[label]={"decision":str(path.resolve()),"field_pass":result["field_gate"]["pass"],"grid_hop_pass":result["grid_hop_gate"]["pass"],"overall_pass":result["overall_pass"]}
accepted=all(row["overall_pass"] for row in domains.values())
report={"schema":"hong2021-v13-final-three-domain-gate-v1","domains":domains,"accepted":accepted,"advance_to_forward_particle_dynamics":accepted}
out.write_text(json.dumps(report,indent=2)+"\n"); print(json.dumps(report,indent=2))
PY
accepted=$(python -c 'import json,sys;print(json.load(open(sys.argv[1]))["accepted"])' \
    "$evaluation/final_gate_decision.json")
write_status complete "accepted=$accepted; $evaluation/final_gate_decision.json"
