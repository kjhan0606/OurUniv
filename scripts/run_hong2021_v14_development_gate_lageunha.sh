#!/usr/bin/env bash
set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
tng=/gpfs/kjhan/IllustrisTNG/TNG100-1
simba=/gpfs/kjhan/CAMELS/SIMBA/L25n256
swift=/gpfs/kjhan/CAMELS/Swift-EAGLE/L25n256
model_root=$tng/derived/hong2021_v14/model
training=$tng/training/tng100_simba_swift_v14_multiscale_edm
evaluation=$tng/evaluation/tng100_simba_swift_v14_multiscale_edm
status=$evaluation/sequence_status.json
tng_indices=49,8,63,0,15,4,32,21,12,74,79,53,76,29,57,62
mkdir -p "$evaluation"
cd "$repo"
export PYTHONPATH=$repo/src

write_status() {
    python - "$status" "$1" "$2" <<'PY'
import json, os, socket, sys
from datetime import datetime, timezone
from pathlib import Path
path, state, detail = sys.argv[1:]
temporary = Path(path).with_suffix(".json.partial")
temporary.write_text(json.dumps({
    "schema": "hong2021-v14-development-gate-sequence-status-v1",
    "state": state, "detail": detail, "host": socket.gethostname(),
    "updated_utc": datetime.now(timezone.utc).isoformat(),
}, indent=2) + "\n")
os.replace(temporary, path)
PY
}

write_status waiting_for_edm "$training/run.json"
while true; do
    if [[ -s $training/run.json ]]; then
        state=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$training/run.json")
        if [[ $state == complete ]]; then break; fi
        if [[ $state == failed* ]]; then
            write_status stopped_edm_failed "$state"
            exit 1
        fi
    fi
    sleep 30
done

simba_indices=$(python -c 'import json; print(",".join(map(str,json.load(open("config/hong2021_simba_dev_representative16_v1.json"))["indices"])))')
swift_indices=$(python -c 'import json; print(",".join(map(str,json.load(open("config/hong2021_swift_eagle_dev_representative16_v1.json"))["indices"])))')

sample_evaluate() {
    local checkpoint=$1 data=$2 cache=$3 indices=$4 seed=$5 root=$6
    mkdir -p "$root"
    if [[ ! -s $root/ensemble16_steps40.h5 ]]; then
        python src/hong2021_v14_edm.py sample \
            --data "$data" --cache "$cache" --checkpoint "$checkpoint" \
            --out "$root/ensemble16_steps40.h5" --indices "$indices" \
            --ensemble 16 --sampling-steps 40 --sigma-min 0.002 \
            --sigma-max 40 --rho 7 --seed "$seed" --device cuda \
            >"$root/sample.log" 2>&1
    fi
    if [[ ! -s $root/ensemble_evaluation/metrics.json ]]; then
        python src/hong2021_residual_evaluate.py \
            --candidate "edm=$root/ensemble16_steps40.h5" \
            --out "$root/ensemble_evaluation" --voxel-mpc-h 0.3125 \
            >"$root/evaluate.log" 2>&1
    fi
}

write_status running_three_domain_development_gate "steps 2000,5000,10000"
for step in 002000 005000 010000; do
    checkpoint=$training/validation_checkpoints/step_${step}.pt
    root=$evaluation/development_candidates/step_${step}
    sample_evaluate "$checkpoint" \
        "$tng/derived/hong2021_v14/cic_data/tng100_validation.h5" \
        "$model_root/tng_validation_standardized.h5" "$tng_indices" 15777 "$root/tng"
    sample_evaluate "$checkpoint" \
        "$simba/derived/hong2021_v14/simba_cv24_26_validation_all_observers.h5" \
        "$model_root/simba_validation_standardized.h5" "$simba_indices" 16777 "$root/simba_dev"
    sample_evaluate "$checkpoint" \
        "$swift/derived/hong2021_v14/swift_eagle_cv20_26_validation_all_observers.h5" \
        "$model_root/swift_eagle_validation_standardized.h5" "$swift_indices" 17777 "$root/swift_eagle_dev"
done

python src/hong2021_v14_development_gate.py \
    --root "$evaluation/development_candidates" --training "$training" \
    --steps 2000 5000 10000 --out "$evaluation/development_decision.json" \
    >"$evaluation/development_decision.log" 2>&1
passed=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["development_pass"])' "$evaluation/development_decision.json")
if [[ $passed == True ]]; then
    write_status complete_passed_development "$evaluation/development_decision.json"
else
    write_status complete_failed_development_no_astrid "$evaluation/development_decision.json"
fi
