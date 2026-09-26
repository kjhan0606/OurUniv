#!/usr/bin/env bash
set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
tng=/gpfs/kjhan/IllustrisTNG/TNG100-1
simba=/gpfs/kjhan/CAMELS/SIMBA/L25n256
train=$tng/training/tng100_simba_v7_multidomain_edm
checkpoint=$train/minimum_validation.pt
tng_data=$tng/derived/hong2021_v2/split00_l0_paper/tng100_validation.h5
tng_cache=$tng/derived/hong2021_v6/tng100_validation_laplacian_sigma2.h5
simba_data=$simba/derived/hong2021_v1/simba_cv16.h5
simba_cache=$simba/derived/hong2021_v1/simba_cv16_laplacian_sigma2.h5
tng_eval=$tng/evaluation/tng100_simba_v7_multidomain_edm
simba_eval=$simba/evaluation/hong2021_v7_multidomain_edm
tng_ensemble=$tng_eval/tng_representative16_ensemble16.h5
simba_ensemble=$simba_eval/simba_cv0_15_ensemble16.h5
tng_metrics=$tng_eval/ensemble_evaluation/metrics.json
simba_metrics=$simba_eval/ensemble_evaluation/metrics.json
decision=$tng_eval/dual_gate_decision.json
hop_adapter=$tng/tools/hop_grid_adapter
indices=49,8,63,0,15,4,32,21,12,74,79,53,76,29,57,62

cd "$repo"
export PYTHONPATH=$repo/src

while tmux has-session -t hong2021_v7_multidomain 2>/dev/null; do
    sleep 30
done
python - "$train/run.json" <<'PY'
import json, sys
report = json.load(open(sys.argv[1]))
if report.get("status") != "complete":
    raise SystemExit(f"training did not complete: {report.get('status')}")
PY
test -s "$checkpoint"
mkdir -p "$tng_eval" "$simba_eval"

python src/hong2021_residual_v6.py sample \
    --data "$tng_data" --cache "$tng_cache" --checkpoint "$checkpoint" \
    --out "$tng_ensemble" --indices "$indices" --ensemble 16 \
    --sampling-steps 40 --seed 777 --device cuda
python src/hong2021_residual_evaluate.py \
    --candidate "edm=$tng_ensemble" --out "$tng_eval/ensemble_evaluation" \
    --voxel-mpc-h 0.3125

python src/hong2021_residual_v6.py sample \
    --data "$simba_data" --cache "$simba_cache" --checkpoint "$checkpoint" \
    --out "$simba_ensemble" --indices 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
    --ensemble 16 --sampling-steps 40 --seed 1777 --device cuda
python src/hong2021_residual_evaluate.py \
    --candidate "edm=$simba_ensemble" --out "$simba_eval/ensemble_evaluation" \
    --voxel-mpc-h 0.3125

python src/hong2021_v7_dual_gate.py \
    --tng-metrics "$tng_metrics" --simba-metrics "$simba_metrics" \
    --out "$decision"

if python - "$decision" <<'PY'
import json, sys
report = json.load(open(sys.argv[1]))
raise SystemExit(0 if report["both_field_gates_pass"] else 1)
PY
then
    python src/hong2021_hop_grid_gate.py \
        --edm "$tng_ensemble" --out "$tng_eval/grid_hop_gate.json" \
        --work "$tng_eval/grid_hop_work" --hop-dir "$hop_adapter" \
        --members 16 --objects 16 --workers 8 --voxel-mpc-h 0.3125 \
        --omega-m 0.3
    python src/hong2021_hop_grid_gate.py \
        --edm "$simba_ensemble" --out "$simba_eval/grid_hop_gate.json" \
        --work "$simba_eval/grid_hop_work" --hop-dir "$hop_adapter" \
        --members 16 --objects 16 --workers 8 --voxel-mpc-h 0.3125 \
        --omega-m 0.3
    python src/hong2021_v7_dual_gate.py \
        --tng-metrics "$tng_metrics" --simba-metrics "$simba_metrics" \
        --tng-hop "$tng_eval/grid_hop_gate.json" \
        --simba-hop "$simba_eval/grid_hop_gate.json" --out "$decision"
else
    printf '%s\n' 'At least one frozen field gate failed; grid-HOP skipped.'
fi
