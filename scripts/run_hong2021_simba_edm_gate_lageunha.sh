#!/usr/bin/env bash
set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
root=/gpfs/kjhan/CAMELS/SIMBA/L25n256
data=$root/derived/hong2021_v1/simba_cv16.h5
evaluation=$root/evaluation/hong2021_v6_edm
deterministic=/gpfs/kjhan/IllustrisTNG/TNG100-1/training/tng100_v4_split00_l0_groupnorm_std_cosine/minimum_validation_loss.pt
edm=/gpfs/kjhan/IllustrisTNG/TNG100-1/training/tng100_v6_edm_laplacian_sigma2/minimum_validation.pt
mean_cache=$root/derived/hong2021_v1/simba_cv16_deterministic_k2_4.h5
v6_cache=$root/derived/hong2021_v1/simba_cv16_laplacian_sigma2.h5
ensemble=$evaluation/edm_cv16_ensemble16.h5
metrics=$evaluation/ensemble_evaluation/metrics.json
decision=$evaluation/independent_gate_decision.json
hop_report=$evaluation/grid_hop_gate.json
hop_work=$evaluation/grid_hop_work
hop_adapter=/gpfs/kjhan/IllustrisTNG/TNG100-1/tools/hop_grid_adapter

cd "$repo"
export PYTHONPATH=$repo/src
mkdir -p "$evaluation"

if [[ ! -s "$mean_cache" ]]; then
    python src/hong2021_residual_diffusion.py prepare \
        --data "$data" --checkpoint "$deterministic" --out "$mean_cache" \
        --batch 4 --workers 1 --device cuda \
        --k-low-h-mpc 2 --k-high-h-mpc 4
fi

if [[ ! -s "$v6_cache" ]]; then
    python src/hong2021_residual_v6.py prepare \
        --data "$data" --mean-cache "$mean_cache" --out "$v6_cache" \
        --sigma-cells 2 --chunk 4
fi

if [[ ! -s "$ensemble" ]]; then
    python src/hong2021_residual_v6.py sample \
        --data "$data" --cache "$v6_cache" --checkpoint "$edm" \
        --out "$ensemble" --indices 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
        --ensemble 16 --sampling-steps 40 --seed 1777 --device cuda
fi

python src/hong2021_residual_evaluate.py \
    --candidate "edm=$ensemble" \
    --out "$evaluation/ensemble_evaluation" --voxel-mpc-h 0.3125

python src/hong2021_independent_gate.py \
    --ensemble-metrics "$metrics" --out "$decision" \
    --simulation CAMELS-SIMBA-CV

if python - "$decision" <<'PY'
import json, sys
report = json.load(open(sys.argv[1]))
raise SystemExit(0 if report["candidates"]["edm"]["field_gate"]["pass"] else 1)
PY
then
    if [[ ! -s "$hop_report" ]]; then
        python src/hong2021_hop_grid_gate.py \
            --edm "$ensemble" --out "$hop_report" --work "$hop_work" \
            --hop-dir "$hop_adapter" --members 16 --objects 16 --workers 8 \
            --voxel-mpc-h 0.3125 --omega-m 0.3
    fi
    python src/hong2021_independent_gate.py \
        --ensemble-metrics "$metrics" --hop "$hop_report" --out "$decision" \
        --simulation CAMELS-SIMBA-CV --bootstrap 50000 --seed 2021
else
    printf '%s\n' 'SIMBA field gate failed; preregistered policy skips grid-HOP.'
fi
