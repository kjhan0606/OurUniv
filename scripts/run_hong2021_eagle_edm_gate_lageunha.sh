#!/usr/bin/env bash
# Wait for EAGLE preparation, then run the locked TNG V6 EDM gate on the Ada GPU.

set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
root=/gpfs/kjhan/EAGLE/RefL0100N1504
derived=$root/derived/hong2021_v1
data=$derived/eagle_ref100_z0_test.h5
evaluation=$root/evaluation/hong2021_v6_edm
deterministic=/gpfs/kjhan/IllustrisTNG/TNG100-1/training/tng100_v4_split00_l0_groupnorm_std_cosine/minimum_validation_loss.pt
edm=/gpfs/kjhan/IllustrisTNG/TNG100-1/training/tng100_v6_edm_laplacian_sigma2/minimum_validation.pt
mean_cache=$derived/eagle_ref100_z0_deterministic_k2_4.h5
v6_cache=$derived/eagle_ref100_z0_laplacian_sigma2.h5
selection=$derived/representative16_indices.json
ensemble=$evaluation/edm_representative16_ensemble16.h5
metrics=$evaluation/ensemble_evaluation/metrics.json
decision=$evaluation/independent_gate_decision.json
hop_report=$evaluation/grid_hop_gate.json
hop_work=$evaluation/grid_hop_work
hop_adapter=/gpfs/kjhan/IllustrisTNG/TNG100-1/tools/hop_grid_adapter
prepare_status=$root/prepare_ref100_hong.status
gate_status=$root/eagle_edm_gate.status
log_dir=$root/logs
run_id=$(date +%Y%m%dT%H%M%S%z)
log=$log_dir/eagle_edm_gate_${run_id}.log

mkdir -p "$evaluation" "$log_dir"
exec 9>"$root/.eagle_edm_gate.lock"
if ! flock -n 9; then
    printf 'Another EAGLE EDM gate holds %s\n' "$root/.eagle_edm_gate.lock" >&2
    exit 3
fi

printf 'waiting_for_preparation\t%s\t%s\t%s\n' \
    "$(date --iso-8601=seconds)" "$(hostname)" "$log" >"$gate_status"
printf '[wait] host=%s time=%s preparation=%s\n' \
    "$(hostname)" "$(date --iso-8601=seconds)" "$prepare_status" | tee "$log"

while [[ ! -s "$data" ]]; do
    if [[ -s "$prepare_status" ]]; then
        prepare_state=$(cut -f1 "$prepare_status")
        if [[ "$prepare_state" == failed ]]; then
            printf '[failed] EAGLE preparation failed; gate will not start.\n' \
                | tee -a "$log"
            printf 'failed\t%s\t%s\t%s\tpreparation_failed\n' \
                "$(date --iso-8601=seconds)" "$(hostname)" "$log" \
                >"$gate_status"
            exit 4
        fi
    fi
    sleep 30
done

printf 'running\t%s\t%s\t%s\n' \
    "$(date --iso-8601=seconds)" "$(hostname)" "$log" >"$gate_status"
cd "$repo"
export PYTHONPATH=$repo/src

set +e
(
    set -euo pipefail
    test -s "$deterministic"
    test -s "$edm"

    if [[ ! -s "$selection" ]]; then
        python - "$data" "$selection" <<'PY'
import json
import sys
from pathlib import Path

import h5py
import numpy as np

from hong2021_prepare_eagle import farthest_point_subset

data_path, output_path = map(Path, sys.argv[1:])
with h5py.File(data_path, "r") as handle:
    position = np.asarray(handle["center_position_mpc_h"], dtype=np.float64)
    galaxy_id = np.asarray(handle["center_galaxy_id"], dtype=np.int64)
selected = farthest_point_subset(position, galaxy_id, 16)
report = {
    "schema": "hong2021-eagle-representative-indices-v1",
    "selection_uses_dark_matter_truth": False,
    "algorithm": "deterministic farthest-point sample; smallest GalaxyID anchor",
    "source": str(data_path.resolve()),
    "indices": selected.tolist(),
    "galaxy_ids": galaxy_id[selected].tolist(),
    "positions_mpc_h": position[selected].tolist(),
}
temporary = output_path.with_suffix(".json.tmp")
temporary.write_text(json.dumps(report, indent=2) + "\n")
temporary.replace(output_path)
print(json.dumps(report, indent=2))
PY
    fi
    indices=$(python -c 'import json,sys; print(",".join(map(str,json.load(open(sys.argv[1]))["indices"])))' "$selection")

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
            --out "$ensemble" --indices "$indices" \
            --ensemble 16 --sampling-steps 40 --seed 2777 --device cuda
    fi

    python src/hong2021_residual_evaluate.py \
        --candidate "edm=$ensemble" \
        --out "$evaluation/ensemble_evaluation" --voxel-mpc-h 0.3125

    python src/hong2021_independent_gate.py \
        --ensemble-metrics "$metrics" --out "$decision" \
        --simulation EAGLE-RefL0100N1504

    if python - "$decision" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1]))
raise SystemExit(0 if report["candidates"]["edm"]["field_gate"]["pass"] else 1)
PY
    then
        if [[ ! -s "$hop_report" ]]; then
            python src/hong2021_hop_grid_gate.py \
                --edm "$ensemble" --out "$hop_report" --work "$hop_work" \
                --hop-dir "$hop_adapter" --members 16 --objects 16 --workers 8 \
                --voxel-mpc-h 0.3125 --omega-m 0.307
        fi
        python src/hong2021_independent_gate.py \
            --ensemble-metrics "$metrics" --hop "$hop_report" --out "$decision" \
            --simulation EAGLE-RefL0100N1504 --bootstrap 50000 --seed 2021
    else
        printf '%s\n' 'EAGLE field gate failed; preregistered policy skips grid-HOP.'
    fi
) 2>&1 | tee -a "$log"
rc=${PIPESTATUS[0]}
set -e

if [[ "$rc" -eq 0 ]]; then
    state=complete
else
    state=failed
fi
printf '%s\t%s\t%s\t%s\texit=%s\n' \
    "$state" "$(date --iso-8601=seconds)" "$(hostname)" "$log" "$rc" \
    >"$gate_status"
printf '[%s] host=%s time=%s exit=%s\n' \
    "$state" "$(hostname)" "$(date --iso-8601=seconds)" "$rc" | tee -a "$log"
exit "$rc"
