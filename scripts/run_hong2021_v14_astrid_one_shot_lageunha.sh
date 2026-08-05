#!/usr/bin/env bash
set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
tng=/gpfs/kjhan/IllustrisTNG/TNG100-1
root=/gpfs/kjhan/CAMELS/Astrid/L25n256
seal=${1:-config/hong2021_v14_astrid_one_shot_seal.json}
freeze_script=${HONG2021_ASTRID_FREEZE_SCRIPT:-src/hong2021_v14_freeze.py}
evaluation=${HONG2021_ASTRID_EVALUATION:-$root/evaluation/hong2021_v14_astrid_one_shot}
state_schema=${HONG2021_ASTRID_STATE_SCHEMA:-hong2021-v14-astrid-one-shot-sequence-status-v1}
sample_mode=${HONG2021_ASTRID_SAMPLE_MODE:-v14}
sample_script=${HONG2021_ASTRID_SAMPLE_SCRIPT:-src/hong2021_v14_edm.py}
state=$evaluation/sequence_status.json
raw_manifest=$root/download/hong2021_v14_astrid_raw_manifest.json
input=$root/derived/hong2021_v14/astrid_cv0_26_one_observer.h5
cache_root=$root/derived/hong2021_v14/model
cache=$cache_root/astrid_standardized.h5
ensemble=$evaluation/ensemble16_steps40_seed28777.h5
metrics=$evaluation/ensemble_evaluation/metrics.json
field_decision=$evaluation/field_decision.json
hop=$evaluation/grid_hop_gate.json
final_decision=$evaluation/final_decision.json
hop_dir=/home/kjhan/BACKUP/lagRamses-de-nonstd/utils/f90/hop_ramses
indices=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26

cd "$repo"
export PYTHONPATH=$repo/src

mkdir -p /gpfs/kjhan/.hong2021_locks
exec 9>/gpfs/kjhan/.hong2021_locks/astrid_one_shot.lock
if ! flock -n 9; then
    printf 'Another sealed Astrid one-shot runner holds the lock.\n' >&2
    exit 2
fi

write_state() {
    local stage=$1 detail=$2
    mkdir -p "$evaluation"
    python - "$state" "$stage" "$detail" "$state_schema" <<'PY'
import json, os, socket, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
path, stage, detail, schema = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
now = datetime.now(timezone.utc).isoformat()
old = json.loads(path.read_text()) if path.exists() else {}
report = {
    "schema": schema,
    "stage": stage, "detail": detail, "host": socket.gethostname(),
    "updated_utc": now,
    "opened_utc": old.get("opened_utc", now),
    "opening_git_head": old.get(
        "opening_git_head",
        subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    ),
    "scientific_configuration_changed_after_opening": False,
}
temporary = path.with_suffix(".json.partial")
temporary.write_text(json.dumps(report, indent=2) + "\n")
os.replace(temporary, path)
PY
}

failure_state() {
    local code=$?
    trap - ERR
    write_state interrupted_exact_configuration_resumable \
        "command failed with exit $code; no alternate scientific configuration was tried"
    exit "$code"
}

if [[ -s $state ]]; then
    previous=$(python -c 'import json,sys;print(json.load(open(sys.argv[1]))["stage"])' "$state")
    if [[ $previous == complete_* ]]; then
        printf 'Astrid one-shot already has terminal state: %s\n' "$previous"
        exit 0
    fi
    python "$freeze_script" verify --repo "$repo" --seal "$seal"
else
    # This is the final target-blind check.  The state file is created only
    # after proving that no Astrid file exists and the exact seal is committed.
    python "$freeze_script" verify \
        --repo "$repo" --seal "$seal" --require-unopened
    write_state opening_frozen_download \
        "exact committed model opened for Astrid CV0-26; no alternate is permitted"
fi
trap failure_state ERR

if [[ ! -s $raw_manifest ]]; then
    write_state downloading_and_validating_raw_cv0_26 \
        "Gadget kpc/h coordinates; raw snapshot CIC only; CMD target forbidden"
    export HONG2021_V14_ASTRID_ONE_SHOT=sealed
    scripts/download_hong2021_v14_astrid_one_shot.sh "$seal" "$root" \
        >"$evaluation/download.log" 2>&1
fi

write_state building_common_cic \
    "27 independent 80^3 periodic cell-centred CIC grids at 0.3125 Mpc/h"
mkdir -p "$root/derived/hong2021_v14/grids" "$evaluation/grid_logs"
for realization in $(seq 0 26); do
    grid=$root/derived/hong2021_v14/dm_cic_${realization}.npy
    if [[ -s $grid && -s ${grid%.npy}.json ]]; then
        continue
    fi
    if [[ -e $grid.partial || -e ${grid%.npy}.json.partial ]]; then
        stamp=$(date -u +%Y%m%dT%H%M%SZ)
        [[ ! -e $grid.partial ]] || mv "$grid.partial" "$grid.partial.interrupted.$stamp"
        [[ ! -e ${grid%.npy}.json.partial ]] || \
            mv "${grid%.npy}.json.partial" "${grid%.npy}.json.partial.interrupted.$stamp"
    fi
    python src/hong2021_build_particle_grid.py \
        --snapshots "$root/raw/CV_${realization}/snapshot_090.hdf5" \
        --out "$grid" --grid 80 --box-mpc-h 25 \
        --coordinate-scale-to-mpc-h 0.001 --assignment cic \
        >"$evaluation/grid_logs/CV_${realization}.log" 2>&1
done

write_state preparing_one_observer_per_realization \
    "frozen stellar-mass-only observer rule and frozen galaxy proxy"
if [[ ! -s $input || ! -s ${input%.h5}.json ]]; then
    python src/hong2021_v14_astrid_prepare.py --seal "$seal" --repo "$repo" \
        input --root "$root" --raw-manifest "$raw_manifest" --out "$input" \
        >"$evaluation/input_preparation.log" 2>&1
fi

write_state preparing_frozen_inference_cache \
    "V4 mean, selected zero-DC correction, and target-free frozen location-scale model"
if [[ ! -s $cache || ! -s ${cache%.h5}.json ]]; then
    python src/hong2021_v14_astrid_prepare.py --seal "$seal" --repo "$repo" \
        model-cache --data "$input" --out "$cache_root" --device cuda \
        >"$evaluation/model_cache_preparation.log" 2>&1
fi

edm_checkpoint=$(python - "$seal" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))["artifacts"]["edm"]["path"])
PY
)
write_state running_single_field_gate \
    "all 27 objects, ensemble16, steps40, seed28777, unchanged eight checks"
if [[ ! -s $ensemble ]]; then
    if [[ -e $ensemble.partial ]]; then
        stamp=$(date -u +%Y%m%dT%H%M%SZ)
        mv "$ensemble.partial" "$ensemble.partial.interrupted.$stamp"
    fi
    if [[ $sample_mode == v14 ]]; then
        python "$sample_script" sample \
            --data "$input" --cache "$cache" --checkpoint "$edm_checkpoint" \
            --out "$ensemble" --indices "$indices" --ensemble 16 \
            --sampling-steps 40 --sigma-min 0.002 --sigma-max 40 --rho 7 \
            --seed 28777 --device cuda >"$evaluation/sample.log" 2>&1
    elif [[ $sample_mode == v18_prior_matched ]]; then
        python "$sample_script" --seal "$seal" --repo "$repo" \
            --data "$input" --cache "$cache" --out "$ensemble" \
            --indices "$indices" --ensemble 16 --sampling-steps 40 \
            --sigma-min 0.002 --sigma-max 40 --rho 7 --seed 28777 \
            --device cuda >"$evaluation/sample.log" 2>&1
    else
        printf 'Unsupported sealed Astrid sample mode: %s\n' "$sample_mode" >&2
        exit 2
    fi
fi
if [[ ! -s $metrics ]]; then
    python src/hong2021_residual_evaluate.py \
        --candidate "edm=$ensemble" --out "$evaluation/ensemble_evaluation" \
        --voxel-mpc-h 0.3125 >"$evaluation/evaluate.log" 2>&1
fi
if [[ ! -s $field_decision ]]; then
    python src/hong2021_independent_gate.py \
        --ensemble-metrics "$metrics" --out "$field_decision" \
        --simulation CAMELS-Astrid-CV0-26-ONE-SHOT \
        --bootstrap 50000 --seed 2024 >"$evaluation/field_decision.log" 2>&1
fi
field_pass=$(python - "$field_decision" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))["candidates"]["edm"]["field_gate"]["pass"])
PY
)
if [[ $field_pass != True ]]; then
    trap - ERR
    write_state complete_failed_field_gate_grid_hop_skipped \
        "$field_decision; failure is sealed and no alternate will be tried"
    exit 0
fi

write_state running_grid_hop_after_field_pass \
    "27 objects, all 16 members, unchanged HOP thresholds, Omega_m=0.3"
if [[ ! -s $hop ]]; then
    python src/hong2021_hop_grid_gate.py \
        --edm "$ensemble" --out "$hop" --work "$evaluation/grid_hop_work" \
        --hop-dir "$hop_dir" --members 16 --objects 27 --workers 8 \
        --voxel-mpc-h 0.3125 --omega-m 0.3 \
        >"$evaluation/grid_hop.log" 2>&1
fi
if [[ ! -s $final_decision ]]; then
    python src/hong2021_independent_gate.py \
        --ensemble-metrics "$metrics" --hop "$hop" --out "$final_decision" \
        --simulation CAMELS-Astrid-CV0-26-ONE-SHOT \
        --bootstrap 50000 --seed 2024 >"$evaluation/final_decision.log" 2>&1
fi
overall=$(python - "$final_decision" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))["candidates"]["edm"]["overall_pass"])
PY
)
trap - ERR
if [[ $overall == True ]]; then
    write_state complete_passed_independent_gate "$final_decision"
else
    write_state complete_failed_grid_hop "$final_decision"
fi
