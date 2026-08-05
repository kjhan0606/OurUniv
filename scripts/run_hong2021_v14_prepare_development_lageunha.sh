#!/usr/bin/env bash
set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
tng=/gpfs/kjhan/IllustrisTNG/TNG100-1
simba=/gpfs/kjhan/CAMELS/SIMBA/L25n256
swift=/gpfs/kjhan/CAMELS/Swift-EAGLE/L25n256
status=$tng/evaluation/tng100_simba_swift_v14_preparation_status.json
log_root=$tng/derived/hong2021_v14/logs
mkdir -p "$(dirname "$status")" "$log_root"
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
    "schema": "hong2021-v14-development-preparation-status-v1",
    "state": state,
    "detail": detail,
    "host": socket.gethostname(),
    "updated_utc": datetime.now(timezone.utc).isoformat(),
}, indent=2) + "\n")
os.replace(temporary, path)
PY
}

wait_for_file() {
    local path=$1 label=$2
    local attempts=0
    while [[ ! -s $path ]]; do
        attempts=$((attempts + 1))
        if [[ $attempts -gt 1440 ]]; then
            write_status failed_wait_timeout "$label: $path"
            return 1
        fi
        sleep 30
    done
}

write_status waiting_for_inputs "TNG CIC cache and SIMBA/Swift-EAGLE download manifests"
wait_for_file "$tng/derived/hong2021_v14/dm_cic_240.npy" tng_cic
wait_for_file "$simba/download/hong2021_v14_raw_development_manifest.json" simba_download
wait_for_file "$swift/download/hong2021_v14_raw_development_manifest.json" swift_eagle_download

build_one() {
    local suite=$1 root=$2 realization=$3
    local out=$root/derived/hong2021_v14/cic_grids/CV_${realization}.npy
    local log=$root/derived/hong2021_v14/cic_grids/CV_${realization}.log
    mkdir -p "$(dirname "$out")"
    if [[ -s $out && -s ${out%.npy}.json ]]; then
        printf '[existing] %s CV_%s\n' "$suite" "$realization"
        return 0
    fi
    python src/hong2021_build_particle_grid.py \
        --snapshots "$root/raw/CV_${realization}/snapshot_090.hdf5" \
        --out "$out" --grid 80 --box-mpc-h 25 \
        --coordinate-scale-to-mpc-h 0.001 --assignment cic \
        --block-particles 20000000 >"$log" 2>&1
    printf '[built] %s CV_%s\n' "$suite" "$realization"
}

build_suite() {
    local suite=$1 root=$2 first=$3 last=$4
    local pids=()
    for realization in $(seq "$first" "$last"); do
        build_one "$suite" "$root" "$realization" &
        pids+=("$!")
        if [[ ${#pids[@]} -ge 4 ]]; then
            wait "${pids[0]}"
            pids=("${pids[@]:1}")
        fi
    done
    for pid in "${pids[@]}"; do wait "$pid"; done
}

write_status building_common_cic_grids "SIMBA CV16-26 and Swift-EAGLE CV0-26"
build_suite SIMBA "$simba" 16 26 >"$log_root/simba_cic_grids.log" 2>&1 &
simba_pid=$!
build_suite Swift-EAGLE "$swift" 0 26 >"$log_root/swift_eagle_cic_grids.log" 2>&1 &
swift_pid=$!
wait "$simba_pid"
wait "$swift_pid"

write_status retargeting_tng "copy inputs/splits exactly; replace target from CIC cache"
tng_out=$tng/derived/hong2021_v14/cic_data
mkdir -p "$tng_out"
if [[ ! -s $tng_out/tng100_train.h5 || ! -s $tng_out/tng100_train.json ]]; then
    python src/hong2021_retarget_tng_cic.py \
        --source "$tng/derived/hong2021_v2/split00_l0_paper/tng100_train.h5" \
        --grid "$tng/derived/hong2021_v14/dm_cic_240.npy" \
        --out "$tng_out/tng100_train.h5" >"$log_root/tng_train_retarget.log" 2>&1
fi
if [[ ! -s $tng_out/tng100_validation.h5 || ! -s $tng_out/tng100_validation.json ]]; then
    python src/hong2021_retarget_tng_cic.py \
        --source "$tng/derived/hong2021_v2/split00_l0_paper/tng100_validation.h5" \
        --grid "$tng/derived/hong2021_v14/dm_cic_240.npy" \
        --out "$tng_out/tng100_validation.h5" >"$log_root/tng_validation_retarget.log" 2>&1
fi

write_status preparing_camels_hdf5 "all development-training and validation observers"
simba_train=$simba/derived/hong2021_v14/simba_cv16_23_train_all_observers.h5
simba_validation=$simba/derived/hong2021_v14/simba_cv24_26_validation_all_observers.h5
swift_train=$swift/derived/hong2021_v14/swift_eagle_cv0_19_train_all_observers.h5
swift_validation=$swift/derived/hong2021_v14/swift_eagle_cv20_26_validation_all_observers.h5
if [[ ! -s $simba_train || ! -s ${simba_train%.h5}.json ]]; then
    python src/hong2021_prepare_camels_raw.py \
        --suite SIMBA --root "$simba" --realizations 16,17,18,19,20,21,22,23 \
        --observers all --role training \
        --grid-pattern "$simba/derived/hong2021_v14/cic_grids/CV_{realization}.npy" \
        --out "$simba_train" >"$log_root/simba_train_prepare.log" 2>&1
fi
if [[ ! -s $simba_validation || ! -s ${simba_validation%.h5}.json ]]; then
    python src/hong2021_prepare_camels_raw.py \
        --suite SIMBA --root "$simba" --realizations 24,25,26 \
        --observers all --role development_validation \
        --grid-pattern "$simba/derived/hong2021_v14/cic_grids/CV_{realization}.npy" \
        --out "$simba_validation" >"$log_root/simba_validation_prepare.log" 2>&1
fi
if [[ ! -s $swift_train || ! -s ${swift_train%.h5}.json ]]; then
    python src/hong2021_prepare_camels_raw.py \
        --suite Swift-EAGLE --root "$swift" \
        --realizations 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19 \
        --observers all --role training \
        --grid-pattern "$swift/derived/hong2021_v14/cic_grids/CV_{realization}.npy" \
        --out "$swift_train" >"$log_root/swift_eagle_train_prepare.log" 2>&1
fi
if [[ ! -s $swift_validation || ! -s ${swift_validation%.h5}.json ]]; then
    python src/hong2021_prepare_camels_raw.py \
        --suite Swift-EAGLE --root "$swift" --realizations 20,21,22,23,24,25,26 \
        --observers all --role development_validation \
        --grid-pattern "$swift/derived/hong2021_v14/cic_grids/CV_{realization}.npy" \
        --out "$swift_validation" >"$log_root/swift_eagle_validation_prepare.log" 2>&1
fi

python - "$status" "$tng_out" "$simba" "$swift" <<'PY'
import json, os, socket, sys
from datetime import datetime, timezone
from pathlib import Path
import h5py

status, tng, simba, swift = map(Path, sys.argv[1:])
paths = {
    "tng_train": tng / "tng100_train.h5",
    "tng_validation": tng / "tng100_validation.h5",
    "simba_train": simba / "derived/hong2021_v14/simba_cv16_23_train_all_observers.h5",
    "simba_validation": simba / "derived/hong2021_v14/simba_cv24_26_validation_all_observers.h5",
    "swift_eagle_train": swift / "derived/hong2021_v14/swift_eagle_cv0_19_train_all_observers.h5",
    "swift_eagle_validation": swift / "derived/hong2021_v14/swift_eagle_cv20_26_validation_all_observers.h5",
}
datasets = {}
for name, path in paths.items():
    with h5py.File(path, "r") as handle:
        datasets[name] = {
            "path": str(path.resolve()),
            "samples": len(handle["input"]),
            "schema": str(handle.attrs.get("schema", "")),
            "target_operator": str(handle.attrs.get("target_operator", "")),
        }
report = {
    "schema": "hong2021-v14-development-preparation-status-v1",
    "state": "complete",
    "detail": "common raw-particle CIC datasets prepared",
    "host": socket.gethostname(),
    "updated_utc": datetime.now(timezone.utc).isoformat(),
    "datasets": datasets,
}
temporary = status.with_suffix(".json.partial")
temporary.write_text(json.dumps(report, indent=2) + "\n")
os.replace(temporary, status)
print(json.dumps(report, indent=2))
PY
