#!/usr/bin/env bash
set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
tng=/gpfs/kjhan/IllustrisTNG/TNG100-1
simba=/gpfs/kjhan/CAMELS/SIMBA/L25n256
swift=/gpfs/kjhan/CAMELS/Swift-EAGLE/L25n256
training=$tng/training/tng100_simba_swift_v14_mean_correction
root=$tng/derived/hong2021_v14/model
status=$root/preparation_status.json
mkdir -p "$root/logs"
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
    "schema": "hong2021-v14-model-residual-preparation-status-v1",
    "state": state, "detail": detail, "host": socket.gethostname(),
    "updated_utc": datetime.now(timezone.utc).isoformat(),
}, indent=2) + "\n")
os.replace(temporary, path)
PY
}

write_status waiting_for_mean_correction "$training/selection.json"
while [[ ! -s $training/selection.json ]]; do sleep 30; done
correction=$(python - "$training/selection.json" <<'PY'
import json, sys
value = json.load(open(sys.argv[1]))
if not value.get("development_pass") or not value.get("selected_checkpoint"):
    raise SystemExit("V14 mean correction failed its frozen selection")
print(value["selected_checkpoint"])
PY
)

prepare_corrected() {
    local domain=$1 data=$2 baseline=$3 label=$4
    local out=$root/${label}_corrected.h5
    if [[ -s $out && -s ${out%.h5}.json ]]; then
        printf '[existing] %s\n' "$out"
        return
    fi
    python src/hong2021_v14_residual_cache.py corrected \
        --domain "$domain" --data "$data" --baseline-cache "$baseline" \
        --correction-checkpoint "$correction" --out "$out" \
        --batch 6 --workers 1 --device cuda >"$root/logs/${label}_corrected.log" 2>&1
}

audit=$tng/evaluation/tng100_simba_swift_v14_baseline_audit
write_status preparing_corrected_residuals "$correction"
prepare_corrected TNG100 "$tng/derived/hong2021_v14/cic_data/tng100_train.h5" "$audit/tng_train.h5" tng_train
prepare_corrected TNG100 "$tng/derived/hong2021_v14/cic_data/tng100_validation.h5" "$audit/tng_validation.h5" tng_validation
prepare_corrected SIMBA "$simba/derived/hong2021_v14/simba_cv16_23_train_all_observers.h5" "$audit/simba_train.h5" simba_train
prepare_corrected SIMBA "$simba/derived/hong2021_v14/simba_cv24_26_validation_all_observers.h5" "$audit/simba_validation.h5" simba_validation
prepare_corrected Swift-EAGLE "$swift/derived/hong2021_v14/swift_eagle_cv0_19_train_all_observers.h5" "$audit/swift_eagle_train.h5" swift_eagle_train
prepare_corrected Swift-EAGLE "$swift/derived/hong2021_v14/swift_eagle_cv20_26_validation_all_observers.h5" "$audit/swift_eagle_validation.h5" swift_eagle_validation

location_scale=$root/location_scale_model.json
write_status fitting_location_scale "three training domains only"
if [[ ! -s $location_scale ]]; then
    python src/hong2021_v14_location_scale.py \
        --tng-train "$root/tng_train_corrected.h5" \
        --tng-validation "$root/tng_validation_corrected.h5" \
        --simba-train "$root/simba_train_corrected.h5" \
        --simba-validation "$root/simba_validation_corrected.h5" \
        --swift-train "$root/swift_eagle_train_corrected.h5" \
        --swift-validation "$root/swift_eagle_validation_corrected.h5" \
        --out "$location_scale" --folds 5 --seed 143021 \
        >"$root/logs/location_scale_fit.log" 2>&1
fi

prepare_standardized() {
    local label=$1
    local corrected=$root/${label}_corrected.h5
    local out=$root/${label}_standardized.h5
    if [[ -s $out && -s ${out%.h5}.json ]]; then
        printf '[existing] %s\n' "$out"
        return
    fi
    python src/hong2021_v14_residual_cache.py standardized \
        --corrected-cache "$corrected" --location-scale-model "$location_scale" \
        --out "$out" --chunk 4 >"$root/logs/${label}_standardized.log" 2>&1
}

write_status preparing_standardized_residuals "$location_scale"
for label in tng_train tng_validation simba_train simba_validation swift_eagle_train swift_eagle_validation; do
    prepare_standardized "$label"
done

python - "$status" "$root" "$correction" "$location_scale" <<'PY'
import h5py, json, os, socket, sys
from datetime import datetime, timezone
from pathlib import Path
status, root = map(Path, sys.argv[1:3])
correction, location_scale = sys.argv[3:]
labels = (
    "tng_train", "tng_validation", "simba_train", "simba_validation",
    "swift_eagle_train", "swift_eagle_validation",
)
caches = {}
for label in labels:
    path = root / f"{label}_standardized.h5"
    with h5py.File(path, "r") as handle:
        caches[label] = {
            "path": str(path.resolve()),
            "samples": len(handle["standardized_residual"]),
            "rms": float(handle.attrs["standardized_residual_rms"]),
            "schema": str(handle.attrs["schema"]),
        }
report = {
    "schema": "hong2021-v14-model-residual-preparation-status-v1",
    "state": "complete", "detail": "corrected and standardized residual caches ready",
    "host": socket.gethostname(),
    "updated_utc": datetime.now(timezone.utc).isoformat(),
    "correction_checkpoint": correction,
    "location_scale_model": str(Path(location_scale).resolve()),
    "caches": caches,
}
temporary = status.with_suffix(".json.partial")
temporary.write_text(json.dumps(report, indent=2) + "\n")
os.replace(temporary, status)
print(json.dumps(report, indent=2))
PY
