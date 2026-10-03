#!/usr/bin/env bash
set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
tng=/gpfs/kjhan/IllustrisTNG/TNG100-1
simba=/gpfs/kjhan/CAMELS/SIMBA/L25n256
swift=/gpfs/kjhan/CAMELS/Swift-EAGLE/L25n256
checkpoint=$tng/training/tng100_v4_split00_l0_groupnorm_std_cosine/minimum_validation_loss.pt
prepare_status=$tng/evaluation/tng100_simba_swift_v14_preparation_status.json
root=$tng/evaluation/tng100_simba_swift_v14_baseline_audit
status=$root/sequence_status.json
mkdir -p "$root"
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
    "schema": "hong2021-v14-baseline-audit-status-v1",
    "state": state, "detail": detail, "host": socket.gethostname(),
    "updated_utc": datetime.now(timezone.utc).isoformat(),
}, indent=2) + "\n")
os.replace(temporary, path)
PY
}

write_status waiting_for_common_cic_development_data "$prepare_status"
while true; do
    if [[ -s $prepare_status ]]; then
        state=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["state"])' "$prepare_status")
        if [[ $state == complete ]]; then break; fi
        if [[ $state == failed* ]]; then
            write_status stopped_preparation_failed "$state"
            exit 1
        fi
    fi
    sleep 30
done

audit_one() {
    local domain=$1 data=$2 label=$3
    local output=$root/${label}.h5
    if [[ -s $output && -s ${output%.h5}.json ]]; then
        printf '[existing] %s\n' "$label"
        return 0
    fi
    python src/hong2021_v14_baseline_audit.py \
        --domain "$domain" --data "$data" --checkpoint "$checkpoint" \
        --out "$output" --batch 6 --workers 1 --device cuda \
        >"$root/${label}.log" 2>&1
}

write_status running "six train/validation common-CIC baseline audits"
audit_one TNG100 "$tng/derived/hong2021_v14/cic_data/tng100_train.h5" tng_train
audit_one TNG100 "$tng/derived/hong2021_v14/cic_data/tng100_validation.h5" tng_validation
audit_one CAMELS-SIMBA "$simba/derived/hong2021_v14/simba_cv16_23_train_all_observers.h5" simba_train
audit_one CAMELS-SIMBA "$simba/derived/hong2021_v14/simba_cv24_26_validation_all_observers.h5" simba_validation
audit_one CAMELS-Swift-EAGLE "$swift/derived/hong2021_v14/swift_eagle_cv0_19_train_all_observers.h5" swift_eagle_train
audit_one CAMELS-Swift-EAGLE "$swift/derived/hong2021_v14/swift_eagle_cv20_26_validation_all_observers.h5" swift_eagle_validation

location_scale=$root/location_scale_predictability.json
write_status auditing_location_scale_predictability \
    "source-balanced train-only ridge versus development-validation constants"
if [[ ! -s $location_scale ]]; then
    python src/hong2021_v14_location_scale_audit.py \
        --train TNG100="$root/tng_train.h5" \
        --train CAMELS-SIMBA="$root/simba_train.h5" \
        --train CAMELS-Swift-EAGLE="$root/swift_eagle_train.h5" \
        --validation TNG100="$root/tng_validation.h5" \
        --validation CAMELS-SIMBA="$root/simba_validation.h5" \
        --validation CAMELS-Swift-EAGLE="$root/swift_eagle_validation.h5" \
        --out "$location_scale" --folds 5 --seed 14021 \
        >"$root/location_scale_predictability.log" 2>&1
fi

python - "$root" "$status" "$location_scale" <<'PY'
import json, os, socket, sys
from datetime import datetime, timezone
from pathlib import Path
root, status, location_scale = map(Path, sys.argv[1:])
labels = (
    "tng_train", "tng_validation", "simba_train", "simba_validation",
    "swift_eagle_train", "swift_eagle_validation",
)
domains = {}
for label in labels:
    path = root / f"{label}.json"
    report = json.loads(path.read_text())
    domains[label] = {
        "report": str(path.resolve()),
        "samples": report["samples"],
        "residual_dc": report["residual_dc"],
        "residual_centered_rms": report["residual_centered_rms"],
        "residual_band_rms": report["residual_band_rms"],
    }
combined = {
    "schema": "hong2021-v14-cross-domain-baseline-audit-v1",
    "uses_eagle_ref100_or_astrid": False,
    "domains": domains,
    "location_scale_predictability": str(location_scale.resolve()),
}
combined_path = root / "cross_domain_report.json"
combined_path.write_text(json.dumps(combined, indent=2) + "\n")
result = {
    "schema": "hong2021-v14-baseline-audit-status-v1",
    "state": "complete", "detail": str(combined_path.resolve()),
    "host": socket.gethostname(),
    "updated_utc": datetime.now(timezone.utc).isoformat(),
}
temporary = status.with_suffix(".json.partial")
temporary.write_text(json.dumps(result, indent=2) + "\n")
os.replace(temporary, status)
print(json.dumps(combined, indent=2))
PY
