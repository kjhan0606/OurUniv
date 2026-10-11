#!/usr/bin/env bash
set -euo pipefail

root=${1:-/gpfs/kjhan/CAMELS/SIMBA/L25n256}
public=https://users.flatironinstitute.org/~camels

download() {
    local realization=$1
    local destination="$root/CV/CV_${realization}/groups_090.hdf5"
    mkdir -p "$(dirname "$destination")"
    curl --fail --location --retry 12 --retry-delay 5 \
        --continue-at - --silent --show-error \
        --output "$destination" \
        "$public/FOF_Subfind/SIMBA/L25n256/CV/CV_${realization}/groups_090.hdf5"
}

pids=()
for realization in $(seq 16 26); do
    download "$realization" &
    pids+=("$!")
    if [[ ${#pids[@]} -ge 4 ]]; then
        wait "${pids[0]}"
        pids=("${pids[@]:1}")
    fi
done
for pid in "${pids[@]}"; do
    wait "$pid"
done

python - "$root" <<'PY'
from pathlib import Path
import sys
import h5py

root = Path(sys.argv[1])
for realization in range(16, 27):
    path = root / "CV" / f"CV_{realization}" / "groups_090.hdf5"
    with h5py.File(path, "r") as handle:
        if abs(float(handle["Header"].attrs["Redshift"])) > 1.0e-6:
            raise SystemExit(f"not z=0: {path}")
        if "Subhalo/SubhaloMassType" not in handle:
            raise SystemExit(f"missing subhalo catalogue: {path}")
print("SIMBA CV 16-26 development catalogues complete")
PY
