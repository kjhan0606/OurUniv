#!/usr/bin/env bash
set -euo pipefail

# Public, authentication-free inputs for the provisional independent-SIMBA
# generalization gate.  The CMD file contains all 27 CV dark-matter density
# grids; only the first 16 independent CV catalogues are used by the frozen
# evaluation manifest.

root=${1:-/gpfs/kjhan/CAMELS/SIMBA/L25n256}
public=https://users.flatironinstitute.org/~camels
grid_name='Grids_Mcdm_SIMBA_CV_256_z=0.0.npy'

mkdir -p "$root/CMD" "$root/CV"

download() {
    local url=$1
    local destination=$2
    mkdir -p "$(dirname "$destination")"
    curl --fail --location --retry 12 --retry-delay 5 \
        --continue-at - --silent --show-error \
        --output "$destination" "$url"
}

download \
    "$public/CMD/3D_grids/data/SIMBA/$grid_name" \
    "$root/CMD/$grid_name" &
grid_pid=$!

pids=()
for realization in $(seq 0 15); do
    if [[ $realization -eq 0 && -s "$root/CV/CV_0/groups_090.hdf5" ]]; then
        continue
    fi
    download \
        "$public/FOF_Subfind/SIMBA/L25n256/CV/CV_${realization}/groups_090.hdf5" \
        "$root/CV/CV_${realization}/groups_090.hdf5" &
    pids+=("$!")
    if [[ ${#pids[@]} -ge 4 ]]; then
        wait "${pids[0]}"
        pids=("${pids[@]:1}")
    fi
done

for pid in "${pids[@]}"; do
    wait "$pid"
done
wait "$grid_pid"

python - "$root" <<'PY'
from pathlib import Path
import sys

import h5py
import numpy as np

root = Path(sys.argv[1])
grid = np.load(
    root / "CMD" / "Grids_Mcdm_SIMBA_CV_256_z=0.0.npy",
    mmap_mode="r",
)
if grid.shape != (27, 256, 256, 256) or grid.dtype != np.float32:
    raise SystemExit(f"unexpected CMD grid: shape={grid.shape} dtype={grid.dtype}")
if not np.isfinite(grid[:16]).all() or np.any(grid[:16] < 0):
    raise SystemExit("non-finite or negative CMD density in selected realizations")

for realization in range(16):
    path = root / "CV" / f"CV_{realization}" / "groups_090.hdf5"
    with h5py.File(path, "r") as handle:
        header = handle["Header"].attrs
        if not np.isclose(float(header["BoxSize"]), 25000.0):
            raise SystemExit(f"unexpected BoxSize in {path}")
        if abs(float(header["Redshift"])) > 1.0e-6:
            raise SystemExit(f"catalog is not z=0: {path}")
        required = (
            "Subhalo/SubhaloPos",
            "Subhalo/SubhaloVel",
            "Subhalo/SubhaloMassType",
        )
        missing = [name for name in required if name not in handle]
        if missing:
            raise SystemExit(f"missing {missing} in {path}")

print(f"CAMELS SIMBA CV gate inputs complete: {root}")
PY
