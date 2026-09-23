#!/usr/bin/env bash
#SBATCH --job-name=cf4_s40349_zoom_l12
#SBATCH --partition=a100_pcie,a40,a100,h100,h200
#SBATCH --exclude=syn06
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=/gpfs/kjhan/CF4/logs/%x_%j.out
#SBATCH --error=/gpfs/kjhan/CF4/logs/%x_%j.err
set -Eeuo pipefail

root=/home/kjhan/BACKUP/CF4
parent=/gpfs/kjhan/CF4/recon/linear_cr/parent_b3292_beta025_lg_survival_calibration_v1/i248_s40349/grafic_l9
transfer=/gpfs/kjhan/CF4/recon/linear_cr/parent_b3292_beta025_lg_survival_calibration_v1/i248_s40349/transfer_l9_eh98_v1.npz
transfer_report=/gpfs/kjhan/CF4/diagnostics/cf4_lg_b3292_s40349_eh98_transfer_v1.json
mask=/gpfs/kjhan/CF4/masks/cf4_lg_b3292_s40349_spatial_mask_l9_v1.npz
trace_report=/gpfs/kjhan/CF4/diagnostics/cf4_lg_b3292_s40349_spatial_trace_v1.json
out=/gpfs/kjhan/CF4/zoom/cf4_lg_b3292_s40349_traceonly_l12_v1
report=/gpfs/kjhan/CF4/diagnostics/cf4_lg_b3292_s40349_zoom_ic_l12_v1.json

test -d "$parent"
test -s "$transfer"
test -s "$transfer_report"
test -s "$mask"
test -s "$trace_report"
test ! -e "$out"
test ! -e "$report"
cd "$root"
export PYTHONPATH="$root/src:$root/scripts"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1

/home/kjhan/miniconda3/envs/circle/bin/python3.11 - "$transfer_report" "$trace_report" <<'PY'
import json
import sys

transfer = json.load(open(sys.argv[1]))
trace = json.load(open(sys.argv[2]))
assert transfer["decision"] == "TRACE_ONLY_LCDM_HIGHK_TRANSFER_PASS"
assert trace["selection_status"] == "trace-only-zoom-candidate"
assert trace["parent_promoted"] is False
assert trace["m33_resolved"] is False
assert trace["z0_selection"]["selected_pair_members_retained"] == 143
PY

available=$(df -hP /gpfs/kjhan/CF4 | awk 'NR==2 {print $4}')
printf 'job=%s host=%s purpose=trace-only-L12-IC parent_promoted=false m33_resolved=false parent=%s transfer=%s mask=%s out=%s expected_output_GiB=3.4 estimated_peak_GiB=12 requested_mem_GiB=16 available=%s\n' \
  "$SLURM_JOB_ID" "$(hostname)" "$parent" "$transfer" "$mask" "$out" "$available"
sha256sum "$transfer" "$transfer_report" "$mask" "$trace_report" \
  "$parent/ic_deltab" "$parent/ic_velcx" "$parent/ic_velcy" "$parent/ic_velcz"

/home/kjhan/miniconda3/envs/circle/bin/python3.11 src/cf4_zoom_ic2.py \
  --parent "$parent" --parent-level 9 --parent-grid-size 512 \
  --transfer "$transfer" --out "$out" \
  --resolution-config config/ic_resolution_v1.json \
  --tier pilot --levelmin 9 --levelmax-ic 12 --runtime-levelmax 19 \
  --box-hmpc 384.0 --mask-npz "$mask" \
  --seed 403495108 --fft-workers 8

/home/kjhan/miniconda3/envs/circle/bin/python3.11 - "$out" "$report" <<'PY'
import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np

root = Path(sys.argv[1])
report_path = Path(sys.argv[2])
manifest_path = root / "zoom_ic_manifest.json"
manifest = json.loads(manifest_path.read_text())
if manifest.get("status") != "complete":
    raise SystemExit("zoom IC manifest is not complete")
spec = manifest["run_spec"]
expected = {
    "global_levelmin": 9,
    "finest_ic_level": 12,
    "runtime_levelmax": 19,
    "parent_grid_size": 512,
    "small_scale_seed": 403495108,
}
for key, value in expected.items():
    if spec.get(key) != value:
        raise SystemExit(f"manifest {key}={spec.get(key)!r}, expected {value!r}")
if spec.get("geometric_mask_diagnostic") is not False:
    raise SystemExit("geometric mask was used unexpectedly")

def header(path):
    with path.open("rb") as stream:
        size = struct.unpack("<i", stream.read(4))[0]
        payload = stream.read(size)
        end = struct.unpack("<i", stream.read(4))[0]
    if size != 44 or end != size:
        raise SystemExit(f"bad GRAFIC header in {path}")
    dims = struct.unpack("<3i", payload[:12])
    values = struct.unpack("<8f", payload[12:])
    return dims, values

levels = {}
expected_side = {9: 512, 10: 84, 11: 168, 12: 336}
for level, side in expected_side.items():
    directory = root / f"level_{level:03d}"
    fields = {}
    headers = {}
    for name in ("ic_deltab", "ic_velcx", "ic_velcy", "ic_velcz", "ic_refmap"):
        path = directory / name
        dims, values = header(path)
        if dims != (side, side, side):
            raise SystemExit(f"{path} dims={dims}, expected {(side, side, side)}")
        if not np.isclose(values[4], 0.02) or not np.isclose(values[5], 0.31):
            raise SystemExit(f"bad cosmology header in {path}")
        fields[name] = path.stat().st_size
        headers[name] = {"dims": list(dims), "dx_mpc": values[0], "offset_mpc": list(values[1:4])}
    levels[str(level)] = {"side_cells": side, "file_sizes": fields, "headers": headers}

def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

payload = {
    "schema": "ouruniv-cf4-s40349-traceonly-zoom-ic-l12-v1",
    "stage": "8/8 bounded zoom-IC pilot",
    "selection_status": "trace-only-zoom-candidate",
    "parent_promoted": False,
    "m33_resolved": False,
    "manifest": str(manifest_path),
    "manifest_sha256": digest(manifest_path),
    "levels": levels,
    "global_base_spacing_cMpc_h": 0.75,
    "finest_particle_spacing_cMpc_h": 0.09375,
    "runtime_levelmax": 19,
    "minimum_runtime_cell_ckpc_h": 0.732421875,
    "decision": "TRACE_ONLY_ZOOM_IC_L12_GENERATED_PREFLIGHT_PENDING",
    "interpretation": (
        "The L12 pilot preserves the exact seed-40349 L9 parent modes and adds one "
        "explicit unconstrained LCDM-shaped high-k realization inside the spatially "
        "traced mask. It is not a validated LG zoom, M33 detection, or production IC."
    ),
}
report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps(payload, sort_keys=True))
PY
printf 'TRACE_ONLY_ZOOM_IC_L12_PASS job=%s report=%s out=%s\n' \
  "$SLURM_JOB_ID" "$report" "$out"
