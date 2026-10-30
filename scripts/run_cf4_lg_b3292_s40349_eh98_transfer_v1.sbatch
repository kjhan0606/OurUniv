#!/usr/bin/env bash
#SBATCH --job-name=cf4_s40349_eh98
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --no-requeue
#SBATCH --output=/gpfs/kjhan/CF4/logs/grammar_%x_%j.out
#SBATCH --error=/gpfs/kjhan/CF4/logs/grammar_%x_%j.err
set -Eeuo pipefail

root=/home/kjhan/BACKUP/CF4
source=/gpfs/kjhan/CF4/recon/linear_cr/parent_b3292_beta025_lg_survival_calibration_v1/i248_s40349/transfer_l9.npz
output=/gpfs/kjhan/CF4/recon/linear_cr/parent_b3292_beta025_lg_survival_calibration_v1/i248_s40349/transfer_l9_eh98_v1.npz
report=/gpfs/kjhan/CF4/diagnostics/cf4_lg_b3292_s40349_eh98_transfer_v1.json

test -s "$source"
test ! -e "$output"
test ! -e "$report"
cd "$root"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/home/kjhan/miniconda3/envs/circle/bin/python3.11 src/cf4_eh98_transfer.py \
  --npz "$source" --Om 0.31 --Ob 0.0403 --h 0.746 --ns 0.9649 \
  --out "$output"
test -s "$output"

/home/kjhan/miniconda3/envs/circle/bin/python3.11 - "$source" "$output" "$report" <<'PY'
import hashlib
import json
import sys

import numpy as np

source, output, report = sys.argv[1:]

def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

src = np.load(source, allow_pickle=False)
out = np.load(output, allow_pickle=False)
required = {"kphys", "T", "r", "kf", "kNyq", "N", "box_hmpc", "tail_coeff", "k_switch"}
if not required.issubset(out.files):
    raise SystemExit(f"missing transfer keys: {sorted(required - set(out.files))}")
if int(out["N"]) != 512 or not np.isclose(float(out["box_hmpc"]), 384.0):
    raise SystemExit("transfer parent geometry mismatch")
if not all(np.isfinite(out[name]).all() for name in required):
    raise SystemExit("non-finite EH98 transfer")

knyq = float(out["kNyq"])
eps = 1.0e-4
t_below = float(np.interp(knyq * (1.0 - eps), out["kphys"], out["T"]))
t_above = float(10.0 ** np.polyval(out["tail_coeff"], np.log10(knyq * (1.0 + eps))))
join_ratio = t_above / t_below
t_2 = float(10.0 ** np.polyval(out["tail_coeff"], np.log10(2.0 * knyq)))
t_4 = float(10.0 ** np.polyval(out["tail_coeff"], np.log10(4.0 * knyq)))
if not 0.9 <= join_ratio <= 1.1:
    raise SystemExit(f"EH98 tail discontinuity at parent Nyquist: {join_ratio}")
if not (0.0 < t_4 < t_2 < t_below):
    raise SystemExit("EH98 high-k tail is not positive and decreasing")

payload = {
    "schema": "ouruniv-cf4-s40349-eh98-transfer-v1",
    "selection_status": "trace-only-zoom-candidate",
    "parent_promoted": False,
    "source": source,
    "source_sha256": digest(source),
    "output": output,
    "output_sha256": digest(output),
    "N": int(out["N"]),
    "box_mpc_h": float(out["box_hmpc"]),
    "k_nyquist_h_mpc": knyq,
    "k_switch_h_mpc": float(out["k_switch"]),
    "nyquist_join_ratio": join_ratio,
    "tail_T_2knyq": t_2,
    "tail_T_4knyq": t_4,
    "decision": "TRACE_ONLY_LCDM_HIGHK_TRANSFER_PASS",
    "interpretation": (
        "Measured seed-40349 L9 modes are retained through the reliable overlap and "
        "the unconstrained high-k tail uses an LCDM-shaped EH98 continuation. This "
        "does not validate or promote the parent realization."
    ),
}
with open(report, "w") as stream:
    json.dump(payload, stream, indent=2, sort_keys=True)
    stream.write("\n")
print(json.dumps(payload, sort_keys=True))
PY
printf 'TRACE_ONLY_LCDM_TRANSFER_PASS job=%s report=%s output=%s\n' \
  "$SLURM_JOB_ID" "$report" "$output"
