#!/usr/bin/env bash
set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
out=/gpfs/kjhan/IllustrisTNG/TNG100-1/training/tng100_simba_v8_observable_context_smoke

cd "$repo"
python src/hong2021_residual_v8_context.py train \
  --initialize /gpfs/kjhan/IllustrisTNG/TNG100-1/training/tng100_simba_v7_multidomain_edm/minimum_validation.pt \
  --tng-train-data /gpfs/kjhan/IllustrisTNG/TNG100-1/derived/hong2021_v2/split00_l0_paper/tng100_train.h5 \
  --tng-train-cache /gpfs/kjhan/IllustrisTNG/TNG100-1/derived/hong2021_v6/tng100_train_laplacian_sigma2.h5 \
  --simba-train-data /gpfs/kjhan/CAMELS/SIMBA/L25n256/derived/hong2021_v1/simba_cv16_23_train_all_observers.h5 \
  --simba-train-cache /gpfs/kjhan/CAMELS/SIMBA/L25n256/derived/hong2021_v1/simba_cv16_23_train_laplacian_sigma2.h5 \
  --tng-validation-data /gpfs/kjhan/IllustrisTNG/TNG100-1/derived/hong2021_v2/split00_l0_paper/tng100_validation.h5 \
  --tng-validation-cache /gpfs/kjhan/IllustrisTNG/TNG100-1/derived/hong2021_v6/tng100_validation_laplacian_sigma2.h5 \
  --simba-validation-data /gpfs/kjhan/CAMELS/SIMBA/L25n256/derived/hong2021_v1/simba_cv24_26_validation_all_observers.h5 \
  --simba-validation-cache /gpfs/kjhan/CAMELS/SIMBA/L25n256/derived/hong2021_v1/simba_cv24_26_validation_laplacian_sigma2.h5 \
  --out "$out" \
  --steps 20 --validation-every 10 --smoke-limit 8 --workers 0

python src/hong2021_residual_v8_context.py sample \
  --data /gpfs/kjhan/CAMELS/SIMBA/L25n256/derived/hong2021_v1/simba_cv24_26_validation_all_observers.h5 \
  --cache /gpfs/kjhan/CAMELS/SIMBA/L25n256/derived/hong2021_v1/simba_cv24_26_validation_laplacian_sigma2.h5 \
  --checkpoint "$out/minimum_validation.pt" \
  --out "$out/smoke_sample.h5" \
  --indices 0 --ensemble 2 --sampling-steps 4

python - "$out" <<'PY'
import h5py
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
with h5py.File(root / "smoke_sample.h5") as handle:
    sample = handle["sample"][:]
report = {
    "status": "passed" if bool((sample == sample).all()) else "failed",
    "shape": list(sample.shape),
    "minimum": float(sample.min()),
    "maximum": float(sample.max()),
}
(root / "smoke_result.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
PY
