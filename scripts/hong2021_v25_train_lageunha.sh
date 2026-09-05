#!/usr/bin/env bash
set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
out=/gpfs/kjhan/IllustrisTNG/TNG100-1/training/tng100_simba_swift_v25_e13_unweighted_edm
preflight=/gpfs/kjhan/IllustrisTNG/TNG100-1/evaluation/tng100_simba_swift_v25_sequence/preflight.json
lock=/gpfs/kjhan/.hong2021_locks/v25_e13_training.lock
cd "$repo"
[[ ${HOSTNAME,,} == lageunha ]] || { echo "V25 training must run on Lageunha" >&2; exit 1; }
python - <<'PY'
import torch
if not torch.cuda.is_available() or "ada" not in torch.cuda.get_device_name(0).lower():
    raise RuntimeError("V25 training requires the Lageunha Ada GPU")
PY
mkdir -p "$(dirname "$lock")"
exec 8>"$lock"
flock -n 8 || { echo "another V25 training process holds the lock" >&2; exit 2; }
exec python -u src/hong2021_v25_edm.py train \
  --registry "$repo/config/hong2021_v25_development_program.json" \
  --repo "$repo" --out "$out" --preflight "$preflight" --device cuda
