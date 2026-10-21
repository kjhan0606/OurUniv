#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-/home/kjhan/miniconda3/envs/circle/bin/python}
RESULT=/gpfs/kjhan/CF4/recon/linear_cr/v3_bgc_p2_n576_parent3096_v1/p2_screen_result.json

while [[ ! -s "$RESULT" ]]; do
  sleep 30
done

if "$PYTHON" - "$RESULT" <<'PY'
import json
import sys

result = json.load(open(sys.argv[1]))
if result.get("status") != "complete":
    raise SystemExit(2)
raise SystemExit(0 if any(row["screen_pass"] for row in result["results"]) else 3)
PY
then
  echo "[fallback] P2 has a passing candidate; blind parent extension not activated"
  exit 0
else
  status=$?
  if [[ $status -ne 3 ]]; then
    exit "$status"
  fi
fi

echo "[fallback] P2 has zero passes; activating preregistered parent extension v3"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1
exec "$ROOT/scripts/run_v3_bgc_parent_extension_v3.sh"
