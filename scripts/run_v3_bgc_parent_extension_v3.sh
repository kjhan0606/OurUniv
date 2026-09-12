#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export OUT=${OUT:-/gpfs/kjhan/CF4/recon/linear_cr/v3_bgc_parent_extension_v3}
export GATE_OUT=${GATE_OUT:-/gpfs/kjhan/CF4/recon/linear_cr/v3_bgc_parent_extension_gate_v3}
export P1_OUT=${P1_OUT:-/gpfs/kjhan/CF4/recon/linear_cr/v3_bgc_p1_observer_extension_v3}
export TAG=${TAG:-v3_bgc_parent_extension_v3}
export SEEDS=${SEEDS:-$(seq -s, 3193 3448)}
exec bash "$ROOT/scripts/run_v3_bgc_parent_extension_v1.sh"
