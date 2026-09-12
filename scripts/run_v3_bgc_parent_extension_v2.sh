#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export OUT=${OUT:-/gpfs/kjhan/CF4/recon/linear_cr/v3_bgc_parent_extension_v2}
export GATE_OUT=${GATE_OUT:-/gpfs/kjhan/CF4/recon/linear_cr/v3_bgc_parent_extension_gate_v2}
export P1_OUT=${P1_OUT:-/gpfs/kjhan/CF4/recon/linear_cr/v3_bgc_p1_observer_extension_v2}
export TAG=${TAG:-v3_bgc_parent_extension_v2}
export SEEDS=${SEEDS:-$(seq -s, 3065 3192)}
exec bash "$ROOT/scripts/run_v3_bgc_parent_extension_v1.sh"
