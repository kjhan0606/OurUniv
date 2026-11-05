#!/usr/bin/env bash
set -Eeuo pipefail
readonly repo=/home/kjhan/BACKUP/CF4
readonly python=/home/kjhan/miniconda3/envs/circle/bin/python
readonly program="$repo/config/cf4_aggregate_evidence_smc_execution_authorization_program_v6_open_pilot.json"
readonly expected_host=lageunha
host=$(hostname); readonly host
short=${host%%.*}; readonly short
lower=$(LC_ALL=C tr '[:upper:]' '[:lower:]' <<<"$short"); readonly lower
if [[ "$lower" != "$expected_host" ]]; then printf 'host gate failed: %s\n' "$host" >&2; exit 69; fi
set +e
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$repo/src" "$python" - "$program" "$host" <<'PY'
from pathlib import Path
import sys
from cf4_aggregate_evidence_smc_execution_authorized_v6_open_pilot import PROGRAM, run_authorized_v6_open_pilot
if Path(sys.argv[1]).resolve() != PROGRAM.resolve():
    raise SystemExit("noncanonical pilot program")
run_authorized_v6_open_pilot(PROGRAM, sys.argv[2])
PY
readonly gate_rc=$?
set -e
printf 'pilot execution boundary refused before mutation (gate_rc=%s)\n' "$gate_rc" >&2
exit 65
