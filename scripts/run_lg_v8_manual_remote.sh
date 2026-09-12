#!/usr/bin/env bash
set -euo pipefail

pidfile=${1:?usage: run_lg_v8_manual_remote.sh PIDFILE}
repo=/home/kjhan/BACKUP/CF4

printf '%s\n' "$$" >"$pidfile"
cd "$repo"
exec bash scripts/run_lg_v8_z0_importance_slurm.sh
