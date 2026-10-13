#!/usr/bin/env bash
set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
seal=${1:-config/hong2021_v15_astrid_one_shot_seal.json}
export HONG2021_ASTRID_FREEZE_SCRIPT=src/hong2021_v15_freeze.py
export HONG2021_ASTRID_EVALUATION=/gpfs/kjhan/CAMELS/Astrid/L25n256/evaluation/hong2021_v15_astrid_one_shot
export HONG2021_ASTRID_STATE_SCHEMA=hong2021-v15-astrid-one-shot-sequence-status-v1
export HONG2021_V15_ASTRID_ONE_SHOT=sealed
cd "$repo"
exec scripts/run_hong2021_v14_astrid_one_shot_lageunha.sh "$seal"
