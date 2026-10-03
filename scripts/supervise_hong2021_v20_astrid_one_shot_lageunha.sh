#!/usr/bin/env bash
set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
seal=${1:-config/hong2021_v20_astrid_one_shot_seal.json}
export HONG2021_ASTRID_EVALUATION=/gpfs/kjhan/CAMELS/Astrid/L25n256/evaluation/hong2021_v20_astrid_one_shot
export HONG2021_ASTRID_RUNNER=scripts/run_hong2021_v20_astrid_one_shot_lageunha.sh
cd "$repo"
exec scripts/supervise_hong2021_v14_astrid_one_shot_lageunha.sh "$seal"
