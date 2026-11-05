#!/usr/bin/env bash
set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
log=/gpfs/kjhan/IllustrisTNG/TNG100-1/training/hong2021_v13_pipeline.log
cd "$repo"
bash scripts/run_hong2021_v13_gate_sequence_lageunha.sh 2>&1 | tee -a "$log"
