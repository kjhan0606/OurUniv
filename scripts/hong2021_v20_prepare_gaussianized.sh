#!/usr/bin/env bash
set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
registry=$repo/config/hong2021_v20_development_program.json
cd "$repo"
export PYTHONPATH=$repo/src

host=$(hostname)
if [[ ${host,,} != lageunha ]]; then
  echo "V20 Gaussianization audit must run on Lageunha, not $host" >&2
  exit 1
fi

python src/hong2021_v20_gaussianize.py verify \
  --registry "$registry" --repo "$repo"
