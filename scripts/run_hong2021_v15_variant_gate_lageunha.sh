#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ($1 != e2 && $1 != e3) ]]; then
  echo "usage: $0 e2|e3" >&2
  exit 2
fi
experiment=$1
repo=/home/kjhan/BACKUP/CF4
tng=/gpfs/kjhan/IllustrisTNG/TNG100-1
simba=/gpfs/kjhan/CAMELS/SIMBA/L25n256
swift=/gpfs/kjhan/CAMELS/Swift-EAGLE/L25n256
model=$tng/derived/hong2021_v14/model
if [[ $experiment == e2 ]]; then
  training=$tng/training/tng100_simba_swift_v15_e2_relative_noise_edm
  evaluation=$tng/evaluation/tng100_simba_swift_v15_e2_relative_noise_edm
  tng_seed=25777
  simba_seed=26777
  swift_seed=27777
else
  training=$tng/training/tng100_simba_swift_v15_e3_relative_noise_tail_quarter_edm
  evaluation=$tng/evaluation/tng100_simba_swift_v15_e3_relative_noise_tail_quarter_edm
  tng_seed=35777
  simba_seed=36777
  swift_seed=37777
fi
cd "$repo"
export PYTHONPATH=$repo/src

state=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$training/run.json")
if [[ $state != complete ]]; then
  echo "$experiment training is not complete: $state" >&2
  exit 1
fi
simba_indices=$(python -c 'import json; print(",".join(map(str,json.load(open("config/hong2021_simba_dev_representative16_v1.json"))["indices"])))')
swift_indices=$(python -c 'import json; print(",".join(map(str,json.load(open("config/hong2021_swift_eagle_dev_representative16_v1.json"))["indices"])))')

sample_evaluate() {
  local checkpoint=$1 data=$2 cache=$3 indices=$4 seed=$5 root=$6
  mkdir -p "$root"
  if [[ ! -s $root/ensemble16_steps40.h5 ]]; then
    python src/hong2021_v14_edm.py sample \
      --data "$data" --cache "$cache" --checkpoint "$checkpoint" \
      --out "$root/ensemble16_steps40.h5" --indices "$indices" \
      --ensemble 16 --sampling-steps 40 --sigma-min 0.002 \
      --sigma-max 40 --rho 7 --seed "$seed" --device cuda \
      >"$root/sample.log" 2>&1
  fi
  if [[ ! -s $root/ensemble_evaluation/metrics.json ]]; then
    python src/hong2021_residual_evaluate.py \
      --candidate "edm=$root/ensemble16_steps40.h5" \
      --out "$root/ensemble_evaluation" --voxel-mpc-h 0.3125 \
      >"$root/evaluate.log" 2>&1
  fi
}

for step in 005000 010000; do
  checkpoint=$training/validation_checkpoints/step_${step}.pt
  root=$evaluation/development_candidates/step_${step}
  sample_evaluate "$checkpoint" \
    "$tng/derived/hong2021_v14/cic_data/tng100_validation.h5" \
    "$model/tng_validation_standardized.h5" \
    49,8,63,0,15,4,32,21,12,74,79,53,76,29,57,62 "$tng_seed" "$root/tng"
  sample_evaluate "$checkpoint" \
    "$simba/derived/hong2021_v14/simba_cv24_26_validation_all_observers.h5" \
    "$model/simba_validation_standardized.h5" "$simba_indices" "$simba_seed" "$root/simba_dev"
  sample_evaluate "$checkpoint" \
    "$swift/derived/hong2021_v14/swift_eagle_cv20_26_validation_all_observers.h5" \
    "$model/swift_eagle_validation_standardized.h5" "$swift_indices" "$swift_seed" "$root/swift_dev"
done

python src/hong2021_v15_development_gate.py \
  --root "$evaluation/development_candidates" --training "$training" \
  --registry config/hong2021_v15_development_program.json \
  --experiment "$experiment" --repo "$repo" \
  --out "$evaluation/development_decision.json" \
  >"$evaluation/development_decision.log" 2>&1
