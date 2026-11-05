#!/usr/bin/env bash
set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
tng=/gpfs/kjhan/IllustrisTNG/TNG100-1
program=$repo/config/hong2021_v54_physical_tail_brier_program.json
expected_program_sha=84e06be3980deeb63456fb53a56d05d22eca68fea663d023b6be5e75460fbd90
threshold_root=$tng/evaluation/tng100_simba_swift_v54_tail_threshold_selection
thresholds=$threshold_root/thresholds.json
derived=$tng/derived/hong2021_v54
checkpoint=$derived/physical_tail_brier_bounded_mixture_step12000.pt
report=$derived/physical_tail_brier_bounded_mixture_step12000.json
sequence=$tng/evaluation/tng100_simba_swift_v54_physical_tail_brier_sequence
preflight=$sequence/preflight.json
train_gate_root=$tng/evaluation/tng100_simba_swift_v54_train_high_backbone_gate
train_gate=$train_gate_root/decision.json
development=$tng/evaluation/tng100_simba_swift_v54_e28_physical_tail_brier
status=$sequence/status
cache=$tng/derived/hong2021_v45/conditioning_cache.h5
expected_cache_sha=f62a074927a1ee67eb8b2a43fd36f0db024bb56545c049af93578abca9412153

cd "$repo"
export PYTHONPATH=$repo/src
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16

[[ $(hostname -s | tr '[:upper:]' '[:lower:]') == lageunha ]] || { echo "V54 requires Lageunha" >&2; exit 1; }
[[ -z $(git status --porcelain) ]] || { echo "V54 requires clean worktree" >&2; exit 1; }
[[ $(sha256sum "$program" | awk '{print $1}') == "$expected_program_sha" ]] || { echo "V54 program hash differs" >&2; exit 1; }
[[ $(sha256sum "$cache" | awk '{print $1}') == "$expected_cache_sha" ]] || { echo "V54 cache hash differs" >&2; exit 1; }
for path in "$threshold_root" "$derived" "$sequence" "$train_gate_root" "$development"; do
  [[ ! -e $path ]] || { echo "V54 refuses existing output: $path" >&2; exit 1; }
done

mkdir -p "$sequence"
trap 'code=$?; current=$(cat "$status" 2>/dev/null || true); if [[ $code -ne 0 ]]; then printf "failed exit=%s\n" "$code" >"$status"; elif [[ $current != complete* ]]; then printf "%s\n" complete >"$status"; fi' EXIT

printf "%s\n" testing >"$status"
pytest -q >"$sequence/pytest.log" 2>&1

printf "%s\n" selecting_thresholds >"$status"
python -u src/hong2021_v54_train.py select-thresholds \
  --program "$program" --repo "$repo" --out "$thresholds" \
  >"$sequence/thresholds.log" 2>&1
threshold_sha=$(sha256sum "$thresholds" | awk '{print $1}')

printf "%s\n" preflight >"$status"
python -u src/hong2021_v54_train.py preflight \
  --program "$program" --repo "$repo" --cache "$cache" --cache-sha256 "$expected_cache_sha" \
  --thresholds "$thresholds" --thresholds-sha256 "$threshold_sha" --out "$preflight" \
  >"$sequence/preflight.log" 2>&1
preflight_sha=$(sha256sum "$preflight" | awk '{print $1}')

printf "%s\n" training >"$status"
python -u src/hong2021_v54_train.py train \
  --program "$program" --repo "$repo" --cache "$cache" --cache-sha256 "$expected_cache_sha" \
  --thresholds "$thresholds" --thresholds-sha256 "$threshold_sha" \
  --preflight "$preflight" --preflight-sha256 "$preflight_sha" \
  --checkpoint "$checkpoint" --report "$report" \
  >"$sequence/train.log" 2>&1
checkpoint_sha=$(sha256sum "$checkpoint" | awk '{print $1}')
report_sha=$(sha256sum "$report" | awk '{print $1}')

printf "%s\n" train_gating >"$status"
python -u src/hong2021_v54_train_gate.py \
  --program "$program" --repo "$repo" --cache "$cache" --cache-sha256 "$expected_cache_sha" \
  --thresholds "$thresholds" --thresholds-sha256 "$threshold_sha" \
  --preflight "$preflight" --preflight-sha256 "$preflight_sha" \
  --checkpoint "$checkpoint" --checkpoint-sha256 "$checkpoint_sha" \
  --report "$report" --report-sha256 "$report_sha" --out "$train_gate" \
  >"$sequence/train_gate.log" 2>&1
train_gate_sha=$(sha256sum "$train_gate" | awk '{print $1}')
train_pass=$(python - "$train_gate" <<'PY'
import json,sys
print("true" if json.load(open(sys.argv[1]))["train_mechanism_pass"] else "false")
PY
)
if [[ $train_pass != true ]]; then
  printf "%s\n" complete_train_gate_failure >"$status"
  exit 0
fi

printf "%s\n" sampling >"$status"
python -u src/hong2021_v54_sample.py \
  --program "$program" --repo "$repo" --cache "$cache" --cache-sha256 "$expected_cache_sha" \
  --thresholds "$thresholds" --thresholds-sha256 "$threshold_sha" \
  --preflight "$preflight" --preflight-sha256 "$preflight_sha" \
  --checkpoint "$checkpoint" --checkpoint-sha256 "$checkpoint_sha" \
  --report "$report" --report-sha256 "$report_sha" \
  --train-gate "$train_gate" --train-gate-sha256 "$train_gate_sha" --out "$development" \
  >"$sequence/sample.log" 2>&1

printf "%s\n" evaluating >"$status"
for arm in bounded_query_local_mixture_copula rolled_parameter_control; do
  for domain in tng simba_dev swift_dev; do
    root=$development/$arm/development_candidate/$domain
    python -u src/hong2021_residual_evaluate.py \
      --candidate "edm=$root/ensemble16.h5" --out "$root/ensemble_evaluation" --voxel-mpc-h .3125 \
      >"$root/evaluate.log" 2>&1
  done
done

printf "%s\n" gating >"$status"
python -u src/hong2021_v54_development_gate.py \
  --root "$development" --program "$program" --repo "$repo" \
  --train-gate "$train_gate" --train-gate-sha256 "$train_gate_sha" \
  --out "$development/development_decision.json" \
  >"$development/development_decision.log" 2>&1
printf "%s\n" complete_development_gate >"$status"
