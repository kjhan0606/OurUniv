#!/usr/bin/env bash
set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
tng=/gpfs/kjhan/IllustrisTNG/TNG100-1
program=$repo/config/hong2021_v56_survival_grid_program.json
expected_program_sha=ec93d8d0894292793279edafad5e8243a272e2c55d52d4c27b3a0cd9ef714f40
thresholds=$tng/evaluation/tng100_simba_swift_v54_tail_threshold_selection/thresholds.json
expected_threshold_sha=e5c1cd480ac47b52d568e1f8c8b8386a68230150e7f2079ad3e832220557d103
grid_root=$tng/evaluation/tng100_simba_swift_v56_survival_grid
grid=$grid_root/grid.json
derived=$tng/derived/hong2021_v56
checkpoint=$derived/upper_survival_grid_step12000.pt
report=$derived/upper_survival_grid_step12000.json
sequence=$tng/evaluation/tng100_simba_swift_v56_survival_grid_sequence
preflight=$sequence/preflight.json
train_gate_root=$tng/evaluation/tng100_simba_swift_v56_train_high_backbone_gate
train_gate=$train_gate_root/decision.json
development=$tng/evaluation/tng100_simba_swift_v56_e29_survival_grid
status=$sequence/status
cache=$tng/derived/hong2021_v45/conditioning_cache.h5
expected_cache_sha=f62a074927a1ee67eb8b2a43fd36f0db024bb56545c049af93578abca9412153

cd "$repo"
export PYTHONPATH=$repo/src
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16

[[ $(hostname -s | tr '[:upper:]' '[:lower:]') == lageunha ]] || { echo "V56 requires Lageunha" >&2; exit 1; }
[[ -z $(git status --porcelain) ]] || { echo "V56 requires clean worktree" >&2; exit 1; }
[[ $(sha256sum "$program" | awk '{print $1}') == "$expected_program_sha" ]] || { echo "V56 program hash differs" >&2; exit 1; }
[[ $(sha256sum "$thresholds" | awk '{print $1}') == "$expected_threshold_sha" ]] || { echo "V56 V54 threshold hash differs" >&2; exit 1; }
[[ $(sha256sum "$cache" | awk '{print $1}') == "$expected_cache_sha" ]] || { echo "V56 cache hash differs" >&2; exit 1; }
for path in "$grid_root" "$derived" "$sequence" "$train_gate_root" "$development"; do
  [[ ! -e $path ]] || { echo "V56 refuses existing output: $path" >&2; exit 1; }
done

mkdir -p "$sequence"
trap 'code=$?; current=$(cat "$status" 2>/dev/null || true); if [[ $code -ne 0 ]]; then printf "failed exit=%s\n" "$code" >"$status"; elif [[ $current != complete* ]]; then printf "%s\n" complete >"$status"; fi' EXIT

printf "%s\n" testing >"$status"
pytest -q >"$sequence/pytest.log" 2>&1

printf "%s\n" materializing_grid >"$status"
python -u src/hong2021_v56_train.py materialize-grid \
  --program "$program" --repo "$repo" \
  --thresholds "$thresholds" --thresholds-sha256 "$expected_threshold_sha" \
  --out "$grid" >"$sequence/grid.log" 2>&1
grid_sha=$(sha256sum "$grid" | awk '{print $1}')

printf "%s\n" preflight >"$status"
python -u src/hong2021_v56_train.py preflight \
  --program "$program" --repo "$repo" --cache "$cache" --cache-sha256 "$expected_cache_sha" \
  --thresholds "$thresholds" --thresholds-sha256 "$expected_threshold_sha" \
  --grid "$grid" --grid-sha256 "$grid_sha" --out "$preflight" \
  >"$sequence/preflight.log" 2>&1
preflight_sha=$(sha256sum "$preflight" | awk '{print $1}')

printf "%s\n" training >"$status"
python -u src/hong2021_v56_train.py train \
  --program "$program" --repo "$repo" --cache "$cache" --cache-sha256 "$expected_cache_sha" \
  --thresholds "$thresholds" --thresholds-sha256 "$expected_threshold_sha" \
  --grid "$grid" --grid-sha256 "$grid_sha" \
  --preflight "$preflight" --preflight-sha256 "$preflight_sha" \
  --checkpoint "$checkpoint" --report "$report" \
  >"$sequence/train.log" 2>&1
checkpoint_sha=$(sha256sum "$checkpoint" | awk '{print $1}')
report_sha=$(sha256sum "$report" | awk '{print $1}')

printf "%s\n" train_gating >"$status"
python -u src/hong2021_v56_train_gate.py \
  --program "$program" --repo "$repo" --cache "$cache" --cache-sha256 "$expected_cache_sha" \
  --thresholds-sha256 "$expected_threshold_sha" \
  --grid "$grid" --grid-sha256 "$grid_sha" \
  --preflight "$preflight" --preflight-sha256 "$preflight_sha" \
  --checkpoint "$checkpoint" --checkpoint-sha256 "$checkpoint_sha" \
  --report "$report" --report-sha256 "$report_sha" --out "$train_gate" \
  >"$sequence/train_gate.log" 2>&1
train_pass=$(jq -r '.train_mechanism_pass' "$train_gate")
if [[ $train_pass == true ]]; then
  printf "%s\n" complete_train_gate_pass_waiting_development_implementation >"$status"
else
  printf "%s\n" complete_train_gate_failure >"$status"
fi
