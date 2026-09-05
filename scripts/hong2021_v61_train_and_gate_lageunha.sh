#!/usr/bin/env bash
set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
tng=/gpfs/kjhan/IllustrisTNG/TNG100-1
program=$repo/config/hong2021_v61_reachable_support_model_program.json
expected_program_sha=327d750774a82885888ff08313e829462d43c877d45effc990f1035358f04cd1
record=$repo/config/hong2021_v61_preflight_record.json
expected_record_sha=3ec5bca83a1088aa733f0c0b8534d9ea09affe46f1d8777e433b57ae2c9cce17
thresholds=$tng/evaluation/tng100_simba_swift_v54_tail_threshold_selection/thresholds.json
expected_threshold_sha=e5c1cd480ac47b52d568e1f8c8b8386a68230150e7f2079ad3e832220557d103
grid=$tng/evaluation/tng100_simba_swift_v60_reachable_support_grid/grid.json
expected_grid_sha=d9c9aa2d91c746139589b10a858ff749b114b3eb8cec0fb134f823b39db0a2db
cache=$tng/derived/hong2021_v45/conditioning_cache.h5
expected_cache_sha=f62a074927a1ee67eb8b2a43fd36f0db024bb56545c049af93578abca9412153
sequence=$tng/evaluation/tng100_simba_swift_v61_reachable_support_sequence
preflight=$sequence/preflight.json
expected_preflight_sha=d900088be814bc969a94e4e4e2ff8b85d358665ab52dc9c31539e17bacb43063
derived=$tng/derived/hong2021_v61
checkpoint=$derived/reachable_support_grid_step12000.pt
report=$derived/reachable_support_grid_step12000.json
train_gate_root=$tng/evaluation/tng100_simba_swift_v61_train_high_backbone_gate
train_gate=$train_gate_root/decision.json
development=$tng/evaluation/tng100_simba_swift_v61_e30_reachable_support_grid
status=$sequence/status

cd "$repo"
export PYTHONPATH=$repo/src
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

[[ $(hostname -s | tr '[:upper:]' '[:lower:]') == lageunha ]] || {
  echo "V61 training requires Lageunha" >&2
  exit 1
}
[[ -z $(git status --porcelain) ]] || {
  echo "V61 training requires clean worktree" >&2
  exit 1
}
[[ -d $sequence && $(cat "$status") == complete_preflight_pass ]] || {
  echo "V61 training requires the completed preflight sequence" >&2
  exit 1
}
for binding in \
  "$program:$expected_program_sha" \
  "$record:$expected_record_sha" \
  "$thresholds:$expected_threshold_sha" \
  "$grid:$expected_grid_sha" \
  "$cache:$expected_cache_sha" \
  "$preflight:$expected_preflight_sha"; do
  path=${binding%:*}
  expected=${binding##*:}
  [[ $(sha256sum "$path" | awk '{print $1}') == "$expected" ]] || {
    echo "V61 frozen input hash differs: $path" >&2
    exit 1
  }
done
for path in "$derived" "$train_gate_root" "$development"; do
  [[ ! -e $path ]] || {
    echo "V61 refuses existing output: $path" >&2
    exit 1
  }
done

record_failure() {
  code=$?
  current=$(cat "$status" 2>/dev/null || true)
  if [[ $code -ne 0 ]]; then
    printf "failed_training_sequence exit=%s previous=%s\n" \
      "$code" "$current" >"$status"
  fi
}
trap record_failure EXIT

printf "%s\n" testing_training_implementation >"$status"
pytest -q >"$sequence/train_pytest.log" 2>&1

printf "%s\n" training >"$status"
python -u src/hong2021_v61_train.py \
  --program "$program" --repo "$repo" \
  --cache "$cache" --cache-sha256 "$expected_cache_sha" \
  --thresholds "$thresholds" --thresholds-sha256 "$expected_threshold_sha" \
  --grid "$grid" --grid-sha256 "$expected_grid_sha" \
  --preflight "$preflight" --preflight-sha256 "$expected_preflight_sha" \
  --checkpoint "$checkpoint" --report "$report" \
  >"$sequence/train.log" 2>&1
checkpoint_sha=$(sha256sum "$checkpoint" | awk '{print $1}')
report_sha=$(sha256sum "$report" | awk '{print $1}')

printf "%s\n" train_gating >"$status"
python -u src/hong2021_v61_train_gate.py \
  --program "$program" --repo "$repo" \
  --cache "$cache" --cache-sha256 "$expected_cache_sha" \
  --thresholds "$thresholds" --thresholds-sha256 "$expected_threshold_sha" \
  --grid "$grid" --grid-sha256 "$expected_grid_sha" \
  --preflight "$preflight" --preflight-sha256 "$expected_preflight_sha" \
  --checkpoint "$checkpoint" --checkpoint-sha256 "$checkpoint_sha" \
  --report "$report" --report-sha256 "$report_sha" --out "$train_gate" \
  >"$sequence/train_gate.log" 2>&1
if [[ $(jq -r '.train_mechanism_pass' "$train_gate") == true ]]; then
  printf "%s\n" complete_train_gate_pass_waiting_locked_development >"$status"
else
  printf "%s\n" complete_train_gate_failure_development_locked >"$status"
fi
trap - EXIT
