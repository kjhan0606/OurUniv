#!/usr/bin/env bash
set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
tng=/gpfs/kjhan/IllustrisTNG/TNG100-1
program=$repo/config/hong2021_v63_conditional_moment_model_program.json
expected_program_sha=ea41d61a2961b3f436ed69662dc39ad8ad151980aca32863c0442948d31b6a48
record=$repo/config/hong2021_v63_preflight_record.json
expected_record_sha=cf0abd8ec0c94db8b7489c9a0200510e1ba60af6e9524a958a96d6fa934208bb
thresholds=$tng/evaluation/tng100_simba_swift_v54_tail_threshold_selection/thresholds.json
expected_threshold_sha=e5c1cd480ac47b52d568e1f8c8b8386a68230150e7f2079ad3e832220557d103
grid=$tng/evaluation/tng100_simba_swift_v56_survival_grid/grid.json
expected_grid_sha=ba0cadb1c921c73918fcf139f121f1d9fa35e0c673ab8f62483315485bfd5fde
cache=$tng/derived/hong2021_v45/conditioning_cache.h5
expected_cache_sha=f62a074927a1ee67eb8b2a43fd36f0db024bb56545c049af93578abca9412153
sequence=$tng/evaluation/tng100_simba_swift_v63_conditional_moment_sequence
preflight=$sequence/preflight.json
expected_preflight_sha=00d7e8fd1ad182645597d52db10b773c151d2e54669775090f55c92dcc76d4db
derived=$tng/derived/hong2021_v63
checkpoint=$derived/conditional_moment_step12000.pt
report=$derived/conditional_moment_step12000.json
train_gate_root=$tng/evaluation/tng100_simba_swift_v63_train_high_backbone_gate
train_gate=$train_gate_root/decision.json
development=$tng/evaluation/tng100_simba_swift_v63_e31_conditional_moment
status=$sequence/status

cd "$repo"
export PYTHONPATH=$repo/src
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

[[ $(hostname -s | tr '[:upper:]' '[:lower:]') == lageunha ]] || {
  echo "V63 training requires Lageunha" >&2
  exit 1
}
[[ -z $(git status --porcelain) ]] || {
  echo "V63 training requires clean worktree" >&2
  exit 1
}
[[ -d $sequence && $(cat "$status") == complete_preflight_pass ]] || {
  echo "V63 training requires the completed preflight sequence" >&2
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
    echo "V63 frozen input hash differs: $path" >&2
    exit 1
  }
done
for path in "$derived" "$train_gate_root" "$development"; do
  [[ ! -e $path ]] || {
    echo "V63 refuses existing output: $path" >&2
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
python -u src/hong2021_v63_train.py \
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
python -u src/hong2021_v63_train_gate.py \
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
