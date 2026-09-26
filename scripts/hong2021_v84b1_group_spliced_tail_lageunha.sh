#!/usr/bin/env bash
set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
tng=/gpfs/kjhan/IllustrisTNG/TNG100-1
program=$repo/config/hong2021_v84b_group_spliced_tail_program.json
program_sha=f07ca34f9e5ab57c2625aba138dba8056193d049bae591abadcb820197896175
freeze_commit=c0f1ae2498d0f4b75f4b182a94d3e6a51c3953b2
cache=$tng/derived/hong2021_v45/conditioning_cache.h5
cache_sha=f62a074927a1ee67eb8b2a43fd36f0db024bb56545c049af93578abca9412153
sequence=$tng/evaluation/tng100_simba_swift_v84b1_group_spliced_tail_sequence
preflight=$sequence/preflight.json
derived=$tng/derived/hong2021_v84b1_group_spliced_tail
checkpoint=$derived/step12000.pt
training_report=$derived/training_report.json
group_gate=$tng/evaluation/tng100_simba_swift_v84b1_group_spliced_tail_gate/decision.json
status=$sequence/status

cd "$repo"
export PYTHONPATH=$repo/src
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[[ $(hostname -s | tr '[:upper:]' '[:lower:]') == lageunha ]] || exit 1
[[ -z $(git status --porcelain) ]] || exit 1
git merge-base --is-ancestor "$freeze_commit" HEAD
[[ $(sha256sum "$program" | awk '{print $1}') == "$program_sha" ]] || {
  echo "V84B1 frozen amended program differs" >&2
  exit 1
}
[[ $(sha256sum "$cache" | awk '{print $1}') == "$cache_sha" ]] || {
  echo "V84B1 conditioning cache differs" >&2
  exit 1
}
for path in "$sequence" "$derived" "$(dirname "$group_gate")"; do
  [[ ! -e $path ]] || {
    echo "V84B1 refuses existing output: $path" >&2
    exit 1
  }
done

mkdir -p "$sequence"
record_failure() {
  code=$?
  if [[ $code -ne 0 ]]; then
    previous=$(cat "$status" 2>/dev/null || true)
    printf 'failed_terminal_V84B1_no_retry exit=%s previous=%s\n' \
      "$code" "$previous" >"$status"
  fi
}
trap record_failure EXIT

printf '%s\n' testing_frozen_V84B1 >"$status"
taskset -c 64-79 nice -n 10 pytest -q >"$sequence/pytest.log" 2>&1

printf '%s\n' running_group_fit_only_hard_preflight >"$status"
taskset -c 64-79 nice -n 10 \
  python -u src/hong2021_v84b_preflight.py \
  --program "$program" --repo "$repo" \
  --conditioning-cache "$cache" --conditioning-cache-sha256 "$cache_sha" \
  --out "$preflight" >"$sequence/preflight.log" 2>&1
preflight_sha=$(sha256sum "$preflight" | awk '{print $1}')

printf '%s\n' training_fixed_12000_step_group_fit_V84B1 >"$status"
taskset -c 64-79 nice -n 10 \
  python -u src/hong2021_v84b_train.py \
  --program "$program" --repo "$repo" \
  --conditioning-cache "$cache" --conditioning-cache-sha256 "$cache_sha" \
  --preflight "$preflight" --preflight-sha256 "$preflight_sha" \
  --checkpoint "$checkpoint" --report "$training_report" \
  >"$sequence/train.log" 2>&1
checkpoint_sha=$(sha256sum "$checkpoint" | awk '{print $1}')
training_report_sha=$(sha256sum "$training_report" | awk '{print $1}')

printf '%s\n' evaluating_group_holdout_once_V84B1 >"$status"
taskset -c 64-79 nice -n 10 \
  python -u src/hong2021_v84b_group_gate.py \
  --program "$program" --repo "$repo" \
  --conditioning-cache "$cache" --conditioning-cache-sha256 "$cache_sha" \
  --checkpoint "$checkpoint" --checkpoint-sha256 "$checkpoint_sha" \
  --report "$training_report" --report-sha256 "$training_report_sha" \
  --out "$group_gate" >"$sequence/group_gate.log" 2>&1

if [[ $(jq -r '.group_held_out_mechanism_pass' "$group_gate") == true ]]; then
  printf '%s\n' complete_V84B1_group_gate_pass_waiting_result_record_before_any_production_refit >"$status"
else
  printf '%s\n' complete_V84B1_group_gate_failure_all_other_payloads_locked >"$status"
fi
trap - EXIT
