#!/usr/bin/env bash
set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
base=/gpfs/kjhan/IllustrisTNG/TNG100-1/evaluation
program=$repo/config/hong2021_v80_single_candidate_program.json
program_sha=43d41ae9722e2321d3a206492d543e39ba5e8b22868e4f3a755cd1b2147dc205
program_freeze=f7cdf927df18cd9047062dbafd3fcecef30de8f2
v79_program=$repo/config/hong2021_v79_complete_candidate_agnostic_gate_program.json
sequence=$base/tng100_simba_swift_v80_sequence
ensemble_root=$base/tng100_simba_swift_v80_calibrated_sqt_single_candidate
gate_root=$base/tng100_simba_swift_v80_v79_single_use_gate
preflight=$sequence/preflight.json
manifest=$sequence/execution_manifest.json
status=$sequence/status

cd "$repo"
export PYTHONPATH=$repo/src
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8

[[ $(hostname -s | tr '[:upper:]' '[:lower:]') == lageunha ]] || {
  echo "V80 requires Lageunha" >&2
  exit 1
}
[[ -z $(git status --porcelain) ]] || {
  echo "V80 requires a clean worktree" >&2
  exit 1
}
[[ $(sha256sum "$program" | awk '{print $1}') == "$program_sha" ]] || exit 1
git merge-base --is-ancestor "$program_freeze" HEAD
[[ ! -e $sequence && ! -e $ensemble_root && ! -e $gate_root ]] || {
  echo "V80 refuses an existing single-use output" >&2
  exit 1
}

mkdir -p "$sequence"
record_failure() {
  code=$?
  if [[ $code -ne 0 ]]; then
    previous=$(cat "$status" 2>/dev/null || true)
    printf 'failed_terminal_V80_no_retry exit=%s previous=%s\n' "$code" "$previous" >"$status"
  fi
}
trap record_failure EXIT

printf '%s\n' testing_frozen_V80_and_V79 >"$status"
pytest -q >"$sequence/pytest.log" 2>&1

printf '%s\n' running_code_only_V80_preflight >"$status"
python -u src/hong2021_v80_sample.py preflight \
  --program "$program" --repo "$repo" --output-root "$ensemble_root" \
  --out "$preflight" >"$sequence/preflight.log" 2>&1

printf '%s\n' sampling_single_frozen_V80_candidate_and_control >"$status"
python -u src/hong2021_v80_sample.py sample \
  --program "$program" --preflight "$preflight" --repo "$repo" \
  --output-root "$ensemble_root" >"$sequence/sample.log" 2>&1

printf '%s\n' evaluating_six_frozen_V80_ensembles >"$status"
pids=()
slot=0
for arm in candidate control; do
  for domain in tng simba_dev swift_dev; do
    root=$ensemble_root/$arm/$domain
    low=$((64 + slot * 10))
    high=$((low + 9))
    mkdir -p "$root/ensemble_evaluation"
    taskset -c "$low-$high" nice -n 10 \
      python -u src/hong2021_v80_evaluate.py \
      --candidate "v80=$root/ensemble16.h5" \
      --out "$root/ensemble_evaluation" --voxel-mpc-h .3125 \
      >"$root/evaluate.log" 2>&1 &
    pids+=("$!")
    slot=$((slot + 1))
  done
done
for pid in "${pids[@]}"; do
  wait "$pid"
done

printf '%s\n' sealing_V80_execution_manifest >"$status"
python -u src/hong2021_v80_manifest.py \
  --candidate-program "$program" --preflight "$preflight" --repo "$repo" \
  --out "$manifest" >"$sequence/manifest.log" 2>&1

printf '%s\n' running_single_use_V79_complete_gate >"$status"
taskset -c 64-95 nice -n 10 \
  python -u src/hong2021_v79_complete_gate.py \
  --program "$v79_program" --manifest "$manifest" --repo "$repo" \
  --output-root "$gate_root" >"$sequence/V79_gate.log" 2>&1

if [[ $(jq -r '.evaluation.candidate_pass' "$gate_root/decision.json") == true ]]; then
  printf '%s\n' complete_V80_V79_pass_sealed_waiting_user_decision >"$status"
else
  printf '%s\n' complete_V80_V79_failure_sealed_no_retry >"$status"
fi
trap - EXIT
