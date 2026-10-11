#!/usr/bin/env bash
set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
tng=/gpfs/kjhan/IllustrisTNG/TNG100-1
sequence=$tng/evaluation/tng100_simba_swift_v63_conditional_moment_sequence
status=$sequence/status
development=$tng/evaluation/tng100_simba_swift_v63_e31_conditional_moment
program=$repo/config/hong2021_v63_conditional_moment_model_program.json
program_sha=ea41d61a2961b3f436ed69662dc39ad8ad151980aca32863c0442948d31b6a48
train_gate=$tng/evaluation/tng100_simba_swift_v63_train_high_backbone_gate/decision.json
train_gate_sha=17cddbc731f9c34b6a471709eb9a28b43e6a5e07b59e063c7ccdd082b1e0f95c

cd "$repo"
export PYTHONPATH=$repo/src
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16

[[ $(hostname -s | tr '[:upper:]' '[:lower:]') == lageunha ]] || {
  echo "V63 development evaluation resume requires Lageunha" >&2
  exit 1
}
[[ -z $(git status --porcelain) ]] || {
  echo "V63 development evaluation resume requires clean worktree" >&2
  exit 1
}
[[ $(cat "$status") == "failed_development_sequence exit=1 previous=development_evaluating" ]] || {
  echo "V63 resume requires the sealed evaluator-schema failure" >&2
  exit 1
}
[[ $(sha256sum "$program" | awk '{print $1}') == "$program_sha" ]] || exit 1
[[ $(sha256sum "$train_gate" | awk '{print $1}') == "$train_gate_sha" ]] || exit 1
[[ ! -e $development/development_decision.json ]] || {
  echo "V63 resume refuses an existing development decision" >&2
  exit 1
}
for arm in bounded_query_local_mixture_copula rolled_parameter_control; do
  for domain in tng simba_dev swift_dev; do
    root=$development/$arm/development_candidate/$domain
    [[ -f $root/ensemble16.h5 \
      && ! -e $root/ensemble_evaluation/metrics.json \
      && ! -e $root/ensemble_evaluation/metrics_edm.json \
      && ! -e $root/ensemble_evaluation/diagnostics.png ]] || {
      echo "V63 resume requires complete unevaluated ensembles" >&2
      exit 1
    }
  done
done

record_failure() {
  code=$?
  current=$(cat "$status" 2>/dev/null || true)
  if [[ $code -ne 0 ]]; then
    printf "failed_development_resume exit=%s previous=%s\n" \
      "$code" "$current" >"$status"
  fi
}
trap record_failure EXIT

printf "%s\n" development_evaluating_resumed_after_schema_registration >"$status"
for arm in bounded_query_local_mixture_copula rolled_parameter_control; do
  for domain in tng simba_dev swift_dev; do
    root=$development/$arm/development_candidate/$domain
    python -u src/hong2021_residual_evaluate.py \
      --candidate "edm=$root/ensemble16.h5" \
      --out "$root/ensemble_evaluation" --voxel-mpc-h .3125 \
      >"$root/evaluate.log" 2>&1
  done
done

printf "%s\n" development_gating >"$status"
python -u src/hong2021_v63_development_gate.py \
  --root "$development" --program "$program" --repo "$repo" \
  --train-gate "$train_gate" --train-gate-sha256 "$train_gate_sha" \
  --out "$development/development_decision.json" \
  >"$development/development_decision.log" 2>&1
if [[ $(jq -r '.development_pass' "$development/development_decision.json") == true ]]; then
  printf "%s\n" complete_development_gate_pass_waiting_explicit_EAGLE_approval >"$status"
else
  printf "%s\n" complete_development_gate_failure_independent_gate_locked >"$status"
fi
trap - EXIT
