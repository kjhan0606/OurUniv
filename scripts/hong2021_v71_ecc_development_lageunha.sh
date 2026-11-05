#!/usr/bin/env bash
set -euo pipefail
repo=/home/kjhan/BACKUP/CF4
tng=/gpfs/kjhan/IllustrisTNG/TNG100-1
program=$repo/config/hong2021_v71_locked_ecc_development_program.json
program_sha=23665bf5d06212113de6feee7cd756c0252a59eac204def9805c5f4595ddf008
development=$tng/evaluation/tng100_simba_swift_v71_tail_preserving_ecc_development
sequence=$tng/evaluation/tng100_simba_swift_v71_tail_preserving_ecc_sequence
preflight=$sequence/preflight.json
status=$sequence/status
sealed=$sequence/sealed_result.json

cd "$repo"
export PYTHONPATH=$repo/src CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[[ $(hostname -s | tr '[:upper:]' '[:lower:]') == lageunha ]] || exit 1
[[ -z $(git status --porcelain) ]] || exit 1
[[ $(sha256sum "$program" | awk '{print $1}') == "$program_sha" ]] || exit 1
[[ ! -e $development && ! -e $sequence ]] || {
  echo "V71 refuses any existing single-use development or sequence output" >&2
  exit 1
}
mkdir -p "$sequence"

record_failure() {
  code=$?
  current=$(cat "$status" 2>/dev/null || true)
  if [[ $code -ne 0 ]]; then
    printf "failed_V71_single_use_ECC exit=%s previous=%s\n" \
      "$code" "$current" >"$status"
  fi
}
trap record_failure EXIT

printf "%s\n" testing_before_V71_code_only_preflight >"$status"
pytest -q >"$sequence/pytest.log" 2>&1

printf "%s\n" running_V71_code_only_preflight >"$status"
python -u src/hong2021_v71_preflight.py \
  --program "$program" --repo "$repo" --out "$preflight" \
  >"$sequence/preflight.log" 2>&1
preflight_sha=$(sha256sum "$preflight" | awk '{print $1}')

printf "%s\n" sampling_single_V71_ECC_development >"$status"
python -u src/hong2021_v71_development_sample.py \
  --program "$program" --repo "$repo" \
  --preflight "$preflight" --preflight-sha256 "$preflight_sha" \
  --out "$development" >"$sequence/development_sample.log" 2>&1

printf "%s\n" evaluating_single_V71_ECC_development >"$status"
for arm in tail_preserving_ECC_V70_copula_V63_marginal independent_voxel_V63_marginal; do
  for domain in tng simba_dev swift_dev; do
    root=$development/$arm/development_candidate/$domain
    python -u src/hong2021_v71_development_evaluate.py \
      --candidate "edm=$root/ensemble16.h5" \
      --out "$root/ensemble_evaluation" --voxel-mpc-h .3125 \
      >"$root/evaluate.log" 2>&1
  done
done

printf "%s\n" gating_single_V71_ECC_development >"$status"
python -u src/hong2021_v71_development_gate.py \
  --root "$development" --program "$program" --repo "$repo" \
  --preflight "$preflight" --preflight-sha256 "$preflight_sha" \
  --out "$development/development_decision.json" \
  >"$development/development_decision.log" 2>&1

printf "%s\n" sealing_single_V71_ECC_development >"$status"
python -u src/hong2021_v71_seal.py \
  --program "$program" --repo "$repo" \
  --preflight "$preflight" --preflight-sha256 "$preflight_sha" \
  --development-decision "$development/development_decision.json" \
  --out "$sealed" >"$sequence/seal.log" 2>&1
if [[ $(jq -r '.development_pass' "$development/development_decision.json") == true ]]; then
  printf "%s\n" complete_V71_development_pass_waiting_explicit_EAGLE_approval >"$status"
else
  printf "%s\n" complete_V71_development_failure_independent_gate_locked >"$status"
fi
trap - EXIT
