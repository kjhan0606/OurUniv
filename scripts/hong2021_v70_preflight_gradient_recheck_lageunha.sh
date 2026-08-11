#!/usr/bin/env bash
set -euo pipefail
repo=/home/kjhan/BACKUP/CF4
tng=/gpfs/kjhan/IllustrisTNG/TNG100-1
program=$repo/config/hong2021_v70_latent_spatial_score_model_program.json
program_sha=79f1b5fe1462664b9b7a237bd82a821e205f3901603d64801a01b328c43f7e42
sequence=$tng/evaluation/tng100_simba_swift_v70_latent_spatial_sequence
attempt=$sequence/attempt1_invalid_unscaled_amp_gradient
preflight=$sequence/preflight.json
status=$sequence/status
cd "$repo"
export PYTHONPATH=$repo/src CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[[ $(hostname -s | tr '[:upper:]' '[:lower:]') == lageunha ]] || exit 1
[[ -z $(git status --porcelain) ]] || exit 1
[[ $(sha256sum "$program" | awk '{print $1}') == "$program_sha" ]] || exit 1
[[ ! -e $attempt ]] || exit 1
[[ $(sha256sum "$preflight" | awk '{print $1}') == 3a4e699724dbe50527c19425a029ebdac429e73c80cc4d8e2b3377cb31340bcc ]] || exit 1
[[ $(sha256sum "$sequence/preflight.log" | awk '{print $1}') == 3a4e699724dbe50527c19425a029ebdac429e73c80cc4d8e2b3377cb31340bcc ]] || exit 1
[[ $(sha256sum "$sequence/pytest.log" | awk '{print $1}') == c1ac769c775c741ee7ff291880859f03d44f1bcd18f2e6d2b027176dcf210c18 ]] || exit 1
[[ $(sha256sum "$status" | awk '{print $1}') == c47a0a9ec8daf082361538280a94d3ee8f205770988d381db7b2152fce3556a4 ]] || exit 1
mkdir "$attempt"
mv "$preflight" "$attempt/preflight_false_positive.json"
mv "$sequence/preflight.log" "$attempt/preflight_false_positive.log"
mv "$sequence/pytest.log" "$attempt/pytest.log"
mv "$status" "$attempt/status"
printf "%s\n" "invalidated: unscaled AMP backward left only 194/8771649 scalar gradients nonzero; no cache or optimizer was created" >"$attempt/reason"
record_failure() {
  code=$?
  current=$(cat "$status" 2>/dev/null || true)
  [[ $code -eq 0 ]] || printf "failed_V70_gradient_recheck exit=%s previous=%s\n" "$code" "$current" >"$status"
}
trap record_failure EXIT
printf "%s\n" retesting_after_gradient_preflight_fix >"$status"
pytest -q >"$sequence/pytest.log" 2>&1
printf "%s\n" rechecking_all_parameter_gradient_tensors >"$status"
python -u src/hong2021_v70_preflight.py \
  --program "$program" --repo "$repo" --out "$preflight" >"$sequence/preflight.log" 2>&1
printf "%s\n" complete_V70_corrected_hard_preflight_pass >"$status"
trap - EXIT
