#!/usr/bin/env bash
set -euo pipefail
repo=/home/kjhan/BACKUP/CF4
tng=/gpfs/kjhan/IllustrisTNG/TNG100-1
program=$repo/config/hong2021_v70_latent_spatial_score_model_program.json
program_sha=79f1b5fe1462664b9b7a237bd82a821e205f3901603d64801a01b328c43f7e42
record=$repo/config/hong2021_v70_latent_cache_result_record.json
record_sha=3419206ce239546d7a2742ead01f20c9e6495c311dda0e4b82da6944a799ef76
cache=$tng/derived/hong2021_v70/train_latent.h5
cache_sha=0ddc9a592bc0eb1ab08d11ce71a5da1864b1fedb241663b2cc9f309094943ad3
cache_report=$tng/derived/hong2021_v70/train_latent.json
cache_report_sha=97bf049e8d5d2d0467a873aa2bf5b02a1939577c35e606be7c5b33899476769d
out=$tng/training/tng100_simba_swift_v70_latent_spatial
sequence=$tng/evaluation/tng100_simba_swift_v70_latent_spatial_sequence
status=$sequence/status
cd "$repo"
export PYTHONPATH=$repo/src CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[[ $(hostname -s | tr '[:upper:]' '[:lower:]') == lageunha ]] || exit 1
[[ -z $(git status --porcelain) ]] || exit 1
[[ $(sha256sum "$program" | awk '{print $1}') == "$program_sha" ]] || exit 1
[[ $(sha256sum "$record" | awk '{print $1}') == "$record_sha" ]] || exit 1
[[ $(sha256sum "$cache" | awk '{print $1}') == "$cache_sha" ]] || exit 1
[[ $(sha256sum "$cache_report" | awk '{print $1}') == "$cache_report_sha" ]] || exit 1
[[ ! -e $out ]] || exit 1
record_failure() {
  code=$?
  current=$(cat "$status" 2>/dev/null || true)
  [[ $code -eq 0 ]] || printf "failed_V70_fixed_training exit=%s previous=%s\n" "$code" "$current" >"$status"
}
trap record_failure EXIT
printf "%s\n" testing_before_fixed_training >"$status"
pytest -q >"$sequence/train_pytest.log" 2>&1
printf "%s\n" training_fixed_V70_step_0_of_30000 >"$status"
python -u src/hong2021_v70_train.py \
  --program "$program" --repo "$repo" \
  --cache "$cache" --cache-sha256 "$cache_sha" \
  --cache-report "$cache_report" --cache-report-sha256 "$cache_report_sha" \
  --out "$out" >"$sequence/train.log" 2>&1
printf "%s\n" complete_V70_fixed_training_pending_train_only_gate >"$status"
trap - EXIT
