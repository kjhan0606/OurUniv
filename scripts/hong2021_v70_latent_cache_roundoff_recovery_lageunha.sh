#!/usr/bin/env bash
set -euo pipefail
repo=/home/kjhan/BACKUP/CF4
tng=/gpfs/kjhan/IllustrisTNG/TNG100-1
sequence=$tng/evaluation/tng100_simba_swift_v70_latent_spatial_sequence
derived=$tng/derived/hong2021_v70
attempt=$sequence/cache_attempt1_strict_CDF_range
partial=$derived/train_latent.h5.partial
cd "$repo"
[[ $(hostname -s | tr '[:upper:]' '[:lower:]') == lageunha ]] || exit 1
[[ -z $(git status --porcelain) ]] || exit 1
[[ ! -e $attempt ]] || exit 1
[[ $(sha256sum "$sequence/cache.log" | awk '{print $1}') == a507751e4f16b78a2cc31560be6aa80f04580b7853b0a53927b6df1992b41a08 ]] || exit 1
[[ $(sha256sum "$sequence/cache_pytest.log" | awk '{print $1}') == a8d40a9e7c31d10350815c553acea1eaeac2f738c525a8caefac9351dbf790a7 ]] || exit 1
[[ $(sha256sum "$sequence/status" | awk '{print $1}') == 14e61307ae323d657ce95e2f7c5af60a0f47310c3c0450eaba47407c61c4c822 ]] || exit 1
[[ $(sha256sum "$partial" | awk '{print $1}') == b8700f0fd1a5a3f3586350c0832dd505e27e9c41b065376d6a00f955e1bddd5b ]] || exit 1
mkdir "$attempt"
mv "$sequence/cache.log" "$attempt/cache.log"
mv "$sequence/cache_pytest.log" "$attempt/pytest.log"
mv "$sequence/status" "$attempt/status"
mv "$partial" "$attempt/train_latent.h5.partial"
printf "%s\n" "TNG100 train object 7 contained one finite float32 mixture-CDF value 1.000000119; all inputs and V63 parameters were finite. Project only CDF roundoff within 5e-7 before the already frozen rank clamp; reject larger excursions and all nonfinite values." >"$attempt/reason"
exec bash scripts/hong2021_v70_latent_cache_lageunha.sh
