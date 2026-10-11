#!/usr/bin/env bash
set -euo pipefail

readonly repo=/home/kjhan/BACKUP/CF4
readonly python=/home/kjhan/miniconda3/envs/circle/bin/python3.11
readonly device_program="$repo/config/cf4_phasec_same_gpu_repeatability_diagnostic_v1.json"
readonly device_implementation="$repo/src/cf4_phasec_same_gpu_repeatability_diagnostic.py"
readonly gate_program="$repo/config/cf4_phasec_scan_generator_gate_v2.json"
readonly gate_implementation="$repo/src/cf4_phasec_scan_generator_gate_v2.py"
readonly device_program_sha=46aba82ea404d13934abe5a1115dca01167a0ffaca322bf7a0e8ac4cb48ee911
readonly device_implementation_sha=79a5080bfb667f5b6294ae741f4246fc516a9b087d498aa4643cd4a7a8d0a8db
readonly gate_program_sha=07b1c0434b954a364f66d78ab6e6b8c63cfe5ba609909ad16b8b2fd897980cf7
readonly gate_implementation_sha=5552198373b2c1507d981c467de9b9a469e053a8af189e51aea20495bf3c8f4b
readonly gate_implementation_commit=b4f33d2a1f408c07ba746a8765d78096a6028e5c

[[ "$#" -eq 3 ]]
readonly slot="$1"
readonly output_root="$2"
readonly expected_commit="$3"
[[ "$slot" =~ ^[01]$ ]]
: "${SLURM_JOB_ID:?}" "${SLURM_STEP_ID:?}" "${SLURM_JOB_GPUS:?}"
[[ "$(hostname -s)" == syn06 ]]
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]
[[ "$(git -C "$repo" rev-parse HEAD)" == "$expected_commit" ]]
[[ "$(git -C "$repo" rev-parse '@{upstream}')" == "$expected_commit" ]]
[[ "$(sha256sum "$device_program" | awk '{print $1}')" == "$device_program_sha" ]]
[[ "$(sha256sum "$device_implementation" | awk '{print $1}')" == "$device_implementation_sha" ]]
[[ "$(sha256sum "$gate_program" | awk '{print $1}')" == "$gate_program_sha" ]]
[[ "$(sha256sum "$gate_implementation" | awk '{print $1}')" == "$gate_implementation_sha" ]]

readonly slot_root="$output_root/slot_$slot"
[[ ! -e "$slot_root" ]]
mkdir -m 700 "$slot_root"
"$python" -I -P "$device_implementation" capture-device \
  --program "$device_program" \
  --output-root "$slot_root"

for task_index in 1 6 2; do
  "$python" -I -P "$gate_implementation" run \
    --program "$gate_program" \
    --output-root "$slot_root" \
    --task-index "$task_index" \
    --implementation-commit "$gate_implementation_commit"
done

for task_index in 1 6 2; do
  seed=$((2026083000 + task_index))
  task_dir=$(printf '%s/seed_%02d_%d' "$slot_root" "$task_index" "$seed")
  "$python" -I -P "$gate_implementation" validate-task --directory "$task_dir"
done
