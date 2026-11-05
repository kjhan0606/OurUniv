#!/usr/bin/env bash
set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
tng=/gpfs/kjhan/IllustrisTNG/TNG100-1
program=$repo/config/hong2021_v37_query_aligned_copula_program.json
descriptor=$tng/derived/hong2021_v37/query_alignment_descriptor.json
sequence=$tng/evaluation/tng100_simba_swift_v37_query_alignment_sequence
preflight=$sequence/preflight.json
evaluation=$tng/evaluation/tng100_simba_swift_v37_e19_query_aligned_copula
decision=$evaluation/development_decision.json

cd "$repo"
export PYTHONPATH=$repo/src
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16

if [[ $(hostname -s | tr '[:upper:]' '[:lower:]') != lageunha ]]; then
    echo "V37 query-aligned transport is frozen on Lageunha" >&2
    exit 1
fi
[[ -z $(git status --porcelain) ]] || {
    echo "V37 requires a clean committed worktree" >&2
    exit 1
}
for path in "$descriptor" "$preflight" "$evaluation"; do
    [[ ! -e $path ]] || {
        echo "V37 refuses pre-existing output: $path" >&2
        exit 1
    }
done

mkdir -p "${descriptor%/*}" "$sequence"
pytest -q >"$sequence/pytest.log" 2>&1
python -u src/hong2021_v37_query_alignment.py fit \
    --program "$program" --repo "$repo" --out "$descriptor" \
    >"$sequence/fit.log" 2>&1
descriptor_sha=$(sha256sum "$descriptor" | awk '{print $1}')
python -u src/hong2021_v37_query_alignment.py preflight \
    --program "$program" --repo "$repo" \
    --descriptor "$descriptor" --descriptor-sha256 "$descriptor_sha" \
    --out "$preflight" >"$sequence/preflight.log" 2>&1
preflight_sha=$(sha256sum "$preflight" | awk '{print $1}')
python -u src/hong2021_v37_query_alignment.py sample \
    --program "$program" --repo "$repo" \
    --descriptor "$descriptor" --descriptor-sha256 "$descriptor_sha" \
    --preflight "$preflight" --preflight-sha256 "$preflight_sha" \
    --out "$evaluation" >"$sequence/sample.log" 2>&1

for arm in aligned shuffled_query_control; do
    for domain in tng simba_dev swift_dev; do
        root=$evaluation/$arm/development_candidate/$domain
        python -u src/hong2021_residual_evaluate.py \
            --candidate "edm=$root/ensemble16.h5" \
            --out "$root/ensemble_evaluation" --voxel-mpc-h 0.3125 \
            >"$root/evaluate.log" 2>&1
    done
done

python -u src/hong2021_v37_development_gate.py \
    --root "$evaluation" --program "$program" --repo "$repo" \
    --out "$decision" >"$evaluation/development_decision.log" 2>&1
