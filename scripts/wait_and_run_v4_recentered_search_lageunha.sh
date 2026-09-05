#!/usr/bin/env bash
set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
root=/gpfs/kjhan/CF4/recon/linear_cr
proposal_dir=$root/v3_bgc_lg_peak_proposals_v4_recentered_search
projection_dir=$root/v3_bgc_lg_peak_parent_projections_v4_recentered_search
p1_dir=$root/v3_bgc_lg_peak_p1_v4_recentered_search
p2_dir=$root/v3_bgc_lg_peak_p2_v4_recentered_search
manifest=$proposal_dir/lg_peak_proposals_manifest.json
projection_manifest=$projection_dir/parent_projection_manifest.json
p1_result=$p1_dir/p1_result.json
p2_config=$p2_dir/p2_targets_frozen.json
python=/home/kjhan/miniconda3/envs/circle/bin/python
log=$root/v3_bgc_lg_peak_v4_recentered_chain.log

mkdir -p "$p1_dir" "$p2_dir"
exec > >(tee -a "$log") 2>&1
echo "[chain] watcher started $(date -Is)"

while [[ ! -s "$manifest" || ! -s "$projection_manifest" ]]; do
    if ! tmux has-session -t cf4_lg_v4_search 2>/dev/null; then
        echo "[chain] proposal session ended without complete manifests"
        exit 1
    fi
    sleep 30
done

echo "[chain] proposal manifests ready $(date -Is)"
cd "$repo"
export JAX_ENABLE_X64=True
export XLA_PYTHON_CLIENT_PREALLOCATE=false

if [[ ! -s "$p1_result" ]]; then
    "$python" src/cf4_parent_p1.py \
        --manifest "$projection_manifest" \
        --config config/p1_targets_v2_observer.json \
        --outdir "$p1_dir"
else
    echo "[chain] reusing completed P1 result $p1_result"
fi

if [[ ! -s "$p2_config" ]]; then
    "$python" src/cf4_prepare_p2_recentered_config.py \
        --proposal-manifest "$manifest" \
        --conditioned-p1-result "$p1_result" \
        --outdir "$p2_dir" \
        --out "$p2_config"
else
    echo "[chain] reusing frozen P2 config $p2_config"
fi

# N576 needs one contiguous device allocation.  Preallocating the Ada GPU is
# the validated path used by the earlier successful P2 batches; disabling it
# fragments the pool and can fail while exporting the first particle arrays.
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.98
"$python" src/cf4_p2_screen.py \
    --p1-result "$root/v3_bgc_p1_observer_extension_v3/p1_result.json" \
    --config "$p2_config" \
    --outdir "$p2_dir"

"$python" src/cf4_p2_recenter_p1_preview.py \
    --p2-result "$p2_dir/p2_screen_result.json" \
    --conditioned-p1-result "$p1_result" \
    --p1-config config/p1_targets_v2_observer.json \
    --halo-directory "$p2_dir" \
    --out "$p2_dir/recentered_p1_preview.json"

touch "$p2_dir/RESEARCH_GATE_COMPLETE"
echo "[chain] recentered P1 gate complete $(date -Is)"
