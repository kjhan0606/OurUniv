#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname | tr '[:upper:]' '[:lower:]')" != lageunha ]]; then
    echo "This promotion pipeline must run on Lageunha." >&2
    exit 2
fi

repo=/home/kjhan/BACKUP/CF4
python=/home/kjhan/miniconda3/envs/circle/bin/python
root=/gpfs/kjhan/CF4/recon/linear_cr
p2_dir=${CF4_P2_DIR:-$root/v3_bgc_lg_peak_p2_v4_recentered_search}
selection=$p2_dir/promotion_selection.json
p2_config=$p2_dir/p2_targets_frozen.json
p1_result=${CF4_CONDITIONED_P1_RESULT:-$root/v3_bgc_lg_peak_p1_v4_recentered_search/p1_result.json}
parent_p1=$root/v3_bgc_p1_observer_extension_v3/p1_result.json
binary=/home/kjhan/BACKUP/lagRamses-de-nonstd/build_lb_minimax/ramses_lb_minimax3d
hop_dir=/home/kjhan/BACKUP/lagRamses-de-nonstd/utils/f90/hop_ramses

read -r parent seed proposal < <("$python" - "$selection" <<'PY'
import json, sys
d=json.load(open(sys.argv[1])); s=d["selected"]
print(s["parent_seed"], s["small_scale_seed"], end=" ")
p2=json.load(open(d["p2_result"]))
for row in p2["results"]:
    if row["parent_seed"] == s["parent_seed"] and row["small_scale_seed"] == s["small_scale_seed"]:
        print(row["conditioned_proposal"]); break
else: raise SystemExit("selected proposal path not found")
PY
)

batch_label=${CF4_BATCH_LABEL:-v4}
work=$root/${batch_label}_p${parent}_s${seed}_auto
trace=$work/p${parent}_s${seed}_r5.npz
mask=$work/p${parent}_s${seed}_r5_l9_mask_pad6.npz
grafic=$work/grafic_n576
transfer=$grafic/transfer_p${parent}_s${seed}_n576.npz
zoom=$work/zoom_ic_l12_pad6
ic_link=/gpfs/kjhan/CF4/ramses/ic/lg${seed}
nml_dir=$work/ramses_config
preflight_nml=$nml_dir/ramses_lg_p3429_s${seed}_preflight.nml
z0_nml=$nml_dir/ramses_lg_p3429_s${seed}_z0.nml
preflight_dir=/gpfs/kjhan/CF4/ramses/lg_p3429_s${seed}_l12_l19_preflight_v1
z0_dir=/gpfs/kjhan/CF4/ramses/lg_p3429_s${seed}_l12_l19_z0_v1
gate_work=/gpfs/kjhan/CF4/recon/lg_p3429_s${seed}_z0_gate_v1
log=$work/automatic_promotion.log

mkdir -p "$work" "$grafic" "$nml_dir" "$gate_work"
exec > >(tee -a "$log") 2>&1
trap 'rc=$?; printf "%s state=failed exit=%s\n" "$(date -Is)" "$rc" >"$work/AUTO_PROMOTION_FAILED"; exit "$rc"' ERR
echo "$(date -Is) promotion start p${parent}/s${seed}"

export JAX_ENABLE_X64=True
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.98
"$python" "$repo/src/cf4_p2_trace.py" \
    --p2-result "$p2_dir/selected_p2_full_result.json" \
    --p1-result "$parent_p1" --config "$p2_config" \
    --parent-seed "$parent" --small-scale-seed "$seed" \
    --radius-mpc-h 5 --out "$trace"

"$python" "$repo/src/cf4_lagrangian_mask.py" \
    --input "$trace" --out "$mask" --box-mpc-h 384 --base-level 9 \
    --buffer-mpc-h 1.5 --subbox-pad-base-cells 6

export XLA_PYTHON_CLIENT_PREALLOCATE=false
"$python" "$repo/src/cf4_export_grafic.py" \
    --s-npz "$proposal" --key s_conditioned --N 576 \
    --spacing 0.6666666666666666 --astart 0.02 --out "$grafic"
"$python" "$repo/src/cf4_measure_transfer.py" \
    --deltab "$grafic/ic_deltab" --sfield "$proposal" \
    --skey s_conditioned --box-hmpc 384 --out "$transfer"
"$python" "$repo/src/cf4_zoom_ic2.py" \
    --parent "$grafic" --parent-level 9 --parent-grid-size 576 \
    --transfer "$transfer" --out "$zoom" --tier pilot \
    --mask-npz "$mask" --seed "$((seed + 3000))" --fft-workers 8

if [[ -L "$ic_link" ]]; then
    [[ "$(readlink -f "$ic_link")" == "$(readlink -f "$zoom")" ]] || {
        echo "Existing IC link points elsewhere: $ic_link" >&2; exit 3; }
elif [[ -e "$ic_link" ]]; then
    echo "Existing non-link IC path: $ic_link" >&2; exit 3
else
    ln -s "$zoom" "$ic_link"
fi
"$python" "$repo/src/cf4_prepare_ramses_candidate.py" \
    --seed "$seed" --ic-link "$ic_link" --outdir "$nml_dir"

mkdir -p "$preflight_dir"
if [[ -e "$preflight_dir/run.log" ]]; then
    echo "Preflight run already exists; refusing overwrite" >&2; exit 3
fi
cd "$preflight_dir"
export OMP_NUM_THREADS=2 OMP_PROC_BIND=close OMP_PLACES=cores
export I_MPI_PIN=1 I_MPI_PIN_DOMAIN=omp I_MPI_PIN_PROCESSOR_LIST=0-7
mpirun -np 4 "$binary" "$preflight_nml" 2>&1 | tee run.log
grep -q 'Run completed' run.log
! grep -Eiq 'fatal|segmentation|out of memory|outside.*initial|MPI_ABORT' run.log
! grep -Eq 'Morton \[step\].*mismatch=[[:space:]]*[1-9]' run.log
echo "$(date -Is) preflight passed" >PREFLIGHT_COMPLETE

while pgrep -f '/zoom_run_sidm[3]/ramses_zoom3d' >/dev/null; do sleep 60; done
mkdir -p "$z0_dir"
if [[ -e "$z0_dir/run.log" ]]; then
    echo "z=0 run already exists; refusing overwrite" >&2; exit 3
fi
cd "$z0_dir"
export I_MPI_PIN_PROCESSOR_LIST=0-31
mpirun -np 16 "$binary" "$z0_nml" 2>&1 | tee run.log
grep -q 'Run completed' run.log
! grep -Eiq 'fatal|segmentation|out of memory|MPI_ABORT' run.log

output=$(find "$z0_dir" -maxdepth 1 -type d -name 'output_*' | sort -V | tail -1)
number=${output##*_}
aexp=$(awk '$1 == "aexp" {print $3; exit}' "$output/info_${number}.txt")
awk -v a="$aexp" 'BEGIN {exit !(a >= 0.999999)}'

hop_work=$gate_work/hop_work
mkdir -p "$hop_work"
cd "$hop_work"
prefix=$output/part_${number}.out
"$hop_dir/hop" -in "$prefix" -p 1. -o hop00010 >hop.log 2>&1
"$hop_dir/regroup" -root hop00010 -douter 80. -dsaddle 200. -dpeak 240. \
    -f77 -o grp00010 >regroup.log 2>&1
"$hop_dir/regroup" -root hop00010 -douter 80. -dsaddle 1e30 -dpeak 240. \
    -f77 -o peaks00010 >regroup_peaks.log 2>&1

"$python" "$repo/src/cf4_zoom_z0_gate_v2.py" --output "$output" \
    --work "$gate_work" --hop-work "$hop_work" --p2-config "$p2_config" \
    --p2-result "$p2_dir/selected_p2_result.json" \
    --p2-halos "$p2_dir/halos_p${parent}_s${seed}.npz" --reuse-catalog
"$python" "$repo/src/cf4_zoom_recenter_p1.py" \
    --gate "$gate_work/gate_result_v2.json" --p1-result "$p1_result" \
    --p1-config "$repo/config/p1_targets_v2_observer.json" \
    --hop-catalog "$gate_work/hop_catalog_exact.npz" \
    --out "$gate_work/environment_result_v2.json"

"$python" - "$gate_work/gate_result_v2.json" "$work" <<'PY'
import json, pathlib, sys
d=json.load(open(sys.argv[1])); w=pathlib.Path(sys.argv[2])
ok=bool(d["verdict"]["overall"])
(w / ("AUTO_FINAL_PASS" if ok else "AUTO_FINAL_FAIL")).write_text(
    json.dumps(d["verdict"], sort_keys=True)+"\n")
print("[final]", json.dumps(d["verdict"], sort_keys=True))
PY
echo "$(date -Is) automatic promotion complete" >"$work/AUTO_PROMOTION_COMPLETE"
