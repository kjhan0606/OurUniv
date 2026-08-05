#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname | tr '[:upper:]' '[:lower:]')" != lageunha ]]; then
    echo "This HOP run must execute on Lageunha." >&2
    exit 2
fi

snapshot=/gpfs/kjhan/CF4/ramses/lg_p3429_s5108_l12_l19_z0_v1/output_00008
work=/gpfs/kjhan/CF4/recon/lg_p3429_s5108_z0_gate_v1/hop_work
hop_dir=/home/kjhan/BACKUP/lagRamses-de-nonstd/utils/f90/hop_ramses
hop="$hop_dir/hop"
regroup="$hop_dir/regroup"
prefix="$snapshot/part_00008.out"

for exe in "$hop" "$regroup"; do
    if [[ ! -x "$exe" ]]; then
        echo "Missing executable: $exe" >&2
        exit 2
    fi
done
if [[ ! -r "${prefix}00001" ]]; then
    echo "Missing z=0 RAMSES particle snapshot: ${prefix}00001" >&2
    exit 2
fi

mkdir -p "$work"
cd "$work"
rm -f HOP_COMPLETE HOP_FAILED

record_failure() {
    local rc=$?
    if (( rc != 0 )); then
        printf '%s HOP pipeline failed with exit code %d\n' \
            "$(date --iso-8601=seconds)" "$rc" >HOP_FAILED
    fi
}
trap record_failure EXIT

echo "$(date --iso-8601=seconds) HOP start host=$(hostname)" | tee runner.log
sha256sum "$hop" "$regroup" | tee -a runner.log

if [[ ! -s hop00010.den || ! -s hop00010.hop || ! -s hop00010.gbound ]]; then
    if compgen -G 'hop00010.*' >/dev/null; then
        echo "Partial hop00010 outputs already exist; refusing to overwrite." | tee -a runner.log >&2
        touch HOP_FAILED
        exit 3
    fi
    echo "$(date --iso-8601=seconds) computing HOP densities and raw peaks" | tee -a runner.log
    "$hop" -in "$prefix" -p 1. -o hop00010 >hop.log 2>&1
fi

if [[ ! -s grp00010.tag ]]; then
    echo "$(date --iso-8601=seconds) standard regroup douter=80 dsaddle=200 dpeak=240" \
        | tee -a runner.log
    "$regroup" -root hop00010 -douter 80. -dsaddle 200. -dpeak 240. \
        -f77 -o grp00010 >regroup.log 2>&1
fi

if [[ ! -s peaks00010.tag ]]; then
    echo "$(date --iso-8601=seconds) unmerged-peak regroup dsaddle=1e30" \
        | tee -a runner.log
    "$regroup" -root hop00010 -douter 80. -dsaddle 1e30 -dpeak 240. \
        -f77 -o peaks00010 >regroup_peaks.log 2>&1
fi

{
    echo "$(date --iso-8601=seconds) HOP products complete"
    stat -c '%n %s bytes' hop00010.den hop00010.hop hop00010.gbound \
        grp00010.tag peaks00010.tag
} | tee -a runner.log
cp -f runner.log HOP_COMPLETE
trap - EXIT
