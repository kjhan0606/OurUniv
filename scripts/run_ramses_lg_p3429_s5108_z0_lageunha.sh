#!/usr/bin/env bash
set -euo pipefail

expected_host="lageunha"
actual_host="$(hostname | tr '[:upper:]' '[:lower:]')"
if [[ "$actual_host" != "$expected_host" ]]; then
    echo "This pilot must run on Lageunha, not $(hostname)." >&2
    exit 2
fi

repo=/home/kjhan/BACKUP/CF4
binary=/home/kjhan/BACKUP/lagRamses-de-nonstd/build_lb_minimax/ramses_lb_minimax3d
namelist="$repo/config/ramses_lg_p3429_s5108_pilot_z0_v1.nml"
run_dir=/gpfs/kjhan/CF4/ramses/lg_p3429_s5108_l12_l19_z0_v1

if [[ ! -x "$binary" || ! -r "$namelist" ]]; then
    echo "Missing RAMSES binary or namelist." >&2
    exit 2
fi
if pgrep -f '/zoom_run_sidm[3]/ramses_zoom3d' >/dev/null; then
    echo "SIDM3 still occupies Lageunha cores 0-31; refusing to overlap it." >&2
    exit 4
fi

mkdir -p "$run_dir"
if compgen -G "$run_dir/output_*" >/dev/null || [[ -e "$run_dir/run.log" ]]; then
    echo "$run_dir already contains a run; refusing to overwrite it." >&2
    exit 3
fi

cd "$run_dir"
export OMP_NUM_THREADS=2
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export I_MPI_PIN=1
export I_MPI_PIN_DOMAIN=omp
export I_MPI_PIN_PROCESSOR_LIST=0-31

echo "host=$(hostname) ranks=16 threads_per_rank=$OMP_NUM_THREADS cpus=0-31"
echo "binary=$binary"
sha256sum "$binary" "$namelist"
exec mpirun -np 16 "$binary" "$namelist" 2>&1 | tee run.log
