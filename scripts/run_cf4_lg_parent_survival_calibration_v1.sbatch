#!/usr/bin/env bash
#SBATCH --job-name=cf4_lg_survive
#SBATCH --partition=a100_pcie,a40,a100,h100,h200
#SBATCH --exclude=syn06
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --array=124,129,242,248%2
#SBATCH --no-requeue
#SBATCH --output=/gpfs/kjhan/CF4/logs/%x_%A_%a.out
#SBATCH --error=/gpfs/kjhan/CF4/logs/%x_%A_%a.err
set -Eeuo pipefail

root=/home/kjhan/BACKUP/CF4
index=${SLURM_ARRAY_TASK_ID:?}
seed=$((40101 + index))
tag=$(printf 'i%03d_s%d' "$index" "$seed")
search=/gpfs/kjhan/CF4/recon/linear_cr/parent_b3292_beta025_lg_pair_diagnostic_v3/result.json
base_config="$root/config/cf4_lg_highk_conditioning_b3292_v3_pair_diagnostic.json"
config_root=/gpfs/kjhan/CF4/recon/linear_cr/survival_calibration_v1_configs
selected_config="$config_root/${tag}.json"
work=/gpfs/kjhan/CF4/recon/linear_cr/parent_b3292_beta025_lg_survival_calibration_v1/$tag
grafic="$work/grafic_l9"
transfer="$work/transfer_l9.npz"
run_root=/gpfs/kjhan/CF4/ramses/cf4_lg_b3292_survival_calibration_v1/$tag
run="$run_root/job_${SLURM_JOB_ID}"
binary=/home/kjhan/BACKUP/lagRamses-de-nonstd/bin/ramses_final3d
template="$root/config/ramses_cf4_lg_parent_survival_screen_v1.nml.in"
galaxyfinder=/gpfs/kjhan/CF4/external/GalaxyFinder

test -x "$binary"
test -r "$template"
if [[ -e "$work" || -e "$selected_config" || -e "$run" ]]; then
  echo "refusing to overwrite candidate artifacts for $tag" >&2
  exit 3
fi
mkdir -p "$config_root" "$run_root"

cd "$root"
export JAX_ENABLE_X64=True XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
/home/kjhan/miniconda3/envs/circle/bin/python3.11 \
  scripts/cf4_select_and_reproduce_lg_environment_candidate.py \
  --search-result "$search" \
  --base-config "$base_config" \
  --search-index "$index" \
  --selected-config "$selected_config" \
  --output "$work"

/home/kjhan/miniconda3/envs/circle/bin/python3.11 src/cf4_export_grafic.py \
  --s-npz "$work/passing_field_000.npz" --key s_conditioned --N 512 --spacing 0.75 \
  --Om 0.31 --h 0.746 --A-s-1e9 1.63 --astart 0.02 --out "$grafic"
/home/kjhan/miniconda3/envs/circle/bin/python3.11 src/cf4_measure_transfer.py \
  --deltab "$grafic/ic_deltab" --sfield "$work/passing_field_000.npz" \
  --skey s_conditioned --box-hmpc 384 --out "$transfer"
sha256sum "$work/passing_field_000.npz" "$grafic/ic_deltab" \
  "$grafic/ic_velcx" "$grafic/ic_velcy" "$grafic/ic_velcz" "$transfer" \
  > "$work/grafic_l9_SHA256SUMS"

mkdir "$run"
namelist="$run/ramses.nml"
sed "s|@INITFILE@|$grafic|g" "$template" > "$namelist"
cd "$run"
printf 'job=%s array_job=%s index=%s seed=%s host=%s screening_only=true expected_outputs=1\n' \
  "$SLURM_JOB_ID" "$SLURM_ARRAY_JOB_ID" "$index" "$seed" "$(hostname)" | tee provenance.txt
sha256sum "$binary" "$namelist" "$grafic/ic_deltab" "$grafic/ic_velcx" \
  "$grafic/ic_velcy" "$grafic/ic_velcz" | tee -a provenance.txt
export OMP_NUM_THREADS=2 OMP_PROC_BIND=close OMP_PLACES=cores
export I_MPI_PIN=1 I_MPI_PIN_DOMAIN=omp
timeout --signal=TERM --kill-after=30s 3300s \
  mpirun -np 32 "$binary" "$namelist" 2>&1 | tee ramses.log
grep -q 'Run completed' ramses.log
grep -q 'FFTW3 CPU direct Poisson solver enabled' ramses.log
! grep -Eiq 'fatal|segmentation|out of memory|MPI_ABORT|Fine multigrid Poisson failed to converge' ramses.log
test -s output_00001/COMPLETE

mkdir -p FoF_Data FoF_Garbage
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export I_MPI_FABRICS=shm I_MPI_HYDRA_BOOTSTRAP=fork
unset I_MPI_PIN I_MPI_PIN_DOMAIN
export NEWDD_DM_ONLY=1 NEWDD_BOXLEN_CMPCH=384
srun --mpi=pmi2 --ntasks=16 --cpus-per-task=1 \
  "$galaxyfinder/NewDD/newdd.exe" 1 256
srun --mpi=pmi2 --ntasks=16 --cpus-per-task=1 \
  "$galaxyfinder/opFoF/opfof.exe" 1 256

fof="$run/FoF_Data/FoF.00001"
diagnostic=/gpfs/kjhan/CF4/diagnostics/cf4_lg_b3292_${tag}_survival_v1.json
members=/gpfs/kjhan/CF4/masks/cf4_lg_b3292_${tag}_survival_members_v1.npz
set +e
/home/kjhan/miniconda3/envs/circle/bin/python3.11 \
  "$root/scripts/cf4_galaxyfinder_parent_validation.py" \
  --catalog "$fof/FoF_halo_cat.00001" \
  --members "$fof/FoF_member_particle.00001" \
  --p1 "$root/config/p1_targets_v2_observer.json" \
  --p2 "$root/config/p2_lg_targets_v11_parent_b3292_beta025_l9.json" \
  --environment-policy standard-lg-isolation \
  --output "$diagnostic" --member-output "$members"
validation_status=$?
set -e
if (( validation_status != 0 && validation_status != 2 )); then
  exit "$validation_status"
fi
printf 'LG_SURVIVAL_CALIBRATION_COMPLETE index=%s seed=%s validator_status=%s result=%s\n' \
  "$index" "$seed" "$validation_status" "$diagnostic" | tee result.txt
