#!/usr/bin/env bash
#SBATCH --job-name=cf4_lg_s40274_ic
#SBATCH --partition=a100_pcie,h200,h100,a100
#SBATCH --exclude=syn06
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=/gpfs/kjhan/CF4/logs/%x_%j.out
#SBATCH --error=/gpfs/kjhan/CF4/logs/%x_%j.err
set -Eeuo pipefail
cd /home/kjhan/BACKUP/CF4
export JAX_ENABLE_X64=True
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
readonly work=/gpfs/kjhan/CF4/recon/linear_cr/parent_b3292_beta025_lg_selected_pair_v3
readonly field="$work/passing_field_000.npz"
readonly grafic="$work/grafic_l9"
readonly transfer="$work/transfer_l9.npz"
if [[ -e "$grafic" || -e "$transfer" ]]; then
  echo "refusing to overwrite existing GRAFIC or transfer artifact" >&2
  exit 3
fi
/home/kjhan/miniconda3/envs/circle/bin/python3.11 src/cf4_export_grafic.py \
  --s-npz "$field" --key s_conditioned --N 512 --spacing 0.75 \
  --Om 0.31 --h 0.746 --A-s-1e9 1.63 --astart 0.02 --out "$grafic"
/home/kjhan/miniconda3/envs/circle/bin/python3.11 src/cf4_measure_transfer.py \
  --deltab "$grafic/ic_deltab" --sfield "$field" --skey s_conditioned \
  --box-hmpc 384 --out "$transfer"
sha256sum "$field" "$grafic/ic_deltab" "$grafic/ic_velcx" \
  "$grafic/ic_velcy" "$grafic/ic_velcz" "$transfer" > "$work/grafic_l9_SHA256SUMS"
