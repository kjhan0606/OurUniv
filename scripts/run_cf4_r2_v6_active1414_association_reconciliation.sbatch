#!/usr/bin/env bash
#SBATCH --job-name=cf4_R2_assoc1414
#SBATCH --partition=h100
#SBATCH --gres=gpu:H100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/gpfs/kjhan/CF4/logs/%x_%j.out
#SBATCH --error=/gpfs/kjhan/CF4/logs/%x_%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?}" "${CF4_EXPECTED_COMMIT:?}"
module purge
cd /home/kjhan/BACKUP/CF4
HEAD_REV=$(git rev-parse --verify HEAD^{commit})
EXPECTED_REV=$(git rev-parse --verify "${CF4_EXPECTED_COMMIT}^{commit}")
if [[ "$HEAD_REV" != "$EXPECTED_REV" ]]; then
    echo "source revision mismatch: expected=$EXPECTED_REV actual=$HEAD_REV" >&2
    exit 2
fi
git diff --cached --quiet "$EXPECTED_REV" -- \
    scripts/cf4_r2_linked_fp_sparse_train.py \
    scripts/cf4_r2_v6_active1414_association_reconciliation.py \
    scripts/cf4_r2_v6_factor_ownership_ledger.py \
    scripts/run_cf4_r2_v6_active1414_association_reconciliation.sbatch \
    tests/test_cf4_r2_v6_active1414_association_reconciliation.py
git diff --quiet "$EXPECTED_REV" -- \
    scripts/cf4_r2_linked_fp_sparse_train.py \
    scripts/cf4_r2_v6_active1414_association_reconciliation.py \
    scripts/cf4_r2_v6_factor_ownership_ledger.py \
    scripts/run_cf4_r2_v6_active1414_association_reconciliation.sbatch \
    tests/test_cf4_r2_v6_active1414_association_reconciliation.py
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$PWD/src:$PWD/scripts${PYTHONPATH:+:$PYTHONPATH}"
PY=/home/kjhan/miniconda3/envs/circle/bin/python3.11
"$PY" tests/test_cf4_r2_v6_active1414_association_reconciliation.py
exec "$PY" scripts/cf4_r2_v6_active1414_association_reconciliation.py
