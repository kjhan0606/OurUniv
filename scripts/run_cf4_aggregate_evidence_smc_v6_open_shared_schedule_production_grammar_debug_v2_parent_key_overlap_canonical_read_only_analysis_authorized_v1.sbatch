#!/bin/bash
#SBATCH --job-name=cf4-parent-overlap-v1
#SBATCH --partition=debug
#SBATCH --nodelist=grammar-debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@60
#SBATCH --kill-on-bad-exit=1
#SBATCH --chdir=/home/kjhan/BACKUP/CF4
#SBATCH --export=NONE
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -u
set -o pipefail

python_bin=/home/kjhan/miniconda3/envs/circle/bin/python3.11
module_name=cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_grammar_debug_v2_parent_key_overlap_canonical_read_only_analysis_authorized_v1
source_path=/home/kjhan/BACKUP/CF4/src/cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_grammar_debug_v2_parent_key_overlap_canonical_read_only_analysis_authorized_v1.py
child_pid=
supervised=0

if [ ! -f "$python_bin" ] || [ -L "$python_bin" ] || [ ! -x "$python_bin" ]; then
    exit 65
fi
printf '%s  %s\n' 9ee5fb16ef60eb6a53af53ae6bd300a5ac8c01d81a8c961e7cdf1497efee3ccc "$python_bin" | /usr/bin/sha256sum --check --status - || exit 65
if [ ! -f "$source_path" ] || [ -L "$source_path" ]; then
    exit 65
fi
printf '%s  %s\n' 15e6e8252fcaf0bfde8a70ce561996d503ddf0db7294681c3817d4c54d9c1842 "$source_path" | /usr/bin/sha256sum --check --status - || exit 65

unset BASH_ENV ENV OURUNIV_CHILD_EXIT_CLASS OURUNIV_LIFECYCLE_MODE
unset PYTHONHOME PYTHONSTARTUP PYTHONINSPECT PYTHONWARNINGS PYTHONBREAKPOINT
unset PYTHONCASEOK PYTHONEXECUTABLE PYTHONMALLOC PYTHONPLATLIBDIR
unset PYTHONPROFILEIMPORTTIME PYTHONTRACEMALLOC PYTHONUSERBASE
unset LD_PRELOAD LD_LIBRARY_PATH

run_supervisor() {
    child_class=$1
    supervised=1
    env -i \
        PATH=/usr/bin:/bin \
        CUDA_VISIBLE_DEVICES= \
        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
        PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 \
        PYTHONPATH=/home/kjhan/BACKUP/CF4/src \
        SLURM_JOB_NUM_NODES="${SLURM_JOB_NUM_NODES-}" \
        SLURM_NTASKS="${SLURM_NTASKS-}" \
        SLURM_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK-}" \
        SLURM_MEM_PER_NODE="${SLURM_MEM_PER_NODE-}" \
        SLURM_JOB_PARTITION="${SLURM_JOB_PARTITION-}" \
        SLURM_JOB_NAME="${SLURM_JOB_NAME-}" \
        OURUNIV_LIFECYCLE_MODE=receipt_supervisor_only \
        OURUNIV_CHILD_EXIT_CLASS="$child_class" \
        "$python_bin" -P -m "$module_name"
}

handle_signal() {
    trap - HUP INT TERM
    if [ -n "$child_pid" ]; then
        kill -TERM "$child_pid" 2>/dev/null || true
        wait "$child_pid" 2>/dev/null || true
    fi
    run_supervisor terminated_143 || true
    exit 143
}

trap handle_signal HUP INT TERM

env -i \
    PATH=/usr/bin:/bin \
    CUDA_VISIBLE_DEVICES= \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 \
    PYTHONPATH=/home/kjhan/BACKUP/CF4/src \
    SLURM_JOB_NUM_NODES="${SLURM_JOB_NUM_NODES-}" \
    SLURM_NTASKS="${SLURM_NTASKS-}" \
    SLURM_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK-}" \
    SLURM_MEM_PER_NODE="${SLURM_MEM_PER_NODE-}" \
    SLURM_JOB_PARTITION="${SLURM_JOB_PARTITION-}" \
    SLURM_JOB_NAME="${SLURM_JOB_NAME-}" \
    "$python_bin" -P -m "$module_name" &
child_pid=$!
wait "$child_pid"
child_rc=$?
child_pid=
trap - HUP INT TERM

case "$child_rc" in
    0) child_class=success_0 ;;
    124) child_class=timeout_124 ;;
    137) child_class=killed_137 ;;
    143) child_class=terminated_143 ;;
    *) child_class=other_nonzero ;;
esac

run_supervisor "$child_class"
supervisor_rc=$?
if [ "$supervised" -ne 1 ]; then
    exit 65
fi
exit "$supervisor_rc"
