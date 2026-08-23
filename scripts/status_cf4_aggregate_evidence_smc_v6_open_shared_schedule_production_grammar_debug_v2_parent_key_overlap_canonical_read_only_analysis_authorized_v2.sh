#!/bin/bash
set -eu

exec env -i \
    PATH=/usr/bin:/bin \
    PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 \
    PYTHONPATH=/home/kjhan/BACKUP/CF4/src \
    OURUNIV_LIFECYCLE_MODE=receipt_status_only \
    /home/kjhan/miniconda3/envs/circle/bin/python3.11 -P -m \
    cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_grammar_debug_v2_parent_key_overlap_canonical_read_only_analysis_authorized_v2
