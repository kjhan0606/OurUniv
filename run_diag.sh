#!/bin/bash
cd /home/kjhan/BACKUP/CF4
export CUDA_VISIBLE_DEVICES=0 JAX_ENABLE_X64=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.92
PY=/home/kjhan/miniconda3/envs/circle/bin/python
# old winner was cr1 (cf4_map_cf4_real192_hr.npz) embed 22 -> did fixed embed_ic kill it?
$PY -u src/cf4_lg_screen3.py --recon recon/cf4_map_cf4_real192_hr.npz --cseed 1 --seed 22 \
   --tsv recon/diag_e22.tsv > logs/diag_e22.log 2>&1
echo DIAG_DONE >> logs/diag_e22.log
