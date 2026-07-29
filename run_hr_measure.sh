#!/bin/bash
cd /home/kjhan/BACKUP/CF4
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.92 CUDA_VISIBLE_DEVICES=0 JAX_ENABLE_X64=0
PY=/home/kjhan/miniconda3/envs/circle/bin/python
$PY -u src/cf4_hr_measure.py --recon recon/cf4_map_cf4_real192_hr.npz --key s_out \
    --Nfine 576 --out recon/cf4_hr_web.png > logs/hr_measure.log 2>&1
echo HR_MEASURE_DONE >> logs/hr_measure.log
