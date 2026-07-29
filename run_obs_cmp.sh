#!/bin/bash
cd /home/kjhan/BACKUP/CF4
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.92 CUDA_VISIBLE_DEVICES=0 JAX_ENABLE_X64=0
PY=/home/kjhan/miniconda3/envs/circle/bin/python
$PY -u src/cf4_obs_compare.py > logs/obs_cmp.log 2>&1
echo OBS_CMP_DONE >> logs/obs_cmp.log
