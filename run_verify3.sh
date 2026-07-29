#!/bin/bash
cd /home/kjhan/BACKUP/CF4
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.92 CUDA_VISIBLE_DEVICES=0 JAX_ENABLE_X64=0
PY=/home/kjhan/miniconda3/envs/circle/bin/python
$PY -u src/cf4_verify3.py > logs/verify3.log 2>&1
echo VERIFY3_DONE >> logs/verify3.log
