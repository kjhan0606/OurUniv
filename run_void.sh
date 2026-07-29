#!/bin/bash
cd /home/kjhan/BACKUP/CF4
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.92 CUDA_VISIBLE_DEVICES=0 JAX_ENABLE_X64=0
PY=/home/kjhan/miniconda3/envs/circle/bin/python
$PY -u src/cf4_void_verify.py > logs/void_verify.log 2>&1
echo VOID_DONE >> logs/void_verify.log
