#!/usr/bin/env bash
# User-space rendering libraries; no sudo or global Python changes.
set -euo pipefail
cd /share/public-local/caoxuanhe/neon-arena-foundation
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_GL=osmesa
export LD_LIBRARY_PATH="$PWD/native/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=2
export PYTHONUNBUFFERED=1
exec .venv/bin/python "$@"
