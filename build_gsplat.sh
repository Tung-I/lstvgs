#!/bin/bash
set -e

GSPLAT_SRC="/work/pi_rsitaram_umass_edu/tungi/lctvgs/gsplat"
CONDA_ENV="/work/pi_rsitaram_umass_edu/tungi/conda/envs/gsplat"
CUDA_DIR="/work/pi_rsitaram_umass_edu/tungi/cuda-13.0"
LOG="/work/pi_rsitaram_umass_edu/tungi/lctvgs/build_gsplat.log"

exec > >(tee -a "$LOG") 2>&1
echo "=== gsplat build started at $(date) ==="

# Compiler environment: use conda g++11, not spack gcc
unset CC CXX
export CUDAHOSTCXX="$CONDA_ENV/bin/g++"
export CUDA_HOME="$CUDA_DIR"
export PATH="$CONDA_ENV/bin:$CUDA_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_ENV/lib:$CUDA_DIR/lib64:$LD_LIBRARY_PATH"

echo "Python:      $($CONDA_ENV/bin/python --version)"
echo "g++:         $($CONDA_ENV/bin/g++ --version | head -1)"
echo "nvcc:        $($CUDA_DIR/bin/nvcc --version | grep release)"
echo "Torch:       $($CONDA_ENV/bin/python -c 'import torch; print(torch.__version__)')"
echo "Source dir:  $GSPLAT_SRC"

cd "$GSPLAT_SRC"

# AOT (ahead-of-time) build — compiles csrc.so against the current torch/CUDA
MAX_JOBS=8 "$CONDA_ENV/bin/pip" install -e . --no-build-isolation -v

echo "=== gsplat build finished at $(date) ==="

# Quick smoke test
"$CONDA_ENV/bin/python" -c "
import gsplat
print('gsplat version:', gsplat.__version__)
from gsplat import rasterization
print('rasterization import OK')
import torch
t = torch.zeros(1, device='cuda')
print('CUDA tensor OK:', t.device)
"
echo "=== smoke test passed ==="
