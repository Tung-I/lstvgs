#!/bin/bash
#SBATCH --job-name=building-citygs-coarse
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/work/pi_rsitaram_umass_edu/tungi/lstvgs/results/building_citygs_coarse/slurm_%j.log
#SBATCH --error=/work/pi_rsitaram_umass_edu/tungi/lstvgs/results/building_citygs_coarse/slurm_%j.log

set -e

WORK_DIR="/work/pi_rsitaram_umass_edu/tungi"
CONDA_ENV="$WORK_DIR/conda/envs/gsplat"
CUDA_DIR="$WORK_DIR/cuda-13.0"
EXAMPLES="$WORK_DIR/lstvgs/gsplat/examples"
DATA_DIR="$WORK_DIR/datasets/building"
RESULT_DIR="$WORK_DIR/lstvgs/results/building_citygs_coarse"

unset CC CXX
export CUDAHOSTCXX="$CONDA_ENV/bin/g++"
export CUDA_HOME="$CUDA_DIR"
export PATH="$CONDA_ENV/bin:$CUDA_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_ENV/lib:$CUDA_DIR/lib64:$LD_LIBRARY_PATH"
export PYTHONPATH="$WORK_DIR/lstvgs/gsplat:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$RESULT_DIR"

echo "=== Building CityGS coarse training started: $(date) ==="
echo "Host: $(hostname), GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

cd "$EXAMPLES"

# CityGaussian coarse model (global prior) for Building
#
# Matches building_coarse.yaml from CityGaussian V1-original:
#   - position_lr_init: 0.00008 (halved vs rubble, for wider scene)
#   - scaling_lr: 0.0025 (halved vs rubble)
#   - densify: 500→15000, every 100 steps
#   - opacity reset every 3000 steps
#
# Paper target (3DGS†, 60k): Building PSNR 20.46, SSIM 0.720, LPIPS 0.305
# CityGS (no LoD):            Building PSNR 21.55, SSIM 0.778, LPIPS 0.246

python simple_trainer.py default \
    --disable-viewer \
    --data-dir "$DATA_DIR" \
    --data-factor 4 \
    --data-type colmap \
    --result-dir "$RESULT_DIR" \
    --test-every 8 \
    --normalize-world-space \
    --max-steps 30000 \
    --eval-steps 7000 30000 \
    --save-steps 7000 30000 \
    --save-ply \
    --ply-steps 7000 30000 \
    --batch-size 1 \
    --init-type random \
    --init-num-pts 200000 \
    --init-extent 3.0 \
    --init-opa 0.1 \
    --init-scale 1.0 \
    --sh-degree 3 \
    --sh-degree-interval 1000 \
    --ssim-lambda 0.2 \
    --means-lr 0.011 \
    --scales-lr 0.005 \
    --opacities-lr 0.05 \
    --quats-lr 0.001 \
    --sh0-lr 0.0025 \
    --shN-lr 0.000125 \
    --strategy.refine-start-iter 500 \
    --strategy.refine-stop-iter 15000 \
    --strategy.refine-every 100 \
    --strategy.reset-every 3000 \
    --strategy.grow-grad2d 0.0002 \
    --strategy.grow-scale3d 0.01 \
    --strategy.prune-opa 0.005 \
    --strategy.verbose \
    --packed \
    --lpips-net alex \
    --no-antialiased \
    --no-random-bkgd

echo "=== Coarse training finished: $(date) ==="
