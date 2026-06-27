#!/bin/bash
#SBATCH --job-name=rubble-citygs-coarse
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/work/pi_rsitaram_umass_edu/tungi/lctvgs/results/rubble_citygs_coarse/slurm_%j.log
#SBATCH --error=/work/pi_rsitaram_umass_edu/tungi/lctvgs/results/rubble_citygs_coarse/slurm_%j.log

set -e

WORK_DIR="/work/pi_rsitaram_umass_edu/tungi"
CONDA_ENV="$WORK_DIR/conda/envs/gsplat"
CUDA_DIR="$WORK_DIR/cuda-13.0"
EXAMPLES="$WORK_DIR/lctvgs/gsplat/examples"
DATA_DIR="$WORK_DIR/datasets/rubble"
RESULT_DIR="$WORK_DIR/lctvgs/results/rubble_citygs_coarse"

unset CC CXX
export CUDAHOSTCXX="$CONDA_ENV/bin/g++"
export CUDA_HOME="$CUDA_DIR"
export PATH="$CONDA_ENV/bin:$CUDA_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_ENV/lib:$CUDA_DIR/lib64:$LD_LIBRARY_PATH"
export PYTHONPATH="$WORK_DIR/lctvgs/gsplat:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$RESULT_DIR"

echo "=== Rubble CityGS coarse training started: $(date) ==="
echo "Host: $(hostname), GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

cd "$EXAMPLES"

# CityGaussian coarse model (global prior) for Rubble
#
# This implements Phase 1 of CityGaussian (ECCV 2024):
#   - Train vanilla 3DGS on the full scene for 30k steps
#   - DefaultStrategy, densification 500→15000 (identical to CityGS rubble_coarse.yaml)
#   - LR: position=0.00016, scaling=0.005 (matching rubble_coarse.yaml)
#   - The coarse model serves as global prior for block finetuning
#
# Dataset notes:
#   - Mega-NeRF format, converted to COLMAP (empty points3D.txt → random init)
#   - scene_scale ≈ 0.856 in normalized coords (= 150m metric / pose_scale_factor)
#   - position LR 0.00016 is passed as means_lr; gsplat multiplies by scene_scale
#     which approximately preserves the same normalized-space gradient step as
#     CityGS (position_lr_init * cameras_extent in metric space).
#
# Paper target (3DGS†, 60k): Rubble PSNR 25.47, SSIM 0.777, LPIPS 0.277
# CityGS (no LoD):            Rubble PSNR 25.77, SSIM 0.813, LPIPS 0.228

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
    --means-lr 0.012 \
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
