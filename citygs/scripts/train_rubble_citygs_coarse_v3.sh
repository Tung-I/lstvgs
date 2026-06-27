#!/bin/bash
#SBATCH --job-name=rubble-citygs-coarse-v3
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/work/pi_rsitaram_umass_edu/tungi/lctvgs/results/rubble_citygs_coarse_v3/slurm_%j.log
#SBATCH --error=/work/pi_rsitaram_umass_edu/tungi/lctvgs/results/rubble_citygs_coarse_v3/slurm_%j.log

set -e

WORK_DIR="/work/pi_rsitaram_umass_edu/tungi"
CONDA_ENV="$WORK_DIR/conda/envs/gsplat"
CUDA_DIR="$WORK_DIR/cuda-13.0"
EXAMPLES="$WORK_DIR/lctvgs/gsplat/examples"
DATA_DIR="$WORK_DIR/datasets/rubble"
RESULT_DIR="$WORK_DIR/lctvgs/results/rubble_citygs_coarse_v3"

unset CC CXX
export CUDAHOSTCXX="$CONDA_ENV/bin/g++"
export CUDA_HOME="$CUDA_DIR"
export PATH="$CONDA_ENV/bin:$CUDA_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_ENV/lib:$CUDA_DIR/lib64:$LD_LIBRARY_PATH"
export PYTHONPATH="$WORK_DIR/lctvgs/gsplat:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$RESULT_DIR"

echo "=== Rubble CityGS coarse v3 training started: $(date) ==="
echo "Host: $(hostname), GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

cd "$EXAMPLES"

# v3: Reduced init_extent from 3.0 to 1.0 to keep random init Gaussians
# inside the actual scene volume. With init_extent=3.0 only ~3k out of 200k
# Gaussians land near the scene (rest in empty space); with init_extent=1.0
# ~100k of 500k Gaussians are inside the camera orbit.
# means_lr=0.012 * scene_scale(1.993) ≈ 0.024 effective LR, matching CityGaussian.

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
    --init-num-pts 500000 \
    --init-extent 1.0 \
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
    --packed \
    --lpips-net alex \
    --no-antialiased \
    --no-random-bkgd

echo "=== Coarse v3 training finished: $(date) ==="
