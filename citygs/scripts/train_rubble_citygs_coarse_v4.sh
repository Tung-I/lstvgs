#!/bin/bash
#SBATCH --job-name=rubble-citygs-coarse-v4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/work/pi_rsitaram_umass_edu/tungi/lctvgs/results/rubble_citygs_coarse_v4/slurm_%j.log
#SBATCH --error=/work/pi_rsitaram_umass_edu/tungi/lctvgs/results/rubble_citygs_coarse_v4/slurm_%j.log

set -e

WORK_DIR="/work/pi_rsitaram_umass_edu/tungi"
CONDA_ENV="$WORK_DIR/conda/envs/gsplat"
CUDA_DIR="$WORK_DIR/cuda-13.0"
EXAMPLES="$WORK_DIR/lctvgs/gsplat/examples"
DATA_DIR="$WORK_DIR/datasets/rubble"
RESULT_DIR="$WORK_DIR/lctvgs/results/rubble_citygs_coarse_v4"

unset CC CXX
export CUDAHOSTCXX="$CONDA_ENV/bin/g++"
export CUDA_HOME="$CUDA_DIR"
export PATH="$CONDA_ENV/bin:$CUDA_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_ENV/lib:$CUDA_DIR/lib64:$LD_LIBRARY_PATH"
export PYTHONPATH="$WORK_DIR/lctvgs/gsplat:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$RESULT_DIR"

echo "=== Rubble CityGS coarse v4 training started: $(date) ==="
echo "Host: $(hostname), GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

cd "$EXAMPLES"

# v4: Ground-plane-aware initialization.
# Analysis of rubble dataset: cameras are at Y~0 in normalized space, looking in +Y direction.
# Scene (rubble ground) is at Y~0.983 (mean cam_dist * Y_viewdir = 0.994 * 0.989 ≈ 0.983).
# v3 (init_extent=1.0) still placed Gaussians in Y∈[-1.993,1.993]; only ~25% near Y≈0.983.
# v4 constrains Y to [0.983 - 0.5, 0.983 + 0.5] = [0.48, 1.48] → ALL Gaussians near scene.
# X, Z remain at ±init_extent*scene_scale = ±1.993.
# means_lr=0.012: same as v3 (empirically better than 0.00016).
# scales_lr=0.005: matches CityGaussian rubble.

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
    --init-y-center 0.983 \
    --init-y-spread 0.5 \
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

echo "=== Coarse v4 training finished: $(date) ==="
