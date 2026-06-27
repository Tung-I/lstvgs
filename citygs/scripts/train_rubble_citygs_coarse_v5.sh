#!/bin/bash
#SBATCH --job-name=rubble-citygs-coarse-v5
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/work/pi_rsitaram_umass_edu/tungi/lstvgs/results/rubble_citygs_coarse_v5/slurm_%j.log
#SBATCH --error=/work/pi_rsitaram_umass_edu/tungi/lstvgs/results/rubble_citygs_coarse_v5/slurm_%j.log

set -e

WORK_DIR="/work/pi_rsitaram_umass_edu/tungi"
CONDA_ENV="$WORK_DIR/conda/envs/gsplat"
CUDA_DIR="$WORK_DIR/cuda-13.0"
EXAMPLES="$WORK_DIR/lstvgs/gsplat/examples"
DATA_DIR="$WORK_DIR/datasets/rubble"
RESULT_DIR="$WORK_DIR/lstvgs/results/rubble_citygs_coarse_v5"
# Use v3 coarse PLY as starting point (981k well-distributed Gaussians)
INIT_PLY="$WORK_DIR/lstvgs/results/rubble_citygs_coarse_v3/ply/point_cloud_29999.ply"

unset CC CXX
export CUDAHOSTCXX="$CONDA_ENV/bin/g++"
export CUDA_HOME="$CUDA_DIR"
export PATH="$CONDA_ENV/bin:$CUDA_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_ENV/lib:$CUDA_DIR/lib64:$LD_LIBRARY_PATH"
export PYTHONPATH="$WORK_DIR/lstvgs/gsplat:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$RESULT_DIR"

echo "=== Rubble CityGS coarse v5 training started: $(date) ==="
echo "Host: $(hostname), GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

if [ ! -f "$INIT_PLY" ]; then
    echo "ERROR: v3 coarse PLY not found: $INIT_PLY"
    exit 1
fi
echo "Init PLY: $INIT_PLY"

cd "$EXAMPLES"

# v5: Continue training from v3 coarse PLY (981k Gaussians, 14.6 dB).
# Uses lower LR (fine-tuning mode) since Gaussians are already partially organized.
# Densification ON (500-15000) to further refine scene coverage.
# 30k additional steps with LR decaying from 0.003 to 0.00003 (10x smaller than coarse).

python simple_trainer.py default \
    --disable-viewer \
    --disable-video \
    --data-dir "$DATA_DIR" \
    --data-factor 4 \
    --data-type colmap \
    --result-dir "$RESULT_DIR" \
    --test-every 8 \
    --normalize-world-space \
    --max-steps 30000 \
    --eval-steps 7000 30000 \
    --save-steps 30000 \
    --save-ply \
    --ply-steps 30000 \
    --batch-size 1 \
    --init-type ply \
    --init-ply-path "$INIT_PLY" \
    --sh-degree 3 \
    --sh-degree-interval 1000 \
    --ssim-lambda 0.2 \
    --means-lr 0.003 \
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

echo "=== Coarse v5 training finished: $(date) ==="
