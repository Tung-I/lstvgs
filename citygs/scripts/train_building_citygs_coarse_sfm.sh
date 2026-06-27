#!/bin/bash
#SBATCH --job-name=building-coarse-sfm
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/work/pi_rsitaram_umass_edu/tungi/lctvgs/results/building_citygs_coarse_sfm/slurm_%j.log
#SBATCH --error=/work/pi_rsitaram_umass_edu/tungi/lctvgs/results/building_citygs_coarse_sfm/slurm_%j.log

set -e

WORK_DIR="/work/pi_rsitaram_umass_edu/tungi"
CONDA_ENV="$WORK_DIR/conda/envs/gsplat"
CUDA_DIR="$WORK_DIR/cuda-13.0"
EXAMPLES="$WORK_DIR/lctvgs/gsplat/examples"
SFM_DATA_DIR="$WORK_DIR/lctvgs/datasets/building_sfm"
RESULT_DIR="$WORK_DIR/lctvgs/results/building_citygs_coarse_sfm"

unset CC CXX
export CUDAHOSTCXX="$CONDA_ENV/bin/g++"
export CUDA_HOME="$CUDA_DIR"
export PATH="$CONDA_ENV/bin:$CUDA_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_ENV/lib:$CUDA_DIR/lib64:$LD_LIBRARY_PATH"
export PYTHONPATH="$WORK_DIR/lctvgs/gsplat:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$RESULT_DIR"

echo "=== Building CityGS coarse (SFM init) started: $(date) ==="
echo "Host: $(hostname), GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

N_POINTS=$(grep -c "^[0-9]" "$SFM_DATA_DIR/sparse/0/points3D.txt" 2>/dev/null || echo 0)
echo "SFM points available: $N_POINTS"

if [ "$N_POINTS" -lt 1000 ]; then
    echo "ERROR: Insufficient SFM points ($N_POINTS). Run colmap_triangulate_building.sh first."
    exit 1
fi

cd "$EXAMPLES"
echo ""
echo "=== Coarse training 30k steps with SFM init ==="

python simple_trainer.py default \
    --disable-viewer \
    --disable-video \
    --data-dir "$SFM_DATA_DIR" \
    --data-factor 4 \
    --data-type colmap \
    --result-dir "$RESULT_DIR" \
    --test-every 8 \
    --normalize-world-space \
    --max-steps 30000 \
    --eval-steps 7000 30000 \
    --save-steps 7000 30000 \
    --save-ply \
    --ply-steps 30000 \
    --batch-size 1 \
    --init-type sfm \
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
    --strategy.refine-stop-iter 25000 \
    --strategy.refine-every 100 \
    --strategy.reset-every 3000 \
    --strategy.grow-grad2d 0.0002 \
    --strategy.grow-scale3d 0.01 \
    --strategy.prune-opa 0.005 \
    --packed \
    --lpips-net alex \
    --no-antialiased \
    --no-random-bkgd

echo ""
echo "=== Coarse training done: $(date) ==="
echo "Result dir: $RESULT_DIR"
