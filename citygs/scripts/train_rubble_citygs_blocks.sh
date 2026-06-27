#!/bin/bash
#SBATCH --job-name=rubble-citygs-blocks
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/work/pi_rsitaram_umass_edu/tungi/lstvgs/results/rubble_citygs_blocks/slurm_%j.log
#SBATCH --error=/work/pi_rsitaram_umass_edu/tungi/lstvgs/results/rubble_citygs_blocks/slurm_%j.log

set -e

WORK_DIR="/work/pi_rsitaram_umass_edu/tungi"
CONDA_ENV="$WORK_DIR/conda/envs/gsplat"
CUDA_DIR="$WORK_DIR/cuda-13.0"
EXAMPLES="$WORK_DIR/lstvgs/gsplat/examples"
CITYGS="$WORK_DIR/lstvgs/citygs"
DATA_DIR="$WORK_DIR/datasets/rubble"
COARSE_DIR="$WORK_DIR/lstvgs/results/rubble_citygs_coarse"
BLOCK_DIR="$WORK_DIR/lstvgs/results/rubble_citygs_blocks"
MERGE_DIR="$WORK_DIR/lstvgs/results/rubble_citygs_merged"

unset CC CXX
export CUDAHOSTCXX="$CONDA_ENV/bin/g++"
export CUDA_HOME="$CUDA_DIR"
export PATH="$CONDA_ENV/bin:$CUDA_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_ENV/lib:$CUDA_DIR/lib64:$LD_LIBRARY_PATH"
export PYTHONPATH="$WORK_DIR/lstvgs/gsplat:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$BLOCK_DIR" "$MERGE_DIR"

echo "=== Rubble CityGS block training started: $(date) ==="
echo "Host: $(hostname), GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

# ── Find coarse PLY ────────────────────────────────────────────────────────────
COARSE_PLY="$COARSE_DIR/ply/point_cloud_29999.ply"
if [ ! -f "$COARSE_PLY" ]; then
    echo "ERROR: Coarse PLY not found: $COARSE_PLY"
    echo "Available PLYs:"
    ls "$COARSE_DIR/ply/" 2>/dev/null || echo "  (none)"
    exit 1
fi
echo "Using coarse PLY: $COARSE_PLY"

# ── Step 1: Partition cameras to blocks ────────────────────────────────────────
PARTITION_DIR="$COARSE_DIR/partition"
echo ""
echo "=== Step 1: Partitioning cameras ($(date)) ==="
cd "$WORK_DIR/lstvgs"
python citygs/partition_citygs.py \
    --ply-path "$COARSE_PLY" \
    --data-dir "$DATA_DIR" \
    --data-factor 4 \
    --test-every 8 \
    --block-dim 3 1 3 \
    --ssim-threshold 0.12 \
    --output-dir "$PARTITION_DIR" \
    --simple-selection 1.5

echo "Partition complete. Camera assignments saved to $PARTITION_DIR"

# ── Step 2: Prepare block data ────────────────────────────────────────────────
echo ""
echo "=== Step 2: Preparing block data ($(date)) ==="
python citygs/prepare_block_data.py \
    --data-dir "$DATA_DIR" \
    --data-factor 4 \
    --test-every 8 \
    --partition-dir "$PARTITION_DIR" \
    --output-dir "$BLOCK_DIR" \
    --block-dim 3 1 3

echo "Block data prepared."

# ── Step 3: Train each block ──────────────────────────────────────────────────
BLOCK_NUM=9  # 3x1x3 grid

# CityGaussian block finetuning hyperparameters for Rubble:
#   position_lr: 0.4× coarse = 0.000064
#   scaling_lr: 0.8× coarse = 0.004
#   densify: 500→15000 (same as coarse)

echo ""
echo "=== Step 3: Block finetuning ($(date)) ==="
cd "$EXAMPLES"

for BLOCK_ID in $(seq 0 $((BLOCK_NUM - 1))); do
    BLOCK_SUBDIR=$(printf "%03d" $BLOCK_ID)
    RESULT_DIR="$BLOCK_DIR/block_$BLOCK_SUBDIR"
    CAM_FILE="$RESULT_DIR/block_info.json"

    # Check if this block has cameras
    N_CAMS=$(python3 -c "import json; d=json.load(open('$CAM_FILE')); print(d['n_cameras'])" 2>/dev/null || echo "0")
    if [ "$N_CAMS" -lt 5 ]; then
        echo "  Block $BLOCK_ID: skipping (only $N_CAMS cameras)"
        continue
    fi

    echo ""
    echo "  Training block $BLOCK_ID ($N_CAMS cameras) → $RESULT_DIR"

    python simple_trainer.py default \
        --disable-viewer \
        --data-dir "$DATA_DIR" \
        --data-factor 4 \
        --data-type colmap \
        --result-dir "$RESULT_DIR" \
        --test-every 8 \
        --normalize-world-space \
        --cam-indices-file "$CAM_FILE" \
        --max-steps 30000 \
        --eval-steps 30000 \
        --save-steps 30000 \
        --save-ply \
        --ply-steps 30000 \
        --batch-size 1 \
        --init-type ply \
        --init-ply-path "$COARSE_PLY" \
        --sh-degree 3 \
        --sh-degree-interval 1000 \
        --ssim-lambda 0.2 \
        --means-lr 0.0048 \
        --scales-lr 0.004 \
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

    echo "  Block $BLOCK_ID done: $(date)"
done

# ── Step 4: Merge blocks ──────────────────────────────────────────────────────
echo ""
echo "=== Step 4: Merging blocks ($(date)) ==="
cd "$WORK_DIR/lstvgs"
MERGED_PLY="$MERGE_DIR/splat_merged.ply"
python citygs/merge_citygs.py \
    --block-results-dir "$BLOCK_DIR" \
    --block-dim 3 1 3 \
    --step 29999 \
    --output "$MERGED_PLY"

echo "Merged PLY: $MERGED_PLY"

# ── Step 5: Evaluate merged model ────────────────────────────────────────────
echo ""
echo "=== Step 5: Evaluation ($(date)) ==="
python citygs/eval_citygs.py \
    --ply-path "$MERGED_PLY" \
    --data-dir "$DATA_DIR" \
    --data-factor 4 \
    --test-every 8 \
    --output-dir "$MERGE_DIR/eval" \
    --save-images

echo ""
echo "=== All done: $(date) ==="
echo "Results:"
cat "$MERGE_DIR/eval/metrics.json"
