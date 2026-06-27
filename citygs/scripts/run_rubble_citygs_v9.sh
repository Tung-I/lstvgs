#!/bin/bash
# CityGS Rubble v9: Full pipeline with proper SFM init (1.69M points)
# Run directly in interactive SSH session (L40S GPU already allocated)
# Usage: bash run_rubble_citygs_v9.sh
set -e

WORK_DIR="/work/pi_rsitaram_umass_edu/tungi"
CONDA_ENV="$WORK_DIR/conda/envs/gsplat"
CUDA_DIR="$WORK_DIR/cuda-13.0"
LCTVGS="$WORK_DIR/lstvgs"
EXAMPLES="$LCTVGS/gsplat/examples"
DATA_DIR="$WORK_DIR/datasets/rubble_citygs"

COARSE_DIR="$LCTVGS/results/rubble_citygs_coarse_v9"
BLOCK_DIR="$LCTVGS/results/rubble_citygs_blocks_v9"
MERGE_DIR="$LCTVGS/results/rubble_citygs_merged_v9"

unset CC CXX
export CUDAHOSTCXX="$CONDA_ENV/bin/g++"
export CUDA_HOME="$CUDA_DIR"
export PATH="$CONDA_ENV/bin:$CUDA_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_ENV/lib:$CUDA_DIR/lib64:$LD_LIBRARY_PATH"
export PYTHONPATH="$LCTVGS/gsplat:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$COARSE_DIR" "$BLOCK_DIR" "$MERGE_DIR"

echo "=== CityGS Rubble v9 started: $(date) ==="
echo "Host: $(hostname), GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Data: $DATA_DIR"
echo "SFM init: 1,694,315 points (from CityGS authors' COLMAP results)"

# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: Coarse model (full scene, SFM init, 30k steps)
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== Stage 1: Coarse training (30k steps, SFM init) === $(date)"
cd "$EXAMPLES"

python simple_trainer.py default \
    --disable-viewer \
    --data-dir "$DATA_DIR" \
    --data-factor 4 \
    --data-type colmap \
    --result-dir "$COARSE_DIR" \
    --test-every 83 \
    --normalize-world-space \
    --max-steps 30000 \
    --eval-steps 7000 30000 \
    --save-steps 7000 30000 \
    --save-ply \
    --ply-steps 7000 30000 \
    --batch-size 1 \
    --init-type sfm \
    --sh-degree 3 \
    --sh-degree-interval 1000 \
    --ssim-lambda 0.2 \
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

echo "=== Stage 1 done: $(date) ==="

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: Partition (3×1×3 blocks, SSIM-based assignment)
# ─────────────────────────────────────────────────────────────────────────────
COARSE_PLY="$COARSE_DIR/ply/point_cloud_29999.ply"
PARTITION_DIR="$COARSE_DIR/partition"

echo ""
echo "=== Stage 2: Partition (3×1×3 blocks) === $(date)"
cd "$LCTVGS"

python citygs/partition_citygs.py \
    --ply-path "$COARSE_PLY" \
    --data-dir "$DATA_DIR" \
    --data-factor 4 \
    --test-every 83 \
    --block-dim 3 1 3 \
    --ssim-threshold 0.12 \
    --output-dir "$PARTITION_DIR"

echo "=== Stage 2 done: $(date) ==="

# ─────────────────────────────────────────────────────────────────────────────
# Stage 3: Prepare block data
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== Stage 3: Prepare block camera lists === $(date)"
python citygs/prepare_block_data.py \
    --data-dir "$DATA_DIR" \
    --data-factor 4 \
    --test-every 83 \
    --partition-dir "$PARTITION_DIR" \
    --output-dir "$BLOCK_DIR" \
    --block-dim 3 1 3

echo "=== Stage 3 done: $(date) ==="

# ─────────────────────────────────────────────────────────────────────────────
# Stage 4: Block finetuning (9 blocks × 30k steps)
# ─────────────────────────────────────────────────────────────────────────────
BLOCK_NUM=9
echo ""
echo "=== Stage 4: Block finetuning (9 blocks × 30k steps, NO opacity reset) === $(date)"
cd "$EXAMPLES"

for BLOCK_ID in $(seq 0 $((BLOCK_NUM - 1))); do
    BLOCK_SUBDIR=$(printf "%03d" $BLOCK_ID)
    RESULT_DIR="$BLOCK_DIR/block_$BLOCK_SUBDIR"
    CAM_FILE="$RESULT_DIR/block_info.json"

    N_CAMS=$(python3 -c "import json; d=json.load(open('$CAM_FILE')); print(d['n_cameras'])" 2>/dev/null || echo "0")
    if [ "$N_CAMS" -lt 5 ]; then
        echo "  Block $BLOCK_ID: skipping (only $N_CAMS cameras)"
        continue
    fi

    echo ""
    echo "  [$(date)] Block $BLOCK_ID ($N_CAMS cameras) → $RESULT_DIR"

    python simple_trainer.py default \
        --disable-viewer \
        --disable-video \
        --data-dir "$DATA_DIR" \
        --data-factor 4 \
        --data-type colmap \
        --result-dir "$RESULT_DIR" \
        --test-every 83 \
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
        --strategy.refine-start-iter 500 \
        --strategy.refine-stop-iter 15000 \
        --strategy.refine-every 100 \
        --strategy.reset-every 100000 \
        --strategy.grow-grad2d 0.0002 \
        --strategy.grow-scale3d 0.01 \
        --strategy.prune-opa 0.005 \
        --packed \
        --lpips-net alex \
        --no-antialiased \
        --no-random-bkgd

    echo "  Block $BLOCK_ID done: $(date)"
done

echo "=== Stage 4 done: $(date) ==="

# ─────────────────────────────────────────────────────────────────────────────
# Stage 5: Merge
# ─────────────────────────────────────────────────────────────────────────────
MERGED_PLY="$MERGE_DIR/splat_merged.ply"
echo ""
echo "=== Stage 5: Merge blocks === $(date)"
cd "$LCTVGS"

python citygs/merge_citygs.py \
    --block-results-dir "$BLOCK_DIR" \
    --block-dim 3 1 3 \
    --step 29999 \
    --output "$MERGED_PLY" \
    --partition-dir "$PARTITION_DIR" \
    --spatial-margin 0.5

echo "=== Stage 5 done: $(date) ==="

# ─────────────────────────────────────────────────────────────────────────────
# Stage 6: Eval
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== Stage 6: Evaluation === $(date)"
python citygs/eval_citygs.py \
    --ply-path "$MERGED_PLY" \
    --data-dir "$DATA_DIR" \
    --data-factor 4 \
    --test-every 83 \
    --output-dir "$MERGE_DIR/eval" \
    --save-images

echo ""
echo "=== All done: $(date) ==="
cat "$MERGE_DIR/eval/metrics.json" 2>/dev/null || true
