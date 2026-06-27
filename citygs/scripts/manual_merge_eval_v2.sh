#!/bin/bash
# Manual merge+eval for rubble citygs blocks v2 (job 61138931) with spatial pruning.
# Run this after job 61138931 completes.
# Usage: bash manual_merge_eval_v2.sh

set -e

WORK_DIR="/work/pi_rsitaram_umass_edu/tungi"
CONDA_ENV="$WORK_DIR/conda/envs/gsplat"
CUDA_DIR="$WORK_DIR/cuda-13.0"

unset CC CXX
export CUDAHOSTCXX="$CONDA_ENV/bin/g++"
export CUDA_HOME="$CUDA_DIR"
export PATH="$CONDA_ENV/bin:$CUDA_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_ENV/lib:$CUDA_DIR/lib64:$LD_LIBRARY_PATH"
export PYTHONPATH="$WORK_DIR/lctvgs/gsplat:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

BLOCK_DIR="$WORK_DIR/lctvgs/results/rubble_citygs_blocks"
MERGE_DIR="$WORK_DIR/lctvgs/results/rubble_citygs_merged_v2_pruned"
PARTITION_DIR="$WORK_DIR/lctvgs/results/rubble_citygs_coarse/partition"
DATA_DIR="$WORK_DIR/datasets/rubble"

mkdir -p "$MERGE_DIR"

cd "$WORK_DIR/lctvgs"

echo "=== Merge with spatial pruning: $(date) ==="
python citygs/merge_citygs.py \
    --block-results-dir "$BLOCK_DIR" \
    --block-dim 3 1 3 \
    --step 29999 \
    --output "$MERGE_DIR/splat_merged.ply" \
    --partition-dir "$PARTITION_DIR" \
    --spatial-margin 0.5

echo ""
echo "=== Evaluation: $(date) ==="
python citygs/eval_citygs.py \
    --ply-path "$MERGE_DIR/splat_merged.ply" \
    --data-dir "$DATA_DIR" \
    --data-factor 4 \
    --test-every 8 \
    --output-dir "$MERGE_DIR/eval" \
    --save-images

echo ""
echo "=== Results: ==="
cat "$MERGE_DIR/eval/metrics.json"
