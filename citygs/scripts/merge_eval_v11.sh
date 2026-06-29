#!/bin/bash
# Merge all v11 block PLYs and evaluate on the 21 val views.
#
# Run after all 7 blocks have point_cloud_29999.ply:
#   ls results/rubble_citygs_blocks_v11/block_*/ply/point_cloud_29999.ply
#
# Usage (from lstvgs repo root):
#   bash citygs/scripts/merge_eval_v11.sh

WORK_DIR="/work/pi_rsitaram_umass_edu/tungi"
CONDA_ENV="$WORK_DIR/conda/envs/gsplat"
CUDA_DIR="$WORK_DIR/cuda-13.0"
LSTVGS="$WORK_DIR/lstvgs"
LCTVGS="$WORK_DIR/lctvgs"
DATA_DIR="$WORK_DIR/datasets/rubble_citygs"

BLOCK_DIR="$LSTVGS/results/rubble_citygs_blocks_v11"
PARTITION_DIR="$LCTVGS/results/rubble_citygs_coarse_v9/partition"
MERGED_DIR="$LSTVGS/results/rubble_citygs_merged_v11"
MERGED_PLY="$MERGED_DIR/merged.ply"

unset CC CXX
export CUDAHOSTCXX="$CONDA_ENV/bin/g++"
export CUDA_HOME="$CUDA_DIR"
export PATH="$CONDA_ENV/bin:$CUDA_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_ENV/lib:$CUDA_DIR/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$LSTVGS/gsplat:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== v11 merge + eval: $(date) ==="

# Verify all blocks are complete
MISSING=0
for BLOCK_ID in 0 1 2 3 4 5 7; do
    PLY="$BLOCK_DIR/block_$(printf '%03d' $BLOCK_ID)/ply/point_cloud_29999.ply"
    if [ ! -f "$PLY" ]; then
        echo "WARNING: Block $BLOCK_ID PLY missing: $PLY"
        MISSING=$((MISSING + 1))
    fi
done
if [ $MISSING -gt 0 ]; then
    echo "ERROR: $MISSING block(s) not complete. Run run_blocks_v11.sh first."
    exit 1
fi

mkdir -p "$MERGED_DIR"

echo ""
echo "--- Merge ---"
# spatial-margin 0.02: strips background GS; tiny overlap for clean block seams
python "$LSTVGS/citygs/merge_citygs.py" \
    --block-results-dir "$BLOCK_DIR" \
    --block-dim 3 1 3 \
    --step 30000 \
    --partition-dir "$PARTITION_DIR" \
    --output "$MERGED_PLY" \
    --prune-opacity 0.005 \
    --spatial-margin 0.02

echo ""
echo "--- Evaluate ---"
python "$LSTVGS/citygs/eval_citygs.py" \
    --ply-path "$MERGED_PLY" \
    --data-dir "$DATA_DIR" \
    --data-factor 4 \
    --test-every 83 \
    --output-dir "$MERGED_DIR"

echo ""
echo "=== Done: $(date) ==="
echo "Results: $MERGED_DIR/metrics.json"
if [ -f "$MERGED_DIR/metrics.json" ]; then
    python -c "
import json
m = json.load(open('$MERGED_DIR/metrics.json'))
print(f'  PSNR:  {m[\"psnr\"]:.2f} dB  (paper: 25.77, coarse: 24.91)')
print(f'  SSIM:  {m[\"ssim\"]:.3f}    (paper: 0.813)')
print(f'  LPIPS: {m[\"lpips\"]:.3f}   (paper: 0.228)')
"
fi
