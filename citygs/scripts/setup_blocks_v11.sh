#!/bin/bash
# Set up v11 block directories in lstvgs.
# - Copies block_info.json from lctvgs v10 (same camera assignments)
# - Generates per-block init.ply: spatial region + 500K background Gaussians
#
# Run once on any node (CPU only, ~2 min):
#   bash citygs/scripts/setup_blocks_v11.sh

WORK_DIR="/work/pi_rsitaram_umass_edu/tungi"
LSTVGS="$WORK_DIR/lstvgs"
LCTVGS="$WORK_DIR/lctvgs"
CONDA_ENV="$WORK_DIR/conda/envs/gsplat"

COARSE_PLY="$LCTVGS/results/rubble_citygs_coarse_v9/ply/point_cloud_29999.ply"
PARTITION_DIR="$LCTVGS/results/rubble_citygs_coarse_v9/partition"
V10_BLOCK_DIR="$LCTVGS/results/rubble_citygs_blocks_v10"
V11_BLOCK_DIR="$LSTVGS/results/rubble_citygs_blocks_v11"

export PATH="$CONDA_ENV/bin:$PATH"

echo "=== CityGS v11 setup: $(date) ==="
echo "Coarse PLY:    $COARSE_PLY"
echo "Partition dir: $PARTITION_DIR"
echo "V10 src:       $V10_BLOCK_DIR"
echo "V11 output:    $V11_BLOCK_DIR"
echo ""

# Verify source files exist
if [ ! -f "$COARSE_PLY" ]; then
    echo "ERROR: coarse PLY not found: $COARSE_PLY"; exit 1
fi
if [ ! -d "$PARTITION_DIR" ]; then
    echo "ERROR: partition dir not found: $PARTITION_DIR"; exit 1
fi

# Copy block_info.json from v10 for each valid block
for BLOCK_ID in 0 1 2 3 4 5 7; do
    SRC="$V10_BLOCK_DIR/block_$(printf '%03d' $BLOCK_ID)/block_info.json"
    DST_DIR="$V11_BLOCK_DIR/block_$(printf '%03d' $BLOCK_ID)"
    if [ ! -f "$SRC" ]; then
        echo "WARNING: block_info.json not found for block $BLOCK_ID at $SRC — skipping"
        continue
    fi
    mkdir -p "$DST_DIR"
    cp "$SRC" "$DST_DIR/block_info.json"
    echo "Copied block_info.json → block_$(printf '%03d' $BLOCK_ID)/"
done

echo ""
echo "=== Generating per-block init PLYs (block region + 500K background) ==="
python "$LSTVGS/citygs/prepare_block_ply.py" \
    --coarse-ply "$COARSE_PLY" \
    --partition-dir "$PARTITION_DIR" \
    --output-dir "$V11_BLOCK_DIR" \
    --margin 0.0 \
    --background-count 500000

echo ""
echo "=== Setup complete: $(date) ==="
echo "Block dirs: $V11_BLOCK_DIR"
ls "$V11_BLOCK_DIR"
