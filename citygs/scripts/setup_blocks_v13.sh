#!/bin/bash
# Generate per-block init PLY for v13: spatially-filtered block region only (no background).
#
# Per CityGS V1 paper: each block is initialized from "the containing coarse global
# Gaussians" (GKj = Gaussians within block's spatial bounds). This is spatially-filtered
# init, NOT the full coarse PLY.
#
# v10 used this same approach but with densification disabled → optimizer distorted block
# GS to cover visible background. v13 uses the same init but with densification ON,
# so new GS are created for background during training and pruned at merge.
#
# Run once on any node (CPU only, ~2 min):
#   cd /work/pi_rsitaram_umass_edu/tungi/lstvgs
#   bash citygs/scripts/setup_blocks_v13.sh

WORK_DIR="/work/pi_rsitaram_umass_edu/tungi"
LSTVGS="$WORK_DIR/lstvgs"
LCTVGS="$WORK_DIR/lctvgs"
CONDA_ENV="$WORK_DIR/conda/envs/gsplat"

COARSE_PLY="$LCTVGS/results/rubble_citygs_coarse_v9/ply/point_cloud_29999.ply"
PARTITION_DIR="$LCTVGS/results/rubble_citygs_coarse_v9/partition"
V11_BLOCK_DIR="$LSTVGS/results/rubble_citygs_blocks_v11"
V13_BLOCK_DIR="$LSTVGS/results/rubble_citygs_blocks_v13"

export PATH="$CONDA_ENV/bin:$PATH"

echo "=== CityGS v13 setup: $(date) ==="
echo "Coarse PLY:    $COARSE_PLY"
echo "Partition dir: $PARTITION_DIR"
echo "V13 output:    $V13_BLOCK_DIR"

if [ ! -f "$COARSE_PLY" ]; then echo "ERROR: coarse PLY not found"; exit 1; fi
if [ ! -d "$PARTITION_DIR" ]; then echo "ERROR: partition dir not found"; exit 1; fi

# Copy block_info.json from v11 (same camera assignments)
for BLOCK_ID in 0 1 2 3 4 5 7; do
    SRC="$V11_BLOCK_DIR/block_$(printf '%03d' $BLOCK_ID)/block_info.json"
    DST_DIR="$V13_BLOCK_DIR/block_$(printf '%03d' $BLOCK_ID)"
    if [ ! -f "$SRC" ]; then echo "WARNING: block_info.json missing for block $BLOCK_ID — skipping"; continue; fi
    mkdir -p "$DST_DIR"
    cp "$SRC" "$DST_DIR/block_info.json"
    echo "Copied block_info.json → block_$(printf '%03d' $BLOCK_ID)/"
done

echo ""
echo "=== Generating per-block init PLYs (block region only, no background) ==="
python "$LSTVGS/citygs/prepare_block_ply.py" \
    --coarse-ply "$COARSE_PLY" \
    --partition-dir "$PARTITION_DIR" \
    --output-dir "$V13_BLOCK_DIR" \
    --margin 0.0 \
    --background-count 0

echo ""
echo "=== Setup complete: $(date) ==="
echo "Block GS counts:"
for BLOCK_ID in 0 1 2 3 4 5 7; do
    PLY="$V13_BLOCK_DIR/block_$(printf '%03d' $BLOCK_ID)/init.ply"
    if [ -f "$PLY" ]; then
        N=$(python3 -c "
with open('$PLY','rb') as f:
    for _ in range(20):
        l=f.readline().decode('ascii').strip()
        if l.startswith('element vertex'): print(l.split()[-1]); break
")
        echo "  Block $BLOCK_ID: ${N} GS"
    fi
done
