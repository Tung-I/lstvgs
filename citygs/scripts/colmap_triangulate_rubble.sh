#!/bin/bash
#SBATCH --job-name=colmap-rubble
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/work/pi_rsitaram_umass_edu/tungi/lctvgs/results/colmap_rubble/slurm_%j.log
#SBATCH --error=/work/pi_rsitaram_umass_edu/tungi/lctvgs/results/colmap_rubble/slurm_%j.log

set -e

COLMAP_BIN="/work/pi_rsitaram_umass_edu/tungi/conda/envs/4dgs/bin/colmap"
DATA_DIR="/work/pi_rsitaram_umass_edu/tungi/datasets/rubble"
WORK_DIR="/work/pi_rsitaram_umass_edu/tungi/lctvgs"
COLMAP_OUT="$WORK_DIR/results/colmap_rubble"
# Synthetic "rubble_sfm" dataset that points to the same images but has SFM points
SFM_DATA_DIR="$WORK_DIR/datasets/rubble_sfm"

mkdir -p "$COLMAP_OUT" "$SFM_DATA_DIR/sparse/0"

echo "=== COLMAP point triangulation for Rubble: $(date) ==="
echo "Host: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# ── Step 1: Create scaled-down sparse reconstruction ──────────────────────
# sparse/0 has camera params for full-res (4608×3456).
# Feature extraction runs on images_4 (1152×864), so scale intrinsics by 1/4.
# Original PINHOLE: fx=2977.53, fy=2977.53, cx=2304.0, cy=1728.0
# Scaled 4x:        fx=744.38,  fy=744.38,  cx=576.0,  cy=432.0

SPARSE_SCALED="$COLMAP_OUT/sparse_scaled"
mkdir -p "$SPARSE_SCALED"

cat > "$SPARSE_SCALED/cameras.txt" << 'EOF'
# Camera list with one line of data per camera:
#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
# Number of cameras: 1
1 PINHOLE 1152 864 744.3825 744.3825 576.0 432.0
EOF

# Copy image poses (poses in world space are scale-invariant)
cp "$DATA_DIR/sparse/0/images.txt" "$SPARSE_SCALED/images.txt"
# Empty points3D.txt
cat > "$SPARSE_SCALED/points3D.txt" << 'EOF'
# 3D point list with one line of data per point:
#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)
# Number of points: 0, mean track length: 0
EOF

echo "Created scaled sparse reconstruction (4x-down, PINHOLE 1152x864 f=744.38)"
echo ""

# ── Step 2: Feature extraction on downscaled images ───────────────────────
DB="$COLMAP_OUT/database.db"
echo "=== Step 2: Feature extraction from images_4 (1152×864): $(date) ==="

"$COLMAP_BIN" feature_extractor \
    --database_path "$DB" \
    --image_path "$DATA_DIR/images_4" \
    --ImageReader.camera_model PINHOLE \
    --ImageReader.single_camera 1 \
    --ImageReader.camera_params "744.3825,744.3825,576.0,432.0" \
    --SiftExtraction.use_gpu 1 \
    --SiftExtraction.num_threads 8 \
    --SiftExtraction.max_image_size 1200 \
    --SiftExtraction.max_num_features 8192

echo "Feature extraction complete: $(date)"
echo ""

# ── Step 3: Sequential matching ───────────────────────────────────────────
# UAV nadir survey images are captured in sequential order (row by row).
# overlap=20 captures 20 preceding and 20 following neighbors.
# quadratic_overlap helps catch images from adjacent rows at grid turns.
echo "=== Step 3: Sequential feature matching (overlap=20): $(date) ==="

"$COLMAP_BIN" sequential_matcher \
    --database_path "$DB" \
    --SequentialMatching.overlap 20 \
    --SequentialMatching.quadratic_overlap 1 \
    --SequentialMatching.loop_detection 0 \
    --SiftMatching.use_gpu 1 \
    --SiftMatching.max_num_matches 16384

echo "Matching complete: $(date)"
echo ""

# ── Step 4: Point triangulation with fixed poses ───────────────────────────
echo "=== Step 4: Point triangulation with fixed camera poses: $(date) ==="

SPARSE_OUT="$COLMAP_OUT/sparse_triangulated"
mkdir -p "$SPARSE_OUT"

"$COLMAP_BIN" point_triangulator \
    --database_path "$DB" \
    --image_path "$DATA_DIR/images_4" \
    --input_path "$SPARSE_SCALED" \
    --output_path "$SPARSE_OUT" \
    --Mapper.ba_refine_focal_length 0 \
    --Mapper.ba_refine_principal_point 0 \
    --Mapper.ba_refine_extra_params 0 \
    --Mapper.filter_max_reproj_error 4.0 \
    --Mapper.min_num_inliers 30

echo "Triangulation complete: $(date)"
echo ""

# ── Step 5: Convert binary → TXT format ───────────────────────────────────
echo "=== Step 5: Convert to TXT format: $(date) ==="

"$COLMAP_BIN" model_converter \
    --input_path "$SPARSE_OUT" \
    --output_path "$SPARSE_OUT" \
    --output_type TXT

# Count triangulated points
N_POINTS=$(grep -c "^[0-9]" "$SPARSE_OUT/points3D.txt" 2>/dev/null || echo 0)
echo "Triangulated 3D points: $N_POINTS"
echo ""

# ── Step 6: Build rubble_sfm dataset directory ─────────────────────────────
# Creates a dataset directory within lctvgs/ that shares images with the
# original dataset but has SFM points from triangulation. Training uses
# this dir so we stay within lctvgs/ and don't modify the original dataset.
echo "=== Step 6: Building rubble_sfm dataset symlinks: $(date) ==="

# Symlink image directories (read-only, no copy)
ln -sfn "$DATA_DIR/images" "$SFM_DATA_DIR/images"
ln -sfn "$DATA_DIR/images_4" "$SFM_DATA_DIR/images_4"
ln -sfn "$DATA_DIR/images_4_png" "$SFM_DATA_DIR/images_4_png"

# Copy the camera and image pose files from original sparse/0
cp "$DATA_DIR/sparse/0/cameras.txt" "$SFM_DATA_DIR/sparse/0/cameras.txt"
cp "$DATA_DIR/sparse/0/images.txt"  "$SFM_DATA_DIR/sparse/0/images.txt"
# Use the triangulated points3D.txt
cp "$SPARSE_OUT/points3D.txt" "$SFM_DATA_DIR/sparse/0/points3D.txt"

echo "rubble_sfm dataset created at: $SFM_DATA_DIR"
echo "  cameras.txt: from original (full-res params, gsplat parser will scale by data_factor=4)"
echo "  images.txt:  from original"
echo "  points3D.txt: from COLMAP triangulation (N=$N_POINTS points)"
echo ""

echo "=== Done: $(date) ==="
echo ""
echo "Next step: train rubble coarse with SFM init:"
echo "  --data-dir $SFM_DATA_DIR"
echo "  --init-type sfm"
echo "  (Estimated PSNR: 20+ dB vs current 14.63 dB with random init)"
