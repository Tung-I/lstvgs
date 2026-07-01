#!/bin/bash
# Block-wise Lapis-CityGS (Milestone 1): train one CityGS block's layered pyramid.
#   L0 = coarse-anchored (shared coarse ply cropped to the block's expanded cell),
#   L1-L3 = LapisGS enhancement at increasing resolution (factors 32->16->8->4 so
#   L3 == the CityGS factor-4 resolution). Trains + evals IN-REGION (cam subset).
# Runs against the lctvgs gsplat (LapisGS's native repo). Skip-aware: trainer skips
# layers whose layer_LL_full.ply exists; prep skips if l0_init.ply exists.
# Usage: bash run_lapis_block.sh BLOCK_ID [STEPS_PER_LAYER]
set -o pipefail
BLOCK_ID="${1:?usage: run_lapis_block.sh BLOCK_ID [STEPS_PER_LAYER]}"
STEPS="${2:-30000}"

WORK_DIR="/work/pi_rsitaram_umass_edu/tungi"
LSTVGS="$WORK_DIR/lstvgs"
LCTVGS="$WORK_DIR/lctvgs"
CONDA_ENV="$WORK_DIR/conda/envs/gsplat"
CUDA_DIR="$WORK_DIR/cuda-13.0"
ORACLE="$LSTVGS/results/rubble_citygs_gsplat_oracle"
DATA_DIR="$ORACLE/data"
COARSE="$WORK_DIR/CityGaussian/output/rubble_coarse/point_cloud/iteration_30000/point_cloud.ply"
BID="$(printf '%03d' "$BLOCK_ID")"
CAMFILE="$ORACLE/blocks/block_$BID/block_info.json"
RESULT="$LSTVGS/results/rubble_lapis_block$BID"
L0INIT="$RESULT/l0_init.ply"
LOG="$RESULT/run_lapis.log"

unset CC CXX
export CUDAHOSTCXX="$CONDA_ENV/bin/g++"
export CUDA_HOME="$CUDA_DIR"
export PATH="$CONDA_ENV/bin:$CUDA_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_ENV/lib:$CUDA_DIR/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$LCTVGS/gsplat:${PYTHONPATH:-}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

mkdir -p "$RESULT"
echo "=== lapis block $BLOCK_ID start $(date); steps/layer=$STEPS ===" | tee -a "$LOG"

# 0) the lctvgs Parser needs images_{factor} to exist (it then resizes images -> images_{factor}_png).
#    Provision symlinks to the full-res source for every pyramid factor (images_4 already exists).
IMG_SRC="$(readlink -f "$DATA_DIR/images")"
for F in 8 16 32; do
    [ -e "$DATA_DIR/images_$F" ] || ln -s "$IMG_SRC" "$DATA_DIR/images_$F"
done

# 1) coarse-anchored L0 init (skip if present)
if [ ! -f "$L0INIT" ]; then
    python "$LSTVGS/citygs/prep_lapis_block_init.py" \
        --coarse-ply "$COARSE" --block-id "$BLOCK_ID" \
        --block-dim 3 1 3 --aabb -50 -100 -135 50 300 -5 --margin 0.5 \
        --out "$L0INIT" 2>&1 | tee -a "$LOG" || exit 1
else
    echo "l0_init.ply exists, skipping prep." | tee -a "$LOG"
fi

# 2) layered pyramid + in-region per-layer eval
cd "$LCTVGS/lapisgs"
python -u train_lapisgs.py \
    --data-dir "$DATA_DIR" --test-every 8 --no-normalize-world-space \
    --result-dir "$RESULT" \
    --cam-indices-file "$CAMFILE" \
    --l0-init-ply "$L0INIT" \
    --data-factors 32 16 8 4 \
    --steps-per-layer "$STEPS" \
    --early-stop --es-min-steps 20000 \
    --eval-fixed-factor 4 \
    --stages layer0 layer1 layer2 layer3 eval eval_fixed >> "$LOG" 2>&1
rc=$?
echo "=== lapis block $BLOCK_ID exit=$rc $(date) ===" | tee -a "$LOG"
exit $rc
