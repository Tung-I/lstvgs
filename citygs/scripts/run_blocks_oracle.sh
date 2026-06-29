#!/bin/bash
# Phase 3 oracle-aligned gsplat CityGS block training.
# Init = FULL INRIA coarse ply (raw frame). Train on each block's cameras only (block_info.json
# built by citygs/phase3_build_blocks.py). Trains in the SAME world frame as the coarse ply, so
# --no-normalize-world-space is REQUIRED. val held out separately (test-every huge => all assigned
# cams train). Skip-aware. Usage: bash run_blocks_oracle.sh [MAX_STEPS] [BLOCK_IDS...]
set -o pipefail
WORK_DIR="/work/pi_rsitaram_umass_edu/tungi"
LSTVGS="$WORK_DIR/lstvgs"
CONDA_ENV="$WORK_DIR/conda/envs/gsplat"
CUDA_DIR="$WORK_DIR/cuda-13.0"
EXAMPLES="$LSTVGS/gsplat/examples"
DATA_DIR="$LSTVGS/results/rubble_citygs_gsplat_oracle/data"
COARSE_PLY="$WORK_DIR/CityGaussian/output/rubble_coarse/point_cloud/iteration_30000/point_cloud.ply"
BLOCK_DIR="$LSTVGS/results/rubble_citygs_gsplat_oracle/blocks"
LOG="$LSTVGS/results/rubble_citygs_gsplat_oracle/run_oracle.log"

unset CC CXX
export CUDAHOSTCXX="$CONDA_ENV/bin/g++"
export CUDA_HOME="$CUDA_DIR"
export PATH="$CONDA_ENV/bin:$CUDA_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_ENV/lib:$CUDA_DIR/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$LSTVGS/gsplat:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

MAX_STEPS="${1:-30000}"; shift || true
BLOCKS="${*:-0 1 2 3 4 5 6 7 8}"
cd "$EXAMPLES"
echo "=== oracle block run started $(date); max_steps=$MAX_STEPS blocks=[$BLOCKS] ===" | tee -a "$LOG"

for BLOCK_ID in $BLOCKS; do
    BR="$BLOCK_DIR/block_$(printf '%03d' $BLOCK_ID)"
    CAM_FILE="$BR/block_info.json"
    DONE_PLY="$BR/ply/point_cloud_$((MAX_STEPS-1)).ply"
    [ -f "$DONE_PLY" ] && { echo "block $BLOCK_ID: done, skipping." | tee -a "$LOG"; continue; }
    [ -f "$CAM_FILE" ] || { echo "block $BLOCK_ID: block_info.json missing." | tee -a "$LOG"; continue; }
    N_CAM=$(python -c "import json;print(json.load(open('$CAM_FILE'))['n_cameras'])")
    echo "=== block $BLOCK_ID ($N_CAM cams) start $(date) ===" | tee -a "$LOG"
    python -u simple_trainer.py default \
        --disable-viewer \
        --data-dir "$DATA_DIR" --data-factor 4 --data-type colmap \
        --result-dir "$BR" \
        --test-every 999999 --no-normalize-world-space \
        --max-steps "$MAX_STEPS" --eval-steps "$MAX_STEPS" --save-steps "$MAX_STEPS" \
        --save-ply --ply-steps "$MAX_STEPS" \
        --batch-size 1 \
        --init-type ply --init-ply-path "$COARSE_PLY" \
        --cam-indices-file "$CAM_FILE" \
        --sh-degree 3 --sh-degree-interval 1000 --ssim-lambda 0.2 \
        --means-lr 0.000064 --scales-lr 0.004 --opacities-lr 0.05 --quats-lr 0.001 \
        --sh0-lr 0.0025 --shN-lr 0.000125 \
        --strategy.reset-every 3000 \
        --strategy.refine-start-iter 500 --strategy.refine-stop-iter 15000 --strategy.refine-every 100 \
        --strategy.prune-opa 0.005 --strategy.grow-grad2d 0.0002 \
        --packed --lpips-net alex --no-antialiased --no-random-bkgd >> "$LOG" 2>&1
    rc=$?
    echo "=== block $BLOCK_ID exit=$rc $(date) ===" | tee -a "$LOG"
done
echo "=== oracle block run done $(date) ===" | tee -a "$LOG"
