#!/bin/bash
# v12 block finetuning: full coarse PLY init + opacity reset disabled.
#
# Root causes fixed vs v11:
#   - v11 used 500K random background (6% of scene) → insufficient context
#   - v11 opacity reset-every 6000 killed GS → only 3.3M survived merge
#
# v12 fixes:
#   - init-ply = full coarse PLY (8.13M GS) → complete rendering context
#   - --strategy.reset-every 100000 → opacity reset disabled (preserves coarse GS)
#   - densification disabled: --strategy.refine-start-iter 100000
#   - 30k steps unchanged
#
# Memory: 8.13M GS, ~10-15GB GPU RAM → L40S (46GB) required.
# Speed: ~5-7 it/s → ~80 min/block → ~560 min total for 7 blocks.
# Needs 2 srun sessions of 480 min (skip-if-done handles resumption).
#
# How to run:
#   srun -t 480 -c 6 -p gpu -G 1 --mem 48G --constraint=[l40s] --pty bash
#   cd /work/pi_rsitaram_umass_edu/tungi/lstvgs
#   bash citygs/scripts/run_blocks_v12.sh

WORK_DIR="/work/pi_rsitaram_umass_edu/tungi"
CONDA_ENV="$WORK_DIR/conda/envs/gsplat"
CUDA_DIR="$WORK_DIR/cuda-13.0"
LSTVGS="$WORK_DIR/lstvgs"
LCTVGS="$WORK_DIR/lctvgs"
EXAMPLES="$LSTVGS/gsplat/examples"
DATA_DIR="$WORK_DIR/datasets/rubble_citygs"
COARSE_PLY="$LCTVGS/results/rubble_citygs_coarse_v9/ply/point_cloud_29999.ply"
# Reuse block_info.json from v11 (same camera assignments)
V11_BLOCK_DIR="$LSTVGS/results/rubble_citygs_blocks_v11"
BLOCK_DIR="$LSTVGS/results/rubble_citygs_blocks_v12"
LOG="$BLOCK_DIR/blocks_v12.log"

unset CC CXX
export CUDAHOSTCXX="$CONDA_ENV/bin/g++"
export CUDA_HOME="$CUDA_DIR"
export PATH="$CONDA_ENV/bin:$CUDA_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_ENV/lib:$CUDA_DIR/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$LSTVGS/gsplat:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

if [ ! -f "$COARSE_PLY" ]; then
    echo "ERROR: coarse PLY not found: $COARSE_PLY"; exit 1
fi

mkdir -p "$BLOCK_DIR"
cd "$EXAMPLES"
echo "=== Block finetuning v12 started: $(date) ===" | tee -a "$LOG"
echo "Init PLY: $COARSE_PLY" | tee -a "$LOG"

for BLOCK_ID in 0 1 2 3 4 5 7; do
    BLOCK_RESULT="$BLOCK_DIR/block_$(printf '%03d' $BLOCK_ID)"
    DONE_PLY="$BLOCK_RESULT/ply/point_cloud_29999.ply"

    if [ -f "$DONE_PLY" ]; then
        echo "Block $BLOCK_ID: already done ($(du -sh $DONE_PLY | cut -f1)), skipping." | tee -a "$LOG"
        continue
    fi

    # Reuse block_info.json from v11
    CAM_FILE="$V11_BLOCK_DIR/block_$(printf '%03d' $BLOCK_ID)/block_info.json"
    if [ ! -f "$CAM_FILE" ]; then
        echo "Block $BLOCK_ID: block_info.json missing at $CAM_FILE — run setup_blocks_v11.sh first." | tee -a "$LOG"
        continue
    fi

    mkdir -p "$BLOCK_RESULT"
    cp "$CAM_FILE" "$BLOCK_RESULT/block_info.json"

    N_CAM=$(python -c "import json; d=json.load(open('$CAM_FILE')); print(d['n_cameras'])")
    N_GS=$(python -c "
with open('$COARSE_PLY','rb') as f:
    for _ in range(20):
        l=f.readline().decode('ascii').strip()
        if l.startswith('element vertex'): print(l.split()[-1]); break
")
    echo "" | tee -a "$LOG"
    echo "=== Block $BLOCK_ID ($N_CAM cams, ${N_GS} GS init) started: $(date) ===" | tee -a "$LOG"

    python -u simple_trainer.py default \
        --disable-viewer \
        --data-dir "$DATA_DIR" \
        --data-factor 4 \
        --data-type colmap \
        --result-dir "$BLOCK_RESULT" \
        --test-every 83 \
        --normalize-world-space \
        --max-steps 30000 \
        --eval-steps 30000 \
        --save-steps 30000 \
        --save-ply \
        --ply-steps 30000 \
        --batch-size 1 \
        --init-type ply \
        --init-ply-path "$COARSE_PLY" \
        --cam-indices-file "$BLOCK_RESULT/block_info.json" \
        --sh-degree 3 \
        --sh-degree-interval 1000 \
        --ssim-lambda 0.2 \
        --strategy.refine-start-iter 100000 \
        --strategy.reset-every 100000 \
        --strategy.prune-opa 0.005 \
        --packed \
        --lpips-net alex \
        --no-antialiased \
        --no-random-bkgd >> "$LOG" 2>&1

    EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
        echo "=== Block $BLOCK_ID FAILED (exit $EXIT_CODE): $(date) ===" | tee -a "$LOG"
        echo "    Check log for OOM or other errors: $LOG"
    else
        echo "=== Block $BLOCK_ID done: $(date) ===" | tee -a "$LOG"
    fi
done

echo "" | tee -a "$LOG"
echo "=== v12 run complete: $(date) ===" | tee -a "$LOG"
echo "Completed blocks:"
for BLOCK_ID in 0 1 2 3 4 5 7; do
    PLY="$BLOCK_DIR/block_$(printf '%03d' $BLOCK_ID)/ply/point_cloud_29999.ply"
    if [ -f "$PLY" ]; then
        echo "  Block $BLOCK_ID: done ($(du -sh $PLY | cut -f1))"
    else
        echo "  Block $BLOCK_ID: MISSING"
    fi
done | tee -a "$LOG"
