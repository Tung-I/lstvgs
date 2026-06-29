#!/bin/bash
# v11 block finetuning: block-region init + 500K background Gaussians.
#
# Key changes vs v10:
#   - init PLY = block region GS + 500K random background GS (rendering context)
#   - opacity reset enabled: --strategy.reset-every 6000
#   - densification disabled: --strategy.refine-start-iter 100000 (2080Ti memory safety)
#   - 30k steps (same as v10)
#
# Skips blocks that already have point_cloud_29999.ply.
#
# How to run (2080Ti interactive session):
#   srun -t 480 -c 6 -p gpu -G 1 --mem 32G --constraint=[2080ti] --pty bash
#   cd /work/pi_rsitaram_umass_edu/tungi/lstvgs
#   bash citygs/scripts/run_blocks_v11.sh
#
# If 2080Ti unavailable or OOM, run in current L40S session instead:
#   bash citygs/scripts/run_blocks_v11.sh
#
# Time estimate (2080Ti, ~4-8 it/s): ~700-900 min total → may need 2 sessions.
# Time estimate (L40S, ~15-30 it/s): ~200-350 min total → fits in one session.

WORK_DIR="/work/pi_rsitaram_umass_edu/tungi"
CONDA_ENV="$WORK_DIR/conda/envs/gsplat"
CUDA_DIR="$WORK_DIR/cuda-13.0"
LSTVGS="$WORK_DIR/lstvgs"
EXAMPLES="$LSTVGS/gsplat/examples"
DATA_DIR="$WORK_DIR/datasets/rubble_citygs"
BLOCK_DIR="$LSTVGS/results/rubble_citygs_blocks_v11"
LOG="$BLOCK_DIR/blocks_v11.log"

unset CC CXX
export CUDAHOSTCXX="$CONDA_ENV/bin/g++"
export CUDA_HOME="$CUDA_DIR"
export PATH="$CONDA_ENV/bin:$CUDA_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_ENV/lib:$CUDA_DIR/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$LSTVGS/gsplat:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

mkdir -p "$BLOCK_DIR"
cd "$EXAMPLES"
echo "=== Block finetuning v11 started: $(date) ===" | tee -a "$LOG"

for BLOCK_ID in 0 1 2 3 4 5 7; do
    BLOCK_RESULT="$BLOCK_DIR/block_$(printf '%03d' $BLOCK_ID)"
    DONE_PLY="$BLOCK_RESULT/ply/point_cloud_29999.ply"

    if [ -f "$DONE_PLY" ]; then
        echo "Block $BLOCK_ID: already done ($(du -sh $DONE_PLY | cut -f1)), skipping." | tee -a "$LOG"
        continue
    fi

    CAM_FILE="$BLOCK_RESULT/block_info.json"
    INIT_PLY="$BLOCK_RESULT/init.ply"

    if [ ! -f "$CAM_FILE" ]; then
        echo "Block $BLOCK_ID: block_info.json missing — run setup_blocks_v11.sh first." | tee -a "$LOG"
        continue
    fi
    if [ ! -f "$INIT_PLY" ]; then
        echo "Block $BLOCK_ID: init.ply missing — run setup_blocks_v11.sh first." | tee -a "$LOG"
        continue
    fi

    N_CAM=$(python -c "import json; d=json.load(open('$CAM_FILE')); print(d['n_cameras'])")
    N_GS=$(python -c "
with open('$INIT_PLY','rb') as f:
    for _ in range(20):
        l=f.readline().decode('ascii').strip()
        if l.startswith('element vertex'): print(l.split()[-1]); break
")
    echo "" | tee -a "$LOG"
    echo "=== Block $BLOCK_ID ($N_CAM cams, ${N_GS} GS) started: $(date) ===" | tee -a "$LOG"

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
        --init-ply-path "$INIT_PLY" \
        --cam-indices-file "$CAM_FILE" \
        --sh-degree 3 \
        --sh-degree-interval 1000 \
        --ssim-lambda 0.2 \
        --strategy.refine-start-iter 100000 \
        --strategy.reset-every 6000 \
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
echo "=== v11 run complete: $(date) ===" | tee -a "$LOG"
echo "Completed blocks:"
for BLOCK_ID in 0 1 2 3 4 5 7; do
    PLY="$BLOCK_DIR/block_$(printf '%03d' $BLOCK_ID)/ply/point_cloud_29999.ply"
    if [ -f "$PLY" ]; then
        echo "  Block $BLOCK_ID: done ($(du -sh $PLY | cut -f1))"
    else
        echo "  Block $BLOCK_ID: MISSING"
    fi
done | tee -a "$LOG"
