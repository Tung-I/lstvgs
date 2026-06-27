#!/bin/bash
# v10 block finetuning: zero-overlap init PLYs + densification enabled
# Key changes from v9:
#   - init PLYs from blocks_v10 (margin=0.0, no spatial overlap)
#   - densification ENABLED (removed --strategy.refine-start-iter 100000)
#   - opacity reset DISABLED (--strategy.reset-every 100000)
# Skips blocks that already have point_cloud_29999.ply.

WORK_DIR="/work/pi_rsitaram_umass_edu/tungi"
CONDA_ENV="$WORK_DIR/conda/envs/gsplat"
CUDA_DIR="$WORK_DIR/cuda-13.0"
LCTVGS="$WORK_DIR/lctvgs"
EXAMPLES="$LCTVGS/gsplat/examples"
DATA_DIR="$WORK_DIR/datasets/rubble_citygs"
BLOCK_DIR="$LCTVGS/results/rubble_citygs_blocks_v10"
LOG="$BLOCK_DIR/blocks_v10.log"

unset CC CXX
export CUDAHOSTCXX="$CONDA_ENV/bin/g++"
export CUDA_HOME="$CUDA_DIR"
export PATH="$CONDA_ENV/bin:$CUDA_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_ENV/lib:$CUDA_DIR/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$LCTVGS/gsplat:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

cd "$EXAMPLES"
echo "=== Block finetuning v10 started: $(date) ===" | tee -a "$LOG"

for BLOCK_ID in 0 1 2 3 4 5 7; do
    BLOCK_RESULT="$BLOCK_DIR/block_$(printf '%03d' $BLOCK_ID)"
    DONE_PLY="$BLOCK_RESULT/ply/point_cloud_29999.ply"

    if [ -f "$DONE_PLY" ]; then
        echo "Block $BLOCK_ID: already done ($(du -sh $DONE_PLY | cut -f1)), skipping." | tee -a "$LOG"
        continue
    fi

    CAM_FILE="$BLOCK_RESULT/block_info.json"
    INIT_PLY="$BLOCK_RESULT/init.ply"
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
        --strategy.reset-every 100000 \
        --strategy.prune-opa 0.005 \
        --packed \
        --lpips-net alex \
        --no-antialiased \
        --no-random-bkgd >> "$LOG" 2>&1
    echo "=== Block $BLOCK_ID done: $(date) ===" | tee -a "$LOG"
done

echo "" | tee -a "$LOG"
echo "=== v10 run complete: $(date) ===" | tee -a "$LOG"
