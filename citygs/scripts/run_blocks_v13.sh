#!/bin/bash
# v13 block finetuning: paper-faithful implementation of CityGS V1 (ECCV 2024).
#
# Key findings from paper review (vs earlier code analysis of V2/main branch):
#   - NO depth/normal regularization in V1 — that is V2 only
#   - Block finetuning is 30K steps (NOT 60K — that was for the 3DGS† baseline)
#   - LR reduction applies to ALL block training: position -60% (6.4e-5), scale -20% (4e-3)
#   - Block init: spatially-filtered coarse GS (block region only), NOT full coarse PLY
#   - Standard 3DGS densification + opacity reset (every 3K) enabled
#   - Camera assignment: SSIM-based (already done by our partition, threshold 0.12)
#
# Why v10 (spatially-filtered init) failed: densification was disabled. Without it, block
# GS had to stretch to cover background regions visible in block cameras → distortion.
# With densification ON, new GS are created for background during training, then pruned
# at merge (strict 0% spatial margin).
#
# Why v12 (full coarse PLY) was wrong: paper uses spatially-filtered init. Full coarse
# PLY with no reset/densification is not how the paper describes the method.
#
# Expected: ~1-2M GS per block → ~20 it/s → ~25 min/block → ~3h total (1 session).
#
# Prerequisites:
#   bash citygs/scripts/setup_blocks_v13.sh   (generates per-block init.ply)
#
# How to run:
#   srun -t 480 -c 6 -p gpu -G 1 --constraint=[l40s] --mem 48G --pty bash
#   cd /work/pi_rsitaram_umass_edu/tungi/lstvgs
#   bash citygs/scripts/setup_blocks_v13.sh
#   bash citygs/scripts/run_blocks_v13.sh

WORK_DIR="/work/pi_rsitaram_umass_edu/tungi"
CONDA_ENV="$WORK_DIR/conda/envs/gsplat"
CUDA_DIR="$WORK_DIR/cuda-13.0"
LSTVGS="$WORK_DIR/lstvgs"
EXAMPLES="$LSTVGS/gsplat/examples"
DATA_DIR="$WORK_DIR/datasets/rubble_citygs"
BLOCK_DIR="$LSTVGS/results/rubble_citygs_blocks_v13"
LOG="$BLOCK_DIR/blocks_v13.log"

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
echo "=== Block finetuning v13 started: $(date) ===" | tee -a "$LOG"
echo "Paper-faithful: spatially-filtered init, densification ON, standard reset (3K), reduced LR" | tee -a "$LOG"

for BLOCK_ID in 0 1 2 3 4 5 7; do
    BLOCK_RESULT="$BLOCK_DIR/block_$(printf '%03d' $BLOCK_ID)"
    DONE_PLY="$BLOCK_RESULT/ply/point_cloud_29999.ply"
    INIT_PLY="$BLOCK_RESULT/init.ply"

    if [ -f "$DONE_PLY" ]; then
        echo "Block $BLOCK_ID: already done ($(du -sh $DONE_PLY | cut -f1)), skipping." | tee -a "$LOG"
        continue
    fi

    CAM_FILE="$BLOCK_RESULT/block_info.json"
    if [ ! -f "$CAM_FILE" ]; then
        echo "Block $BLOCK_ID: block_info.json missing — run setup_blocks_v13.sh first." | tee -a "$LOG"
        continue
    fi
    if [ ! -f "$INIT_PLY" ]; then
        echo "Block $BLOCK_ID: init.ply missing — run setup_blocks_v13.sh first." | tee -a "$LOG"
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
        --init-ply-path "$INIT_PLY" \
        --cam-indices-file "$CAM_FILE" \
        --sh-degree 3 \
        --sh-degree-interval 1000 \
        --ssim-lambda 0.2 \
        --means-lr 0.000064 \
        --scales-lr 0.004 \
        --strategy.reset-every 3000 \
        --strategy.refine-start-iter 500 \
        --strategy.refine-stop-iter 15000 \
        --strategy.refine-every 100 \
        --strategy.prune-opa 0.005 \
        --packed \
        --lpips-net alex \
        --no-antialiased \
        --no-random-bkgd >> "$LOG" 2>&1

    EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
        echo "=== Block $BLOCK_ID FAILED (exit $EXIT_CODE): $(date) ===" | tee -a "$LOG"
    else
        echo "=== Block $BLOCK_ID done: $(date) ===" | tee -a "$LOG"
    fi
done

echo "" | tee -a "$LOG"
echo "=== v13 run complete: $(date) ===" | tee -a "$LOG"
for BLOCK_ID in 0 1 2 3 4 5 7; do
    PLY="$BLOCK_DIR/block_$(printf '%03d' $BLOCK_ID)/ply/point_cloud_29999.ply"
    if [ -f "$PLY" ]; then
        echo "  Block $BLOCK_ID: done ($(du -sh $PLY | cut -f1))"
    else
        echo "  Block $BLOCK_ID: MISSING"
    fi
done | tee -a "$LOG"
