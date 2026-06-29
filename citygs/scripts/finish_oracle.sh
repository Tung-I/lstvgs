#!/bin/bash
# Phase 3 finish: crop each trained gsplat block to its block region, concat into merged.ply,
# then eval on the 21 INRIA val images (raw frame, no-normalize) for an apples-to-apples gate
# vs vanilla coarse (24.96) and INRIA CityGS (26.05). Run after run_blocks_oracle.sh completes.
set -o pipefail
WORK_DIR="/work/pi_rsitaram_umass_edu/tungi"
LSTVGS="$WORK_DIR/lstvgs"
CONDA_ENV="$WORK_DIR/conda/envs/gsplat"
CUDA_DIR="$WORK_DIR/cuda-13.0"
OUT="$LSTVGS/results/rubble_citygs_gsplat_oracle"

unset CC CXX
export CUDAHOSTCXX="$CONDA_ENV/bin/g++"
export CUDA_HOME="$CUDA_DIR"
export PATH="$CONDA_ENV/bin:$CUDA_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_ENV/lib:$CUDA_DIR/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$LSTVGS/gsplat:${PYTHONPATH:-}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
cd "$LSTVGS"

echo "=== crop + merge $(date) ==="
python citygs/phase3_crop_merge.py \
    --blocks-dir "$OUT/blocks" --step 29999 \
    --block-dim 3 1 3 --aabb -50 -100 -135 50 300 -5 \
    --out "$OUT/merged.ply" || exit 1

echo "=== eval merged on 21 val views (raw frame) $(date) ==="
python citygs/eval_citygs.py \
    --ply-path "$OUT/merged.ply" \
    --data-dir "$OUT/data_val" --data-factor 4 \
    --test-every 1 --no-normalize \
    --output-dir "$OUT/eval_val" --lpips-net vgg || exit 1

echo "=== GATE: gsplat CityGS vs vanilla 24.96 / INRIA 26.05 / paper 25.77 ==="
python - <<'PY'
import json
r=json.load(open("/work/pi_rsitaram_umass_edu/tungi/lstvgs/results/rubble_citygs_gsplat_oracle/eval_val/metrics.json"))
print(f"gsplat CityGS merged: PSNR {r['psnr']:.3f} SSIM {r['ssim']:.4f} LPIPS {r['lpips']:.4f}  (N={r['n_gaussians']})")
print(f"  vs vanilla coarse 24.96 / 0.774 / 0.284 ; INRIA CityGS 26.05 / 0.813 / 0.232")
print("  GATE PASS" if r['psnr']>24.96 else "  GATE MISS (<= vanilla) -> debug LR/scene_scale")
PY
echo "=== finish_oracle done $(date) ==="
