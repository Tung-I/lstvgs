#!/bin/bash
# Lapis-CityGS finish: merge each block's per-LoD layer into global merged_lodL.ply, then eval
# every level on the same 21 INRIA val views as the flat CityGS gate (raw frame, no-normalize,
# vgg LPIPS) to produce a scene-level rate-distortion curve vs flat CityGS (25.78/0.832/0.276).
# Run after enough blocks (covering the val cameras) finish run_lapis_block.sh.
# Usage: bash finish_lapis_lod.sh [LAYERS...]   (default: 0 1 2 3)
set -o pipefail
WORK_DIR="/work/pi_rsitaram_umass_edu/tungi"
LSTVGS="$WORK_DIR/lstvgs"
CONDA_ENV="$WORK_DIR/conda/envs/gsplat"
CUDA_DIR="$WORK_DIR/cuda-13.0"
RESULTS="$LSTVGS/results"
OUT="$RESULTS/rubble_lapis_lod"
DATA_VAL="$RESULTS/rubble_citygs_gsplat_oracle/data_val"
LAYERS=("${@:-0 1 2 3}")

unset CC CXX
export CUDAHOSTCXX="$CONDA_ENV/bin/g++"
export CUDA_HOME="$CUDA_DIR"
export PATH="$CONDA_ENV/bin:$CUDA_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_ENV/lib:$CUDA_DIR/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$LSTVGS/gsplat:${PYTHONPATH:-}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
cd "$LSTVGS"

echo "=== merge per-LoD $(date) ==="
python citygs/merge_lapis_lod.py \
    --blocks-dir "$RESULTS" --n-blocks 9 --layers ${LAYERS[@]} \
    --block-dim 3 1 3 --aabb -50 -100 -135 50 300 -5 \
    --out-dir "$OUT" || exit 1

for L in ${LAYERS[@]}; do
    PLY="$OUT/merged_lod${L}.ply"
    [ -f "$PLY" ] || { echo "skip L$L (no $PLY)"; continue; }
    echo "=== eval LoD $L on 21 val views $(date) ==="
    python citygs/eval_citygs.py \
        --ply-path "$PLY" \
        --data-dir "$DATA_VAL" --data-factor 4 \
        --test-every 1 --no-normalize \
        --output-dir "$OUT/eval_lod${L}" --lpips-net vgg || exit 1
done

echo "=== Lapis-CityGS scene-level LoD RD curve (vs flat CityGS 25.78/0.832/0.276 @ 9.08M) ==="
OUT="$OUT" python - <<'PY'
import json, os
out = os.environ["OUT"]
print(f"{'LoD':>3} {'#GS':>12} {'PSNR':>7} {'SSIM':>7} {'LPIPS':>7}")
print("-" * 42)
for L in range(4):
    mp = os.path.join(out, f"eval_lod{L}", "metrics.json")
    if not os.path.exists(mp):
        continue
    r = json.load(open(mp))
    print(f"{L:>3} {r['n_gaussians']:>12,} {r['psnr']:>7.3f} {r['ssim']:>7.4f} {r['lpips']:>7.4f}")
print("-" * 42)
print("flat CityGS  9,080,000  25.778  0.8318  0.2758")
PY
echo "=== finish_lapis_lod done $(date) ==="
