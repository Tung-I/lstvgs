# Block Finetuning — Rubble CityGS

---

## v10 Results (2026-06-26) — FAILED

**Merged v10: 20.46 dB / 0.694 SSIM** — worse than coarse model (24.91 dB).

### Per-block v10 val PSNRs

| Block | Init GS | Cams | Val PSNR | Val SSIM |
|-------|---------|------|----------|----------|
| 0 | 892K | 332 | 12.92 | 0.281 |
| 1 | 1.97M | 977 | 20.51 | 0.556 |
| 2 | 1.66M | 561 | 15.05 | 0.398 |
| 3 | 366K | 244 | 12.86 | 0.284 |
| 4 | 1.72M | 832 | 18.58 | 0.501 |
| 5 | 916K | 345 | 12.97 | 0.343 |
| 7 | 247K | 47 | 10.39 | 0.202 |

Note: val PSNR is across all 21 global val views (not just block-local views). Low PSNR on small blocks is expected — what matters is the merged result.

### Root Cause

**No rendering context for block training.** v10 used `--margin 0.0` (no spatial overlap at init), so each block started with only its own Gaussians (~0.9–2M). But cameras assigned to a block see the *full scene*. Without background Gaussians present during rendering, the optimizer distorts the block Gaussians trying to cover missing regions. After merge, these distorted Gaussians conflict with neighboring blocks → 4+ dB degradation versus the coarse model.

---

## Key Differences from Official CityGaussian

Identified by comparing against https://github.com/Linketic/CityGaussian.

| Aspect | Our impl (v10) | Official CityGS |
|--------|---------------|-----------------|
| **Block init** | Spatially pruned PLY (block region only, 0% margin) | Full coarse checkpoint; block Gaussians extracted at merge |
| **Block training steps** | 30K | **60K** (plus extra epochs scaled by camera count) |
| **Opacity reset** | Disabled (`reset-every 100000`) | Every **6000 steps** (scaled by block size) |
| **Densification interval** | Default 100 steps | 200 steps, scaled by `sqrt(n_cams / 300)` |
| **Densify until** | Step 15000 | Step 30000, scaled |
| **SH degree** | 3 | 2 |
| **Merge pruning** | Spatial margin (10%) | Visibility-based (min 32 training images) |
| **Normal/depth loss** | None | Normal regularization weight 0.0125 + depth loss |

The first two rows (background context + 60K steps) are the primary quality drivers.

---

## v11 Fix

**Core change:** initialize each block from the **full coarse PLY** (8.13M Gaussians) instead of a spatially pruned subset. This gives every block full rendering background during training. At merge time, use **strict 0% spatial pruning** to extract only the block's own Gaussians from each trained model.

Additional changes vs v10:

- `--max-steps 60000` (2× more training)
- Opacity reset enabled: `--strategy.reset-every 6000`
- Densification on (default), `--strategy.refine-stop-iter 30000`
- Merge margin 0% (extract exact block region only)

**Expected tradeoff:** training is slower (~3–4 it/s with 8M Gaussians vs ~20 it/s with 1M), but each block produces correct Gaussians that sum cleanly at merge.

Script: `citygs/scripts/run_blocks_v11.sh`  
Results: `results/rubble_citygs_blocks_v11/`, `results/rubble_citygs_merged_v11/`

---

## Status as of 2026-06-25 (v9 history)

Block finetuning for `results/rubble_citygs_blocks_v9/` was interrupted partway through.

### Completed (PLY saved)
| Block | Cameras | Init GS | Final loss | Global-val PSNR | PLY path |
|-------|---------|---------|-----------|----------------|---------|
| 0 | 332 | 1.14M | 0.068 | 12.19 dB | `block_000/ply/point_cloud_29999.ply` |
| 1 | 977 | 3.01M | 0.193 | 19.72 dB | `block_001/ply/point_cloud_29999.ply` |
| 2 | 561 | 2.38M | 0.090 | 14.50 dB | `block_002/ply/point_cloud_29999.ply` |
| 3 | 244 | 0.76M | 0.078 | 11.76 dB | `block_003/ply/point_cloud_29999.ply` |
| 4 | 832 | 2.87M | ≈0.08 | — | `block_004/ply/point_cloud_29999.ply` |

Note: global-val PSNR (11–20 dB) reflects only ~1/9 of the 21 val views being visible per block.
This is **expected** — not a bug. The merged model is what matters for final quality.

### Not completed
| Block | Status |
|-------|--------|
| 5 | Interrupted at step ~4,834/30,000 — **no PLY saved, must re-run** |
| 6 | Skipped (0 cameras) |
| 7 | Not started (47 cameras) |
| 8 | Skipped (1 camera) |

---

## How to Resume

The resume script skips blocks that already have `point_cloud_29999.ply`:

```bash
cd /work/pi_rsitaram_umass_edu/tungi/lctvgs

WORK_DIR="/work/pi_rsitaram_umass_edu/tungi"
CONDA_ENV="$WORK_DIR/conda/envs/gsplat"
CUDA_DIR="$WORK_DIR/cuda-13.0"
unset CC CXX
export CUDAHOSTCXX="$CONDA_ENV/bin/g++"
export CUDA_HOME="$CUDA_DIR"
export PATH="$CONDA_ENV/bin:$CUDA_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_ENV/lib:$CUDA_DIR/lib64:$LD_LIBRARY_PATH"
export PYTHONPATH="$WORK_DIR/lctvgs/gsplat:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

bash run_blocks_resume.sh
```

Expected behavior:
- Blocks 0–4: prints "already done, skipping"
- Block 5: runs from scratch (~30 min, 1.82M init GS)
- Block 7: runs from scratch (~8 min, 0.44M init GS)
- Total remaining time: ~40 minutes

---

## After Blocks Complete: Stage 5 + 6

### Stage 5: Merge
```bash
python citygs/merge_citygs.py \
    --block-results-dir results/rubble_citygs_blocks_v9 \
    --block-dim 3 1 3 \
    --partition-dir results/rubble_citygs_coarse_v9/partition \
    --output results/rubble_citygs_merged_v9/merged.ply
```

### Stage 6: Evaluate
```bash
python citygs/eval_citygs.py \
    --ply-path results/rubble_citygs_merged_v9/merged.ply \
    --data-dir /work/pi_rsitaram_umass_edu/tungi/datasets/rubble_citygs \
    --data-factor 4 --test-every 83 \
    --output-dir results/rubble_citygs_merged_v9
```

Target: PSNR ~25 dB (paper: 25.77 dB / SSIM 0.813 / LPIPS 0.228)

---

## Optional: Per-block sanity eval (after finetuning, before merge)

Run `citygs/eval_block.py` on each completed block to verify quality on its own cameras.
Do NOT run while any block trainer is occupying the GPU.

```bash
for BLOCK in 000 001 002 003 004 005 007; do
    python citygs/eval_block.py \
        --block-dir results/rubble_citygs_blocks_v9/block_$BLOCK \
        --data-dir /work/pi_rsitaram_umass_edu/tungi/datasets/rubble_citygs
done
```

Results will be saved to `block_NNN/stats/block_eval.json`.

---

## Key paths

- Coarse PLY:    `results/rubble_citygs_coarse_v9/ply/point_cloud_29999.ply`
- Partition dir: `results/rubble_citygs_coarse_v9/partition/`
- Block results: `results/rubble_citygs_blocks_v9/block_NNN/`
- Dataset:       `/work/pi_rsitaram_umass_edu/tungi/datasets/rubble_citygs/`
- Resume script: `run_blocks_resume.sh`

## Full implementation docs

See `documentation/citygs_implementation.md` for the complete pipeline reference.
