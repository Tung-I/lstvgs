# Block Finetuning — Rubble CityGS

---

## Version Summary

| Version | PSNR | SSIM | LPIPS | Status |
|---------|------|------|-------|--------|
| Coarse v9 (baseline) | 24.91 | 0.768 | 0.252 | Done |
| v10 merged | 20.46 | 0.694 | — | Failed (no background context) |
| v11 merged | 21.97 | 0.723 | 0.321 | Done — still below coarse |
| v12 merged | — | — | — | In progress (blocks 1-7 training, 2026-06-27) |
| v13 merged | — | — | — | Scripts ready; run after v12 |
| Paper target | 25.77 | 0.813 | 0.228 | CityGS V1 (ECCV 2024) |

**v12:** Full coarse PLY (8.13M) init + opacity reset disabled + densification disabled + 30K steps + 5% spatial margin merge.
**v13:** Same init + reduced LR (6.4e-5 pos, 4e-3 scale) + densification enabled + SH degree 2 + **0% spatial margin** (strict, no double-counting at block seams).

---

## Official CityGaussian Analysis (2026-06-27)

Reviewed https://github.com/Linketic/CityGaussian (main = V2; V1 branch for ECCV 2024 results).

### V2 block training (main branch) key parameters
- **Block init**: spatially-filtered coarse checkpoint (XY position only, strict bounds)
- **Camera assignment**: location-based (camera center in block XY bounds) + visibility-based (SSIM loss > 0.08 when rendering with/without block GS)
- **Total steps**: `30000 × scale_up + extra_epoch × max(n_cams, 300)` where `scale_up = max(n_cams/300, 1.0)`, `extra_epoch=30`
- **Densification interval**: `100 × scale_up`; opacity reset: `3000 × scale_up`
- **Position LR**: 0.000064 (40% of default 0.00016)
- **Scale LR**: 0.004 (80% of default 0.005)
- **Depth loss** (Depth Anything V2 monocular) + **normal regularization** (weight 0.0125)
- **Merge**: strict spatial pruning (0% margin, each GS in exactly one block by XY position)
- **Pre-block pruning**: 60% opacity-importance-based pruning of coarse model before block training

### V1 (ECCV 2024) key differences from V2
- 3DGS backbone (not 2DGS)
- 60K base training steps (not 30K)
- `reset_every = 6000 × sqrt(n_cams/300)`; `densify_until = 30000 × sqrt(n_cams/300)`
- Merge: visibility-based (keep GS seen by ≥ 32 training images)

### Why our implementations fall short
1. **Depth/normal regularization**: requires Depth Anything V2 monocular depths — complex to set up
2. **Camera count scaling**: large blocks need proportionally more steps (977 cams → 127K steps in V2)
3. **30K steps insufficient for large blocks** at official reset intervals
4. **Our camera assignment** (SSIM threshold 0.12 from partition) assigns too many cameras per block (block 1 = 977 of 1657 = 59%)

---

## v11 Results (2026-06-27) — PARTIAL IMPROVEMENT

**Merged v11: 21.97 dB / 0.723 SSIM / 0.321 LPIPS** — better than v10 but still worse than coarse (24.91 dB).

### Settings (vs v10)
- Init PLY: block region GS + 500K random background GS (provided some rendering context)
- `--strategy.reset-every 6000` (opacity reset enabled)
- `--strategy.refine-start-iter 100000` (densification disabled)
- `--max-steps 30000`
- Merge: `--spatial-margin 0.02`

### Root causes of remaining gap

**1. 500K background too sparse.** 500K / 8.13M = only 6% of scene GS sampled randomly.
Coverage of background regions seen by block cameras was insufficient, causing continued
(if milder) distortion of block Gaussians. Fix: use full coarse PLY (8.13M GS) as init.

**2. Opacity reset kills GS.** With `reset-every 6000` and 30K steps, 5 resets occur.
Last reset at step 24K leaves only 6K recovery steps → many GS remain low-opacity →
stripped at merge. Merged model has only 3.3M GS (vs 8.13M in coarse). Fix: disable
opacity reset (`reset-every 100000`).

---

## v12 Plan

**Core changes vs v11:**
- Init PLY: **full coarse PLY** (8.13M GS) — complete rendering context, no distortion
- `--strategy.reset-every 100000` — disable opacity reset to preserve coarse GS quality
- `--strategy.refine-start-iter 100000` — keep densification disabled (memory safety)
- `--max-steps 30000` — unchanged
- Merge: `--spatial-margin 0.05` — slightly larger to avoid losing boundary GS

**Memory:** 8.13M GS on L40S (46GB) is feasible. Estimate 5–7 it/s → ~80 min/block → ~560 min total → 2 srun sessions of 480 min.

**Scripts to create:**
- `citygs/scripts/run_blocks_v12.sh` — same structure as v11, but init-ply-path = full coarse PLY
- `citygs/scripts/merge_eval_v12.sh` — merge with spatial-margin 0.05, step 30000

**Key paths:**
- Coarse PLY: `/work/pi_rsitaram_umass_edu/tungi/lctvgs/results/rubble_citygs_coarse_v9/ply/point_cloud_29999.ply`
- Partition:  `/work/pi_rsitaram_umass_edu/tungi/lctvgs/results/rubble_citygs_coarse_v9/partition/`
- Block info: reuse `lstvgs/results/rubble_citygs_blocks_v11/block_NNN/block_info.json`
- v12 results: `lstvgs/results/rubble_citygs_blocks_v12/`, `lstvgs/results/rubble_citygs_merged_v12/`

### Block completion status (as of 2026-06-27 06:31 UTC)

| Block | Cams | Status | Notes |
|-------|------|--------|-------|
| 0 | 332 | ✓ DONE (89 min) | from previous session |
| 1 | 977 | IN PROGRESS (started 06:17 UTC) | ETA 07:42 UTC |
| 2 | 561 | NOT DONE | — |
| 3 | 244 | NOT DONE | — |
| 4 | 832 | NOT DONE | — |
| 5 | 345 | NOT DONE | — |
| 7 | 47  | NOT DONE | May get cut by session expire (14:03 UTC) |

Session expires at 14:03 UTC. Blocks 0-5 will finish by ~13:31 UTC. Block 7 will get ~32 min of 89 min → killed → re-run in new session. `run_blocks_v12.sh` is running in background (job 61206288).

**After v12 blocks done:**
```bash
bash citygs/scripts/merge_eval_v12.sh
```

**Then run v13:**
```bash
bash citygs/scripts/run_blocks_v13.sh   # reduced LR, densification ON, SH degree 2
bash citygs/scripts/merge_eval_v13.sh   # 0% spatial margin
```

---

## v13 Plan — Paper-Faithful Implementation (2026-06-27)

**Key paper findings (CityGS V1, ECCV 2024):**
- **No depth/normal regularization** — that is V2 only (different codebase, different paper)
- **Block finetuning = 30K steps** — the "60K" was for the 3DGS† baseline, NOT CityGS blocks
- **LR reduction for ALL block training**: position −60% (→ 6.4e-5), scale −20% (→ 4e-3)
- **Block init = spatially-filtered coarse GS** (only Gaussians within block's spatial bounds)
- **Standard 3DGS densification + opacity reset (every 3K)** — paper doesn't override these

**Why previous versions failed:**
- v10: spatially-filtered init ✓, but **densification disabled** → block GS stretched to cover background → distortion
- v11: 500K background GS, opacity reset every 6K → reset killed GS (3.3M survived vs 8.13M)
- v12: full coarse PLY (wrong per paper) + no reset + no densification → doesn't match paper

**v13 fixes:**
- Spatially-filtered init (block region only, `--margin 0.0 --background-count 0`)
- **Densification enabled** (standard: start 500, stop 15K, every 100) — new GS created for background during training, pruned at merge
- Opacity reset every 3000 (standard 3DGS)
- `--means-lr 0.000064` (40% of default), `--scales-lr 0.004` (80%)
- `--max-steps 30000`
- Merge: `--spatial-margin 0.0` (strict, no double-counting)

**Expected speed:** ~1-2M GS per block (not 8.13M) → ~20 it/s → ~25 min/block → ~3h total (fits one session)

**Scripts:**
- `citygs/scripts/setup_blocks_v13.sh` — generates per-block init.ply (run once, CPU only)
- `citygs/scripts/run_blocks_v13.sh` — block training
- `citygs/scripts/merge_eval_v13.sh` — merge (0% margin) + eval

**Key paths:**
- Block results: `lstvgs/results/rubble_citygs_blocks_v13/`
- Merged: `lstvgs/results/rubble_citygs_merged_v13/`

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
