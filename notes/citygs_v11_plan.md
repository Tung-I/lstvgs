# CityGS v11: Block Finetuning with Background Context (2080Ti-compatible)

## Context

v10 merged result was 20.46 dB — worse than the 24.91 dB coarse model.
Root cause: each block was initialized from a spatially-pruned PLY with **zero background Gaussians**. Cameras assigned to a block see the full scene, so the optimizer distorted each block's Gaussians to cover missing regions. After merge, adjacent blocks' distorted Gaussians conflicted.

**v11 fix:** Initialize each block from its spatial region Gaussians **plus** ~500K randomly-sampled background Gaussians from the rest of the scene. This gives rendering context without loading the full 8.13M GS (which won't fit in 2080Ti's 11GB VRAM). At merge time, strip background GS with strict spatial pruning.

GPU constraint: **2080Ti, 11GB VRAM**. The full coarse PLY (8.13M GS, 1.8GB on disk, ~5GB+ in optimizer) is too large. Per-block budget of ~2–2.5M GS is safe.

---

## Step-by-step Plan

### Step 0: Save this plan to `notes/` ✓ (done)

---
## Overall Progress (last updated 2026-06-27)

| Version | PSNR | SSIM | LPIPS | Status |
|---------|------|------|-------|--------|
| Coarse v9 (baseline) | 24.91 | 0.768 | 0.252 | Done |
| v10 merged | 20.46 | 0.694 | — | Failed (no background context) |
| v11 merged | 21.97 | 0.723 | 0.321 | Done — still below coarse |
| **v12** | — | — | — | **Training (blocks 1-7 in progress, 2026-06-27)** |
| **v13** | — | — | — | **Scripts ready** (`run_blocks_v13.sh`, `merge_eval_v13.sh`) |
| Paper target | 25.77 | 0.813 | 0.228 | — |

**v12:** Full coarse PLY + opacity reset disabled + 5% merge margin.
**v13:** Same + reduced LR (6.4e-5/4e-3) + densification enabled + SH degree 2 + **0% merge margin** (eliminates double-counting).
**Next if v13 fails:** Run official CityGaussian code directly to verify paper claims.
**Resume v12:** Job 61206288 running. After completion: `bash citygs/scripts/merge_eval_v12.sh` → then `bash citygs/scripts/run_blocks_v13.sh` → `bash citygs/scripts/merge_eval_v13.sh`

---

### Step 1: Modify `citygs/prepare_block_ply.py` — add background sampling

Add `--background-count INT` (default 500000): number of background GS to sample from
outside the block region. Logic after existing spatial masking:

```python
bg_raw = raw[~mask]
if args.background_count > 0 and len(bg_raw) > 0:
    n_bg = min(args.background_count, len(bg_raw))
    idx = rng.choice(len(bg_raw), n_bg, replace=False)
    combined = np.concatenate([block_raw, bg_raw[idx]], axis=0)
else:
    combined = block_raw
```

Use `rng = np.random.default_rng(seed=42)` for reproducibility.

### Step 2: Create `citygs/scripts/setup_blocks_v11.sh`

Creates `lstvgs/results/rubble_citygs_blocks_v11/block_NNN/` for blocks 0,1,2,3,4,5,7.
Copies `block_info.json` from `lctvgs/results/rubble_citygs_blocks_v10/block_NNN/`.
Runs `prepare_block_ply.py --margin 0.0 --background-count 500000`.

Expected init GS counts (0% margin block + 500K background):
- Block 0: 892K + 500K = 1.39M
- Block 1: 1.97M + 500K = 2.47M
- Block 2: 1.66M + 500K = 2.16M
- Block 3: 366K + 500K = 866K
- Block 4: 1.72M + 500K = 2.22M
- Block 5: 916K + 500K = 1.42M
- Block 7: 247K + 500K = 747K

### Step 3: Create `citygs/scripts/run_blocks_v11.sh`

Sequential loop over blocks 0,1,2,3,4,5,7. Skips blocks with existing `point_cloud_29999.ply`.

Key flags vs v10:
- `--init-ply-path`: block region + 500K background GS (not per-block pruned PLY)
- `--strategy.reset-every 6000` (opacity reset enabled)
- `--strategy.refine-start-iter 100000` (densification disabled — 2080Ti memory safety)
- `--max-steps 30000`

Submit via:
```bash
srun -t 480 -c 6 -p gpu -G 1 --mem 32G --constraint=[2080ti] --pty bash
cd /work/pi_rsitaram_umass_edu/tungi/lstvgs
bash citygs/scripts/run_blocks_v11.sh
```

Time estimate on 2080Ti (~4–8 it/s with 1–2.5M GS):
- Total ~700–900 min → need 2 srun sessions (480 min each)
- Skip-if-done logic handles resumption across sessions

### Step 4: Job monitoring and fallback to L40S

**a) GPU allocation failure** (srun hangs >10 min):
```bash
scancel <jobid>
# Then run in this session (L40S):
bash citygs/scripts/run_blocks_v11.sh
```

**b) OOM on 2080Ti** (`torch.cuda.OutOfMemoryError`, no PLY saved):
Run only the failed blocks in this current L40S session using the same script
(skip-if-done skips already-completed blocks automatically).

### Step 5: Create `citygs/scripts/merge_eval_v11.sh`

```bash
python citygs/merge_citygs.py \
    --block-results-dir results/rubble_citygs_blocks_v11 \
    --block-dim 3 1 3 --step 30000 \
    --partition-dir /work/pi_rsitaram_umass_edu/tungi/lctvgs/results/rubble_citygs_coarse_v9/partition \
    --output results/rubble_citygs_merged_v11/merged.ply \
    --prune-opacity 0.005 --spatial-margin 0.02

python citygs/eval_citygs.py \
    --ply-path results/rubble_citygs_merged_v11/merged.ply \
    --data-dir /work/pi_rsitaram_umass_edu/tungi/datasets/rubble_citygs \
    --data-factor 4 --test-every 83 \
    --output-dir results/rubble_citygs_merged_v11
```

Target: PSNR > 24.91 dB (beat coarse); paper target 25.77 / SSIM 0.813 / LPIPS 0.228.

### Step 6: Iterative quality improvement if gap remains

If merged PSNR < paper target, apply fixes in priority order (each as a new version):

| Priority | Fix | Impl notes |
|----------|-----|------------|
| 1 | 60K steps | Feasible on L40S if 2080Ti too slow |
| 2 | Enable densification (scaled by `sqrt(n_cams/300)`) | Watch memory on 2080Ti |
| 3 | Visibility-based merge pruning (min 32 train images) | Modify `merge_citygs.py` |
| 4 | SH degree 2 instead of 3 | Minor; match official exactly |
| 5 | Normal/depth regularization | Needs depth maps; most complex |

Official repo: https://github.com/Linketic/CityGaussian (V1 branch)

Stop when PSNR ≥ 25.5 dB or all major differences are resolved.
Update `notes/resume_block_finetuning.md` with each version's results.

---

## Key Paths

- Coarse PLY:     `lctvgs/results/rubble_citygs_coarse_v9/ply/point_cloud_29999.ply`
- Partition dir:  `lctvgs/results/rubble_citygs_coarse_v9/partition/`
- v10 block dirs: `lctvgs/results/rubble_citygs_blocks_v10/block_NNN/` (block_info.json source)
- v11 block dirs: `lstvgs/results/rubble_citygs_blocks_v11/block_NNN/`
- v11 merged:     `lstvgs/results/rubble_citygs_merged_v11/`
- Dataset:        `/work/pi_rsitaram_umass_edu/tungi/datasets/rubble_citygs`
