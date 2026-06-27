# CityGaussian (CityGS) Reproduction — Implementation Notes

Reference: "CityGaussian: Real-time High-quality Large-Scale Scene Rendering with Gaussians"  
           Liu et al., ECCV 2024  
Official repo: https://github.com/Linketic/CityGaussian (V1 branch for Mill19/UrbanScene3D)  
Our impl: gsplat as the 3DGS backend (`gsplat/examples/simple_trainer.py`)

---

## Overview

CityGS is a divide-and-conquer 3DGS method for large outdoor scenes. The pipeline has
three stages:

1. **Coarse training** — train a full-scene 3DGS model with SFM initialization
2. **Block partition + finetuning** — partition scene into spatial blocks, finetune each
3. **Merge + eval** — spatially prune and concatenate block models, evaluate

Paper targets (Rubble scene, Mill19 benchmark):
  PSNR: 25.77 dB, SSIM: 0.813, LPIPS: 0.228

Our coarse model (30k steps, SFM init): PSNR 24.91, SSIM 0.768, LPIPS 0.252

---

## Dataset Setup

CityGS uses a specific naming convention and COLMAP format that requires careful setup.

**Source data:**
- Images: `/work/.../datasets/rubble-pixsfm/train/rgbs/` (1657 images, named `000001.jpg`...)
           `/work/.../datasets/rubble-pixsfm/val/rgbs/`   (21 images,   named `000001.jpg`...)
- COLMAP: `/work/.../datasets/cityGSdata/colmap_results/rubble/train/sparse/0/`
          (cameras.bin, images.bin with 1657 entries, points3D.bin with 1,694,315 points)

**Assembled dataset: `/work/.../datasets/rubble_citygs/`**

```
rubble_citygs/
  images/               # 1678 symlinks (000000.jpg ... 001677.jpg)
                        # sorted order: train images first, then val
  images_4/             # symlink → images/  (triggers Parser's resize path)
  images_4_png/         # 1678 PNGs at 1152×864 (4× downsampled)
                        # created by Parser's _resize_image_folder on first run
  sparse/0/
    cameras.bin         # copied from cityGSdata/colmap_results/rubble/train/sparse/0/
    images.bin          # merged: 1657 train (original IDs) + 21 val (new IDs)
    points3D.bin        # copied from cityGSdata COLMAP (1,694,315 SFM points, 155 MB)
```

**Critical: val split alignment**  
With 1678 total images sorted by name, `--test-every 83` places test images at
positions 0, 83, 166, ..., 1660 — exactly the 21 CityGS/MegaNeRF val frames.

**Critical: image IDs in images.bin**  
Train images keep their original COLMAP image IDs. Val images get new IDs starting at
`max_train_id + 1`. Do NOT reassign train image IDs or pycolmap's cross-validation of
points3D.bin track references will fail.

**Setup script:** `setup_rubble_citygs.py`  
Reads images.bin and points3D.bin via raw struct binary I/O (not pycolmap API, which
has breaking changes in v4.0.4 where `cam_from_world` is read-only).

**Known issue: empty PNG bug**  
`images_4_png/000215.png` was created as a 0-byte file by Parser's `_resize_image_folder`
(silent failure on one JPEG). This causes a `ValueError: Could not find a backend`
crash in the DataLoader around step 1500. Fixed by manually regenerating with PIL:
```python
from PIL import Image
img = Image.open('images/000215.jpg')
img.resize((1152, 864), Image.LANCZOS).save('images_4_png/000215.png')
```

---

## Stage 1: Coarse Training

**Script:** `gsplat/examples/simple_trainer.py`  
**Result dir:** `results/rubble_citygs_coarse_v9/`

```bash
python simple_trainer.py default \
    --disable-viewer \
    --data-dir $DATA_DIR --data-factor 4 --data-type colmap \
    --result-dir $COARSE_DIR \
    --test-every 83 --normalize-world-space \
    --max-steps 30000 \
    --eval-steps 7000 30000 --save-steps 7000 30000 \
    --save-ply --ply-steps 7000 30000 \
    --batch-size 1 --init-type sfm \
    --sh-degree 3 --sh-degree-interval 1000 --ssim-lambda 0.2 \
    --strategy.refine-start-iter 500 --strategy.refine-stop-iter 15000 \
    --strategy.refine-every 100 --strategy.reset-every 3000 \
    --strategy.grow-grad2d 0.0002 --strategy.grow-scale3d 0.01 \
    --strategy.prune-opa 0.005 \
    --packed --lpips-net alex --no-antialiased --no-random-bkgd
```

Results:
  Step 7k:  PSNR 20.34, SSIM 0.550, LPIPS 0.578, 4.36M GS
  Step 30k: PSNR 24.91, SSIM 0.768, LPIPS 0.252, 8.13M GS

Speed: ~25 it/s initially, slows to ~5-7 it/s as GS count grows (densification).
Total time: ~45 minutes on L40S.

---

## Stage 2: Scene Partition

**Script:** `citygs/partition_citygs.py`  
**Result dir:** `results/rubble_citygs_coarse_v9/partition/`

```bash
python citygs/partition_citygs.py \
    --ply-path results/rubble_citygs_coarse_v9/ply/point_cloud_29999.ply \
    --data-dir $DATA_DIR --data-factor 4 --test-every 83 \
    --block-dim 3 1 3 --ssim-threshold 0.12 \
    --output-dir results/rubble_citygs_coarse_v9/partition
```

Outputs: `camera_mask.npy` (shape [1657, 9], bool), `aabb.npy`, `partition_info.json`

Block indexing (3×1×3 grid, x-fastest convention):
  block_z = block_id // (dim_x * dim_y)
  block_y = (block_id % (dim_x * dim_y)) // dim_x
  block_x = (block_id % (dim_x * dim_y)) % dim_x

Cameras per block (rubble): [332, 977, 561, 244, 832, 345, 0, 47, 1]
  → Block 6 (0 cameras) and Block 8 (1 camera) are skipped.

**Must pass `--test-every 83`** — default is 8, causing IndexError because
train_indices size mismatches the camera_mask row count (1657 vs ~1468).

---

## Stage 3: Block Data Preparation

**Script:** `citygs/prepare_block_data.py`

```bash
python citygs/prepare_block_data.py \
    --data-dir $DATA_DIR --data-factor 4 --test-every 83 \
    --partition-dir results/rubble_citygs_coarse_v9/partition \
    --output-dir results/rubble_citygs_blocks_v9
```

Creates `block_NNN/block_info.json` per block:
  { "block_id", "n_cameras", "global_indices", "image_names", ... }

`global_indices` are indices into the full 1678-image list (not the train subset).
The trainer's `--cam-indices-file` flag reads this and filters the training dataset.

---

## Stage 3b: Per-Block PLY Initialization

**Script:** `citygs/prepare_block_ply.py`  (written by us, not in original CityGS)

The coarse PLY has 8.13M Gaussians. Training each block from the full coarse PLY
causes two problems:
  1. Slow speed (3-4 it/s at 8M GS → ~2h per block × 7 blocks = ~14h total)
  2. Uncontrolled densification: small camera subsets → high per-GS gradients →
     block explodes to 22M+ GS within 15k steps

Fix: spatially prune coarse PLY to each block's region (+ 5% margin).

```bash
python citygs/prepare_block_ply.py \
    --coarse-ply results/rubble_citygs_coarse_v9/ply/point_cloud_29999.ply \
    --partition-dir results/rubble_citygs_coarse_v9/partition \
    --output-dir results/rubble_citygs_blocks_v9 \
    --margin 0.05
```

Pruning is done in the contracted [0,1]^3 space (via `contract_to_unisphere`).
Margin of 0.05 means each block gets its grid cell ± 5% of the full scene range.

Resulting GS counts per block:
  Block 0: 1.14M, Block 1: 3.01M, Block 2: 2.38M, Block 3: 0.76M,
  Block 4: 2.87M, Block 5: 1.82M, Block 7: 0.44M

---

## Stage 4: Block Finetuning

**Critical flags:**
- `--init-type ply --init-ply-path block_NNN/init.ply` — start from spatially pruned PLY
- `--cam-indices-file block_NNN/block_info.json` — restrict to block's cameras
- `--strategy.refine-start-iter 100000` — **DISABLE densification** (> max-steps=30000)
- `--strategy.reset-every 100000` — disable opacity reset

Densification MUST be disabled for block finetuning. With only ~300–1000 cameras
per block, the per-Gaussian gradient is much higher than full-scene training,
causing uncontrolled GS growth even from spatially pruned PLY files.

```bash
python simple_trainer.py default \
    --disable-viewer \
    --data-dir $DATA_DIR --data-factor 4 --data-type colmap \
    --result-dir results/rubble_citygs_blocks_v9/block_NNN \
    --test-every 83 --normalize-world-space \
    --max-steps 30000 --eval-steps 30000 --save-steps 30000 \
    --save-ply --ply-steps 30000 --batch-size 1 \
    --init-type ply --init-ply-path block_NNN/init.ply \
    --cam-indices-file block_NNN/block_info.json \
    --sh-degree 3 --sh-degree-interval 1000 --ssim-lambda 0.2 \
    --strategy.refine-start-iter 100000 \
    --strategy.reset-every 100000 --strategy.prune-opa 0.005 \
    --packed --lpips-net alex --no-antialiased --no-random-bkgd
```

Speed with spatially pruned init PLY: 15–36 it/s (vs 3-4 it/s from full coarse PLY).
Time per block: 8–37 min. Total for 7 blocks: ~2 hours.

**Resume script:** `run_blocks_resume.sh` — skips blocks with existing `point_cloud_29999.ply`.

Block eval sanity check: use `citygs/eval_block.py` to render each block on its own
assigned cameras. NOTE: cannot run concurrently with block training (GPU memory conflict).
The trainer's built-in `val_step29999.json` uses GLOBAL val cameras (21 views spread
across the full scene), so individual block PSNRs will look low (11–20 dB) — this is
expected and NOT a bug; each block renders most global val views as empty.

---

## Stage 5: Merge

**Script:** `citygs/merge_citygs.py`

```bash
python citygs/merge_citygs.py \
    --block-results-dir results/rubble_citygs_blocks_v9 \
    --block-dim 3 1 3 \
    --partition-dir results/rubble_citygs_coarse_v9/partition \
    --output results/rubble_citygs_merged_v9/merged.ply
```

Applies opacity pruning (`--prune-opacity 0.005`) and spatial margin pruning
(`--spatial-margin 0.5`) per block before concatenating.

---

## Stage 6: Evaluation

**Script:** `citygs/eval_citygs.py`

```bash
python citygs/eval_citygs.py \
    --ply-path results/rubble_citygs_merged_v9/merged.ply \
    --data-dir $DATA_DIR --data-factor 4 --test-every 83 \
    --output-dir results/rubble_citygs_merged_v9
```

Renders all 21 val views, saves `metrics.json` with PSNR/SSIM/LPIPS.

---

## Pipeline Scripts

| Script | Purpose |
|--------|---------|
| `setup_rubble_citygs.py` | Assemble rubble_citygs/ dataset from pixsfm + CityGS COLMAP |
| `run_rubble_citygs_v9.sh` | Full pipeline (sequential, interactive session) |
| `run_blocks_resume.sh` | Resume block finetuning, skipping completed blocks |
| `citygs/partition_citygs.py` | Stage 2: SSIM-based block partition |
| `citygs/prepare_block_data.py` | Stage 3: write block_info.json per block |
| `citygs/prepare_block_ply.py` | Stage 3b: spatial pruning of coarse PLY per block |
| `citygs/merge_citygs.py` | Stage 5: opacity + spatial prune and merge |
| `citygs/eval_citygs.py` | Stage 6: render test views, compute metrics |
| `citygs/eval_block.py` | Per-block sanity eval on own cameras (post-training) |

---

## Known Issues Summary

| Issue | Symptom | Fix |
|-------|---------|-----|
| Empty points3D.txt | ~14 dB PSNR (random init) | Use cityGSdata COLMAP (1.69M pts) |
| images_4_png/000215.png empty | ValueError crash at step ~1500 | Manually resize with PIL |
| prepare_block_data.py wrong test_every | IndexError (list index out of range) | Pass `--test-every 83` |
| Densification in block finetuning | GS explodes to 22M+, 3-4 it/s | `--strategy.refine-start-iter 100000` |
| Full coarse PLY for block init | Too slow (14h total) | Use `prepare_block_ply.py` with margin 0.05 |
| Opacity reset in block finetuning | Destroys converged structure | `--strategy.reset-every 100000` |
| pycolmap v4.0.4 API | cam_from_world is read-only | Use struct binary I/O directly |
