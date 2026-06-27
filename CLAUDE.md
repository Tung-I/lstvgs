# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goal

Implement and evaluate large-scale 3DGS baseline methods for stadium-scale scenes. The primary method here is CityGaussian (CityGS) using gsplat as the 3DGS backend. The primary benchmarks are Mill19 (rubble, building), UrbanScene3D (residence, sciart), and MatrixCity (aerial, street).

This repo (`lstvgs`) focuses on baseline reproductions. The sibling repo `lctvgs` (`/work/pi_rsitaram_umass_edu/tungi/lctvgs`) contains streaming/compression extensions (L3GS, LapisGS, EvoGS, SGSS, VQ-HEVC, etc.).

## Environment Setup

All training runs on SLURM with L40S GPUs. The conda environment is at `$WORK/conda/envs/gsplat`. Every SLURM script must set:

```bash
WORK_DIR="/work/pi_rsitaram_umass_edu/tungi"
CONDA_ENV="$WORK_DIR/conda/envs/gsplat"
CUDA_DIR="$WORK_DIR/cuda-13.0"
unset CC CXX
export CUDAHOSTCXX="$CONDA_ENV/bin/g++"
export CUDA_HOME="$CUDA_DIR"
export PATH="$CONDA_ENV/bin:$CUDA_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_ENV/lib:$CUDA_DIR/lib64:$LD_LIBRARY_PATH"
export PYTHONPATH="$WORK_DIR/lstvgs/gsplat:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Submit jobs with `sbatch <script>.sh`. Check status with `squeue -u tungichen_umass_edu`.

## Key Paths

- **Repo**: `/work/pi_rsitaram_umass_edu/tungi/lstvgs/`
- **gsplat submodule**: `gsplat/` — the training engine; `gsplat/examples/simple_trainer.py` is the main trainer
- **CityGS scripts**: `citygs/` — partition, merge, eval, block data prep
- **Datasets**: `/work/pi_rsitaram_umass_edu/tungi/datasets/` (symlinked as `datasets/`)
  - `rubble/`, `building/` — flat COLMAP format (images_4/, sparse/0/ with cameras.txt/images.txt/points3D.txt)
  - `rubble-pixsfm/`, `building-pixsfm/` — MegaNeRF split format (train/val with rgbs/ and metadata/)
  - `cityGSdata/colmap_results/{rubble,building,residence,sciart,matrix_city_aerial}/` — pre-computed COLMAP binary (cameras.bin, images.bin, points3D.bin); this is the critical SFM init source
  - `cityGSdata/geometry_gt/` — ground-truth point clouds for surface eval
- **Results**: `results/` — named `{scene}_{method}_{variant}/`

## CityGS Pipeline (3 stages)

### Stage 1: Coarse Training
Train a full-scene 3DGS on all cameras with SFM initialization:
```bash
python gsplat/examples/simple_trainer.py default \
    --data-dir <scene_dir> --data-factor 4 --data-type colmap \
    --init-type sfm \
    --max-steps 30000 --save-ply --ply-steps 30000 \
    --sh-degree 3 --ssim-lambda 0.2 \
    --strategy.reset-every 3000  # or 100000 to disable
```
`--init-type sfm` reads `sparse/0/points3D.{bin,txt}`. **Empty points3D → ~14 dB; proper SFM points → expected 20+ dB.**

### Stage 2: Block Partition + Finetuning
```bash
# Partition: assigns cameras to 3×1×3 = 9 blocks via SSIM or expanded-box
python citygs/partition_citygs.py --ply-path <coarse.ply> --data-dir <scene> \
    --block-dim 3 1 3 --output-dir <partition_dir>

# Prepare per-block camera lists
python citygs/prepare_block_data.py --data-dir <scene> \
    --partition-dir <partition_dir> --output-dir <blocks_dir>

# Train each block (loop over 0..8), init from coarse PLY
python gsplat/examples/simple_trainer.py default \
    --init-type ply --init-ply-path <coarse.ply> \
    --cam-indices-file <blocks_dir>/block_NNN/block_info.json \
    --strategy.reset-every 100000  # CRITICAL: disable opacity reset for blocks
```

### Stage 3: Merge + Eval
```bash
python citygs/merge_citygs.py --block-results-dir <blocks_dir> \
    --block-dim 3 1 3 --output <merged.ply> --partition-dir <partition_dir>

python citygs/eval_citygs.py --ply-path <merged.ply> \
    --data-dir <scene> --data-factor 4 --test-every 8 --output-dir <eval_dir>
```

## Known Issues / History

- **Empty points3D**: `rubble/sparse/0/points3D.txt` and `building/sparse/0/points3D.txt` are **empty** (0 points). This is the root cause of ~14 dB PSNR vs paper's ~25 dB. Fix: use `cityGSdata/colmap_results/` binary files.
- **Opacity reset during block finetuning**: `--strategy.reset-every 3000` (default) destroys converged coarse structure and makes merged models *worse* than coarse. Always use `--strategy.reset-every 100000` for block training.
- **COLMAP triangulation failure**: The installed COLMAP binary does not support `--Mapper.min_num_inliers`; do not use it.
- **v7/v8 merged results**: Despite disabling opacity reset (v8) and various coarse experiments (v3–v5), merged PSNR stagnated at ~13–14 dB because the coarse model itself was trained with random init (empty points3D).
- **SFM dataset setup**: The `rubble-pixsfm` dataset needs `sparse/` injected from `cityGSdata/colmap_results/rubble/` to enable `--init-type sfm`. CityGS's `data_proc_mill19.sh` does: symlink `train/images → train/rgbs`, move `colmap_results/rubble/train/sparse → rubble-pixsfm/train/`.
- **images_4_png corruption**: `images_4_png/000215.png` was 0 bytes after `_resize_image_folder` silently failed. Fixed by manual PIL resize. All 1678 PNGs in `rubble_citygs/images_4_png/` are now valid.

## Paper Targets (CityGS)
- Rubble: PSNR 25.77, SSIM 0.813, LPIPS 0.228
- Building: PSNR 21.55, SSIM 0.778, LPIPS 0.246

## Progress Log
See `progress_notes.txt` in the repo root for a running log of experiments and current status.
