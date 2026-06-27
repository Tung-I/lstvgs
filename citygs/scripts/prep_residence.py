#!/usr/bin/env python3
"""
Prepare UrbanScene3D Residence dataset for simple_trainer.py (colmap data_type).

Prerequisites:
  1. residence-pixsfm/ metadata extracted (train/metadata/*.pt, val/metadata/*.pt)
     Already done: $WORK/datasets/residence-pixsfm/
  2. Raw UrbanScene3D photos downloaded to $RAW_IMAGES_DIR
     Download from: https://drive.google.com/drive/folders/1e91lEw56DUBbQgRTo48T3lVjo53SzEOd
     or NAS: http://szuvccnas.quickconnect.cn/d/s/lSvWkTMbFjecrEwZDx3cV72M5scS2tKA/
     Files needed: building/ (or residence/) folder with DJI_XXXX.JPG images

This script:
  1. Reads mappings.txt to map original filenames to metadata .pt files
  2. Loads each .pt file (H, W, c2w, intrinsics, distortion)
  3. Undistorts raw images using OpenCV
  4. Writes undistorted images as 000000.JPG, 000001.JPG, ... (matching COLMAP names)
  5. Copies pre-computed COLMAP sparse reconstruction
  6. Creates images_4/ (4x downscaled) for training

Usage:
  python prep_residence.py --raw-dir <path_to_raw_DJI_images> [--out-dir <path>]
"""

import argparse, shutil
import cv2
import numpy as np
import torch
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

WORK = Path("/work/pi_rsitaram_umass_edu/tungi")
PIXSFM = WORK / "datasets/residence-pixsfm"
COLMAP = WORK / "datasets/cityGSdata/colmap_results/residence"

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", required=True,
                   help="Directory containing raw DJI_XXXX.JPG photos from UrbanScene3D")
    p.add_argument("--out-dir", default=str(WORK / "datasets/residence_sfm"),
                   help="Output dataset directory")
    p.add_argument("--workers", type=int, default=8)
    return p.parse_args()

def undistort_image(src_path, out_path, H, W, intrinsics, distortion):
    """Undistort image using per-frame camera model from pixsfm metadata."""
    fx, fy, cx, cy = intrinsics.tolist()
    k1, k2, p1, p2 = distortion.tolist()
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    D = np.array([k1, k2, p1, p2], dtype=np.float64)
    img = cv2.imread(str(src_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read {src_path}")
    img_ud = cv2.undistort(img, K, D)
    cv2.imwrite(str(out_path), img_ud)

def main():
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)

    # Read mappings: original_path → metadata_stem
    mappings = {}
    with open(PIXSFM / "mappings.txt") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            orig, meta = line.split(",")
            mappings[orig.strip()] = meta.strip()  # e.g. "B/DJI_0872.JPG" → "001458.pt"

    print(f"Mappings: {len(mappings)} entries")

    for split in ["train", "val"]:
        meta_dir = PIXSFM / split / "metadata"
        out_images = out_dir / split / "images"
        out_images_4 = out_dir / split / "images_4"
        out_images.mkdir(parents=True, exist_ok=True)
        out_images_4.mkdir(parents=True, exist_ok=True)

        # Build list: (src_image_path, out_stem, metadata_path)
        pt_files = sorted(meta_dir.glob("*.pt"))
        print(f"[{split}] {len(pt_files)} metadata files")

        # Reverse-lookup: metadata_stem → original image path
        reverse_map = {v: k for k, v in mappings.items()}  # "001458.pt" → "B/DJI_0872.JPG"

        def process_one(pt_file):
            stem = pt_file.name  # e.g. "001458.pt"
            out_stem = pt_file.stem  # "001458" → becomes "001458.JPG"
            dst_full = out_images   / f"{out_stem}.JPG"
            dst_4    = out_images_4 / f"{out_stem}.JPG"
            if dst_full.exists() and dst_4.exists():
                return "skip"
            orig_rel = reverse_map.get(stem)
            if orig_rel is None:
                return f"no mapping for {stem}"
            # orig_rel like "B/DJI_0872.JPG" — search under raw_dir
            src = raw_dir / orig_rel
            if not src.exists():
                # Try just the filename
                src = raw_dir / Path(orig_rel).name
            if not src.exists():
                return f"missing {orig_rel}"
            meta = torch.load(pt_file, map_location="cpu", weights_only=False)
            if not dst_full.exists():
                undistort_image(src, dst_full, meta["H"], meta["W"],
                                meta["intrinsics"], meta["distortion"])
            if not dst_4.exists():
                img = cv2.imread(str(dst_full))
                H4, W4 = meta["H"] // 4, meta["W"] // 4
                img_small = cv2.resize(img, (W4, H4), interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(dst_4), img_small)
            return "ok"

        results = {}
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for pt_file, result in zip(pt_files, ex.map(process_one, pt_files)):
                results[result] = results.get(result, 0) + 1

        print(f"  [{split}] results: {results}")

        # Install COLMAP sparse
        dst_sparse = out_dir / split / "sparse"
        if not dst_sparse.exists():
            shutil.copytree(COLMAP / split / "sparse", dst_sparse)
            print(f"  [{split}] sparse installed from {COLMAP / split / 'sparse'}")
        else:
            print(f"  [{split}] sparse already exists")

    print("\n=== Residence dataset ready ===")
    print(f"  {out_dir}")

if __name__ == "__main__":
    main()
