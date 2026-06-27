"""
Prepare per-block COLMAP data for CityGaussian block finetuning.

Given the camera_mask from partition_citygs.py, this script creates a lightweight
data directory for each block that contains:
  - A JSON file listing which camera indices belong to this block
  - The block's assigned camera intrinsics/extrinsics
  - Symlinks to the original image directory (images are shared)

The block trainer then uses a custom dataset filter to only load the assigned cameras.

Usage:
    python citygs/prepare_block_data.py \
        --data-dir /path/to/rubble \
        --data-factor 4 \
        --partition-dir results/rubble_citygs_coarse/partition \
        --output-dir results/rubble_citygs_blocks \
        --block-dim 3 1 3
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pycolmap

SCRIPT_DIR = Path(__file__).parent
EXAMPLES_DIR = SCRIPT_DIR.parent / "gsplat" / "examples"
sys.path.insert(0, str(EXAMPLES_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent / "gsplat"))

from datasets.colmap import Parser


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--data-factor", type=int, default=4)
    parser.add_argument("--test-every", type=int, default=8)
    parser.add_argument("--partition-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--block-dim", type=int, nargs=3, default=[3, 1, 3])
    args = parser.parse_args()

    # Load partition results
    camera_mask = np.load(os.path.join(args.partition_dir, "camera_mask.npy"))  # [N_train, B]
    with open(os.path.join(args.partition_dir, "partition_info.json")) as f:
        info = json.load(f)

    block_num = info["block_num"]
    print(f"Preparing {block_num} block data directories")

    # Load parser to get image names and indices
    colmap_parser = Parser(
        data_dir=args.data_dir,
        factor=args.data_factor,
        normalize=True,
        test_every=args.test_every,
    )
    train_indices = [i for i in range(len(colmap_parser.image_names))
                     if i % args.test_every != 0]
    all_image_names = colmap_parser.image_names

    os.makedirs(args.output_dir, exist_ok=True)

    for block_id in range(block_num):
        block_dir = os.path.join(args.output_dir, f"block_{block_id:03d}")
        os.makedirs(block_dir, exist_ok=True)

        # Find which training cameras belong to this block
        block_cam_mask = camera_mask[:, block_id]  # [N_train,]
        assigned_train_positions = np.where(block_cam_mask)[0].tolist()  # positions in train_indices list
        # Map back to global indices
        assigned_global_indices = [train_indices[p] for p in assigned_train_positions]
        assigned_image_names = [all_image_names[i] for i in assigned_global_indices]

        block_info = {
            "block_id": block_id,
            "n_cameras": len(assigned_global_indices),
            "global_indices": assigned_global_indices,
            "image_names": assigned_image_names,
            "data_dir": args.data_dir,
            "data_factor": args.data_factor,
            "test_every": args.test_every,
        }
        with open(os.path.join(block_dir, "block_info.json"), "w") as f:
            json.dump(block_info, f, indent=2)

        print(f"Block {block_id:3d}: {len(assigned_global_indices):4d} cameras")

    print(f"\nBlock data prepared in {args.output_dir}")
    print("Next: run block finetuning using train_blocks.py or per-block SLURM scripts")


if __name__ == "__main__":
    main()
