"""
Spatially prune the coarse PLY for each block before finetuning.

Each block covers 1/9th of the scene. Starting block finetuning from the full
8M Gaussian coarse PLY is slow (3-4 it/s). This script prunes to only the
Gaussians within each block's spatial region (+ margin), reducing starting
count to ~1M per block and restoring ~20 it/s training speed.

Usage:
    python citygs/prepare_block_ply.py \
        --coarse-ply results/rubble_citygs_coarse_v9/ply/point_cloud_29999.ply \
        --partition-dir results/rubble_citygs_coarse_v9/partition \
        --output-dir results/rubble_citygs_blocks_v9 \
        --margin 0.15
"""

import argparse
import json
import os
import struct
from pathlib import Path

import numpy as np


def read_ply_gaussians(path):
    """Read a 3DGS PLY file, return (data_dict, vertex_names, header_bytes)."""
    with open(path, "rb") as f:
        header_lines = []
        while True:
            line = f.readline().decode("ascii").strip()
            header_lines.append(line)
            if line == "end_header":
                break
        header_bytes = b"\n".join(l.encode("ascii") for l in header_lines) + b"\n"

        # Parse properties
        props = []
        n_verts = 0
        for line in header_lines:
            if line.startswith("element vertex"):
                n_verts = int(line.split()[-1])
            elif line.startswith("property float "):
                props.append(line.split()[-1])

        n_props = len(props)
        dtype = np.dtype([("data", np.float32, (n_props,))])
        raw = np.frombuffer(f.read(n_verts * n_props * 4), dtype=np.float32).reshape(n_verts, n_props)

    return raw, props, header_lines, n_verts


def write_ply_gaussians(path, raw, props, header_lines):
    """Write a 3DGS PLY file from raw float32 array."""
    n_verts = raw.shape[0]
    # Rebuild header with updated vertex count
    new_header_lines = []
    for line in header_lines:
        if line.startswith("element vertex"):
            new_header_lines.append(f"element vertex {n_verts}")
        else:
            new_header_lines.append(line)
    header_bytes = b"\n".join(l.encode("ascii") for l in new_header_lines) + b"\n"
    with open(path, "wb") as f:
        f.write(header_bytes)
        f.write(raw.tobytes())


def contract_to_unisphere(xyz, aabb):
    """Map xyz from [aabb_min, aabb_max] to [0, 1]^3."""
    aabb_min = aabb[:3]
    aabb_max = aabb[3:]
    return (xyz - aabb_min) / (aabb_max - aabb_min)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coarse-ply", required=True)
    parser.add_argument("--partition-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--margin", type=float, default=0.15,
                        help="Fractional margin beyond each block's boundary (default 0.15)")
    args = parser.parse_args()

    # Load partition info
    with open(os.path.join(args.partition_dir, "partition_info.json")) as f:
        info = json.load(f)
    aabb = np.array(info["aabb"], dtype=np.float32)
    block_dim = info["block_dim"]  # [3, 1, 3]
    block_num = info["block_num"]
    cameras_per_block = info["cameras_per_block"]

    print(f"Loading coarse PLY: {args.coarse_ply}")
    raw, props, header_lines, n_verts = read_ply_gaussians(args.coarse_ply)
    print(f"  {n_verts:,} Gaussians, {len(props)} properties")

    # Get xyz indices in property list
    xi = props.index("x")
    yi = props.index("y")
    zi = props.index("z")
    xyz = raw[:, [xi, yi, zi]]  # [N, 3]

    # Contract to [0,1]^3
    xyz_norm = contract_to_unisphere(xyz, aabb)

    os.makedirs(args.output_dir, exist_ok=True)

    for block_id in range(block_num):
        n_cam = cameras_per_block[block_id]
        if n_cam < 2:
            print(f"Block {block_id:2d}: {n_cam} cameras — skipping PLY")
            continue

        # Compute block's normalized [0,1]^3 grid cell
        bz = block_id // (block_dim[0] * block_dim[1])
        by = (block_id % (block_dim[0] * block_dim[1])) // block_dim[0]
        bx = (block_id % (block_dim[0] * block_dim[1])) % block_dim[0]

        lo = np.array([bx / block_dim[0], by / block_dim[1], bz / block_dim[2]])
        hi = np.array([(bx+1) / block_dim[0], (by+1) / block_dim[1], (bz+1) / block_dim[2]])

        # Add margin
        lo_m = np.maximum(lo - args.margin, 0.0)
        hi_m = np.minimum(hi + args.margin, 1.0)

        mask = (
            (xyz_norm[:, 0] >= lo_m[0]) & (xyz_norm[:, 0] < hi_m[0]) &
            (xyz_norm[:, 1] >= lo_m[1]) & (xyz_norm[:, 1] < hi_m[1]) &
            (xyz_norm[:, 2] >= lo_m[2]) & (xyz_norm[:, 2] < hi_m[2])
        )
        block_raw = raw[mask]

        block_dir = os.path.join(args.output_dir, f"block_{block_id:03d}")
        os.makedirs(block_dir, exist_ok=True)
        out_ply = os.path.join(block_dir, "init.ply")
        write_ply_gaussians(out_ply, block_raw, props, header_lines)

        print(f"Block {block_id:2d} [{bx},{by},{bz}] ({n_cam:4d} cams): "
              f"{block_raw.shape[0]:>7,} / {n_verts:,} Gaussians → {out_ply}")

    print("\nDone.")


if __name__ == "__main__":
    main()
