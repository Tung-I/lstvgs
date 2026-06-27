"""
Merge all per-block PLY files from CityGaussian block finetuning into a single PLY.

Usage:
    python citygs/merge_citygs.py \
        --block-results-dir results/rubble_citygs_blocks \
        --block-dim 3 1 3 \
        --step 30000 \
        --output results/rubble_citygs_merged/splat_merged.ply \
        --partition-dir results/rubble_citygs_coarse/partition

The per-block PLY files are expected at:
    {block-results-dir}/block_{NNN}/ply/point_cloud_{step-1}.ply
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "gsplat"))

from gsplat.exporter import load_ply_to_splats


def save_merged_ply(means, scales, quats, opacities, sh0, shN, out_path: str):
    """Save merged Gaussian splat model as standard 3DGS PLY."""
    from gsplat.exporter import export_splats
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    export_splats(
        means=means,
        scales=scales,
        quats=quats,
        opacities=opacities,
        sh0=sh0,
        shN=shN,
        format="ply",
        save_to=out_path,
    )
    print(f"Saved merged PLY: {out_path} ({means.shape[0]} Gaussians)")


def contract_to_unisphere(x: torch.Tensor, aabb: torch.Tensor) -> torch.Tensor:
    """Map x from [aabb_min, aabb_max] to [0, 1]³."""
    aabb_min = aabb[:3]
    aabb_max = aabb[3:]
    return (x - aabb_min) / (aabb_max - aabb_min)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--block-results-dir", required=True,
                        help="Directory containing block_NNN/ subdirectories")
    parser.add_argument("--block-dim", type=int, nargs=3, default=[3, 1, 3])
    parser.add_argument("--step", type=int, default=30000,
                        help="Training step for PLY file (e.g., 30000 → file is point_cloud_29999.ply)")
    parser.add_argument("--output", required=True, help="Output merged PLY path")
    parser.add_argument("--prune-opacity", type=float, default=0.005,
                        help="Prune Gaussians with sigmoid(opacity) < this threshold")
    parser.add_argument("--partition-dir", default="",
                        help="Partition directory from partition_citygs.py (enables spatial pruning)")
    parser.add_argument("--spatial-margin", type=float, default=0.5,
                        help="Fractional margin beyond block bounds for spatial pruning (default 0.5 = 50%%)")
    args = parser.parse_args()

    block_num = args.block_dim[0] * args.block_dim[1] * args.block_dim[2]
    print(f"Merging {block_num} blocks from {args.block_results_dir}")

    # Load AABB for spatial pruning
    aabb = None
    if args.partition_dir:
        aabb_path = os.path.join(args.partition_dir, "aabb.npy")
        if os.path.exists(aabb_path):
            aabb = torch.from_numpy(np.load(aabb_path)).float()
            print(f"Loaded scene AABB: {aabb.tolist()}")
        else:
            print(f"Warning: AABB not found at {aabb_path}, skipping spatial pruning")

    all_means = []
    all_scales = []
    all_quats = []
    all_opacities = []
    all_sh0 = []
    all_shN = []

    missing = []
    for b in range(block_num):
        block_dir = os.path.join(args.block_results_dir, f"block_{b:03d}")
        ply_path = os.path.join(block_dir, "ply", f"point_cloud_{args.step}.ply")
        if not os.path.exists(ply_path):
            # Try finding any available PLY
            ply_dir = os.path.join(block_dir, "ply")
            if os.path.exists(ply_dir):
                candidates = sorted(os.listdir(ply_dir))
                if candidates:
                    ply_path = os.path.join(ply_dir, candidates[-1])
                    print(f"  Block {b:3d}: using {candidates[-1]} (step {args.step} not found)")
                else:
                    print(f"  Block {b:3d}: MISSING (no PLY found)")
                    missing.append(b)
                    continue
            else:
                print(f"  Block {b:3d}: MISSING ({ply_path})")
                missing.append(b)
                continue

        splats = load_ply_to_splats(ply_path)
        n = splats["means"].shape[0]

        # Opacity pruning
        if args.prune_opacity > 0:
            keep = torch.sigmoid(splats["opacities"]) > args.prune_opacity
            splats = {k: v[keep] for k, v in splats.items()}
            n_after_opacity = splats["means"].shape[0]
        else:
            n_after_opacity = n

        # Spatial pruning: keep only Gaussians within (expanded) block bounding box
        if aabb is not None:
            bz = b // (args.block_dim[0] * args.block_dim[1])
            by = (b % (args.block_dim[0] * args.block_dim[1])) // args.block_dim[0]
            bx = (b % (args.block_dim[0] * args.block_dim[1])) % args.block_dim[0]

            # Nominal block bounds in [0,1]^3
            Bx, By, Bz = args.block_dim
            margin = args.spatial_margin
            block_min = torch.tensor([
                (bx - margin) / Bx,
                (by - margin) / By,
                (bz - margin) / Bz,
            ], dtype=torch.float32)
            block_max = torch.tensor([
                (bx + 1 + margin) / Bx,
                (by + 1 + margin) / By,
                (bz + 1 + margin) / Bz,
            ], dtype=torch.float32)

            xyz_c = contract_to_unisphere(splats["means"], aabb)  # [N, 3]
            in_block = (
                (xyz_c[:, 0] >= block_min[0]) & (xyz_c[:, 0] <= block_max[0]) &
                (xyz_c[:, 1] >= block_min[1]) & (xyz_c[:, 1] <= block_max[1]) &
                (xyz_c[:, 2] >= block_min[2]) & (xyz_c[:, 2] <= block_max[2])
            )
            splats = {k: v[in_block] for k, v in splats.items()}
            n_kept = splats["means"].shape[0]
            print(f"  Block {b:3d} [{bx},{by},{bz}]: {n} → {n_after_opacity} (opacity) → {n_kept} (spatial)")
        else:
            n_kept = n_after_opacity
            print(f"  Block {b:3d}: {n} → {n_kept} Gaussians")

        all_means.append(splats["means"])
        all_scales.append(splats["scales"])
        all_quats.append(splats["quats"])
        all_opacities.append(splats["opacities"])
        all_sh0.append(splats["sh0"])
        all_shN.append(splats["shN"])

    if missing:
        print(f"\nWarning: {len(missing)} blocks missing: {missing}")

    if not all_means:
        print("ERROR: No block PLY files found!")
        return

    print("\nConcatenating...")
    merged = {
        "means":     torch.cat(all_means, dim=0),
        "scales":    torch.cat(all_scales, dim=0),
        "quats":     torch.cat(all_quats, dim=0),
        "opacities": torch.cat(all_opacities, dim=0),
        "sh0":       torch.cat(all_sh0, dim=0),
        "shN":       torch.cat(all_shN, dim=0),
    }
    N_total = merged["means"].shape[0]
    print(f"Total Gaussians: {N_total}")

    save_merged_ply(
        merged["means"], merged["scales"], merged["quats"],
        merged["opacities"], merged["sh0"], merged["shN"],
        args.output,
    )


if __name__ == "__main__":
    main()
