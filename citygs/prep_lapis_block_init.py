#!/usr/bin/env python3
"""Prep the coarse-anchored L0 init for a block-wise Lapis-CityGS run.

Crops the shared coarse PLY to block `i`'s **expanded** owned cell (owned cell ±
`margin` cell-widths on each contracted axis) and writes `l0_init.ply`. LapisGS L0
then initializes from this instead of SfM, so every block's base layer shares the
same coarse Gaussians — including, in the overlap band, the *identical* Gaussians a
neighbor sees — which is what keeps CityGS seams soft at merge.

The crop is the same contracted-space test as the merge (`phase3_crop_merge.block_mask`)
but widened by `margin`; at merge time each layer is re-cropped to the *strict* owned
cell (no margin), so the margin here only provides boundary training context.

Run in the gsplat env. GPU-free.
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, "/work/pi_rsitaram_umass_edu/tungi/lstvgs/gsplat")
from gsplat.exporter import load_ply_to_splats, export_splats

# Reuse the exact contraction used by the merge so init/crop frames are identical.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase3_crop_merge import contract_to_unisphere


def expanded_block_mask(xyz, aabb, block_dim, block_id, margin):
    """Keep Gaussians whose contracted center lies in block_id's owned cell grown
    by `margin` cell-widths on each side (clamped to [0,1])."""
    bx = block_id % block_dim[0]
    by = (block_id % (block_dim[0] * block_dim[1])) // block_dim[0]
    bz = block_id // (block_dim[0] * block_dim[1])
    xc = contract_to_unisphere(xyz, aabb, ord=torch.inf)
    b = torch.tensor([bx, by, bz], dtype=torch.float32)
    B = torch.tensor(block_dim, dtype=torch.float32)
    lo = (b - margin) / B
    hi = (b + 1 + margin) / B
    lo = lo.clamp(0.0, 1.0)
    hi = hi.clamp(0.0, 1.0)
    return ((xc[:, 0] >= lo[0]) & (xc[:, 0] < hi[0]) &
            (xc[:, 1] >= lo[1]) & (xc[:, 1] < hi[1]) &
            (xc[:, 2] >= lo[2]) & (xc[:, 2] < hi[2]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coarse-ply", default="/work/pi_rsitaram_umass_edu/tungi/CityGaussian/"
                    "output/rubble_coarse/point_cloud/iteration_30000/point_cloud.ply")
    ap.add_argument("--block-id", type=int, required=True)
    ap.add_argument("--block-dim", type=int, nargs=3, default=[3, 1, 3])
    ap.add_argument("--aabb", type=float, nargs=6, default=[-50, -100, -135, 50, 300, -5])
    ap.add_argument("--margin", type=float, default=0.5,
                    help="cell-widths to expand the owned cell on each side")
    ap.add_argument("--out", required=True, help="path to write l0_init.ply")
    args = ap.parse_args()

    sp = load_ply_to_splats(args.coarse_ply)
    xyz = sp["means"].float()
    m = expanded_block_mask(xyz, args.aabb, args.block_dim, args.block_id, args.margin)
    n_in, n_keep = xyz.shape[0], int(m.sum())
    print(f"[block {args.block_id}] coarse {n_in} -> {n_keep} kept "
          f"({100*n_keep/max(n_in,1):.1f}%, margin={args.margin})")
    if n_keep == 0:
        raise SystemExit("ERROR: 0 Gaussians kept — check aabb/block_dim/margin.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    export_splats(
        means=sp["means"][m], scales=sp["scales"][m], quats=sp["quats"][m],
        opacities=sp["opacities"].reshape(-1)[m], sh0=sp["sh0"][m], shN=sp["shN"][m],
        format="ply", save_to=args.out,
    )
    print(f"[block {args.block_id}] wrote {args.out}")


if __name__ == "__main__":
    main()
