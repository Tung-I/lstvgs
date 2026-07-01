#!/usr/bin/env python3
"""Cross-block per-LoD merge for the block-wise Lapis-CityGS pipeline.

A per-layer generalization of `phase3_crop_merge.py`: for each LoD level L, take every
block's cumulative `layers/layer_{L:02d}_full.ply`, crop it to that block's STRICT owned
cell (the same `block_mask` the flat CityGS merge uses — no margin), concat across blocks,
and write `<out-dir>/merged_lod{L}.ply`. The result is a global scene model at LoD L, built
from exactly the Gaussians each block transmits for its region.

Each block's per-layer ply is a self-contained model (L0 base ∪ all enhancement deltas up
to L), so merging the same level L across blocks gives a consistent global level-L scene.
Blocks missing `layer_{L:02d}_full.ply` are skipped, so this runs on a partial block set.

Run in the gsplat env. GPU-free.
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, "/work/pi_rsitaram_umass_edu/tungi/lstvgs/gsplat")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gsplat.exporter import load_ply_to_splats, export_splats
# Reuse the EXACT crop used by the flat CityGS merge — do not re-implement.
from phase3_crop_merge import block_mask


def merge_level(L, blocks_dir, n_blocks, block_dim, aabb, prune_opacity, out_path):
    parts = {k: [] for k in ["means", "scales", "quats", "opacities", "sh0", "shN"]}
    total_in = total_kept = 0
    present = 0
    for b in range(n_blocks):
        ply = os.path.join(blocks_dir, f"rubble_lapis_block{b:03d}", "layers",
                           f"layer_{L:02d}_full.ply")
        if not os.path.exists(ply):
            print(f"  [L{L} block {b}] MISSING {os.path.basename(ply)} — skipping")
            continue
        present += 1
        sp = load_ply_to_splats(ply)
        xyz = sp["means"].float()
        m = block_mask(xyz, aabb, block_dim, b)
        op = sp["opacities"].reshape(-1)
        if prune_opacity > 0:
            m = m & (torch.sigmoid(op) > prune_opacity)
        n_in, n_keep = xyz.shape[0], int(m.sum())
        total_in += n_in
        total_kept += n_keep
        print(f"  [L{L} block {b}] {n_in:,} -> {n_keep:,} kept ({100*n_keep/max(n_in,1):.1f}%)")
        parts["means"].append(sp["means"][m])
        parts["scales"].append(sp["scales"][m])
        parts["quats"].append(sp["quats"][m])
        parts["opacities"].append(op[m])
        parts["sh0"].append(sp["sh0"][m])
        parts["shN"].append(sp["shN"][m])

    if present == 0:
        print(f"  [L{L}] no blocks present — skipping level")
        return None

    merged = {k: torch.cat(v, 0) for k, v in parts.items()}
    n = merged["means"].shape[0]
    print(f"  [L{L}] {present} blocks; total {total_in:,} -> kept {total_kept:,}; merged = {n:,}")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    export_splats(means=merged["means"], scales=merged["scales"], quats=merged["quats"],
                  opacities=merged["opacities"], sh0=merged["sh0"], shN=merged["shN"],
                  format="ply", save_to=out_path)
    mb = os.path.getsize(out_path) / 1e6
    print(f"  [L{L}] wrote {out_path} ({mb:.1f} MB, {n:,} GS)")
    return {"layer": L, "n_gs": n, "mb": round(mb, 1), "blocks": present}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks-dir",
                    default="/work/pi_rsitaram_umass_edu/tungi/lstvgs/results",
                    help="dir containing rubble_lapis_blockNNN/ subdirs")
    ap.add_argument("--n-blocks", type=int, default=9)
    ap.add_argument("--layers", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--block-dim", type=int, nargs=3, default=[3, 1, 3])
    ap.add_argument("--aabb", type=float, nargs=6, default=[-50, -100, -135, 50, 300, -5])
    ap.add_argument("--prune-opacity", type=float, default=0.005)
    ap.add_argument("--out-dir",
                    default="/work/pi_rsitaram_umass_edu/tungi/lstvgs/results/rubble_lapis_lod")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    summary = []
    for L in args.layers:
        print(f"{'='*70}\nMerging LoD level {L}")
        out_path = os.path.join(args.out_dir, f"merged_lod{L}.ply")
        r = merge_level(L, args.blocks_dir, args.n_blocks, args.block_dim,
                        args.aabb, args.prune_opacity, out_path)
        if r:
            summary.append(r)

    print(f"\n{'='*70}\nMerged LoD summary  (rate = merged owned-cell crop)")
    print(f"{'LoD':>3} {'#blocks':>7} {'#GS':>14} {'size MB':>9}")
    print("-" * 40)
    for r in summary:
        print(f"{r['layer']:>3} {r['blocks']:>7} {r['n_gs']:>14,} {r['mb']:>9.1f}")
    print("=" * 40)


if __name__ == "__main__":
    main()
