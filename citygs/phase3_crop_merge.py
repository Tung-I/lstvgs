#!/usr/bin/env python3
"""Phase 3: crop each trained gsplat block to its block region (exact replication of the official
CityGaussian `block_filtering` / scene.save crop), then concatenate the 9 crops into a merged ply.

Official crop (utils/large_utils.py:block_filtering, scene/__init__.py:save):
  contract_to_unisphere(xyz, aabb, ord=inf) maps the aabb box to [0.25,0.75] and background shells
  into [0,0.25]u[0.75,1] of a [0,1] cube; keep gaussians whose contracted xyz lie in the block's
  [bx/Bx,(bx+1)/Bx) x [by/By,(by+1)/By) x [bz/Bz,(bz+1)/Bz) (strict, scale=1.0, no margin).
Run in gsplat env. GPU-free.
"""
import argparse, os, sys
import torch

sys.path.insert(0, "/work/pi_rsitaram_umass_edu/tungi/lstvgs/gsplat")
from gsplat.exporter import load_ply_to_splats, export_splats


def contract_to_unisphere(x, aabb, ord=torch.inf):
    aabb = torch.as_tensor(aabb, dtype=torch.float32)
    aabb_min, aabb_max = aabb[:3], aabb[3:]
    x = (x - aabb_min) / (aabb_max - aabb_min)
    x = x * 2 - 1
    mag = torch.linalg.norm(x, ord=ord, dim=-1, keepdim=True)
    mask = (mag.squeeze(-1) > 1)
    x[mask] = (2 - 1 / mag[mask]) * (x[mask] / mag[mask])
    x = x / 4 + 0.5
    return x


def block_mask(xyz, aabb, block_dim, block_id):
    bx = block_id % block_dim[0]
    by = (block_id % (block_dim[0] * block_dim[1])) // block_dim[0]
    bz = block_id // (block_dim[0] * block_dim[1])
    xc = contract_to_unisphere(xyz, aabb, ord=torch.inf)
    mnx, mxx = bx / block_dim[0], (bx + 1) / block_dim[0]
    mny, mxy = by / block_dim[1], (by + 1) / block_dim[1]
    mnz, mxz = bz / block_dim[2], (bz + 1) / block_dim[2]
    return ((xc[:, 0] >= mnx) & (xc[:, 0] < mxx) &
            (xc[:, 1] >= mny) & (xc[:, 1] < mxy) &
            (xc[:, 2] >= mnz) & (xc[:, 2] < mxz))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks-dir", default="/work/pi_rsitaram_umass_edu/tungi/lstvgs/results/rubble_citygs_gsplat_oracle/blocks")
    ap.add_argument("--step", type=int, default=29999)
    ap.add_argument("--block-dim", type=int, nargs=3, default=[3, 1, 3])
    ap.add_argument("--aabb", type=float, nargs=6, default=[-50, -100, -135, 50, 300, -5])
    ap.add_argument("--n-blocks", type=int, default=9)
    ap.add_argument("--prune-opacity", type=float, default=0.005)
    ap.add_argument("--out", default="/work/pi_rsitaram_umass_edu/tungi/lstvgs/results/rubble_citygs_gsplat_oracle/merged.ply")
    args = ap.parse_args()

    parts = {k: [] for k in ["means", "scales", "quats", "opacities", "sh0", "shN"]}
    total_in = total_kept = 0
    for b in range(args.n_blocks):
        ply = os.path.join(args.blocks_dir, f"block_{b:03d}", "ply", f"point_cloud_{args.step}.ply")
        if not os.path.exists(ply):
            print(f"[block {b}] MISSING {ply} — skipping")
            continue
        sp = load_ply_to_splats(ply)
        xyz = sp["means"].float()
        m = block_mask(xyz, args.aabb, args.block_dim, b)
        op = sp["opacities"].reshape(-1)
        if args.prune_opacity > 0:
            m = m & (torch.sigmoid(op) > args.prune_opacity)
        n_in, n_keep = xyz.shape[0], int(m.sum())
        total_in += n_in; total_kept += n_keep
        print(f"[block {b}] {n_in} -> {n_keep} kept ({100*n_keep/max(n_in,1):.1f}%)")
        parts["means"].append(sp["means"][m])
        parts["scales"].append(sp["scales"][m])
        parts["quats"].append(sp["quats"][m])
        parts["opacities"].append(op[m])
        parts["sh0"].append(sp["sh0"][m])
        parts["shN"].append(sp["shN"][m])

    merged = {k: torch.cat(v, 0) for k, v in parts.items()}
    n = merged["means"].shape[0]
    print(f"[merge] total {total_in} -> kept {total_kept}; merged gaussians = {n}")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    export_splats(means=merged["means"], scales=merged["scales"], quats=merged["quats"],
                  opacities=merged["opacities"], sh0=merged["sh0"], shN=merged["shN"],
                  format="ply", save_to=args.out)
    print(f"[merge] wrote {args.out}")


if __name__ == "__main__":
    main()
