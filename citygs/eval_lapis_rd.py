#!/usr/bin/env python3
"""Summarize the per-block rate-distortion curve for a Lapis-CityGS block.

For each layer L0..L3 of a single block:
  - quality = the in-region PSNR/SSIM/LPIPS already measured by the LapisGS
    `run_eval` (rendered with the owned+margin model on the block's val cams),
    read from <result_dir>/eval/metrics.json;
  - rate    = the size/#GS of the layer cropped to its STRICT owned cell (no
    margin) via phase3_crop_merge.block_mask — i.e. exactly what the merge keeps
    and what a client transmits for this block.

Writes <result_dir>/rd_summary.json and prints an RD table. GPU-free.
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, "/work/pi_rsitaram_umass_edu/tungi/lstvgs/gsplat")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gsplat.exporter import load_ply_to_splats, export_splats
from phase3_crop_merge import block_mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-dir", required=True, help="rubble_lapis_blockNNN dir")
    ap.add_argument("--block-id", type=int, required=True)
    ap.add_argument("--block-dim", type=int, nargs=3, default=[3, 1, 3])
    ap.add_argument("--aabb", type=float, nargs=6, default=[-50, -100, -135, 50, 300, -5])
    ap.add_argument("--prune-opacity", type=float, default=0.005)
    ap.add_argument("--eval-factor", type=int, default=4,
                    help="fixed eval factor whose metrics file to read for quality")
    ap.add_argument("--save-cropped", action="store_true",
                    help="also write the owned-cell-cropped per-layer plys")
    args = ap.parse_args()

    layer_dir = os.path.join(args.result_dir, "layers")
    # Prefer the fixed-factor metrics (ALL layers rendered at the same resolution) —
    # that is the apples-to-apples RD quality. Fall back to native per-layer metrics.
    eval_dir = os.path.join(args.result_dir, "eval")
    fixed = os.path.join(eval_dir, f"metrics_fixed_factor{args.eval_factor}.json")
    native = os.path.join(eval_dir, "metrics.json")
    eval_path = fixed if os.path.exists(fixed) else native
    quality = {}
    if os.path.exists(eval_path):
        print(f"[quality] {os.path.basename(eval_path)}"
              f"{'  (FIXED-factor, apples-to-apples)' if eval_path == fixed else '  (NATIVE per-layer — not comparable across layers!)'}")
        for r in json.load(open(eval_path)):
            quality[r["layer"]] = r
    else:
        print(f"[warn] no eval metrics in {eval_dir} — quality columns blank")

    rows = []
    for L in range(4):
        full = os.path.join(layer_dir, f"layer_{L:02d}_full.ply")
        if not os.path.exists(full):
            continue
        sp = load_ply_to_splats(full)
        xyz = sp["means"].float()
        m = block_mask(xyz, args.aabb, args.block_dim, args.block_id)
        op = sp["opacities"].reshape(-1)
        if args.prune_opacity > 0:
            m = m & (torch.sigmoid(op) > args.prune_opacity)
        n_full, n_owned = xyz.shape[0], int(m.sum())

        # cropped transmitted size: write a temp (or kept) ply and measure bytes
        crop_path = os.path.join(layer_dir, f"layer_{L:02d}_owned.ply")
        export_splats(means=sp["means"][m], scales=sp["scales"][m], quats=sp["quats"][m],
                      opacities=op[m], sh0=sp["sh0"][m], shN=sp["shN"][m],
                      format="ply", save_to=crop_path)
        owned_mb = os.path.getsize(crop_path) / 1e6
        if not args.save_cropped:
            os.remove(crop_path)

        q = quality.get(L, {})
        rows.append({
            "layer": L,
            "factor": q.get("factor"),
            "n_full": n_full,
            "n_owned": n_owned,
            "owned_mb": round(owned_mb, 2),
            "psnr": q.get("psnr"),
            "ssim": q.get("ssim"),
            "lpips": q.get("lpips"),
        })

    out = os.path.join(args.result_dir, "rd_summary.json")
    json.dump(rows, open(out, "w"), indent=2)

    print(f"\n{'='*78}\nBlock {args.block_id} RD curve  (rate = owned-cell crop, quality = in-region)")
    print(f"{'L':>2} {'fac':>4} {'#GS owned':>11} {'owned MB':>9} {'PSNR':>7} {'SSIM':>7} {'LPIPS':>7}")
    print("-" * 78)
    for r in rows:
        def f(x, p): return (f"%.{p}f" % x) if isinstance(x, (int, float)) else "  -  "
        print(f"{r['layer']:>2} {str(r['factor']):>4} {r['n_owned']:>11,} {r['owned_mb']:>9.1f} "
              f"{f(r['psnr'],3):>7} {f(r['ssim'],4):>7} {f(r['lpips'],3):>7}")
    print("=" * 78)
    print(f"saved → {out}")


if __name__ == "__main__":
    main()
