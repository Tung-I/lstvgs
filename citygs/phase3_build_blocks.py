#!/usr/bin/env python3
"""Phase 3 oracle-aligned scaffolding for gsplat CityGS reproduction.

Builds per-block block_info.json (camera global-index lists) by mapping the official
CityGaussian per-block camera lists (output/rubble_c9_r4/cells/cellN/cameras.json, keyed by
img_name) onto the gsplat Parser's image ordering of the SAME INRIA dataset dir (so the world
frame matches the coarse ply we init from). Also verifies the INRIA coarse ply loads via gsplat.

Run in the gsplat conda env. GPU-free.
"""
import argparse, json, os, sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/work/pi_rsitaram_umass_edu/tungi/CityGaussian/data/mill19/rubble-pixsfm/train",
                    help="INRIA train dir (images/ + sparse/) — same frame as the coarse ply")
    ap.add_argument("--cells-dir", default="/work/pi_rsitaram_umass_edu/tungi/CityGaussian/output/rubble_c9_r4/cells",
                    help="dir with cell0..cell8/cameras.json")
    ap.add_argument("--coarse-ply", default="/work/pi_rsitaram_umass_edu/tungi/CityGaussian/output/rubble_coarse/point_cloud/iteration_30000/point_cloud.ply")
    ap.add_argument("--n-blocks", type=int, default=9)
    ap.add_argument("--out-dir", default="/work/pi_rsitaram_umass_edu/tungi/lstvgs/results/rubble_citygs_gsplat_oracle/blocks")
    args = ap.parse_args()

    sys.path.insert(0, "/work/pi_rsitaram_umass_edu/tungi/lstvgs/gsplat/examples")
    from datasets.colmap import Parser

    # Build gsplat global camera ordering on the SAME dataset dir (factor=1: no resize at this step).
    print(f"[parser] loading {args.data_dir} ...")
    parser = Parser(data_dir=args.data_dir, factor=1, normalize=False, test_every=8)
    image_names = list(parser.image_names)
    print(f"[parser] {len(image_names)} images; e.g. {image_names[:2]} ... {image_names[-2:]}")
    # name (no extension / basename) -> global index
    def stem(n):
        return os.path.splitext(os.path.basename(n))[0]
    name2idx = {stem(n): i for i, n in enumerate(image_names)}

    os.makedirs(args.out_dir, exist_ok=True)
    total = 0
    counts = []
    for b in range(args.n_blocks):
        cj = os.path.join(args.cells_dir, f"cell{b}", "cameras.json")
        cams = json.load(open(cj))
        names = [stem(str(c["img_name"])) for c in cams]
        idxs = []
        missing = []
        for nm in names:
            if nm in name2idx:
                idxs.append(name2idx[nm])
            else:
                missing.append(nm)
        if missing:
            print(f"[block {b}] WARNING {len(missing)} img_names not found in parser, e.g. {missing[:3]}")
        idxs = sorted(set(idxs))
        counts.append(len(idxs))
        total += len(idxs)
        bdir = os.path.join(args.out_dir, f"block_{b:03d}")
        os.makedirs(bdir, exist_ok=True)
        json.dump({
            "block_id": b,
            "n_cameras": len(idxs),
            "global_indices": idxs,
            "data_dir": args.data_dir,
        }, open(os.path.join(bdir, "block_info.json"), "w"))
        print(f"[block {b}] {len(idxs)} cameras -> {bdir}/block_info.json")

    print(f"[summary] per-block camera counts: {counts}")
    print(f"[summary] total (with overlap across blocks counted separately) = {total}")
    print(f"[summary] expected INRIA counts ~ [298,487,344,588,747,501,426,528,239]")

    # Verify coarse ply loads in gsplat.
    print("[coarse] loading via gsplat load_ply_to_splats ...")
    sys.path.insert(0, "/work/pi_rsitaram_umass_edu/tungi/lstvgs/gsplat")
    from gsplat.exporter import load_ply_to_splats
    sp = load_ply_to_splats(args.coarse_ply)
    n = sp["means"].shape[0]
    print(f"[coarse] OK: {n} gaussians; keys={list(sp.keys())}; "
          f"sh0={tuple(sp['sh0'].shape)} shN={tuple(sp['shN'].shape)}")


if __name__ == "__main__":
    main()
