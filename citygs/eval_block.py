"""
Evaluate a single finetuned block on its own assigned cameras.
Renders each assigned camera using the block's PLY, computes PSNR/SSIM/LPIPS.

Usage:
    python citygs/eval_block.py \
        --block-dir results/rubble_citygs_blocks_v9/block_000 \
        --data-dir /work/.../datasets/rubble_citygs \
        [--data-factor 4]
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import numpy as np
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

SCRIPT_DIR = Path(__file__).parent
EXAMPLES_DIR = SCRIPT_DIR.parent / "gsplat" / "examples"
sys.path.insert(0, str(EXAMPLES_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent / "gsplat"))

from datasets.colmap import Parser, Dataset
from gsplat import rasterization
from gsplat.exporter import load_ply_to_splats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--block-dir", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--data-factor", type=int, default=4)
    parser.add_argument("--max-images", type=int, default=50,
                        help="Max cameras to eval (sample evenly if more)")
    args = parser.parse_args()

    block_dir = Path(args.block_dir)
    block_id = int(block_dir.name.split("_")[-1])

    # Load block info
    with open(block_dir / "block_info.json") as f:
        info = json.load(f)
    global_indices = info["global_indices"]
    n_cam = len(global_indices)
    print(f"Block {block_id}: {n_cam} assigned cameras")

    # Find PLY
    ply_candidates = sorted((block_dir / "ply").glob("point_cloud_*.ply"))
    if not ply_candidates:
        print(f"No PLY found in {block_dir}/ply/"); return
    ply_path = ply_candidates[-1]
    print(f"PLY: {ply_path.name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Gaussians
    gaussians = load_ply_to_splats(str(ply_path))
    means    = gaussians["means"].to(device)
    quats    = gaussians["quats"].to(device)
    scales   = gaussians["scales"].to(device)
    opacities = gaussians["opacities"].to(device)
    sh0      = gaussians["sh0"].to(device)
    shN      = gaussians.get("shN", None)
    if shN is not None:
        shN = shN.to(device)
    n_gs = means.shape[0]
    print(f"Loaded {n_gs:,} Gaussians")

    # Load dataset
    colmap_parser = Parser(
        data_dir=args.data_dir,
        factor=args.data_factor,
        normalize=True,
        test_every=83,
    )
    full_dataset = Dataset(colmap_parser, split="train", load_depths=False)

    # Sample cameras to eval
    if n_cam > args.max_images:
        step = n_cam // args.max_images
        indices_to_eval = global_indices[::step][:args.max_images]
    else:
        indices_to_eval = global_indices
    print(f"Evaluating {len(indices_to_eval)} cameras...")

    # Build index mapping: global_index → position in full train dataset
    # full_dataset indices are the training set positions (not global image indices)
    # The Dataset returns items at position i within the training split
    train_positions = [i for i in range(len(colmap_parser.image_names))
                       if i % 83 != 0]
    global_to_train_pos = {g: p for p, g in enumerate(train_positions)}

    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type="alex").to(device)

    psnrs, ssims, lpipss = [], [], []

    for g_idx in indices_to_eval:
        pos = global_to_train_pos.get(g_idx)
        if pos is None:
            continue
        data = full_dataset[pos]
        K = data["K"].to(device)           # [3, 3]
        c2w = data["camtoworld"].to(device) # [4, 4]
        image = data["image"].to(device)    # [H, W, 3]
        H, W = image.shape[:2]

        w2c = torch.linalg.inv(c2w)
        viewmat = w2c.unsqueeze(0)  # [1, 4, 4]
        Ks = K.unsqueeze(0)         # [1, 3, 3]

        sh_degree = 0
        colors = sh0  # [N, 1, 3]
        if shN is not None:
            colors = torch.cat([sh0, shN], dim=1)
            sh_degree = int(round((colors.shape[1] ** 0.5) - 1))

        with torch.no_grad():
            render_colors, _, _ = rasterization(
                means=means, quats=quats, scales=scales,
                opacities=opacities.squeeze(-1),
                colors=colors,
                viewmats=viewmat,
                Ks=Ks,
                width=W, height=H,
                sh_degree=sh_degree,
                near_plane=0.01, far_plane=1e10,
                render_mode="RGB",
                packed=True,
            )
        render = render_colors[0].clamp(0, 1)  # [H, W, 3]

        gt = image.clamp(0, 1)
        r = render.permute(2, 0, 1).unsqueeze(0)
        g = gt.permute(2, 0, 1).unsqueeze(0)

        psnrs.append(psnr_metric(r, g).item())
        ssims.append(ssim_metric(r, g).item())
        lpipss.append(lpips_metric(r * 2 - 1, g * 2 - 1).item())

    if not psnrs:
        print("No cameras evaluated."); return

    print(f"\n=== Block {block_id} eval on own cameras ===")
    print(f"  PSNR:  {np.mean(psnrs):.3f} dB  (min {np.min(psnrs):.1f}, max {np.max(psnrs):.1f})")
    print(f"  SSIM:  {np.mean(ssims):.4f}")
    print(f"  LPIPS: {np.mean(lpipss):.4f}")
    print(f"  Cameras evaluated: {len(psnrs)}")

    result = {
        "block_id": block_id,
        "ply": ply_path.name,
        "n_cameras_eval": len(psnrs),
        "psnr": float(np.mean(psnrs)),
        "ssim": float(np.mean(ssims)),
        "lpips": float(np.mean(lpipss)),
    }
    out_path = block_dir / "stats" / "block_eval.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
