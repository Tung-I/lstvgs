"""
Evaluate a merged CityGaussian model (or any single PLY) on the test split.

Renders each test view with gsplat's rasterization and computes PSNR/SSIM/LPIPS.
Matches the evaluation protocol of the paper (test_every=8, downscale factor=4).

Usage:
    python citygs/eval_citygs.py \
        --ply-path results/rubble_citygs_merged/splat_merged.ply \
        --data-dir /path/to/rubble \
        --data-factor 4 \
        --test-every 8 \
        --output-dir results/rubble_citygs_merged/eval \
        [--save-images]
"""

import argparse
import json
import os
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).parent
EXAMPLES_DIR = SCRIPT_DIR.parent / "gsplat" / "examples"
sys.path.insert(0, str(EXAMPLES_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent / "gsplat"))

from gsplat.exporter import load_ply_to_splats
from gsplat.rendering import rasterization
from datasets.colmap import Parser


def compute_psnr(pred: torch.Tensor, gt: torch.Tensor) -> float:
    mse = F.mse_loss(pred, gt).item()
    return float(10.0 * np.log10(1.0 / max(mse, 1e-10)))


def compute_ssim(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """Compute SSIM on [H, W, 3] tensors in [0,1]."""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    img1 = img1.permute(2, 0, 1).unsqueeze(0)
    img2 = img2.permute(2, 0, 1).unsqueeze(0)
    ks = 11
    mu1 = F.avg_pool2d(img1, ks, 1, ks // 2)
    mu2 = F.avg_pool2d(img2, ks, 1, ks // 2)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu12 = mu1 * mu2
    sig1 = F.avg_pool2d(img1 ** 2, ks, 1, ks // 2) - mu1_sq
    sig2 = F.avg_pool2d(img2 ** 2, ks, 1, ks // 2) - mu2_sq
    sig12 = F.avg_pool2d(img1 * img2, ks, 1, ks // 2) - mu12
    ssim_map = ((2 * mu12 + C1) * (2 * sig12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sig1 + sig2 + C2))
    return float(ssim_map.mean().item())


@torch.no_grad()
def render_view(means, quats, scales, opacities, sh0, shN, sh_degree,
                camtoworld, K, width, height, device, bg_color=0.0):
    """Render a single view. Returns [H, W, 3] float tensor."""
    viewmat = torch.linalg.inv(camtoworld.to(device)).unsqueeze(0)
    K_t = K.to(device).unsqueeze(0)

    # Pass SH coefficients [N, K, 3] directly to rasterization
    sh_coeffs = torch.cat([sh0, shN], dim=1)  # [N, K, 3]

    render_colors, render_alphas, _ = rasterization(
        means=means,
        quats=quats,
        scales=torch.exp(scales),
        opacities=torch.sigmoid(opacities),
        colors=sh_coeffs,
        viewmats=viewmat,
        Ks=K_t,
        width=width,
        height=height,
        sh_degree=sh_degree,
        packed=True,
        absgrad=False,
        sparse_grad=False,
        rasterize_mode="classic",
    )
    # Composite with background color (matching training with --no-random-bkgd which uses black bg)
    colors = render_colors[0]  # [H, W, C]
    alphas = render_alphas[0]  # [H, W, 1]
    bg = torch.full_like(colors, bg_color)
    colors = colors + bg * (1.0 - alphas)
    return colors.clamp(0.0, 1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply-path", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--data-factor", type=int, default=4)
    parser.add_argument("--test-every", type=int, default=8)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--save-images", action="store_true")
    parser.add_argument("--lpips-net", default="alex", choices=["alex", "vgg"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--bg-color", type=float, default=0.0, help="Background color (0=black)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    if args.save_images:
        os.makedirs(os.path.join(args.output_dir, "renders"), exist_ok=True)

    device = args.device

    # ── Load LPIPS ─────────────────────────────────────────────────────────────
    try:
        import lpips
        lpips_fn = lpips.LPIPS(net=args.lpips_net).to(device)
        use_lpips = True
    except ImportError:
        print("Warning: lpips not available, skipping LPIPS metric")
        use_lpips = False

    # ── Load model ─────────────────────────────────────────────────────────────
    print(f"Loading PLY: {args.ply_path}")
    splats = load_ply_to_splats(args.ply_path)
    means     = splats["means"].to(device)
    scales    = splats["scales"].to(device)
    quats     = splats["quats"].to(device)
    opacities = splats["opacities"].to(device)
    sh0       = splats["sh0"].to(device)
    shN       = splats["shN"].to(device)
    N = means.shape[0]
    sh_degree = round(np.sqrt(shN.shape[1] + 1) - 1)
    print(f"  {N} Gaussians, SH degree={sh_degree}")

    # ── Load dataset ───────────────────────────────────────────────────────────
    print(f"Loading dataset: {args.data_dir}")
    colmap_parser = Parser(
        data_dir=args.data_dir,
        factor=args.data_factor,
        normalize=True,
        test_every=args.test_every,
    )
    test_indices = [i for i in range(len(colmap_parser.image_names))
                    if i % args.test_every == 0]
    print(f"  {len(test_indices)} test views")

    # ── Evaluate ───────────────────────────────────────────────────────────────
    psnrs, ssims, lpipss = [], [], []

    for i, idx in enumerate(tqdm(test_indices, desc="Evaluating")):
        # Load GT image
        img_gt = imageio.imread(colmap_parser.image_paths[idx])[..., :3]
        camera_id = colmap_parser.camera_ids[idx]
        params = colmap_parser.params_dict[camera_id]
        if len(params) > 0:
            import cv2
            mapx = colmap_parser.mapx_dict[camera_id]
            mapy = colmap_parser.mapy_dict[camera_id]
            img_gt = cv2.remap(img_gt, mapx, mapy, cv2.INTER_LINEAR)
            x, y, w, h = colmap_parser.roi_undist_dict[camera_id]
            img_gt = img_gt[y:y+h, x:x+w]

        H, W = img_gt.shape[:2]
        K_np = colmap_parser.Ks_dict[camera_id]
        K_t = torch.from_numpy(K_np).float()
        c2w = torch.from_numpy(colmap_parser.camtoworlds[idx]).float()

        gt_tensor = torch.from_numpy(img_gt).float().to(device) / 255.0  # [H, W, 3]

        # Render
        pred = render_view(
            means, quats, scales, opacities, sh0, shN, sh_degree,
            c2w, K_t, W, H, device, bg_color=args.bg_color,
        )

        psnr = compute_psnr(pred, gt_tensor)
        ssim = compute_ssim(pred, gt_tensor)
        psnrs.append(psnr)
        ssims.append(ssim)

        if use_lpips:
            pred_lp = pred.permute(2, 0, 1).unsqueeze(0) * 2 - 1
            gt_lp   = gt_tensor.permute(2, 0, 1).unsqueeze(0) * 2 - 1
            lp = lpips_fn(pred_lp, gt_lp).item()
            lpipss.append(lp)

        if args.save_images:
            img_name = os.path.basename(colmap_parser.image_names[idx])
            pred_np = (pred.cpu().numpy() * 255).astype(np.uint8)
            imageio.imwrite(os.path.join(args.output_dir, "renders", img_name), pred_np)

    # ── Report ─────────────────────────────────────────────────────────────────
    mean_psnr = float(np.mean(psnrs))
    mean_ssim = float(np.mean(ssims))
    mean_lpips = float(np.mean(lpipss)) if lpipss else None

    print("\n" + "=" * 50)
    print(f"Results on {len(test_indices)} test views:")
    print(f"  PSNR  : {mean_psnr:.4f}")
    print(f"  SSIM  : {mean_ssim:.4f}")
    if mean_lpips is not None:
        print(f"  LPIPS : {mean_lpips:.4f}")
    print("=" * 50)

    # Paper targets:
    # Rubble  CityGS: PSNR 25.77, SSIM 0.813, LPIPS 0.228
    # Building CityGS: PSNR 21.55, SSIM 0.778, LPIPS 0.246

    results = {
        "ply_path": args.ply_path,
        "n_test_views": len(test_indices),
        "n_gaussians": N,
        "psnr": mean_psnr,
        "ssim": mean_ssim,
        "lpips": mean_lpips,
        "per_image_psnr": psnrs,
        "per_image_ssim": ssims,
        "per_image_lpips": lpipss if lpipss else [],
    }
    out_json = os.path.join(args.output_dir, "metrics.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved metrics to {out_json}")


if __name__ == "__main__":
    main()
