"""
CityGaussian block partitioning for gsplat-based training.

Assigns each training camera to one or more spatial blocks using:
  1. Location-based: cameras whose center falls within the (optionally expanded) block bounding box
  2. SSIM-based: cameras where rendering WITH vs WITHOUT the block's Gaussians differs significantly

Usage:
    python citygs/partition_citygs.py \
        --ply-path results/rubble_citygs_coarse/ply/splat_30000.ply \
        --data-dir /path/to/rubble \
        --data-factor 4 \
        --block-dim 3 1 3 \
        --ssim-threshold 0.12 \
        --output-dir results/rubble_citygs_coarse/partition

Outputs:
    {output-dir}/camera_mask.npy  — bool array [N_train_cams, N_blocks]
    {output-dir}/aabb.npy         — float array [6,] (xmin ymin zmin xmax ymax zmax)
    {output-dir}/partition_info.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# Make sure gsplat examples datasets are importable
SCRIPT_DIR = Path(__file__).parent
EXAMPLES_DIR = SCRIPT_DIR.parent / "gsplat" / "examples"
sys.path.insert(0, str(EXAMPLES_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent / "gsplat"))

from gsplat.exporter import load_ply_to_splats
from gsplat.rendering import rasterization
from datasets.colmap import Parser


def ssim_metric(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    """Compute SSIM between two [H, W, 3] tensors. Returns scalar."""
    C1 = (0.01 ** 2)
    C2 = (0.03 ** 2)
    # Use 11×11 Gaussian kernel approximation via average pooling
    from torch.nn.functional import avg_pool2d
    img1 = img1.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
    img2 = img2.permute(2, 0, 1).unsqueeze(0)
    kernel_size = 11
    pad = kernel_size // 2
    mu1 = avg_pool2d(img1, kernel_size, stride=1, padding=pad)
    mu2 = avg_pool2d(img2, kernel_size, stride=1, padding=pad)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = avg_pool2d(img1 ** 2, kernel_size, stride=1, padding=pad) - mu1_sq
    sigma2_sq = avg_pool2d(img2 ** 2, kernel_size, stride=1, padding=pad) - mu2_sq
    sigma12 = avg_pool2d(img1 * img2, kernel_size, stride=1, padding=pad) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def compute_aabb_from_gaussians(means: torch.Tensor, percentile: float = 98.0, margin: float = 0.1) -> torch.Tensor:
    """Compute AABB from Gaussian positions with margin."""
    lo = torch.tensor(
        [np.percentile(means[:, i].cpu().numpy(), 100.0 - percentile) for i in range(3)],
        dtype=torch.float32,
    )
    hi = torch.tensor(
        [np.percentile(means[:, i].cpu().numpy(), percentile) for i in range(3)],
        dtype=torch.float32,
    )
    extent = hi - lo
    lo = lo - margin * extent
    hi = hi + margin * extent
    return torch.cat([lo, hi])


def contract_to_unisphere(x: torch.Tensor, aabb: torch.Tensor) -> torch.Tensor:
    """Map x from [aabb_min, aabb_max] to [0, 1]³ (inf-norm contraction)."""
    aabb_min = aabb[:3]
    aabb_max = aabb[3:]
    return (x - aabb_min) / (aabb_max - aabb_min)


@torch.no_grad()
def render_single(
    means, quats, scales, opacities, sh0, shN,
    camtoworld, K, width, height, device, sh_degree=3,
):
    """Render a single camera view. Returns [H, W, 3] float tensor."""
    viewmat = torch.linalg.inv(camtoworld.to(device)).unsqueeze(0)  # [1, 4, 4]
    K_t = K.to(device).unsqueeze(0)  # [1, 3, 3]

    # Pass SH coefficients [N, K, 3] directly to rasterization
    sh_coeffs = torch.cat([sh0, shN], dim=1)  # [N, K, 3]

    render_colors, _, _ = rasterization(
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
    return render_colors[0].clamp(0.0, 1.0)  # [H, W, 3]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply-path", required=True, help="Path to coarse model PLY file")
    parser.add_argument("--data-dir", required=True, help="COLMAP dataset directory")
    parser.add_argument("--data-factor", type=int, default=4, help="Image downscale factor")
    parser.add_argument("--test-every", type=int, default=8, help="Test split interval (match coarse training)")
    parser.add_argument("--block-dim", type=int, nargs=3, default=[3, 1, 3], metavar=("BX", "BY", "BZ"))
    parser.add_argument("--ssim-threshold", type=float, default=0.12,
                        help="1-SSIM threshold for block assignment (rubble=0.12, building=0.1)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--aabb", type=float, nargs=6, default=None,
                        help="Manual AABB: xmin ymin zmin xmax ymax zmax (auto-computed if not given)")
    parser.add_argument("--simple-selection", type=float, default=0.0,
                        help="If > 1.0, use expanded-box location assignment instead of SSIM (e.g., 1.5)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-ssim", action="store_true", help="Skip SSIM-based assignment, use only location-based")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = args.device

    # ── Load coarse Gaussians ──────────────────────────────────────────────────
    print(f"Loading coarse PLY: {args.ply_path}")
    splats = load_ply_to_splats(args.ply_path)
    means    = splats["means"].to(device)     # [N, 3]
    scales   = splats["scales"].to(device)    # [N, 3]
    quats    = splats["quats"].to(device)     # [N, 4]
    opacities = splats["opacities"].to(device)  # [N,]
    sh0      = splats["sh0"].to(device)       # [N, 1, 3]
    shN      = splats["shN"].to(device)       # [N, K-1, 3]
    N = means.shape[0]
    sh_degree = round(np.sqrt(shN.shape[1] + 1) - 1)
    print(f"  Loaded {N} Gaussians, SH degree={sh_degree}")

    # ── Load camera data ───────────────────────────────────────────────────────
    print(f"Parsing dataset: {args.data_dir}")
    colmap_parser = Parser(
        data_dir=args.data_dir,
        factor=args.data_factor,
        normalize=True,  # apply similarity transform (same as coarse training)
        test_every=args.test_every,
    )
    train_indices = [i for i in range(len(colmap_parser.image_names))
                     if i % args.test_every != 0]
    camtoworlds = torch.from_numpy(colmap_parser.camtoworlds).float()  # [N_all, 4, 4]
    train_c2w = camtoworlds[train_indices]  # [N_train, 4, 4]
    N_train = len(train_indices)
    print(f"  {N_train} training cameras (of {len(colmap_parser.image_names)} total)")

    # ── Compute AABB ──────────────────────────────────────────────────────────
    if args.aabb is not None:
        aabb = torch.tensor(args.aabb, dtype=torch.float32)
        print(f"Using manual AABB: {aabb.tolist()}")
    else:
        aabb = compute_aabb_from_gaussians(means.cpu(), percentile=98.0, margin=0.1)
        print(f"Auto-computed AABB: {aabb.tolist()}")
    np.save(os.path.join(args.output_dir, "aabb.npy"), aabb.numpy())

    block_dim = args.block_dim
    block_num = block_dim[0] * block_dim[1] * block_dim[2]
    print(f"Block grid: {block_dim} → {block_num} blocks")

    # ── Block partitioning ────────────────────────────────────────────────────
    # Contracted Gaussian positions in [0,1]³
    xyz_contracted = contract_to_unisphere(means.cpu(), aabb)  # [N, 3]

    camera_mask = torch.zeros((N_train, block_num), dtype=torch.bool)

    # Camera positions in contracted space
    cam_centers = train_c2w[:, :3, 3]  # [N_train, 3]
    cam_contracted = contract_to_unisphere(cam_centers, aabb)  # [N_train, 3]

    for block_id in range(block_num):
        # 3D grid index (x fastest, z slowest — matching CityGS convention)
        block_z = block_id // (block_dim[0] * block_dim[1])
        block_y = (block_id % (block_dim[0] * block_dim[1])) // block_dim[0]
        block_x = (block_id % (block_dim[0] * block_dim[1])) % block_dim[0]

        # Nominal block bounds in [0,1]³
        org_min = torch.tensor([
            float(block_x) / block_dim[0],
            float(block_y) / block_dim[1],
            float(block_z) / block_dim[2],
        ])
        org_max = torch.tensor([
            float(block_x + 1) / block_dim[0],
            float(block_y + 1) / block_dim[1],
            float(block_z + 1) / block_dim[2],
        ])

        # Find Gaussians in this block (with adaptive expansion if too few)
        num_threshold = 100
        cur_min, cur_max = org_min.clone(), org_max.clone()
        while True:
            in_block = (
                (xyz_contracted[:, 0] >= cur_min[0]) & (xyz_contracted[:, 0] < cur_max[0]) &
                (xyz_contracted[:, 1] >= cur_min[1]) & (xyz_contracted[:, 1] < cur_max[1]) &
                (xyz_contracted[:, 2] >= cur_min[2]) & (xyz_contracted[:, 2] < cur_max[2])
            )
            if in_block.sum() >= num_threshold:
                break
            cur_min = cur_min - 0.01
            cur_max = cur_max + 0.01

        n_block_gs = in_block.sum().item()
        outside_block = ~in_block

        if args.simple_selection > 1.0:
            # Expanded-box location assignment
            rate = (args.simple_selection - 1.0) / 2.0
            exp_min = org_min - rate * (org_max - org_min)
            exp_max = org_max + rate * (org_max - org_min)
            in_box = (
                (cam_contracted[:, 0] > exp_min[0]) & (cam_contracted[:, 0] < exp_max[0]) &
                (cam_contracted[:, 1] > exp_min[1]) & (cam_contracted[:, 1] < exp_max[1]) &
                (cam_contracted[:, 2] > exp_min[2]) & (cam_contracted[:, 2] < exp_max[2])
            )
            camera_mask[:, block_id] = in_box
            n_assigned = in_box.sum().item()
            print(f"Block {block_id+1}/{block_num} [{block_x},{block_y},{block_z}]: "
                  f"{n_block_gs} Gaussians, {n_assigned} cameras (expanded-box)")
            continue

        # ── SSIM-based assignment ──────────────────────────────────────────────
        # Pre-build masked Gaussian tensors (everything OUTSIDE this block)
        masked_means     = means[outside_block]
        masked_scales    = scales[outside_block]
        masked_quats     = quats[outside_block]
        masked_opacities = opacities[outside_block]
        masked_sh0       = sh0[outside_block]
        masked_shN       = shN[outside_block]

        n_assigned = 0
        for cam_idx in tqdm(range(N_train), desc=f"Block {block_id+1}/{block_num}", leave=False):
            c = cam_contracted[cam_idx]

            # Location-based: camera center inside block → always assign
            if (c[0] > org_min[0] and c[0] < org_max[0] and
                c[1] > org_min[1] and c[1] < org_max[1] and
                c[2] > org_min[2] and c[2] < org_max[2]):
                camera_mask[cam_idx, block_id] = True
                n_assigned += 1
                continue

            if args.no_ssim:
                continue

            # SSIM-based: render with all vs. without block, check difference
            cam_idx_all = train_indices[cam_idx]
            cam_id = colmap_parser.camera_ids[cam_idx_all]
            K_np = colmap_parser.Ks_dict[cam_id]
            K_t = torch.from_numpy(K_np).float()
            w, h = colmap_parser.imsize_dict[cam_id]
            c2w = train_c2w[cam_idx]

            try:
                img_full = render_single(
                    means, quats, scales, opacities, sh0, shN,
                    c2w, K_t, w, h, device, sh_degree
                )
                img_masked = render_single(
                    masked_means, masked_quats, masked_scales, masked_opacities,
                    masked_sh0, masked_shN,
                    c2w, K_t, w, h, device, sh_degree
                )
                ssim_loss = 1.0 - ssim_metric(img_full, img_masked).item()
                if ssim_loss > args.ssim_threshold:
                    camera_mask[cam_idx, block_id] = True
                    n_assigned += 1
            except Exception as e:
                print(f"  Warning: render failed for cam {cam_idx}: {e}")

        print(f"Block {block_id+1}/{block_num} [{block_x},{block_y},{block_z}]: "
              f"{n_block_gs} Gaussians, {n_assigned} cameras assigned")

    # ── Save results ──────────────────────────────────────────────────────────
    camera_mask_np = camera_mask.numpy()
    np.save(os.path.join(args.output_dir, "camera_mask.npy"), camera_mask_np)

    info = {
        "block_dim": block_dim,
        "block_num": block_num,
        "n_train_cameras": N_train,
        "n_gaussians": N,
        "aabb": aabb.tolist(),
        "ssim_threshold": args.ssim_threshold,
        "simple_selection": args.simple_selection,
        "cameras_per_block": [int(camera_mask_np[:, b].sum()) for b in range(block_num)],
    }
    with open(os.path.join(args.output_dir, "partition_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    print("\nPartition complete!")
    for b in range(block_num):
        bz = b // (block_dim[0] * block_dim[1])
        by = (b % (block_dim[0] * block_dim[1])) // block_dim[0]
        bx = (b % (block_dim[0] * block_dim[1])) % block_dim[0]
        print(f"  Block {b:2d} [{bx},{by},{bz}]: {info['cameras_per_block'][b]} cameras")


if __name__ == "__main__":
    main()
