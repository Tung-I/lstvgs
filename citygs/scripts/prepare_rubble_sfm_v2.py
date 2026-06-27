"""
Prepare rubble_sfm_v2 dataset for vanilla 3DGS with SFM init.

Problem with v1: CityGS points3D are in metric COLMAP space; our images.txt
poses are pre-normalized to unit sphere — two incompatible world frames.

Fix: use the CityGS reconstruction directly (cameras.bin + images.bin +
points3D.bin, all consistent in metric space), but create image symlinks
that match the CityGS filenames:
  CityGS "000001.jpg" → rubble/images/train_000001.jpg  (1-to-1 by name)
  CityGS "000001.jpg" → rubble/images_4/train_000001.jpg (same for factor-4)

The 21 val images (val_000000.jpg etc.) are not in the CityGS train
reconstruction; test images will instead be selected from the 1657 train
frames via --test-every=8, giving ~207 held-out views.
"""

import os
from pathlib import Path

RUBBLE_DIR  = Path("/work/pi_rsitaram_umass_edu/tungi/datasets/rubble")
CITYGS_DIR  = Path("/work/pi_rsitaram_umass_edu/tungi/datasets/cityGSdata/colmap_results/rubble/train/sparse/0")
OUT_DIR     = Path("/home/tungichen_umass_edu/lstvgs-copy/datasets/rubble_sfm_v2")


def make_image_symlinks(src_dir, dst_dir, cg_names):
    """Create dst_dir/NNNNNN.jpg -> src_dir/train_NNNNNN.jpg symlinks."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    created = skipped = 0
    for cg_name in cg_names:
        stem = cg_name.replace(".jpg", "")
        src  = src_dir / f"train_{stem}.jpg"
        dst  = dst_dir / cg_name
        if not dst.exists():
            if src.exists():
                os.symlink(src, dst)
                created += 1
            else:
                print(f"  WARNING: source missing: {src}")
        else:
            skipped += 1
    print(f"  {dst_dir.name}: {created} created, {skipped} already existed")


def main():
    import pycolmap

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Read CityGS reconstruction to get image names ─────────────────────
    r_cg = pycolmap.Reconstruction(str(CITYGS_DIR))
    cg_names = sorted([img.name for img in r_cg.images.values()])
    print(f"CityGS reconstruction: {len(r_cg.cameras)} cameras, "
          f"{len(r_cg.images)} images, {len(r_cg.points3D):,} points")
    print(f"Image names: {cg_names[:3]} ... {cg_names[-2:]}")

    # ── 2. Symlink images and images_4 with CityGS filenames ─────────────────
    print("\nCreating image symlinks:")
    make_image_symlinks(RUBBLE_DIR / "images",   OUT_DIR / "images",   cg_names)
    make_image_symlinks(RUBBLE_DIR / "images_4", OUT_DIR / "images_4", cg_names)

    # ── 3. Symlink sparse/0 directly to CityGS binary files ──────────────────
    sparse0 = OUT_DIR / "sparse" / "0"
    sparse0.mkdir(parents=True, exist_ok=True)
    for fname in ["cameras.bin", "images.bin", "points3D.bin"]:
        src = CITYGS_DIR / fname
        dst = sparse0 / fname
        if not dst.exists():
            os.symlink(src, dst)
            print(f"Symlinked sparse/0/{fname}")
        else:
            print(f"Already exists: sparse/0/{fname}")

    # ── 4. Verify ─────────────────────────────────────────────────────────────
    print("\nVerifying merged dataset with pycolmap:")
    r = pycolmap.Reconstruction(str(sparse0))
    print(f"  cameras={len(r.cameras)}, images={len(r.images)}, "
          f"points3D={len(r.points3D):,}")

    import numpy as np
    pts = np.array([p.xyz for p in r.points3D.values()])
    cams = np.array([img.projection_center() for img in r.images.values()])
    print(f"  Points  X=[{pts[:,0].min():.1f},{pts[:,0].max():.1f}] "
          f"Y=[{pts[:,1].min():.1f},{pts[:,1].max():.1f}] "
          f"Z=[{pts[:,2].min():.1f},{pts[:,2].max():.1f}]")
    print(f"  Cameras X=[{cams[:,0].min():.1f},{cams[:,0].max():.1f}] "
          f"Y=[{cams[:,1].min():.1f},{cams[:,1].max():.1f}] "
          f"Z=[{cams[:,2].min():.1f},{cams[:,2].max():.1f}]")

    n_imgs = len(list((OUT_DIR / "images").iterdir()))
    n_imgs4 = len(list((OUT_DIR / "images_4").iterdir()))
    print(f"\n  images/ count:   {n_imgs}")
    print(f"  images_4/ count: {n_imgs4}")
    print(f"\nDataset ready at: {OUT_DIR}")


if __name__ == "__main__":
    main()
