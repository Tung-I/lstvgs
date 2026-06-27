"""
Convert Mega-NeRF dataset format (rubble-pixsfm / building-pixsfm) to COLMAP text format
so that gsplat's simple_trainer.py can load it via --data-type colmap.

Mega-NeRF format:
  <scene>/train/metadata/XXXXXX.pt  -> dict: H, W, c2w [3x4], intrinsics [fx,fy,cx,cy], distortion [k1,...]
  <scene>/train/rgbs/XXXXXX.jpg
  <scene>/val/metadata/XXXXXX.pt
  <scene>/val/rgbs/XXXXXX.jpg
  <scene>/coordinates.pt            -> {origin_drb, pose_scale_factor}

Output COLMAP layout (written to <out_dir>):
  images/          symlinks to all train+val images
  sparse/0/
    cameras.txt    one OPENCV camera per unique (H,W,intrinsics) tuple
    images.txt     one entry per image with w2c quaternion+translation
    points3D.txt   empty (gsplat will use --init-type random)
"""

import argparse
import os
import shutil
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation


def c2w_to_colmap_qvec_tvec(c2w: np.ndarray):
    """
    Convert a 3x4 camera-to-world matrix to COLMAP's w2c quaternion + translation.
    COLMAP stores extrinsics as the transformation from world to camera:
        p_cam = R_w2c @ p_world + t_w2c
    """
    R_c2w = c2w[:3, :3]   # 3x3
    t_c2w = c2w[:3, 3]    # 3,

    R_w2c = R_c2w.T
    t_w2c = -R_w2c @ t_c2w

    # scipy gives [x, y, z, w]; COLMAP wants [w, x, y, z]
    r = Rotation.from_matrix(R_w2c)
    xyzw = r.as_quat()
    qvec = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])
    return qvec, t_w2c


def collect_images(scene_dir: Path, split: str):
    """Return sorted list of (stem, metadata_path, image_path)."""
    meta_dir = scene_dir / split / "metadata"
    rgb_dir  = scene_dir / split / "rgbs"
    entries = []
    for meta_path in sorted(meta_dir.glob("*.pt")):
        stem = meta_path.stem
        # try .jpg then .png
        for ext in [".jpg", ".JPG", ".png", ".PNG"]:
            img_path = rgb_dir / (stem + ext)
            if img_path.exists():
                entries.append((stem, meta_path, img_path))
                break
    return entries


def build_camera_id(H, W, intrinsics):
    """Unique camera key based on resolution + focal length (rounded to 2dp)."""
    fx, fy, cx, cy = [round(float(v), 2) for v in intrinsics]
    return (H, W, fx, fy, cx, cy)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scene_dir", help="Path to rubble-pixsfm or building-pixsfm")
    parser.add_argument("out_dir",   help="Output directory (will be created)")
    parser.add_argument("--splits", nargs="+", default=["train", "val"],
                        help="Which splits to include (default: train val)")
    args = parser.parse_args()

    scene_dir = Path(args.scene_dir)
    out_dir   = Path(args.out_dir)
    images_dir = out_dir / "images"
    sparse_dir = out_dir / "sparse" / "0"
    images_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)

    # ── collect all images across requested splits ──────────────────────────
    all_entries = []  # (name_in_colmap, meta_path, img_path)
    for split in args.splits:
        split_dir = scene_dir / split
        if not split_dir.exists():
            print(f"[warn] split '{split}' not found, skipping")
            continue
        entries = collect_images(scene_dir, split)
        for stem, meta_path, img_path in entries:
            colmap_name = f"{split}_{stem}{img_path.suffix}"
            all_entries.append((colmap_name, meta_path, img_path))
    print(f"Total images: {len(all_entries)}")

    # ── symlink images ───────────────────────────────────────────────────────
    for colmap_name, _, img_path in all_entries:
        dst = images_dir / colmap_name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.symlink(img_path.resolve(), dst)

    # ── build camera registry + image records ───────────────────────────────
    camera_registry = {}   # key -> camera_id (1-indexed)
    image_records   = []   # list of dicts

    for image_id, (colmap_name, meta_path, _) in enumerate(all_entries, start=1):
        meta = torch.load(meta_path, map_location="cpu", weights_only=False)
        H          = int(meta["H"])
        W          = int(meta["W"])
        intrinsics = meta["intrinsics"].numpy().astype(float)   # [fx, fy, cx, cy]
        c2w        = meta["c2w"].numpy().astype(float)          # [3, 4]

        cam_key = build_camera_id(H, W, intrinsics)
        if cam_key not in camera_registry:
            camera_registry[cam_key] = len(camera_registry) + 1

        qvec, tvec = c2w_to_colmap_qvec_tvec(c2w)
        image_records.append({
            "image_id":  image_id,
            "qvec":      qvec,
            "tvec":      tvec,
            "camera_id": camera_registry[cam_key],
            "name":      colmap_name,
        })

    # ── write cameras.txt ────────────────────────────────────────────────────
    # Use PINHOLE model: Mega-NeRF k1 distortion is negligible (~-0.002) and
    # gsplat's COLMAP parser warns on non-PINHOLE cameras.
    cameras_path = sparse_dir / "cameras.txt"
    with open(cameras_path, "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: {len(camera_registry)}\n")
        for (H, W, fx, fy, cx, cy), cam_id in camera_registry.items():
            f.write(f"{cam_id} PINHOLE {W} {H} {fx} {fy} {cx} {cy}\n")
    print(f"Wrote {len(camera_registry)} camera(s) → {cameras_path}")

    # ── write images.txt ─────────────────────────────────────────────────────
    images_path = sparse_dir / "images.txt"
    with open(images_path, "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(image_records)}\n")
        for rec in image_records:
            qw, qx, qy, qz = rec["qvec"]
            tx, ty, tz      = rec["tvec"]
            f.write(
                f"{rec['image_id']} {qw:.9f} {qx:.9f} {qy:.9f} {qz:.9f} "
                f"{tx:.9f} {ty:.9f} {tz:.9f} {rec['camera_id']} {rec['name']}\n"
            )
            f.write("\n")  # empty 2D-points line
    print(f"Wrote {len(image_records)} image records → {images_path}")

    # ── write empty points3D.txt ─────────────────────────────────────────────
    points_path = sparse_dir / "points3D.txt"
    with open(points_path, "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
        f.write("# Number of points: 0\n")
    print(f"Wrote empty points3D.txt → {points_path}")
    print("Done. Use --init-type random when training with gsplat (no SFM points).")


if __name__ == "__main__":
    main()
