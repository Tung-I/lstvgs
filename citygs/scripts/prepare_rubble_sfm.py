"""
Prepare rubble_sfm dataset directory for vanilla 3DGS training with SFM init.

Merges:
  - Camera poses: our existing rubble dataset (cameras.txt + images.txt, 1678 images)
  - 3D points:    CityGaussian pre-generated COLMAP results (points3D.bin, 1.69M pts)

The CityGS images.bin uses different filenames (000001.jpg vs train_000001.jpg),
so we cannot use their cameras/images. Instead we:
  1. Copy cameras.txt + images.txt from our rubble dataset verbatim
  2. Read CityGS points3D.bin and write as points3D.txt (no tracks, since track
     elements reference CityGS image IDs that don't match ours — pycolmap skips
     missing image_ids gracefully, so only XYZ/RGB matter for sfm init)
"""

import os
import shutil
import struct
from pathlib import Path

RUBBLE_DIR = Path("/work/pi_rsitaram_umass_edu/tungi/datasets/rubble")
CITYGS_COLMAP_DIR = Path(
    "/work/pi_rsitaram_umass_edu/tungi/datasets/cityGSdata/colmap_results/rubble/train/sparse/0"
)
OUT_DIR = Path("/home/tungichen_umass_edu/lctvgs-copy/datasets/rubble_sfm")


def read_points3d_bin(path):
    """Read COLMAP points3D.bin, return list of (id, x, y, z, r, g, b, error)."""
    points = []
    with open(path, "rb") as f:
        num_points = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_points):
            point_id = struct.unpack("<Q", f.read(8))[0]
            xyz = struct.unpack("<ddd", f.read(24))
            rgb = struct.unpack("<BBB", f.read(3))
            error = struct.unpack("<d", f.read(8))[0]
            track_len = struct.unpack("<Q", f.read(8))[0]
            # Skip track elements (image_id, point2D_idx pairs)
            f.read(track_len * 8)
            points.append((point_id, xyz[0], xyz[1], xyz[2],
                           rgb[0], rgb[1], rgb[2], error))
    return points


def write_points3d_txt(points, path):
    """Write points as COLMAP points3D.txt without track elements."""
    with open(path, "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write(f"# Number of points: {len(points)}, mean track length: 0\n")
        for pt in points:
            pid, x, y, z, r, g, b, err = pt
            f.write(f"{pid} {x:.6f} {y:.6f} {z:.6f} {r} {g} {b} {err:.6f}\n")


def main():
    sparse0 = OUT_DIR / "sparse" / "0"
    sparse0.mkdir(parents=True, exist_ok=True)

    # ── 1. Symlink image directories ──────────────────────────────────────────
    for subdir in ["images", "images_4", "images_4_png"]:
        src = RUBBLE_DIR / subdir
        dst = OUT_DIR / subdir
        if src.exists() and not dst.exists():
            os.symlink(src, dst)
            print(f"Symlinked {subdir} -> {src}")
        elif dst.exists():
            print(f"Already exists: {dst}")

    # ── 2. Copy cameras.txt and images.txt from our reconstruction ────────────
    src_sparse = RUBBLE_DIR / "sparse" / "0"
    for fname in ["cameras.txt", "images.txt"]:
        src = src_sparse / fname
        dst = sparse0 / fname
        if not dst.exists():
            shutil.copy2(src, dst)
            print(f"Copied {fname}")
        else:
            print(f"Already exists: {fname}")

    # ── 3. Convert CityGS points3D.bin → points3D.txt (no tracks) ────────────
    pts_dst = sparse0 / "points3D.txt"
    if not pts_dst.exists():
        pts_bin = CITYGS_COLMAP_DIR / "points3D.bin"
        print(f"Reading {pts_bin} ...")
        points = read_points3d_bin(pts_bin)
        print(f"Read {len(points):,} points. Writing points3D.txt ...")
        write_points3d_txt(points, pts_dst)
        print(f"Written: {pts_dst}")
    else:
        print(f"Already exists: points3D.txt")

    # ── 4. Quick verification via pycolmap ────────────────────────────────────
    print("\nVerifying with pycolmap ...")
    import pycolmap
    r = pycolmap.Reconstruction(str(sparse0))
    print(
        f"  cameras={len(r.cameras)}, images={len(r.images)}, "
        f"points3D={len(r.points3D):,}"
    )
    if len(r.points3D) == 0:
        print("  WARNING: pycolmap read 0 points — check points3D.txt format!")
    else:
        sample = list(r.points3D.values())[0]
        print(f"  Sample point: xyz={sample.xyz}, color={sample.color}")

    print("\nDone. Dataset ready at:", OUT_DIR)


if __name__ == "__main__":
    main()
