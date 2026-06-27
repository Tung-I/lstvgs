"""
Set up the rubble_citygs dataset directory for CityGS reproduction.

Reads CityGS COLMAP binary files directly (no pycolmap Reconstruction building)
and creates a merged dataset:
  - train: 1657 images + 1.69M SFM points
  - val: 21 images (poses only, no new points)

Output:
  datasets/rubble_citygs/
  ├── images/         ← symlinks to all 1678 full-res images (000001.jpg etc.)
  └── sparse/0/
      ├── cameras.bin
      ├── images.bin  (1678 images = 1657 train + 21 val)
      └── points3D.bin (1.69M SFM points from train)

Train with: --data-dir .../rubble_citygs --data-factor 4 --test-every 83
→ exactly 21 test views matching the CityGS/MegaNeRF val split.
"""
import os
import shutil
import struct

WORK        = "/work/pi_rsitaram_umass_edu/tungi"
PIXSFM_TRAIN = f"{WORK}/datasets/rubble-pixsfm/train/rgbs"
PIXSFM_VAL   = f"{WORK}/datasets/rubble-pixsfm/val/rgbs"
COLMAP_TRAIN = f"{WORK}/datasets/cityGSdata/colmap_results/rubble/train/sparse/0"
COLMAP_VAL   = f"{WORK}/datasets/cityGSdata/colmap_results/rubble/val/sparse/0"
OUT_DIR      = f"{WORK}/datasets/rubble_citygs"
OUT_IMAGES   = f"{OUT_DIR}/images"
OUT_SPARSE   = f"{OUT_DIR}/sparse/0"


# ── COLMAP binary I/O ──────────────────────────────────────────────────────────

def read_cameras_bin(path):
    """Returns dict {camera_id: dict(model_id, width, height, params)}"""
    cameras = {}
    with open(path, "rb") as f:
        num_cameras = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_cameras):
            cam_id   = struct.unpack("<I", f.read(4))[0]
            model_id = struct.unpack("<i", f.read(4))[0]
            width    = struct.unpack("<Q", f.read(8))[0]
            height   = struct.unpack("<Q", f.read(8))[0]
            # Number of params depends on model:
            # 0=SIMPLE_PINHOLE(3), 1=PINHOLE(4), 2=SIMPLE_RADIAL(4),
            # 3=RADIAL(5), 4=OPENCV(8), 5=OPENCV_FISHEYE(8)
            num_params = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8}.get(model_id, 4)
            params = list(struct.unpack(f"<{num_params}d", f.read(8 * num_params)))
            cameras[cam_id] = {"model_id": model_id, "width": width,
                               "height": height, "params": params}
    return cameras


def read_images_bin(path):
    """Returns list of dicts: {image_id, qvec, tvec, camera_id, name, points2D}"""
    images = []
    with open(path, "rb") as f:
        num_images = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_images):
            image_id  = struct.unpack("<I", f.read(4))[0]
            qvec      = struct.unpack("<4d", f.read(32))   # qw qx qy qz
            tvec      = struct.unpack("<3d", f.read(24))   # tx ty tz
            camera_id = struct.unpack("<I", f.read(4))[0]
            # Read null-terminated name
            name_bytes = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name_bytes += c
            name = name_bytes.decode("utf-8")
            num_pts = struct.unpack("<Q", f.read(8))[0]
            points2D = []
            for _ in range(num_pts):
                x = struct.unpack("<d", f.read(8))[0]
                y = struct.unpack("<d", f.read(8))[0]
                pt3d_id = struct.unpack("<Q", f.read(8))[0]
                points2D.append((x, y, pt3d_id))
            images.append({
                "image_id": image_id, "qvec": qvec, "tvec": tvec,
                "camera_id": camera_id, "name": name, "points2D": points2D,
            })
    return images


def write_cameras_bin(path, cameras):
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(cameras)))
        for cam_id, c in sorted(cameras.items()):
            f.write(struct.pack("<I", cam_id))
            f.write(struct.pack("<i", c["model_id"]))
            f.write(struct.pack("<Q", c["width"]))
            f.write(struct.pack("<Q", c["height"]))
            f.write(struct.pack(f"<{len(c['params'])}d", *c["params"]))


def write_images_bin(path, images):
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(images)))
        for img in images:
            f.write(struct.pack("<I", img["image_id"]))
            f.write(struct.pack("<4d", *img["qvec"]))
            f.write(struct.pack("<3d", *img["tvec"]))
            f.write(struct.pack("<I", img["camera_id"]))
            f.write(img["name"].encode("utf-8") + b"\x00")
            pts = img["points2D"]
            f.write(struct.pack("<Q", len(pts)))
            for x, y, pt3d_id in pts:
                f.write(struct.pack("<d", x))
                f.write(struct.pack("<d", y))
                f.write(struct.pack("<Q", pt3d_id))


def main():
    os.makedirs(OUT_IMAGES, exist_ok=True)
    os.makedirs(OUT_SPARSE, exist_ok=True)

    # ── 1. Copy cameras.bin from train (authoritative) ─────────────────────────
    src_cam = os.path.join(COLMAP_TRAIN, "cameras.bin")
    dst_cam = os.path.join(OUT_SPARSE, "cameras.bin")
    shutil.copy2(src_cam, dst_cam)
    cameras = read_cameras_bin(dst_cam)
    cam = list(cameras.values())[0]
    print(f"Camera: model_id={cam['model_id']} {cam['width']}x{cam['height']} "
          f"params={cam['params']}")

    # ── 2. Read train images ───────────────────────────────────────────────────
    print(f"Reading train images from {COLMAP_TRAIN}/images.bin ...")
    train_images = read_images_bin(os.path.join(COLMAP_TRAIN, "images.bin"))
    print(f"  {len(train_images)} train images")

    # ── 3. Read val images ────────────────────────────────────────────────────
    print(f"Reading val images from {COLMAP_VAL}/images.bin ...")
    val_images_raw = read_images_bin(os.path.join(COLMAP_VAL, "images.bin"))
    print(f"  {len(val_images_raw)} val images")

    # ── 4. Merge: keep original train IDs, append val with non-conflicting IDs ─
    # CRITICAL: do NOT reassign train image IDs — points3D.bin track references
    # the original IDs. Reassigning breaks pycolmap's cross-validation check.
    train_ids = set(img["image_id"] for img in train_images)
    max_train_id = max(train_ids)
    cam_id_to_use = list(cameras.keys())[0]

    merged_images = list(train_images)  # keep original IDs
    new_id = max_train_id + 1
    for img in sorted(val_images_raw, key=lambda x: x["name"]):
        img_copy = dict(img)
        img_copy["image_id"] = new_id
        img_copy["camera_id"] = cam_id_to_use
        img_copy["points2D"] = []  # val's 31-point reconstruction is irrelevant
        merged_images.append(img_copy)
        new_id += 1

    print(f"Merged: {len(merged_images)} images total")
    names = sorted(img["name"] for img in merged_images)
    print(f"  First 5: {names[:5]}")
    print(f"  Last 5:  {names[-5:]}")

    # ── 5. Write merged images.bin ────────────────────────────────────────────
    dst_img = os.path.join(OUT_SPARSE, "images.bin")
    print(f"Writing {dst_img} ...")
    write_images_bin(dst_img, merged_images)
    print("  Done.")

    # ── 6. Copy points3D.bin from train (1.69M points) ───────────────────────
    src_pts = os.path.join(COLMAP_TRAIN, "points3D.bin")
    dst_pts = os.path.join(OUT_SPARSE, "points3D.bin")
    print(f"Copying points3D.bin ({os.path.getsize(src_pts)/1024/1024:.1f} MB) ...")
    shutil.copy2(src_pts, dst_pts)
    print("  Done.")

    # ── 7. Create image symlinks ───────────────────────────────────────────────
    print(f"Creating image symlinks in {OUT_IMAGES} ...")
    n_linked = 0
    for src_dir in [PIXSFM_TRAIN, PIXSFM_VAL]:
        for fname in sorted(os.listdir(src_dir)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            src = os.path.join(src_dir, fname)
            dst = os.path.join(OUT_IMAGES, fname)
            if os.path.lexists(dst):
                os.unlink(dst)
            os.symlink(src, dst)
            n_linked += 1
    print(f"  Linked {n_linked} images.")

    # ── 8. Verify alignment ────────────────────────────────────────────────────
    colmap_names = set(img["name"] for img in merged_images)
    image_files  = set(os.listdir(OUT_IMAGES))
    missing = colmap_names - image_files
    extra   = image_files - colmap_names
    print(f"\nVerification:")
    print(f"  COLMAP image names: {len(colmap_names)}")
    print(f"  Image files on disk: {len(image_files)}")
    if missing:
        print(f"  WARNING: {len(missing)} COLMAP names missing from images/: {list(missing)[:5]}")
    else:
        print(f"  All COLMAP images have corresponding files. ✓")
    if extra:
        print(f"  Note: {len(extra)} extra files in images/ (not in COLMAP): {list(extra)[:3]}")

    # ── 9. Print train/test split info ────────────────────────────────────────
    all_sorted = sorted(colmap_names)
    test_every = 83
    test_names = [n for i, n in enumerate(all_sorted) if i % test_every == 0]
    val_names  = sorted(os.listdir(PIXSFM_VAL))
    test_set   = set(test_names)
    val_set    = set(val_names)
    overlap    = test_set & val_set
    print(f"\nTrain/test split with test_every={test_every}:")
    print(f"  Total images: {len(all_sorted)}")
    print(f"  Test frames (every {test_every}): {len(test_names)}")
    print(f"  Actual val frames: {len(val_names)}")
    print(f"  Overlap (test_every={test_every} ∩ val): {len(overlap)}")
    if overlap == val_set:
        print(f"  ✓ test_every={test_every} perfectly recovers the CityGS val split!")
    else:
        print(f"  Note: test_every={test_every} gives {len(test_names)} frames, "
              f"not all matching val. Consider test_every=80.")
        # Try test_every=80
        test_80 = [n for i, n in enumerate(all_sorted) if i % 80 == 0]
        overlap_80 = set(test_80) & val_set
        print(f"  test_every=80 overlap with val: {len(overlap_80)}/{len(val_names)}")

    print(f"\n=== Dataset ready at {OUT_DIR} ===")
    print(f"Training command (coarse):")
    print(f"  python gsplat/examples/simple_trainer.py default \\")
    print(f"    --data-dir {OUT_DIR} --data-factor 4 --test-every 83 \\")
    print(f"    --init-type sfm --max-steps 30000")


if __name__ == "__main__":
    main()
