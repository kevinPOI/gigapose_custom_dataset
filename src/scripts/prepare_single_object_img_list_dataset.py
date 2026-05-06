import argparse
import json
import os
import re
import struct
import tarfile
from pathlib import Path

import numpy as np
from PIL import Image


DEFAULT_CAM_K = [572.4114, 0.0, 320.0, 0.0, 573.57043, 240.0, 0.0, 0.0, 1.0]


def binary_mask_to_rle(binary_mask):
    rle = {"counts": [], "size": list(binary_mask.shape)}
    counts = rle["counts"]
    mask = binary_mask.ravel(order="F")
    if len(mask) > 0 and mask[0] == 1:
        counts.append(0)
    if len(mask) > 0:
        mask_changes = mask[:-1] != mask[1:]
        changes_idx = np.where(np.concatenate(([True], mask_changes, [True]), 0))[0]
        counts.extend(np.diff(changes_idx).tolist())
    return rle


def parse_object_id(mesh_path, explicit_obj_id):
    if explicit_obj_id is not None:
        return int(explicit_obj_id)
    numbers = re.findall(r"\d+", mesh_path.stem)
    if numbers:
        return int(numbers[-1])
    return 1


def parse_image_id(image_path, used_ids, fallback_id):
    numbers = re.findall(r"\d+", image_path.stem)
    image_id = int(numbers[-1]) if numbers else fallback_id
    while image_id in used_ids:
        image_id += 1
    used_ids.add(image_id)
    return image_id


def symlink_or_replace(src, dst):
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(src, dst)


def load_stl_triangles(mesh_path):
    data = mesh_path.read_bytes()
    if len(data) >= 84:
        tri_count = struct.unpack_from("<I", data, 80)[0]
        expected_size = 84 + tri_count * 50
        if expected_size == len(data):
            triangles = np.empty((tri_count, 3, 3), dtype=np.float64)
            offset = 84
            for tri_idx in range(tri_count):
                values = struct.unpack_from("<12fH", data, offset)
                triangles[tri_idx] = np.asarray(values[3:12], dtype=np.float64).reshape(3, 3)
                offset += 50
            return triangles

    vertices = []
    for line in data.decode("utf-8", errors="ignore").splitlines():
        parts = line.strip().split()
        if len(parts) == 4 and parts[0].lower() == "vertex":
            vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if len(vertices) % 3 != 0 or not vertices:
        raise ValueError(f"Could not parse STL triangles from {mesh_path}")
    return np.asarray(vertices, dtype=np.float64).reshape(-1, 3, 3)


def export_stl_as_obj(mesh_path, dst):
    triangles = load_stl_triangles(mesh_path)
    with dst.open("w") as f:
        f.write(f"o {mesh_path.stem}\n")
        for vertex in triangles.reshape(-1, 3):
            f.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
        for face_idx in range(len(triangles)):
            base = face_idx * 3 + 1
            f.write(f"f {base} {base + 1} {base + 2}\n")
    extent = np.ptp(triangles.reshape(-1, 3), axis=0)
    diameter = float(np.linalg.norm(extent))
    if not np.isfinite(diameter) or diameter <= 0:
        diameter = 1.0
    return diameter


def ensure_obj_mesh(mesh_path, model_dir, object_id):
    model_dir.mkdir(parents=True, exist_ok=True)
    dst = model_dir / f"obj_{object_id:06d}.obj"
    if mesh_path.suffix.lower() == ".obj":
        symlink_or_replace(mesh_path, dst)
        return dst, 1.0
    if mesh_path.suffix.lower() == ".stl":
        diameter = export_stl_as_obj(mesh_path, dst)
        return dst, diameter

    try:
        import trimesh
    except ImportError as exc:
        raise SystemExit(
            "This mesh is not an OBJ. Install trimesh in this environment, or convert "
            f"{mesh_path} to {dst} before preparing the dataset."
        ) from exc

    mesh = trimesh.load_mesh(mesh_path, force="mesh")
    mesh.export(dst)
    diameter = float(np.linalg.norm(np.asarray(mesh.extents, dtype=np.float64)))
    if not np.isfinite(diameter) or diameter <= 0:
        diameter = 1.0
    return dst, diameter


def write_camera(path, cam_k):
    camera = {
        "cam_K": cam_k,
        "cam_R_w2c": np.eye(3).reshape(-1).tolist(),
        "cam_t_w2c": [0.0, 0.0, 0.0],
    }
    path.write_text(json.dumps(camera))


def add_file_to_tar(tar, path, arcname):
    real_path = path.resolve()
    info = tar.gettarinfo(str(real_path), arcname=arcname)
    info.type = tarfile.REGTYPE
    info.linkname = ""
    with real_path.open("rb") as f:
        tar.addfile(info, f)


def write_webdataset_shard(imagewise_dir, test_dir, image_keys):
    test_dir.mkdir(parents=True, exist_ok=True)
    shard_path = test_dir / "shard-000000.tar"
    key_to_shard = {}
    with tarfile.open(shard_path, "w") as tar:
        for image_key in image_keys:
            for ext in ["rgb.png", "depth.png", "camera.json"]:
                add_file_to_tar(tar, imagewise_dir / f"{image_key}.{ext}", f"{image_key}.{ext}")
            key_to_shard[image_key] = 0
    (test_dir / "key_to_shard.json").write_text(json.dumps(key_to_shard))


def main():
    parser = argparse.ArgumentParser(
        description="Prepare a one-object image-list dataset for GigaPose inference."
    )
    parser.add_argument("--image-list", type=Path, required=True)
    parser.add_argument("--mesh-path", type=Path, required=True)
    parser.add_argument("--dataset-name", default="single_object")
    parser.add_argument("--dataset-root", type=Path, default=Path("gigaPose_datasets/datasets"))
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--obj-id", type=int, default=None)
    parser.add_argument(
        "--cam-k",
        type=float,
        nargs=9,
        default=DEFAULT_CAM_K,
        metavar=("fx", "s0", "cx", "s1", "fy", "cy", "s2", "s3", "s4"),
    )
    args = parser.parse_args()

    repo_root = args.repo_root.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    dataset_dir = dataset_root / args.dataset_name
    imagewise_dir = dataset_dir / "test_imagewise"
    test_dir = dataset_dir / "test"
    model_dir = dataset_dir / "models"
    detections_dir = (
        dataset_root / "default_detections" / "core19_model_based_unseen" / "cnos-fastsam"
    )

    image_paths = [
        Path(line.strip()).expanduser().resolve()
        for line in args.image_list.read_text().splitlines()
        if line.strip()
    ]
    if not image_paths:
        raise SystemExit(f"No images found in {args.image_list}")

    mesh_path = args.mesh_path.expanduser().resolve()
    object_id = parse_object_id(mesh_path, args.obj_id)
    _, diameter = ensure_obj_mesh(mesh_path, model_dir, object_id)

    (dataset_dir / "name_to_obj_id.json").write_text(json.dumps({mesh_path.stem: object_id}, indent=2))
    (dataset_dir / "obj_id_to_name.json").write_text(json.dumps({str(object_id): mesh_path.stem}, indent=2))
    (model_dir / "models_info.json").write_text(
        json.dumps({str(object_id): {"diameter": diameter, "name": mesh_path.stem}}, indent=2)
    )

    detections = []
    targets = []
    image_keys = []
    imagewise_dir.mkdir(parents=True, exist_ok=True)
    used_ids = set()
    for fallback_id, rgb_path in enumerate(image_paths, start=1):
        if not rgb_path.exists():
            raise FileNotFoundError(rgb_path)
        image_id = parse_image_id(rgb_path, used_ids, fallback_id)
        image_key = f"{1:06d}_{image_id:06d}"
        image_keys.append(image_key)

        rgb_dst = imagewise_dir / f"{image_key}.rgb.png"
        symlink_or_replace(rgb_path, rgb_dst)

        rgb = np.array(Image.open(rgb_path).convert("RGB"))
        height, width = rgb.shape[:2]
        Image.fromarray(np.zeros((height, width), dtype=np.uint16)).save(
            imagewise_dir / f"{image_key}.depth.png"
        )
        write_camera(imagewise_dir / f"{image_key}.camera.json", args.cam_k)

        full_mask = np.ones((height, width), dtype=np.uint8)
        detections.append(
            {
                "scene_id": 1,
                "image_id": image_id,
                "category_id": object_id,
                "bbox": [0, 0, width, height],
                "segmentation": binary_mask_to_rle(full_mask),
                "score": 1.0,
                "time": 0.0,
            }
        )
        targets.append({"scene_id": 1, "im_id": image_id, "obj_id": object_id, "inst_count": 1})

    detections_dir.mkdir(parents=True, exist_ok=True)
    detections_path = detections_dir / f"cnos-fastsam_{args.dataset_name}-test_gt_masks.json"
    detections_path.write_text(json.dumps(detections))
    (dataset_dir / "test_targets_bop19.json").write_text(json.dumps(targets, indent=2))

    if test_dir.exists():
        for tar_path in test_dir.glob("*.tar"):
            tar_path.unlink()
        key_to_shard = test_dir / "key_to_shard.json"
        if key_to_shard.exists():
            key_to_shard.unlink()

    write_webdataset_shard(imagewise_dir, test_dir, image_keys)

    print(f"Prepared dataset: {dataset_dir}")
    print(f"Object id: {object_id}")
    print(f"Images: {len(image_paths)}")
    print(f"Detections: {detections_path}")


if __name__ == "__main__":
    main()
