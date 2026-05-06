import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
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


def binary_mask_to_bbox(binary_mask):
    ys, xs = np.where(binary_mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("Empty mask encountered while generating detections.")
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return [x0, y0, x1 - x0 + 1, y1 - y0 + 1]


def parse_color_key(color_key):
    values = color_key.strip()[1:-1].split(",")
    return tuple(int(v.strip()) for v in values)


def find_mesh_names(mesh_dir):
    mesh_paths = sorted(
        [p for p in mesh_dir.iterdir() if p.is_file() and p.suffix.lower() in {".stl", ".obj", ".ply"}],
        key=lambda p: p.stem.lower(),
    )
    return [p.stem for p in mesh_paths]


def build_name_to_obj_id(mesh_names):
    return {name: idx for idx, name in enumerate(sorted(mesh_names), start=1)}


def normalize_object_name(name):
    suffixes = [".stl", ".obj", ".ply"]
    normalized = name.strip()
    lower = normalized.lower()
    for suffix in suffixes:
        if lower.endswith(suffix):
            normalized = normalized[: -len(suffix)]
            break
    return normalized


def extract_object_name_from_obj(obj_path):
    with open(obj_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line.startswith("o ") or line.startswith("g "):
                return normalize_object_name(line.split(maxsplit=1)[1])
    return normalize_object_name(obj_path.stem)


def build_name_to_obj_id_from_model_dir(model_dir):
    obj_paths = sorted(model_dir.glob("obj_*.obj"))
    if not obj_paths:
        return None

    name_to_obj_id = {}
    for obj_path in obj_paths:
        obj_id = int(obj_path.stem.split("_")[-1])
        object_name = extract_object_name_from_obj(obj_path)
        if object_name in name_to_obj_id and name_to_obj_id[object_name] != obj_id:
            raise ValueError(
                f"Duplicate object name with conflicting ids: {object_name} -> "
                f"{name_to_obj_id[object_name]} and {obj_id}"
            )
        name_to_obj_id[object_name] = obj_id
    return name_to_obj_id


def write_models_info(models_info_path, name_to_obj_id):
    models_info_path.parent.mkdir(parents=True, exist_ok=True)
    models_info = {}
    for name, obj_id in sorted(name_to_obj_id.items(), key=lambda x: x[1]):
        models_info[str(obj_id)] = {"diameter": 1.0, "name": name}
    models_info_path.write_text(json.dumps(models_info, indent=2))


def write_mapping_files(dataset_dir, name_to_obj_id):
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "name_to_obj_id.json").write_text(json.dumps(name_to_obj_id, indent=2))
    obj_id_to_name = {str(obj_id): name for name, obj_id in name_to_obj_id.items()}
    (dataset_dir / "obj_id_to_name.json").write_text(json.dumps(obj_id_to_name, indent=2))


def write_imagewise_sample(output_dir, image_key, rgb_path, cam_k):
    output_dir.mkdir(parents=True, exist_ok=True)
    rgb_dst = output_dir / f"{image_key}.rgb.png"
    if not rgb_dst.exists():
        os.symlink(rgb_path, rgb_dst)
    depth_dst = output_dir / f"{image_key}.depth.png"
    if not depth_dst.exists():
        rgb = np.array(Image.open(rgb_path))
        zero_depth = np.zeros(rgb.shape[:2], dtype=np.uint16)
        Image.fromarray(zero_depth).save(depth_dst)
    camera = {
        "cam_K": cam_k,
        "cam_R_w2c": np.eye(3).reshape(-1).tolist(),
        "cam_t_w2c": [0.0, 0.0, 0.0],
    }
    (output_dir / f"{image_key}.camera.json").write_text(json.dumps(camera))


def collect_dataset_annotations(raw_root, name_to_obj_id):
    detections = []
    targets = []
    missing_names = defaultdict(int)

    rgb_paths = sorted(raw_root.glob("rgb_*.png"))
    for rgb_path in rgb_paths:
        image_id = int(rgb_path.stem.split("_")[-1])
        image_key = f"{1:06d}_{image_id:06d}"
        mask_path = raw_root / f"instance_segmentation_{image_id:04d}.png"
        mapping_path = raw_root / f"instance_segmentation_mapping_{image_id:04d}.json"
        if not mask_path.exists() or not mapping_path.exists():
            continue

        mapping = json.loads(mapping_path.read_text())
        rgba = np.array(Image.open(mask_path).convert("RGBA"))
        object_counts = defaultdict(int)
        image_detections = []

        for color_key, object_name in mapping.items():
            if object_name not in name_to_obj_id:
                missing_names[object_name] += 1
                continue
            color = parse_color_key(color_key)
            binary_mask = np.all(rgba == np.array(color, dtype=rgba.dtype), axis=-1)
            if not np.any(binary_mask):
                continue
            obj_id = name_to_obj_id[object_name]
            image_detections.append(
                {
                    "scene_id": 1,
                    "image_id": image_id,
                    "category_id": obj_id,
                    "bbox": binary_mask_to_bbox(binary_mask),
                    "segmentation": binary_mask_to_rle(binary_mask.astype(np.uint8)),
                    "score": 1.0,
                    "time": 0.0,
                }
            )
            object_counts[obj_id] += 1

        detections.extend(image_detections)
        for obj_id, inst_count in sorted(object_counts.items()):
            targets.append(
                {
                    "scene_id": 1,
                    "im_id": image_id,
                    "obj_id": obj_id,
                    "inst_count": inst_count,
                }
            )

    return detections, targets, missing_names


def convert_imagewise_to_webdataset(repo_root, imagewise_dir, test_dir):
    cmd = [
        sys.executable,
        "-m",
        "src.scripts.convert_imagewise_to_webdataset",
        "--input",
        str(imagewise_dir),
        "--output",
        str(test_dir),
        "--maxcount",
        "1000",
    ]
    subprocess.run(cmd, check=True, cwd=repo_root)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare a flat RGB+instance-mask dataset for GigaPose inference."
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("/sata1/data/kevin/realworld_datasets/3d_printing_dataset"),
        help="Flat dataset directory with rgb_XXXX.png and instance_segmentation_XXXX.png.",
    )
    parser.add_argument(
        "--mesh-dir",
        type=Path,
        default=Path("/sata1/data/kevin/realworld_datasets/3d_printing_meshes/stls"),
        help="Fallback mesh directory used to build a name-to-obj_id mapping when obj_*.obj is unavailable.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("gigaPose_datasets/datasets"),
        help="Root directory that contains templates/, dataset folders, and default_detections/.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="myprints",
        help="Dataset name to create inside dataset-root.",
    )
    parser.add_argument(
        "--cam-k",
        type=float,
        nargs=9,
        default=DEFAULT_CAM_K,
        metavar=("fx", "s0", "cx", "s1", "fy", "cy", "s2", "s3", "s4"),
        help="Flattened 3x3 intrinsic matrix written into camera.json.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repository root used to call the webdataset converter.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    raw_root = args.raw_root.expanduser().resolve()
    mesh_dir = args.mesh_dir.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    repo_root = args.repo_root.expanduser().resolve()

    dataset_dir = dataset_root / args.dataset_name
    model_dir = dataset_dir / "models"
    imagewise_dir = dataset_dir / "test_imagewise"
    test_dir = dataset_dir / "test"
    detections_dir = (
        dataset_root / "default_detections" / "core19_model_based_unseen" / "cnos-fastsam"
    )
    detections_path = detections_dir / f"cnos-fastsam_{args.dataset_name}-test_gt_masks.json"

    name_to_obj_id = build_name_to_obj_id_from_model_dir(model_dir)
    if name_to_obj_id is None:
        mesh_names = find_mesh_names(mesh_dir)
        name_to_obj_id = build_name_to_obj_id(mesh_names)
    write_mapping_files(dataset_dir, name_to_obj_id)
    write_models_info(dataset_dir / "models" / "models_info.json", name_to_obj_id)

    detections, targets, missing_names = collect_dataset_annotations(raw_root, name_to_obj_id)
    if missing_names:
        missing_names_str = ", ".join(
            f"{name}({count})" for name, count in sorted(missing_names.items())
        )
        print(
            "Warning: skipping segmentation entries whose object names are missing from the "
            f"mesh-derived name_to_obj_id mapping: {missing_names_str}"
        )

    for rgb_path in sorted(raw_root.glob("rgb_*.png")):
        image_id = int(rgb_path.stem.split("_")[-1])
        image_key = f"{1:06d}_{image_id:06d}"
        write_imagewise_sample(imagewise_dir, image_key, rgb_path.resolve(), args.cam_k)

    detections_dir.mkdir(parents=True, exist_ok=True)
    detections_path.write_text(json.dumps(detections))
    (dataset_dir / "test_targets_bop19.json").write_text(json.dumps(targets, indent=2))

    if test_dir.exists():
        for tar_path in test_dir.glob("*.tar"):
            tar_path.unlink()
        key_to_shard = test_dir / "key_to_shard.json"
        if key_to_shard.exists():
            key_to_shard.unlink()
    convert_imagewise_to_webdataset(repo_root, imagewise_dir, test_dir)

    template_dir = dataset_root / "templates" / args.dataset_name
    available_template_ids = {
        int(path.name)
        for path in template_dir.iterdir()
        if path.is_dir() and path.name.isdigit()
    } if template_dir.exists() else set()
    missing_template_names = [
        name for name, obj_id in sorted(name_to_obj_id.items(), key=lambda x: x[1])
        if obj_id not in available_template_ids
    ]

    print(f"Prepared dataset at {dataset_dir}")
    print(f"Saved detections to {detections_path}")
    print(f"Saved {len(targets)} target entries and {len(detections)} detections")
    if missing_template_names:
        print("Missing templates for:")
        for name in missing_template_names:
            print(f"  - {name}")


if __name__ == "__main__":
    main()
