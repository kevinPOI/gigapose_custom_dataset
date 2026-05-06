import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from bop_toolkit_lib import inout


def binary_mask_to_rle(binary_mask):
    rle = {"counts": [], "size": list(binary_mask.shape)}
    counts = rle["counts"]
    mask = binary_mask.ravel(order="F")
    if len(mask) > 0 and mask[0] == 1:
        counts.append(0)

    if len(mask) > 0:
        mask_changes = mask[:-1] != mask[1:]
        changes_indx = np.where(np.concatenate(([True], mask_changes, [True]), 0))[0]
        counts.extend(np.diff(changes_indx).tolist())
    return rle


def binary_mask_to_bbox(binary_mask):
    ys, xs = np.where(binary_mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("Empty mask encountered while generating detections.")
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return [x0, y0, x1 - x0 + 1, y1 - y0 + 1]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate ITODD detection JSON from GT visible masks."
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("/sata1/data/kevin/bop_datasets/ITODD"),
        help="Raw ITODD root containing test/ and itodd/test_targets_bop19.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/sata1/data/kevin/bop_datasets/datasets/default_detections/"
            "core19_model_based_unseen/cnos-fastsam/"
            "cnos-fastsam_itodd-test_gt_masks.json"
        ),
        help="Output JSON path in the format expected by GigaPose.",
    )
    parser.add_argument(
        "--score",
        type=float,
        default=1.0,
        help="Confidence score assigned to every GT-derived detection.",
    )
    parser.add_argument(
        "--time",
        type=float,
        default=0.0,
        help="Detection runtime assigned to every GT-derived detection.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    raw_root = args.raw_root.expanduser().resolve()
    target_path = raw_root / "itodd" / "test_targets_bop19.json"
    test_root = raw_root / "test"

    if not target_path.exists():
        raise FileNotFoundError(f"Missing target file: {target_path}")
    if not test_root.exists():
        raise FileNotFoundError(f"Missing ITODD test directory: {test_root}")

    targets = json.loads(target_path.read_text())
    targets_by_image = {}
    for target in targets:
        key = (int(target["scene_id"]), int(target["im_id"]))
        targets_by_image[key] = target

    detections = []
    mismatched_images = []
    for scene_dir in sorted(test_root.iterdir()):
        if not scene_dir.is_dir():
            continue
        scene_id = int(scene_dir.name)
        mask_dir = scene_dir / "mask_visib"
        if not mask_dir.exists():
            continue

        masks_per_image = defaultdict(list)
        for mask_path in sorted(mask_dir.glob("*.png")):
            im_id = int(mask_path.stem.split("_")[0])
            masks_per_image[im_id].append(mask_path)

        for im_id, mask_paths in sorted(masks_per_image.items()):
            key = (scene_id, im_id)
            if key not in targets_by_image:
                continue
            target = targets_by_image[key]
            obj_id = int(target["obj_id"])
            expected_inst_count = int(target["inst_count"])
            if expected_inst_count != len(mask_paths):
                mismatched_images.append((scene_id, im_id, expected_inst_count, len(mask_paths)))

            for mask_path in mask_paths:
                mask = inout.load_im(mask_path)
                binary_mask = np.asarray(mask) > 0
                try:
                    detections.append(
                        {
                            "scene_id": scene_id,
                            "image_id": im_id,
                            "category_id": obj_id,
                            "bbox": binary_mask_to_bbox(binary_mask),
                            "segmentation": binary_mask_to_rle(binary_mask.astype(np.uint8)),
                            "score": args.score,
                            "time": args.time,
                        }
                    )
                except Exception as e:
                    print(f"Error processing mask {mask_path}: {e}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(detections))
    print(f"Saved {len(detections)} detections to {args.output}")
    if mismatched_images:
        print("Found mask/inst_count mismatches for these images:")
        for scene_id, im_id, expected, found in mismatched_images[:20]:
            print(
                f"  scene_id={scene_id} im_id={im_id} "
                f"expected={expected} found_masks={found}"
            )
        if len(mismatched_images) > 20:
            print(f"  ... and {len(mismatched_images) - 20} more")


if __name__ == "__main__":
    main()
