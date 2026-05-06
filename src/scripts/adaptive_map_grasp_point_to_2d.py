import csv
import ast
import json
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.scripts.map_grasp_point_to_2d import (
    DEFAULT_CAM_K,
    PREDICTION_TRANSLATION_SCALE_TO_M,
    build_exact_mesh_cache,
    build_vis_mesh_cache,
    group_by_image_level,
    import_open3d,
    load_bop_results,
    load_camera_matrix,
    load_image,
    load_object_mappings,
    print_available_objects,
    print_mesh_bbox_info,
    render_open3d_scene,
    resize_for_concat,
    resolve_object_id,
)


# Edit this table for your object.
# Points are in the centered mesh/object coordinate frame used by GigaPose.
# The selected key is the object side whose normal points most toward the camera.
SIDE_NORMALS_OBJECT = {
    "pos_x": np.array([1.0, 0.0, 0.0], dtype=np.float64),
    "neg_x": np.array([-1.0, 0.0, 0.0], dtype=np.float64),
    "pos_y": np.array([0.0, 1.0, 0.0], dtype=np.float64),
    "neg_y": np.array([0.0, -1.0, 0.0], dtype=np.float64),
    "pos_z": np.array([0.0, 0.0, 1.0], dtype=np.float64),
    "neg_z": np.array([0.0, 0.0, -1.0], dtype=np.float64),
}

SIDE_GRASP_POINTS_OBJECT = {
    "pos_x": [
        [0.0020, -0.0012, 0.0],
        [0.0020, 0.0012, 0.0],
    ],
    "neg_x": [
        [-0.0020, -0.0012, 0.0],
        [-0.0020, 0.0012, 0.0],
    ],
    "pos_y": [
        [-0.0012, 0.0020, 0.0],
        [0.0012, 0.0020, 0.0],
    ],
    "neg_y": [
        [-0.0012, -0.0020, 0.0],
        [0.0012, -0.0020, 0.0],
    ],
    "pos_z": [
        [-0.0012, 0.0, 0.0010],
        [0.0012, 0.0, 0.0010],
    ],
    "neg_z": [
        [-0.0012, 0.0, -0.0010],
        [0.0012, 0.0, -0.0010],
    ],
}

SIDE_COLORS = {
    "pos_x": (230, 50, 50),
    "neg_x": (180, 30, 30),
    "pos_y": (50, 160, 230),
    "neg_y": (30, 110, 180),
    "pos_z": (50, 190, 90),
    "neg_z": (35, 130, 65),
}

WORLD_UP_AXIS = np.array([0.0, 0.0, 1.0], dtype=np.float64)


def parse_args():
    parser = ArgumentParser(
        description=(
            "Project grasp points selected adaptively by object orientation. "
            "The script prefers feasible grasps from a JSON library when the predicted "
            "orientation matches a known tabletop pose, and otherwise falls back to "
            "manual side-based grasp presets."
        )
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("/sata1/data/kevin/realworld_datasets/3d_printing_dataset"),
        help="Flat dataset directory containing rgb_XXXX.png.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("/home/zhenrant/gigapose/gigaPose_datasets/datasets/myprints"),
        help="Prepared dataset directory containing models/ and test_imagewise/.",
    )
    parser.add_argument(
        "--result-csv",
        type=Path,
        default=Path(
            "/home/zhenrant/gigapose/gigaPose_datasets/results/large_myprints/predictions/"
            "large-pbrreal-rgb-mmodel_myprints-test_myprints.csv"
        ),
        help="Prediction CSV in BOP format.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/home/zhenrant/gigapose/gigaPose_datasets/results/large_myprints/adaptive_grasp_point_mapping"
        ),
        help="Where to save visualization images and projected_points.csv.",
    )
    parser.add_argument(
        "--object-name",
        type=str,
        default=None,
        help="Object name, object id, or obj_000001-style identifier.",
    )
    parser.add_argument("--scene-id", type=int, default=1, help="Optional scene id filter.")
    parser.add_argument("--im-id", type=int, default=None, help="Optional image id filter.")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--axis-size", type=float, default=0.03)
    parser.add_argument("--scene-width", type=int, default=960)
    parser.add_argument("--scene-height", type=int, default=960)
    parser.add_argument("--point-radius", type=int, default=3)
    parser.add_argument(
        "--feasible-grasps-json",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "feasible_grasps.json",
        help="Feasible grasp library JSON. Used when the predicted orientation matches a pose.",
    )
    parser.add_argument(
        "--feasible-pose-match-threshold",
        type=float,
        default=0.95,
        help="Minimum cosine similarity to treat a predicted orientation as matching a feasible pose.",
    )
    parser.add_argument(
        "--center-on-mask",
        action="store_true",
        default=True,
        help="Shift each grasp-point pair so its 2D center matches the object mask centroid.",
    )
    parser.add_argument(
        "--mask-annotation-dir",
        type=Path,
        default=Path("/sata1/data/kevin/v2_imgs/_out_v2_merged"),
        help=(
            "Directory with semantic_segmentation_*.png and "
            "semantic_segmentation_labels_*.json files. These masks are preferred "
            "over fallback detection masks."
        ),
    )
    parser.add_argument(
        "--no-center-on-mask",
        action="store_false",
        dest="center_on_mask",
        help="Use raw projection without centroid correction.",
    )
    parser.add_argument(
        "--camera-up-axis",
        type=float,
        nargs=3,
        default=[0.0, 0.0, -1.0],
        metavar=("X", "Y", "Z"),
        help=(
            "Direction in camera coordinates corresponding to a block face pointing up/toward "
            "the camera. For OpenCV-style cameras looking down at a table, use 0 0 -1."
        ),
    )
    parser.add_argument(
        "--list-objects",
        default=False,
        action="store_true",
        help="Print available object names and exit.",
    )
    return parser.parse_args()


def normalize(vector):
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError(f"Cannot normalize near-zero vector: {vector}")
    return vector / norm


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def select_up_side(pred, camera_up_axis):
    rotation = np.asarray(pred["R"], dtype=np.float64).reshape(3, 3)
    camera_up_axis = normalize(camera_up_axis)
    scores = {}
    for side_name, normal_object in SIDE_NORMALS_OBJECT.items():
        normal_camera = normalize(rotation @ normal_object)
        scores[side_name] = float(np.dot(normal_camera, camera_up_axis))
    side_name = max(scores, key=scores.get)
    return side_name, scores[side_name], scores


def quaternion_xyzw_to_rotation_matrix(quaternion_xyzw):
    x, y, z, w = normalize(quaternion_xyzw)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def infer_pose_normal_object(planned_orientations):
    best_side_name = None
    best_score = -np.inf
    for side_name, normal_object in SIDE_NORMALS_OBJECT.items():
        scores = []
        for quaternion_xyzw in planned_orientations:
            rotation_world = quaternion_xyzw_to_rotation_matrix(quaternion_xyzw)
            normal_world = normalize(rotation_world @ normal_object)
            scores.append(float(np.dot(normal_world, WORLD_UP_AXIS)))
        side_score = min(scores)
        if side_score > best_score:
            best_side_name = side_name
            best_score = side_score
    return best_side_name, best_score


def load_feasible_grasp_library(path):
    path = Path(path).expanduser().resolve()
    if not path.exists():
        return None

    payload = load_json(path)
    feasible_cases = payload.get("feasible_cases", [])
    poses = {}
    for case in feasible_cases:
        pose_id = int(case["pose_id"])
        pose_entry = poses.setdefault(
            pose_id,
            {
                "pose_id": pose_id,
                "planned_orientations": [],
                "grasps": {},
            },
        )

        planned_orientation = tuple(
            float(v) for v in case["object"]["planned_pose"]["orientation_xyzw"]
        )
        if planned_orientation not in pose_entry["planned_orientations"]:
            pose_entry["planned_orientations"].append(planned_orientation)

        grasp = case["grasp"]
        grasp_id = int(grasp["grasp_id"])
        pose_entry["grasps"].setdefault(
            grasp_id,
            {
                "grasp_id": grasp_id,
                "point_object": np.asarray(grasp["grasp_point"], dtype=np.float64),
                "approach_direction": np.asarray(
                    grasp["approach_direction"], dtype=np.float64
                ),
                "object_orientation_xyzw": np.asarray(
                    grasp["planned_object_relative_pose"]["object_orientation_xyzw"],
                    dtype=np.float64,
                ),
            },
        )

    if not poses:
        return None

    for pose_entry in poses.values():
        normal_side_name, normal_score = infer_pose_normal_object(
            pose_entry["planned_orientations"]
        )
        pose_entry["normal_side_name"] = normal_side_name
        pose_entry["normal_object"] = SIDE_NORMALS_OBJECT[normal_side_name]
        pose_entry["normal_score"] = normal_score
        pose_entry["color"] = SIDE_COLORS[normal_side_name]
        pose_entry["grasps"] = [
            pose_entry["grasps"][grasp_id]
            for grasp_id in sorted(pose_entry["grasps"])
        ]

    return {
        "path": path,
        "poses": {pose_id: poses[pose_id] for pose_id in sorted(poses)},
    }


def select_feasible_pose(pred, camera_up_axis, feasible_grasp_library, match_threshold):
    if not feasible_grasp_library:
        return None, None, None

    rotation = np.asarray(pred["R"], dtype=np.float64).reshape(3, 3)
    camera_up_axis = normalize(camera_up_axis)
    scores = {}
    for pose_id, pose_entry in feasible_grasp_library["poses"].items():
        normal_camera = normalize(rotation @ pose_entry["normal_object"])
        scores[pose_id] = float(np.dot(normal_camera, camera_up_axis))

    if not scores:
        return None, None, scores

    pose_id = max(scores, key=scores.get)
    pose_score = scores[pose_id]
    if pose_score < match_threshold:
        return None, pose_score, scores
    return feasible_grasp_library["poses"][pose_id], pose_score, scores


def get_translation_in_meters(pred):
    return np.asarray(pred["t"], dtype=np.float64).reshape(3) * PREDICTION_TRANSLATION_SCALE_TO_M


def rle_to_binary_mask(rle):
    height, width = [int(v) for v in rle["size"]]
    values = []
    value = 0
    for count in rle["counts"]:
        values.extend([value] * int(count))
        value = 1 - value
    flat = np.asarray(values, dtype=np.uint8)
    if flat.size != height * width:
        raise ValueError(
            f"Invalid RLE size: got {flat.size} values, expected {height * width}"
        )
    return flat.reshape((height, width), order="F")


def binary_mask_centroid(mask):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return np.array([float(xs.mean()), float(ys.mean())], dtype=np.float64)


def overlay_mask(image, mask, color=(255, 255, 0), alpha=0.35):
    if mask is None:
        return image
    overlay = image.astype(np.float32).copy()
    mask_bool = mask.astype(bool)
    color_array = np.asarray(color, dtype=np.float32)
    overlay[mask_bool] = alpha * color_array + (1.0 - alpha) * overlay[mask_bool]
    return np.clip(overlay, 0, 255).astype(np.uint8)


def find_detection_file(dataset_dir, dataset_name):
    detection_dir = (
        dataset_dir.parent
        / "default_detections"
        / "core19_model_based_unseen"
        / "cnos-fastsam"
    )
    candidates = sorted(detection_dir.glob(f"cnos-fastsam_{dataset_name}-test*.json"))
    if not candidates:
        return None
    return candidates[0]


def load_masks_and_centroids(dataset_dir, obj_id):
    detection_file = find_detection_file(dataset_dir, dataset_dir.name)
    if detection_file is None:
        return {}
    with detection_file.open("r", encoding="utf-8") as f:
        detections = json.load(f)

    masks = {}
    for det in detections:
        if int(det.get("category_id", -1)) != int(obj_id):
            continue
        key = f"{int(det['scene_id']):06d}_{int(det['image_id']):06d}"
        mask = rle_to_binary_mask(det["segmentation"])
        centroid = binary_mask_centroid(mask)
        if centroid is not None:
            masks[key] = {"mask": mask, "centroid": centroid}
    return masks


def parse_color_key(color_key):
    return tuple(int(v) for v in ast.literal_eval(color_key))


def load_semantic_mask(mask_annotation_dir, im_id, object_class):
    seg_path = mask_annotation_dir / f"semantic_segmentation_{im_id}.png"
    labels_path = mask_annotation_dir / f"semantic_segmentation_labels_{im_id}.json"
    if not seg_path.exists() or not labels_path.exists():
        return None

    labels = json.load(labels_path.open("r", encoding="utf-8"))
    target_colors = [
        parse_color_key(color_key)
        for color_key, info in labels.items()
        if str(info.get("class")) == object_class
    ]
    if not target_colors:
        return None

    segmentation = np.asarray(Image.open(seg_path).convert("RGBA"))
    mask = np.zeros(segmentation.shape[:2], dtype=np.uint8)
    for color in target_colors:
        mask |= np.all(segmentation == np.asarray(color, dtype=np.uint8), axis=-1)
    if not np.any(mask):
        return None
    return mask.astype(np.uint8)


def load_semantic_masks_and_centroids(mask_annotation_dir, image_keys, object_class):
    masks = {}
    mask_annotation_dir = mask_annotation_dir.expanduser().resolve()
    for image_key in image_keys:
        _, im_id = [int(x) for x in image_key.split("_")]
        mask = load_semantic_mask(mask_annotation_dir, im_id, object_class)
        if mask is None:
            continue
        centroid = binary_mask_centroid(mask)
        if centroid is not None:
            masks[image_key] = {"mask": mask, "centroid": centroid}
    return masks


def project_point(point_object, pred, cam_k):
    point_cam = pred["R"] @ point_object + get_translation_in_meters(pred)
    if point_cam[2] <= 1e-9:
        return None, point_cam
    uvw = cam_k @ point_cam
    return uvw[:2] / uvw[2], point_cam


def draw_adaptive_grasp_points(
    image,
    selected_predictions,
    cam_k,
    point_radius,
    camera_up_axis,
    feasible_grasp_library,
    feasible_pose_match_threshold,
    mask_data=None,
):
    mask = None if mask_data is None else mask_data["mask"]
    mask_centroid = None if mask_data is None else mask_data["centroid"]
    # image = overlay_mask(image, mask)
    image_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(image_pil)
    width, height = image_pil.size
    projected_rows = []

    # if mask_centroid is not None:
    #     cx, cy = float(mask_centroid[0]), float(mask_centroid[1])
    #     draw.line((cx - 10, cy, cx + 10, cy), fill=(255, 255, 255), width=3)
    #     draw.line((cx, cy - 10, cx, cy + 10), fill=(255, 255, 255), width=3)
    #     draw.ellipse((cx - 4, cy - 4, cx + 4, cy + 4), fill=(255, 0, 255))

    for det_idx, pred in enumerate(selected_predictions, start=1):
        matched_pose_entry, pose_score, _ = select_feasible_pose(
            pred,
            camera_up_axis,
            feasible_grasp_library,
            feasible_pose_match_threshold,
        )
        if matched_pose_entry is not None:
            grasp_source = "feasible_grasp_json"
            side_name = matched_pose_entry["normal_side_name"]
            side_score = pose_score
            pose_id = matched_pose_entry["pose_id"]
            grasp_entries = matched_pose_entry["grasps"]
            color = matched_pose_entry["color"]
            mode_label = f"{grasp_source}:pose_{pose_id}"
        else:
            grasp_source = "manual_presets"
            side_name, side_score, _ = select_up_side(pred, camera_up_axis)
            pose_id = ""
            grasp_entries = [
                {
                    "grasp_id": point_idx - 1,
                    "point_object": np.asarray(point, dtype=np.float64),
                    "approach_direction": np.full(3, np.nan, dtype=np.float64),
                    "object_orientation_xyzw": np.full(4, np.nan, dtype=np.float64),
                }
                for point_idx, point in enumerate(
                    SIDE_GRASP_POINTS_OBJECT[side_name], start=1
                )
            ]
            color = SIDE_COLORS[side_name]
            mode_label = grasp_source

        point_payloads = []

        for point_idx, grasp_entry in enumerate(grasp_entries, start=1):
            point = grasp_entry["point_object"]
            uv, point_cam = project_point(point, pred, cam_k)
            point_payloads.append((point_idx, grasp_entry, uv, point_cam))

        uv_values = [uv for _, _, uv, _ in point_payloads if uv is not None]
        uv_shift = np.zeros(2, dtype=np.float64)
        if mask_centroid is not None and uv_values:
            uv_shift = np.asarray(mask_centroid, dtype=np.float64) - np.mean(
                np.stack(uv_values, axis=0), axis=0
            )

        visible_anchor_points = []
        for point_idx, grasp_entry, uv, point_cam in point_payloads:
            point = grasp_entry["point_object"]
            approach_direction = grasp_entry["approach_direction"]
            object_orientation_xyzw = grasp_entry["object_orientation_xyzw"]
            row = {
                "scene_id": int(pred["scene_id"]),
                "im_id": int(pred["im_id"]),
                "obj_id": int(pred["obj_id"]),
                "det_idx": det_idx,
                "grasp_source": grasp_source,
                "matched_pose_id": pose_id,
                "side": side_name,
                "side_score": side_score,
                "grasp_id": int(grasp_entry["grasp_id"]),
                "point_idx": point_idx,
                "point_object_x": float(point[0]),
                "point_object_y": float(point[1]),
                "point_object_z": float(point[2]),
                "approach_direction_x": (
                    "" if np.isnan(approach_direction[0]) else float(approach_direction[0])
                ),
                "approach_direction_y": (
                    "" if np.isnan(approach_direction[1]) else float(approach_direction[1])
                ),
                "approach_direction_z": (
                    "" if np.isnan(approach_direction[2]) else float(approach_direction[2])
                ),
                "grasp_orientation_x": (
                    "" if np.isnan(object_orientation_xyzw[0]) else float(object_orientation_xyzw[0])
                ),
                "grasp_orientation_y": (
                    "" if np.isnan(object_orientation_xyzw[1]) else float(object_orientation_xyzw[1])
                ),
                "grasp_orientation_z": (
                    "" if np.isnan(object_orientation_xyzw[2]) else float(object_orientation_xyzw[2])
                ),
                "grasp_orientation_w": (
                    "" if np.isnan(object_orientation_xyzw[3]) else float(object_orientation_xyzw[3])
                ),
                "point_camera_x": float(point_cam[0]),
                "point_camera_y": float(point_cam[1]),
                "point_camera_z": float(point_cam[2]),
                "raw_u": "",
                "raw_v": "",
                "mask_center_u": "" if mask_centroid is None else float(mask_centroid[0]),
                "mask_center_v": "" if mask_centroid is None else float(mask_centroid[1]),
                "center_shift_u": float(uv_shift[0]),
                "center_shift_v": float(uv_shift[1]),
                "u": "",
                "v": "",
                "in_image": False,
            }

            if uv is not None:
                row["raw_u"] = float(uv[0])
                row["raw_v"] = float(uv[1])
                shifted_uv = uv + uv_shift
                x, y = float(shifted_uv[0]), float(shifted_uv[1])
                in_image = 0.0 <= x < width and 0.0 <= y < height
                row["u"] = x
                row["v"] = y
                row["in_image"] = in_image

                if in_image:
                    visible_anchor_points.append((x, y))
                    draw.ellipse(
                        (
                            x - point_radius,
                            y - point_radius,
                            x + point_radius,
                            y + point_radius,
                        ),
                        fill=color,
                        outline=(255, 255, 255),
                        width=2,
                    )
                    draw.text(
                        (x + point_radius + 2, y - point_radius - 2),
                        f"{side_name}:{point_idx}",
                        fill=color,
                    )

            projected_rows.append(row)

        if visible_anchor_points:
            anchor_x = min(point[0] for point in visible_anchor_points)
            anchor_y = min(point[1] for point in visible_anchor_points) - (point_radius + 12)
            draw.text((anchor_x, anchor_y), mode_label, fill=color)

    return np.array(image_pil), projected_rows


def write_projection_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()

    raw_dir = args.raw_dir.expanduser().resolve()
    dataset_dir = args.dataset_dir.expanduser().resolve()
    result_csv = args.result_csv.expanduser().resolve()
    output_root = args.output_dir.expanduser().resolve()
    camera_up_axis = normalize(args.camera_up_axis)
    feasible_grasp_library = load_feasible_grasp_library(args.feasible_grasps_json)

    name_to_obj_id, obj_id_to_name = load_object_mappings(dataset_dir)
    if args.list_objects:
        print_available_objects(name_to_obj_id)
        return

    selected_obj_id, selected_object_name = resolve_object_id(
        args.object_name, name_to_obj_id, obj_id_to_name
    )

    o3d = import_open3d()
    exact_mesh_cache = build_exact_mesh_cache(o3d, dataset_dir)
    vis_mesh_cache = build_vis_mesh_cache(o3d, exact_mesh_cache)
    if selected_obj_id not in exact_mesh_cache:
        raise SystemExit(
            f"Could not load a mesh for '{selected_object_name}' (obj_id={selected_obj_id})."
        )

    all_predictions = load_bop_results(result_csv)
    predictions = [
        pred
        for pred in all_predictions
        if pred["obj_id"] == selected_obj_id
        and (args.scene_id is None or pred["scene_id"] == args.scene_id)
        and (args.im_id is None or pred["im_id"] == args.im_id)
    ]
    predictions_per_image = group_by_image_level(predictions)
    selected_keys = sorted(predictions_per_image)
    if args.max_images is not None:
        selected_keys = selected_keys[: args.max_images]

    if not selected_keys:
        raise SystemExit(
            f"No predictions found for '{selected_object_name}' (obj_id={selected_obj_id})."
        )

    all_predictions_per_image = group_by_image_level(all_predictions)
    output_dir = output_root / selected_object_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print_mesh_bbox_info(exact_mesh_cache[selected_obj_id], selected_object_name, selected_obj_id)
    if feasible_grasp_library is not None:
        print(
            f"Loaded feasible grasp library with {len(feasible_grasp_library['poses'])} pose ids "
            f"from {feasible_grasp_library['path']}"
        )
    else:
        print(
            f"No feasible grasp library found at {args.feasible_grasps_json.expanduser().resolve()}; "
            "using manual presets only."
        )
    all_projected_rows = []
    mask_data_by_image = {}
    if args.center_on_mask:
        object_class = f"{selected_obj_id:08d}"
        mask_data_by_image = load_semantic_masks_and_centroids(
            args.mask_annotation_dir,
            selected_keys,
            object_class,
        )
        fallback_mask_data = load_masks_and_centroids(dataset_dir, selected_obj_id)
        for image_key, mask_data in fallback_mask_data.items():
            mask_data_by_image.setdefault(image_key, mask_data)
        print(
            f"Loaded {len(mask_data_by_image)} masks for obj_id={selected_obj_id} "
            f"(semantic class {object_class}); semantic masks are preferred."
        )

    for image_key in tqdm(selected_keys, desc=f"Adaptive grasp mapping for {selected_object_name}"):
        scene_id, im_id = [int(x) for x in image_key.split("_")]
        image = load_image(raw_dir, dataset_dir, scene_id, im_id)
        cam_k = load_camera_matrix(dataset_dir, scene_id, im_id)

        overlay, projected_rows = draw_adaptive_grasp_points(
            image=image,
            selected_predictions=predictions_per_image[image_key],
            cam_k=cam_k,
            point_radius=args.point_radius,
            camera_up_axis=camera_up_axis,
            feasible_grasp_library=feasible_grasp_library,
            feasible_pose_match_threshold=args.feasible_pose_match_threshold,
            mask_data=mask_data_by_image.get(image_key),
        )
        all_projected_rows.extend(projected_rows)

        scene_view = render_open3d_scene(
            o3d,
            vis_mesh_cache,
            all_predictions_per_image.get(image_key, predictions_per_image[image_key]),
            width=args.scene_width,
            height=args.scene_height,
            axis_size=args.axis_size,
        )
        overlay = resize_for_concat(overlay, scene_view)
        vis = np.concatenate([overlay, scene_view], axis=1)
        Image.fromarray(vis).save(output_dir / f"{image_key}.png")

    write_projection_csv(output_dir / "projected_points.csv", all_projected_rows)
    print(
        f"Saved {len(selected_keys)} adaptive grasp visualizations for "
        f"'{selected_object_name}' to {output_dir}"
    )


if __name__ == "__main__":
    main()
