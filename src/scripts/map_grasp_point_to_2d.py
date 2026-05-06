import csv
import json
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm


DEFAULT_CAM_K = np.array(
    [572.4114, 0.0, 320.0, 0.0, 573.57043, 240.0, 0.0, 0.0, 1.0],
    dtype=np.float64,
).reshape(3, 3)
DEFAULT_OBJECT_NAME = "flipper_extension"
DEFAULT_GRASP_POINTS = [
    [0.01, 0.0, -0.005],
    [0.01, 0.0, 0.005],
]
PREDICTION_TRANSLATION_SCALE_TO_M = 0.001
MIN_LONGEST_SIDE_M = 0.01
MAX_LONGEST_SIDE_M = 0.10
VISIBLE_COLOR = (255, 0, 0)
BACKSIDE_COLOR = (0, 0, 255)
POINT_LABELS = ("A", "B")


def parse_args():
    parser = ArgumentParser(
        description=(
            "Project two 3D grasp points for a selected object onto every image that "
            "contains that object in a BOP-style prediction CSV."
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
            "/home/zhenrant/gigapose/gigaPose_datasets/results/large_myprints/grasp_point_mapping"
        ),
        help="Where to save visualization images.",
    )
    parser.add_argument(
        "--object-name",
        type=str,
        default=DEFAULT_OBJECT_NAME,
        help="Object name, object id, or obj_000001-style identifier.",
    )
    parser.add_argument(
        "--grasp-point",
        type=float,
        nargs=3,
        action="append",
        metavar=("X", "Y", "Z"),
        default=None,
        help=(
            "One grasp point in mesh coordinates. Pass this flag twice, for example: "
            "--grasp-point 1 2 3 --grasp-point 4 5 6"
        ),
    )
    parser.add_argument(
        "--scene-id",
        type=int,
        default=1,
        help="Optional scene id filter.",
    )
    parser.add_argument(
        "--im-id",
        type=int,
        default=None,
        help="Optional image id filter.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional maximum number of output images.",
    )
    parser.add_argument(
        "--axis-size",
        type=float,
        default=0.03,
        help="Axis size in meters for the Open3D scene view.",
    )
    parser.add_argument(
        "--scene-width",
        type=int,
        default=960,
        help="Open3D panel width.",
    )
    parser.add_argument(
        "--scene-height",
        type=int,
        default=960,
        help="Open3D panel height.",
    )
    parser.add_argument(
        "--point-radius",
        type=int,
        default=12,
        help="Radius in pixels for projected grasp points.",
    )
    parser.add_argument(
        "--list-objects",
        default = False,
        action="store_true",
        help="Print available object names and exit.",
    )
    return parser.parse_args()


def import_open3d():
    try:
        import open3d as o3d
    except ImportError as exc:
        raise SystemExit(
            "Install Open3D in the current env with `pip install open3d` and rerun."
        ) from exc
    return o3d


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_bop_results(result_csv: Path):
    predictions = []
    with result_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            predictions.append(
                {
                    "scene_id": int(row["scene_id"]),
                    "im_id": int(row["im_id"]),
                    "obj_id": int(row["obj_id"]),
                    "score": float(row["score"]),
                    "R": np.fromstring(row["R"], sep=" ", dtype=np.float64).reshape(3, 3),
                    "t": np.fromstring(row["t"], sep=" ", dtype=np.float64).reshape(3),
                    "time": float(row["time"]),
                }
            )
    return predictions


def group_by_image_level(data, image_key="im_id"):
    data_per_image = {}
    for det in data:
        scene_id, im_id = int(det["scene_id"]), int(det[image_key])
        key = f"{scene_id:06d}_{im_id:06d}"
        data_per_image.setdefault(key, []).append(det)
    return data_per_image


def load_object_mappings(dataset_dir: Path):
    obj_id_to_name_path = dataset_dir / "obj_id_to_name.json"
    if obj_id_to_name_path.exists():
        obj_id_to_name = {int(k): v for k, v in load_json(obj_id_to_name_path).items()}
    else:
        models_info = load_json(dataset_dir / "models" / "models_info.json")
        obj_id_to_name = {}
        for obj_id, info in models_info.items():
            name = info.get("name", f"obj_{int(obj_id):06d}")
            obj_id_to_name[int(obj_id)] = name
    name_to_obj_id = {name: obj_id for obj_id, name in obj_id_to_name.items()}
    return name_to_obj_id, obj_id_to_name


def resolve_object_id(object_name, name_to_obj_id, obj_id_to_name):
    if object_name is None:
        available = ", ".join(sorted(name_to_obj_id))
        raise SystemExit(
            "--object-name is required. Use --list-objects to inspect the available names.\n"
            f"Available objects: {available}"
        )

    candidate = object_name.strip()
    if candidate in name_to_obj_id:
        obj_id = name_to_obj_id[candidate]
        return obj_id, obj_id_to_name[obj_id]

    lower = candidate.lower()
    if lower.startswith("obj_"):
        lower = lower[4:]
    if lower.isdigit():
        obj_id = int(lower)
        if obj_id in obj_id_to_name:
            return obj_id, obj_id_to_name[obj_id]

    available = ", ".join(sorted(name_to_obj_id))
    raise SystemExit(
        f"Could not resolve object '{object_name}'. Available objects: {available}"
    )


def validate_grasp_points(points):
    if points is None:
        points = DEFAULT_GRASP_POINTS
    if len(points) != 2:
        raise SystemExit(
            "Pass exactly two grasp points with "
            "`--grasp-point X1 Y1 Z1 --grasp-point X2 Y2 Z2`."
        )
    return [np.asarray(point, dtype=np.float64) for point in points]


def load_image(raw_dir: Path, dataset_dir: Path, scene_id: int, im_id: int):
    candidate_paths = [
        raw_dir / f"rgb_{im_id:04d}.png",
        raw_dir / f"rgb_{im_id:06d}.png",
        dataset_dir / "test_imagewise" / f"{scene_id:06d}_{im_id:06d}.rgb.png",
    ]
    for img_path in candidate_paths:
        if img_path.exists():
            return np.array(Image.open(img_path).convert("RGB"))
    raise FileNotFoundError(
        f"Could not locate an RGB image for scene_id={scene_id}, im_id={im_id}."
    )


def load_camera_matrix(dataset_dir: Path, scene_id: int, im_id: int):
    camera_path = dataset_dir / "test_imagewise" / f"{scene_id:06d}_{im_id:06d}.camera.json"
    if not camera_path.exists():
        return DEFAULT_CAM_K.copy()
    camera_data = load_json(camera_path)
    cam_k = camera_data.get("cam_K")
    if cam_k is None:
        return DEFAULT_CAM_K.copy()
    return np.asarray(cam_k, dtype=np.float64).reshape(3, 3)


def build_exact_mesh_cache(o3d, dataset_dir: Path):
    models_info = load_json(dataset_dir / "models" / "models_info.json")
    mesh_cache = {}
    for obj_id in models_info:
        obj_id_int = int(obj_id)
        mesh_path = dataset_dir / "models" / f"obj_{obj_id_int:06d}.obj"
        if not mesh_path.exists():
            mesh_path = dataset_dir / "models" / f"obj_{obj_id_int:06d}.ply"
        if not mesh_path.exists():
            continue
        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        if mesh.is_empty():
            continue
        bbox = mesh.get_axis_aligned_bounding_box()
        mesh.translate(-np.asarray(bbox.get_center(), dtype=np.float64))
        mesh.compute_vertex_normals()
        mesh_cache[obj_id_int] = mesh
    return mesh_cache


def build_vis_mesh_cache(o3d, exact_mesh_cache):
    mesh_cache = {}
    for obj_id, mesh in exact_mesh_cache.items():
        mesh_vis = o3d.geometry.TriangleMesh(mesh)
        bbox = mesh_vis.get_axis_aligned_bounding_box()
        extent = np.asarray(bbox.get_extent(), dtype=np.float64)
        longest_side = float(np.max(extent))
        if longest_side > 0:
            target_longest_side = min(max(longest_side * 0.001, MIN_LONGEST_SIDE_M), MAX_LONGEST_SIDE_M)
            mesh_vis.scale(target_longest_side / longest_side, center=bbox.get_center())
        mesh_vis.compute_vertex_normals()
        mesh_cache[obj_id] = mesh_vis
    return mesh_cache


def make_transform(pred, translation_scale):
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = np.asarray(pred["R"], dtype=np.float64).reshape(3, 3)
    pose[:3, 3] = np.asarray(pred["t"], dtype=np.float64).reshape(3) * translation_scale
    return pose


def get_translation_in_meters(pred):
    return np.asarray(pred["t"], dtype=np.float64).reshape(3) * PREDICTION_TRANSLATION_SCALE_TO_M


def create_material(o3d, base_color):
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultLit"
    material.base_color = np.asarray(base_color, dtype=np.float32)
    return material


def render_open3d_scene(o3d, mesh_cache, predictions, width, height, axis_size):
    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    scene = renderer.scene
    scene.set_background([1.0, 1.0, 1.0, 1.0])
    scene.scene.enable_sun_light(True)
    scene.scene.set_sun_light([0.0, 0.0, -1.0], [1.0, 1.0, 1.0], 50000)
    scene.scene.enable_indirect_light(True)

    palette = [
        [0.8, 0.2, 0.2, 1.0],
        [0.2, 0.6, 0.9, 1.0],
        [0.2, 0.7, 0.3, 1.0],
        [0.8, 0.6, 0.2, 1.0],
        [0.7, 0.3, 0.8, 1.0],
    ]

    all_points = [np.zeros((1, 3), dtype=np.float64)]
    for idx, pred in enumerate(predictions):
        obj_id = int(pred["obj_id"])
        if obj_id not in mesh_cache:
            continue
        pose = make_transform(pred, translation_scale=PREDICTION_TRANSLATION_SCALE_TO_M)
        mesh = o3d.geometry.TriangleMesh(mesh_cache[obj_id])
        mesh.transform(pose)
        scene.add_geometry(
            f"mesh_{idx}_{obj_id}",
            mesh,
            create_material(o3d, palette[idx % len(palette)]),
        )
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size)
        axis.transform(pose)
        scene.add_geometry(
            f"axis_{idx}_{obj_id}",
            axis,
            create_material(o3d, [1.0, 1.0, 1.0, 1.0]),
        )
        all_points.append(np.asarray(mesh.vertices))

    all_points = np.concatenate(all_points, axis=0)
    bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
        o3d.utility.Vector3dVector(all_points)
    )
    extent = max(float(np.max(bbox.get_extent())), 0.2)

    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width,
        height,
        DEFAULT_CAM_K[0, 0],
        DEFAULT_CAM_K[1, 1],
        DEFAULT_CAM_K[0, 2],
        DEFAULT_CAM_K[1, 2],
    )
    extrinsic = np.eye(4, dtype=np.float64)
    renderer.setup_camera(intrinsic, extrinsic)
    scene.camera.set_projection(
        DEFAULT_CAM_K,
        0.01,
        max(5.0, extent * 10.0),
        width,
        height,
    )

    return np.asarray(renderer.render_to_image())


def project_point(point_object, pred, cam_k):
    point_cam = pred["R"] @ point_object + get_translation_in_meters(pred)
    if point_cam[2] <= 1e-9:
        return None, point_cam
    uvw = cam_k @ point_cam
    return uvw[:2] / uvw[2], point_cam


def classify_point_visibility(point_object, pred, mesh):
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    normals = np.asarray(mesh.vertex_normals, dtype=np.float64)
    if len(vertices) == 0 or len(normals) == 0:
        return False

    nearest_vertex_idx = int(np.argmin(np.sum((vertices - point_object[None, :]) ** 2, axis=1)))
    normal_object = normals[nearest_vertex_idx]
    normal_camera = pred["R"] @ normal_object
    point_camera = pred["R"] @ point_object + get_translation_in_meters(pred)
    if point_camera[2] <= 1e-9:
        return False
    camera_to_point = point_camera / np.linalg.norm(point_camera)
    return float(np.dot(normal_camera, -camera_to_point)) > 0.0


def draw_grasp_points(image, selected_predictions, mesh, grasp_points, cam_k, point_radius):
    image_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(image_pil)
    width, height = image_pil.size

    for det_idx, pred in enumerate(selected_predictions, start=1):
        for point_idx, point in enumerate(grasp_points):
            uv, _ = project_point(point, pred, cam_k)
            is_visible = classify_point_visibility(point, pred, mesh)
            color = VISIBLE_COLOR if is_visible else BACKSIDE_COLOR

            if uv is None:
                continue

            x, y = float(uv[0]), float(uv[1])
            if not (0.0 <= x < width and 0.0 <= y < height):
                print("Projection out of image for point", point_idx, "in detection", det_idx)
                continue

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
            label = f"{POINT_LABELS[point_idx]}{det_idx}"
            draw.text((x + point_radius + 2, y - point_radius - 2), label, fill=color)

    return np.array(image_pil)


def resize_for_concat(original, scene_view):
    if original.shape[0] == scene_view.shape[0]:
        return original
    scale = scene_view.shape[0] / original.shape[0]
    new_width = int(round(original.shape[1] * scale))
    return np.array(Image.fromarray(original).resize((new_width, scene_view.shape[0])))


def print_available_objects(name_to_obj_id):
    for name, obj_id in sorted(name_to_obj_id.items(), key=lambda item: item[1]):
        print(f"{obj_id:02d}: {name}")


def print_mesh_bbox_info(mesh, object_name, obj_id):
    bbox = mesh.get_axis_aligned_bounding_box()
    center = np.asarray(bbox.get_center(), dtype=np.float64)
    extent = np.asarray(bbox.get_extent(), dtype=np.float64)
    print(
        f"Object '{object_name}' (obj_id={obj_id}) bbox center: "
        f"{center.tolist()}"
    )
    print(
        f"Object '{object_name}' (obj_id={obj_id}) bbox size: "
        f"{extent.tolist()}"
    )


def main():
    args = parse_args()

    raw_dir = args.raw_dir.expanduser().resolve()
    dataset_dir = args.dataset_dir.expanduser().resolve()
    result_csv = args.result_csv.expanduser().resolve()
    output_root = args.output_dir.expanduser().resolve()

    name_to_obj_id, obj_id_to_name = load_object_mappings(dataset_dir)
    if args.list_objects:
        print_available_objects(name_to_obj_id)
        return

    grasp_points = validate_grasp_points(args.grasp_point)
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

    exact_mesh = exact_mesh_cache[selected_obj_id]
    print_mesh_bbox_info(exact_mesh, selected_object_name, selected_obj_id)

    for image_key in tqdm(selected_keys, desc=f"Mapping grasp points for {selected_object_name}"):
        scene_id, im_id = [int(x) for x in image_key.split("_")]
        image = load_image(raw_dir, dataset_dir, scene_id, im_id)
        cam_k = load_camera_matrix(dataset_dir, scene_id, im_id)

        overlay = draw_grasp_points(
            image=image,
            selected_predictions=predictions_per_image[image_key],
            mesh=exact_mesh,
            grasp_points=grasp_points,
            cam_k=cam_k,
            point_radius=args.point_radius,
        )
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

    print(
        f"Saved {len(selected_keys)} visualizations for '{selected_object_name}' "
        f"to {output_dir}"
    )


if __name__ == "__main__":
    main()
