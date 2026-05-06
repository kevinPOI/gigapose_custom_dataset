from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from bop_toolkit_lib import inout


DEFAULT_CAM_K = np.array(
    [572.4114, 0.0, 320.0, 0.0, 573.57043, 240.0, 0.0, 0.0, 1.0],
    dtype=np.float64,
).reshape(3, 3)
MIN_LONGEST_SIDE_M = 0.01
MAX_LONGEST_SIDE_M = 0.10


def parse_args():
    parser = ArgumentParser(
        description="Visualize custom-dataset GigaPose predictions with Open3D."
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
        help="Prepared GigaPose dataset directory containing models/models_info.json.",
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
            "/home/zhenrant/gigapose/gigaPose_datasets/results/large_myprints/visualizations"
        ),
        help="Where to save visualization images.",
    )
    parser.add_argument("--scene-id", type=int, default=1, help="Scene id filter.")
    parser.add_argument("--im-id", type=int, default=None, help="Optional image id filter.")
    parser.add_argument("--max-images", type=int, default=50, help="Maximum images to render.")
    parser.add_argument("--axis-size", type=float, default=0.03, help="Axis size in meters.")
    parser.add_argument("--scene-width", type=int, default=960, help="Open3D panel width.")
    parser.add_argument("--scene-height", type=int, default=960, help="Open3D panel height.")
    return parser.parse_args()


def import_open3d():
    try:
        import open3d as o3d
    except ImportError as exc:
        raise SystemExit(
            "Install Open3D in the current env with `pip install open3d` and rerun."
        ) from exc
    return o3d


def group_by_image_level(data, image_key="im_id"):
    data_per_image = {}
    for det in data:
        scene_id, im_id = int(det["scene_id"]), int(det[image_key])
        key = f"{scene_id:06d}_{im_id:06d}"
        data_per_image.setdefault(key, []).append(det)
    return data_per_image


def load_image(raw_dir: Path, dataset_dir: Path, scene_id: int, im_id: int):
    img_path = raw_dir / f"rgb_{im_id:04d}.png"
    if not img_path.exists():
        img_path = dataset_dir / "test_imagewise" / f"{scene_id:06d}_{im_id:06d}.rgb.png"
    return np.array(Image.open(img_path).convert("RGB"))


def build_mesh_cache(o3d, dataset_dir: Path):
    model_infos = inout.load_json(dataset_dir / "models" / "models_info.json")
    mesh_cache = {}
    for obj_id in model_infos:
        obj_id_int = int(obj_id)
        mesh_path = dataset_dir / "models" / f"obj_{obj_id_int:06d}.obj"
        if not mesh_path.exists():
            mesh_path = dataset_dir / "models" / f"obj_{obj_id_int:06d}.ply"
        if not mesh_path.exists():
            continue
        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        bbox = mesh.get_axis_aligned_bounding_box()
        extent = np.asarray(bbox.get_extent(), dtype=np.float64)
        longest_side = float(np.max(extent))
        if longest_side > 0:
            target_longest_side = min(max(longest_side, MIN_LONGEST_SIDE_M), MAX_LONGEST_SIDE_M)
            mesh.scale(target_longest_side / longest_side, center=bbox.get_center())
        mesh.compute_vertex_normals()
        mesh_cache[obj_id_int] = mesh
    return mesh_cache


def make_transform(pred):
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = np.asarray(pred["R"], dtype=np.float64).reshape(3, 3)
    pose[:3, 3] = np.asarray(pred["t"], dtype=np.float64).reshape(3) * 0.001
    return pose


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
        pose = make_transform(pred)
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

    intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, *DEFAULT_CAM_K[[0, 1], [0, 1]], DEFAULT_CAM_K[0, 2], DEFAULT_CAM_K[1, 2])
    extrinsic = np.eye(4, dtype=np.float64)
    renderer.setup_camera(intrinsic, extrinsic)
    scene.camera.set_projection(
        DEFAULT_CAM_K,
        0.01,
        max(5.0, extent * 10.0),
        width,
        height,
    )

    image = np.asarray(renderer.render_to_image())
    return image


def main():
    args = parse_args()
    o3d = import_open3d()

    raw_dir = args.raw_dir.expanduser().resolve()
    dataset_dir = args.dataset_dir.expanduser().resolve()
    result_csv = args.result_csv.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = inout.load_bop_results(result_csv)
    predictions = group_by_image_level(predictions)
    mesh_cache = build_mesh_cache(o3d, dataset_dir)

    selected_keys = sorted(predictions.keys())
    if args.scene_id is not None:
        selected_keys = [k for k in selected_keys if int(k.split("_")[0]) == args.scene_id]
    if args.im_id is not None:
        selected_keys = [k for k in selected_keys if int(k.split("_")[1]) == args.im_id]
    if args.max_images is not None:
        selected_keys = selected_keys[: args.max_images]

    for image_key in tqdm(selected_keys, desc="Visualizing predictions"):
        scene_id, im_id = [int(x) for x in image_key.split("_")]
        original = load_image(raw_dir, dataset_dir, scene_id, im_id)
        scene_view = render_open3d_scene(
            o3d,
            mesh_cache,
            predictions[image_key],
            width=args.scene_width,
            height=args.scene_height,
            axis_size=args.axis_size,
        )

        if original.shape[0] != scene_view.shape[0]:
            scale = scene_view.shape[0] / original.shape[0]
            new_width = int(original.shape[1] * scale)
            original = np.array(Image.fromarray(original).resize((new_width, scene_view.shape[0])))

        vis = np.concatenate([original, scene_view], axis=1)
        Image.fromarray(vis).save(output_dir / f"{image_key}.png")


if __name__ == "__main__":
    main()
