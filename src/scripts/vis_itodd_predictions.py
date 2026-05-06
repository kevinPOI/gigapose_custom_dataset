from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from bop_toolkit_lib import inout


def parse_args():
    parser = ArgumentParser(
        description="Visualize ITODD prediction CSV with side-by-side Open3D scenes."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("/sata1/data/kevin/bop_datasets/ITODD"),
        help="Raw ITODD dataset root containing test/ and models/.",
    )
    parser.add_argument(
        "--result-csv",
        type=Path,
        default=Path(
            "/sata1/data/kevin/bop_datasets/results/large_itodd_only/predictions/"
            "large-pbrreal-rgb-mmodel_itodd-test_itodd_only.csv"
        ),
        help="Prediction CSV in BOP format.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/sata1/data/kevin/bop_datasets/results/large_itodd_only/vis_open3d"),
        help="Where to save visualization images.",
    )
    parser.add_argument(
        "--scene-id",
        type=int,
        default=None,
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
        default=50,
        help="Maximum number of images to render.",
    )
    parser.add_argument(
        "--axis-size",
        type=float,
        default=0.03,
        help="Axis size in meters for each predicted object pose.",
    )
    parser.add_argument(
        "--scene-width",
        type=int,
        default=960,
        help="Width of the Open3D panel.",
    )
    parser.add_argument(
        "--scene-height",
        type=int,
        default=960,
        help="Height of the Open3D panel.",
    )
    return parser.parse_args()


def import_open3d():
    try:
        import open3d as o3d
    except ImportError as exc:
        raise SystemExit(
            "This visualizer now uses Open3D. Install it in perseve-sam2 with "
            "`pip install open3d` and rerun."
        ) from exc
    return o3d


def group_by_image_level(data, image_key="im_id"):
    data_per_image = {}
    for det in data:
        scene_id, im_id = int(det["scene_id"]), int(det[image_key])
        key = f"{scene_id:06d}_{im_id:06d}"
        if key not in data_per_image:
            data_per_image[key] = []
        data_per_image[key].append(det)
    return data_per_image


def load_image(scene_dir: Path, im_id: int):
    img_path = scene_dir / "gray" / f"{im_id:06d}.tif"
    image = Image.open(img_path)
    gray = np.array(image)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


def build_mesh_cache(o3d, dataset_dir: Path):
    model_infos = inout.load_json(dataset_dir / "models" / "models_info.json")
    mesh_cache = {}
    for obj_id in model_infos:
        obj_label = f"obj_{int(obj_id):06d}"
        mesh_path = (dataset_dir / "models" / obj_label).with_suffix(".ply")
        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        # ITODD meshes are in millimeters; convert to meters to match pose translations.
        mesh.scale(0.001, center=(0.0, 0.0, 0.0))
        mesh.compute_vertex_normals()
        mesh_cache[int(obj_id)] = mesh
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

    # Camera frame follows CV convention: z points forward, x right, y down.
    eye = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    center = np.array([0.0, 0.0, max(0.3, bbox.get_max_bound()[2] + 0.2)], dtype=np.float32)
    up = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    scene.camera.look_at(center, eye, up)
    scene.camera.set_projection(
        60.0,
        width / height,
        0.01,
        max(5.0, extent * 10.0),
        o3d.visualization.rendering.Camera.FovType.Vertical,
    )

    image = np.asarray(renderer.render_to_image())
    return image


def main():
    args = parse_args()
    o3d = import_open3d()

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
        scene_dir = dataset_dir / "test" / f"{scene_id:06d}"
        original = load_image(scene_dir, im_id)
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
            original = cv2.resize(
                original,
                (int(original.shape[1] * scale), scene_view.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

        vis = np.concatenate([original, scene_view], axis=1)
        Image.fromarray(vis).save(output_dir / f"{scene_id:06d}_{im_id:06d}.png")


if __name__ == "__main__":
    main()
