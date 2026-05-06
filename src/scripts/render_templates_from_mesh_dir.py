import argparse
import multiprocessing
import shutil
import subprocess
import time
from functools import partial
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.lib3d.template_transform import get_obj_poses_from_template_level
from src.utils.logging import get_logger

logger = get_logger(__name__)


def render_one(
    idx_obj,
    mesh_paths,
    output_dirs,
    pose_paths,
    num_gpus,
    disable_output,
    renderer,
    mesh_scale,
):
    mesh_path = Path(mesh_paths[idx_obj]).resolve()
    output_dir = Path(output_dirs[idx_obj]).resolve()
    pose_path = Path(pose_paths[idx_obj]).resolve()

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gpu_id = idx_obj % max(num_gpus, 1)

    if renderer == "blenderproc":
        cmd = [
            "blenderproc",
            "run",
            "./src/lib3d/blenderproc.py",
            str(mesh_path),
            str(pose_path),
            str(output_dir),
            str(gpu_id),
            "true" if disable_output else "false",
            "true",
            str(mesh_scale),
        ]
    else:
        cmd = [
            "python",
            "-m",
            "src.custom_megapose.call_panda3d",
            str(mesh_path),
            str(pose_path),
            str(output_dir),
            str(gpu_id),
            "true" if disable_output else "false",
            "true",
            str(mesh_scale),
        ]

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        logger.error("Renderer failed for %s with exit code %s", mesh_path, result.returncode)
        return False

    expected_num_pngs = len(np.load(pose_path)) * 2
    actual_num_pngs = len(list(output_dir.glob("*.png")))
    if actual_num_pngs != expected_num_pngs:
        logger.warning(
            "Unexpected number of pngs for %s: got %s, expected %s",
            mesh_path,
            actual_num_pngs,
            expected_num_pngs,
        )
        return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mesh_dir", help="Directory containing obj_XXXXXX.ply/obj files")
    parser.add_argument(
        "--output-dir",
        default="gigaPose_datasets/datasets/templates/itodd",
        help="Directory where templates will be written",
    )
    parser.add_argument(
        "--renderer",
        choices=["auto", "blenderproc", "panda3d"],
        default="auto",
        help="Rendering backend",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--disable-output", action="store_true")
    parser.add_argument(
        "--mesh-scale",
        type=float,
        default=1.0,
        help="Additional scale applied to rendered mesh geometry.",
    )
    args = parser.parse_args()

    mesh_dir = Path(args.mesh_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    object_pose_dir = output_dir / "object_poses"
    object_pose_dir.mkdir(parents=True, exist_ok=True)

    mesh_paths = sorted(mesh_dir.glob("obj_*.ply")) + sorted(mesh_dir.glob("obj_*.obj"))
    if not mesh_paths:
        raise FileNotFoundError(f"No obj_*.ply or obj_*.obj files found in {mesh_dir}")

    renderer = args.renderer
    if renderer == "auto":
        renderer = "blenderproc" if any(p.suffix.lower() == ".obj" for p in mesh_paths) else "blenderproc"

    template_poses = get_obj_poses_from_template_level(level=1, pose_distribution="all")
    template_poses[:, :3, 3] *= 0.4

    output_dirs = []
    pose_paths = []
    for mesh_path in mesh_paths:
        object_id = int(mesh_path.stem[4:])
        obj_output_dir = output_dir / f"{object_id:06d}"
        obj_pose_path = object_pose_dir / f"{object_id:06d}.npy"
        np.save(obj_pose_path, template_poses)
        output_dirs.append(obj_output_dir)
        pose_paths.append(obj_pose_path)

    logger.info("Rendering %s objects from %s", len(mesh_paths), mesh_dir)
    logger.info("Writing templates to %s", output_dir)
    logger.info("Using renderer: %s", renderer)

    start_time = time.time()
    worker = partial(
        render_one,
        mesh_paths=mesh_paths,
        output_dirs=output_dirs,
        pose_paths=pose_paths,
        num_gpus=args.num_gpus,
        disable_output=args.disable_output,
        renderer=renderer,
        mesh_scale=args.mesh_scale,
    )

    with multiprocessing.Pool(processes=args.num_workers) as pool:
        results = list(
            tqdm(pool.imap_unordered(worker, range(len(mesh_paths))), total=len(mesh_paths))
        )

    logger.info("Finished %s/%s objects", sum(results), len(mesh_paths))
    logger.info("Total time: %.2fs", time.time() - start_time)


if __name__ == "__main__":
    main()
