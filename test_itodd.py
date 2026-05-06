import argparse
import importlib
import os
import subprocess
import sys
import traceback
from pathlib import Path


DEFAULT_ROOT_DIR = Path("/sata1/data/kevin/bop_datasets")
DEFAULT_RUN_ID = "itodd_only"
DEFAULT_TEST_SETTING = "localization"
DEFAULT_SELECTED_OBJ_ID = 25
DEFAULT_CHECKPOINT_PATH = (
    Path(__file__).resolve().parent / "gigaPose_datasets" / "pretrained" / "gigaPose_v1.ckpt"
)
DEFAULT_TEMPLATE_DIR = (
    Path(__file__).resolve().parent / "gigaPose_datasets" / "datasets" / "templates"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run GigaPose coarse evaluation on ITODD with sensible defaults."
    )
    parser.add_argument(
        "--root-dir",
        default=str(DEFAULT_ROOT_DIR),
        help="Root directory that contains datasets/, pretrained/, and results/.",
    )
    parser.add_argument(
        "--run-id",
        default=DEFAULT_RUN_ID,
        help="Run identifier used in the output directory name.",
    )
    parser.add_argument(
        "--checkpoint-path",
        default=str(DEFAULT_CHECKPOINT_PATH),
        help="Checkpoint path for the coarse GigaPose model.",
    )
    parser.add_argument(
        "--template-dir",
        default=str(DEFAULT_TEMPLATE_DIR),
        help="Template root directory containing the dataset-specific template folders.",
    )
    parser.add_argument(
        "--test-setting",
        default=DEFAULT_TEST_SETTING,
        choices=["localization", "detection"],
        help="Evaluation setting for ITODD.",
    )
    parser.add_argument(
        "--selected-obj-id",
        type=int,
        default=DEFAULT_SELECTED_OBJ_ID,
        help="Only onboard and evaluate this object id; images without it are skipped.",
    )
    parser.add_argument(
        "--disable-output",
        action="store_true",
        help="Mirror test.py's disable_output=True behavior.",
    )
    parser.add_argument(
        "--dryrun",
        action="store_true",
        help="Run with machine.dryrun=True.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Extra Hydra overrides forwarded to test.py.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root_dir = Path(args.root_dir).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint_path).expanduser().resolve()
    template_dir = Path(args.template_dir).expanduser().resolve()
    processed_target_path = root_dir / "datasets" / "itodd" / "test_targets_bop19.json"
    raw_target_path = root_dir / "ITODD" / "itodd" / "test_targets_bop19.json"

    if not processed_target_path.exists() and raw_target_path.exists():
        processed_target_path.parent.mkdir(parents=True, exist_ok=True)
        if processed_target_path.exists() or processed_target_path.is_symlink():
            processed_target_path.unlink()
        os.symlink(raw_target_path, processed_target_path)

    try:
        importlib.import_module("src.models.gigaPose")
    except Exception:
        traceback.print_exc()
        raise SystemExit(
            "Failed to import src.models.gigaPose. The Hydra target is correct, "
            "but a Python dependency for the model is missing or broken in this environment."
        )
    try:
        importlib.import_module("src.dataloader.test")
    except Exception:
        traceback.print_exc()
        raise SystemExit(
            "Failed to import src.dataloader.test. The Hydra target is correct, "
            "but a Python dependency for the dataloader is missing or broken in this environment."
        )

    command = [
        sys.executable,
        "test.py",
        "test_dataset_name=itodd",
        f"machine.root_dir={root_dir}",
        f"model.checkpoint_path={checkpoint_path}",
        f"data.test.dataloader.template_config.dir={template_dir}",
        f"data.test.dataloader.selected_obj_id={args.selected_obj_id}",
        f"run_id={args.run_id}",
        f"test_setting={args.test_setting}",
        f"disable_output={str(args.disable_output).lower()}",
        f"machine.dryrun={str(args.dryrun).lower()}",
    ]
    command.extend(args.overrides)
    raise SystemExit(subprocess.call(command))


if __name__ == "__main__":
    main()
