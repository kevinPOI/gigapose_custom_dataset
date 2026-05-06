import argparse
import importlib
import subprocess
import sys
import traceback
from pathlib import Path


DEFAULT_ROOT_DIR = Path(__file__).resolve().parent / "gigaPose_datasets"
DEFAULT_RUN_ID = "myprints"
DEFAULT_TEST_SETTING = "localization"
DEFAULT_CHECKPOINT_PATH = DEFAULT_ROOT_DIR / "pretrained" / "gigaPose_v1.ckpt"
DEFAULT_TEMPLATE_DIR = DEFAULT_ROOT_DIR / "datasets" / "templates"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run GigaPose coarse inference on the custom myprints dataset."
    )
    parser.add_argument(
        "--root-dir",
        default=str(DEFAULT_ROOT_DIR),
        help="Root directory that contains datasets/, pretrained/, and results/.",
    )
    parser.add_argument(
        "--dataset-name",
        default="myprints",
        help="Custom dataset name under datasets/ and templates/.",
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
        help="Evaluation setting.",
    )
    parser.add_argument(
        "--selected-obj-id",
        type=int,
        default=None,
        help="Optional object id filter.",
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

    try:
        importlib.import_module("src.models.gigaPose")
        importlib.import_module("src.dataloader.test")
    except Exception:
        traceback.print_exc()
        raise SystemExit("Failed to import GigaPose modules in this environment.")

    command = [
        sys.executable,
        "test.py",
        f"test_dataset_name={args.dataset_name}",
        f"machine.root_dir={root_dir}",
        f"model.checkpoint_path={checkpoint_path}",
        f"data.test.dataloader.template_config.dir={template_dir}",
        f"run_id={args.run_id}",
        f"test_setting={args.test_setting}",
        f"disable_output={str(args.disable_output).lower()}",
        f"machine.dryrun={str(args.dryrun).lower()}",
    ]
    if args.selected_obj_id is not None:
        command.append(f"+data.test.dataloader.selected_obj_id={args.selected_obj_id}")
    command.extend(args.overrides)
    raise SystemExit(subprocess.call(command))


if __name__ == "__main__":
    main()
