import argparse
from pathlib import Path

from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description="Scan a template directory for empty RGBA renders."
    )
    parser.add_argument(
        "--template-root",
        type=Path,
        required=True,
        help="Dataset-specific template directory, e.g. .../templates/myprints",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    template_root = args.template_root.expanduser().resolve()

    bad = []
    for obj_dir in sorted(
        [p for p in template_root.iterdir() if p.is_dir() and p.name.isdigit()]
    ):
        for png_path in sorted(obj_dir.glob("[0-9][0-9][0-9][0-9][0-9][0-9].png")):
            rgba = Image.open(png_path).convert("RGBA")
            if rgba.getbbox() is None:
                bad.append(png_path)

    print(f"Found {len(bad)} empty template images")
    for path in bad:
        print(path)


if __name__ == "__main__":
    main()
