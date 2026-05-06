import os
from pathlib import Path
import zipfile
import hydra
from omegaconf import DictConfig
from src.utils.logging import get_logger

logger = get_logger(__name__)


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="train",
)
def download(cfg: DictConfig) -> None:
    root_dir = Path(cfg.machine.root_dir)
    source_url = (
        "https://huggingface.co/datasets/nv-nguyen/gigaPose/resolve/main/templates.zip"
    )
    tmp_dir = root_dir / "datasets/tmp/"
    os.makedirs(tmp_dir, exist_ok=True)

    download_cmd = f"wget -O {tmp_dir}/templates.zip {source_url}"
    logger.info(f"Running {download_cmd}")
    os.system(download_cmd)

    zip_path = tmp_dir / "templates.zip"
    logger.info(f"Extracting {zip_path} to {tmp_dir}")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(tmp_dir)
    
    os.rename(
        tmp_dir / "templates",
        root_dir / "datasets/templates",
    )

if __name__ == "__main__":
    download()
