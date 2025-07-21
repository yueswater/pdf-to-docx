import logging
import shutil
from pathlib import Path
from typing import Dict
from typing import List
from typing import Union

from pdf_to_docx.utils.logging_tools.logger import LoggerStatus
from pdf_to_docx.utils.logging_tools.logger import setup_logger

setup_logger(LoggerStatus.FULL)


def make_subfolder(base_folder: Union[str, Path], subfolder: str) -> str:
    if isinstance(base_folder, Path):
        base_folder = str(base_folder)

    return f"{base_folder}/{subfolder}"


def make_output_dirs(
    output_base: Path, subfolders: List[str], overwrite: bool = True
) -> Dict[str, str]:
    # remove existing output folder
    output_base = Path(output_base)
    if overwrite and output_base.exists():
        shutil.rmtree(output_base)
    # create a new & clean output folder
    output_base.mkdir(parents=True, exist_ok=True)

    paths: Dict[str, Path] = {}
    for name in subfolders:
        subdir = output_base / name
        subdir.mkdir(parents=True, exist_ok=True)
        paths[name] = subdir

    all_paths = ", ".join(str(p) for p in paths.values())
    logging.info(f"Create folders: {all_paths}")
    return paths
