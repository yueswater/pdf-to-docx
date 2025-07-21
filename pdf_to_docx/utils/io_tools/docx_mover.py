import shutil
from pathlib import Path


def move_docx(origin_path: Path, dest_path: Path):
    shutil.move(str(origin_path), str(dest_path))
