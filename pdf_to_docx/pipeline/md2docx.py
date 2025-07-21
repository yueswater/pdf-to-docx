import logging
import re
from pathlib import Path
from typing import Optional

from Markdown2docx import Markdown2docx
from pdf_to_docx.utils.io_tools.docx_mover import move_docx
from pdf_to_docx.utils.logging_tools.logger import setup_logger, LoggerStatus

setup_logger(LoggerStatus.NORMAL)
INVALID_XML_CHARS = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")


class DocxExporter:
    def __init__(
        self,
        md_path: Path,
        output_dir: Path = Path("./"),  # default output folder
        target_dir: Optional[Path] = None,
        docx_name: Optional[Path] = None,
    ):
        self.output_dir = output_dir
        self.target_dir = target_dir
        self.md_path = md_path
        self.file_name = md_path.stem
        if docx_name:
            self.docx_name = (
                docx_name
                if docx_name.suffix == ".docx"
                else docx_name.with_suffix(".docx")
            )
        else:
            self.docx_name = Path(self.file_name + ".docx")

    def clean_md_file(self):
        try:
            text = self.md_path.read_text(encoding="utf-8", errors="ignore")
            clean_text = INVALID_XML_CHARS.sub("", text)
            self.md_path.write_text(clean_text, encoding="utf-8")
        except Exception as e:
            logging.warning(f"Failed to clean markdown file {self.md_path}: {e}")

    def export_docx(self):
        try:
            self.clean_md_file()
            md_file = str(self.md_path.with_suffix(""))
            project = Markdown2docx(md_file)  # stem path without .md
            project.eat_soup()
            project.save()
            self.move_to_target()
        except Exception as e:
            logging.error(f"Docx export failed for {self.md_path}: {e}")
            return

    def move_to_target(self):
        if not self.docx_name or not isinstance(self.docx_name, Path):
            logging.error("Invalid docx_name. Cannot move file.")
            return

        src_path = self.output_dir / f"{self.file_name}.docx"

        if not src_path.exists():
            logging.error(f"Source file not found: {src_path}")
            return

        if self.target_dir:
            dest_path = self.target_dir / self.docx_name
            try:
                self.target_dir.mkdir(parents=True, exist_ok=True)
                move_docx(src_path, dest_path)
            except OSError:
                logging.error("Failed to move docx file.")
            else:
                logging.debug(f"Moved docx to {dest_path}")
        else:
            logging.debug(f"No target dir specified, file remains at {src_path}")
