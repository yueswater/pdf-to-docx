from pathlib import Path
from typing import Optional
from typing import Tuple

from pdf_to_docx.utils.time_tools.generate_timestamp import generate_timestamp


def generate_paths(
    pdf_path: Path,
    txt_path: Optional[str],
    md_path: Optional[str],
    docx_path: Optional[str],
    json_path: Optional[str],
    pdf_out_path: Optional[str],
    base_output_dir: Optional[Path] = None,
) -> Tuple[str, str, str, str, str]:
    """
    Generate default output paths if not provided.
    """
    if base_output_dir is None:
        base_output_dir = pdf_path.parent

    stem = pdf_path.stem
    time_stamp = generate_timestamp()

    if not txt_path:
        txt_path = base_output_dir / "tx" / f"{time_stamp}_{stem}.txt"
    if not md_path:
        md_path = base_output_dir / "md" / f"{time_stamp}_{stem}.md"
    if not docx_path:
        docx_path = base_output_dir / "doc" / f"{time_stamp}_{stem}.docx"
    if not json_path:
        json_path = base_output_dir / "json" / f"{time_stamp}_{stem}.json"
    if not pdf_out_path:
        pdf_out_path = base_output_dir / "pdf" / f"{time_stamp}_{stem}(annotated).pdf"

    return txt_path, md_path, docx_path, json_path, pdf_out_path
