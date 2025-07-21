import builtins
import csv
import json
import logging
import re
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Union

import fitz
import numpy as np
from json_compare import JSONComparator
from pdf_to_docx.main import process_pdf_file
from pdf_to_docx.ocr.processor import PDFTextExtractor
from pdf_to_docx.utils.io_tools.generate_paths import generate_paths
from pdf_to_docx.utils.io_tools.json_reader import utf8_open
from pdf_to_docx.utils.io_tools.make_output_dirs import make_output_dirs
from pdf_to_docx.utils.io_tools.make_output_dirs import make_subfolder
from pdf_to_docx.utils.logging_tools.logger import LoggerStatus
from pdf_to_docx.utils.logging_tools.logger import setup_logger
from pdf_to_docx.utils.text_tools.format_size import format_size
from pdf_to_docx.utils.time_tools.timestamp import TimeStamp
from tqdm import tqdm

# Basic Setting
setup_logger(LoggerStatus.NORMAL)
builtins.open = utf8_open  # Force JSON file opened as UTF-8

@dataclass
class JSONCompare:
    log_folder: Path = Path("./comparison_logs")
    _comparator: JSONComparator = field(init=False, repr=False)

    def __post_init__(self):
        self.log_folder.mkdir(parents=True, exist_ok=True)

    def save_comparison_log(self, timestamp: str, prev_name: str, latest_name: str, diff_counts: int) -> None:
        # Generate timestamp
        if not timestamp:
            timestamp = TimeStamp.now()

        # Save comparison log file
        log_path = self.log_folder / f"{timestamp}_diff_summary.log"

        # Write in log file
        with open(log_path, "a", encoding="utf-8") as f:
            current_time = TimeStamp.iso()
            content = f"[{current_time}] {prev_name} vs {latest_name} ⭢ {diff_counts} diffs\n"
            f.write(content)
        return log_path

    def diff_counts(self, comparator: JSONComparator) -> int:
        return len(comparator.diff_log.log)

    def compare_json_files(self, prev_version: Union[str, Path], latest_version: Union[str, Path]) -> int:
        if not prev_version or not latest_version:
            raise ValueError("Previous & latest JSON files must be provided.")

        # TODO: Confirm whether JSONComparator is available
        comparator = JSONComparator(left_file_path=str(prev_version), right_file_path=str(latest_version))

        diff_counts = self.diff_counts(comparator=comparator)

        # Save comparison log
        self.save_comparison_log(
            timestamp=None,
            diff_counts=diff_counts,
            prev_name=Path(prev_version).name,
            latest_name=Path(latest_version).name,
        )
        return diff_counts

    def compare_versions(self, json_folder: str, prev_version: str, latest_version: str) -> List[int]:
        # Initialize folders
        base_path = Path(json_folder)
        prev_folder = base_path / prev_version
        latest_folder = base_path / latest_version

        # Fetch JSON files
        prev_jsons = {f.name: f for f in prev_folder.glob("*.json")}
        latest_jsons = {f.name: f for f in latest_folder.glob("*.json")}

        # Find common files
        common_files = set(prev_jsons.keys()) & set(latest_jsons.keys())

        if not common_files:
            logging.warning("No common JSON files found between versions.")

        # Compare JSON files
        diff_counts = []
        for fname in sorted(common_files):
            diff = self.compare_json_files(str(prev_jsons[fname]), str(latest_jsons[fname]))
            diff_counts.append(diff)

        # Show average diff counts
        avg_diff = np.mean(diff_counts) if diff_counts else 0
        logging.info(f"Compared {len(common_files)} JSON files")
        logging.info(f"Average diff count: {avg_diff:.2f}")
        return diff_counts

    def compare_latest_two_versions(self, json_folder: str) -> None:
        base_path = Path(json_folder)
        versions = sorted([f.name for f in base_path.iterdir() if f.is_dir() and re.match(r"\d{8}_\d{6}", f.name)])

        if len(versions) < 2:
            logging.warning("Not enoung versions to compare.")
            return

        prev_version = versions[-2]
        latest_version = versions[-1]
        diff_counts = self.compare_versions(
            json_folder=json_folder,
            prev_version=prev_version,
            latest_version=latest_version,
        )
        logging.info(f"Comparing latest two versions:\nPrev: {prev_version}\nLatest: {latest_version}\nDiff Counts: {diff_counts}")


@dataclass
class PDFBatchTest:
    version_name: str = field(init=False)
    data_dir: Path
    base_output_dir: Path

    def __post_init__(self):
        self.version_name = TimeStamp.now()
        self.json_key = f"json/{self.version_name}"
        self.data_dir = Path(self.data_dir)
        self.base_output_dir = Path(self.base_output_dir)
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        self.subfolders = {
            "tx": make_subfolder("tx", self.version_name),
            "md": make_subfolder("md", self.version_name),
            "doc": make_subfolder("doc", self.version_name),
            "json": make_subfolder("json", self.version_name),
            "pdf": make_subfolder("pdf", self.version_name),
        }
        self.folders = make_output_dirs(
            output_base=self.base_output_dir,
            subfolders=list(self.subfolders.values()),
            overwrite=False,
        )

        self.txt_output_dir = self.folders[self.subfolders["tx"]]
        self.md_output_dir = self.folders[self.subfolders["md"]]
        self.docx_output_dir = self.folders[self.subfolders["doc"]]
        self.json_output_dir = self.folders[self.subfolders["json"]]
        self.pdf_output_dir = self.folders[self.subfolders["pdf"]]

        self.mode_record: Dict[str, str] = {}

    @staticmethod
    def get_sort_key(f: Path):
        # Rank the page number first
        try:
            page_count = len(fitz.open(f))
        except Exception:
            page_count = float("inf")

        # Replace file size
        try:
            size = f.stat().st_size
        except Exception:
            size = float("inft")
        return (page_count, size)

    def list_files(self) -> List[Path]:
        files = [f for f in self.data_dir.iterdir() if f.name.endswith(".pdf") and not f.name.startswith(".")]

        # Add progress bar
        keys = []
        for f in tqdm(files, desc="計算頁數與大小並排序"):
            keys.append((self.get_sort_key(f), f))

        sorted_files = [f for _, f in sorted(keys, key=lambda x: x[0])]
        return sorted_files

    def pdf_batch(self):
        info_path = self.base_output_dir / "files_info.json"
        record_path = self.base_output_dir / f"{self.version_name}_result.json"

        record = {}

        try:
            with open(info_path, encoding="utf-8") as f:
                infos = json.load(f)
                logging.info(f"Loaded files info from {info_path}.")
        except (FileNotFoundError, json.JSONDecodeError):
            logging.warning(f"{info_path} not found or invalid. Rebuilding files info.")
            infos = self.files_info()

        files = [Path(f) for f in infos.keys()]
        pbar = tqdm(files, desc="Surya OCR batch", position=1)

        # TODO: disable global tqdm and enable only tqdm here
        for pdf_path in pbar:
            # Display recent processed file
            filename = str(pdf_path)
            info = infos.get(filename, {})
            pages = info.get("pages", "?")
            size = info.get("size", "?")

            display_file_name = Path(pdf_path).stem[:10]
            pbar.set_description(f"{display_file_name:<15} | {pages:>3}頁 | {size:>8}")

            try:
                extractor = PDFTextExtractor(pdf=pdf_path, auto_clean=True)
                _ = extractor(detect_only=True)
                mode = extractor.mode.lower()
            except Exception as e:
                logging.warning(f"Error detecting mode for {pdf_path.name}: {e}")
                self.mode_record[pdf_path.name] = "MODE_ERROR"
                continue

            if "ocr" in mode:
                self.mode_record[pdf_path.name] = "OCR_SKIPPED"
                continue
            elif "text" in mode:
                self.mode_record[pdf_path.name] = "PDF"

            # Define each output path
            txt_path = self.txt_output_dir / (pdf_path.stem + ".txt")
            md_path = self.md_output_dir / (pdf_path.stem + ".md")
            docx_path = self.docx_output_dir / (pdf_path.stem + ".docx")
            json_path = self.json_output_dir / (pdf_path.stem + ".json")
            pdf_out_path = self.pdf_output_dir / (pdf_path.stem + "(annotated).pdf")

            txt_path, md_path, docx_path, json_path, pdf_out_path = generate_paths(
                pdf_path=pdf_path,
                txt_path=txt_path,
                md_path=md_path,
                docx_path=docx_path,
                json_path=json_path,
                pdf_out_path=pdf_out_path,
                base_output_dir=self.base_output_dir,
            )

            try:
                convert_result = process_pdf_file(
                    pdf_path=pdf_path,
                    txt_path=txt_path,
                    md_path=md_path,
                    docx_path=docx_path,
                    json_path=json_path,
                    pdf_out_path=pdf_out_path,
                    use_openai=False,
                )
                if convert_result["success"]:
                    logging.debug(f"Converted: {pdf_path.name}")
                else:
                    logging.warning(f"Failed: {pdf_path.name}")
            except Exception as e:
                logging.exception(f"Error with {pdf_path.name}: {e}")

        logging.info("Batch processing complete.")
        json_files = [str(jf) for jf in self.json_output_dir.glob("*.json")]
        logging.info(f"JSON Files saved: {len(json_files)}")

        with open(record_path, "w", encoding="utf-8") as f:
            json.dump(self.mode_record, f, ensure_ascii=False, indent=4)
            logging.debug(f"Batch result record saved in {record_path}")
        return json_files

    def run(self):
        logging.info("Start running PDF batch testing")
        self.pdf_batch()

        # Save mode log
        with open(self.base_output_dir / f"mode_log_{self.version_name}.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "mode"])
            for fname, mode in self.mode_record.items():
                writer.writerow([fname, mode])

        # Compare JSON files
        json_folder = self.base_output_dir / "json"
        log_folder = self.base_output_dir / "logs"
        JSONCompare(log_folder=log_folder).compare_latest_two_versions(json_folder)

    def files_info(self, autosave: bool = True) -> Dict[str, Dict[str, Any]]:
        files = self.list_files()
        infos = {}
        info_path = self.base_output_dir / "files_info.json"

        for file in tqdm(files, desc="建立檔案基本資訊"):
            # Fetch file basic info
            filename = str(file)
            file_size = file.stat().st_size
            file_pages = len(fitz.open(filename=filename))

            # Save file info
            infos[filename] = {"size": format_size(file_size), "pages": file_pages}

        if autosave:
            try:
                if not info_path.exists():
                    with open(info_path, "w", encoding="utf-8") as f:
                        json.dump(infos, f, ensure_ascii=False, indent=4)
                        logging.info(f"File info has been saved in {info_path}.")
                else:
                    logging.info(f"Files info has been created in {info_path}.")
            except Exception:
                logging.exception(f"Failed to save file info to {info_path}.")

        return infos


if __name__ == "__main__":
    task = PDFBatchTest(data_dir="data", base_output_dir="batch/output")
    task.run()
