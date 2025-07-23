import gc
import math
import logging
import os
import re
from collections import Counter
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any
from typing import Callable
from typing import ClassVar
from typing import Dict
from typing import List
from typing import Optional
from typing import Protocol
from typing import Tuple
from typing import Union

import fitz
import numpy as np
import torch
from pdf2image import convert_from_bytes
from pdf2image import convert_from_path
from pdf_to_docx.parsing.text_cleaner import TextCleaner
from pdf_to_docx.utils.device_tools.device import get_best_device
from pdf_to_docx.utils.text_tools.text_patterns import HTML_PATTERN
from pdf_to_docx.utils.validation.valid_pdf import valid_pdf
from pdf_to_docx.utils.geometry_tools import geometry
from pdf_to_docx.pipeline.txt2md import TextToMarkdown
from pdf_to_docx.parsing.text_cleaner import TextCleaner
from PIL import Image
from PIL import ImageDraw
from surya.detection import DetectionPredictor
from surya.recognition import RecognitionPredictor
from surya.recognition.schema import TextLine
from tqdm import tqdm

@dataclass
class PseudoLine:
    text: str
    polygon: List[Tuple[float, float]]  # [top-left, top-right, bottom-right, bottom-left]


@dataclass
class OCRModelManager:
    _instance: ClassVar[Optional["OCRModelManager"]] = None

    # Surya OCR tool initializer
    recognition_predictor: RecognitionPredictor
    detection_predictor: DetectionPredictor

    # Progress bar
    disable_tqdm = True

    @classmethod
    def init(cls, device: Optional[torch.device] = None, disable_tqdm: bool = True):
        if cls._instance is None:
            device = device or get_best_device()
            rec = RecognitionPredictor(device=device)
            det = DetectionPredictor(device=device)

            rec.disable_tqdm = disable_tqdm
            det.disable_tqdm = disable_tqdm
            cls._instance = cls(rec, det)
        return cls._instance

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            raise RuntimeError("OCRModelManager not initialized.")
        return cls._instance


class PDFTextStrategy(Protocol):
    def extract(self, extractor: "PDFTextExtractor") -> List[str]:
        ...


class TextLayerStrategy:
    def extract(self, extractor: "PDFTextExtractor") -> List[str]:
        return extractor.extract_from_pdf()


class OCRLayerStrategy:
    def extract(self, extractor: "PDFTextExtractor") -> List[str]:
        return extractor.extract_from_ocr()


class StrategySelector:
    @staticmethod
    def choose(extractor: "PDFTextExtractor") -> PDFTextStrategy:
        try:
            first_page_text = extractor.doc[0].get_text("text").strip()
        except Exception:
            return OCRLayerStrategy()
        # Check if the doc is a genuine PDF
        if valid_pdf(first_page_text):
            return TextLayerStrategy()
        else:
            return OCRLayerStrategy()


@dataclass
class PDFTextExtractor:
    pdf: Union[str, bytes]
    auto_clean: bool
    temp_path: str = "./temp"
    image_format: str = "PNG"

    # Progess Bar Setting
    disable_tqdm: bool = False

    # Page number detector debug
    debug: bool = False

    # Image & block storage
    images: List[Image.Image] = field(default_factory=list)
    pagenum_blocks: Dict = field(default_factory=dict)
    pagenum_info: Dict = field(default_factory=dict)
    page_numbers: Dict = field(default_factory=dict)

    ocr_page_number_boxes: Dict = field(default_factory=dict)
    ocr_page_info: Dict = field(default_factory=dict)
    ocr_page_numbers: Dict = field(default_factory=dict)

    global_coord_groups: Dict[int, List[Tuple]] = field(default_factory=dict)
    next_group_id: int = 0

    pagenum_ratio_threshold: float = 0.5

    # Parameters setting
    EPSILON: float = 1e-3  # tolerence of block criterion
    DPI: int = 200
    BIN_SIZE: int = 20
    MARGIN: int = 60
    Z_THRESHOLD: float = 1.5
    CONFIDENCE_LEVEL: float = 0.5  # confidence of the block being a valid sentence
    SHORT_LINE_NUM: int = 8
    PAGE_NUM_BOX_WIDTH: float = 1.0
    IOU_THRESHOLD: float = 0.5

    # Open pdf file
    doc: fitz.Document = field(init=False)

    def __post_init__(self):
        # Create temporary folder
        os.makedirs(self.temp_path, exist_ok=True)
        # Read pdf file
        if isinstance(self.pdf, (str, Path)):
            self.doc = fitz.open(filename=self.pdf)
        elif isinstance(self.pdf, bytes):
            self.doc = fitz.open(stream=self.pdf, filetype="pdf")
        else:
            raise TypeError(f"Unsupported pdf_path type: {type(self.pdf)}")
        # Initialize device
        OCRModelManager.init(disable_tqdm=True)

    HTML_TAG_RE = re.compile(HTML_PATTERN, re.DOTALL)
    DASH_TRANS = str.maketrans("－—‐-", "    ")
    PAGENUM_PATTERNS: ClassVar[List[Tuple[str, str]]] = [
        (r"第\s*\d+\s*頁\s*/\s*共\s*\d+\s*頁", "中式複合格式（含斜線）"),
        (r"第\s*(\d+)\s*頁\s*[,，]?\s*共\s*\d+\s*頁", "中文複合格式（含總頁數）"),
        (r"第\s*(\d+)\s*頁", "中文格式"),
        (r"Page\s+(\d+)", "英文格式"),
        (r"(\d+)\s*/\s*\d+", "分數格式"),
        (r"([ivxlcdmIVXLCDM]{1,10})\s*$", "羅馬數字"),
        (r"\b(\d{1,4})\b", "單一數字"),
        (r"^\d{1,4}/\d{1,4}$", "分數格式"),
        (r"^[\u4e00-\u9fa5]{1,4}\s*\d{1,3}$", "中文短詞 + 數字結尾"),
        (r".*[\u4e00-\u9fa5]{1,4}[-‐－—]\d{1,3}$", "短詞-數字格式（含多種破折號）"),
        (r"^[\u4e00-\u9fa5]{1,4}\s*\d{1,3}[-‐－—]\d{1,3}$", "中文短詞 + 數字-數字"),
        (r"^[\s\u4e00-\u9fa5A-Za-z\-－—]{2,}-\d{1,3}(\s+\d{2,4}\.\d{2}\s*版)?\s*$", "詞組破折號數字 + 可選後綴"),
        (r"^[\u4e00-\u9fa5A-Za-z\-－—]+-\d+(?:\s+)?\d{3}\.\d{2}版?$", "長中文-數字混合頁碼"),
        (r"^附件\s*\d{1,3}\s*$", "附件頁碼")
    ]

    NON_PAGENUM_PATTERNS: ClassVar[List[Tuple[str, str]]] = [
        (r"^[(（][一二三四五六七八九十]{1,3}[)）]", "條文編號開頭"),
        (r"^[A-Z]{1,4}-\d+(\.\d+)*$", "英文開頭條號，如 R10L-2.7.1、F100-3.2"),
        (r"^\d+(\.\d+)+$", "多層章節號格式，如 2.1.3、5.1.2.3"),
        (r"^[A-Z]+\d{2,}$", "英文代碼與數字結尾，如 R088、T100"),
        (r"^[A-Z]{1,3}-[A-Z]{1,3}-\d+$", "英文模組代號，如 DOC-INT-001"),
        (r"^第\s*[一二三四五六七八九十百千\d]+\s*條$", "法律條文「第X條」格式")
    ]

    def __call__(self, detect_only: bool = False):  # Can be called as a function
        strategy = StrategySelector.choose(self)
        self.mode = type(strategy).__name__
        if detect_only:
            return None
        return strategy.extract(self)  # i.e., text = extractor(), equivalent to text = extractor.extract()

    def clean_up(self):
        # Make sure clean existing temp folder
        for f in os.listdir(self.temp_path):
            try:
                os.remove(os.path.join(self.temp_path, f))
            except Exception:
                pass
        gc.collect()

    def _load_images(self) -> List[Image.Image]:
        """
        Different load images method:
            - Load images from path
            - Load images from bytes, i.e., load from an opened PDF file
        """
        if isinstance(self.pdf, (str, Path)):
            images = convert_from_path(str(self.pdf), dpi=self.DPI, thread_count=2)
        elif isinstance(self.pdf, bytes):
            images = convert_from_bytes(self.pdf, dpi=self.DPI)
        else:
            raise TypeError(f"Unsupported pdf_path type: {type(self.pdf)}")

        # Save images
        self.images = images
        for i, img in enumerate(tqdm(self.images, desc="Saving images", disable=self.disable_tqdm)):
            img_path = os.path.join(self.temp_path, f"p{i + 1:02d}.png")
            img.save(img_path, self.image_format)
        logging.debug(f"Saved {len(self.images)} images from PDF.")
        return images

    @staticmethod
    def is_same_block(block: Tuple, polygon: List[Tuple[float, float]]) -> bool:
        # coords from extracted PDF page
        x0, y0, x1, y1 = block[:4]
        # coords from detected PDF page
        bx0 = min(p[0] for p in polygon)
        by0 = min(p[1] for p in polygon)
        bx1 = max(p[0] for p in polygon)
        by1 = max(p[1] for p in polygon)
        return abs(x0 - bx0) < 1 and abs(y0 - by0) < 1 and abs(x1 - bx1) < 1 and abs(y1 - by1) < 1
    
    @staticmethod
    def _normalize_pagenum_text(text: str) -> str:
        # Clear all blanks (including full-shaped blanks \u3000)
        return re.sub(r"[\s\u3000]+", "", text)
    
    def _is_similar_pagenum_text(self, t1: str, t2: str) -> bool:
        norm1 = self._normalize_pagenum_text(t1)
        norm2 = self._normalize_pagenum_text(t2)
        for pattern, _ in self.PAGENUM_PATTERNS:
            if re.match(pattern, norm1) and re.match(pattern, norm2):
                return True
        return False

    def extract_from_pdf(self) -> List[str]:
        """
        Extract full text directly from a PDF file.
        Only retain page numbers that appear in consistent positions across multiple pages.
        """
        full_text = []
        try:
            # Prescan each page and create a page number candidate block group
            for i, page in enumerate(self.doc):
                blocks = page.get_text("blocks")
                self.detect_pagenums(blocks, page.rect.height, page_index=i)

            # Page group filtering: Only groups appearing on multiple pages are retained
            if self.doc.page_count > 1:
                total_pages = self.doc.page_count

                valid_groups = []
                for group_id, group in self.global_coord_groups.items():
                    polygons = [
                        f"[{round(min(p[0] for p in g[0]), 1)}, {round(min(p[1] for p in g[0]), 1)}, "
                        f"{round(max(p[0] for p in g[0]), 1)}, {round(max(p[1] for p in g[0]), 1)}]" 
                        for g in group
                    ]
                    texts = {g[1] for g in group}
                    pages = {g[2] for g in group}
                    ratio = len(pages) / total_pages
                    logging.debug(
                        f"[PAGE NUM GROUP] Group {group_id}: appears on {len(pages)} pages (ratio={ratio:.3f}), "
                        f"texts={texts}, pages={sorted(pages)}, polygons={polygons} → {'✅ keep' if ratio >= self.pagenum_ratio_threshold else '❌ drop'}"
                    )
                    if ratio >= self.pagenum_ratio_threshold:
                        valid_groups.append(group)
            else:
                valid_groups = list(self.global_coord_groups.values())

            # Create page number information for each page
            for polygon, text, page_index, _ in [g for group in valid_groups for g in group]:
                self.pagenum_blocks.setdefault(page_index, []).append(polygon)
                self.page_numbers.setdefault(page_index, []).append(TextCleaner.get_page_line(text))
                self.pagenum_info.setdefault(page_index, []).append({
                    "box": polygon,
                    "text": text,
                })

            #Second scan: Extract text and exclude valid page number blocks
            for i, page in enumerate(self.doc):
                blocks = page.get_text("blocks")
                pagenum_blocks = self.pagenum_blocks.get(i, [])

                page_text = ""
                for block in blocks:
                    if any(self.is_same_block(block, polygon) for polygon in pagenum_blocks):
                        continue
                    if len(block) >= self.SHORT_LINE_NUM and block[7] < self.CONFIDENCE_LEVEL:
                        continue
                    page_text += block[4]
                full_text.append(page_text.strip())

            if self.auto_clean:
                self.clean_up()

        except Exception as e:
            logging.exception("Failed on running extract from PDF: %s", e)

        return full_text


    def extract_from_ocr(self) -> List[str]:
        """
        Extract full text from a PDF file by using Surya-OCR method.
        """
        # Initialize recognizer and detector
        model_mgr = OCRModelManager.get_instance()
        recognition_predictor = model_mgr.recognition_predictor
        detection_predictor = model_mgr.detection_predictor

        # Start parsing
        full_text = []
        try:
            # Load images from PDF file
            self._load_images()
            for i, image in enumerate(tqdm(self.images, desc="Running OCR", disable=self.disable_tqdm, leave=False)):
                image = image.convert("RGB")
                predictions = recognition_predictor([image], det_predictor=detection_predictor)

                if not predictions:
                    full_text.append("")
                    continue

                # Fetch OCR result & image height
                ocr_result = predictions[0].text_lines
                image_height = image.height

                # Check if a line contains page number
                kept_lines, ocr_pn_boxes, ocr_page_numbers = self.detect_ocr_pagenums(ocr_result, image_height, page_index=i)

                if ocr_pn_boxes:
                    self.ocr_page_number_boxes[i] = ocr_pn_boxes
                    self.ocr_page_numbers[i] = [TextCleaner.get_page_line(text) for text in ocr_page_numbers]

                # Save parsing result
                page_text = "\n".join([line.text for line in kept_lines])
                full_text.append(page_text)

            total_pages = len(self.images)
            extracted_page_nums = self.ocr_page_numbers
            success_ratio = f"{len(extracted_page_nums)} / {total_pages}"

            self.ocr_page_info = {"Total Page": total_pages, "Extracted Page Numbers": self.ocr_page_numbers, "Success Rate": success_ratio}

            # Clean temporary folder to release memory
            if self.auto_clean:
                self.clean_up()
        except Exception as e:
            logging.debug("Failed on running OCR: %s", e)
            # Clean temporary folder to release memory
            if self.auto_clean:
                self.clean_up()

        return full_text

    def is_html_contains_num(self, line: str) -> bool:
        html_tag = re.findall(self.HTML_TAG_RE, line)
        if not html_tag:
            return False
        return any(re.search(r"\d+", text) for text in html_tag)

    @staticmethod
    def print_line_info(text: str, **kwargs: Any):
        line_info = {"text": text}
        for k, v in kwargs.items():
            # Convert NumPy types to native Python types
            if hasattr(v, "item"):  # np.float64, np.int64, np.bool_ all have .item()
                v = v.item()

            # Round native float
            if isinstance(v, float):
                v = round(v, 3)

            line_info[k] = v
        return line_info

    @staticmethod
    def normalize_lines(lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        normalized = []
        for line in lines:
            y = line["y_value"]
            for part in line["text"].splitlines():  # Split into multiple lines
                part = part.strip()
                if part:
                    normalized.append({"text": part, "y_value": y})
        return normalized

    # TODO: Instead, find the location of the page number of the same file, determine whether it has intersection with the existing coord, and see if it meets the page number format
    def is_line_pn(
        self,
        text: str,
        y: float,
        image_height: int,
        body_range: tuple,
        mean_y: float,
        std_y: float,
    ) -> bool:
        text = text.strip()

        if TextToMarkdown.is_structured_headline(text):
            return False
        if text.isdigit() and len(text) <= 3:
            is_pagenum_format = re.search(r"(第\s*\d+\s*頁|Page\s+\d+|\d+\s*/\s*\d+)", text)
            if not is_pagenum_format and not (y < image_height * 0.25 or y > image_height * 0.75):
                return False

        # Remove the tag and continue to make general judgments
        text = TextCleaner.remove_html_tag(text)
        text = TextCleaner.remove_dashes(text).strip()

        z = abs(y - mean_y) / max(std_y, self.EPSILON)
        # Necessary condition
        is_top_or_bottom = y < image_height * 0.25 or y > image_height * 0.75
        is_page_number = any(re.search(p[0], text) for p in self.PAGENUM_PATTERNS)
        necessary_cond = is_top_or_bottom and is_page_number
        
        if not necessary_cond:
            return False

        # Sufficient condition
        z = abs(y - mean_y) / max(std_y, self.EPSILON)
        add_score = sum(
            [
                z > self.Z_THRESHOLD,  # outlier
                not (body_range[0] <= y <= body_range[1]),  # not in body
                text.isdigit(),  # pure digit
                bool(re.search(r"\d+\s*$", text)),  # end with number
                bool(re.match(r"^[ivxlcdmIVXLCDM]+$", text)),  # roman numeral
            ]
        )

        if add_score >= 1:
            logging.debug(f"[PAGE_NUM?] '{text}' -> score: {add_score}, z={z:.2f}, y={y:.1f}")

        logging.debug(f"\nText: {text}")
        logging.debug(f"Length: {len(text)}")
        logging.debug(f"Position top and bottom: {is_top_or_bottom}")
        logging.debug(f"Required Conditions: {necessary_cond}")
        logging.debug(f"Extra score: {add_score}")
        logging.debug(f"Judgement result: {necessary_cond and add_score >= 1}")

        return add_score >= 1

    @staticmethod
    def box_contains(a, b):
        # a completely include b (excluding equal boundaries)
        ax0, ay0 = min(p[0] for p in a), min(p[1] for p in a)
        ax1, ay1 = max(p[0] for p in a), max(p[1] for p in a)
        bx0, by0 = min(p[0] for p in b), min(p[1] for p in b)
        bx1, by1 = max(p[0] for p in b), max(p[1] for p in b)
        return ax0 < bx0 and ay0 < by0 and ax1 > bx1 and ay1 > by1

    def refine_pagenum_blocks(self, pn_boxes: List, pn_texts: str, page_height: int):
        if not pn_boxes:
            return [], []

        y_centers = [(box[0][1] + box[2][1]) / 2 for box in pn_boxes]
        blocks = list(zip(pn_boxes, pn_texts, y_centers))

        upper_blocks = [b for b in blocks if b[2] < 0.5 * page_height]
        lower_blocks = [b for b in blocks if b[2] >= 0.5 * page_height]

        selected_blocks = []

        # Handle the upper area
        if upper_blocks:
            upper_blocks.sort(key=lambda x: x[2])  # y value is small before
            main_box, main_text, _ = upper_blocks[0]
            if len(upper_blocks) > 1:
                # _, _, second_y = upper_blocks[1]
                if self.box_contains(main_box, upper_blocks[1][0]) or self.box_contains(upper_blocks[1][0], main_box):
                    selected_blocks.append((main_box, main_text))
                else:
                    selected_blocks.append((main_box, main_text))
            else:
                selected_blocks.append((main_box, main_text))

        # Handle the lower area
        if lower_blocks:
            lower_blocks.sort(key=lambda x: -x[2])  # y value is ahead
            main_box, main_text, _ = lower_blocks[0]
            if len(lower_blocks) > 1:
                # _, _, second_y = lower_blocks[1]
                if self.box_contains(main_box, lower_blocks[1][0]) or self.box_contains(lower_blocks[1][0], main_box):
                    selected_blocks.append((main_box, main_text))
                else:
                    selected_blocks.append((main_box, main_text))
            else:
                selected_blocks.append((main_box, main_text))

        # Return to the unpacked box, text
        refined_boxes, refined_texts = zip(*[(b, t) for b, t in selected_blocks]) if selected_blocks else ([], [])
        return list(refined_boxes), list(refined_texts)

    def filter_pagenum(self, lines: List, get_text: Callable, get_polygon: Callable, page_height: int, page_index: int) -> Tuple[List, List, List]:
        kept_lines = []
        pn_boxes = []
        pn_texts = []
        
        coord_groups = {}
        groud_id = 0

        # Calculate y-axis centers & body range
        y_centers = np.array([(get_polygon(line)[0][1] + get_polygon(line)[2][1]) / 2 for line in lines])
        y_min, y_max = min(y_centers).tolist(), max(y_centers).tolist()
        mean_y = np.mean(y_centers)
        std_y = np.std(y_centers)
        body_range = (y_min + self.BIN_SIZE, y_max - self.BIN_SIZE)

        for line, y in zip(lines, y_centers):
            text = get_text(line)
            polygon = get_polygon(line)
            matched = False

            for subtext in text.splitlines():
                is_page_num = self.is_line_pn(
                    text=subtext,
                    y=y,
                    body_range=body_range,
                    image_height=page_height,
                    mean_y=mean_y,
                    std_y=std_y,
                )
                if is_page_num:
                    pn_boxes.append(get_polygon(line))
                    pn_texts.append(subtext)
                    matched = True
                    added = False
                    for _, group in self.global_coord_groups.items():
                        if any(geometry.compute_iou(polygon, other_block[0]) > self.IOU_THRESHOLD for other_block in group):
                            group.append((polygon, subtext, page_index, "unknown"))
                            added = True
                            break
                    if not added:
                        # fallback
                        merged_by_pattern = False
                        for gid, group in self.global_coord_groups.items():
                            if any(
                                self._is_similar_pagenum_text(subtext, existing_text) or
                                self._is_similar_pagenum_text(existing_text, subtext)
                                for _, existing_text, _, _ in group
                            ):
                                group.append((polygon, subtext, page_index, "fallback-text-match"))
                                logging.debug(f" → Fallback merged with Group {gid} by text pattern match: {subtext!r}")
                                merged_by_pattern = True
                                break

                        if not merged_by_pattern:
                            self.global_coord_groups[self.next_group_id] = [(polygon, subtext, page_index, "unknown")]
                            logging.debug(f" → New group #{self.next_group_id} created with: {subtext!r}, polygon={polygon}")
                            self.next_group_id += 1

                    break
            if not matched:
                kept_lines.append(line)

        if self.doc.page_count > 1:
            valid_groups = [
                group for group in self.global_coord_groups.values()
                if len(set(g[2] for g in group)) > 1
            ]
        else:
            valid_groups = list(coord_groups.values())

        pn_boxes = [g[0] for group in valid_groups for g in group]
        pn_texts = [g[1] for group in valid_groups for g in group]

        pn_boxes, pn_texts = self.refine_pagenum_blocks(pn_boxes, pn_texts, page_height)

        return kept_lines, [], []

    @staticmethod
    def convert_block_to_pseudoline(blocks: List[Tuple]) -> List[PseudoLine]:
        pseudo_lines: List[PseudoLine] = []

        # Convert blocks to PseudoLine and get y_center
        for block in blocks:
            text = block[4].strip()
            if not text:
                continue
            x0, y0, x1, y1 = block[:4]
            polygon = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
            pseudo_lines.append(PseudoLine(text=text, polygon=polygon))
        return pseudo_lines

    def detect_pagenums(self, blocks: List[Tuple], page_height: int, page_index: int) -> Tuple[List, List, List]:
        pseudo_lines = self.convert_block_to_pseudoline(blocks)
        if not pseudo_lines:
            return [], [], []

        # The page number content is not returned here, and the processing is delayed
        return self.filter_pagenum(
            lines=pseudo_lines,
            get_polygon=lambda l: l.polygon,
            get_text=lambda l: l.text,
            page_height=page_height,
            page_index=page_index
        )

    def detect_ocr_pagenums(self, ocr_result: List[TextLine], image_height: int, page_index: int) -> Tuple[List, List, List]:
        return self.filter_pagenum(lines=ocr_result, get_polygon=lambda l: l.polygon, get_text=lambda l: l.text, page_height=image_height, page_index=page_index)

    def original_pdf(self, output_path: Path) -> None:
        self.doc.save(output_path)

    def draw_pagenum_boxes(self, output_path: Path) -> None:
        for idx, page in enumerate(self.doc):
            if idx in self.pagenum_blocks:
                for polygon in self.pagenum_blocks[idx]:  # Every box must be drawn
                    x0 = min(p[0] for p in polygon)
                    y0 = min(p[1] for p in polygon)
                    x1 = max(p[0] for p in polygon)
                    y1 = max(p[1] for p in polygon)
                    rect = fitz.Rect(x0, y0, x1, y1)
                    page.draw_rect(rect, color=(1, 0, 0), width=self.PAGE_NUM_BOX_WIDTH)
        self.doc.save(output_path)

    def draw_ocr_pagenum_boxes(self, output_path: Path) -> None:
        # Attempt to get images list
        if not hasattr(self, "images"):
            raise RuntimeError("No images found. Run extract_from_ocr() first.")

        images = []
        for i, image in enumerate(self.images):
            im = image.convert("RGB")
            draw = ImageDraw.Draw(im)
            if i in self.ocr_page_number_boxes:
                for polygon in self.ocr_page_number_boxes[i]:
                    x0 = min(p[0] for p in polygon)
                    y0 = min(p[1] for p in polygon)
                    x1 = max(p[0] for p in polygon)
                    y1 = max(p[1] for p in polygon)
                    draw.rectangle([x0, y0, x1, y1], outline="red", width=2)

            images.append(im.copy())

        if images:
            images[0].save(output_path, save_all=True, append_images=images[1:])

    def get_page_info(self) -> Dict:
        """
        Return page number info for both PDF and OCR mode.
        """
        if self.ocr_page_numbers:
            return {
                "mode": self.mode,
                "total_pages": len(self.images),
                "page_numbers": self.ocr_page_numbers,
                "page_boxes": self.ocr_page_number_boxes,
                "info": self.ocr_page_info,
            }
        else:
            return {
                "mode": self.mode,
                "total_pages": len(self.doc),
                "page_numbers": self.page_numbers,
                "page_boxes": self.pagenum_blocks,
                "info": self.pagenum_info,
            }
