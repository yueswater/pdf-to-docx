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
class PageNumberLocator:
    reference_text: str
    reference_box: List[Tuple[float, float]]  # polygon of first detected page number
    tolerance: float = 20.0  # 可以自定義 XY 容忍範圍

    def is_same_position(self, box: List[Tuple[float, float]]) -> bool:
        def get_center(polygon):
            xs = [p[0] for p in polygon]
            ys = [p[1] for p in polygon]
            return sum(xs) / len(xs), sum(ys) / len(ys)

        cx1, cy1 = get_center(self.reference_box)
        cx2, cy2 = get_center(box)

        return abs(cx1 - cx2) <= self.tolerance and abs(cy1 - cy2) <= self.tolerance

    def is_similar_text(self, text: str) -> bool:
        return self.reference_text.strip() == text.strip()

    def match(self, text: str, box: List[Tuple[float, float]]) -> bool:
        return self.is_same_position(box)

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

    page_locator: Optional[PageNumberLocator] = field(default=None)

    # Parameters setting
    EPSILON: float = 1e-3  # tolerence of block criterion
    DPI: int = 200
    BIN_SIZE: int = 20
    MARGIN: int = 60
    Z_THRESHOLD: float = 1.5
    CONFIDENCE_LEVEL: float = 0.5  # confidence of the block being a valid sentence
    SHORT_LINE_NUM: int = 8
    PAGE_NUM_BOX_WIDTH: float = 1.0

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
    ]

    NON_PAGENUM_PATTERNS: ClassVar[List[Tuple[str, str]]] = [
        (r"^[(（][一二三四五六七八九十]{1,3}[)）]", "條文編號開頭"),
        (r"^[A-Z]{1,4}-\d+(\.\d+)*$", "英文開頭條號，如 R10L-2.7.1、F100-3.2"),
        (r"^\d+(\.\d+)+$", "多層章節號格式，如 2.1.3、5.1.2.3"),
        (r"^[A-Z]+\d{2,}$", "英文代碼與數字結尾，如 R088、T100"),
        (r"^[A-Z]{1,3}-[A-Z]{1,3}-\d+$", "英文模組代號，如 DOC-INT-001"),
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

    def extract_from_pdf(self) -> List[str]:
        """
        Extract full text directly from a PDF file.
        """
        full_text = []
        try:
            for i, page in enumerate(self.doc):
                blocks = page.get_text("blocks")
                _, pagenum_blocks, pagenum_texts = self.detect_pagenums(blocks, page.rect.height, i)

                # Save page number block and page number content
                # TODO: Don't save only the first one, you may be able to tell if you hate multiple pages (RESOLVED)
                if pagenum_blocks:
                    self.pagenum_blocks[i] = pagenum_blocks
                    self.page_numbers[i] = [TextCleaner.get_page_line(t) for t in pagenum_texts]
                    self.pagenum_info[i] = [{"box": b, "text": t} for b, t in zip(pagenum_blocks, pagenum_texts)]

                # Create page text
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
                kept_lines, ocr_pn_boxes, ocr_page_numbers = self.detect_ocr_pagenums(ocr_result, image_height)

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

    # TODO: 改為找到同一份文件頁碼的位置，判斷是否與既有的 coord 有交集，並看一下是否符合頁碼格式
    def is_line_pn(self, text: str, y: float, image_height: int, body_range: Tuple, mean_y: float, std_y: float, all_lines: List[Dict[str, Any]]) -> bool:
        # print(image_height)
        text = text.strip()
        body_range = tuple(sorted(body_range))
        all_lines = self.normalize_lines(all_lines)

        # Direct exclusion of non-page number formats
        if any(re.fullmatch(pattern, text) for pattern, _ in self.NON_PAGENUM_PATTERNS):
            return False
        if self.is_html_contains_num(text):
            return True
        if max(body_range) - min(body_range) <= self.MARGIN:
            return True

        # Preprocessing
        text = TextCleaner.remove_html_tag(text)
        text = TextCleaner.remove_dashes(text).strip()
        phrases = re.split(r"[，,。:：\s]+", text)

        # Position
        in_body = body_range[0] <= y <= body_range[1]
        is_top_or_bottom = y < image_height * 0.25 or y > image_height * 0.75
        z = abs(y - mean_y) / max(std_y, self.EPSILON)

        # Scan each phrase to see if there are decent page numbers
        for phrase in phrases:
            # If empty, skip
            if not phrase:
                continue

            # If it is too long, split the text block
            phrase = phrase.strip()

            # Exclude common formats
            if re.search(r"\d{4}/\d{1,2}/\d{1,2}", phrase):  # 2025/04/21
                continue
            if re.search(r"\d{2}:\d{2}(:\d{2})?", phrase):  # 10:48 or 10:48:57
                continue
            if re.search(r"\d{3}[./．]\d{2}[./．]\d{2}", phrase):  # 111.06.17
                continue

            for pattern, _ in self.NON_PAGENUM_PATTERNS:
                if re.fullmatch(pattern, phrase):
                    return False

            if any(re.fullmatch(p[0], phrase) for p in self.PAGENUM_PATTERNS):
                rounded_y = round(y, 3)
                same_y_lines = [line for line in all_lines if round(line["y_value"], 3) == rounded_y]
                if len(same_y_lines) > 1:
                    for line in same_y_lines:
                        other_text = line["text"]
                        if other_text.strip() == text.strip():
                            continue
                        if not any(re.fullmatch(p[0], other_text.strip()) for p in self.PAGENUM_PATTERNS):
                            if self.debug:
                                logging.debug(f"y 值為 {rounded_y} 的多行中包含非頁碼樣式「{other_text.strip()}」，略過頁碼判斷")
                            return False

                score: float = 0.0

                # bonus points: It looks really like a page number
                if re.search(r"第\s*\d+\s*頁\s*/\s*共\s*\d+\s*頁", phrase):
                    score += 2.0

                # Points deducted: Don't look like a page
                if re.search(r"[a-zA-Z]{2,}", phrase) and not re.fullmatch(r"[ivxlcdmIVXLCDM]+", phrase):
                    score -= 1.0
                if re.search(r"[年月日][:：]", phrase):
                    score -= 1.0
                if re.search(r"[^\w\s/]", phrase):
                    score -= 0.5

                if in_body:
                    score -= 0.8
                else:
                    score += 1.2

                score += 0.8 if z > self.Z_THRESHOLD else 0
                # score += 0.6 if is_top_or_bottom else 0

                weight = 1 / (1 + math.exp(-(z - 1.5))) # sigmoid
                score += weight * 1.2
                score += min(z, 2.0) * 0.2

                score += 1.2 if phrase.isdigit() and len(phrases) == 1 and len(phrase) <= 2 else 0
                score += 0.3 if re.fullmatch(r"[ivxlcdmIVXLCDM]+", phrase) else 0
                score += 0.4 if re.fullmatch(r"第?\s*\d{1,3}\s*頁", phrase) else 0

                line_info = self.print_line_info(
                    text,
                    y=y,
                    z=z,
                    in_body=in_body,
                    weight=weight,
                    score=score,
                    is_page_num=score>=2.0
                )

                if self.debug:
                    logging.debug(line_info)
                if score >= 1.8:
                    return True

        return False

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

    def filter_pagenum(
        self,
        lines: List,
        get_text: Callable,
        get_polygon: Callable,
        page_height: int
    ) -> Tuple[List, List, List]:
        kept_lines = []
        pn_boxes = []
        pn_texts = []

        for line in lines:
            text = get_text(line).strip()
            polygon = get_polygon(line)

            # 初始化 page_locator（若尚未建立）
            if self.page_locator is None:
                for pattern, _ in self.PAGENUM_PATTERNS:
                    if re.fullmatch(pattern, text):
                        self.page_locator = PageNumberLocator(reference_text=text, reference_box=polygon)
                        break

            # 若符合定位器的文字與位置
            if self.page_locator and self.page_locator.match(text, polygon):
                pn_boxes.append(polygon)
                pn_texts.append(text)
            else:
                kept_lines.append(line)

        # 過濾重複位置的頁碼區塊
        pn_boxes, pn_texts = self.refine_pagenum_blocks(pn_boxes, pn_texts, page_height)

        return kept_lines, pn_boxes, pn_texts


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

        # 第一次建立 page locator
        if self.page_locator is None:
            for line in pseudo_lines:
                text = line.text
                for pattern, _ in self.PAGENUM_PATTERNS:
                    if re.fullmatch(pattern, text.strip()):
                        self.page_locator = PageNumberLocator(text, line.polygon)
                        if self.debug:
                            logging.debug(f"[PageLocator Init] ref_text: {text}, box: {line.polygon}")
                        break
                if self.page_locator is not None:
                    break

        # 第二次全部走定位比對
        pn_boxes, pn_texts, kept_lines = [], [], []
        for line in pseudo_lines:
            if self.page_locator and self.page_locator.match(line.text, line.polygon):
                if self.debug:
                    logging.debug(f"[Match] Page {page_index}, matched text: {line.text}")
                pn_boxes.append(line.polygon)
                pn_texts.append(line.text)
            else:
                if self.debug:
                    logging.debug(f"[No Match] Page {page_index}, text: {line.text}")
                kept_lines.append(line)

        return kept_lines, pn_boxes, pn_texts


    def detect_ocr_pagenums(self, ocr_result: List[TextLine], image_height: int) -> Tuple[List, List, List]:
        pn_boxes, pn_texts, kept_lines = [], [], []

        for line in ocr_result:
            text = line.text.strip()
            polygon = line.polygon

            # 第一次建立定位器
            if self.page_locator is None:
                for pattern, _ in self.PAGENUM_PATTERNS:
                    if re.fullmatch(pattern, text):
                        self.page_locator = PageNumberLocator(text, polygon)
                        break

            if self.page_locator and self.page_locator.match(text, polygon):
                pn_boxes.append(polygon)
                pn_texts.append(text)
            else:
                kept_lines.append(line)

        return kept_lines, pn_boxes, pn_texts

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
