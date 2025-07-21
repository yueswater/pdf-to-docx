import re
from typing import List

from pdf_to_docx.utils.text_tools.text_patterns import VALID_TEXT

VALID_TEXT_RATIO: float = 0.7
MIN_VALID_CHAR_COUNT: int = 100  # Minimum legal number of characters


def valid_pdf(text: List[str]) -> bool:
    all_text = "\n".join(text)
    total = len(all_text.replace("\n", ""))
    valid_chars = sum(len(m.group()) for m in re.finditer(VALID_TEXT, all_text))
    ratio_ok = total > 0 and (valid_chars / total) >= VALID_TEXT_RATIO
    count_ok = valid_chars >= MIN_VALID_CHAR_COUNT
    return ratio_ok and count_ok
