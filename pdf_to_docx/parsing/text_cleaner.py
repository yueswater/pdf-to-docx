import re

from pdf_to_docx.utils.text_tools.text_patterns import CJK_CHARACTERS
from pdf_to_docx.utils.text_tools.text_patterns import HTML_PATTERN
from pdf_to_docx.utils.text_tools.text_patterns import HTML_TAG


class TextCleaner:
    @staticmethod
    def get_page_line(line: str) -> str:
        return re.sub(HTML_PATTERN, r"\1", line)

    @staticmethod
    def remove_html_tag(line: str) -> str:
        """
        Remove html tag, e.g., <tag>Something inside</tag>
        """
        return re.sub(HTML_TAG, "", line)

    @staticmethod
    def normalize_line(line: str) -> str:
        """
        Normalize spacing among sentences.
        """
        return line.replace("\u3000", " ").replace("\xa0", " ").strip()

    @staticmethod
    def remove_dashes(line: str) -> str:
        """
        Replace dashes in different patterns with spacing.
        """
        dashes = str.maketrans("－—‐-", "    ")
        return line.translate(dashes)

    @staticmethod
    def clean_spacing(line: str) -> str:
        line = line.replace(r"\s+", " ")

        invalid_space_patterns = [
            # zh + space
            (rf"([{CJK_CHARACTERS}])\s+([{CJK_CHARACTERS}])", r"\1\2"),
            # zh + space + punct
            (rf"([{CJK_CHARACTERS}])\s+([，。！？；：、）】』」])", r"\1\2"),
            # punct + space + zh
            (rf"([，。！？；：、（【『「])\s+([{CJK_CHARACTERS}])", r"\1\2"),
            # number + space + zh
            (rf"(\d)\s+([{CJK_CHARACTERS}])", r"\1\2"),
            # zh + space + number
            (rf"([{CJK_CHARACTERS}])\s+(\d)", r"\1\2"),
        ]

        for pattern, repl in invalid_space_patterns:
            line = re.sub(pattern, repl, line)

        return line
