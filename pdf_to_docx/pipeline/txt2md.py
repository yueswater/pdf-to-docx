import logging
import re
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Callable
from typing import ClassVar
from typing import Dict
from typing import List
from typing import Literal
from typing import Pattern
from typing import TypedDict

from pdf_to_docx.parsing.text_cleaner import TextCleaner
from pdf_to_docx.utils.logging_tools.logger import setup_logger, LoggerStatus
from pdf_to_docx.utils.text_tools.text_patterns import CJK_NUM

setup_logger(LoggerStatus.NORMAL)


def with_pattern(level_name):
    def decorator(func):
        def wrapper(self, line: str):
            if level_name not in self.LEVEL_PATTERNS:
                raise ValueError(f"Invalid level name: {level_name}")
            pattern = self.LEVEL_PATTERNS[level_name]["pattern"]
            function = self.LEVEL_PATTERNS[level_name]["function"]
            matcher = pattern.match if function == "match" else pattern.search
            try:
                match = matcher(line)
            except Exception as e:
                logging.exception(f"Regex match error on line {line}: {e}")

            if not match:
                return [line]
            return func(self, line, match)

        return wrapper

    return decorator


class LevelConfig(TypedDict):
    pattern: Pattern[str]
    md_level: str
    function: Literal["match", "search"]


@dataclass(kw_only=True)  # Only keyword argument accepted
class TextToMarkdown:
    input_path: Path
    output_path: Path
    lines: List[str] = field(default_factory=list)

    # Common sentence ending
    COMMON_END: ClassVar[Pattern[str]] = re.compile(r"[。.！!？?：:；;]$")
    # Markdown level heading
    MD_LEVELS: ClassVar[Dict[int, str]] = {1: "#", 2: "##", 3: "###", 4: "####"}
    # Level heading mapping
    LEVEL_PATTERNS: ClassVar[Dict[str, LevelConfig]] = {
        "chapter": {
            "pattern": re.compile(rf"^(第[{CJK_NUM}]+章)(?:\s+(.*))?"),
            "md_level": MD_LEVELS[1],
            "function": "match",
        },
        "section": {
            "pattern": re.compile(rf"^(第[{CJK_NUM}]+節)(?:\s+(.*))?"),
            "md_level": MD_LEVELS[2],
            "function": "match",
        },
        "article": {
            "pattern": re.compile(rf"([{CJK_NUM}]{{1,3}}、)"),
            "md_level": MD_LEVELS[3],
            "function": "match",
        },
        "paragraph": {
            "pattern": re.compile(rf"([（\(][{CJK_NUM}]{{1,3}}[）\)])"),
            "md_level": MD_LEVELS[4],
            "function": "match",
        },
        "item": {
            "pattern": re.compile(r"(\d+[\.、。．)])"),
            "md_level": "ordered_item",
            "function": "search",
        },
    }
    # Level heading mapping handler
    LEVEL_HANDLERS: Dict[str, Callable[[str], List[str]]] = field(init=False)

    def __repr__(self):
        return f"<TextToMarkdown: {len(self.lines)} lines, output: {self.output_path}>"

    def __post_init__(self):
        self.LEVEL_HANDLERS = {
            "chapter": self._split_chapter_title,
            "section": self._split_section_title,
            "article": self._split_article,
            "paragraph": self._split_paragraph,
            "item": self._split_item,
        }
        if not self.lines:
            logging.warning(
                f"Warning: lines is empty. Check input path: {self.input_path}"
            )

    @classmethod
    def from_file(cls, input_path: Path, output_path: Path):
        if not input_path.exists():
            raise FileNotFoundError(f"File not found: {input_path}")
        with open(input_path, encoding="utf-8") as f:
            lines = [
                TextCleaner.remove_html_tag(TextCleaner.normalize_line(line))
                for line in f.readlines()
            ]
        return cls(input_path=input_path, output_path=output_path, lines=lines)

    @classmethod
    def from_text(cls, full_text: List[str], input_path: Path, output_path: Path):
        if not full_text:
            raise ValueError("Required full text.")
        lines = [
            TextCleaner.remove_html_tag(TextCleaner.normalize_line(line))
            for raw in full_text
            for line in raw.splitlines()
        ]
        return cls(input_path=input_path, output_path=output_path, lines=lines)
    
    @classmethod
    def is_structured_headline(cls, text: str) -> bool:
        for conf in cls.LEVEL_PATTERNS.values():
            pattern = conf["pattern"]
            matcher = pattern.match if conf["function"] == "match" else pattern.search
            if matcher(text.strip()):
                return True
        return False
    
    @with_pattern("chapter")
    def _split_chapter_title(self, _: str, match) -> List[str]:
        groups = match.groups()
        chapter = groups[0]
        rest = groups[1] if len(groups) > 1 and groups[1] else ""
        return [f"{self.LEVEL_PATTERNS['chapter']['md_level']} {chapter}"] + (
            [rest.strip()] if rest.strip() else []
        )

    @with_pattern("section")
    def _split_section_title(self, line: str, match) -> List[str]:
        groups = match.groups()
        section = groups[0]
        rest = groups[1] if len(groups) > 1 and groups[1] else ""
        before = line[: match.start()].strip()
        result = [before] if before else []
        result.append(f"{self.LEVEL_PATTERNS['section']['md_level']} {section}")
        if rest.strip():
            result.append(rest.strip())
        return result

    @with_pattern("article")
    def _split_article(self, line: str, match) -> List[str]:
        article = match.group(1)
        rest = line[match.end() :].strip()
        return [f"{self.LEVEL_PATTERNS['article']['md_level']} {article}{rest}"]

    @with_pattern("paragraph")
    def _split_paragraph(self, line: str, match) -> List[str]:
        paragraph = match.group(1)
        rest = line[match.end() :].strip()
        return [f"{self.LEVEL_PATTERNS['paragraph']['md_level']} {paragraph}{rest}"]

    def _split_item(self, line: str) -> List[str]:
        pattern = self.LEVEL_PATTERNS["item"]["pattern"]
        result = []

        matches = list(pattern.finditer(line))
        if not matches:
            return [line]

        for idx, match in enumerate(matches):
            num = re.sub(r"[、。．.)]", "", match.group(1))
            start = match.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(line)
            content = line[start:end].strip()
            result.append(f"{num}. {content.strip()}")
        return result

    def _is_new_block(self, line: str) -> bool:
        for _, mapping in self.LEVEL_PATTERNS.items():
            pattern = mapping["pattern"]
            function = mapping["function"]
            matcher = pattern.match if function == "match" else pattern.search
            if matcher(line):
                return True
        return False

    def _merge_lines(self) -> List[str]:
        merged = []
        for line in self.lines:
            if not line:
                continue
            if merged:
                prev_line = merged[-1]
                if not re.search(self.COMMON_END, prev_line) and not self._is_new_block(
                    line
                ):
                    merged[-1] += " " + line.strip()
                    continue
            merged.append(line.strip())
        return merged

    def _process_line(self, line: str) -> List[str]:
        try:
            for key, handler in self.LEVEL_HANDLERS.items():
                pattern, function = (
                    self.LEVEL_PATTERNS[key]["pattern"],
                    self.LEVEL_PATTERNS[key]["function"],
                )
                matcher = pattern.match if function == "match" else pattern.search
                if matcher(line):
                    return handler(line)
        except Exception as e:
            logging.exception(f"Failed to process line on {line}")
        return [line]

    def convert_to_markdown(self, debug: bool = False) -> List[str]:
        merged = self._merge_lines()
        result = []

        for line in merged:
            for md_line in self._process_line(line):
                cleaned = TextCleaner.clean_spacing(md_line)
                if cleaned.strip():
                    if debug:
                        logging.info(f"{line} → {cleaned}")
                    result.append(cleaned)

        # Filter spacing before return
        result = [line for line in result if line.strip()]
        return result

    def save(self):
        if not self.output_path:
            raise ValueError("Required output path.")

        processed = self.convert_to_markdown()

        try:
            with open(self.output_path, "w", encoding="utf-8") as f:
                f.write("\n".join(processed))
                logging.debug(f"Saved to {self.output_path}")
        except FileNotFoundError:
            logging.error(f"File not found: {self.output_path}")
        except PermissionError:
            logging.error(f"Permission denied: {self.output_path}")
        except OSError as e:
            logging.error(f"OS error when writing to file: {e}")
