import logging
from datetime import datetime
from enum import Enum
from pathlib import Path


class LoggerStatus(Enum):
    QUIET = 1
    NORMAL = 2
    FULL = 3


def setup_logger(
    mode: LoggerStatus = LoggerStatus.FULL, level=logging.DEBUG, log_dir="logs"
):
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{datetime.now():%Y-%m-%d-%H-%M-%S}.log"

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    if mode == LoggerStatus.QUIET:
        logging.disable(logging.CRITICAL + 1)
        return

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(level)  # 控制寫入檔案的最低等級

    handlers = [file_handler]

    stream_handler = logging.StreamHandler()
    if mode == LoggerStatus.FULL:
        stream_handler.setLevel(logging.DEBUG)
    elif mode == LoggerStatus.NORMAL:
        stream_handler.setLevel(logging.INFO)
    handlers.append(stream_handler)

    formatter = logging.Formatter(
        "[%(levelname)s] %(asctime)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s"
    )

    for h in handlers:
        h.setFormatter(formatter)
        logger.addHandler(h)
