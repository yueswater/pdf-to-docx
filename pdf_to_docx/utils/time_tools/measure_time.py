import logging
import time
from functools import wraps

from pdf_to_docx.utils.logging_tools.logger import setup_logger, LoggerStatus
from pdf_to_docx.utils.time_tools.convert_hhmmss import convert_hhmmss

setup_logger(LoggerStatus.NORMAL)


def measure_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        logging.debug(f"Start running: {func.__name__}")
        result = func(*args, **kwargs)
        end = time.time()
        elapsed = end - start
        hh, mm, ss = convert_hhmmss(elapsed)
        logging.debug(f"Total elapsed time: {hh} hour(s) {mm} minute(s) {ss} second(s)")
        return result

    return wrapper
