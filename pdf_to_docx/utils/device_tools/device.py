import logging
import platform

import torch
from pdf_to_docx.utils.logging_tools.logger import LoggerStatus
from pdf_to_docx.utils.logging_tools.logger import setup_logger

setup_logger(LoggerStatus.FULL)


def get_best_device() -> torch.device:
    """
    Set up GPU depends on Operating System working on.
    """
    system = platform.system()

    # macOS
    if system == "Darwin":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Linux
    elif system == "Linux":
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            device = torch.device("xpu")
        else:
            device = torch.device("cpu")

    # Windows
    elif system == "Windows":
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            device = torch.device("xpu")
        else:
            device = torch.device("cpu")

    # Others
    else:
        device = torch.device("cpu")

    logging.critical(f"Working on system {system} with device: {device}\n")
    return device
