import logging
from datetime import datetime
import os

def get_rag_logger():
    logger = logging.getLogger("rag_logger")
    if logger.hasHandlers():
        return logger  # جلوگیری از چند بار تنظیم شدن لاگر

    logger.setLevel(logging.DEBUG)
    log_dir = os.path.join(os.path.dirname(__file__), "../logs")
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(log_dir, f"rag_log_{datetime.now().strftime('%Y%m%d')}.log")

    file_handler = logging.FileHandler(log_filename, encoding="utf-8")
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    return logger
