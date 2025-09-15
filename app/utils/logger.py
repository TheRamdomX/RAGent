import logging
import sys

def get_logger(name: str = "study-bot"):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.DEBUG)
        fmt = logging.Formatter("%(asctime)s — %(name)s — %(levelname)s — %(message)s")
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    return logger

logger = get_logger()
