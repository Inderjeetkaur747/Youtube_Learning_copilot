import logging
import time
from contextlib import contextmanager
from .config import settings

logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(name)s - %(message)s")

def get_logger(name: str):
    return logging.getLogger(name)

@contextmanager
def timed(bucket: dict, key: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        bucket[key] = round(time.perf_counter() - start, 3)
