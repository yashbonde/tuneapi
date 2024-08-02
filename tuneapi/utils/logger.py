# Copyright © 2024- Frello Technology Private Limited
# REMEMBER: nothing from outside tune should be imported in utils

import logging
from typing import Optional
from contextlib import contextmanager

from tuneapi.utils.env import ENV


def get_logger(name: str = "tuneapi") -> logging.Logger:
    """Returns a logger object"""
    lvl = ENV.TUNEAPI_LOG_LEVEL("info").upper()
    name = name or ENV.TUNEAPI_LOG_NAME(name)
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, lvl))
    log_handler = logging.StreamHandler()
    log_handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        )
    )
    logger.addHandler(log_handler)
    return logger


logger = get_logger()
"""
Logger provided with the package, try to use this logger for all logging purposes
"""


@contextmanager
def warning_with_fix(msg: str, fix: Optional[str]):
    msg = f"""
Deprecation Warning:
    err: {msg}
""".strip()
    if fix:
        msg += f"   fix: {fix}"
    logger.warning(msg)
