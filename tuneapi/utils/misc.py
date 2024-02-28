# Copyright © 2024- Frello Technology Private Limited

import os
import hashlib
from datetime import datetime, timezone
from google.protobuf.timestamp_pb2 import Timestamp as Timestamp_pb

from tuneapi.utils.logger import logger


class SimplerTimes:
    """
    A class that provides a simpler interface to datetime and time modules.
    """

    tz = timezone.utc

    def get_now_datetime() -> datetime:  # type: ignore
        """Get the current datetime in UTC timezone"""
        return datetime.now(SimplerTimes.tz)

    def get_now_float() -> float:  # type: ignore
        """Get the current datetime in UTC timezone as a float"""
        return SimplerTimes.get_now_datetime().timestamp()

    def get_now_i64() -> int:  # type: ignore
        """Get the current datetime in UTC timezone as a int"""
        return int(SimplerTimes.get_now_datetime().timestamp())

    def get_now_str() -> str:  # type: ignore
        """Get the current datetime in UTC timezone as a string"""
        return SimplerTimes.get_now_datetime().strftime("%Y-%m-%d %H:%M:%S.%f")

    def i64_to_datetime(i64: int) -> datetime:  # type: ignore
        """Convert an int to datetime in UTC timezone"""
        return datetime.fromtimestamp(i64, SimplerTimes.tz)

    def get_now_human() -> str:  # type: ignore
        """Get the current datetime in UTC timezone as a human readable string"""
        return SimplerTimes.get_now_datetime().strftime("%A %d %B, %Y at %I:%M %p")

    def get_now_pb() -> Timestamp_pb:
        ts = Timestamp_pb()
        ts.GetCurrentTime()
        return ts

    def get_now_ns() -> int:
        return SimplerTimes.get_now_pb().ToNanoseconds()


def unsafe_exit(code=0):
    """
    why use os._exit over sys.exit:
    https://stackoverflow.com/questions/9591350/what-is-difference-between-sys-exit0-and-os-exit0
    https://stackoverflow.com/questions/19747371/python-exit-commands-why-so-many-and-when-should-each-be-used
    tl;dr: os._exit kills without cleanup and so it's okay on the Pod
    """
    logger.warning(
        f"Hard Exiting with code {code}. This can cause memory leaks and loose threads. Use safe_exit when local."
    )
    os._exit(code)


def safe_exit(code=0):
    """
    This function is used to exit the program safely. It is used to make sure that the program exits properly and all the resources are cleaned up.
    """
    logger.warning(f"Exiting with code {code}. This will try to cleanup resources.")
    exit(code)


def hashstr(item: str, fn="md5"):
    """Hash sting of any item"""
    return getattr(hashlib, fn)(item.encode("utf-8")).hexdigest()
