# Copyright © 2024-2025 Frello Technology Private Limited
# Copyright © 2025-2025 Yash Bonde github.com/yashbonde
# MIT License
# REMEMBER: nothing from outside tune should be imported in utils

import os
import base64
import hashlib

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from datetime import datetime, timezone, timedelta

from tuneapi.utils.logger import logger
from tuneapi.utils.serdeser import to_json


class SimplerTimes:
    """
    A class that provides a simpler interface to datetime and time modules.
    """

    tz = timezone.utc
    IST = timezone(timedelta(hours=5, minutes=30))

    def get_now_datetime() -> datetime:  # type: ignore
        """Get the current datetime in UTC timezone"""
        return datetime.now(SimplerTimes.tz)

    def get_now_float() -> float:  # type: ignore
        """Get the current datetime in UTC timezone as a float"""
        return SimplerTimes.get_now_datetime().timestamp()

    def get_now_fp64() -> float:  # type: ignore
        """Get the current datetime in UTC timezone as a float"""
        return SimplerTimes.get_now_datetime().timestamp()

    def get_now_i64() -> int:  # type: ignore
        """Get the current datetime in UTC timezone as a int"""
        return int(SimplerTimes.get_now_datetime().timestamp())

    def get_now_str() -> str:  # type: ignore
        """Get the current datetime in UTC timezone as a string"""
        return SimplerTimes.get_now_datetime().strftime("%Y-%m-%d_%H:%M:%S.%f")

    def get_now_human(date: bool = True, tz=None) -> str:
        """Get the current datetime in timezone as a human readable string"""
        fmt_str = "%I:%M %p"
        if date:
            fmt_str = "%A %d %B, %Y at " + fmt_str
        dt = SimplerTimes.get_now_datetime()
        if tz:
            fmt_str += " %Z"
            dt = dt.astimezone(tz)
        return dt.strftime(fmt_str)

    def get_now_pb():
        from google.protobuf.timestamp_pb2 import Timestamp as Timestamp_pb

        ts = Timestamp_pb()
        ts.GetCurrentTime()
        return ts

    def get_now_ns() -> int:
        return SimplerTimes.get_now_pb().ToNanoseconds()

    def i64_to_datetime(i64: int) -> datetime:  # type: ignore
        """Convert an int to datetime in UTC timezone"""
        return datetime.fromtimestamp(i64, SimplerTimes.tz)


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


def encrypt(text: str, password: str, salt: str):
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt.encode("utf-8"),
        iterations=480000,
    )
    key = base64.urlsafe_b64encode(kdf.derive(password.encode("utf-8")))
    f = Fernet(key)
    return f.encrypt(text.encode("utf-8"))


def decrypt(token: str, password: str, salt: str):
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt.encode("utf-8"),
        iterations=480000,
    )
    key = base64.urlsafe_b64encode(kdf.derive(password.encode("utf-8")))
    f = Fernet(key)
    return f.decrypt(token).decode("utf-8")


def generator_to_api_events(model, generator):
    """Takes in a token generator and creates OpenAI compatible event stream"""
    for token in generator:
        temp = {
            "model": None,
            "choices": [
                {"index": 0, "delta": {"content": None}, "finish_reason": None}
            ],
        }
        if model and type(model) == str:
            temp["model"] = model
        temp["choices"][0]["delta"]["content"] = token
        yield b"data: " + to_json(temp, tight=True).encode() + b"\r\n"
        yield b"\r\n"
    yield b"done: [DONE]"
