# Copyright © 2024- Frello Technology Private Limited
# REMEMBER: nothing from outside tune should be imported in utils

import random
import string
from threading import Lock
from snowflake import SnowflakeGenerator


def get_random_string(length: int, numbers: bool = True, special: bool = False) -> str:
    choice_items = string.ascii_letters
    if numbers:
        choice_items += string.digits
    if special:
        choice_items += string.punctuation
    return "".join((random.choice(choice_items) for _ in range(length)))


class SFGen:
    CURRENT_EPOCH_START = 1705905900000  # UTC timezone
    """Start of the current epoch, used for generating snowflake ids"""

    def __init__(self, instance: int = None, epoch=CURRENT_EPOCH_START):
        if instance is None:
            # largest value of instance has to be <1024, so we randomly create one when initialising ie. < `0x400`
            instance = "0x3" + "".join(random.choices("0123456789abcedf", k=2))
            instance = int(instance, 16)
        self.gen = SnowflakeGenerator(instance, epoch=epoch)
        self.lock = Lock()

    def __call__(self, as_int=False) -> int:
        with self.lock:
            if as_int:
                return next(self.gen)
            else:
                return f"{next(self.gen):17d}"


get_snowflake = SFGen()
