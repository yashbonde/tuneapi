# Copyright © 2024- Frello Technology Private Limited
# REMEMBER: nothing from outside tune should be imported in utils

import random
import string
from threading import Lock
from snowflake import SnowflakeGenerator

from tuneapi.utils.logger import logger


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


def reservoir_sampling(stream, k, seed: int = 4):
    """
    Perform reservoir sampling on the given stream to select k items.

    :param stream: An iterable representing the input stream.
    :param k: The number of items to sample.
    :param seed: The seed to use for random number generation. Only used if numpy is available. If -1 is passed, the seed
                    will be randomly generated.
    :return: A list containing k sampled items.
    """
    try:
        import numpy as np

        if seed == -1:
            seed = None
        rng = np.random.default_rng(seed)
        foo = rng.integers
    except ImportError:
        logger.warning(
            "Numpy not found, using python's random module. Seed will be ignored."
        )
        foo = random.randint

    # Initialize an empty reservoir
    reservoir = []

    # Fill the reservoir with the first k elements from the stream
    for i, item in enumerate(stream):
        if i < k:
            reservoir.append(item)
        else:
            # Randomly replace elements in the reservoir with decreasing probability
            j = foo(0, i)
            if j < k:
                reservoir[j] = item

    return reservoir
