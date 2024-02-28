"""
Parallel processing
"""

# Copyright © 2024- Frello Technology Private Limited

import os
import re
import json
import time
import time
import random
import string
import logging
from uuid import uuid4
from urllib.parse import quote
from datetime import datetime, timezone
from typing import Any, Dict, List, Union, Tuple, Optional

from concurrent.futures import ThreadPoolExecutor, as_completed, Future


def threaded_map(
    fn,
    inputs: List[Tuple],
    wait: bool = True,
    max_threads=20,
    post_fn=None,
    _name: str = "",
    safe: bool = False,
) -> Union[Dict[Future, int], List[Any]]:
    """
    inputs is a list of tuples, each tuple is the input for single invocation of fn. order is preserved.

    Args:
        fn (function): The function to call
        inputs (List[Tuple[Any]]): All the inputs to the function, can be a generator
        wait (bool, optional): If true, wait for all the threads to finish, otherwise return a dict of futures. Defaults to True.
        max_threads (int, optional): The maximum number of threads to use. Defaults to 20.
        post_fn (function, optional): A function to call with the result. Defaults to None.
        _name (str, optional): The name of the thread pool. Defaults to "".
        safe (bool, optional): If true, all caughts exceptions are in the results. Defaults to False.
    """
    _name = _name or str(uuid4())
    results = [None for _ in range(len(inputs))]
    with ThreadPoolExecutor(max_workers=max_threads, thread_name_prefix=_name) as exe:
        _fn = lambda i, x: [i, fn(*x)]
        futures = {exe.submit(_fn, i, x): i for i, x in enumerate(inputs)}
        if not wait:
            return futures
        for future in as_completed(futures):
            try:
                i, res = future.result()
                if post_fn:
                    res = post_fn(res)
                results[i] = res
            except Exception as e:
                if safe:
                    results[i] = e
                else:
                    raise e
    return results


def batched(iterable, n):
    """Convert any ``iterable`` to a generator of batches of size ``n``, last one may be smaller.
    Python 3.12 has ``itertools.batched`` which does the same thing.

    Example:
        >>> for x in batched(range(10), 3):
        ...    print(x)
        [0, 1, 2]
        [3, 4, 5]
        [6, 7, 8]
        [9]

    Args:
        iterable (Iterable): The iterable to convert to batches
        n (int): The batch size

    Yields:
        Iterator: The batched iterator
    """
    done = False
    buffer = []
    _iter = iter(iterable)
    while not done:
        try:
            buffer.append(next(_iter))
            if len(buffer) == n:
                yield buffer
                buffer = []
        except StopIteration:
            done = True
    if buffer:
        yield buffer
