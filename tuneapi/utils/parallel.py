# Copyright © 2024- Frello Technology Private Limited
# REMEMBER: nothing from outside tune should be imported in utils

import time
from uuid import uuid4
from tqdm import trange
from typing import Any, Dict, List, Union, Tuple, Generator

from concurrent.futures import ThreadPoolExecutor, as_completed, Future


def threaded_map(
    fn,
    inputs: List[Tuple] | Generator,
    wait: bool = True,
    max_threads=20,
    post_fn=None,
    _name: str = "",
    safe: bool = False,
    pbar: bool = False,
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
        safe (bool, optional): If true, all exceptions are caught and returned with results. Defaults to False.
        pbar (bool, optional): If true, show a progress bar. Defaults to False.
    """
    _name = _name or str(uuid4())
    if isinstance(inputs, Generator):
        inputs = list(inputs)
    results = [None for _ in range(len(inputs))]
    errors = []
    _pbar = trange(len(inputs), desc="Processing", unit="input") if pbar else None
    with ThreadPoolExecutor(max_workers=max_threads, thread_name_prefix=_name) as exe:
        _fn = lambda i, x: [i, fn(*x)]
        futures = {exe.submit(_fn, i, x): i for i, x in enumerate(inputs)}
        if not wait:
            return futures
        for future in as_completed(futures):
            try:
                if _pbar:
                    _pbar.update(1)
                i, res = future.result()
                if post_fn is not None:
                    res = post_fn(res)
                results[i] = res
            except Exception as e:
                if safe:
                    errors.append(e)
                else:
                    raise e
    if safe:
        return results, errors
    return results


def batched(iterable, n, ol=0, expand: bool = False, last: bool = True):
    """Convert any ``iterable`` to a generator of batches of size ``n``, last one may be smaller.
    Python 3.12 has ``itertools.batched`` which does the same thing.

    Example:
        >>> for x in batched(range(10), 3):
        ...    print(x)
        [0, 1, 2]
        [3, 4, 5]
        [6, 7, 8]
        [9]

        >>> for x in batched(range(10), 3, ol = 1): # overlap = 1
        ...    print(x)
        [0, 1, 2]
        [2, 3, 4]
        [4, 5, 6]
        [6, 7, 8]
        [8, 9]

        >>> for x in batched(range(10), 3, last = False):
        ...    print(x)
        [0, 1, 2]
        [3, 4, 5]
        [6, 7, 8]

    Args:
        iterable (Iterable): The iterable to convert to batches
        n (int): The batch size
        ol (int): amount of overlap between batches
        expand (bool, optional): If true, each item in batch is tuple, eg. in numpy ``x[ ... , None]``
        last (bool, optional): If true, return the last batch even if it is smaller than n. Defaults to True.

    Yields:
        Iterator: The batched iterator
    """
    if ol == 0:
        done = False
        buffer = []
        _iter = iter(iterable)
        while not done:
            try:
                buffer.append(next(_iter))
                if len(buffer) == n:
                    if expand:
                        for x in buffer:
                            yield (x,)
                    else:
                        yield buffer
                    buffer = []
            except StopIteration:
                done = True
        if buffer and last:
            if expand:
                for x in buffer:
                    yield (x,)
            else:
                yield buffer
    elif ol >= n:
        raise ValueError("Overlap cannot be greater than or equal to the batch size")
    elif ol < 0 or n < 0:
        raise ValueError("Overlap and batch size cannot be negative")
    else:
        b = []
        for item in iterable:
            b.append(item)
            if len(b) == n:
                if expand:
                    for x in b:
                        yield (x,)
                else:
                    yield b
                b = b[n - ol :]  # Keep the overlap for the next batch

        # Yield any remaining items as the last batch
        if b and last:
            if expand:
                for x in b:
                    yield (x,)
            else:
                yield b
