# Parts used from https://github.com/hephex/asyncache/blob/master/asyncache/__init__.py
import asyncio
import functools
import inspect
from collections.abc import Callable
from typing import Any, NamedTuple

from cachetools import Cache, keys


class _CacheInfo(NamedTuple):
    """Cache information stores hits and misses."""

    hits: int
    misses: int


class NullContext:
    """NullContext acts as a proxy for the async cache lock."""

    async def __aenter__(self) -> "NullContext":
        """Return ``self`` upon entering the runtime context."""
        return self

    async def __aexit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """Raise any exception triggered within the runtime context."""
        return None


def acached(
    cache: Cache,
    ignore_args: list[str] = ["self"],
    store_none: bool = True,
) -> Any:
    """
    Normal `cached` decorator from `functools` does not work with async functions.
    This is a modified version that works with async functions.

    Usage
    -----

    >>> from cachetools import TTLCache
    >>> cache = TTLCache(maxsize=100, ttl=60)
    >>> @acached(cache)
    >>> async def foo(arg1, arg2):
    >>>     # ...
    >>>     return result

    If you have to use this with a descriptor (staticmethod, classmethod, property, etc) then you should
    keep acached innermost and the descriptor outermost like this:

    >>> @staticmethod
    >>> @acached(cache)
    >>> async def foo():

    This is because the staticmethod convert the underlying function into an object. The acached function
    does no know how to handle objects it only knows how to handle functions.

    You can check the cache info using the `cache_info` method.
    >>> print(foo.cache_info())
    ... CacheInfo(hits=0, misses=0, maxsize=100, currsize=0)

    You can clear the cache using the `cache_clear` method.
    >>> foo.cache_clear()

    If you pass `ignore_args=["self", "cls"]` then the cache key will not include arguments with those names.
    This works by filtering out arguments based on their parameter names.

    Example:
    >>> @acached(cache, ignore_args=["self"])
    >>> async def foo(self, arg1, arg2):
    >>>     return arg1 + arg2
    >>>
    >>> foo(1, 2)
    >>> foo(1, 2)
    >>> foo(1, 3)
    >>> foo.cache_info()
    ... CacheInfo(hits=2, misses=2, maxsize=100, currsize=2)

    """
    lock = NullContext()

    def decorator(func: Callable[..., Any]) -> Any:
        if not asyncio.iscoroutinefunction(func):
            raise ValueError("Function must be a coroutine")

        hits = misses = 0
        sig = inspect.signature(func)
        param_names = list(sig.parameters.keys())

        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            nonlocal hits, misses

            # Filter out ignored arguments from cache key
            k_args = []
            k_kwargs = {}

            # Process positional arguments
            for i, arg in enumerate(args):
                if i < len(param_names) and param_names[i] not in ignore_args:
                    k_args.append(arg)

            # Process keyword arguments
            for key, value in kwargs.items():
                if key not in ignore_args:
                    k_kwargs[key] = value

            k = keys.hashkey(*k_args, **k_kwargs)
            try:
                async with lock:
                    result = cache[k]
                    hits += 1
                    return result
            except KeyError:
                async with lock:
                    misses += 1

            val = await func(*args, **kwargs)  # run func

            if not store_none and val is None:
                return val

            try:
                async with lock:
                    cache[k] = val
            except ValueError:
                pass  # val too large

            return val

        def cache_clear() -> None:
            nonlocal hits, misses
            cache.clear()
            hits = misses = 0

        def cache_info() -> _CacheInfo:
            nonlocal hits, misses
            return _CacheInfo(hits, misses)

        wrapper_with_attrs = functools.wraps(func)(wrapper)
        wrapper_with_attrs.cache = cache
        wrapper_with_attrs.cache_key = keys.hashkey
        wrapper_with_attrs.cache_lock = lock
        wrapper_with_attrs.cache_clear = cache_clear
        wrapper_with_attrs.cache_info = cache_info

        return wrapper_with_attrs

    return decorator
