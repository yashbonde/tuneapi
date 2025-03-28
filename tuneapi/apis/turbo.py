# Copyright © 2024-2025 Frello Technology Private Limited
# Copyright © 2025-2025 Yash Bonde github.com/yashbonde
# MIT License

import queue
import asyncio
import threading
from tqdm import trange
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass

from tuneapi.types import Thread, ModelInterface, Usage
from tuneapi.utils import logger, SimplerTimes


def distributed_chat(
    model: ModelInterface,
    prompts: List[Thread],
    post_logic: Optional[callable] = None,
    max_threads: int = 10,
    retry: int = 3,
    pbar=True,
    debug=False,
    usage: bool = False,
    time_metrics: bool = False,
    **kwargs,
) -> List | Tuple[List, Usage]:
    """
    Distributes multiple chat prompts across a thread pool for parallel processing.

    This function creates a pool of worker threads to process multiple chat prompts concurrently. It handles retry
    logic for failed requests and maintains the order of responses corresponding to the input prompts.

    Args:
        model (ModelInterface): The base model instance to clone for each worker thread. Each thread gets its own model
            instance to ensure thread safety.

        prompts (List[Thread]): A list of chat prompts to process. The order of responses will match the order of these
            prompts.

        post_logic (Optional[callable], default=None): A function to process each chat response before storing. If None,
            raw responses are stored. Function signature should be: f(chat_response) -> processed_response

        max_threads (int, default=10): Maximum number of concurrent worker threads. Adjust based on API rate limits and
            system capabilities.

        retry (int, default=3): Number of retry attempts for failed requests. Set to 0 to disable retries.

        pbar (bool, default=True): Whether to display a progress bar.

        debug (bool, default=False): Whether to log debug information.

        usage (bool, default=False): Whether to return usage statistics. If True, the function will return a tuple of
            (responses, usage) where usage is an instance of Usage.

        time_metrics (bool, default=False): Whether to return time metrics. If True, the function will return a tuple of
            (responses, time_metrics) where time_metrics is a list of time taken for each prompt.

    Returns:
        List[Any]: A list of responses or errors, maintaining the same order as input prompts.
            Successful responses will be either raw or processed (if post_logic provided).
            Failed requests (after retries) will contain the last error encountered.

    Raises:
        ValueError: If max_threads < 1 or retry < 0
        TypeError: If model is not an instance of ModelInterface

    Example:
        >>> from tuneapi import ta, tt
        >>> model = ta.Gemini()
        >>> prompts = [
        ...     tt.Thread([tt.human("What is 2+2?")]),
        ...     tt.Thread([tt.human("What is Python?")])
        ... ]
        >>> responses = distributed_chat(model, prompts, max_threads=5)
        >>> for prompt, response in zip(prompts, responses):
        ...     print(f"Q: {prompt}\nA: {response}\n")

    Note:
        - Each worker thread gets its own model instance to prevent sharing state
        - Progress bar shows both initial processing and retries
        - The function maintains thread safety through message passing channels
    """
    task_channel = queue.Queue()
    result_channel = queue.Queue()

    # Initialize results container
    results = [None for _ in range(len(prompts))]

    def worker():
        while True:
            try:
                task: _Task = task_channel.get(timeout=1)
                if task is None:  # Poison pill
                    break

                st = SimplerTimes.get_now_fp64()
                try:
                    out = task.model.chat(chats=task.prompt, **task.kwargs)
                    if usage:
                        out, _usage = out
                    if post_logic:
                        out = post_logic(out)
                    if not usage:
                        result_channel.put(
                            _Result(
                                index=task.index,
                                data=out,
                                success=True,
                                time_elapsed=SimplerTimes.get_now_fp64() - st,
                            )
                        )
                    else:
                        result_channel.put(
                            _Result(
                                index=task.index,
                                data=(out, _usage),
                                success=True,
                                time_elapsed=SimplerTimes.get_now_fp64() - st,
                            )
                        )
                except Exception as e:
                    if task.retry_count < retry:
                        # Create new model instance for retry
                        nm = model.__class__(
                            id=model.model_id,
                            base_url=model.base_url,
                            extra_headers=model.extra_headers,
                        )
                        nm.set_api_token(model.api_token)
                        # Increment retry count and requeue
                        task_channel.put(
                            _Task(
                                index=task.index,
                                model=nm,
                                prompt=task.prompt,
                                retry_count=task.retry_count + 1,
                                kwargs=task.kwargs,
                            )
                        )
                    else:
                        # If we've exhausted retries, store the error
                        result_channel.put(
                            _Result(
                                index=task.index,
                                data=None,
                                success=False,
                                error=e,
                                time_elapsed=SimplerTimes.get_now_fp64() - st,
                            )
                        )
                finally:
                    task_channel.task_done()
            except queue.Empty:
                continue

    # Create and start worker threads
    workers = []
    for _ in range(max_threads):
        t = threading.Thread(target=worker)
        t.start()
        workers.append(t)

    if debug:
        logger.info(f"Processing {len(prompts)} prompts with {max_threads} workers")

    # Initialize progress bar
    _pbar = trange(len(prompts), unit="Thread") if pbar else None

    # Queue initial tasks
    kwargs.update({"usage": usage})
    for i, p in enumerate(prompts):
        nm = model.__class__(
            id=model.model_id,
            base_url=model.base_url,
            extra_headers=model.extra_headers,
        )
        nm.set_api_token(model.api_token)
        task_channel.put(
            _Task(
                index=i,
                model=nm,
                prompt=p,
                retry_count=0,
                kwargs=kwargs,
            )
        )

    # Process results
    completed = 0
    all_usage: List[Usage] = []
    time_taken: List[float] = []
    while completed < len(prompts):
        try:
            result: _Result = result_channel.get(timeout=1)
            if result.success:
                if usage:
                    results[result.index], _usage = result.data
                    all_usage.append(_usage)
                else:
                    results[result.index] = result.data
            else:
                results[result.index] = result.error
            if _pbar:
                _pbar.update(1)
            completed += 1
            result_channel.task_done()
            time_taken.append(result.time_elapsed)
        except queue.Empty:
            continue

    # Cleanup
    for _ in workers:
        task_channel.put(None)  # Send poison pills
    for w in workers:
        w.join()

    if _pbar:
        _pbar.close()

    return_items = [results]
    if usage:
        _usage = Usage(
            input_tokens=sum([u.input_tokens for u in all_usage]),
            output_tokens=sum([u.output_tokens for u in all_usage]),
            cached_tokens=sum([u.cached_tokens for u in all_usage]),
            all_usage=all_usage,
        )
        return_items.append(_usage)
    if time_metrics:
        return_items.append(time_taken)

    if len(return_items) > 1:
        return tuple(return_items)
    return return_items[0]


async def distributed_chat_async(
    model: ModelInterface,
    prompts: List[Thread],
    post_logic: Optional[callable] = None,
    max_threads: int = 10,
    retry: int = 3,
    pbar=True,
    debug=False,
    usage: bool = False,
    time_metrics: bool = False,
    **kwargs,
):
    _pbar = trange(len(prompts), unit="Thread") if pbar else None
    results = [None for _ in range(len(prompts))]
    kwargs.update({"usage": usage})

    async def process_prompt(index, prompt, retry_count=0) -> _Result:
        st = SimplerTimes.get_now_fp64()
        try:
            out = await model.chat_async(chats=prompt, **kwargs)
            if usage:
                out, _usage = out
            if post_logic:
                out = post_logic(out)
            _pbar.update(1)
            if not usage:
                return _Result(
                    index=index,
                    data=out,
                    success=True,
                    time_elapsed=SimplerTimes.get_now_fp64() - st,
                )
            else:
                return _Result(
                    index=index,
                    data=(out, _usage),
                    success=True,
                    time_elapsed=SimplerTimes.get_now_fp64() - st,
                )
        except Exception as e:
            if retry_count < retry:
                return await process_prompt(index, prompt, retry_count + 1)
            else:
                return _Result(
                    index=index,
                    data=None,
                    success=False,
                    error=e,
                    time_elapsed=SimplerTimes.get_now_fp64() - st,
                )

    # Run all tasks concurrently using asyncio.gather
    tasks = []
    for i, prompt in enumerate(prompts):
        tasks.append(process_prompt(i, prompt))

    if debug:
        logger.info(f"Processing {len(prompts)} prompts with {max_threads} workers")

    results_from_gather: List[_Result] = await asyncio.gather(*tasks)

    # Process results
    all_usage: List[Usage] = []
    time_taken: List[float] = []
    for r in results_from_gather:
        if r.success:
            if usage:
                results[r.index], _usage = r.data
                all_usage.append(_usage)
            else:
                results[r.index] = r.data
        else:
            results[r.index] = r.error[0] if r.error else None

        # whether success or failure, record time taken
        time_taken.append(r.time_elapsed)

    if _pbar:
        _pbar.close()

    return_items = [results]
    if usage:
        _usage = Usage(
            input_tokens=sum([u.input_tokens for u in all_usage]),
            output_tokens=sum([u.output_tokens for u in all_usage]),
            cached_tokens=sum([u.cached_tokens for u in all_usage]),
            all_usage=all_usage,
        )
        return_items.append(_usage)
    if time_metrics:
        return_items.append(time_taken)

    if len(return_items) > 1:
        return tuple(return_items)
    return return_items[0]


# helpers


@dataclass
class _Task:
    """Individual Task"""

    index: int
    model: ModelInterface
    prompt: Thread
    retry_count: int = 0
    kwargs: Optional[Dict] = None


@dataclass
class _Result:
    """The Result object"""

    index: int
    data: any
    success: bool
    error: Optional[Exception] = None
    time_elapsed: Optional[float] = None
