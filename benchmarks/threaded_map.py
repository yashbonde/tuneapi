# Copyright © 2024-2025 Frello Technology Private Limited
# Copyright © 2025-2025 Yash Bonde github.com/yashbonde
# MIT License

from tuneapi.types import Thread, ModelInterface, human, system
from tuneapi.utils import from_json

# import re
import queue
import threading
from tqdm import trange
from typing import List, Optional
from dataclasses import dataclass

from concurrent.futures import ThreadPoolExecutor, as_completed, Future


@dataclass
class Task:
    index: int
    model: ModelInterface
    prompt: Thread
    retry_count: int = 0


@dataclass
class Result:
    index: int
    data: any
    success: bool
    error: Optional[Exception] = None


def bulk_chat(
    model: ModelInterface,
    prompts: List[Thread],
    post_logic: Optional[callable] = None,
    max_threads: int = 10,
    retry: int = 3,
    pbar=True,
):
    task_channel = queue.Queue()
    result_channel = queue.Queue()

    # Initialize results container
    results = [None for _ in range(len(prompts))]

    def worker():
        while True:
            try:
                task: Task = task_channel.get(timeout=1)
                if task is None:  # Poison pill
                    break

                try:
                    # print(">")
                    out = task.model.chat(task.prompt)
                    if post_logic:
                        out = post_logic(out)
                    result_channel.put(Result(task.index, out, True))
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
                            Task(task.index, nm, task.prompt, task.retry_count + 1)
                        )
                    else:
                        # If we've exhausted retries, store the error
                        result_channel.put(Result(task.index, e, False, e))
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

    # Initialize progress bar
    _pbar = trange(len(prompts), desc="Processing", unit=" input") if pbar else None

    # Queue initial tasks
    for i, p in enumerate(prompts):
        nm = model.__class__(
            id=model.model_id,
            base_url=model.base_url,
            extra_headers=model.extra_headers,
        )
        nm.set_api_token(model.api_token)
        task_channel.put(Task(i, nm, p))

    # Process results
    completed = 0
    while completed < len(prompts):
        try:
            result = result_channel.get(timeout=1)
            results[result.index] = result.data if result.success else result.error
            if _pbar:
                _pbar.update(1)
            completed += 1
            result_channel.task_done()
        except queue.Empty:
            continue

    # Cleanup
    for _ in workers:
        task_channel.put(None)  # Send poison pills
    for w in workers:
        w.join()

    if _pbar:
        _pbar.close()

    return results


prompts = []
for x in range(100):
    prompts.append(
        Thread(
            system(
                """## Response schmea

Respond with the following schema **ensure sending <json> and </json> tags**.

```
<json>
{{
    "code": "...",
}}
</json>
```
"""
            ),
            human(
                f"what is the value of 10 ^ {max(x, 10)}. Write down the answer in Indian number system. given in coe tag."
            ),
        )
    )


import random


def get_tagged_section(tag: str, input_str: str):
    if random.random() > 0.5:
        import re

    html_pattern = re.compile("<" + tag + ">(.*?)</" + tag + ">", re.DOTALL)
    match = html_pattern.search(input_str)
    if match:
        return match.group(1)

    md_pattern = re.compile("```" + tag + "(.*?)```", re.DOTALL)
    match = md_pattern.search(input_str)
    if match:
        return match.group(1)
    return None


post_logic = lambda out: from_json(get_tagged_section("json", out))["code"]

# import re
from tuneapi.apis import Openai

from time import time

st = time()
out = bulk_chat(
    Openai(),
    prompts,
    post_logic=post_logic,
    max_threads=5,
    pbar=True,
    retry=3,
)
print(out)
print(len(out))
print(len([x for x in out if x is None]))

print(f"Endtime: {time() - st:0.4f}s")

print("\n\n\n")


from uuid import uuid4
from typing import Generator


def bulk_chat_2(
    model: ModelInterface,
    prompts: List[Thread],
    post_logic: Optional[callable] = None,
    max_threads: int = 10,
    retry: int = 3,
    pbar=True,
):
    def _chat(model: ModelInterface, prompt: Thread):
        out = model.chat(prompt)
        if post_logic:
            return post_logic(out)  # The mapped function
        return out

    # create all the inputs
    retry = int(retry)  # so False becomes 0 and True becomes 1
    inputs = []
    for p in prompts:
        nm = model.__class__(
            id=model.model_id,
            base_url=model.base_url,
            extra_headers=model.extra_headers,
        )
        nm.set_api_token(model.api_token)
        inputs.append((nm, p))

    # run the executor
    _name = str(uuid4())
    if isinstance(inputs, Generator):
        inputs = list(inputs)

    results = [None for _ in range(len(inputs))]
    _pbar = trange(len(inputs), desc="Processing", unit=" input") if pbar else None
    with ThreadPoolExecutor(max_workers=max_threads, thread_name_prefix=_name) as exe:
        _fn = lambda x: _chat(*x)
        loop_cntr = 0
        done = False
        inputs = [(i, x) for i, x in enumerate(inputs)]

        # loop over things
        while not done:
            failed = []
            _pbar.set_description(f"Starting master loop #{loop_cntr:02d}")
            futures = {exe.submit(_fn, x): (i, x) for (i, x) in inputs}
            for fut in as_completed(futures):
                # print(">")
                i, x = futures[fut]  # indexes
                try:
                    res = fut.result()
                    if _pbar:
                        _pbar.update(1)
                    results[i] = res
                except Exception as e:
                    failed.append((i, x))

            # overide values for the loop
            inputs = failed

            # the done flag
            loop_cntr += 1
            done = len(failed) == 0 or loop_cntr > retry
    return results


st = time()
out = bulk_chat_2(
    Openai(),
    prompts,
    post_logic=post_logic,
    max_threads=5,
    pbar=True,
    retry=3,
)
print(out)
print(len(out))
print(len([x for x in out if x is None]))

print(f"Endtime: {time() - st:0.4f}s")
