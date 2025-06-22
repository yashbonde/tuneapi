"""
Connect to the `Anthropic API <https://console.anthropic.com/>`_ to use Claude series of LLMs
"""

# Copyright © 2024-2025 Frello Technology Private Limited
# Copyright © 2025-2025 Yash Bonde github.com/yashbonde
# MIT License

from copy import deepcopy
from typing import Optional, Dict, Any, List, Tuple

import tuneapi.utils as tu
import tuneapi.types as tt
from tuneapi.apis.turbo import distributed_chat, distributed_chat_async


class Anthropic(tt.ModelInterface):
    def __init__(
        self,
        id: Optional[str] = "claude-3-haiku-20240307",
        base_url: str = "https://api.anthropic.com/v1/messages",
        api_token: Optional[str] = None,
        extra_headers: Optional[Dict[str, str]] = None,
    ):
        super().__init__()

        self.model_id = id
        self.base_url = base_url
        self.batch_url = base_url + "/batches"
        self.api_token = api_token or tu.ENV.ANTHROPIC_TOKEN("")
        self.extra_headers = extra_headers
        self.client = None

    def set_api_token(self, token: str) -> None:
        self.api_token = token

    def _process_header(self, token: str) -> Dict[str, str]:
        if not token and not self.api_token:  # type: ignore
            raise Exception(
                "Please set ANTHROPIC_TOKEN environment variable or pass through function"
            )
        token = token or self.api_token
        return {
            "x-api-key": token,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
            "anthropic-beta": "tools-2024-05-16",
        }

    def _process_thread(self, thread: tt.Thread) -> Tuple[str, List[Dict[str, Any]]]:
        # create the anthropic style data
        system = ""
        if thread.chats[0].role == tt.Message.SYSTEM:
            system = thread.chats[0].value

        claude_messages = []
        prev_tool_id = tu.get_random_string(5)
        for m in thread.chats[int(system != "") :]:
            if m.role == tt.Message.HUMAN:
                if isinstance(m.value, str):
                    content = [{"type": "text", "text": m.value}]
                elif isinstance(m.value, list):
                    content = deepcopy(m.value)
                else:
                    raise Exception(
                        f"Unknown message type. Got: '{type(m.value)}', expected 'List[Dict[str, Any]]' or 'str'"
                    )
                if m.images:
                    for i in m.images:
                        content.append(
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": i,
                                },
                            }
                        )
                msg = {"role": "user", "content": content}
            elif m.role == tt.Message.GPT:
                if isinstance(m.value, str):
                    content = [{"type": "text", "text": m.value}]
                elif isinstance(m.value, list):
                    content = deepcopy(m.value)
                else:
                    raise Exception(
                        f"Unknown message type. Got: '{type(m.value)}', expected 'List[Dict[str, Any]]' or 'str'"
                    )
                if m.images:
                    for i in m.images:
                        content.append(
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": i,
                                },
                            }
                        )
                msg = {"role": "assistant", "content": content}
            elif m.role == tt.Message.FUNCTION_CALL:
                _m = tu.from_json(m.value) if isinstance(m.value, str) else m.value
                msg = {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": prev_tool_id,
                            "name": _m["name"],
                            "input": _m["arguments"],
                        }
                    ],
                }
            elif m.role == tt.Message.FUNCTION_RESP:
                # _m = tu.from_json(m.value) if isinstance(m.value, str) else m.value
                msg = {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": prev_tool_id,
                            "content": tu.to_json(m.value, tight=True),
                        }
                    ],
                }
            else:
                raise Exception(f"Unknown role: {m.role}")
            claude_messages.append(msg)

        # if the last messag eis
        if thread.schema:
            assert (
                claude_messages[-1]["role"] == thread.chats[-1].HUMAN
            ), f"Last message should be from {thread.chats[-1].HUMAN} to generate structured output with Anthropic model."
            resp_schema = thread.schema.model_json_schema()
            resp_schema["additionalProperties"] = False
            for _, defs in resp_schema.get("$defs", dict()).items():
                defs["additionalProperties"] = False
            claude_messages[-1]["content"].append(
                {
                    "type": "text",
                    "text": (
                        "You are given this JSON schema. You will generate only the output filled in this schema and "
                        "nothing else. No thinking or extra tokens. Only the JSON. It is crucial you do this because "
                        "the response text will be parsed from JSON. If it fails, everything has gone to waste. Here "
                        "is the schema:\n\n" + tu.to_json(resp_schema)
                    ),
                }
            )

        return system, claude_messages

    def _process_input(
        self,
        chats: tt.Thread | str,
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: Optional[float] = None,
        token: Optional[str] = None,
        debug: bool = False,
        extra_headers: Optional[Dict[str, str]] = None,
        stream: bool = True,
        **kwargs,
    ):
        if isinstance(chats, tt.Thread):
            thread = chats
        elif isinstance(chats, str):
            thread = tt.Thread(tt.human(chats))
        else:
            raise Exception("Invalid input")

        system, claude_messages = self._process_thread(thread)
        headers = self._process_header(token)

        tools = []
        if isinstance(chats, tt.Thread) and chats.tools:
            tools = [x.to_dict() for x in chats.tools]
            for t in tools:
                t["input_schema"] = t.pop("parameters")
        extra_headers = extra_headers or self.extra_headers
        if extra_headers:
            headers.update(extra_headers)

        data = {
            "model": model or self.model_id,
            "max_tokens": max_tokens,
            "messages": claude_messages,
            "system": system,
            "tools": tools,
            "stream": stream,
        }
        if temperature:
            data["temperature"] = temperature
        if kwargs:
            data.update(kwargs)

        if debug:
            fp = "sample_anthropic.json"
            print("Saving at path " + fp)
            tu.to_json(data, fp=fp)

        return headers, data

    def _process_output(self, raw: bool, lines_fn: callable, yield_usage: bool):
        fn_call = None
        usage_dict = {}
        for line in lines_fn():
            if isinstance(line, bytes):
                line = line.decode().strip()
            if not line or not "data:" in line:
                continue

            resp = tu.from_json(line.replace("data:", "").strip())
            if resp["type"] == "message_start":
                usage = resp["message"]["usage"]
                usage_dict.update(usage)
            elif resp["type"] == "content_block_start":
                if resp["content_block"]["type"] == "tool_use":
                    fn_call = {
                        "name": resp["content_block"]["name"],
                        "arguments": "",
                    }
            elif resp["type"] == "content_block_delta":
                delta = resp["delta"]
                delta_type = delta["type"]
                if delta_type == "text_delta":
                    if raw:
                        yield b"data: " + tu.to_json(
                            {
                                "object": delta_type,
                                "choices": [{"delta": {"content": delta["text"]}}],
                            },
                            tight=True,
                        ).encode()
                        yield b""  # uncomment this line if you want 1:1 with OpenAI
                    else:
                        yield delta["text"]
                elif delta_type == "input_json_delta":
                    fn_call["arguments"] += delta["partial_json"]
            elif resp["type"] == "content_block_stop":
                if fn_call:
                    fn_call["arguments"] = tu.from_json(fn_call["arguments"] or "{}")
                    yield fn_call
                    fn_call = None
            elif resp["type"] == "message_delta":
                usage_dict["output_tokens"] += resp["usage"]["output_tokens"]
                cached_tokens = usage_dict.get(
                    "cache_read_input_tokens", 0
                ) or usage_dict.get("cache_creation_input_tokens", 0)
                usage_obj = tt.Usage(
                    input_tokens=usage_dict.pop("input_tokens"),
                    output_tokens=usage_dict.pop("output_tokens"),
                    cached_tokens=cached_tokens,
                    **usage_dict,
                )
        if yield_usage:
            if raw:
                yield b"data: " + usage_obj.to_json(tight=True).encode()
                yield b""  # uncomment this line if you want 1:1 with OpenAI
            else:
                yield usage_obj

    # Interaction methods

    def chat(
        self,
        chats: tt.Thread | str,
        model: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: Optional[float] = None,
        token: Optional[str] = None,
        usage: bool = False,
        extra_headers: Optional[Dict[str, str]] = None,
        **kwargs,
    ):
        output = ""
        usage_obj = None
        fn_call = None
        for i in self.stream_chat(
            chats=chats,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            token=token,
            usage=usage,
            extra_headers=extra_headers,
            raw=False,
            **kwargs,
        ):
            if isinstance(i, dict):
                fn_call = i.copy()
            elif isinstance(i, tt.Usage):
                usage_obj = i
            else:
                output += i
        if fn_call:
            output = fn_call

        if isinstance(chats, tt.Thread) and chats.schema:
            output = chats.schema(**tu.from_json(output))

        if usage:
            return output, usage_obj
        return output

    def stream_chat(
        self,
        chats: tt.Thread | str,
        model: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: Optional[float] = None,
        token: Optional[str] = None,
        debug: bool = False,
        usage: bool = False,
        extra_headers: Optional[Dict[str, str]] = None,
        timeout=(5, 30),
        raw: bool = False,
        **kwargs,
    ) -> Any:

        headers, data = self._process_input(
            chats=chats,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            token=token,
            debug=debug,
            extra_headers=extra_headers,
            **kwargs,
        )

        if self.client is None:
            self.set_client()

        r = self.client.post(
            self.base_url,
            headers=headers,
            json=data,
            timeout=timeout,
        )
        try:
            r.raise_for_status()
        except Exception as e:
            yield r.text
            raise e

        yield from self._process_output(
            raw=raw,
            lines_fn=r.iter_lines,
            yield_usage=usage,
        )

    async def chat_async(
        self,
        chats: tt.Thread | str,
        model: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: Optional[float] = None,
        token: Optional[str] = None,
        usage: bool = False,
        extra_headers: Optional[Dict[str, str]] = None,
        **kwargs,
    ):
        output = ""
        usage_obj = None
        fn_call = None
        async for i in self.stream_chat_async(
            chats=chats,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            token=token,
            usage=usage,
            extra_headers=extra_headers,
            raw=False,
            **kwargs,
        ):
            if isinstance(i, dict):
                fn_call = i.copy()
            elif isinstance(i, tt.Usage):
                usage_obj = i
            else:
                output += i
        if fn_call:
            output = fn_call

        if isinstance(chats, tt.Thread) and chats.schema:
            output = chats.schema(**tu.from_json(output))

        if usage:
            return output, usage_obj
        return output

    async def stream_chat_async(
        self,
        chats: tt.Thread | str,
        model: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: Optional[float] = None,
        token: Optional[str] = None,
        debug: bool = False,
        usage: bool = False,
        extra_headers: Optional[Dict[str, str]] = None,
        timeout=(5, 30),
        raw: bool = False,
        **kwargs,
    ) -> Any:

        headers, data = self._process_input(
            chats=chats,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            token=token,
            debug=debug,
            extra_headers=extra_headers,
            **kwargs,
        )

        if self.async_client is None:
            self.set_async_client()

        response = await self.async_client.post(
            self.base_url,
            headers=headers,
            json=data,
            timeout=timeout,
        )
        try:
            response.raise_for_status()
        except Exception as e:
            yield str(e)
            return

        async for chunk in response.aiter_bytes():
            for x in self._process_output(
                raw=raw,
                lines_fn=chunk.decode("utf-8").splitlines,
                yield_usage=usage,
            ):
                yield x

    def distributed_chat(
        self,
        prompts: List[tt.Thread],
        post_logic: Optional[callable] = None,
        max_threads: int = 10,
        retry: int = 3,
        pbar: bool = True,
        debug: bool = False,
        time_metrics: bool = False,
        **kwargs,
    ):
        return distributed_chat(
            self,
            prompts=prompts,
            post_logic=post_logic,
            max_threads=max_threads,
            retry=retry,
            pbar=pbar,
            debug=debug,
            time_metrics=time_metrics,
            **kwargs,
        )

    async def distributed_chat_async(
        self,
        prompts: List[tt.Thread],
        post_logic: Optional[callable] = None,
        max_threads: int = 10,
        retry: int = 3,
        pbar: bool = True,
        debug: bool = False,
        time_metrics: bool = False,
        **kwargs,
    ):
        return await distributed_chat_async(
            self,
            prompts=prompts,
            post_logic=post_logic,
            max_threads=max_threads,
            retry=retry,
            pbar=pbar,
            debug=debug,
            time_metrics=time_metrics,
            **kwargs,
        )

    def submit_batch(
        self,
        threads: List[tt.Thread | str],
        model: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: Optional[float] = None,
        token: Optional[str] = None,
        debug: bool = False,
        extra_headers: Optional[Dict[str, str]] = None,
        timeout=(5, 30),
        raw: bool = False,
        **kwargs,
    ) -> Tuple[str, List[str]] | Dict:
        bodies = []
        custom_ids = []
        for chats in threads:
            headers, data = self._process_input(
                chats=chats,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                token=token,
                extra_headers=extra_headers,
                stream=False,
                **kwargs,
            )
            custom_id = "tuneapi_" + tu.get_random_string(10)
            custom_ids.append(custom_id)
            bodies.append({"custom_id": custom_id, "params": data})
        body = {"requests": bodies}
        if debug:
            fp = "sample_anthropic_batch.json"
            print("Saving at path " + fp)
            tu.to_json(body, fp=fp)

        if self.client is None:
            self.set_client()

        r = self.client.post(
            url=self.batch_url,
            headers=headers,
            timeout=timeout,
            json=body,
        )
        try:
            r.raise_for_status()
        except Exception as e:
            tu.logger.error(f"Coudn't submit batch: {r.text}")
            raise e
        resp = r.json()

        if raw:
            return resp
        return resp["id"], custom_ids

    def get_batch(
        self,
        batch_id: str,
        custom_ids: Optional[List[str]] = None,
        usage: bool = False,
        token: Optional[str] = None,
        raw: bool = False,
        verbose: bool = False,
    ) -> Tuple[List[Any] | Dict, str | None]:
        headers = self._process_header(token)

        if self.client is None:
            self.set_client()

        r = self.client.get(self.batch_url + "/" + batch_id, headers=headers)
        try:
            r.raise_for_status()
        except Exception as e:
            tu.logger.error(f"Coudn't get batch: {r.text}")
            raise e
        resp = r.json()
        if resp["processing_status"] != "ended":
            if verbose:
                tu.logger.info(
                    f"Batch {batch_id} has not ended. Status: {resp['processing_status']}"
                )
            return None, resp["processing_status"]
        results_url = resp["results_url"]

        # fetch the results, response is a JSONL, fucntion return shoudl be a List of JSONs
        r = self.client.get(results_url, headers=headers)
        try:
            r.raise_for_status()
        except Exception as e:
            tu.logger.error(f"Coudn't get batch results: {r.text}")
            raise e

        output = []
        for line in r.iter_lines():
            if not line:
                continue
            output.append(tu.from_json(line))

        if custom_ids:
            # each item in output has a key called "custom_id" sort on the basis of incoming custom_ids
            output = sorted(output, key=lambda x: custom_ids.index(x["custom_id"]))

        if raw:
            return output, None

        _usage = tt.Usage(0, 0)
        for o in output:
            u = o["result"]["message"]["usage"]
            _usage += tt.Usage(
                input_tokens=u.pop("input_tokens"),
                output_tokens=u.pop("output_tokens"),
                cached_tokens=u.get("cache_read_input_tokens", 0)
                or u.get("cache_creation_input_tokens", 0),
                **u,
            )

        parsed_output = [o["result"]["message"]["content"][0] for o in output]
        final_output = []
        for o in parsed_output:
            if o["type"] == "text":
                final_output.append(o["text"])
            elif o["type"] == "tool_use":
                final_output.append(
                    {
                        "name": o["name"],
                        "arguments": o["input"],
                    }
                )
            else:
                raise ValueError(f"Unknown message content: {o['type']}")

        if usage:
            return final_output, None, _usage
        return final_output, None
