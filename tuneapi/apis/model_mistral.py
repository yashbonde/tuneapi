"""
Connect to the `Mistral API <https://console.mistral.ai/>`_ and use their LLMs.
"""

# Copyright © 2024- Frello Technology Private Limited

import json
import requests
from typing import Optional, Dict, Any, Tuple, List

import tuneapi.utils as tu
import tuneapi.types as tt
from tuneapi.utils import ENV, SimplerTimes as stime, from_json, to_json
from tuneapi.types import Thread, human, Message


class Mistral(tt.ModelInterface):
    def __init__(
        self,
        id: Optional[str] = "mistral-small-latest",
        base_url: str = "https://api.mistral.ai/v1/chat/completions",
        extra_headers: Optional[Dict[str, str]] = None,
    ):
        self.model_id = id
        self.base_url = base_url
        self.api_token = ENV.MISTRAL_TOKEN("")
        self.extra_headers = extra_headers

    def set_api_token(self, token: str) -> None:
        self.api_token = token

    def _process_input(self, chats, token: Optional[str] = None):
        if not token and not self.api_token:  # type: ignore
            raise Exception(
                "Please set MISTRAL_TOKEN environment variable or pass through function"
            )
        token = token or self.api_token

        if isinstance(chats, tt.Thread):
            thread = chats
        elif isinstance(chats, str):
            thread = tt.Thread(tt.human(chats))
        else:
            raise Exception("Invalid input")

        prev_tool_id = tu.get_random_string(5)
        final_messages = []
        for i, m in enumerate(thread.chats):
            if m.role == tt.Message.SYSTEM:
                final_messages.append({"role": "system", "content": m.value})
            elif m.role == tt.Message.HUMAN:
                final_messages.append({"role": "user", "content": m.value})
            elif m.role == tt.Message.GPT:
                final_messages.append(
                    {
                        "role": "assistant",
                        "content": m.value.strip(),
                    }
                )
            elif m.role == tt.Message.FUNCTION_CALL:
                _m = tu.from_json(m.value) if isinstance(m.value, str) else m.value
                final_messages.append(
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "type": "function",
                                "id": prev_tool_id,
                                "function": {
                                    "name": _m["name"],
                                    "arguments": tu.to_json(_m["arguments"]),
                                },
                            }
                        ],
                    }
                )
            elif m.role == tt.Message.FUNCTION_RESP:
                # _m = tu.from_json(m.value) if isinstance(m.value, str) else m.value
                final_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": prev_tool_id,
                        "content": tu.to_json(m.value, tight=True),
                    }
                )
                prev_tool_id = tu.get_random_string(5)  # reset tool id
            else:
                raise Exception(f"Invalid message role: {m.role}")

        headers = {
            "Authorization": "Bearer " + token,
            "Content-Type": "application/json",
        }
        return headers, final_messages

    def chat(
        self,
        chats: Thread | str,
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 1,
        token: Optional[str] = None,
        timeout=(5, 30),
        extra_headers: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> str | Dict[str, Any]:
        output = ""
        for x in self.stream_chat(
            chats=chats,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            token=token,
            timeout=timeout,
            extra_headers=extra_headers,
            raw=False,
            **kwargs,
        ):
            if isinstance(x, dict):
                output = x
            else:
                output += x
        return output

    def stream_chat(
        self,
        chats: Thread | str,
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 1,
        token: Optional[str] = None,
        timeout=(5, 60),
        raw: bool = False,
        debug: bool = False,
        extra_headers: Optional[Dict[str, str]] = None,
    ):
        tools = []
        if isinstance(chats, Thread):
            tools = [{"type": "function", "function": x.to_dict()} for x in chats.tools]
        headers, messages = self._process_input(chats, token)
        extra_headers = extra_headers or self.extra_headers
        if extra_headers:
            headers.update(extra_headers)
        data = {
            "messages": messages,
            "model": model or self.model_id,
            "stream": True,
            "max_tokens": max_tokens,
            "tools": tools,
        }
        if temperature:
            data["temperature"] = temperature
        if debug:
            fp = "sample_mistral.json"
            print("Saving at path " + fp)
            tu.to_json(data, fp=fp)

        response = requests.post(
            self.base_url,
            headers=headers,
            json=data,
            stream=True,
            timeout=timeout,
        )
        try:
            response.raise_for_status()
        except Exception as e:
            yield response.text
            raise e

        fn_call = None
        for line in response.iter_lines():
            if raw:
                yield line
                continue

            line = line.decode().strip()
            if line:
                try:
                    x = json.loads(line.replace("data: ", ""))["choices"][0]["delta"]
                    if "tool_calls" in x:
                        y = x["tool_calls"][0]["function"]
                        if fn_call is None:
                            fn_call = {"name": y["name"], "arguments": y["arguments"]}
                        else:
                            fn_call["arguments"] += y["arguments"]
                    elif "content" in x:
                        c = x["content"]
                        if c:
                            yield c
                except:
                    break
        if fn_call:
            fn_call["arguments"] = from_json(fn_call["arguments"])
            yield fn_call
        return
