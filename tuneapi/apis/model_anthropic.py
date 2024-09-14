"""
Connect to the `Anthropic API <https://console.anthropic.com/>`_ to use Claude series of LLMs
"""

# Copyright © 2024- Frello Technology Private Limited

import re
import json
import requests
from typing import Optional, Dict, Any, Tuple, List

import tuneapi.utils as tu
import tuneapi.types as tt


class Anthropic(tt.ModelInterface):
    def __init__(
        self,
        id: Optional[str] = "claude-3-haiku-20240307",
        base_url: str = "https://api.anthropic.com/v1/messages",
        extra_headers: Optional[Dict[str, str]] = None,
    ):
        self.model_id = id
        self.base_url = base_url
        self.api_token = tu.ENV.ANTHROPIC_TOKEN("")
        self.extra_headers = extra_headers

    def set_api_token(self, token: str) -> None:
        self.api_token = token

    def _process_input(self, chats, token: Optional[str] = None):
        if not token and not self.api_token:  # type: ignore
            raise Exception(
                "Please set ANTHROPIC_TOKEN environment variable or pass through function"
            )
        token = token or self.api_token
        if isinstance(chats, tt.Thread):
            thread = chats
        elif isinstance(chats, str):
            thread = tt.Thread(tt.human(chats))
        else:
            raise Exception("Invalid input")

        # create the anthropic style data
        system = ""
        if thread.chats[0].role == tt.Message.SYSTEM:
            system = thread.chats[0].value

        claude_messages = []
        prev_tool_id = tu.get_random_string(5)
        for m in thread.chats[int(system != "") :]:
            if m.role == tt.Message.HUMAN:
                msg = {
                    "role": "user",
                    "content": [{"type": "text", "text": m.value.strip()}],
                }
                if m.images:
                    for i in m.images:
                        msg["content"].append(
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": i,
                                },
                            }
                        )
            elif m.role == tt.Message.GPT:
                msg = {
                    "role": "assistant",
                    "content": [{"type": "text", "text": m.value.strip()}],
                }
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

        headers = {
            "x-api-key": token,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
            "anthropic-beta": "tools-2024-05-16",
        }
        return headers, system.strip(), claude_messages

    def chat(
        self,
        chats: tt.Thread | str,
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: Optional[float] = None,
        token: Optional[str] = None,
        return_message: bool = False,
        extra_headers: Optional[Dict[str, str]] = None,
        **kwargs,
    ):
        output = ""
        fn_call = None
        for i in self.stream_chat(
            chats=chats,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            token=token,
            extra_headers=extra_headers,
            raw=False,
            **kwargs,
        ):
            if isinstance(i, dict):
                fn_call = i.copy()
            else:
                output += i
        if return_message:
            return output, fn_call
        if fn_call:
            return fn_call
        return output

    def stream_chat(
        self,
        chats: tt.Thread | str,
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: Optional[float] = None,
        token: Optional[str] = None,
        timeout=(5, 30),
        raw: bool = False,
        debug: bool = False,
        extra_headers: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> Any:

        tools = []
        if isinstance(chats, tt.Thread):
            tools = [x.to_dict() for x in chats.tools]
            for t in tools:
                t["input_schema"] = t.pop("parameters")
        headers, system, claude_messages = self._process_input(chats=chats, token=token)
        extra_headers = extra_headers or self.extra_headers
        if extra_headers:
            headers.update(extra_headers)

        data = {
            "model": model or self.model_id,
            "max_tokens": max_tokens,
            "messages": claude_messages,
            "system": system,
            "tools": tools,
            "stream": True,
        }
        if temperature:
            data["temperature"] = temperature
        if kwargs:
            data.update(kwargs)

        if debug:
            fp = "sample_anthropic.json"
            print("Saving at path " + fp)
            tu.to_json(data, fp=fp)

        r = requests.post(
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

        fn_call = None
        for line in r.iter_lines():
            line = line.decode().strip()
            if not line or not "data:" in line:
                continue

            try:
                # print(line)
                resp = json.loads(line.replace("data:", "").strip())
                if resp["type"] == "content_block_start":
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
                        fn_call["arguments"] = json.loads(fn_call["arguments"] or "{}")
                        yield fn_call
                        fn_call = None
            except:
                break
        return


# helper methods


def get_section(tag: str, out: str) -> Optional[str]:
    pattern = re.compile("<" + tag + ">(.*?)</" + tag + ">", re.DOTALL)
    match = pattern.search(out)
    if match:
        content = match.group(1)
        return content
    return None
