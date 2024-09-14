"""
Connect to the `TuneAI Proxy API <https://studio.tune.app/>`_ and use our standard endpoint for AI.
"""

# Copyright © 2024- Frello Technology Private Limited

import json
import requests
from typing import Optional, Dict, Any, List

import tuneapi.utils as tu
import tuneapi.types as tt


class TuneModel(tt.ModelInterface):
    """Defines the model used in tune.app. See [Tune Studio](https://studio.tune.app/) for more information."""

    def __init__(
        self,
        id: Optional[str] = None,
        base_url: str = "https://proxy.tune.app/chat/completions",
        api_token: Optional[str] = None,
        org_id: Optional[str] = None,
        extra_headers: Optional[Dict[str, str]] = None,
    ):
        self.model_id = id or tu.ENV.TUNEAPI_MODEL("")
        self.base_url = base_url
        self.api_token = api_token or tu.ENV.TUNEAPI_TOKEN("")
        self.org_id = org_id or tu.ENV.TUNEORG_ID("")
        self.extra_headers = extra_headers

    def __repr__(self) -> str:
        out = f"<TuneModel: {self.model_id}"
        if self.org_id:
            out += f" | {self.org_id}"
        out += ">"
        return out

    def set_api_token(self, token: str) -> None:
        self.api_token = token

    def set_org_id(self, org_id: str) -> None:
        self.org_id = org_id

    def _process_input(self, chats, token: Optional[str] = None):
        if not token and not self.api_token:  # type: ignore
            raise Exception(
                "Tune API key not found. Please set TUNEAPI_TOKEN environment variable or pass through function"
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
            "Authorization": token,
            "Content-Type": "application/json",
        }
        if self.org_id:
            headers["X-Org-Id"] = self.org_id
        return headers, final_messages

    def chat(
        self,
        chats: tt.Thread | str,
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        token: Optional[str] = None,
        timeout=(5, 60),
        stop: Optional[List[str]] = None,
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
            stop=stop,
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
        chats: tt.Thread | str,
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        token: Optional[str] = None,
        timeout=(5, 60),
        stop: Optional[List[str]] = None,
        raw: bool = False,
        debug: bool = False,
        extra_headers: Optional[Dict[str, str]] = None,
    ):
        model = model or self.model_id
        if not model:
            raise Exception(
                "Tune Model ID not found. Please set TUNEAPI_MODEL environment variable or pass through function"
            )
        headers, messages = self._process_input(chats, token)
        extra_headers = extra_headers or self.extra_headers
        if extra_headers:
            headers.update(extra_headers)
        data = {
            "temperature": temperature,
            "messages": messages,
            "model": model,
            "stream": True,
            "max_tokens": max_tokens,
        }
        if stop:
            data["stop"] = stop
        if isinstance(chats, tt.Thread) and len(chats.tools):
            data["tools"] = [
                {"type": "function", "function": x.to_dict()} for x in chats.tools
            ]
        if debug:
            fp = "sample_tune.json"
            tu.logger.info("Saving at path " + fp)
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
            tu.logger.error(response.text)
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
                    delta = json.loads(line.replace("data: ", ""))["choices"][0][
                        "delta"
                    ]
                    if "tool_calls" in delta:
                        y = delta["tool_calls"][0]["function"]
                        if fn_call is None:
                            fn_call = {
                                "name": y["name"],
                                "arguments": y.get("arguments", ""),
                            }
                        else:
                            fn_call["arguments"] += y["arguments"]
                    elif "content" in delta:
                        yield delta["content"]
                except:
                    break
        if fn_call:
            fn_call["arguments"] = tu.from_json(fn_call["arguments"])
            yield fn_call
        return
