"""
Connect to the `OpenAI API <https://playground.openai.com/>`_ and use their LLMs.
"""

# Copyright © 2024- Frello Technology Private Limited

import json
import requests

from typing import Optional, Any, List, Dict

import tuneapi.utils as tu
import tuneapi.types as tt


class Openai(tt.ModelInterface):
    def __init__(
        self,
        id: Optional[str] = "gpt-4o",
        base_url: str = "https://api.openai.com/v1/chat/completions",
        extra_headers: Optional[Dict[str, str]] = None,
    ):
        self.model_id = id
        self.base_url = base_url
        self.api_token = tu.ENV.OPENAI_TOKEN("")
        self.extra_headers = extra_headers

    def set_api_token(self, token: str) -> None:
        self.api_token = token

    def _process_input(self, chats, token: Optional[str] = None):
        if not token and not self.api_token:  # type: ignore
            raise Exception(
                "OpenAI API key not found. Please set OPENAI_TOKEN environment variable or pass through function"
            )
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

        headers = self._process_header(token)
        return headers, final_messages

    def _process_header(self, token: Optional[str] = None):
        return {
            "Authorization": "Bearer " + (token or self.api_token),
            "Content-Type": "application/json",
        }

    def chat(
        self,
        chats: tt.Thread | str,
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 1,
        parallel_tool_calls: bool = False,
        token: Optional[str] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> Any:
        output = ""
        for x in self.stream_chat(
            chats=chats,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            parallel_tool_calls=parallel_tool_calls,
            token=token,
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
        temperature: float = 1,
        parallel_tool_calls: bool = False,
        token: Optional[str] = None,
        timeout=(5, 60),
        extra_headers: Optional[Dict[str, str]] = None,
        debug: bool = False,
        raw: bool = False,
    ):
        headers, messages = self._process_input(chats, token)
        extra_headers = extra_headers or self.extra_headers
        if extra_headers:
            headers.update(extra_headers)
        data = {
            "temperature": temperature,
            "messages": messages,
            "model": model or self.model_id,
            "stream": True,
            "max_tokens": max_tokens,
            "parallel_tool_calls": parallel_tool_calls,
        }
        if isinstance(chats, tt.Thread) and len(chats.tools):
            data["tools"] = [
                {"type": "function", "function": x.to_dict()} for x in chats.tools
            ]
        if debug:
            fp = "sample_oai.json"
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
                    if "tool_calls" not in x:
                        yield x["content"]
                    else:
                        y = x["tool_calls"][0]["function"]
                        if fn_call is None:
                            fn_call = {"name": y["name"], "arguments": y["arguments"]}
                        else:
                            fn_call["arguments"] += y["arguments"]
                except:
                    break
        if fn_call:
            fn_call["arguments"] = tu.from_json(fn_call["arguments"])
            yield fn_call
        return

    def embedding(
        self,
        chats: tt.Thread | List[str] | str,
        cum: bool = False,
        model: str = "text-embedding-3-small",
        token: Optional[str] = None,
        timeout=(5, 60),
        raw: bool = False,
        extra_headers: Optional[Dict[str, str]] = None,
    ):
        """If you pass a list then returned items are in the insertion order"""
        text = []

        headers = self._process_header(token)
        extra_headers = extra_headers or self.extra_headers
        if extra_headers:
            headers.update(extra_headers)
        if isinstance(chats, tt.Thread):
            _, messages = self._process_input(chats, token)
            for i, m in enumerate(messages):
                x = f"<{m['role']}> : {m['content']}\n\n"
                text.append(x)

            if cum:
                text = ["".join(text)]
        elif isinstance(chats, tt.Message):
            text = [chats.value]
        elif isinstance(chats, list) and len(chats) and isinstance(chats[0], str):
            # this is an exception
            text = chats
        elif isinstance(chats, str):
            text = [chats]
        else:
            raise ValueError(f"Invalid input type. Got {type(chats)}")

        r = requests.post(
            "https://api.openai.com/v1/embeddings",
            json={
                "input": text,
                "model": model,
                "encoding_format": "float",
            },
            headers=headers,
            timeout=timeout,
        )
        try:
            r.raise_for_status()
        except Exception as e:
            print(r.text)
            raise e

        if raw:
            return r.json()

        emb = [
            x["embedding"] for x in sorted(r.json()["data"], key=lambda x: x["index"])
        ]
        return emb
