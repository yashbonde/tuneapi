# Copyright © 2024- Frello Technology Private Limited

import json
import requests
from typing import Optional, Dict, Any, Tuple, List

from tuneapi.utils import ENV, SimplerTimes as stime, from_json, to_json
from tuneapi.types import Thread, human, Message


class Openai:
    def __init__(
        self,
        id: Optional[str] = "gpt-3.5-turbo",
        base_url: str = "https://api.openai.com/v1/chat/completions",
    ):
        self._openai_model_id = id
        self.base_url = base_url
        self.openai_api_token = ENV.OPENAI_TOKEN("")

    def set_api_token(self, token: str) -> None:
        self.openai_api_token = token

    def _process_input(self, chats, token: Optional[str] = None):
        if not token and not self.openai_api_token:  # type: ignore
            raise Exception(
                "OpenAI API key not found. Please set OPENAI_TOKEN environment variable or pass through function"
            )
        if isinstance(chats, Thread):
            messages = chats.to_dict()["chats"]
        elif isinstance(chats, str):
            messages = Thread(human(chats)).to_dict()["chats"]
        else:
            messages = chats
        headers = self._process_header(token)
        return headers, messages

    def _process_header(self, token: Optional[str] = None):
        return {
            "Authorization": "Bearer " + (token or self.openai_api_token),
            "Content-Type": "application/json",
        }

    def chat(
        self,
        chats: Thread | str,
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 1,
        token: Optional[str] = None,
        **kwargs,
    ) -> Any:
        output = ""
        for i in self.stream_chat(
            chats=chats,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            token=token,
            **kwargs,
        ):
            output += i
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
    ):
        headers, messages = self._process_input(chats, token)
        data = {
            "temperature": temperature,
            "messages": messages,
            "model": model or self._openai_model_id,
            "stream": True,
            "max_tokens": max_tokens,
        }
        # for m in messages:
        #     print(m)
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
            print(response.text)
            raise e

        for line in response.iter_lines():
            if raw:
                yield line
                continue

            line = line.decode().strip()
            if line:
                try:
                    yield json.loads(line.replace("data: ", ""))["choices"][0]["delta"][
                        "content"
                    ]
                except:
                    break
        return

    def function_call(
        self,
        chats: Thread | str,
        tools: List,
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = None,
        token: Optional[str] = None,
        timeout=(5, 60),
    ):
        headers, messages = self._process_input(chats, token)
        data = {
            "temperature": temperature,
            "messages": messages,
            "model": model or self._openai_model_id,
            "tools": tools,
            "stream": False,
            "max_tokens": max_tokens,
        }
        # for m in messages:
        #     print(m)
        r = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=timeout,
        )
        try:
            r.raise_for_status()
        except Exception as e:
            print(r.text)
            raise e
        x = r.json()["choices"][0]["message"]
        if "tool_calls" not in x:
            return x["content"]
        else:
            y = x["tool_calls"][0]["function"]
            # print(x)
            return {
                "name": y["name"],
                "arguments": from_json(y["arguments"]),
            }

    def embedding(
        self,
        chats: Thread | List[str] | str,
        cum: bool = False,
        model: str = "text-embedding-3-small",
        token: Optional[str] = None,
        timeout=(5, 60),
    ):
        text = []

        headers = self._process_header(token)
        if isinstance(chats, Thread):
            headers, messages = self._process_input(chats, token)
            for i, m in enumerate(messages):
                x = f"<{m['role']}> : {m['content']}\n\n"
                if not text:
                    text.append(x)
                    continue
                if cum:
                    # create a cumulative string from start till now
                    x = ""
                    for j in range(i + 1):
                        x += f"<{messages[j]['role']}> : {messages[j]['content']}\n\n"
                else:
                    # attach to previous message
                    text[-1] += x
                text.append(x)
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
        emb = [
            x["embedding"] for x in sorted(r.json()["data"], key=lambda x: x["index"])
        ]
        return emb
