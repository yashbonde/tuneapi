# Copyright © 2024- Frello Technology Private Limited

import json
import requests
from typing import Optional, Dict, Any, Tuple, List

from tuneapi.utils import ENV, SimplerTimes as stime, from_json, to_json
from tuneapi.types import Thread, human, Message


class Groq:
    def __init__(
        self,
        id: Optional[str] = "mixtral-8x7b-32768",
        base_url: str = "https://api.groq.com/openai/v1/chat/completions",
    ):
        self.groq_model_id = id
        self.base_url = base_url
        self.groq_api_token = ENV.GROQ_TOKEN("")

    def set_api_token(self, token: str) -> None:
        self.groq_api_token = token

    def _process_input(self, chats, token: Optional[str] = None):
        if not token and not self.groq_api_token:  # type: ignore
            raise Exception(
                "Please set GROQ_TOKEN environment variable or pass through function"
            )
        token = token or self.groq_api_token
        if isinstance(chats, Thread):
            messages = chats.to_dict()["chats"]
        elif isinstance(chats, str):
            messages = Thread(human(chats)).to_dict()["chats"]
        else:
            messages = chats

        headers = {
            "Authorization": "Bearer " + token,
            "Content-Type": "application/json",
        }
        return headers, messages

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
            "model": model or self.groq_model_id,
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
