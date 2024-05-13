# Copyright © 2024- Frello Technology Private Limited

import json
import requests
from typing import Optional, Dict, Any, Tuple, List

from tuneapi.utils import ENV, SimplerTimes as stime, from_json, to_json
from tuneapi.types import Thread, human, Message


class Gemini:
    def __init__(
        self,
        id: Optional[str] = "gemini-1.5-pro-latest",
        base_url: str = "https://generativelanguage.googleapis.com/v1beta/models/{id}:{rpc}",
    ):
        self._gemeni_model_id = id
        self.base_url = base_url
        self.gemini_token = ENV.GEMINI_TOKEN("")

    def set_api_token(self, token: str) -> None:
        self.gemini_token = token

    def _process_input(self, chats, token: Optional[str] = None):
        if not token and not self.gemini_token:  # type: ignore
            raise Exception(
                "Gemini API key not found. Please set GEMINI_TOKEN environment variable or pass through function"
            )
        if isinstance(chats, Thread):
            messages_tt = chats.to_dict()["chats"]
        elif isinstance(chats, str):
            messages_tt = Thread(human(chats)).to_dict()["chats"]
        else:
            messages_tt = chats

        # create body
        # - multiple assistants works
        messages = []
        for m in messages_tt:
            messages.append(
                {
                    "role": m["role"],
                    "parts": [{"text": m["content"]}],
                }
            )

        # create headers
        headers = self._process_header()
        params = {"key": self.gemini_token}
        return headers, messages, params

    def _process_header(self):
        return {
            "Content-Type": "application/json",
        }

    def chat(
        self,
        chats: Thread | str,
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 1,
        token: Optional[str] = None,
        timeout=None,
        raw: bool = False,
        **kwargs,
    ) -> Any:
        output = ""
        for i in self.stream_chat(
            chats=chats,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            token=token,
            timeout=timeout,
            raw=raw,
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
        **kwargs,
    ):
        headers, messages, params = self._process_input(chats, token)
        data = {
            "contents": messages,
            "generationConfig": {
                "temperature": temperature,
                "topK": 0,
                "topP": 0.95,
                "maxOutputTokens": max_tokens,
                "stopSequences": [],
            },
            "safetySettings": [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE",
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE",
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE",
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE",
                },
            ],
        }
        data.update(kwargs)

        response = requests.post(
            self.base_url.format(
                id=model or self._gemeni_model_id,
                rpc="streamGenerateContent",
            ),
            headers=headers,
            params=params,
            json=data,
            stream=True,
            timeout=timeout,
        )
        try:
            response.raise_for_status()
        except Exception as e:
            print(response.text)
            raise e

        block_lines = ""
        done = False
        for lno, line in enumerate(response.iter_lines()):
            line = line.decode("utf-8")
            # print(f"[{lno:03d}] {line}")
            if raw:
                yield line
                continue

            # get the clean line for block
            if line == ("[{"):  # first line
                line = line[1:]
            elif line == "," or line == "]":  # intermediate or last line
                continue
            block_lines += line

            # is the block done?
            if line == "{":
                done = False
            elif line == "}":
                done = True

            if done:
                yield json.loads(block_lines)["candidates"][0]["content"]["parts"][0][
                    "text"
                ]
                block_lines = ""
