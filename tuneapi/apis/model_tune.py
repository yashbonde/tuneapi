# Copyright © 2024- Frello Technology Private Limited

import json
import requests
from typing import Optional, Dict, Any, Tuple, List

from tuneapi.utils import ENV, SimplerTimes as stime, from_json, to_json
from tuneapi.types import Thread, human, Message


class TuneModel:
    """Defines the model used in tune.app. See [Tune Studio](https://studio.tune.app/) for more information."""

    def __init__(self, id: Optional[str] = "rohan/mixtral-8x7b-inst-v0-1-32k"):
        self._tune_model_id = id
        self.tune_api_token = ENV.TUNEAPI_TOKEN("")

    def set_api_token(self, token: str) -> None:
        self.tune_api_token = token

    def _process_input(self, chats, token: Optional[str] = None):
        if not token and not self.tune_api_token:  # type: ignore
            raise Exception(
                "Tune API key not found. Please set TUNEAPI_TOKEN environment variable or pass through function"
            )
        token = token or self.tune_api_token
        if isinstance(chats, Thread):
            messages = chats.to_dict()["chats"]
        elif isinstance(chats, str):
            messages = Thread(human(chats)).to_dict()["chats"]
        else:
            messages = chats

        headers = {
            "Authorization": token,
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
        usage: bool = False,
        timeout=(5, 30),
    ) -> str | Tuple[str, int]:
        """
        Chat with the Tune Studio APIs, see more at https://studio.tune.app/

        Note: This is a API is partially compatible with OpenAI's API, so `messages` should be of type :code:`[{"role": ..., "content": ...}]`

        Args:
            model (str): The model to use, see https://studio.nbox.ai/ for more info
            messages (List[Dict[str, str]]): A list of messages to send to the API which are OpenAI compatible
            token (Secret, optional): The API key to use or set TUNEAPI_TOKEN environment variable
            max_tokens (int, optional): The maximum number of tokens to generate. Defaults to 1024.
            temperature (float, optional): The higher the temperature, the crazier the text. Defaults to 1.

        Returns:
            Dict[str, Any]: The response from the API
        """
        _st = stime.get_now_float()
        headers, messages = self._process_input(chats, token)
        data = {
            "temperature": temperature,
            "messages": messages,
            "model": model or self._tune_model_id,
            "stream": False,
            "max_tokens": max_tokens,
        }
        response = requests.post(
            "https://proxy.tune.app/chat/completions",
            headers=headers,
            json=data,
            timeout=timeout,
        )
        try:
            response.raise_for_status()
        except Exception as e:
            raise e
        odata = response.json()
        text = odata["choices"][0]["message"]["content"]
        if usage:
            return text, {
                "time_ms": int((stime.get_now_float() - _st) * 1000),
                **odata["usage"],
            }
        return text

    def stream_chat(
        self,
        chats: Thread | str,
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 1,
        token: Optional[str] = None,
        timeout=(5, 60),
        usage: bool = False,
    ):
        """
        Chat with the ChatNBX API with OpenAI compatability, see more at https://chat.nbox.ai/

        Note: This is a API is partially compatible with OpenAI's API, so `messages` should be of type :code:`[{"role": ..., "content": ...}]`

        Args:
            model (str): The model to use, see https://chat.nbox.ai/ for more info
            messages (List[Dict[str, str]]): A list of messages to send to the API which are OpenAI compatible
            token (Secret, optional): The API key to use or set TUNEAPI_TOKEN environment variable
            max_tokens (int, optional): The maximum number of tokens to generate. Defaults to 1024.
            temperature (float, optional): The higher the temperature, the crazier the text. Defaults to 1.

        Returns:
            Dict[str, Any]: The response from the API
        """
        _st = stime.get_now_float()
        headers, messages = self._process_input(chats, token)
        data = {
            "temperature": temperature,
            "messages": messages,
            "model": model,
            "stream": True,
            "max_tokens": max_tokens,
        }
        response = requests.post(
            "https://proxy.tune.app/chat/completions",
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
            line = line.decode().strip()
            if line:
                try:
                    yield json.loads(line.replace("data: ", ""))["choices"][0]["delta"][
                        "content"
                    ]
                except:
                    break
        if usage:
            return {"time_ms": int((stime.get_now_float() - _st) * 1000)}
        return
