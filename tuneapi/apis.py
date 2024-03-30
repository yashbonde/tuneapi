# Copyright © 2024- Frello Technology Private Limited

import json
import requests
from typing import Optional, Dict, Any, Tuple, List

from tuneapi.utils import ENV, SimplerTimes as stime, from_json
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


class Openai:
    def __init__(self, id: Optional[str] = "gpt-3.5-turbo"):
        self._openai_model_id = id
        self.openai_api_token = ENV.OPENAI_TOKEN("")

    def set_api_token(self, token: str) -> None:
        self.openai_api_token = token

    def _process_input(self, chats, token: Optional[str] = None):
        if not token and not self.openai_api_token:  # type: ignore
            raise Exception(
                "OpenAI API key not found. Please set OPENAI_TOKEN environment variable or pass through function"
            )
        token = token or self.openai_api_token
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
        """
        Returns a JSON object containing the OpenAI's API chat response. See [OpenAI documentation](https://platform.openai.com/docs/api-reference/chat/create).

        Args:
            messages: A list of messages describing the conversation so far
            model: ID of the model to use. See [OpenAI documentation](https://platform.openai.com/docs/api-reference/chat/create).
            token (Secret): The OpenAI API key. Defaults to "" or the OPENAI_TOKEN environment variable.
            temperature: Optional. What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic. We generally recommend altering this or top_p but not both. Defaults to 1.
            top_p: Optional. An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens comprising the top 10% probability mass are considered. We generally recommend altering this or temperature but not both. Defaults to 1.
            n: Optional. How many chat completion choices to generate for each input message. Defaults to 1.
            stop: Optional. Up to 4 sequences where the API will stop generating further tokens.
            max_tokens: Optional. The maximum number of tokens to generate in the chat completion. The total length of input tokens and generated tokens is limited by the model's context length. Defaults to infinity.
            presence_penalty: Optional. Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics. See more information about frequency and presence penalties. Defaults to 0.
            frequency_penalty: Optional. Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim. See more information about frequency and presence penalties. Defaults to 0.
            logit_bias: Optional. Modify the likelihood of specified tokens appearing in the completion. Accepts a json object that maps tokens (specified by their token ID in the tokenizer) to an associated bias value from -100 to 100. Mathematically, the bias is added to the logits generated by the model prior to sampling. The exact effect will vary per model, but values between -1 and 1 should decrease or increase likelihood of selection; values like -100 or 100 should result in a ban or exclusive selection of the relevant
            user: Optional. A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse. Defaults to None.
            retry_count: Optional. Number of times to retry the API call. Defaults to 3.
            retry_delay: Optional. Number of seconds to wait before retrying the API call. Defaults to 1.

        Returns:
            Any: The completion(s) generated by the API.
        """
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
            "https://api.openai.com/v1/chat/completions",
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
            return x["content"], False
        else:
            y = x["tool_calls"][0]["function"]
            # print(x)
            return {
                "name": y["name"],
                "arguments": from_json(y["arguments"]),
            }, True


class Anthropic:
    def __init__(self, model: Optional[str] = "claude-3-haiku-20240307"):
        self.anthropic_model = model
        self.anthropic_token = ENV.ANTHROPIC_TOKEN("")

    def set_api_token(self, token: str) -> None:
        self.anthropic_token = token

    def tool_to_claude_xml(self, tool):
        tool_signature = ""
        if len(tool["parameters"]) > 0:
            for name, p in tool["parameters"]["properties"].items():
                param = f"""<parameter>
                <name> {name} </name>
                <type> {p['type']} </type>
                <description> {p['description']} </description>
                """
                if name in tool["parameters"]["required"]:
                    param += "<required> true </required>\n"
                param += "</parameter>"
                tool_signature += param + "\n"
        tool_signature = tool_signature.strip()

        constructed_prompt = (
            "<tool_description>\n"
            f"<tool_name> {tool['name']} </tool_name>\n"
            "<description>\n"
            f"{tool['description']}\n"
            "</description>\n"
            "<parameters>\n"
            f"{tool_signature}\n"
            "</parameters>\n"
            "</tool_description>"
        )
        return constructed_prompt

    def _process_input(
        self, chats, tools: Optional[List] = None, token: Optional[str] = None
    ):
        if not token and not self.anthropic_token:  # type: ignore
            raise Exception(
                "OpenAI API key not found. Please set OPENAI_TOKEN environment variable or pass through function"
            )
        token = token or self.anthropic_token
        if isinstance(chats, Thread):
            messages = chats.to_dict()["chats"]
        elif isinstance(chats, str):
            messages = Thread(human(chats)).to_dict()["chats"]
        else:
            messages = chats

        # create the anthropic style data
        system = ""
        claude_messages = []
        if messages[0]["role"] == Message.SYSTEM:
            system = messages[0]["content"]
        start_idx = 1 if system else 0
        for m in messages[start_idx:]:
            role = m["role"]
            if m["role"] == Message.HUMAN:
                role = "user"
            claude_messages.append(
                {"role": role, "content": [{"type": "text", "text": m["content"]}]}
            )

        if tools:
            tool_use_system_prompt = (
                "In this environment you have access to a set of tools you can use to answer the user's question.\n"
                "\n"
                "You may call them like this:\n"
                "<function_calls>\n"
                "<invoke>\n"
                "<tool_name>$TOOL_NAME</tool_name>\n"
                "<parameters>\n"
                "<$PARAMETER_NAME>$PARAMETER_VALUE</$PARAMETER_NAME>\n"
                "...\n"
                "</parameters>\n"
                "</invoke>\n"
                "</function_calls>\n"
                "\n"
                "Here are the tools available:\n"
                "<tools>\n"
                + "\n".join([self.tool_to_claude_xml(tool) for tool in tools])
                + "\n</tools>"
            )
            system += "\n\n" + tool_use_system_prompt
        system = system.strip()

        headers = {
            "x-api-key": token,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }
        return headers, system, claude_messages

    def chat(
        self,
        chats: Thread | str,
        tools: Optional[List] = None,
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 1,
        token: Optional[str] = None,
        **kwargs,
    ):
        output = ""
        for i in self.stream_chat(
            chats=chats,
            tools=tools,
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
        tools: Optional[List] = None,
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 1,
        token: Optional[str] = None,
        timeout=(5, 60),
        raw: bool = False,
        **kwargs,
    ) -> Any:
        headers, system, messages = self._process_input(
            chats=chats,
            tools=tools,
            token=token,
        )
        data = {
            "model": model or self.anthropic_model,
            "max_tokens": max_tokens,
            "messages": messages,
            "temperature": temperature,
            "system": system,
            "stream": True,
            **kwargs,
        }
        r = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=data,
            timeout=timeout,
        )
        try:
            r.raise_for_status()
        except Exception as e:
            print(r.text)
            raise e

        for line in r.iter_lines():
            if raw:
                yield line
                continue

            line = line.decode().strip()
            if line and "data:" in line:
                try:
                    resp = json.loads(line.replace("data:", "").strip())
                    if "delta" in resp:
                        yield resp["delta"]["text"]
                except:
                    break
        return
