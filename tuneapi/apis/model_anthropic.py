# Copyright © 2024- Frello Technology Private Limited

import json
import requests
from typing import Optional, Dict, Any, Tuple, List

from tuneapi.utils import ENV, SimplerTimes as stime, from_json, to_json
from tuneapi.types import Thread, human, Message


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
                "Anthropic API key not found. Please set ANTHROPIC_TOKEN environment variable or pass through function"
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
            content = m["content"]
            if type(content) == str:
                content = [{"type": "text", "text": content.strip()}]
            claude_messages.append(
                {
                    "role": role,
                    "content": content,
                }
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
            line = line.decode().strip()
            if not "data:" in line or not line:
                continue

            # create openai style raw response
            if raw:
                try:
                    resp = json.loads(line.replace("data:", "").strip())
                    if "delta" in resp:
                        yield (
                            "data: "
                            + to_json(
                                {
                                    "choices": [
                                        {"delta": {"content": resp["delta"]["text"]}}
                                    ]
                                },
                                tight=True,
                            )
                        ).encode()
                except:
                    yield line
                continue

            # return token
            try:
                resp = json.loads(line.replace("data:", "").strip())
                if "delta" in resp:
                    yield resp["delta"]["text"]
            except:
                break
        return
