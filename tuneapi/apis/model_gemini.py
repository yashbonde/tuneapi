"""
Connect to the Google Gemini API to their LLMs. See more `Gemini <https://ai.google.dev>`_.
"""

# Copyright © 2024-2025 Frello Technology Private Limited
# Copyright © 2025-2025 Yash Bonde github.com/yashbonde
# MIT License
# https://ai.google.dev/gemini-api/docs/function-calling

import httpx
from pydantic import BaseModel
from typing import get_args, get_origin, List, Optional, Dict, Any, Union, Tuple

import tuneapi.utils as tu
import tuneapi.types as tt
from tuneapi.apis.turbo import distributed_chat, distributed_chat_async


class Gemini(tt.ModelInterface):

    def __init__(
        self,
        id: Optional[str] = "gemini-2.0-flash-exp",
        base_url: str = "https://generativelanguage.googleapis.com/v1beta/models/{id}:{rpc}",
        extra_headers: Optional[Dict[str, str]] = None,
        api_token: Optional[str] = None,
        emebdding_url: Optional[str] = None,
    ):
        super().__init__()

        self.model_id = id
        self.base_url = base_url
        self.api_token = api_token or tu.ENV.GEMINI_TOKEN("")
        self.extra_headers = extra_headers
        self.embedding_url = emebdding_url or base_url
        self.client = None

    def set_api_token(self, token: str) -> None:
        self.api_token = token

    def _process_header(self):
        return {
            "Content-Type": "application/json",
        }

    def _process_input(
        self,
        chats: tt.Thread | str,
        model: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 1,
        token: Optional[str] = None,
        debug: bool = False,
        extra_headers: Optional[Dict[str, str]] = None,
        **kwargs,
    ):
        if not token and not self.api_token:  # type: ignore
            raise Exception(
                "Gemini API key not found. Please set GEMINI_TOKEN environment variable or pass through function"
            )
        if isinstance(chats, tt.Thread):
            thread = chats
        elif isinstance(chats, str):
            thread = tt.Thread(tt.human(chats))
        else:
            raise Exception("Invalid input")

        system = ""
        if thread.chats[0].role == tt.Message.SYSTEM:
            system = thread.chats[0].value

        messages = []
        prev_fn_name = ""
        for m in thread.chats[int(system != "") :]:
            if m.role == tt.Message.HUMAN:
                inline_objects = [{"text": m.value}]
                for img in m.images:
                    inline_objects.append(
                        {
                            "inline_data": {
                                "mime_type": "image/png",
                                "data": img,
                            },
                        }
                    )
                messages.append({"role": "user", "parts": inline_objects})
            elif m.role == tt.Message.GPT:
                inline_objects = [{"text": m.value}]
                for img in m.images:
                    inline_objects.append(
                        {
                            "inline_data": {
                                "mime_type": "image/png",
                                "data": img,
                            },
                        }
                    )
                messages.append({"role": "model", "parts": inline_objects})
            elif m.role == tt.Message.FUNCTION_CALL:
                _m = tu.from_json(m.value) if isinstance(m.value, str) else m.value
                prev_fn_name = _m["name"]
                messages.append(
                    {
                        "role": "model",
                        "parts": [
                            {
                                "functionCall": {
                                    "name": _m["name"],
                                    "args": _m["arguments"],
                                }
                            }
                        ],
                    }
                )
            elif m.role == tt.Message.FUNCTION_RESP:
                # _m = tu.from_json(m.value) if isinstance(m.value, str) else m.value
                messages.append(
                    {
                        "role": "function",
                        "parts": [
                            {
                                "functionResponse": {
                                    "name": prev_fn_name,
                                    "response": {
                                        "name": prev_fn_name,
                                        "content": m.value,
                                    },
                                }
                            }
                        ],
                    }
                )
            else:
                raise Exception(f"Unknown role: {m.role}")

        # create headers and params
        headers = self._process_header()
        url = self.base_url.format(
            id=model or self.model_id,
            rpc="streamGenerateContent",
        )

        # create the body
        tools = []
        if isinstance(chats, tt.Thread) and chats.tools:
            tools = [x.to_dict() for x in chats.tools]
        extra_headers = extra_headers or self.extra_headers
        if extra_headers:
            headers.update(extra_headers)

        data = {
            "systemInstruction": {
                "parts": [{"text": system}],
            },
            "contents": messages,
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

        generation_config = {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
            "stopSequences": [],
        }

        if isinstance(chats, tt.Thread) and chats.schema:
            generation_config.update(
                {
                    "response_mime_type": "application/json",
                    "response_schema": get_structured_schema(chats.schema),
                }
            )
        data["generationConfig"] = generation_config

        if tools:
            data["tool_config"] = {
                "function_calling_config": {
                    "mode": "ANY",
                }
            }
            std_tools = []
            for i, t in enumerate(tools):
                props = t["parameters"]["properties"]
                t_copy = t.copy()
                if not props:
                    t_copy.pop("parameters")
                std_tools.append(t_copy)
            data["tools"] = [{"function_declarations": std_tools}]
        data.update(kwargs)

        if debug:
            fp = "sample_gemini.json"
            tu.logger.info("Saving gemini prompt at " + fp)
            tu.to_json(data, fp=fp)

        return url, headers, data

    def _process_output(self, raw: bool, lines_fn: callable):
        block_lines = ""
        done = False
        for line in lines_fn():
            if isinstance(line, bytes):
                line = line.decode("utf-8")

            # get the clean line for block
            if line == ("[{"):  # first line
                line = line[1:]
            elif line == "," or line == "]":  # intermediate or last line
                continue
            block_lines += line

            done = False
            try:
                tu.from_json(block_lines)
                done = True
            except Exception as e:
                pass

            # print(f"{block_lines=}")
            if done:
                part_data = tu.from_json(block_lines)["candidates"][0]["content"][
                    "parts"
                ][0]
                if "text" in part_data:
                    if raw:
                        yield b"data: " + tu.to_json(
                            {
                                "object": "gemini_text",
                                "choices": [{"delta": {"content": part_data["text"]}}],
                            },
                            tight=True,
                        ).encode()
                        yield b""
                    else:
                        yield part_data["text"]
                elif "functionCall" in part_data:
                    fn_call = part_data["functionCall"]
                    fn_call["arguments"] = fn_call.pop("args")
                    yield fn_call
                block_lines = ""

    # Interaction methods

    def stream_chat(
        self,
        chats: tt.Thread | str,
        model: Optional[str] = None,
        max_tokens: int = None,
        temperature: float = 1,
        token: Optional[str] = None,
        timeout=(5, 60),
        usage: bool = False,
        extra_headers: Optional[Dict[str, str]] = None,
        debug: bool = False,
        raw: bool = False,
        **kwargs,
    ):
        url, headers, data = self._process_input(
            chats=chats,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            token=token,
            debug=debug,
            extra_headers=extra_headers,
            **kwargs,
        )

        if self.client is None:
            self.set_client()

        response = self.client.post(
            url=url,
            headers=headers,
            params={"key": self.api_token},
            json=data,
            timeout=timeout,
        )
        try:
            response.raise_for_status()
        except Exception as e:
            yield response.text
            raise e

        yield from self._process_output(
            raw=raw,
            lines_fn=response.iter_lines,
        )

    def chat(
        self,
        chats: tt.Thread | str,
        model: Optional[str] = None,
        max_tokens: int = None,
        temperature: float = 1,
        token: Optional[str] = None,
        usage: bool = False,
        extra_headers: Optional[Dict[str, str]] = None,
        debug: bool = False,
        timeout=(5, 60),
        **kwargs,
    ) -> Any:
        output = ""
        x = None
        try:
            for x in self.stream_chat(
                chats=chats,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                token=token,
                timeout=timeout,
                extra_headers=extra_headers,
                debug=debug,
                raw=False,
                **kwargs,
            ):
                if isinstance(x, dict):
                    output = x
                else:
                    output += x
        except httpx.HTTPError as e:
            print(e.response.text)
            raise e

        if isinstance(chats, tt.Thread) and chats.schema:
            output = chats.schema(**tu.from_json(output))
            return output
        return output

    async def stream_chat_async(
        self,
        chats: tt.Thread | str,
        model: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 1,
        token: Optional[str] = None,
        raw: bool = False,
        debug: bool = False,
        extra_headers: Optional[Dict[str, str]] = None,
        timeout=(5, 60),
        **kwargs,
    ):
        url, headers, data = self._process_input(
            chats=chats,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            token=token,
            debug=debug,
            extra_headers=extra_headers,
            **kwargs,
        )

        if self.async_client is None:
            self.set_async_client()

        response = await self.async_client.post(
            url=url,
            headers=headers,
            params={"key": self.api_token},
            json=data,
            timeout=timeout,
        )
        try:
            response.raise_for_status()
        except Exception as e:
            yield str(e)
            return

        async for chunk in response.aiter_bytes():
            for x in self._process_output(
                raw=raw,
                lines_fn=chunk.decode("utf-8").splitlines,
            ):
                yield x

    async def chat_async(
        self,
        chats: tt.Thread | str,
        model: Optional[str] = None,
        max_tokens: int = None,
        temperature: float = 1,
        token: Optional[str] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        debug: bool = False,
        timeout=(5, 60),
        **kwargs,
    ) -> Any:
        output = ""
        x = None
        try:
            async for x in self.stream_chat_async(
                chats=chats,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                token=token,
                timeout=timeout,
                extra_headers=extra_headers,
                raw=False,
                debug=debug,
                **kwargs,
            ):
                if isinstance(x, dict):
                    output = x
                else:
                    output += x
        except Exception as e:
            if not x:
                raise e
            else:
                raise ValueError(x)

        if isinstance(chats, tt.Thread) and chats.schema:
            output = chats.schema(**tu.from_json(output))
            return output
        return output

    def distributed_chat(
        self,
        prompts: List[tt.Thread],
        post_logic: Optional[callable] = None,
        max_threads: int = 10,
        retry: int = 3,
        pbar: bool = True,
        debug: bool = False,
        time_metrics: bool = False,
        **kwargs,
    ):
        return distributed_chat(
            self,
            prompts=prompts,
            post_logic=post_logic,
            max_threads=max_threads,
            retry=retry,
            pbar=pbar,
            debug=debug,
            time_metrics=time_metrics,
            **kwargs,
        )

    async def distributed_chat_async(
        self,
        prompts: List[tt.Thread],
        post_logic: Optional[callable] = None,
        max_threads: int = 10,
        retry: int = 3,
        pbar: bool = True,
        debug: bool = False,
        time_metrics: bool = False,
        **kwargs,
    ):
        return await distributed_chat_async(
            self,
            prompts=prompts,
            post_logic=post_logic,
            max_threads=max_threads,
            retry=retry,
            pbar=pbar,
            debug=debug,
            time_metrics=time_metrics,
            **kwargs,
        )

    # Embedding methods

    def _prepare_embedding_input(
        self,
        chats: tt.Thread | List[str] | str,
        model: str = "text-embedding-004:embedContent",
        extra_headers: Optional[Dict[str, str]] = None,
    ):
        headers = self._process_header()
        extra_headers = extra_headers or self.extra_headers
        if extra_headers:
            headers.update(extra_headers)
        if isinstance(chats, tt.Thread):
            text = []
            for i, m in enumerate(chats.chats):
                x = f"<{m.role}> : {m.value}"
                text.append(x)
        elif isinstance(chats, tt.Message):
            text = [chats.value]
        elif isinstance(chats, list) and len(chats) and isinstance(chats[0], str):
            if len(chats) > 1:
                raise ValueError(
                    f"Only one string can be passed for list of strings. Got: {len(chats)}"
                )
            text = chats
        elif isinstance(chats, str):
            text = [chats]
        else:
            raise ValueError(f"Invalid input type. Got {type(chats)}")
        data = {
            "model": f"models/{model}",
            "content": {"parts": [{"text": x} for x in text]},
        }
        url = self.base_url.format(id=model, rpc="embedContent")
        return url, headers, data

    def embedding(
        self,
        chats: tt.Thread | List[str] | str,
        model: str = "text-embedding-004",
        extra_headers: Optional[Dict[str, str]] = None,
        token: Optional[str] = None,
        timeout: Tuple[int, int] = (5, 60),
        raw: bool = False,
    ) -> tt.EmbeddingGen:
        url, headers, data = self._prepare_embedding_input(
            chats=chats,
            model=model,
            extra_headers=extra_headers,
        )

        if self.client is None:
            self.set_client()

        response = self.client.post(
            url=url,
            headers=headers,
            json=data,
            timeout=timeout,
            params={"key": token or self.api_token},
        )
        try:
            response.raise_for_status()
        except httpx.HTTPError as e:
            tu.logger.error(f"Cannot get emeddings: {response.text}")
            raise e
        except Exception as e:
            tu.logger.error(e)
            raise e
        resp = response.json()
        if raw:
            return resp

        return tt.EmbeddingGen(embedding=[resp["embedding"]["values"]])

    async def embedding_async(
        self,
        chats: tt.Thread | List[str] | str,
        model: str = "text-embedding-004",
        extra_headers: Optional[Dict[str, str]] = None,
        token: Optional[str] = None,
        timeout: Tuple[float, float] = (5.0, 60.0),  # httpx uses float for timeouts
        raw: bool = False,
    ) -> tt.EmbeddingGen:
        url, headers, data = self._prepare_embedding_input(
            chats=chats,
            model=model,
            extra_headers=extra_headers,
        )

        if self.async_client is None:
            self.set_async_client()

        response = await self.async_client.post(
            url=url,
            headers=headers,
            json=data,
            timeout=timeout,
            params={"key": token or self.api_token},
        )
        try:
            response.raise_for_status()  # Raise an exception for bad status codes
        except httpx.HTTPError as e:
            tu.logger.error(f"Cannot get embeddings: {e.response.text}")
            raise
        except Exception as e:
            tu.logger.error(e)
            raise

        resp = response.json()
        if raw:
            return resp

        return tt.EmbeddingGen(embedding=[resp["embedding"]["values"]])


# helpers


def get_structured_schema(model: type[BaseModel]) -> Dict[str, Any]:
    """
    Converts a Pydantic BaseModel to a JSON schema compatible with Gemini API,
    including `anyOf` for optional or union types and handling nested structures correctly.

    Args:
        model: The Pydantic BaseModel class to convert.

    Returns:
        A dictionary representing the JSON schema.
    """

    def _process_field(
        field_name: str, field_type: Any, field_description: str = None
    ) -> dict:
        """Helper function to process a single field."""
        schema = {}
        origin = get_origin(field_type)
        args = get_args(field_type)

        if origin is list:
            schema["type"] = "array"
            if args:
                item_schema = _process_field_type(args[0])
                schema["items"] = item_schema
                if "type" not in item_schema and "anyOf" not in item_schema:
                    schema["items"]["type"] = "object"  # default item type for list
            else:
                schema["items"] = {}
        elif origin is Optional:
            if args:
                inner_schema = _process_field_type(args[0])
                schema["anyOf"] = [inner_schema, {"type": "null"}]
            else:
                schema = {"type": "null"}
        elif origin is dict:
            schema["type"] = "object"
            if len(args) == 2:
                schema["additionalProperties"] = _process_field_type(args[1])
        else:
            schema = _process_field_type(field_type)

        if field_description:
            schema["description"] = field_description
        return schema

    def _process_field_type(field_type: Any) -> dict:
        """Helper function to process the type of a field."""

        origin = get_origin(field_type)
        args = get_args(field_type)

        if field_type is str:
            return {"type": "string"}
        elif field_type is int:
            return {"type": "integer"}
        elif field_type is float:
            return {"type": "number"}
        elif field_type is bool:
            return {"type": "boolean"}
        elif isinstance(field_type, type) and issubclass(field_type, BaseModel):
            return get_structured_schema(field_type)  # Recursive call for nested models
        elif origin is list:
            schema = {"type": "array"}
            if args:
                item_schema = _process_field_type(args[0])
                schema["items"] = item_schema
                if "type" not in item_schema and "anyOf" not in item_schema:
                    schema["items"]["type"] = "object"
            return schema
        elif origin is Optional:
            return _process_field_type(args[0])
        elif origin is dict:
            schema = {"type": "object"}
            if len(args) == 2:
                schema["additionalProperties"] = _process_field_type(args[1])
            return schema
        elif origin is Union:
            return _process_field_type(args[0])
        else:
            return {"type": "string"}  # default any object to string

    schema = {"type": "object", "properties": {}, "required": []}

    for field_name, field in model.model_fields.items():
        field_description = field.description
        if field.is_required():
            schema["required"].append(field_name)

        schema["properties"][field_name] = _process_field(
            field_name, field.annotation, field_description
        )

    if model.__doc__:
        schema["description"] = model.__doc__.strip()
    return schema
