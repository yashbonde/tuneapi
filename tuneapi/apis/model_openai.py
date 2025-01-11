"""
Connect to the `OpenAI API <https://playground.openai.com/>`_ and use their LLMs.
"""

# Copyright © 2024-2025 Frello Technology Private Limited

import httpx
import requests
from typing import Optional, Any, List, Dict

import tuneapi.utils as tu
import tuneapi.types as tt
from tuneapi.apis.turbo import distributed_chat, distributed_chat_async


class Openai(tt.ModelInterface):
    def __init__(
        self,
        id: str = "gpt-4o",
        base_url: str = "https://api.openai.com/v1/chat/completions",
        extra_headers: Optional[Dict[str, str]] = None,
        api_token: Optional[str] = None,
    ):
        self.model_id = id
        self.base_url = base_url
        self.api_token = api_token or tu.ENV.OPENAI_TOKEN("")
        self.extra_headers = extra_headers

    def set_api_token(self, token: str) -> None:
        self.api_token = token

    def _process_header(self, token: Optional[str] = None):
        return {
            "Authorization": "Bearer " + (token or self.api_token),
            "Content-Type": "application/json",
        }

    def _process_input(
        self,
        chats,
        model: Optional[str] = None,
        max_tokens: int = None,
        temperature: float = 1,
        parallel_tool_calls: bool = False,
        token: Optional[str] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        debug: bool = False,
        **kwargs,
    ):
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
                content = [{"type": "text", "text": m.value}]
                for img in m.images:
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img}"},
                        }
                    )
                final_messages.append({"role": "user", "content": content})
            elif m.role == tt.Message.GPT:
                content = [{"type": "text", "text": m.value}]
                for img in m.images:
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img}"},
                        }
                    )
                final_messages.append({"role": "assistant", "content": content})
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
        # return headers, final_messages

        extra_headers = extra_headers or self.extra_headers
        if extra_headers:
            headers.update(extra_headers)
        data = {
            "temperature": temperature,
            "messages": final_messages,
            "model": model or self.model_id,
            "stream": True,
        }
        if max_tokens:
            data["max_tokens"] = max_tokens
        if isinstance(chats, tt.Thread) and len(chats.tools):
            data["tools"] = [
                {"type": "function", "function": x.to_dict()} for x in chats.tools
            ]
            data["parallel_tool_calls"] = parallel_tool_calls
        if isinstance(chats, tt.Thread) and chats.schema:
            data["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "schema": chats.schema.model_json_schema(),
                    "name": "chat",
                },
            }

        if kwargs:
            data.update(kwargs)

        if debug:
            fp = "sample_oai.json"
            print("Saving at path " + fp)
            tu.to_json(data, fp=fp)

        return headers, data

    def _process_output(self, raw: bool, lines_fn: callable):
        fn_call = None
        for line in lines_fn():
            if isinstance(line, bytes):
                line = line.decode("utf-8")
            if raw:
                yield line
                continue

            line = line.strip()
            if line:
                try:
                    x = tu.from_json(line.replace("data: ", ""))["choices"][0]["delta"]
                    if "tool_calls" not in x:
                        yield x["content"]
                    else:
                        y = x["tool_calls"][0]["function"]
                        if fn_call is None:
                            fn_call = {
                                "name": y["name"],
                                "arguments": y["arguments"],
                            }
                        else:
                            fn_call["arguments"] += y["arguments"]
                except:
                    break
        if fn_call:
            fn_call["arguments"] = tu.from_json(fn_call["arguments"])
            yield fn_call

    # Interface methods

    def chat(
        self,
        chats: tt.Thread | str,
        model: Optional[str] = None,
        max_tokens: int = None,
        temperature: float = 1,
        parallel_tool_calls: bool = False,
        token: Optional[str] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> Any:
        output = ""
        try:
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
        except requests.HTTPError as e:
            print(e.response.text)
            raise e

        if isinstance(chats, tt.Thread) and chats.schema:
            output = chats.schema(**tu.from_json(output))
            return output
        return output

    def stream_chat(
        self,
        chats: tt.Thread | str,
        model: Optional[str] = None,
        max_tokens: int = None,
        temperature: float = 1,
        parallel_tool_calls: bool = False,
        token: Optional[str] = None,
        timeout=(5, 60),
        extra_headers: Optional[Dict[str, str]] = None,
        debug: bool = False,
        raw: bool = False,
        **kwargs,
    ):
        headers, data = self._process_input(
            chats=chats,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            parallel_tool_calls=parallel_tool_calls,
            token=token,
            extra_headers=extra_headers,
            debug=debug,
            **kwargs,
        )
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

        yield from self._process_output(raw, response.iter_lines)

    async def chat_async(
        self,
        chats: tt.Thread | str,
        model: Optional[str] = None,
        max_tokens: int = None,
        temperature: float = 1,
        parallel_tool_calls: bool = False,
        token: Optional[str] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> Any:
        output = ""
        async for x in self.stream_chat_async(
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
        if isinstance(chats, tt.Thread) and chats.schema:
            output = chats.schema(**tu.from_json(output))
            return output
        return output

    async def stream_chat_async(
        self,
        chats: tt.Thread | str,
        model: Optional[str] = None,
        max_tokens: int = None,
        temperature: float = 1,
        parallel_tool_calls: bool = False,
        token: Optional[str] = None,
        timeout=(5, 60),
        extra_headers: Optional[Dict[str, str]] = None,
        debug: bool = False,
        raw: bool = False,
        **kwargs,
    ):
        headers, data = self._process_input(
            chats=chats,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            parallel_tool_calls=parallel_tool_calls,
            token=token,
            extra_headers=extra_headers,
            debug=debug,
            **kwargs,
        )

        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.base_url,
                headers=headers,
                json=data,
                timeout=timeout,
            )
            try:
                response.raise_for_status()
            except Exception as e:
                yield response.text
                raise e

            async for chunk in response.aiter_bytes():
                for x in self._process_output(
                    raw=raw, lines_fn=chunk.decode("utf-8").splitlines
                ):
                    yield x

    def distributed_chat(
        self,
        prompts: List[tt.Thread],
        post_logic: Optional[callable] = None,
        max_threads: int = 10,
        retry: int = 3,
        pbar=True,
        debug=False,
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
            **kwargs,
        )

    async def distributed_chat_async(
        self,
        prompts: List[tt.Thread],
        post_logic: Optional[callable] = None,
        max_threads: int = 10,
        retry: int = 3,
        pbar=True,
        debug=False,
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
            **kwargs,
        )

    # Embedding models

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


# Other OpenAI compatible models


class Mistral(Openai):
    def __init__(
        self,
        id: str = "mistral-small-latest",
        base_url: str = "https://api.mistral.ai/v1/chat/completions",
        extra_headers: Optional[Dict[str, str]] = None,
        api_token: Optional[str] = None,
    ):
        super().__init__(
            id=id,
            base_url=base_url,
            extra_headers=extra_headers,
            api_token=api_token or tu.ENV.MISTRAL_TOKEN(),
        )

    def embedding(*a, **k):
        raise NotImplementedError("Mistral does not support embeddings")


class Groq(Openai):
    def __init__(
        self,
        id: str = "llama3-70b-8192",
        base_url: str = "https://api.groq.com/openai/v1/chat/completions",
        extra_headers: Optional[Dict[str, str]] = None,
        api_token: Optional[str] = None,
    ):
        super().__init__(
            id=id,
            base_url=base_url,
            extra_headers=extra_headers,
            api_token=api_token or tu.ENV.GROQ_TOKEN(),
        )

    def embedding(*a, **k):
        raise NotImplementedError("Groq does not support embeddings")


class TuneModel(Openai):
    def __init__(
        self,
        id: str = "meta/llama-3.1-8b-instruct",
        base_url: str = "https://proxy.tune.app/chat/completions",
        api_token: Optional[str] = None,
        org_id: Optional[str] = None,
        extra_headers: Optional[Dict[str, str]] = None,
    ):
        if extra_headers is None:
            extra_headers = {}
        if org_id is not None:
            extra_headers["X-Org-Id"] = org_id

        super().__init__(
            id=id,
            base_url=base_url,
            extra_headers=extra_headers,
            api_token=api_token or tu.ENV.TUNEAPI_TOKEN(),
        )

    def embedding(*a, **k):
        raise NotImplementedError("TuneModel does not support embeddings")
