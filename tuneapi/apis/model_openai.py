"""
Connect to the `OpenAI API <https://playground.openai.com/>`_ and use their LLMs.
"""

# Copyright © 2024-2025 Frello Technology Private Limited
# Copyright © 2025-2025 Yash Bonde github.com/yashbonde
# MIT License

import os
import httpx
import aiofiles
from PIL import Image
from io import BytesIO
from copy import deepcopy
from typing import Optional, Any, List, Dict, Tuple

import tuneapi.utils as tu
import tuneapi.types as tt
from tuneapi.apis.turbo import distributed_chat, distributed_chat_async


class OpenAIProtocol(tt.ModelInterface):
    def __init__(
        self,
        id: str,
        base_url: str,
        extra_headers: Optional[Dict[str, str]],
        api_token: Optional[str],
        emebdding_url: Optional[str],
        image_gen_url: Optional[str],
        audio_transcribe_url: Optional[str],
        audio_gen_url: Optional[str],
        batch_url: Optional[str],
        files_url: Optional[str],
    ):
        super().__init__()

        self.model_id = id
        self.base_url = base_url
        self.api_token = api_token
        self.extra_headers = extra_headers
        self.emebdding_url = emebdding_url
        self.image_gen_url = image_gen_url
        self.audio_transcribe_url = audio_transcribe_url
        self.audio_gen_url = audio_gen_url
        self.batch_url = batch_url
        self.files_url = files_url
        self.client = None

    # setters

    def set_api_token(self, token: str) -> None:
        self.api_token = token

    # common methods

    def _process_header(
        self,
        token: Optional[str] = None,
        content_type: str = "application/json",
    ):
        return {
            "Authorization": "Bearer " + (token or self.api_token),
            "Content-Type": content_type,
        }

    # Chat methods

    def _process_thread(self, thread: tt.Thread) -> List[Dict[str, Any]]:
        prev_tool_id = tu.get_random_string(5)
        final_messages = []
        for i, m in enumerate(thread.chats):
            if m.role == tt.Message.SYSTEM:
                final_messages.append({"role": "system", "content": m.value})
            elif m.role == tt.Message.HUMAN:
                if isinstance(m.value, str):
                    content = [{"type": "text", "text": m.value}]
                elif isinstance(m.value, list):
                    content = deepcopy(m.value)
                else:
                    raise Exception(
                        f"Unknown message type. Got: '{type(m.value)}', expected 'List[Dict[str, Any]]' or 'str'"
                    )

                for img in m.images:
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img}"},
                        }
                    )
                final_messages.append({"role": "user", "content": content})
            elif m.role == tt.Message.GPT:
                if isinstance(m.value, str):
                    content = [{"type": "text", "text": m.value}]
                elif isinstance(m.value, list):
                    content = deepcopy(m.value)
                else:
                    raise Exception(
                        f"Unknown message type. Got: '{type(m.value)}', expected 'List[Dict[str, Any]]' or 'str'"
                    )

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
        return final_messages

    def _process_input(
        self,
        chats,
        model: Optional[str] = None,
        max_tokens: int = None,
        temperature: Optional[float] = None,
        parallel_tool_calls: bool = False,
        token: Optional[str] = None,
        usage: bool = False,
        extra_headers: Optional[Dict[str, str]] = None,
        debug: bool = False,
        stream: bool = True,
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

        final_messages = self._process_thread(thread)
        headers = self._process_header(token)

        extra_headers = extra_headers or self.extra_headers
        if extra_headers:
            headers.update(extra_headers)
        data = {
            "temperature": temperature,
            "messages": final_messages,
            "model": model or self.model_id,
            "stream": stream,
        }
        if stream:
            data["stream_options"] = {"include_usage": usage}
        if max_tokens:
            data["max_tokens"] = max_tokens
        if isinstance(chats, tt.Thread) and len(chats.tools):
            data["tools"] = [
                {"type": "function", "function": x.to_dict()} for x in chats.tools
            ]
            data["parallel_tool_calls"] = parallel_tool_calls
        if isinstance(chats, tt.Thread) and chats.schema:
            resp_schema = chats.schema.model_json_schema()
            resp_schema["additionalProperties"] = False
            for _, defs in resp_schema.get("$defs", dict()).items():
                defs["additionalProperties"] = False
            data["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "strict": True,
                    "schema": resp_schema,
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

    def _process_output(self, raw: bool, lines_fn: callable, yield_usage: bool = False):
        fn_call = None
        for line in lines_fn():
            if isinstance(line, bytes):
                line = line.decode("utf-8")
            if raw:
                yield line
                continue

            if line.endswith("[DONE]"):
                break

            if line:
                # print(line)
                resp = tu.from_json(line.replace("data: ", ""))
                choices = resp.get("choices", [])
                if len(choices):
                    x = resp["choices"][0]["delta"]
                    if "tool_calls" not in x:
                        if "content" in x:
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
                elif "usage" in resp and yield_usage:
                    usage = resp["usage"]
                    yield tt.Usage(
                        input_tokens=usage.pop("prompt_tokens"),
                        output_tokens=usage.pop("completion_tokens"),
                        cached_tokens=usage["prompt_tokens_details"]["cached_tokens"],
                        **usage,
                    )

        if fn_call:
            fn_call["arguments"] = tu.from_json(fn_call["arguments"])
            yield fn_call

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
        parallel_tool_calls: bool = False,
        **kwargs,
    ):
        headers, data = self._process_input(
            chats=chats,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            parallel_tool_calls=parallel_tool_calls,
            token=token,
            usage=usage,
            extra_headers=extra_headers,
            debug=debug,
            **kwargs,
        )
        if self.client is None:
            self.set_client()

        response = self.client.post(
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

        yield from self._process_output(
            raw=raw,
            lines_fn=response.iter_lines,
            yield_usage=usage,
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
        parallel_tool_calls: bool = False,
        **kwargs,
    ) -> Any:
        output = ""
        usage_obj = None
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
                debug=debug,
                timeout=timeout,
                **kwargs,
            ):
                if isinstance(x, dict):
                    output = x
                elif isinstance(x, tt.Usage):
                    usage_obj = x
                else:
                    output += x
        except httpx.HTTPError as e:
            tu.logger.error(f"API Error: {e.response.text}")
            raise e

        if isinstance(chats, tt.Thread) and chats.schema:
            output = chats.schema(**tu.from_json(output))

        if usage:
            return output, usage_obj
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

    async def stream_chat_async(
        self,
        chats: tt.Thread | str,
        model: Optional[str] = None,
        max_tokens: int = None,
        temperature: float = 1,
        parallel_tool_calls: bool = False,
        token: Optional[str] = None,
        usage: bool = False,
        extra_headers: Optional[Dict[str, str]] = None,
        debug: bool = False,
        raw: bool = False,
        timeout=(5, 60),
        **kwargs,
    ):
        headers, data = self._process_input(
            chats=chats,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            parallel_tool_calls=parallel_tool_calls,
            token=token,
            usage=usage,
            extra_headers=extra_headers,
            debug=debug,
            **kwargs,
        )

        if self.async_client is None:
            self.set_async_client()

        response = await self.async_client.post(
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
                raw=raw,
                lines_fn=chunk.decode("utf-8").splitlines,
                yield_usage=usage,
            ):
                yield x

    async def chat_async(
        self,
        chats: tt.Thread | str,
        model: Optional[str] = None,
        max_tokens: int = None,
        temperature: float = 1,
        parallel_tool_calls: bool = False,
        token: Optional[str] = None,
        usage: bool = False,
        extra_headers: Optional[Dict[str, str]] = None,
        timeout=(5, 60),
        **kwargs,
    ) -> Any:
        output = ""
        usage_obj = None
        try:
            async for x in self.stream_chat_async(
                chats=chats,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                parallel_tool_calls=parallel_tool_calls,
                token=token,
                extra_headers=extra_headers,
                raw=False,
                timeout=timeout,
                **kwargs,
            ):
                if isinstance(x, dict):
                    output = x
                elif isinstance(x, tt.Usage):
                    usage_obj = x
                else:
                    output += x
        except httpx.HTTPError as e:
            tu.logger.error(f"API Error: {e.response.text}")
            raise e

        if isinstance(chats, tt.Thread) and chats.schema:
            output = chats.schema(**tu.from_json(output))

        if usage:
            return output, usage_obj
        return output

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
        model: str,
        token: str,
        extra_headers: Optional[Dict[str, str]],
    ):
        headers = self._process_header(token)
        extra_headers = extra_headers or self.extra_headers
        if extra_headers:
            headers.update(extra_headers)
        if isinstance(chats, tt.Thread):
            text = []
            for i, m in enumerate(chats.chats):
                x = f"<{m.role}> : {m.value}"
                text.append(x)
            text = ["".join(text)]
        elif isinstance(chats, tt.Message):
            text = [chats.value]
        elif isinstance(chats, list) and len(chats) and isinstance(chats[0], str):
            text = chats
        elif isinstance(chats, str):
            text = [chats]
        else:
            raise ValueError(f"Invalid input type. Got {type(chats)}")
        data = {
            "input": text,
            "model": model,
            "encoding_format": "float",
        }
        return headers, data

    def embedding(
        self,
        chats: tt.Thread | List[str] | str,
        model: str = "text-embedding-3-small",
        token: Optional[str] = None,
        raw: bool = False,
        extra_headers: Optional[Dict[str, str]] = None,
        timeout: Tuple[int, int] = (5, 60),
    ) -> tt.EmbeddingGen:
        """If you pass a list then returned items are in the insertion order"""
        if self.emebdding_url is None:
            raise ValueError(
                "Embedding URL is not set. Does this model support embeddings?"
            )

        headers, data = self._prepare_embedding_input(
            chats=chats,
            model=model,
            token=token,
            extra_headers=extra_headers,
        )

        if self.client is None:
            self.set_client()

        r = self.client.post(
            url=self.emebdding_url,
            json=data,
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

        return tt.EmbeddingGen(
            embedding=[
                x["embedding"]
                for x in sorted(
                    r.json()["data"],
                    key=lambda x: x["index"],
                )
            ]
        )

    async def embedding_async(
        self,
        chats: tt.Thread | List[str] | str,
        model: str = "text-embedding-3-small",
        token: Optional[str] = None,
        timeout: Tuple[int, int] = (10, 60),  # Increased connection timeout
        raw: bool = False,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> tt.EmbeddingGen:
        """If you pass a list then returned items are in the insertion order"""
        if self.emebdding_url is None:
            raise ValueError(
                "Embedding URL is not set. Does this model support embeddings?"
            )

        headers, data = self._prepare_embedding_input(
            chats=chats,
            model=model,
            token=token,
            extra_headers=extra_headers,
        )

        if self.async_client is None:
            self.set_async_client()

        r = await self.async_client.post(
            url=self.emebdding_url,
            json=data,
            headers=headers,
            timeout=timeout,
        )
        try:
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            print(e.response.text)
            raise e
        except Exception as e:
            print(f"An unexpected error occured: {e}")
            raise e

        resp = ""
        async for chunk in r.aiter_bytes():
            resp += chunk.decode("utf-8")

        if raw:
            return tu.from_json(resp)

        return tt.EmbeddingGen(
            embedding=[
                x["embedding"]
                for x in sorted(
                    r.json()["data"],
                    key=lambda x: x["index"],
                )
            ]
        )

    # Image methods

    def _prepare_image_gen_input(
        self,
        prompt: str,
        style: str,
        model: str,
        n: int,
        size: str,
        extra_headers,
        **kwargs,
    ):
        assert size in [
            "1024x1024",
            "1792x1024",
            "1024x1792",
        ], "Only these size are allowed: https://platform.openai.com/docs/api-reference/images/create"
        data = {
            "model": model,
            "prompt": prompt,
            "n": n,
            "size": size,
            "style": style,
            "response_format": "url",
        }
        if kwargs:
            data.update(kwargs)
        headers = self._process_header()
        if extra_headers:
            headers.update(extra_headers)
        return headers, data

    def image_gen(
        self,
        prompt: str,
        style: str = "natural",
        model: str = "dall-e-3",
        n: int = 1,
        size: str = "1024x1024",
        extra_headers: Optional[Dict[str, str]] = None,
        timeout: Tuple[int, int] = (5, 60),
        **kwargs,
    ) -> tt.ImageGen:

        if self.image_gen_url is None:
            raise ValueError(
                "Image Generation URL is not set. Does this model support image generation?"
            )

        headers, data = self._prepare_image_gen_input(
            prompt=prompt,
            style=style,
            model=model,
            n=n,
            size=size,
            extra_headers=extra_headers,
            **kwargs,
        )

        if self.client is None:
            self.set_client()

        r = self.client.post(
            url=self.image_gen_url,
            json=data,
            headers=headers,
            timeout=timeout,
        )
        try:
            r.raise_for_status()
        except Exception as e:
            tu.logger.error(f"Cannot generate image: {r.text}")
            raise e
        out = r.json()

        img_r = self.client.get(out["data"][0]["url"])
        try:
            img_r.raise_for_status()
        except httpx.HTTPError as e:
            print(e.response.text)
            raise e
        except Exception as e:
            tu.logger.error(f"Cannot fetch image: {r.text}")
            raise e
        cont = BytesIO(img_r.content)
        return tt.ImageGen(
            image=Image.open(cont),
        )

    async def image_gen_async(
        self,
        prompt: str,
        style: str = "natural",
        model: str = "dall-e-3",
        n: int = 1,
        size: str = "1024x1024",
        extra_headers: Optional[Dict[str, str]] = None,
        timeout: Tuple[int, int] = (5, 60),
        **kwargs,
    ) -> tt.ImageGen:
        if self.image_gen_url is None:
            raise ValueError(
                "Image Generation URL is not set. Does this model support image generation?"
            )

        headers, data = self._prepare_image_gen_input(
            prompt=prompt,
            style=style,
            model=model,
            n=n,
            size=size,
            extra_headers=extra_headers,
            **kwargs,
        )

        if self.async_client is None:
            self.set_async_client()

        # generate image
        r = await self.async_client.post(
            url=self.image_gen_url,
            json=data,
            headers=headers,
            timeout=timeout,
        )
        try:
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            tu.logger.error(f"Cannot generate image: {e.response.text}")
            raise e
        out = r.json()

        # now fetch the image
        img_r = await self.async_client.get(out["data"][0]["url"])
        try:
            img_r.raise_for_status()
        except httpx.HTTPStatusError as e:
            tu.logger.error(f"Cannot fetch image: {e.response.text}")
            raise e
        cont = BytesIO(img_r.content)
        return tt.ImageGen(
            image=Image.open(cont),
        )

    # Audio methods

    def speech_to_text(
        self,
        prompt: str,
        audio: str,
        model="whisper-1",
        timestamp_granularities=["segment"],
        token: Optional[str] = None,
        timeout: Tuple[int, int] = (5, 300),
        **kwargs,
    ) -> tt.Transcript:
        if self.audio_transcribe_url is None:
            raise ValueError(
                "Audio Transcription URL is not set. Does this model support audio transcription?"
            )

        headers = self._process_header(token=token)
        headers.pop("Content-Type")

        if not (isinstance(audio, str) and os.path.exists(audio)):
            raise ValueError("Invalid audio file path")

        # max file size can be 25 MB
        if tu.file_size(audio) / (1024 * 1024) > 25:
            raise ValueError("Audio file size should be less than 25 MB")

        if self.client is None:
            self.set_client()

        with open(audio, "rb") as f:
            files = {"file": (os.path.basename(audio), f)}
            data = {
                "model": model,
                "prompt": prompt,
                "response_format": "vtt",
                "timestamp_granularities": timestamp_granularities,
            }
            if kwargs:
                data.update(kwargs)
            r = self.client.post(
                url=self.audio_transcribe_url,
                headers=headers,
                files=files,
                data=data,
                timeout=timeout,
            )

        try:
            r.raise_for_status()
        except Exception as e:
            tu.logger.error(f"Could not transcribe audio: {r.text}")
            raise e

        return tt.get_transcript(text=r.content.decode("utf-8"))

    async def speech_to_text_async(
        self,
        prompt: str,
        audio: str,
        model="whisper-1",
        timestamp_granularities=["segment"],
        token: Optional[str] = None,
        timeout: Tuple[int, int] = (5, 300),
        **kwargs,
    ) -> tt.Transcript:
        if self.audio_transcribe_url is None:
            raise ValueError(
                "Audio Transcription URL is not set. Does this model support audio transcription?"
            )

        headers = self._process_header(token=token)
        headers.pop("Content-Type")

        if not (isinstance(audio, str) and os.path.exists(audio)):
            raise ValueError("Invalid audio file path")

        # max file size can be 25 MB
        if tu.file_size(audio) / (1024 * 1024) > 25:
            raise ValueError("Audio file size should be less than 25 MB")

        if self.async_client is None:
            self.set_async_client()

        async with aiofiles.open(audio, mode="rb") as f:
            files = {"file": (os.path.basename(audio), await f.read())}
            data = {
                "model": model,
                "prompt": prompt,
                "response_format": "vtt",
                "timestamp_granularities": timestamp_granularities,
            }
            if kwargs:
                data.update(kwargs)

            r = await self.async_client.post(
                url=self.audio_transcribe_url,
                headers=headers,
                files=files,
                data=data,
            )

        try:
            r.raise_for_status()
        except Exception as e:
            tu.logger.error(f"Could not transcribe audio: {r.text}")
            raise e

        return tt.get_transcript(text=r.content.decode("utf-8"))

    def _prepare_audio_gen_input(
        self,
        prompt: str,
        voice: str,
        model: str,
        response_format: str,
        extra_headers,
        **kwargs,
    ):
        data = {
            "model": model,
            "input": prompt,
            "voice": voice,
            "response_format": response_format,
        }
        if kwargs:
            data.update(kwargs)
        headers = self._process_header()
        if extra_headers:
            headers.update(extra_headers)
        return headers, data

    def text_to_speech(
        self,
        prompt: str,
        voice: str = "shimmer",
        model="tts-1",
        response_format="wav",
        extra_headers: Optional[Dict[str, str]] = None,
        timeout: Tuple[int, int] = (5, 60),
        **kwargs,
    ) -> bytes:

        if self.audio_gen_url is None:
            raise ValueError(
                "Audio Generation URL is not set. Does this model support audio generation?"
            )

        headers, data = self._prepare_audio_gen_input(
            prompt=prompt,
            voice=voice,
            model=model,
            response_format=response_format,
            extra_headers=extra_headers,
            **kwargs,
        )

        if self.client is None:
            self.set_client()

        r = self.client.post(
            url=self.audio_gen_url,
            json=data,
            headers=headers,
            timeout=timeout,
        )
        try:
            r.raise_for_status()
        except httpx.HTTPError as e:
            tu.logger.error(f"Cannot to text to speech: {e.response.text}")
            raise e
        except Exception as e:
            tu.logger.error(f"An unexpected error occured: {e}")
            raise e

        return r.content

    async def text_to_speech_async(
        self,
        prompt: str,
        voice: str = "shimmer",
        model="tts-1",
        response_format="wav",
        extra_headers: Optional[Dict[str, str]] = None,
        timeout: Tuple[int, int] = (5, 60),
        **kwargs,
    ) -> bytes:

        if self.audio_gen_url is None:
            raise ValueError(
                "Audio Generation URL is not set. Does this model support audio generation?"
            )

        headers, data = self._prepare_audio_gen_input(
            prompt=prompt,
            voice=voice,
            model=model,
            response_format=response_format,
            extra_headers=extra_headers,
            **kwargs,
        )

        if self.async_client is None:
            self.set_async_client()

        r = await self.async_client.post(
            url=self.audio_gen_url,
            json=data,
            headers=headers,
            timeout=timeout,
        )
        try:
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            tu.logger.error(f"Cannot to text to speech: {e.response.text}")
            raise e
        except Exception as e:
            tu.logger.error(f"An unexpected error occured: {e}")
            raise e

        return r.content

    # Submit batches

    def submit_batch(
        self,
        threads: List[tt.Thread | str],
        model: Optional[str] = None,
        endpoint: str = "/v1/chat/completions",
        max_tokens: int = None,
        temperature: float = 1,
        parallel_tool_calls: bool = False,
        token: Optional[str] = None,
        usage: bool = False,
        extra_headers: Optional[Dict[str, str]] = None,
        debug: bool = False,
        timeout=(5, 60),
        raw: bool = False,
        verbose: bool = False,
        **kwargs,
    ) -> Tuple[str, List[str]] | Dict:
        if self.batch_url is None:
            raise ValueError("Batch URL is not set. Does this model support batches?")

        headers = self._process_header(token)
        headers.pop("Content-Type")

        # create the bodies
        bodies = []
        custom_ids = []
        for chats in threads:
            _, data = self._process_input(
                chats=chats,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                parallel_tool_calls=parallel_tool_calls,
                token=token,
                usage=usage,
                extra_headers=extra_headers,
                debug=debug,
                stream=False,
                **kwargs,
            )
            custom_id = "tuneapi_" + tu.get_random_string(10)
            custom_ids.append(custom_id)
            bodies.append(
                {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": data,
                }
            )

        # upload the file
        fp = f"./tuneapi_batch_{tu.SimplerTimes.get_now_float()}.jsonl"
        if verbose:
            tu.logger.info("Creating temporary file at " + fp)
        tu.to_json(bodies, fp)

        if self.client is None:
            self.set_client()

        with open(fp, "rb") as f:
            files = {"file": (os.path.basename(fp), f)}
            data = {"purpose": "batch"}
            response = self.client.post(
                url=self.files_url,
                headers=headers,
                files=files,
                data=data,
                timeout=timeout,
            )

        try:
            response.raise_for_status()
        except Exception as e:
            tu.logger.error(f"Could not upload file: {response.text}")
            raise e
        finally:
            if verbose:
                tu.logger.info("Removing temporary JSONL file")
            os.remove(fp)

        data = response.json()
        file_id = data["id"]

        # Now submit the batch
        body = {
            "input_file_id": file_id,
            "endpoint": endpoint,
            "completion_window": "24h",
        }
        headers = self._process_header(token)
        if debug:
            tu.logger.info("Saving batch at path batch_oai.json")
            tu.to_json(body, fp="batch_oai.json")

        if self.client is None:
            self.set_client()

        r = self.client.post(
            url=self.batch_url,
            headers=headers,
            json=body,
            timeout=timeout,
        )
        try:
            r.raise_for_status()
        except Exception as e:
            tu.logger.error(f"Coudn't submit batch: {r.text}")
            raise e

        resp = r.json()
        if raw:
            return resp
        return resp["id"], custom_ids

    def get_batch(
        self,
        batch_id: str,
        custom_ids: Optional[List[str]] = None,
        usage: bool = False,
        token: Optional[str] = None,
        raw: bool = False,
        verbose: bool = False,
    ):
        if self.batch_url is None:
            raise ValueError("Batch URL is not set. Does this model support batches?")

        if self.client is None:
            self.set_client()

        headers = self._process_header(token)
        r = self.client.get(self.batch_url + "/" + batch_id, headers=headers)
        try:
            r.raise_for_status()
        except Exception as e:
            tu.logger.error(f"Coudn't get batch: {r.text}")
            raise e
        resp = r.json()

        if resp["status"] in ["failed", "expired", "cancelled"]:
            if verbose:
                tu.logger.info(
                    f"Batch {batch_id} has ended unsuccesfully. Status: {resp['processing_status']}"
                )
            return None, resp["status"]
        elif resp["status"] != "completed":
            if verbose:
                tu.logger.info(
                    f"Batch {batch_id} has not ended. Status: {resp['processing_status']}"
                )
            return None, resp["status"]

        # get the results
        results_file = resp["output_file_id"]
        r = self.client.get(
            self.files_url + "/" + results_file + "/content",
            headers=headers,
        )
        try:
            r.raise_for_status()
        except Exception as e:
            tu.logger.error(f"Coudn't get batch results: {r.text}")
            raise e

        output = []
        for line in r.iter_lines():
            if not line:
                continue
            output.append(tu.from_json(line))

        if custom_ids:
            # each item in output has a key called "custom_id" sort on the basis of incoming custom_ids
            output = sorted(output, key=lambda x: custom_ids.index(x["custom_id"]))

        if raw:
            return output, None

        _usage = tt.Usage(0, 0)
        for o in output:
            u = o["response"]["body"]["usage"]
            _usage += tt.Usage(
                input_tokens=u.pop("prompt_tokens"),
                output_tokens=u.pop("completion_tokens"),
                cached_tokens=u["prompt_tokens_details"]["cached_tokens"],
                **u,
            )

        parsed_output = [o["response"]["body"]["choices"][0]["message"] for o in output]
        final_output = []
        for o in parsed_output:
            if "tool_calls" in o:
                final_output.append(o["tool_calls"][0]["function"])
            elif "content" in o:
                final_output.append(o["content"])

        if usage:
            return final_output, None, _usage
        return final_output, None


# Other OpenAI compatible models


class Openai(OpenAIProtocol):
    def __init__(
        self,
        id: str = "gpt-4o",
        base_url: str = "https://api.openai.com/v1/chat/completions",
        extra_headers: Optional[Dict[str, str]] = None,
        api_token: Optional[str] = None,
        emebdding_url: Optional[str] = None,
        image_gen_url: Optional[str] = None,
        audio_transcribe: Optional[str] = None,
        audio_gen_url: Optional[str] = None,
        batch_url: Optional[str] = None,
        files_url: Optional[str] = None,
    ):
        super().__init__(
            id=id,
            base_url=base_url,
            api_token=api_token or tu.ENV.OPENAI_TOKEN(""),
            extra_headers=extra_headers,
            emebdding_url=emebdding_url
            or base_url.replace(
                "/chat/completions",
                "/embeddings",
            ),
            image_gen_url=image_gen_url
            or base_url.replace(
                "/chat/completions",
                "/images/generations",
            ),
            audio_transcribe_url=audio_transcribe
            or base_url.replace(
                "/chat/completions",
                "/audio/transcriptions",
            ),
            audio_gen_url=audio_gen_url
            or base_url.replace(
                "/chat/completions",
                "/audio/speech",
            ),
            batch_url=batch_url
            or base_url.replace(
                "/chat/completions",
                "/batches",
            ),
            files_url=files_url
            or base_url.replace(
                "/chat/completions",
                "/files",
            ),
        )


class Mistral(OpenAIProtocol):
    """
    A class to interact with Mistral's Large Language Models (LLMs) via their API. Note this class does not contain the
    `embedding` method.

    Attributes:
        id (str): Identifier for the Mistral model.
        base_url (str): The base URL for the Mistral API. Defaults to "https://api.mistral.ai/v1/chat/completions".
        extra_headers (Optional[Dict[str, str]]): Additional headers to include in API requests.
        api_token (Optional[str]): API token for authenticating requests. If not provided, it will use the token from the environment variable MISTRAL_TOKEN.

    Methods:
        embedding(*a, **k): Raises NotImplementedError as Mistral does not support embeddings.

    Note:
        For more information, visit the Mistral API documentation at https://console.mistral.ai/
    """

    def __init__(
        self,
        id: str = "mistral-small-latest",
        base_url: str = "https://api.mistral.ai/v1/chat/completions",
        extra_headers: Optional[Dict[str, str]] = None,
        api_token: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            id=id,
            base_url=base_url,
            extra_headers=extra_headers,
            api_token=api_token or tu.ENV.MISTRAL_TOKEN(),
            emebdding_url=None,
            image_gen_url=None,
            audio_transcribe_url=None,
            audio_gen_url=None,
            batch_url=None,
            files_url=None,
        )

    def embedding(*a, **k):
        raise NotImplementedError("Mistral does not support embeddings")


class Groq(OpenAIProtocol):
    """
    A class to interact with Groq's Large Language Models (LLMs) via their API. Note this class does not contain the
    `embedding` method.

    Attributes:
        id (str): Identifier for the Mistral model.
        base_url (str): The base URL for the Mistral API. Defaults to "https://api.groq.com/openai/v1/chat/completions".
        extra_headers (Optional[Dict[str, str]]): Additional headers to include in API requests.
        api_token (Optional[str]): API token for authenticating requests. If not provided, it will use the token from the environment variable MISTRAL_TOKEN.

    Note:
        For more information, visit the Mistral API documentation at https://console.groq.com/
    """

    def __init__(
        self,
        id: str = "llama3-70b-8192",
        base_url: str = "https://api.groq.com/openai/v1/chat/completions",
        extra_headers: Optional[Dict[str, str]] = None,
        api_token: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            id=id,
            base_url=base_url,
            extra_headers=extra_headers,
            api_token=api_token or tu.ENV.GROQ_TOKEN(),
            emebdding_url=None,
            image_gen_url=None,
            audio_transcribe_url=None,
            audio_gen_url=None,
            batch_url=None,
            files_url=None,
        )

    def embedding(*a, **k):
        raise NotImplementedError("Groq does not support embeddings")


class TuneModel(OpenAIProtocol):
    """
    A class to interact with Groq's Large Language Models (LLMs) via their API.

    Attributes:
        id (str): Identifier for the Mistral model.
        base_url (str): The base URL for the Mistral API. Defaults to "https://proxy.tune.app/chat/completions".
        org_id (Optional[str]): Organization ID for the Tune API.
        extra_headers (Optional[Dict[str, str]]): Additional headers to include in API requests.
        api_token (Optional[str]): API token for authenticating requests. If not provided, it will use the token from the environment variable MISTRAL_TOKEN.

    Note:
        For more information, visit the Mistral API documentation at https://tune.app/
    """

    def __init__(
        self,
        id: str = "meta/llama-3.1-8b-instruct",
        base_url: str = "https://proxy.tune.app/chat/completions",
        org_id: Optional[str] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        api_token: Optional[str] = None,
        **kwargs,
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
            emebdding_url="https://proxy.tune.app/v1/embeddings",
            image_gen_url=None,
            audio_transcribe_url=None,
            audio_gen_url=None,
            batch_url=None,
            files_url=None,
        )

    def embedding(
        self,
        chats: tt.Thread | List[str] | str,
        model: str = "openai/text-embedding-3-small",
        token: Optional[str] = None,
        timeout: Tuple[int, int] = (5, 60),
        raw: bool = False,
        extra_headers: Optional[Dict[str, str]] = None,
    ):
        return super().embedding(
            chats=chats,
            model=model,
            token=token,
            timeout=timeout,
            raw=raw,
            extra_headers=extra_headers,
        )

    async def embedding_async(
        self,
        chats: tt.Thread | List[str] | str,
        model: str = "openai/text-embedding-3-small",
        token: Optional[str] = None,
        timeout: Tuple[int, int] = (5, 60),
        raw: bool = False,
        extra_headers: Optional[Dict[str, str]] = None,
    ):
        return await super().embedding_async(
            chats=chats,
            model=model,
            token=token,
            timeout=timeout,
            raw=raw,
            extra_headers=extra_headers,
        )


class Ollama(OpenAIProtocol):
    def __init__(
        self,
        id: str,
        base_url: str = "http://localhost:11434/v1/chat/completions",
        **kwargs,
    ):
        super().__init__(
            id=id,
            base_url=base_url,
            api_token="ollama",
            extra_headers={},
            emebdding_url=None,
            image_gen_url=None,
            audio_transcribe_url=None,
            audio_gen_url=None,
            batch_url=None,
            files_url=None,
        )
