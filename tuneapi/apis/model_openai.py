"""
A single OpenAI-protocol wire. All model access goes through an ``openai.OpenAI`` or
``openai.AsyncOpenAI`` client object -- this module never builds URLs or headers itself.
"""

# Copyright © 2024-2025 Frello Technology Private Limited
# Copyright © 2025- Yash Bonde github.com/yashbonde
# MIT License

import asyncio
import inspect
from copy import deepcopy
from typing import Any, AsyncIterator

import tuneapi.utils as tu
import tuneapi.types as tt

__all__ = ["OpenAIProtocol", "Price", "get_prices", "clear_price_cache"]


class Price(tt.BM):
    """Dollars per million tokens, which is how everyone quotes them."""

    input: float = 0.0
    cached: float = 0.0
    output: float = 0.0

    def is_empty(self) -> bool:
        return not (self.input or self.cached or self.output)


# One entry per API host, not per client and not per kernel: a server running a
# hundred kernels against the same endpoint fetches the price list once. The
# lock makes that literal -- a hundred kernels starting at the same moment queue
# on it and only the first one goes out to the network.
_PRICES: dict[str, dict[str, Price]] = {}
_PRICE_LOCKS: dict[str, asyncio.Lock] = {}


def clear_price_cache(base_url: str | None = None) -> None:
    """Forget cached prices, for the whole process or one host. Prices move
    rarely enough that nothing expires them on its own."""
    if base_url is None:
        _PRICES.clear()
    else:
        _PRICES.pop(str(base_url).rstrip("/"), None)


async def get_prices(client) -> dict[str, Price]:
    """``GET /models`` once per host, then serve everyone from memory.

    Only providers that publish pricing on that endpoint (OpenRouter does,
    OpenAI does not) come back with anything; for the rest this is an empty
    map and costs read 0.0.
    """
    host = str(getattr(client, "base_url", "")).rstrip("/")
    if host in _PRICES:
        return _PRICES[host]

    lock = _PRICE_LOCKS.setdefault(host, asyncio.Lock())
    async with lock:
        if host in _PRICES:  # someone else fetched while we waited
            return _PRICES[host]

        prices: dict[str, Price] = {}
        try:
            page = await client.models.list()
            for model in page.data:
                raw = getattr(model, "pricing", None)
                if raw is None:
                    raw = (getattr(model, "model_extra", None) or {}).get("pricing")
                if not raw:
                    continue
                raw = raw if isinstance(raw, dict) else raw.model_dump()
                # the endpoint quotes dollars per token, as strings
                prices[model.id] = Price(
                    input=_per_million(raw.get("prompt")),
                    cached=_per_million(raw.get("input_cache_read")),
                    output=_per_million(raw.get("completion")),
                )
        except Exception as e:
            tu.logger.warning(f"could not fetch prices from {host}: {e}")

        _PRICES[host] = prices
        return prices


def _per_million(value) -> float:
    try:
        return float(value) * 1e6
    except (TypeError, ValueError):
        return 0.0


class OpenAIProtocol(tt.ModelInterface):
    """Thin adapter over an ``openai`` client speaking the chat-completions protocol."""

    def __init__(self, client, model_id: str = "", **default_kwargs):
        """``client`` is an ``openai.AsyncOpenAI`` instance -- we never build URLs or headers
        ourselves anymore, the client owns transport, auth and retries. The wire is async only."""
        super().__init__()
        try:
            import openai as _openai

            if isinstance(client, _openai.OpenAI):
                raise TypeError(
                    "OpenAIProtocol is async only: pass an `openai.AsyncOpenAI` client, "
                    "not a sync `openai.OpenAI` one."
                )
        except ImportError:  # pragma: no cover - openai is a hard dep
            pass
        self.client = client
        self.model_id = model_id
        self.default_kwargs = default_kwargs

    async def get_price(self, model_id: str = "") -> "Price":
        """Per-million-token prices for a model, from the process-wide cache."""
        prices = await get_prices(self.client)
        return prices.get(model_id or self.model_id) or Price()

    # ---------------------------------------------------------------- inputs

    def _process_thread(self, thread: tt.Thread) -> list[dict[str, Any]]:
        prev_tool_id = tu.get_random_string(5)
        final_messages = []
        for i, m in enumerate(thread.chats):
            if m.role == tt.Message.SYSTEM:
                if i != 0:
                    raise ValueError(
                        "Only the first message in thread can be the system message."
                    )
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
                # Audio rides only on the user turn. ``input_audio`` is an
                # input part: what a model sends *back* as audio is a
                # different shape entirely, and putting one here would be
                # describing a reply in the vocabulary of a request.
                for clip in m.audio:
                    content.append({"type": "input_audio", "input_audio": clip})
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

        if final_messages[0]["role"] == "system" and thread.tools:
            tool_prompt = "# Tool Usage Instructions\n\n"
            for tool in thread.tools:
                if tool.system:
                    tool_prompt += f"{tool.system}\n"
            system = final_messages[0]["content"]
            if isinstance(system, list):
                system.append(
                    {
                        "type": "text",
                        "text": tool_prompt,
                    }
                )
            else:
                system += tool_prompt
            final_messages[0]["content"] = system

        return final_messages

    def _process_input(
        self,
        thread: tt.Thread,
        *,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        parallel_tool_calls: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        """Build the kwargs dict for ``client.chat.completions.create(...)``."""
        if isinstance(thread, str):
            thread = tt.Thread(tt.human(thread))
        elif not isinstance(thread, tt.Thread):
            raise Exception(f"Invalid input: {type(thread)}")

        model_id = model or self.model_id
        data: dict[str, Any] = {
            "messages": self._process_thread(thread),
            "model": model_id,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        is_reasoning_family = (
            model_id.startswith("gpt-5")
            or model_id.startswith("o4-")
            or model_id.startswith("o3-")
            or model_id.startswith("o1-")
        )
        if not is_reasoning_family and temperature is not None:
            data["temperature"] = temperature
        if max_tokens:
            data["max_tokens"] = max_tokens

        if thread.tools:
            data["tools"] = [
                {"type": "function", "function": x.to_dict()} for x in thread.tools
            ]
            if not (
                model_id.startswith("o4-")
                or model_id.startswith("o3-")
                or model_id.startswith("o1-")
            ):
                data["parallel_tool_calls"] = parallel_tool_calls

        if thread.schema:
            resp_schema = thread.schema.model_json_schema()
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

        for k, v in self.default_kwargs.items():
            data.setdefault(k, v)
        if kwargs:
            data.update(kwargs)
        return data

    # ------------------------------------------------------------- transport

    async def _achunks(self, data: dict[str, Any]) -> AsyncIterator[Any]:
        """Yield raw chunks from the async openai client."""
        stream = self.client.chat.completions.create(**data)
        if inspect.isawaitable(stream):
            stream = await stream
        async for chunk in stream:
            yield chunk

    # ---------------------------------------------------------------- stream

    async def stream(
        self,
        thread: tt.Thread,
        *,
        request: dict[str, Any] | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        parallel_tool_calls: bool = False,
        **kwargs,
    ) -> AsyncIterator[str | tt.ReasoningBlock | tt.ToolCall | tt.Usage]:
        """Yield, in arrival order: text deltas (``str``), complete ``ReasoningBlock`` s,
        complete ``ToolCall`` s and finally ``Usage`` if the provider sent it.

        If ``request`` is given it is sent verbatim to the client (built earlier via
        :meth:`_process_input` and possibly mutated); ``thread`` is still used to resolve
        streamed tool calls back to ``Tool`` objects."""
        if request is not None:
            # sent verbatim; the caller (kernel) owns the payload
            data = request
        else:
            data = self._process_input(
                thread,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                parallel_tool_calls=parallel_tool_calls,
                **kwargs,
            )

        tools = list(getattr(thread, "tools", None) or [])
        usage_obj = None
        reasoning_buf: str = ""
        reasoning_idx: int = 0
        # index -> {"name": str, "arguments": str}
        tool_frags: dict[int, dict[str, str]] = {}
        tool_order: list[int] = []

        def _flush_reasoning():
            nonlocal reasoning_buf
            if reasoning_buf:
                block = tt.ReasoningBlock(text=reasoning_buf, index=reasoning_idx)
                reasoning_buf = ""
                return block
            return None

        def _build_tool_call(frag: dict[str, str]) -> tt.ToolCall:
            name = frag.get("name") or ""
            tool_fn = [t for t in tools if t.name == name]
            if not tool_fn:
                raise ValueError(f"Tool '{name}' not found in thread.tools")
            raw_args = frag.get("arguments") or "{}"
            try:
                args = tu.from_json(raw_args) if isinstance(raw_args, str) else raw_args
            except Exception as e:
                tu.logger.error(f"Could not parse tool arguments: {raw_args}")
                raise e
            return tt.ToolCall(tool=tool_fn[0], arguments=args)

        async for chunk in self._achunks(data):
            usage = getattr(chunk, "usage", None)
            if usage is not None:
                usage_obj = self._to_usage(usage, data.get("model", self.model_id))

            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            delta = getattr(choices[0], "delta", None)
            if delta is None:
                continue

            # reasoning: OpenRouter uses `reasoning`, others `reasoning_content`
            rtext = getattr(delta, "reasoning", None)
            if rtext is None:
                rtext = getattr(delta, "reasoning_content", None)
            ridx = getattr(delta, "reasoning_index", None) or 0
            if rtext:
                if reasoning_buf and ridx != reasoning_idx:
                    block = _flush_reasoning()
                    if block is not None:
                        yield block
                reasoning_idx = ridx
                reasoning_buf += rtext

            content = getattr(delta, "content", None)
            tool_calls = getattr(delta, "tool_calls", None)

            if (content or tool_calls) and reasoning_buf:
                block = _flush_reasoning()
                if block is not None:
                    yield block

            if content:
                yield content

            for tc in tool_calls or []:
                idx = getattr(tc, "index", None)
                if idx is None:
                    idx = len(tool_order)
                fn = getattr(tc, "function", None)
                if idx not in tool_frags:
                    tool_frags[idx] = {"name": "", "arguments": ""}
                    tool_order.append(idx)
                if fn is not None:
                    if getattr(fn, "name", None):
                        tool_frags[idx]["name"] += fn.name
                    if getattr(fn, "arguments", None):
                        tool_frags[idx]["arguments"] += fn.arguments

        block = _flush_reasoning()
        if block is not None:
            yield block

        for idx in tool_order:
            yield _build_tool_call(tool_frags[idx])

        if usage_obj is not None:
            yield usage_obj

    def _to_usage(self, usage, model_id: str) -> tt.Usage:
        def _get(obj, key, default=0):
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default) or default

        details = _get(usage, "prompt_tokens_details", None)
        cached = _get(details, "cached_tokens", 0) if details else 0
        return tt.Usage(
            input_tokens=_get(usage, "prompt_tokens", 0),
            output_tokens=_get(usage, "completion_tokens", 0),
            cached_tokens=cached,
            model=model_id,
        )

    # ------------------------------------------------------------------- run

    async def run(self, thread: tt.Thread, **kwargs) -> tt.ChatResponse:
        """Drains ``stream()`` and returns the assembled response."""
        parts: list[Any] = []
        reasoning: list[tt.ReasoningBlock] = []
        usage_obj = None
        async for x in self.stream(thread, **kwargs):
            if isinstance(x, tt.ReasoningBlock):
                reasoning.append(x)
            elif isinstance(x, tt.Usage):
                usage_obj = x
            elif isinstance(x, tt.ToolCall):
                parts.append(x)
            else:
                parts.append(x)

        structured = None
        schema = getattr(thread, "schema", None) if not isinstance(thread, str) else None
        if schema and len(parts) == 1 and isinstance(parts[0], str):
            try:
                structured = schema(**tu.from_json(parts[0]))
            except Exception as e:
                tu.logger.error(f"Error loading schema: {parts[0]}")
                raise e

        return tt.ChatResponse(
            parts=parts,
            reasoning=reasoning,
            usage=usage_obj,
            structured=structured,
        )
