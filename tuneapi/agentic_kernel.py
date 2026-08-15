"""
The agentic kernel: a small state machine that drives a model through tool calls.

The loop, in the vocabulary of ``tuneapi.types.events``::

    idle ──UserPromptSubmit──▶ context assembled ──LLMStep──▶ llm step in flight
                                      ▲                             │
                                      │                          LLMOutput
                              PostToolUse(Failure)                  │
                                      │                             ▼
                                 tool execution ◀──PreToolUse── response received
                                                                    │
                                     idle ◀──AssistantChatResponse──┘
                                              (no tool calls left)

``LLMFailed`` and ``Interrupt`` also return the kernel to idle, but emit no
``AssistantChatResponse`` -- a caller awaiting only that event will hang on
those two paths.

Everything here is async. The kernel awaits its subscribers, the wire, and the
tools; the only sync things left are user-supplied sync tools, which
``ToolCall.run_async`` hands off to a worker thread.
"""

# Copyright © 2025- Yash Bonde github.com/yashbonde
# MIT License

from __future__ import annotations

import os
import time
import inspect
import contextlib
from pathlib import Path
from typing import Any, Callable

import tuneapi.utils as tu
import tuneapi.types as tt
from tuneapi.types import events as ev


COMPACT_PROMPT = """\
Summarise the conversation so far. Keep every decision taken, every file or \
path touched, every tool result that later steps still depend on, and anything \
the user asked for that is not done yet. Drop pleasantries and superseded \
attempts. Write it as notes to yourself, not as a report to the user."""


class KernelConfig(tt.BM):
    """Everything the kernel needs that is not the wire or the subscribers."""

    model_config = {"arbitrary_types_allowed": True, "protected_namespaces": ()}

    model_id: str
    base_folder: str = ""
    """All tool calls run with this as the working directory. Defaults to the
    folder the main code is running in."""

    tools: list[tt.Tool] = []
    system_prompt: str = ""

    max_turns: int = 20
    """Hard stop on the tool loop, so a model that keeps calling tools cannot
    run forever."""

    temperature: float | None = None
    max_tokens: int | None = None

    compact_at_tokens: int = 0
    """Input tokens after which the next step compacts first. 0 disables it."""

    compact_keep_last: int = 4
    """Messages held out of the summary and kept verbatim."""

    # Only used to fill in cost_usd on the events. Left at 0, the kernel asks
    # the wire for the model's published price at start(); set them to override
    # that (a negotiated rate, or a provider that publishes nothing).
    price_in_per_m: float = 0.0
    price_cached_per_m: float = 0.0
    price_out_per_m: float = 0.0


class _ChatRollup:
    """Mutable running totals for one chat, folded into AssistantChatResponse."""

    def __init__(self):
        self.t0 = time.time()
        self.n_turns = 0
        self.n_llm_steps = 0
        self.n_tool_calls = 0
        self.n_tool_failures = 0
        self.tokens_in = 0
        self.tokens_out = 0
        self.cached_tokens = 0
        self.cost_usd = 0.0
        self.compacted = False
        self.last_input_tokens = 0


class Kernel:
    """The state machine. One kernel is one transcript.

    Args:
        config: model, tools, folder and loop bounds.
        wire: anything implementing ``tuneapi.types.ModelInterface`` -- i.e.
            ``async stream(thread, request=...)`` and ``async run(thread)``.
        event_subscribers: ``{event: fn}`` where the key is an ``EventKind``, a
            payload class (``events.LLMStep``) or its name, and the value is one
            callable or a list of them. Each is called with the ``Event`` and may
            mutate ``event.body`` in place -- that is how guardrails redact an
            outbound request before it is sent.
    """

    def __init__(
        self,
        config: KernelConfig,
        wire: tt.ModelInterface,
        event_subscribers: dict[Any, Callable | list[Callable]] | None = None,
    ):
        self.config = config
        self.wire = wire
        self.thread = tt.Thread(tools=list(config.tools))

        self.session_id = "sess_" + str(tu.get_snowflake())
        self.base_folder = Path(config.base_folder or os.getcwd()).resolve()

        self._subs: dict[ev.EventKind, list[Callable]] = {}
        for key, fns in (event_subscribers or {}).items():
            kind = ev.resolve_kind(key)
            if not isinstance(fns, (list, tuple)):
                fns = [fns]
            self._subs.setdefault(kind, []).extend(fns)

        self._seq = 0
        self._started = False
        self._stopped = False
        self._interrupt: str | None = None
        self._chat_id: str | None = None
        self._turn_id: str | None = None
        self._step_id: str | None = None
        self._n_chats = 0
        self._totals = _ChatRollup()
        self._price = (0.0, 0.0, 0.0)

    def __repr__(self) -> str:
        return f"<Kernel '{self.config.model_id}' {self.session_id} @ {self.base_folder}>"

    # ------------------------------------------------------------------ #
    # session lifecycle
    # ------------------------------------------------------------------ #

    async def start(self) -> "Kernel":
        """Open the session. Must be awaited before ``chat``."""
        if self._started:
            return self
        self._started = True
        self._totals = _ChatRollup()
        await self._resolve_price()
        if self.config.system_prompt:
            self.thread.append(tt.system(self.config.system_prompt))
        await self._emit(
            ev.EventKind.SESSION_START,
            ev.SessionStart(
                model=self.config.model_id,
                system_prompt=self.config.system_prompt,
                tools=[t.name for t in self.config.tools],
                cwd=str(self.base_folder),
                config=self.config.model_dump(exclude={"tools"}),
            ),
        )
        return self

    async def stop(self, reason: str = "user") -> "Kernel":
        """Close the session gracefully. Nothing is emitted after this."""
        if self._stopped or not self._started:
            return self
        self._stopped = True
        t = self._totals
        await self._emit(
            ev.EventKind.SESSION_END,
            ev.SessionEnd(
                reason=reason,
                n_chats=self._n_chats,
                total_tokens_in=t.tokens_in,
                total_tokens_out=t.tokens_out,
                total_cost_usd=t.cost_usd,
                duration_s=time.time() - t.t0,
            ),
        )
        return self

    def interrupt(self, reason: str = "asked to stop") -> None:
        """Ask the loop to stop. Sync on purpose: this is called from outside
        the loop -- a signal handler, another task, a UI thread -- and only
        flips a flag that the loop checks at its next boundary."""
        self._interrupt = reason

    # ------------------------------------------------------------------ #
    # the loop
    # ------------------------------------------------------------------ #

    async def chat(
        self,
        message: str,
        files: list[str] | None = None,
        output_structure: tt.BM | None = None,
        stream: bool = True,
        **kwargs,
    ) -> str | tt.BM | None:
        """One chat: a prompt in, the final answer out.

        Returns the answer string, or an instance of ``output_structure`` when
        one is given. Returns ``None`` when the chat died -- interrupted, or the
        model call failed unrecoverably. Watch the ``Interrupt`` and
        ``LLMFailed`` events to tell those apart.
        """
        if not self._started:
            raise RuntimeError("await Kernel.start() before chatting")
        if self._stopped:
            raise RuntimeError("this kernel has been stopped")

        output_structure = output_structure or kwargs.pop("output", None)
        self._chat_id = "chat_" + str(tu.get_snowflake())
        self._interrupt = None
        self._n_chats += 1
        c = _ChatRollup()

        prompt = await self._emit(
            ev.EventKind.USER_PROMPT_SUBMIT,
            ev.UserPromptSubmit(text=message, attachments=list(files or [])),
        )
        self._build_thread(prompt.text, prompt.attachments, output_structure)

        while True:
            if self._interrupt:
                return await self._interrupted()

            await self._compact(c)

            out = await self._llm_step(c, stream=stream)
            if out is None:  # LLMFailed or interrupted mid-stream
                if self._interrupt:
                    return await self._interrupted()
                return None

            if not out.tool_calls:
                return await self._respond(out, c)

            for tc in out.tool_calls:
                if self._interrupt:
                    return await self._interrupted()
                await self._perform_tool_call(tc, c, stream=stream)

            if c.n_turns >= self.config.max_turns:
                tu.logger.warning(
                    f"max_turns ({self.config.max_turns}) hit, forcing an answer"
                )
                self.thread.append(
                    tt.human(
                        "You have hit the tool call limit for this chat. "
                        "Answer now with what you have, and call no more tools."
                    )
                )

    async def _llm_step(self, c: _ChatRollup, stream: bool) -> tt.ChatResponse | None:
        """One llm-step. Returns the assembled response, or None if it died."""
        self._step_id = "step_" + str(tu.get_snowflake())
        request = self.wire._process_input(
            self.thread,
            model=self.config.model_id,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )

        # subscribers may rewrite the request in place before it goes out
        step = await self._emit(
            ev.EventKind.LLM_STEP,
            ev.LLMStep(
                request=request,
                model=self.config.model_id,
                n_messages=len(self.thread.chats),
                est_tokens_in=_est_tokens(request),
                attempt=1,
            ),
        )
        c.n_llm_steps += 1

        t0 = time.time()
        text, tool_calls, reasoning, usage = "", [], [], None
        try:
            async for item in self.wire.stream(self.thread, request=step.request):
                if self._interrupt:
                    break
                if isinstance(item, str):
                    text += item
                elif isinstance(item, tt.ReasoningBlock):
                    reasoning.append(item)
                    await self._emit(
                        ev.EventKind.REASONING,
                        ev.Reasoning(
                            text=item.text,
                            signature=item.signature,
                            index=item.index,
                        ),
                    )
                    if stream:
                        _show("reasoning", item.text)
                elif isinstance(item, tt.ToolCall):
                    tool_calls.append(item)
                elif isinstance(item, tt.Usage):
                    usage = item
        except Exception as e:
            await self._emit(
                ev.EventKind.LLM_FAILED,
                ev.LLMFailed(
                    error=str(e),
                    error_type=type(e).__name__,
                    status_code=getattr(e, "status_code", None),
                    attempt=1,
                    retryable=False,
                    request=step.request,
                ),
            )
            return None

        if self._interrupt:
            return None

        self._account(c, usage)
        out = tt.ChatResponse(
            parts=([text] if text else []) + tool_calls,
            reasoning=reasoning,
            usage=usage,
        )
        await self._emit(
            ev.EventKind.LLM_OUTPUT,
            ev.LLMOutput(
                response=step.request.get("_raw", {}),
                text=text,
                tool_calls=[
                    {"name": t.tool.name, "arguments": t.arguments} for t in tool_calls
                ],
                stop_reason="tool_calls" if tool_calls else "end_turn",
                tokens_in=usage.input_tokens if usage else 0,
                tokens_out=usage.output_tokens if usage else 0,
                cached_tokens=usage.cached_tokens if usage else 0,
                cost_usd=self._cost(usage),
                latency_s=time.time() - t0,
            ),
        )
        return out

    async def _perform_tool_call(
        self, tc: tt.ToolCall, c: _ChatRollup, stream: bool = True
    ) -> None:
        """Run one tool and feed its response back into the thread."""
        self._turn_id = "turn_" + str(tu.get_snowflake())
        c.n_turns += 1
        c.n_tool_calls += 1

        pre = await self._emit(
            ev.EventKind.PRE_TOOL_USE,
            ev.PreToolUse(
                tool_call_id=self._turn_id, name=tc.tool.name, args=tc.arguments
            ),
        )
        if stream:
            _show("tool", f"{pre.name}({tu.to_json(pre.args, tight=True)})")

        self.thread.append(
            tt.function_call({"name": pre.name, "arguments": pre.args})
        )

        if pre.blocked:
            # a subscriber refused the call: the model is told, and the loop
            # carries on rather than dying
            self.thread.append(tt.function_resp(pre.block_reason))
            c.n_tool_failures += 1
            await self._emit(
                ev.EventKind.POST_TOOL_USE_FAILURE,
                ev.PostToolUseFailure(
                    tool_call_id=self._turn_id,
                    name=pre.name,
                    args=pre.args,
                    error=pre.block_reason,
                    error_type="blocked",
                    traceback=None,
                    duration_s=0.0,
                    sent_to_llm=pre.block_reason,
                ),
            )
            return

        t0 = time.time()
        try:
            with contextlib.chdir(self.base_folder):
                result = await tc.run_async()
        except Exception as e:
            import traceback as _tb

            c.n_tool_failures += 1
            sent = f"{type(e).__name__}: {e}"
            post = await self._emit(
                ev.EventKind.POST_TOOL_USE_FAILURE,
                ev.PostToolUseFailure(
                    tool_call_id=self._turn_id,
                    name=tc.tool.name,
                    args=tc.arguments,
                    error=str(e),
                    error_type=type(e).__name__,
                    traceback=_tb.format_exc(),
                    duration_s=time.time() - t0,
                    sent_to_llm=sent,
                ),
            )
            # the failure is not fatal: it goes back as the tool result and the
            # model gets to decide what to do about it
            self.thread.append(tt.function_resp(post.sent_to_llm))
            if stream:
                _show("tool-failed", post.sent_to_llm)
            return

        post = await self._emit(
            ev.EventKind.POST_TOOL_USE,
            ev.PostToolUse(
                tool_call_id=self._turn_id,
                name=tc.tool.name,
                result=result,
                duration_s=time.time() - t0,
            ),
        )
        self.thread.append(tt.function_resp(post.result))
        if stream:
            _show("tool-done", str(post.result))

    # ------------------------------------------------------------------ #
    # exits
    # ------------------------------------------------------------------ #

    async def _respond(self, out: tt.ChatResponse, c: _ChatRollup):
        """The model returned no tool calls: this is the last AI message."""
        structured = None
        if self.thread.schema is not None and out.text:
            try:
                structured = self.thread.schema(**tu.from_json(out.text))
            except Exception as e:
                tu.logger.error(f"could not parse structured output: {out.text}")
                raise e

        self.thread.append(tt.assistant(out.text))
        body = await self._emit(
            ev.EventKind.ASSISTANT_CHAT_RESPONSE,
            ev.AssistantChatResponse(
                text=out.text,
                content=[{"type": "text", "text": out.text}]
                + [{"type": "thinking", "text": r.text} for r in out.reasoning],
                structured=structured,
                stop_reason="end_turn",
                n_turns=c.n_turns,
                n_llm_steps=c.n_llm_steps,
                n_tool_calls=c.n_tool_calls,
                n_tool_failures=c.n_tool_failures,
                tokens_in=c.tokens_in,
                tokens_out=c.tokens_out,
                cached_tokens=c.cached_tokens,
                cost_usd=c.cost_usd,
                duration_s=time.time() - c.t0,
                compacted=c.compacted,
            ),
        )
        return body.structured if body.structured is not None else body.text

    async def _interrupted(self) -> None:
        """Someone called ``interrupt()``. Back to idle, no response event."""
        await self._emit(
            ev.EventKind.INTERRUPT,
            ev.Interrupt(
                reason=self._interrupt or "",
                interrupted_at=ev.EventKind.LLM_STEP,
                partial_text=None,
            ),
        )
        self._interrupt = None
        return None

    # ------------------------------------------------------------------ #
    # thread management
    # ------------------------------------------------------------------ #

    def _build_thread(
        self,
        message: str,
        files: list[str] | None = None,
        output_structure: tt.BM | None = None,
    ) -> tt.Thread:
        """Append the prompt (and any attachments) to the running thread.

        Attachments are routed by mime type: pictures and sound go to the
        model as themselves, and everything else is inlined as text. The
        ``else`` is a real fallback, not a default -- it calls ``read_text()``,
        so anything binary that is not recognised here fails on the decode
        rather than being sent as mojibake.
        """
        images, texts, audio = [], [], []
        for f in files or []:
            p = Path(f)
            if not p.is_absolute():
                p = self.base_folder / p
            mime = tu.get_mime_type(str(p))
            if mime.startswith("image/"):
                images.append(tu.to_b64(p.read_bytes()))
            elif mime.startswith("audio/"):
                audio.append(p)
            else:
                try:
                    texts.append(f"<file path='{p}'>\n{p.read_text()}\n</file>")
                except UnicodeDecodeError as e:
                    # Naming the type beats a decode error from three frames
                    # down: someone attaching a video wants to be told that
                    # video is not carried, not that byte 148 is not utf-8.
                    raise ValueError(
                        f"cannot attach {p.name}: {mime} is not text, and only "
                        f"image/* and audio/* are sent as themselves"
                    ) from e

        value = "\n\n".join(texts + [message]) if texts else message
        self.thread.append(tt.human(value, images=images, audio=audio))
        self.thread.schema = output_structure
        return self.thread

    async def _compact(self, c: _ChatRollup) -> None:
        """Summarise the thread when it has grown past its budget."""
        limit = self.config.compact_at_tokens
        if not limit or c.last_input_tokens < limit:
            return

        keep = self.config.compact_keep_last
        head = self.thread.chats[:1] if _is_system(self.thread.chats) else []
        pre = await self._emit(
            ev.EventKind.PRE_COMPACT,
            ev.PreCompact(
                trigger="auto",
                n_messages=len(self.thread.chats),
                tokens_before=c.last_input_tokens,
                threshold=limit,
                keep_last=keep,
            ),
        )

        t0 = time.time()
        before = len(self.thread.chats)
        ask = tt.Thread(*self.thread.chats, tt.human(COMPACT_PROMPT))
        resp = await self.wire.run(ask)
        summary = resp.text

        tail = _trim_dangling(self.thread.chats[len(head) :][-pre.keep_last :])
        self.thread.chats = head + [tt.human(f"<summary>\n{summary}\n</summary>")] + tail

        c.compacted = True
        c.last_input_tokens = 0
        self._account(c, resp.usage)
        await self._emit(
            ev.EventKind.POST_COMPACT,
            ev.PostCompact(
                summary=summary,
                n_messages_before=before,
                n_messages_after=len(self.thread.chats),
                tokens_before=pre.tokens_before,
                tokens_after=_est_tokens({"messages": self.thread.chats}),
                cost_usd=self._cost(resp.usage),
                duration_s=time.time() - t0,
            ),
        )

    # ------------------------------------------------------------------ #
    # plumbing
    # ------------------------------------------------------------------ #

    async def _emit(self, kind: ev.EventKind, body: Any) -> Any:
        """Hand the event to its subscribers and return the (possibly mutated)
        body. Subscribers run in registration order and mutate in place."""
        self._seq += 1
        event = ev.Event(
            kind=kind,
            session_id=self.session_id,
            ts=time.time(),
            seq=self._seq,
            body=body,
            chat_id=self._chat_id,
            turn_id=self._turn_id,
            step_id=self._step_id,
        )
        for fn in self._subs.get(kind, []):
            out = fn(event)
            if inspect.isawaitable(out):
                await out
        return event.body

    def _account(self, c: _ChatRollup, usage: tt.Usage | None) -> None:
        if usage is None:
            return
        for t in (c, self._totals):
            t.tokens_in += usage.input_tokens
            t.tokens_out += usage.output_tokens
            t.cached_tokens += usage.cached_tokens
            t.cost_usd += self._cost(usage)
        c.last_input_tokens = usage.input_tokens + usage.output_tokens

    async def _resolve_price(self) -> None:
        """Take the price from the config if it is set, otherwise ask the wire
        for the model's published one. The wire caches that per host for the
        life of the process, so a hundred kernels cost one HTTP call between
        them -- and none at all after the first."""
        c = self.config
        self._price = (c.price_in_per_m, c.price_cached_per_m, c.price_out_per_m)
        if any(self._price) or not hasattr(self.wire, "get_price"):
            return
        price = await self.wire.get_price(c.model_id)
        self._price = (price.input, price.cached, price.output)
        if price.is_empty():
            tu.logger.warning(
                f"no published price for '{c.model_id}', costs will read 0.0"
            )

    def _cost(self, usage: tt.Usage | None) -> float:
        if usage is None:
            return 0.0
        return usage.cost(*self._price)


# ---------------------------------------------------------------------- #
# module helpers
# ---------------------------------------------------------------------- #


def _is_system(chats: list[tt.Message]) -> bool:
    return bool(chats) and chats[0].role == tt.Message.SYSTEM


def _trim_dangling(chats: list[tt.Message]) -> list[tt.Message]:
    """Drop a leading tool response whose tool call was just summarised away --
    providers reject a tool result with no matching call."""
    while chats and chats[0].role == tt.Message.FUNCTION_RESP:
        chats = chats[1:]
    return chats


def _est_tokens(request: dict) -> int:
    """Rough count, for the events only. Four characters to a token."""
    return len(tu.to_json(request.get("messages", []), tight=True)) // 4


def _show(what: str, text: str) -> None:
    text = text if len(text) < 500 else text[:500] + "..."
    tu.logger.info(f"[{what}] {text}")
