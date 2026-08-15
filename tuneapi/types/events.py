# Copyright © 2024-2025 Frello Technology Private Limited
# Copyright © 2025- Yash Bonde github.com/yashbonde

"""Event kinds emitted by the kernel.

Terminology, outermost to innermost::

    transcript ── a full conversation (one complete session)
     └─ chat ── one human prompt → agent turns → final AI response
        └─ turn ── one tool call + its tool response (a loop step)
           └─ llm-step ── one LLM call (one litellm_request)

Every event carries the ids of the scopes it sits inside, so a consumer can
fold the flat stream back into that tree without holding any state:
``session_id`` on everything, ``chat_id`` from UserPromptSubmit onwards,
``turn_id`` inside the loop, ``step_id`` inside an llm-step.
"""

from __future__ import annotations

import functools
import inspect
from enum import Enum
from typing import Any, Literal, Union

from tuneapi.types.bm import BM, F


class EventKind(str, Enum):
    # --- session ---------------------------------------------------------
    SESSION_START = "SessionStart"  # kernel.start()
    SESSION_END = "SessionEnd"  # kernel.end(), graceful only

    # --- chat ------------------------------------------------------------
    USER_PROMPT_SUBMIT = "UserPromptSubmit"  # kernel.chat(...)
    ASSISTANT_CHAT_RESPONSE = "AssistantChatResponse"  # loop stopped, answer out

    # --- llm-step --------------------------------------------------------
    LLM_STEP = "LLMStep"  # request body going out
    LLM_OUTPUT = "LLMOutput"  # response body coming back
    LLM_FAILED = "LLMFailed"  # call errored --> kernel goes idle

    REASONING = "Reasoning"  # one complete thinking block

    # --- turn ------------------------------------------------------------
    PRE_TOOL_USE = "PreToolUse"
    POST_TOOL_USE = "PostToolUse"
    POST_TOOL_USE_FAILURE = "PostToolUseFailure"  # error fed back to the LLM

    # --- out of band -----------------------------------------------------
    INTERRUPT = "Interrupt"  # kernel.interrupt() --> kernel goes idle
    PRE_COMPACT = "PreCompact"  # kernel._compact()
    POST_COMPACT = "PostCompact"  # kernel._compact(), carries the summary


# alias used by the kernel's public API
EventType = EventKind


class Event(BM):
    """Envelope. ``body`` is one of the payloads below, matching ``kind``."""

    model_config = {"arbitrary_types_allowed": True}

    kind: EventKind
    session_id: str
    ts: float = F("unix seconds, when the kernel emitted it")
    body: Any = F("payload matching ``kind``")
    chat_id: Union[str, None] = F("set from UserPromptSubmit onwards", None)
    turn_id: Union[str, None] = F("set inside the loop", None)
    step_id: Union[str, None] = F("set inside an llm-step", None)
    seq: int = F("monotonic per session, for ordering and replay", 0)


# ---------------------------------------------------------------------------
# session
# ---------------------------------------------------------------------------


class SessionStart(BM):
    model_config = {"arbitrary_types_allowed": True}

    model: str
    system_prompt: str
    tools: list[str] = F("tool names registered at start")
    cwd: str
    resumed_from: Union[str, None] = F("prior session id, if continued", None)
    config: dict[str, Any] = F("kernel config snapshot", default_factory=dict)


class SessionEnd(BM):
    reason: Literal["user", "idle_timeout", "shutdown"]
    n_chats: int
    total_tokens_in: int
    total_tokens_out: int
    total_cost_usd: float
    duration_s: float


# ---------------------------------------------------------------------------
# chat
# ---------------------------------------------------------------------------


class UserPromptSubmit(BM):
    text: str
    attachments: list[str] = F("paths or uris", default_factory=list)
    source: Literal["human", "schedule", "api"] = "human"


class AssistantChatResponse(BM):
    """Closes a chat normally: an LLMOutput arrived carrying no tool calls.

    This is the last AI message of the chat. It is NOT emitted when the chat
    dies -- Interrupt and an unretryable LLMFailed return the kernel to idle
    on their own, so a consumer waiting only on this event will hang on those
    paths and must watch all three.

    The rollup fields cover the whole chat -- every llm-step and every tool
    call under it -- so a consumer never has to sum LLMOutput events itself.
    """

    model_config = {"arbitrary_types_allowed": True}

    text: str = F("the final assistant answer, as shown to the user")
    content: list[dict[str, Any]] = F("final structured blocks: thinking + text")
    stop_reason: str = F("the model's own reason: end_turn, max_tokens, ...")
    structured: Any = F("the parsed object when the chat asked for a schema", None)

    n_turns: int = F("tool loop steps taken")
    n_llm_steps: int
    n_tool_calls: int
    n_tool_failures: int

    tokens_in: int
    tokens_out: int
    cached_tokens: int
    cost_usd: float = F("total for the chat, all steps summed")
    duration_s: float

    compacted: bool = F("a PreCompact/PostCompact pair fired inside", False)


# ---------------------------------------------------------------------------
# llm-step
# ---------------------------------------------------------------------------


class LLMStep(BM):
    model_config = {"arbitrary_types_allowed": True}

    request: dict[str, Any] = F("the litellm kwargs, verbatim")
    model: str
    n_messages: int
    est_tokens_in: int
    attempt: int = F(">1 after a retry", 1)


class LLMOutput(BM):
    model_config = {"arbitrary_types_allowed": True}

    response: dict[str, Any] = F("the provider response, verbatim")
    text: str = F("concatenated text blocks, for display")
    tool_calls: list[dict[str, Any]] = F(
        "name + args + id, empty ends the loop"
    )
    stop_reason: str
    tokens_in: int
    tokens_out: int
    cached_tokens: int
    cost_usd: float
    latency_s: float


class LLMFailed(BM):
    model_config = {"arbitrary_types_allowed": True}

    error: str
    error_type: str = F("provider class: rate_limit, context_length, auth, ...")
    status_code: Union[int, None]
    attempt: int
    retryable: bool = F("False means the kernel gave up and went idle")
    request: dict[str, Any]


class Reasoning(BM):
    text: str = F("one complete thinking block, not tokens")
    signature: Union[str, None] = F("provider block signature, if any", None)
    index: int = F("nth thinking block in this llm-step", 0)


# ---------------------------------------------------------------------------
# turn
# ---------------------------------------------------------------------------


class PreToolUse(BM):
    model_config = {"arbitrary_types_allowed": True}

    tool_call_id: str
    name: str
    args: dict[str, Any]

    blocked: bool = F("set by a subscriber to stop the call from running", False)
    block_reason: str = F("what the model is told instead of the tool result", "")


class PostToolUse(BM):
    model_config = {"arbitrary_types_allowed": True}

    tool_call_id: str
    name: str
    result: Any = F("what is fed back to the LLM")
    duration_s: float
    truncated: bool = F("payload was cut down before being sent", False)


class PostToolUseFailure(BM):
    model_config = {"arbitrary_types_allowed": True}

    tool_call_id: str
    name: str
    args: dict[str, Any]
    error: str
    error_type: str = F("not_found, timeout, bad_args, unhandled, ...")
    traceback: Union[str, None]
    duration_s: float
    sent_to_llm: str = F("the error string the model actually sees")


# ---------------------------------------------------------------------------
# out of band
# ---------------------------------------------------------------------------


class Interrupt(BM):
    reason: str
    interrupted_at: EventKind = F("what was in flight when the ask came in")
    partial_text: Union[str, None] = F("streamed so far, if anything", None)


class PreCompact(BM):
    trigger: Literal["auto", "manual"]
    n_messages: int
    tokens_before: int
    threshold: int = F("the budget that was crossed")
    keep_last: int = F("messages held out of the summary")


class PostCompact(BM):
    summary: str = F("the LLM-generated summary")
    n_messages_before: int
    n_messages_after: int
    tokens_before: int
    tokens_after: int
    cost_usd: float
    duration_s: float


# ---------------------------------------------------------------------------
# kernel plumbing: subscriber keying and registration
# ---------------------------------------------------------------------------

PAYLOAD_TO_KIND: dict[type, EventKind] = {
    SessionStart: EventKind.SESSION_START,
    SessionEnd: EventKind.SESSION_END,
    UserPromptSubmit: EventKind.USER_PROMPT_SUBMIT,
    AssistantChatResponse: EventKind.ASSISTANT_CHAT_RESPONSE,
    LLMStep: EventKind.LLM_STEP,
    LLMOutput: EventKind.LLM_OUTPUT,
    LLMFailed: EventKind.LLM_FAILED,
    Reasoning: EventKind.REASONING,
    PreToolUse: EventKind.PRE_TOOL_USE,
    PostToolUse: EventKind.POST_TOOL_USE,
    PostToolUseFailure: EventKind.POST_TOOL_USE_FAILURE,
    Interrupt: EventKind.INTERRUPT,
    PreCompact: EventKind.PRE_COMPACT,
    PostCompact: EventKind.POST_COMPACT,
}

_NAME_TO_KIND: dict[str, EventKind] = {kind.value: kind for kind in EventKind}


def resolve_kind(key: Any) -> EventKind:
    """Accept an EventKind, a payload class, or the raw string name, return the EventKind."""
    if isinstance(key, EventKind):
        return key
    if isinstance(key, type) and key in PAYLOAD_TO_KIND:
        return PAYLOAD_TO_KIND[key]
    if isinstance(key, str):
        if key in _NAME_TO_KIND:
            return _NAME_TO_KIND[key]
        try:
            return EventKind(key)
        except ValueError:
            pass
    raise ValueError(f"cannot resolve EventKind from: {key!r}")


def event_subscriber(fn):
    """Mark a callable as an event subscriber.

    Sync and async callables are both fine; the kernel awaits async ones and
    calls sync ones directly.
    """

    fn._is_event_subscriber = True
    fn._is_async = inspect.iscoroutinefunction(fn)
    functools.wraps(fn)(fn)
    return fn
