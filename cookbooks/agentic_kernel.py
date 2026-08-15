"""Drive the agentic kernel against OpenRouter.

    export OPENROUTER_TOKEN=sk-or-...
    uv run python cookbooks/agentic_kernel.py

Shows the whole loop: a tool call, an event subscriber that rewrites the
outbound request, a second chat on the same kernel, and structured output.
"""

import os
import asyncio
import openai

import tuneapi as tu
import tuneapi.types as tt
from tuneapi.types import events as ev
from tuneapi.agentic_kernel import Kernel, KernelConfig


# ----------------------------------------------------------------- tools


@tt.tool()
def list_files(folder: str = ".") -> tt.Message:
    """List the files in a folder with their sizes in bytes.

    Args:
        folder: path to list, relative to the kernel's base folder
    """
    out = []
    for f in sorted(os.listdir(folder)):
        p = os.path.join(folder, f)
        if os.path.isfile(p):
            out.append({"name": f, "bytes": os.path.getsize(p)})
    return out


# ----------------------------------------------------------- subscribers


def guardrails(email: bool = False, ip: bool = False):
    """Redact things out of the request before it reaches the model. The
    subscriber mutates the event body in place; the kernel sends what is left."""
    import re

    EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
    IP = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")

    @ev.event_subscriber
    async def _subscriber(event: ev.Event):
        for message in event.body.request.get("messages", []):
            content = message.get("content")
            if not isinstance(content, str):
                continue
            if email:
                content = EMAIL.sub("<EMAIL>", content)
            if ip:
                content = IP.sub("<IP>", content)
            message["content"] = content

    return _subscriber


def trace(event: ev.Event):
    """Print every event as it goes past."""
    extra = ""
    if hasattr(event.body, "cost_usd"):
        extra = f"  ${event.body.cost_usd:.6f}"
    print(f"  · {event.seq:>3} {event.kind.value}{extra}")


# ------------------------------------------------------------------ main


class Summary(tt.BM):
    text: str
    file_count: int


async def main():
    token = os.environ["OPENROUTER_TOKEN"]
    client = openai.AsyncOpenAI(
        api_key=token,
        base_url="https://openrouter.ai/api/v1",
    )

    k = Kernel(
        config=KernelConfig(
            model_id=os.getenv("JAMINI_KERNEL_MODEL", "openai/gpt-4o-mini"),
            base_folder=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            tools=[list_files],
            system_prompt="You are a terse assistant working inside a code repo.",
        ),
        wire=tu.ta.OpenAIProtocol(client=client),
        event_subscribers={
            ev.LLMStep: [guardrails(email=True, ip=True), trace],
            ev.LLMOutput: trace,
            ev.PreToolUse: trace,
            ev.PostToolUse: trace,
            ev.AssistantChatResponse: trace,
        },
    )
    await k.start()
    print(f"price per 1M tokens (in, cached, out): {k._price}")

    print("\n--- chat 1: tool use ---")
    answer = await k.chat("What are the three largest files in the current folder?")
    print(answer)

    print("\n--- chat 2: structured output, same kernel ---")
    summary = await k.chat(
        "Summarise what you just did.",
        output_structure=Summary,
    )
    print(repr(summary))

    await k.stop()


if __name__ == "__main__":
    asyncio.run(main())
