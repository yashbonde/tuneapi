# Copyright © 2023- Frello Technology Private Limited

from tuneapi import ta, tt

import asyncio
from pydantic import BaseModel

# fmt: off
print("[BLOCKING] [Anthropic]", ta.Anthropic().chat("2 + 2 =", max_tokens=10))
print("[BLOCKING] [Openai]", ta.Openai().chat("2 + 2 =", max_tokens=10))
print("[BLOCKING] [Gemini]", ta.Gemini().chat("2 + 2 =", max_tokens=10))
print("[BLOCKING] [Mistral]", ta.Mistral().chat("2 + 2 =", max_tokens=10))
print("[BLOCKING] [Groq]", ta.Groq().chat("2 + 2 =", max_tokens=10))
print("[BLOCKING] [TuneAI]", ta.TuneModel("meta/llama-3.1-8b-instruct").chat("2 + 2 =", max_tokens=10))
# fmt: on


class Result(BaseModel):
    result: int


# fmt: off
print("[BLOCKING] [STRUCTURE] [Openai]", ta.Openai().chat(tt.Thread(tt.human("2 + 2 ="), schema=Result), max_tokens=10).result)
print("[BLOCKING] [STRUCTURE] [Gemini]", ta.Gemini().chat(tt.Thread(tt.human("2 + 2 ="), schema=Result), max_tokens=10).result)
# fmt: on


async def main():
    # fmt: off
    print("[ASYNC] [Anthropic]", await ta.Anthropic().chat_async("2 + 2 =", max_tokens=10))
    print("[ASYNC] [Openai]", await ta.Openai().chat_async("2 + 2 =", max_tokens=10))
    print("[ASYNC] [Gemini]", await ta.Gemini().chat_async("2 + 2 =", max_tokens=10))
    print("[ASYNC] [Mistral]", await ta.Mistral().chat_async("2 + 2 =", max_tokens=10))
    print("[ASYNC] [Groq]", await ta.Groq().chat_async("2 + 2 =", max_tokens=10))
    print("[ASYNC] [TuneAI]", await ta.TuneModel("meta/llama-3.1-8b-instruct").chat_async("2 + 2 =", max_tokens=10))
    # fmt: on


asyncio.run(main())
