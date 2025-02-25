# Copyright © 2023- Frello Technology Private Limited

from tuneapi import tu, ta

import asyncio
from pydantic import BaseModel

AUDIO_FP = tu.joinp(tu.folder(__file__), "madukya_2m.mp3")


# models
ant = ta.Anthropic()
gemini = ta.Gemini()
oai = ta.Openai()

# fmt: off
print("[BLOCKING] [Anthropic]", ant.chat("2 + 2 =", max_tokens=10))
print("[BLOCKING] [Gemini]", gemini.chat("2 + 2 =", max_tokens=10).strip())
print("[BLOCKING] [Gemini.embedding]", len(gemini.embedding("2 + 2 =").embedding[0]))
print("[BLOCKING] [Openai]", oai.chat("2 + 2 =", max_tokens=10))
print("[BLOCKING] [Openai.embedding]", len(oai.embedding("2 + 2 =").embedding[0]))
print("[BLOCKING] [Openai.text_to_speech]", len(oai.text_to_speech("2 + 2 =")), "bytes")
print("[BLOCKING] [Openai.speech_to_text]", len(oai.speech_to_text("transcribe", AUDIO_FP).segments), "segments")
print("[BLOCKING] [Openai.image_gen]", oai.image_gen("2 + 2 =").image.size, "dimensions")
# fmt: on


async def main():
    # fmt: off
    print("[ASYNC] [Anthropic]", await ant.chat_async("2 + 2 =", max_tokens=10))
    print("[ASYNC] [Gemini]", (await gemini.chat_async("2 + 2 =", max_tokens=10)).strip())
    print("[ASYNC] [Gemini.embedding]", len((await gemini.embedding_async("2 + 2 =")).embedding[0]))
    print("[ASYNC] [Openai]", await oai.chat_async("2 + 2 =", max_tokens=10))
    print("[ASYNC] [Openai.embedding]", len((await oai.embedding_async("2 + 2 =")).embedding[0]))
    print("[ASYNC] [Openai.text_to_speech]", len(await oai.text_to_speech_async("2 + 2 =")), "bytes")
    print("[ASYNC] [Openai.image_gen]", (await oai.image_gen_async("2 + 2 =")).image.size, "dimensions")
    # fmt: on


asyncio.run(main())
