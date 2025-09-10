# Copyright © 2023- Frello Technology Private Limited

from tuneapi import tu, ta, tt

from fire import Fire

import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed

AUDIO_FP = tu.joinp(tu.folder(__file__), "madukya_2m.mp3")


# models


async def main(
    ant: tt.ModelInterface,
    gemini: tt.ModelInterface,
    oai: tt.ModelInterface,
):
    # Async concurrent execution - each function runs concurrently
    tasks = []

    # fmt: off
    if ant:
        tasks.append(("anthropic_chat", ant.chat_async("2 + 2 =", max_tokens=10)))
    if gemini:
        tasks.append(("gemini_chat", gemini.chat_async("2 + 2 =", max_tokens=10)))
        tasks.append(("gemini_embedding", gemini.embedding_async("2 + 2 =")))
    if oai:
        tasks.append(("openai_chat", oai.chat_async("2 + 2 =", max_tokens=10)))
        tasks.append(("openai_embedding", oai.embedding_async("2 + 2 =")))
        tasks.append(("openai_tts", oai.text_to_speech_async("2 + 2 =")))
        tasks.append(("openai_stt", oai.speech_to_text_async("transcribe", AUDIO_FP)))
        tasks.append(("openai_image", oai.image_gen_async("2 + 2 =")))
    # fmt: on

    # Run all tasks concurrently
    if tasks:
        results = await asyncio.gather(
            *[task[1] for task in tasks], return_exceptions=True
        )

        # Process results
        for (name, _), result in zip(tasks, results):
            if isinstance(result, Exception):
                print(f"[ASYNC] [{name}] Error: {result}")
            else:
                if name == "anthropic_chat":
                    print("[ASYNC] [Anthropic]", result)
                elif name == "gemini_chat":
                    print("[ASYNC] [Gemini]", result.strip())
                elif name == "gemini_embedding":
                    print("[ASYNC] [Gemini.embedding]", len(result.embedding[0]))
                elif name == "openai_chat":
                    print("[ASYNC] [Openai]", result)
                elif name == "openai_embedding":
                    print("[ASYNC] [Openai.embedding]", len(result.embedding[0]))
                elif name == "openai_tts":
                    print("[ASYNC] [Openai.text_to_speech]", len(result), "bytes")
                elif name == "openai_stt":
                    print(
                        "[ASYNC] [Openai.speech_to_text]",
                        len(result.segments),
                        "segments",
                    )
                elif name == "openai_image":
                    print("[ASYNC] [Openai.image_gen]", result.image.size, "dimensions")


def main_(
    openai: bool = False,
    anthropic: bool = False,
    gemini: bool = False,
):
    ant = ta.Anthropic()
    gemini_model = ta.Gemini()
    oai = ta.Openai()

    # Threadpool - each function runs as a thread in the pool
    tasks = []

    # fmt: off
    if anthropic:
        tasks.append(("anthropic_chat", lambda: ant.chat("2 + 2 =", max_tokens=10)))
    if gemini:
        tasks.append(("gemini_chat", lambda: gemini_model.chat("2 + 2 =", max_tokens=10)))
        tasks.append(("gemini_embedding", lambda: gemini_model.embedding("2 + 2 =")))
    if openai:
        tasks.append(("openai_chat", lambda: oai.chat("2 + 2 =", max_tokens=10)))
        tasks.append(("openai_embedding", lambda: oai.embedding("2 + 2 =")))
        tasks.append(("openai_tts", lambda: oai.text_to_speech("2 + 2 =")))
        tasks.append(("openai_stt", lambda: oai.speech_to_text("transcribe", AUDIO_FP)))
        tasks.append(("openai_image", lambda: oai.image_gen("2 + 2 =")))
    # fmt: on

    # Execute all tasks in parallel using ThreadPoolExecutor
    if tasks:
        with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
            # Submit all tasks
            future_to_name = {executor.submit(task[1]): task[0] for task in tasks}

            # Process results as they complete
            for future in as_completed(future_to_name):
                name = future_to_name[future]
                try:
                    result = future.result()
                    if name == "anthropic_chat":
                        print("[BLOCKING] [Anthropic]", result)
                    elif name == "gemini_chat":
                        print("[BLOCKING] [Gemini]", result.strip())
                    elif name == "gemini_embedding":
                        print("[BLOCKING] [Gemini.embedding]", len(result.embedding[0]))
                    elif name == "openai_chat":
                        print("[BLOCKING] [Openai]", result)
                    elif name == "openai_embedding":
                        print("[BLOCKING] [Openai.embedding]", len(result.embedding[0]))
                    elif name == "openai_tts":
                        print(
                            "[BLOCKING] [Openai.text_to_speech]", len(result), "bytes"
                        )
                    elif name == "openai_stt":
                        print(
                            "[BLOCKING] [Openai.speech_to_text]",
                            len(result.segments),
                            "segments",
                        )
                    elif name == "openai_image":
                        print(
                            "[BLOCKING] [Openai.image_gen]",
                            result.image.size,
                            "dimensions",
                        )
                except Exception as exc:
                    print(f"[BLOCKING] [{name}] Error: {exc}")

    asyncio.run(
        main(
            ant if anthropic else None,
            gemini_model if gemini else None,
            oai if openai else None,
        )
    )

    pass


if __name__ == "__main__":
    Fire(main_)
