"""Audio on a message: normalised on the way in, an ``input_audio`` part on the wire.

The live call is skipped without ``OPENROUTER_TOKEN`` -- everything up to the
request is checked without one, because the shape of what gets sent is the
part that breaks, and it can be asserted for free.
"""

import os
import base64
import asyncio

import pytest

import tuneapi.types as tt
from tuneapi.types.chats import _as_audio
from tuneapi.apis.model_openai import OpenAIProtocol

HERE = os.path.dirname(os.path.abspath(__file__))
CLIP = os.path.join(HERE, "madukya_2m.mp3")


# --- normalising ----------------------------------------------------------


def test_a_path_is_read_and_its_format_taken_from_the_extension():
    (clip,) = tt.human("what is this", audio=[CLIP]).audio

    assert clip["format"] == "mp3"
    assert base64.b64decode(clip["data"])[:3] in (b"ID3", b"\xff\xfb\x00")


def test_a_dict_is_taken_as_given():
    """Already-encoded audio should not have to be written to a file first."""
    (clip,) = tt.human("x", audio=[{"data": "QUJD", "format": "wav"}]).audio

    assert clip == {"data": "QUJD", "format": "wav"}


def test_a_dict_missing_its_format_is_refused():
    """Format cannot be guessed from base64, and a wrong guess fails upstream."""
    with pytest.raises(ValueError, match="both 'data'.*and 'format'"):
        tt.human("x", audio=[{"data": "QUJD"}])


def test_a_missing_file_says_so_here_rather_than_at_the_wire():
    with pytest.raises(ValueError, match="audio file not found"):
        tt.human("x", audio=["/no/such/clip.mp3"])


def test_audio_survives_a_round_trip_through_a_dict():
    """A thread that is stored and reloaded must still carry its sound."""
    original = tt.human("what is this", audio=[CLIP])
    restored = tt.Message.from_dict(
        {**original.to_dict(), "value": original.value, "role": "user"}
    )

    assert restored.audio == original.audio


def test_the_default_argument_is_not_shared_between_messages():
    """The mutable default is only safe because nothing writes back into it."""
    first = tt.human("one")
    tt.human("two", audio=[{"data": "QUJD", "format": "mp3"}])

    assert first.audio == []


# --- the wire -------------------------------------------------------------


def _messages(thread):
    protocol = OpenAIProtocol.__new__(OpenAIProtocol)
    return protocol._process_thread(thread)


def test_audio_becomes_an_input_audio_part_after_the_text():
    thread = tt.Thread(tt.human("what is said here", audio=[CLIP]))

    (message,) = _messages(thread)
    kinds = [part["type"] for part in message["content"]]

    assert message["role"] == "user"
    assert kinds == ["text", "input_audio"]
    assert message["content"][1]["input_audio"]["format"] == "mp3"


def test_audio_rides_alongside_images_and_a_parts_list():
    """A caller who built their own content list keeps it, plus the audio."""
    parts = [{"type": "text", "text": "compare these"}]
    thread = tt.Thread(
        tt.human(parts, images=["QUJD"], audio=[{"data": "QUJD", "format": "wav"}])
    )

    (message,) = _messages(thread)

    assert [p["type"] for p in message["content"]] == [
        "text", "image_url", "input_audio",
    ]


def test_an_assistant_turn_carries_no_audio():
    """``input_audio`` is a request shape; a reply is not described with it."""
    thread = tt.Thread(
        tt.human("listen", audio=[{"data": "QUJD", "format": "mp3"}]),
        tt.assistant("I heard it"),
    )

    human_message, assistant_message = _messages(thread)

    assert any(p["type"] == "input_audio" for p in human_message["content"])
    assert all(p["type"] != "input_audio" for p in assistant_message["content"])


# --- the kernel's file router --------------------------------------------


def test_the_kernel_routes_an_audio_file_to_the_audio_field():
    from tuneapi.agentic_kernel import Kernel, KernelConfig

    kernel = Kernel.__new__(Kernel)
    kernel.thread = tt.Thread()
    kernel.base_folder = HERE
    kernel._build_thread("what is this", files=[CLIP])

    (message,) = kernel.thread.chats
    assert message.audio and message.audio[0]["format"] == "mp3"
    # The prompt stays the prompt: audio does not get inlined into the text
    # the way an attached file does.
    assert message.value == "what is this"


def test_attaching_something_binary_says_what_it_is(tmp_path):
    from tuneapi.agentic_kernel import Kernel

    clip = tmp_path / "recording.mp4"
    clip.write_bytes(b"\x00\x00\x00\x18ftypmp42\xff\xfb")

    kernel = Kernel.__new__(Kernel)
    kernel.thread = tt.Thread()
    kernel.base_folder = str(tmp_path)

    with pytest.raises(ValueError, match="video/mp4 is not text"):
        kernel._build_thread("watch this", files=[str(clip)])


# --- against the real provider -------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("OPENROUTER_TOKEN"), reason="needs OPENROUTER_TOKEN"
)
def test_a_real_model_hears_the_clip():
    from openai import AsyncOpenAI
    from tuneapi.agentic_kernel import Kernel, KernelConfig

    model = os.environ.get("JAMINI_MODEL", "google/gemini-3.7-flash")

    class Heard(tt.BM):
        language: str
        transcript: str

    async def run():
        client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_TOKEN"],
        )
        kernel = Kernel(
            KernelConfig(model_id=model),
            wire=OpenAIProtocol(client, model_id=model),
        )
        await kernel.start()
        try:
            return await kernel.chat(
                "Transcribe this audio.",
                files=[CLIP],
                output_structure=Heard,
                stream=False,
            )
        finally:
            await kernel.stop()

    heard = asyncio.run(run())

    assert isinstance(heard, Heard)
    assert heard.transcript.strip()
