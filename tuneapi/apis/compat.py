# Copyright © 2024-2025 Frello Technology Private Limited
# Copyright © 2025- Yash Bonde github.com/yashbonde

# functions for compatibility

from tuneapi.types import Message
from tuneapi.utils import SimplerTimes, get_snowflake, to_json


def to_openai_message(message: Message) -> dict:
    """Convert a message into full OpenAI response format"""
    role = "assistant" if message.role == message.GPT else None
    role = role or ("user" if message.role == message.HUMAN else None)
    if role is None:
        raise ValueError(f"Unknown role: {message.role}")
    return {
        "id": f"tuneapi-{get_snowflake()}",
        "object": "chat.completion",
        "created": SimplerTimes.get_now_i64(),
        "model": "model",
        "choices": [
            {
                "index": 0,
                "message": {"role": role, "content": message.value},
                "finish_reason": "stop",
            }
        ],
    }


def to_openai_chunk(message: Message, last: bool = False) -> str:
    role = "assistant" if message.role == message.GPT else None
    role = role or ("user" if message.role == message.HUMAN else None)
    if role is None:
        raise ValueError(f"Unknown role: {message.role}")
    data = {
        "id": f"tuneapi-{get_snowflake()}",
        "object": "chat.completion.chunk",
        "created": SimplerTimes.get_now_i64(),
        "model": "model",
        "choices": [
            {
                "delta": {"role": role, "content": message.value},
                "finish_reason": "stop" if last else None,
            }
        ],
    }
    return "data: " + to_json(data, tight=True) + "\n\n"
