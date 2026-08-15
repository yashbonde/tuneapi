# Copyright © 2024-2025 Frello Technology Private Limited
# Copyright © 2025- Yash Bonde github.com/yashbonde
# MIT License

# the single OpenAI-protocol wire
from tuneapi.apis.model_openai import OpenAIProtocol
from tuneapi.apis.compat import to_openai_message, to_openai_chunk

__all__ = ["OpenAIProtocol", "to_openai_message", "to_openai_chunk"]
