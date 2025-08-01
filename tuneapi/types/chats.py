"""
This file contains all the datatypes relevant for a chat conversation. In general this is the nomenclature that we follow:
    * Message: a unit of information produced by a ``role``
    * Thread: a group of messages is called a thread. There can be many 2 types of threads, linear and tree based.
    * Threadslist: a group of linear threads is called a threads list.
    * Dataset: a container for grouping threads lists is called a dataset

Almost all the classes contain ``to_dict`` and ``from_dict`` for serialisation and deserialisation.
"""

# Copyright © 2024-2025 Frello Technology Private Limited
# Copyright © 2025- Yash Bonde github.com/yashbonde

import io
import os
import re
import copy
import httpx
import nutree as nt
from functools import partial
from abc import ABC, abstractmethod
from PIL.Image import Image as ImageType
from typing import Any, Callable, Generator

import tuneapi.utils as tu
from tuneapi.types.bm import BM, F


########################################################################################################################
#
# The code in this section contains the primitive of this new chat API. The ``Tool`` class defines tools that the model
# can predict. The ``Message`` class defines the container for storing the chat messages.
#
########################################################################################################################


class Prop(BM):
    """
    An individual property is called a prop.
    """

    name: str
    type: str = F("The kind of variable this is. 'number', 'text', 'enum', etc.")
    required: bool

    def __init__(
        self,
        name: str,
        type: str,
        required: bool = False,
        description: str | None = "",
        items: dict | None = None,
        enum: list[str] | None = None,
    ):
        self.name = name
        self.description = description
        self.required = required
        self.type = type
        self.items = items
        self.enum = enum

    def __repr__(self) -> str:
        return f"<Tool.Prop: " + ("*" if self.required else "") + f"{self.name}>"


class Tool(BM):
    """A tool is a container for telling the LLM what it can do. This is a standard definition."""

    name: str
    description: str
    parameters: list[Prop]

    def __repr__(self) -> str:
        return f"<Tool: {self.name}>"

    def to_dict(self):
        properties = {}
        required = []
        for x in self.parameters:
            properties[x.name] = {
                "type": x.type,
                "description": x.description,
            }
            if x.items:
                properties[x.name]["items"] = x.items
            if x.enum:
                properties[x.name]["enum"] = x.enum
            if x.required:
                required.append(x.name)

        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }

    @classmethod
    def from_dict(cls, x):
        if "type" in x and x["type"] == "function":
            return cls.from_dict(x["function"])
        parameters = []
        for k, v in x["parameters"].get("properties", {}).items():
            parameters.append(Prop(name=k, **v))
        return cls(
            name=x["name"],
            description=x["description"],
            parameters=parameters,
        )


class Message:
    """
    A message is the unit element of information in a thread. You should avoid using directly and use the convinience
    aliases ``tuneapi.types.chat. human/assistant/system/...``.

    Args:
        - value: this is generally a string or a list of dictionary objects for more advanced use cases
        - role: the role who produced this information
        - images: a list of PIL images or base64 strings
    """

    # names that are our standards roles
    SYSTEM = "system"
    HUMAN = "human"
    GPT = "gpt"
    FUNCTION_CALL = "function_call"
    FUNCTION_RESP = "function_resp"

    # mapping from known roles to our standard roles
    KNOWN_ROLES = {
        # system
        "system": SYSTEM,
        "sys": SYSTEM,
        # user
        "user": HUMAN,
        "human": HUMAN,
        # assistants
        "gpt": GPT,
        "assistant": GPT,
        "machine": GPT,
        # function calls
        "function_call": FUNCTION_CALL,
        "function-call": FUNCTION_CALL,
        # function response
        "function_resp": FUNCTION_RESP,
        "function-resp": FUNCTION_RESP,
        "tool": FUNCTION_RESP,
    }
    """A map that contains the popularly known mappings to make life simpler"""

    # start initialization here
    def __init__(
        self,
        value: str | list[dict[str, Any]],
        role: str,
        images: list[str | ImageType] = [],
        id: str = None,
        **kwargs,
    ):
        if role not in self.KNOWN_ROLES:
            raise ValueError(f"Unknown role: {role}. Update dictionary ``KNOWN_ROLES``")
        if value is None:
            raise ValueError("value cannot be None")

        self.role = self.KNOWN_ROLES[role]
        self.value = value
        self.id = id or "msg_" + str(tu.get_snowflake())
        self.metadata = kwargs
        self.images = images
        for i, img in enumerate(self.images):
            if isinstance(img, ImageType):
                buf = io.BytesIO()
                img.save(buf, "png")
                self.images[i] = tu.to_b64(buf.getvalue())

        # validations
        if self.role == self.FUNCTION_CALL:
            assert "name" in self.value, "key 'name' not found in function call"
            assert (
                "arguments" in self.value
            ), "key 'arguments' not found in function call"

    def __str__(self) -> str:
        try:
            idx = max(os.get_terminal_size().columns - len(self.role) - 40, 10)
        except OSError:
            idx = 50
        if isinstance(self.value, str):
            _msg = "\\n".join(self.value.splitlines())
        else:
            _msg = tu.to_json(self.value, tight=True)
        return f"<{self.role}: {_msg[:idx]}>"

    def __radd__(self, other: str):
        return Message(self.value + other, self.role)

    def __add__(self, other: str):
        return Message(self.value + other, self.role)

    def __repr__(self) -> str:
        out = str(self.value)
        return out

    def __getitem__(self, x):
        if x == "content":
            return self.value
        return getattr(self, x)

    def __getattr__(self, __name: str) -> Any:
        if __name in self.metadata:
            return self.metadata[__name]
        raise AttributeError(f"Attribute {__name} not found")

    def to_dict(
        self,
        format: str | None = None,
        meta: bool = False,
    ):
        """
        Serialise the Message into a dictionary of different formats:
            - format == ``ft`` then export to following format: ``{"from": "system/human/gpt", "value": "..."}``
            - format == ``api`` then ``{"role": "system/user/assistant", "content": [{"type": "text", "text": {"value": "..."}]}``. This is used with TuneAPI
            - format == ``full`` then ``{"id": 1234421123, "role": "system/user/assistant", "content": [{"type": "text", "text": {"value": "..."}]}``
            - default: ``{"role": "system/user/assistant", "content": "..."}``
        """
        role = self.role

        ft = format == "ft"
        api = format in ["api", "full"]
        full = format == "full"

        if not ft:
            if self.role == self.HUMAN:
                role = "user"
            elif self.role == self.GPT:
                role = "assistant"

        chat_message: dict[str, str | float]
        if ft:
            chat_message = {"from": role}
        else:
            chat_message = {"role": role}

        if ft:
            chat_message["value"] = self.value
        elif api:
            chat_message["content"] = [{"type": "text", "text": self.value}]
        else:
            chat_message["content"] = self.value

        if meta:
            chat_message["metadata"] = self.metadata

        if full:
            chat_message["id"] = self.id

        return chat_message

    @classmethod
    def from_dict(cls, data):
        """Deserialise and construct a message from a dictionary"""
        return cls(
            value=data.get("value") or data.get("content"),
            role=data.get("from") or data.get("role"),
            id=data.get("id", ""),
            images=data.get("images", []),
            **data.get("metadata", {}),
        )  # type: ignore


### Aliases
human = partial(Message, role=Message.HUMAN)
"""Convinience for creating a human message"""

system = partial(Message, role=Message.SYSTEM)
"""Convinience for creating a system message"""

assistant = partial(Message, role=Message.GPT)
"""Convinience for creating an assistant message"""

function_call = partial(Message, role=Message.FUNCTION_CALL)
"""Convinience for creating a function call message"""

function_resp = partial(Message, role=Message.FUNCTION_RESP)
"""Convinience for creating a function response message"""


########################################################################################################################
#
# Thread is an array of Messages and / or Tools.
#
########################################################################################################################


class Thread:
    """
    This is a container for a list of chat messages. This follows a similar interface to a list in python. See the methods
    below for more information.

    Args:
        *chats: list of chat ``Message`` objects
    """

    def __init__(
        self,
        *chats: list[Message] | Message,
        evals: dict[str, Any] | None = None,
        model: str | None = None,
        id: str = "",
        title: str = "",
        tools: list[Tool] | None = None,
        schema: BM | None = None,
        **kwargs,
    ):
        self.chats = list(chats)
        self.evals = evals
        self.model = model
        self.id = id or "thread_" + str(tu.get_snowflake())
        self.title = title
        self.tools = tools or []
        self.schema = schema

        #
        kwargs = {k: v for k, v in sorted(kwargs.items())}
        self.meta = kwargs
        self.keys = list(kwargs.keys())
        self.values = tuple(kwargs.values())

        # avoid special character BS.
        assert not any(["=" in x or "&" in x for x in self.keys])
        if self.values:
            assert all([type(x) in [int, str, float, bool] for x in self.values])

        self.value_hash = hash(self.values)

    def __repr__(self) -> str:
        x = "<Thread "
        if self.id:
            x += f"'{self.id}' "
        for k, v in self.meta.items():
            x += f"{k}={v} "
        for c in self.chats:
            x += f"\n  {c}"
        if self.tools:
            x += f"\n  <tools: {[x.name for x in self.tools]}>"
        if self.evals:
            x += f"\n  <evals: {[x for x in self.evals]}>"
        x += "\n>"
        return x

    def __getattr__(self, __name: str) -> Any:
        if __name in self.meta:
            return self.meta[__name]
        raise AttributeError(f"Attribute {__name} not found")

    def __getitem__(self, __x) -> Any:
        if isinstance(__x, tuple):
            new_thread = self.copy()
            start, end, step = __x
            new_thread.chats = new_thread.chats[start:end:step]
            return new_thread
        return self.chats[__x]

    def __radd__(self, other: "Thread"):
        thread = self.copy()
        thread.chats = other.chats + thread.chats
        tools_added = []
        for tool in other.tools + thread.tools:
            if tool.name not in tools_added:
                tools_added.append(tool)
                thread.tools.append(tool)
        thread.meta.update(other.meta)
        thread.keys = list(thread.meta.keys())
        thread.values = tuple(thread.meta.values())
        thread.value_hash = hash(thread.values)
        return thread

    def __add__(self, other: "Thread"):
        thread = self.copy()
        thread.chats = other.chats + thread.chats
        tools_added = []
        for tool in other.tools + thread.tools:
            if tool.name not in tools_added:
                tools_added.append(tool)
                thread.tools.append(tool)
        thread.meta.update(other.meta)
        thread.keys = list(thread.meta.keys())
        thread.values = tuple(thread.meta.values())
        thread.value_hash = hash(thread.values)
        return thread

    def __len__(self) -> int:
        return len(self.chats)

    # ser/deser

    def to_dict(self, full: bool = False):
        if full:
            return {
                "chats": [x.to_dict() for x in self.chats],
                "evals": self.evals,
                "model": self.model,
                "meta": self.meta,
                "title": self.title,
                "id": self.id,
                "tools": [x.to_dict() for x in self.tools],
                "schema": self.schema,
            }
        return {
            "chats": [x.to_dict() for x in self.chats],
            "tools": [x.to_dict() for x in self.tools],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Thread":
        chats = (
            data.get("chats", [])
            or data.get("conversations", [])
            or data.get("messages", [])
        )
        if not chats:
            raise ValueError("No chats found")
        return cls(
            *[Message.from_dict(x) for x in chats],
            id=data.get("id", ""),
            evals=data.get("evals", ""),
            model=data.get("model", ""),
            title=data.get("title", ""),
            tools=[Tool.from_dict(x) for x in data.get("tools", [])],
            schema=data.get("schema", {}),
            **data.get("meta", {}),
        )

    def copy(self) -> "Thread":
        return self.from_dict(self.to_dict())

    def to_ft(
        self, id: Any = None, drop_last: bool = False
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        chats = self.chats if not drop_last else self.chats[:-1]
        ft_dict = {
            "id": id or tu.get_random_string(6),
            "conversations": [x.to_dict(format="ft") for x in chats],
        }
        if drop_last:
            ft_dict["last"] = self.chats[-1].to_dict(format="ft")
        return ft_dict, self.meta

    # modifications

    def copy(self) -> "Thread":
        return Thread(
            *[x for x in self.chats],
            evals=self.evals,
            model=self.model,
            title="Copy: " + self.title,
            **self.meta,
        )

    def append(self, message: Message):
        self.chats.append(message)
        return self

    def pop(self, message: Message = None):
        if message:
            if isinstance(message, int):
                self.chats.pop(message)
            elif isinstance(message, Message):
                for i, x in enumerate(self.chats):
                    if x.id == message.id:
                        self.chats.pop(i)
                        break
        else:
            self.chats.pop()
        return self

    # actions
    def _eval(self, out):
        if self.evals:
            evals = {}
            for k, e in self.evals.items():
                evals[k] = tu.json_logic(e, {"response": out})
            return evals

    # def step_streaming(self, model: ModelInterface, /, eval: bool = False):
    #     out = ""
    #     for x in model.stream_chat(self):
    #         yield x
    #         if isinstance(x, dict):
    #             out = x
    #         else:
    #             out += x
    #     if eval:
    #         yield self._eval(out)
    #     self.append(assistant(out))

    # def step(self, model: ModelInterface, /, eval: bool = False):
    #     out = model.chat(self)
    #     self.append(assistant(out))
    #     if eval:
    #         return out, self._eval(out)
    #     return out


########################################################################################################################
#
# The code in this section contains a default model interface that each model API has to provide. All the APIs should
# follow this interface to be compatible with the chat API.
#
########################################################################################################################


class ModelInterface(ABC):
    """This is the generic abstract interface implemented by all the model APIs"""

    model_id: str
    """This is the model ID for the model"""

    api_token: str
    """This is the API token for the model"""

    extra_headers: dict[str, Any]
    """This is the placeholder for any extra headers to be passed during request"""

    base_url: str
    """This is the default URL that has to be pinged. This may not be the REST endpoint URL but anything"""

    client: httpx.Client | None
    """This is the client that is used to make the requests"""

    async_client: httpx.AsyncClient | None
    """This is the async client that is used to make the requests"""

    def __init__(self):
        self.model_id = ""
        self.api_token = ""
        self.extra_headers = {}
        self.base_url = ""
        self.client = None
        self.async_client = None

    def __repr__(self):
        return f"ta.{self.__class__.__name__}('{self.model_id}')"

    def set_api_token(self, token: str) -> None:
        """This are used to set the API token for the model"""
        self.api_token = token

    def set_async_client(self, client: httpx.AsyncClient | None = None):
        if client is None:
            client = httpx.AsyncClient()
        self.async_client = client

    def set_client(self, client: httpx.Client | None = None):
        if client is None:
            client = httpx.Client()
        self.client = client

    # Chat methods

    @abstractmethod
    def stream_chat(
        self,
        chats: Thread | str,
        model: str | None = None,
        max_tokens: int = None,
        temperature: float = 1,
        token: str | None = None,
        usage: bool = False,
        extra_headers: dict[str, str] | None = None,
        debug: bool = False,
        raw: bool = False,
        timeout=(5, 60),
        **kwargs,
    ):
        """This is the blocking function to stream chat with the model where each token is iteratively generated"""
        pass

    @abstractmethod
    def chat(
        self,
        chats: Thread | str,
        model: str | None = None,
        max_tokens: int = None,
        temperature: float = 1,
        parallel_tool_calls: bool = False,
        token: str | None = None,
        usage: bool = False,
        extra_headers: dict[str, str] | None = None,
        debug: bool = False,
        timeout=(5, 60),
        **kwargs,
    ) -> str | dict[str, Any]:
        """This is the blocking function to block chat with the model"""
        pass

    @abstractmethod
    async def stream_chat_async(
        self,
        chats: Thread | str,
        model: str | None = None,
        max_tokens: int = None,
        temperature: float = 1,
        token: str | None = None,
        usage: bool = False,
        extra_headers: dict[str, str] | None = None,
        debug: bool = False,
        raw: bool = False,
        timeout=(5, 60),
        **kwargs,
    ) -> str | dict[str, Any]:
        """This is the async function to stream chat with the model where each token is iteratively generated"""
        pass

    @abstractmethod
    async def chat_async(
        self,
        chats: Thread | str,
        model: str | None = None,
        max_tokens: int = None,
        temperature: float = 1,
        parallel_tool_calls: bool = False,
        token: str | None = None,
        usage: bool = False,
        extra_headers: dict[str, str] | None = None,
        debug: bool = False,
        timeout=(5, 60),
        **kwargs,
    ) -> str | dict[str, Any]:
        """This is the async function to block chat with the model"""
        pass

    @abstractmethod
    def distributed_chat(
        self,
        prompts: list[Thread],
        post_logic: Callable | None = None,
        max_threads: int = 10,
        retry: int = 3,
        pbar=True,
        debug=False,
        **kwargs,
    ):
        """This is the blocking function to chat with the model in a distributed manner"""
        pass

    @abstractmethod
    async def distributed_chat_async(
        self,
        prompts: list[Thread],
        post_logic: Callable | None = None,
        max_threads: int = 10,
        retry: int = 3,
        pbar=True,
        debug=False,
        **kwargs,
    ):
        """This is the async function to chat with the model in a distributed manner"""
        pass

    # Embedding methods

    @abstractmethod
    def embedding(
        self,
        chats: Thread | list[str] | str,
        model: str,
        token: str | None,
        timeout: tuple[int, int],
        raw: bool,
        extra_headers: dict[str, str] | None,
    ) -> "EmbeddingGen":
        """This is the blocking function to get embeddings for the chat"""
        pass

    @abstractmethod
    async def embedding_async(
        self,
        chats: Thread | list[str] | str,
        model: str,
        token: str | None,
        timeout: tuple[int, int],
        raw: bool,
        extra_headers: dict[str, str] | None,
    ) -> "EmbeddingGen":
        """This is the async function to get embeddings for the chat"""
        pass

    # Image methods

    @abstractmethod
    def image_gen(
        self,
        prompt: str,
        style: str,
        model: str,
        n: int,
        size: str,
        **kwargs,
    ) -> "ImageGen":
        """This is the blocking function to generate images"""
        pass

    @abstractmethod
    async def image_gen_async(
        self,
        prompt: str,
        style: str,
        model: str,
        n: int,
        size: str,
        **kwargs,
    ) -> "ImageGen":
        """This is the async function to generate images"""
        pass

    # Speech methods
    @abstractmethod
    def text_to_speech(
        self,
        prompt: str,
        voice: str = "shimmer",
        model="tts-1",
        response_format="wav",
        extra_headers: dict[str, str] | None = None,
        timeout: tuple[int, int] = (5, 60),
        **kwargs,
    ) -> bytes:
        """This is the blocking function to convert text to speech"""
        pass

    @abstractmethod
    async def text_to_speech_async(
        self,
        prompt: str,
        voice: str = "shimmer",
        model="tts-1",
        response_format="wav",
        extra_headers: dict[str, str] | None = None,
        timeout: tuple[int, int] = (5, 60),
        **kwargs,
    ) -> bytes:
        """This is the async function to convert text to speech"""
        pass

    @abstractmethod
    def speech_to_text(
        self,
        prompt: str,
        audio: str,
        model: str,
        timestamp_granularities: list[str],
        **kwargs,
    ) -> "Transcript":
        """This is the blocking function to convert speech to text"""
        pass

    @abstractmethod
    async def speech_to_text_async(
        self,
        prompt: str,
        audio: str,
        model: str,
        timestamp_granularities=["segment"],
        **kwargs,
    ) -> "Transcript":
        """This is the async function to convert speech to text"""
        pass

    # batching

    @abstractmethod
    def submit_batch(
        self,
        threads: list[Thread | str],
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float | None = None,
        token: str | None = None,
        debug: bool = False,
        extra_headers: dict[str, str] | None = None,
        timeout=(5, 30),
        raw: bool = False,
        **kwargs,
    ) -> tuple[str, list[str]] | dict:
        """This is the blocking function to submit a batch of threads. It will return the batch_id and custom_ids
        for ordering the responses"""
        pass

    @abstractmethod
    def get_batch(
        self,
        batch_id: str,
        custom_ids: list[str] | None = None,
        token: str | None = None,
        raw: bool = False,
    ) -> tuple[list[Any] | dict, str | None]:
        """This is the blocking function to get the batch results"""
        pass


class Usage:
    def __init__(
        self,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int = 0,
        **kwargs,
    ):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cached_tokens = cached_tokens
        self.total_tokens = input_tokens + output_tokens
        self.extra = kwargs

    def __getitem__(self, x):
        return getattr(self, x)

    def __repr__(self) -> str:
        return f"<Usage: {self.input_tokens} [Cached: {self.cached_tokens}] -> {self.output_tokens}>"

    def __radd__(self, other: "Usage"):
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
        )

    def __add__(self, other: "Usage"):
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
        )

    def to_json(self, *a, **k) -> str:
        return tu.to_json(self.__dict__, *a, **k)

    def cost(
        self,
        input_token_per_million: float,
        cache_token_per_million: float,
        output_token_per_million: float,
    ) -> float:
        return (
            self.input_tokens * input_token_per_million / 1e6
            + self.cached_tokens * cache_token_per_million / 1e6
            + self.output_tokens * output_token_per_million / 1e6
        )


########################################################################################################################
#
# The code here is the ultimate representation of thoughts and processes. A tree of messages is a ThreadsTree.
#
########################################################################################################################


class ThreadsTree:
    """
    This is the tree representation of a thread, where each node is a Message object. Useful for regeneration and
    searching through a tree of conversations. This is a container providing all the necessary APIs.
    """

    def __init__(self, *msgs: list[list[Message] | Message], id: str = None):
        system = None
        if len(msgs):
            if isinstance(msgs[0], Message) and msgs[0].role == Message.SYSTEM:
                system = msgs[0].value
            elif isinstance(msgs[0], str):
                system = msgs[0]
        if system:
            msgs = msgs[1:]

        self.id = id or "tree_" + str(tu.get_snowflake())
        self.system = system
        self.msg_counter = 0  # monotonically increasing counter
        self.messages_map = {}
        self.messages: dict[str, Message] = {}

        self.tree = nt.Tree()
        if msgs:
            self._add_children_to_parent(self.tree, msgs)

    def __repr__(self) -> str:
        return f"<ThreadsTree: {self.id} " + self.tree.format() + ">"

    def __getitem__(self, x) -> Message:
        try:
            if type(x) == int:
                return self.messages[self.messages_map[x]]
            elif type(x) == str:
                return self.messages[x]
            elif isinstance(x, Message):
                return self.messages[x.id]
            elif isinstance(x, nt.Node):
                return self.messages[x.data_id]
        except KeyError:
            raise ValueError(f"Message with id '{x}' not found")
        raise ValueError(f"Unknown type: {type(x)}")

    def _get_parent_message(self, message: Message) -> Message:
        if not isinstance(message, Message):
            message = self[message]
        parent_node = self.tree.find(data_id=message.id).parent
        return self.messages[parent_node.data_id]

    def _get_parent_node(self, message: Message) -> nt.Node:
        message = self._get_parent_message(message)
        return self.tree.find(data_id=message.id).parent

    def _add_children_to_parent(
        self,
        parent_node: nt.Tree | nt.Node,
        children: list[list[Message] | Message],
    ) -> None:
        for child in children:
            if isinstance(child, Message):
                if not isinstance(parent_node, nt.Tree):
                    lm = self.messages[parent_node.data_id]
                    if lm.role == child.role:
                        raise ValueError(
                            f"Same consecutive roles: {self.latest_message.role} -> {child.role}"
                        )

                parent_node = parent_node.add(
                    f"[{self.msg_counter:02d}] " + str(child),
                    data_id=child.id,
                )
                self.messages_map[self.msg_counter] = child.id
                self.messages[child.id] = child
                self.msg_counter += 1
            if isinstance(child, list):
                # if this is a list then there are two possibilities:
                # - regeneration: all items are assistant type
                # - reprompt: all items are human type
                self._add_children_to_parent(parent_node, child)

    @property
    def latest_node(self) -> nt.Node:
        done = False
        if not self.msg_counter:
            return self.tree
        cntr = copy.deepcopy(self.msg_counter)
        while not done:
            try:
                return self.tree.find(data_id=self.messages_map[cntr - 1])
            except KeyError:
                cntr -= 1
                if cntr == 0:
                    done = True
        raise ValueError("No latest node found")

    @property
    def latest_message(self) -> Message:
        ln = self.latest_node
        if isinstance(ln, nt.Tree):
            return None
        return self.messages[ln.data_id]

    @property
    def degree_of_tree(self) -> int:
        # The degree of a tree is the maximum degree of a node in the tree.
        degree = len(self.tree.children)

        def _update_degree(node, _):
            nonlocal degree
            degree = max(degree, len(node.children))

        self.tree.visit(_update_degree)
        return degree

    @property
    def size(self) -> int:
        # Number of nodes in the tree.
        return len(self.messages)

    @property
    def breadth(self) -> int:
        # The number of leaves.
        brt = 0

        def _update_breadth(node, _):
            nonlocal brt
            if not node.children:
                # A leaf, by definition, has degree zero
                brt += 1

        self.tree.visit(_update_breadth)
        return brt

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "system": self.system,
            "messages": [x.to_dict(format="full") for x in self.messages.values()],
            "tree": self.tree.to_dict_list(),
            "messages_map": self.messages_map,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ThreadsTree":
        tree = cls()
        tree.id = data.get("id", "tree_" + str(tu.get_snowflake()))
        tree.system = data["system"]
        tree.tree = nt.Tree.from_dict(data["tree"])
        messages = [Message.from_dict(x) for x in data["messages"]]
        tree.messages = {x.id: x for x in messages}
        tree.messages_map = data["messages_map"]
        tree.msg_counter = len(tree.messages) + 1
        return tree

    def copy(self) -> "ThreadsTree":
        return ThreadsTree.from_dict(self.to_dict())

    def add(self, child: Message, to: Message = "root") -> "ThreadsTree":
        if child.id in self.messages:
            raise ValueError(
                f"Message with id '{child.id}' already exists. Cycle detected."
            )
        if to is None:
            # find the latest inserted message and just add this as a child
            node = self.latest_node
        elif to == "root":
            node = self.tree
        else:
            to = self[to]
            node = self.tree.find(data_id=to.id)

        print("******", node, child)
        self._add_children_to_parent(node, [child])
        return self

    def delete(self, from_: Message) -> "ThreadsTree":
        from_ = self[from_]
        if from_.id not in self.messages:
            raise ValueError(
                f"Parent with id {from_.id} not found. Insert parent first."
            )
        from_node = self.tree.find(data_id=from_.id)

        messages_map_inv = {v: k for k, v in self.messages_map.items()}

        def _cleanup(node, _):
            del self.messages[node.data_id]
            del self.messages_map[messages_map_inv[node.data_id]]

        if from_node.children:
            from_node.visit(_cleanup, add_self=True)
        else:
            del self.messages[from_node.data_id]
            del self.messages_map[messages_map_inv[from_node.data_id]]
        from_node.remove(keep_children=False)
        return self

    def undo(self) -> "ThreadsTree":
        return self.delete(self.latest_message)

    def pick(self, to: Message = None, from_: Message = None) -> Thread:
        """
        A poerful methods to get a thread from the Tree srtucture by telling ``to`` and ``from_`` in the tree
        """
        if self.system:
            thread = Thread(system(self.system))
        else:
            thread = Thread()
        to = self.latest_message if to is None else self[to]
        if to is None:
            return thread
        to_node = self.tree.find(data_id=to.id)
        if from_ is not None:
            from_ = self[from_]
            from_node = self.tree.find(data_id=from_.id)
            if not from_node.is_ancestor_of(to_node):
                raise ValueError(
                    f"Message with id '{from_.id}' is not an ancestor of '{to.id}'. Cannot build a thread."
                )
            from_parents = set(x.data_id for x in from_node.get_parent_list())
        else:
            from_parents = set()
        for p in to_node.get_parent_list():
            if p.data_id in from_parents:
                continue
            thread.append(self.messages[p.data_id])
        return thread

    def step_stream(
        self, api: ModelInterface, /, from_: Message
    ) -> Generator[Message, None, None]:
        latest_thread = self.pick()
        if latest_thread.chats[-1].role == from_.role:
            raise ValueError(
                f"Cannot step towards same role '{from_.role}' to '{latest_thread.chats[-1].role}'"
            )
        latest_thread.append(from_)

        for token in api.stream_chat(latest_thread):
            if isinstance(token, dict):
                fnc = token
            else:
                yield token

        if fnc:
            yield function_call(fnc)
            yield function_resp("... I have been generated ...")

    def step(self, api: ModelInterface, /, from_: Message) -> Message:
        out = ""
        fnc = None
        for token in self.step_stream(api, from_):
            if isinstance(token, dict):
                fnc = token
            else:
                out += token

    def regenerate_stream(
        self,
        api: ModelInterface,
        /,
        from_: Message = None,
        prompt: str = None,
        dry: bool = False,
        **api_kwargs,
    ):
        if from_ is None:
            from_ = self.latest_message
        else:
            from_ = self[from_]

        if from_.role == Message.HUMAN:
            # if we are regenerating for a human, then we need to add a prompt to the tree and then regenerate
            if not prompt:
                raise ValueError(
                    f"Regenerating for role 'human' but no ``prompt`` provided. pass ``prompt``"
                )
            if type(prompt) == str:
                prompt = human(prompt)
            self.add(prompt, to=self._get_parent_message(from_))
            thread = Thread()
            for x in self.tree.find(data_id=prompt.id).get_parent_list():
                thread.append(self.messages[x.data_id])
            thread.append(prompt)
        else:
            # if regenerating AI response then we just need to get till the parent message because that is guaranteed
            # to be a human
            if prompt:
                raise ValueError(
                    f"Regenerating for role 'gpt' but ``prompt`` provided. remove ``prompt``"
                )
            thread = Thread()
            for x in self.tree.find(data_id=from_.id).get_parent_list():
                thread.append(self.messages[x.data_id])

        if dry:
            stream = (x + " " for x in "... I have been generated ...".split())
        else:
            stream = api.stream_chat(thread, **api_kwargs)

        full_str = ""
        for token in stream:
            yield token
            if isinstance(token, dict):
                raise ValueError("Function call occured, not sure what to do")
            full_str += token
        self.add(assistant(full_str), to=thread.chats[-1])

    def regenerate(
        self,
        api: ModelInterface,
        /,
        from_: Message = None,
        prompt: str = None,
        dry: bool = False,
        **api_kwargs,
    ):
        return "".join(
            list(
                self.regenerate_stream(
                    api,
                    from_,
                    prompt=prompt,
                    dry=dry,
                    **api_kwargs,
                )
            )
        )

    class ROLLOUT:
        Continue = "continue"
        OneMoreRanker = "one_more_ranker"
        StopRollout = "stop_rollout"

    def rollout(
        self,
        /,
        message_gen_fn: callable = None,
        value_fn: callable = None,
        from_: Message = None,
        max_rollouts: int = 20,
        depth: int = 5,
        children: int = 5,
        retry: int = 1,
    ):
        # perform a full on rollout of the tree and provide necesary callbacks. The underlying threads contain
        # all the necessary information to perform the rollouts
        raise NotImplementedError("Not implemented yet, contact developers if urgent!")


########################################################################################################################
#
# Modalities
# ==========
#
# The code in this section is for the different modalities that are supported by the library.
#
########################################################################################################################

########################################################################################################################
# Embedding


class EmbeddingGen(BM):
    embedding: list[list[float]] = F("The generated embedding as a list of floats.")


########################################################################################################################
# Image


class ImageGen(BM):
    image: ImageType = F("The generated image in PIL.Image format.")

    class Config:
        arbitrary_types_allowed = True


########################################################################################################################
# Audio


class WebVTTCue(BM):
    start: str = F("The start time of the cue.")
    end: str = F("The end time of the cue.")
    text: str = F("The text of the cue.")


class Transcript(BM):
    segments: list[WebVTTCue] = F(
        "A list of WebVTTCue objects representing the audio segments."
    )

    @property
    def text(self):
        return "\n".join(
            [f"{cue.start} --> {cue.end}\t{cue.text}" for cue in self.segments]
        )

    @classmethod
    def from_text(cls, text: str):
        cues = []
        lines = text.strip().split("\n")

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # Skip WEBVTT header and blank lines
            if line == "WEBVTT" or not line:
                i += 1
                continue

            # Extract the timestamp
            match = re.match(
                r"(\d{2}:\d{2}:\d{2}\.\d{3}) --> (\d{2}:\d{2}:\d{2}\.\d{3})", line
            )
            if match:
                start_time = match.group(1)
                end_time = match.group(2)
                i += 1

                # Collect text for this cue
                text_lines = []
                while (
                    i < len(lines)
                    and lines[i].strip()
                    and not re.match(
                        r"(\d{2}:\d{2}:\d{2}\.\d{3}) --> (\d{2}:\d{2}:\d{2}\.\d{3})",
                        lines[i].strip(),
                    )
                ):
                    text_lines.append(lines[i].strip())
                    i += 1
                text = " ".join(text_lines)

                cue = WebVTTCue(
                    start=start_time,
                    end=end_time,
                    text=text,
                )
                cues.append(cue)
            else:
                i += 1
        return cls(segments=cues)

    def to(self, format: str = "text"):
        if format == "vtt":
            return self.text
        elif format == "srt":
            return "\n".join(
                [
                    f"{i + 1}\n{cue.start.replace('.', ',')} --> {cue.end.replace('.', ',')}\n{cue.text}"
                    for i, cue in enumerate(self.segments)
                ]
            )
        elif format == "text":
            return "\n".join([f"{cue.text}" for cue in self.segments])
        else:
            raise ValueError(f"Unsupported format: {format}")
