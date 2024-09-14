"""
This file contains all the datatypes relevant for a chat conversation. In general this is the nomenclature that we follow:
    * Message: a unit of information produced by a ``role``
    * Thread: a group of messages is called a thread. There can be many 2 types of threads, linear and tree based.
    * ThreadsList: a group of linear threads is called a threads list.
    * Dataset: a container for grouping threads lists is called a dataset

Almost all the classes contain ``to_dict`` and ``from_dict`` for serialisation and deserialisation.
"""

# Copyright © 2023- Frello Technology Private Limited

import io
import os
import json
import copy
import random
from PIL.Image import Image
from functools import partial
from collections.abc import Iterable
from typing import Dict, List, Any, Tuple, Optional, Generator, Union
import nutree as nt

import tuneapi.utils as tu


########################################################################################################################
#
# The code in this section contains the primitive of this new chat API. The ``Tool`` class defines tools that the model
# can predict. The ``Message`` class defines the container for storing the chat messages.
#
########################################################################################################################


class Tool:
    """A tool is a container for telling the LLM what it can do. This is a standard definition."""

    class Prop:
        """
        An individual property is called a prop.
        """

        def __init__(
            self,
            name: str,
            description: str,
            type: str,
            required: bool = False,
            items: Optional[Dict] = None,
            enum: Optional[List[str]] = None,
        ):
            self.name = name
            self.description = description
            self.required = required
            self.type = type
            self.items = items
            self.enum = enum

        def __repr__(self) -> str:
            return f"<Tool.Prop: " + ("*" if self.required else "") + f"{self.name}>"

    def __init__(
        self,
        name: str,
        description: str,
        parameters: List["Tool.Prop"],
    ):
        self.name = name
        self.description = description
        self.parameters = parameters

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
            parameters.append(cls.Prop(name=k, **v))
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
        value: str | List[Dict[str, Any]],
        role: str,
        images: List[str | Image] = [],
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
            if isinstance(img, Image):
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
        format: Optional[str] = None,
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

        chat_message: Dict[str, str | float]
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
# The code in this section contains a default model interface that each model API has to provide. All the APIs should
# follow this interface to be compatible with the chat API.
#
########################################################################################################################


class ModelInterface:
    """This is the generic interface implemented by all the model APIs"""

    model_id: str
    """This is the model ID for the model"""

    api_token: str
    """This is the API token for the model"""

    def set_api_token(self, token: str) -> None:
        """This are used to set the API token for the model"""
        raise NotImplementedError("This model has no operation for this.")

    def set_org_id(self, org_id: str) -> None:
        """This are used to set the Organisation ID for the model"""
        raise NotImplementedError("This model has no operation for this.")

    def chat(
        self,
        chats: "Thread",
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 1,
        token: Optional[str] = None,
        timeout=(5, 30),
        extra_headers: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> str | Dict[str, Any]:
        """This is the main function to block chat with the model"""

    def stream_chat(
        self,
        chats: "Thread",
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 1,
        token: Optional[str] = None,
        timeout=(5, 60),
        raw: bool = False,
        debug: bool = False,
        extra_headers: Optional[Dict[str, str]] = None,
    ):
        """This is the main function to stream chat with the model where each token is iteratively generated"""


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
        *chats: List of chat ``Message`` objects
        evals: JSON logic and
    """

    def __init__(
        self,
        *chats: Union[List[Message], Message],
        evals: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        id: str = "",
        title: str = "",
        tools: List[Tool] = [],
        **kwargs,
    ):
        self.chats = list(chats)
        self.evals = evals
        self.model = model
        self.id = id or "thread_" + str(tu.get_snowflake())
        self.title = title
        self.tools = tools

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
            }
        return {
            "chats": [x.to_dict() for x in self.chats],
            "tools": [x.to_dict() for x in self.tools],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Thread":
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
            **data.get("meta", {}),
        )

    def to_ft(
        self, id: Any = None, drop_last: bool = False
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
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
# The code here is the ultimate representation of thoughts and processes. A tree of messages is a ThreadsTree.
#
########################################################################################################################


class ThreadsTree:
    """
    This is the tree representation of a thread, where each node is a Message object. Useful for regeneration and
    searching through a tree of conversations. This is a container providing all the necessary APIs.
    """

    def __init__(
        self, *msgs: Union[List[Union[List, Message]], Message], id: str = None
    ):
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
        self.messages = {}

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
        parent_node: Union[nt.Tree, nt.Node],
        children: List[Union[List, Message]],
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
        cntr = copy.deepcopy(self.msg_counter)
        if not cntr:
            return self.tree
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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "system": self.system,
            "messages": [x.to_dict(format="full") for x in self.messages.values()],
            "tree": self.tree.to_dict_list(),
            "messages_map": self.messages_map,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ThreadsTree":
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
# The code in this section is copied from another repository which was originally used for research (r-stack)
#
########################################################################################################################


class ThreadsList(list):
    """This class implements some basic container methods for a list of Chat objects"""

    def __init__(self):
        self.keys = {}
        self.items: List[Thread] = []
        self.idx_dict: Dict[int, Tuple[Any, ...]] = {}
        self.key_to_items_idx: Dict[int, List[int]] = {}

    def __repr__(self) -> str:
        return f"ThreadsList(unq_keys={len(self.key_to_items_idx)}, items={len(self.items)})"

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self) -> Generator[Thread, None, None]:
        for x in self.items:
            yield x

    def __getitem__(self, __index) -> List[Thread]:
        return self.items[__index]

    def table(self) -> str:
        try:
            from tabulate import tabulate
        except ImportError:
            raise ImportError("Install tabulate to use this method")

        table = []
        for k, v in self.idx_dict.items():
            table.append(
                [
                    *v,
                    len(self.key_to_items_idx[k]),
                    f"{len(self.key_to_items_idx[k])/len(self)*100:0.2f}%",
                ]
            )
        return tabulate(table, headers=[*list(self.keys), "count", "percentage"])

    # data manipulation

    def append(self, __object: Thread) -> None:
        if not self.items:
            self.keys = __object.meta.keys()
        if self.keys != __object.meta.keys():
            raise ValueError("Keys should match")
        self.idx_dict.setdefault(__object.value_hash, __object.values)
        self.key_to_items_idx.setdefault(__object.value_hash, [])
        self.key_to_items_idx[__object.value_hash].append(len(self.items))
        self.items.append(__object)

    def add(self, x: Thread):
        return self.append(x)

    def extend(self, __iterable: Iterable) -> None:
        if hasattr(__iterable, "items"):
            for x in __iterable.items:  # type: ignore
                self.append(x)
        elif isinstance(__iterable, Iterable):
            for x in __iterable:
                self.append(x)
        else:
            raise ValueError("Unknown iterable")

    def shuffle(self, seed: Optional[int] = None) -> None:
        """Perform in place shuffle"""
        # shuffle using indices, self.items and self.key_to_items_idx
        idx = list(range(len(self.items)))
        if seed:
            rng = random.Random(seed)
            rng.shuffle(idx)
        else:
            random.shuffle(idx)
        self.items = [self.items[i] for i in idx]
        self.key_to_items_idx = {}
        for i, x in enumerate(self.items):
            self.key_to_items_idx.setdefault(x.value_hash, [])
            self.key_to_items_idx[x.value_hash].append(i)

    def create_te_split(
        self, test_items: int | float = 0.1
    ) -> Tuple["ThreadsList", ...]:
        try:
            import numpy as np
        except ImportError:
            raise ImportError("Install numpy to use ``create_te_split`` method")

        train_ds = ThreadsList()
        eval_ds = ThreadsList()
        items_np_arr = np.array(self.items)
        for k, v in self.key_to_items_idx.items():
            if isinstance(test_items, float):
                if int(len(v) * test_items) < 1:
                    raise ValueError(
                        f"Test percentage {test_items} is too high for the dataset key '{k}'"
                    )
                split_ids = random.sample(v, int(len(v) * test_items))
            else:
                if test_items > len(v):
                    raise ValueError(
                        f"Test items {test_items} is too high for the dataset key '{k}'"
                    )
                split_ids = random.sample(v, test_items)

            # get items
            eval_items = items_np_arr[split_ids]
            train_items = items_np_arr[np.setdiff1d(v, split_ids)]
            train_ds.extend(train_items)
            eval_ds.extend(eval_items)

        return train_ds, eval_ds

    # ser / deser

    def to_dict(self):
        return {"items": [x.to_dict() for x in self.items]}

    @classmethod
    def from_dict(cls, data):
        bench_dataset = cls()
        for item in data["items"]:
            bench_dataset.append(Thread.from_dict(item))
        return bench_dataset

    def to_disk(self, folder: str, fmt: Optional[str] = None, override: bool = False):
        if fmt:
            tu.logger.warn(
                f"exporting to {fmt} format, you cannot recreate the dataset from this."
            )
        os.makedirs(folder, exist_ok=override)
        fp = f"{folder}/tuneds.jsonl"
        with open(fp, "w") as f:
            for sample in self.items:
                if fmt == "ft":
                    item, _ = sample.to_ft()
                elif fmt == "full":
                    item = sample.to_dict(full=True)
                elif fmt is None:
                    item = sample.to_dict()
                else:
                    raise ValueError(f"Unknown format: {fmt}")
                f.write(tu.to_json(item, tight=True) + "\n")  # type: ignore
        return fp

    @classmethod
    def from_disk(cls, folder: str):
        bench_dataset = cls()
        with open(f"{folder}/tuneds.jsonl", "r") as f:
            for line in f:
                item = json.loads(line)
                bench_dataset.append(Thread.from_dict(item))
        return bench_dataset

    def to_hf_dataset(self) -> Tuple["datasets.Dataset", List]:  # type: ignore
        try:
            import datasets as dst
        except ImportError:
            raise ImportError("Install huggingface datasets library to use this method")

        _ds_list = []
        meta_list = []
        for x in self.items:
            sample, meta = x.to_ft()
            _ds_list.append(sample)
            meta_list.append(meta)
        return dst.Dataset.from_list(_ds_list), meta_list


class Dataset:
    """This class is a container for training and evaulation datasets, useful for serialising items to and from disk"""

    def __init__(self, train: ThreadsList, eval: ThreadsList):
        self.train_ds = train
        self.eval_ds = eval

    def __repr__(self) -> str:
        return f"Dataset(\n  train={self.train_ds},\n  eval={self.eval_ds}\n)"

    @classmethod
    def from_list(cls, items: List["Dataset"]):
        train_ds = ThreadsList()
        eval_ds = ThreadsList()
        for item in items:
            train_ds.extend(item.train_ds)
            eval_ds.extend(item.eval_ds)
        return cls(train=train_ds, eval=eval_ds)

    def to_hf_dict(self) -> Tuple["datasets.DatasetDict", Dict[str, List]]:  # type: ignore
        try:
            import datasets as dst
        except ImportError:
            raise ImportError("Install huggingface datasets library to use this method")

        train_ds, train_meta = self.train_ds.to_hf_dataset()
        eval_ds, eval_meta = self.eval_ds.to_hf_dataset()
        return dst.DatasetDict(train=train_ds, eval=eval_ds), {
            "train": train_meta,
            "eval": eval_meta,
        }

    def to_disk(self, folder: str, fmt: Optional[str] = None):
        """
        Serialise all the items of the container to a folder on the disk
        """
        config = {}
        config["type"] = "tune"
        config["hf_type"] = fmt
        os.makedirs(folder)
        self.train_ds.to_disk(f"{folder}/train", fmt=fmt)
        self.eval_ds.to_disk(f"{folder}/eval", fmt=fmt)
        tu.to_json(config, fp=f"{folder}/tune_config.json", tight=True)

    @classmethod
    def from_disk(cls, folder: str):
        """
        Deserialise and rebuild the container from a folder on the disk
        """
        if not os.path.exists(folder):
            raise ValueError(f"Folder '{folder}' does not exist")
        if not os.path.exists(f"{folder}/train"):
            raise ValueError(f"Folder '{folder}/train' does not exist")
        if not os.path.exists(f"{folder}/eval"):
            raise ValueError(f"Folder '{folder}/eval' does not exist")
        if not os.path.exists(f"{folder}/tune_config.json"):
            raise ValueError(f"File '{folder}/tune_config.json' does not exist")

        # not sure what to do with these
        config = tu.from_json(f"{folder}/tune_config.json")
        return cls(
            train=ThreadsList.from_disk(f"{folder}/train"),
            eval=ThreadsList.from_disk(f"{folder}/eval"),
        )
