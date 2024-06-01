# Copyright © 2023- Frello Technology Private Limited

import io
import os
import json
import copy
import random
from functools import partial
from collections.abc import Iterable
from typing import Dict, List, Any, Tuple, Optional, Generator, Union
import nutree as nt

import tuneapi.utils as tu


class Tool:

    class Prop:
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
        parameters = []
        for k, v in x["parameters"].get("properties", {}).items():
            parameters.append(cls.Prop(name=k, **v))
        return cls(
            name=x["name"],
            description=x["description"],
            parameters=parameters,
        )


class Message:
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
        # functions
        "function_call": FUNCTION_CALL,
        "function-call": FUNCTION_CALL,
        "function_resp": FUNCTION_RESP,
        "function-resp": FUNCTION_RESP,
    }

    # start initialization here
    def __init__(
        self,
        value: str | float | List[Dict[str, Any]],
        role: str,
        id: str = None,
        fn_pairs: Optional[Tuple["Message", "Message"]] = None,
        **kwargs,
    ):
        if role not in self.KNOWN_ROLES:
            raise ValueError(f"Unknown role: {role}. Update dictionary `KNOWN_ROLES`")
        if value is None:
            raise ValueError("value cannot be None")

        self.role = self.KNOWN_ROLES[role]
        self.value = value
        self.id = id or "msg_" + str(tu.get_snowflake())
        self.metadata = kwargs
        self.fn_pairs = fn_pairs

    def __str__(self) -> str:
        try:
            idx = max(os.get_terminal_size().columns - len(self.role) - 40, 10)
        except OSError:
            idx = 50
        return f"<{self.role}: {json.dumps(self.value)[:idx]}>"

    def __radd__(self, other: str):
        return Message(self.value + other, self.role)

    def __add__(self, other: str):
        return Message(self.value + other, self.role)

    def __repr__(self) -> str:
        out = ""
        if self.fn_pairs:
            for fc, fr in self.fn_pairs:
                out += f"[[FC] {fc} => [FR] {fr}]"
        if out:
            out += " " + str(self.value)
        else:
            out = str(self.value)
        return out

    def __getitem__(self, x):
        if x == "content":
            return self.value
        return getattr(self, x)

    def to_dict(
        self,
        format: Optional[str] = None,
        meta: bool = False,
    ):  #  ft: bool = False
        """
        if format == `ft` then export to following format: `{"from": "system/human/gpt", "value": "..."}`
        elif format == `api` then `{"role": "system/user/assistant", "content": [{"type": "text", "text": {"value": "..."}]}`
        elif format == `full` then `{"id": 1234421123, "role": "system/user/assistant", "content": [{"type": "text", "text": {"value": "..."}]}`
        else export to following format: `{"role": "system/user/assistant", "content": "..."}`
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
            chat_message["content"] = [{"type": "text", "text": {"value": self.value}}]
        else:
            chat_message["content"] = self.value

        if meta:
            chat_message["metadata"] = self.metadata

        if full:
            chat_message["id"] = self.id

        return chat_message

    @classmethod
    def from_dict(cls, data):
        return cls(
            value=data.get("value") or data.get("content"),
            role=data.get("from") or data.get("role"),
            id=data.get("id"),
            **data.get("metadata", {}),
        )  # type: ignore


### Aliases
human = partial(Message, role=Message.HUMAN)
system = partial(Message, role=Message.SYSTEM)
assistant = partial(Message, role=Message.GPT)
function_call = partial(Message, role=Message.FUNCTION_CALL)
function_resp = partial(Message, role=Message.FUNCTION_RESP)


class Thread:
    """
    If the last Message is a "value".

    Args:
        chats (List[Message]): List of chat messages
        jl (Dict[str, Any]): Optional json-logic
    """

    def __init__(
        self,
        *chats: Union[List[Message], Message],
        jl: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        id: str = "",
        title: str = "",
        tools: List[Tool] = [],
        **kwargs,
    ):
        self.chats = list(chats)
        self.jl = jl
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
        x += "\n>"
        return x

    def __getattr__(self, __name: str) -> Any:
        if __name in self.meta:
            return self.meta[__name]
        raise AttributeError(f"Attribute {__name} not found")

    def __getitem__(self, __x) -> Any:
        return self.chats[__x]

    # ser/deser

    def to_dict(self, full: bool = False):
        if full:
            return {
                "chats": [x.to_dict() for x in self.chats],
                "jl": self.jl,
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
        chats = data.get("chats", []) or data.get("conversations", [])
        if not chats:
            raise ValueError("No chats found")
        return cls(
            *[Message.from_dict(x) for x in chats],
            id=data.get("id", ""),
            jl=data.get("jl", ""),
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
            chats=[x for x in self.chats],
            jl=self.jl,
            model=self.model,
            title="Copy: " + self.title,
            **self.meta,
        )

    def add(self, message: Message):
        self.chats.append(message)

    def append(self, message: Message):
        self.chats.append(message)


class TreeThread:
    """
    This is the tree representation of a thread, where each node is a Message object. Useful for regeneration and
    searching through a tree of conversations. This is a container providing all the necessary APIs.
    """

    def __init__(
        self, *msgs: Union[List[Union[List, Message]], Message], id: str = None
    ):
        system = ""
        if (
            len(msgs)
            and isinstance(msgs[0], Message)
            and msgs[0].role == Message.SYSTEM
        ):
            system = msgs[0].value
        if system:
            msgs = msgs[1:]

        self.id = id or "tree_" + str(tu.get_snowflake())
        self.system = system
        self.msg_counter = 0  # monotonically increasing counter
        self.messages_map = {}
        self.messages = {}

        self.tree = nt.Tree(name=self.id)
        if msgs:
            self._add_children_to_parent(self.tree, msgs)

    def __repr__(self) -> str:
        if self.system and " <system>" not in self.tree.name:
            self.tree.name += " <system>"
        elif not self.system and self.tree.name.endswith(" <system>"):
            self.tree.name = self.tree.name[:-9]
        return self.tree.format()

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
        return self.messages[self.latest_node.data_id]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "system": self.system,
            "messages": [x.to_dict(format="full") for x in self.messages.values()],
            "tree": self.tree.to_dict_list(),
            "messages_map": self.messages_map,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TreeThread":
        tree = cls()
        tree.id = data["id"]
        tree.system = data["system"]
        tree.tree = nt.Tree.from_dict(data["tree"])
        messages = [Message.from_dict(x) for x in data["messages"]]
        tree.messages = {x.id: x for x in messages}
        tree.messages_map = data["messages_map"]
        tree.msg_counter = len(tree.messages) + 1
        return tree

    def add(self, child: Message, to: Message = None) -> "TreeThread":
        if child.id in self.messages:
            raise ValueError(
                f"Message with id '{child.id}' already exists. Cycle detected."
            )
        if to is None:
            # find the latest inserted message and just add this as a child
            to = self.latest_message
        to = self[to]
        if to.id not in self.messages:
            raise ValueError(f"Parent with id {to.id} not found. Insert parent first.")

        self._add_children_to_parent(
            self.tree.find(data_id=to.id),  # find the actual nutree.Node
            [child],
        )
        return self

    def delete(self, from_: Message) -> "TreeThread":
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

    def undo(self) -> "TreeThread":
        return self.delete(self.latest_message)

    def regenerate_stream(
        self,
        from_: Message,
        api: object,
        prompt: str = None,
        dry: bool = False,
        **api_kwargs,
    ):
        # validation on inputs for regeneration
        from_ = self[from_]

        if from_.role == Message.HUMAN:
            # if we are regenerating for a human, then we need to add a prompt to the tree and then regenerate
            if not prompt:
                raise ValueError(
                    f"Regenerating for role 'human' but no `prompt` provided. pass `prompt`"
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
                    f"Regenerating for role 'gpt' but `prompt` provided. remove `prompt`"
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
        from_: Message,
        api: object,
        prompt: str = None,
        dry: bool = False,
        **api_kwargs,
    ):
        return "".join(
            list(
                self.regenerate_stream(
                    from_,
                    api,
                    prompt=prompt,
                    dry=dry,
                    **api_kwargs,
                )
            )
        )


# these are the classes that we use for tune datasets from r-stack


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

    def stream(self) -> Generator[Thread, None, None]:
        for x in self:
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

    def append(self, __object: Any) -> None:
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
            raise ImportError("Install numpy to use `create_te_split` method")

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

    def to_disk(self, folder: str, fmt: Optional[str] = None):
        if fmt:
            tu.logger.warn(
                f"exporting to {fmt} format, you cannot recreate the dataset from this."
            )
        os.makedirs(folder)
        with open(f"{folder}/tuneds.jsonl", "w") as f:
            for sample in self.items:
                if fmt == "sharegpt":
                    item, _ = sample.to_ft()
                elif fmt is None:
                    item = sample.to_dict()
                else:
                    raise ValueError(f"Unknown format: {fmt}")
                f.write(tu.to_json(item, tight=True) + "\n")  # type: ignore

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

    # properties

    def can_train_koro_regression(self) -> bool:
        return all(["koro.regression" in x.meta for x in self])


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
        config = {}
        config["type"] = "tune"
        config["hf_type"] = fmt
        os.makedirs(folder)
        self.train_ds.to_disk(f"{folder}/train", fmt=fmt)
        self.eval_ds.to_disk(f"{folder}/eval", fmt=fmt)
        tu.to_json(config, fp=f"{folder}/tune_config.json", tight=True)

    @classmethod
    def from_disk(cls, folder: str):
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
