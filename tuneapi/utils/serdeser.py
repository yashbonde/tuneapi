# Copyright © 2024- Frello Technology Private Limited

import os
import json
import cloudpickle
from typing import Any, Dict
from base64 import b64encode, b64decode
from google.protobuf.struct_pb2 import Struct


def to_json(x: dict, fp: str = "", indent=2, tight: bool = False) -> str:
    """
    Convert a dict to json string and write to file if ``fp`` is provided.

    Args:
        x (dict): The dict to convert
        fp (str, optional): The file path to write to. Defaults to "".
        indent (int, optional): The indentation level. Defaults to 2.
        tight (bool, optional): If true, remove all the whitespaces, ignores ``indent``. Defaults to False.

    Returns:
        Optional[str]: The json string if ``fp`` is not provided
    """
    kwargs: Dict[str, Any] = {}
    if tight:
        kwargs["separators"] = (",", ":")  # type: ignore
    else:
        kwargs["indent"] = indent
    if fp:
        with open(fp, "w") as f:
            f.write(json.dumps(x, **kwargs))
    else:
        return json.dumps(x, **kwargs)


def from_json(fp: str = "") -> Dict[str, Any]:
    """
    Load a JSON string or filepath and return a dictionary.

    Args:
        fp (str): The filepath or JSON-ified string

    Returns:

    """
    if os.path.exists(fp):
        with open(fp, "r") as f:
            return json.load(f)
    else:
        return json.loads(fp)


def to_pickle(obj, path):
    """Save an object to a pickle file

    Args:
      obj: object to save
      path: path to save to
    """
    with open(path, "wb") as f:
        cloudpickle.dump(obj, f)


def from_pickle(path):
    """Load an object from a pickle file

    Args:
      path: path to load from
    """
    with open(path, "rb") as f:
        return cloudpickle.load(f)


def to_b64(x: bytes):
    return b64encode(x).decode("utf-8")


def from_b64(x: str):
    return b64decode(x.encode("utf-8"))


def dict_to_structpb(data: Dict) -> Struct:
    s = Struct()
    s.update(data)
    return s


def structpb_to_dict(struct: Struct, out: Dict = None) -> Dict:
    if not out:
        out = {}
    for key, value in struct.items():
        if isinstance(value, Struct):
            out[key] = dict_to_structpb(value)
        elif isinstance(value, float):
            if value.is_integer():
                out[key] = int(value)
            else:
                out[key] = value
        else:
            out[key] = value
    return out
