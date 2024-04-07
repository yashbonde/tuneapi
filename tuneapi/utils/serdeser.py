# Copyright © 2024- Frello Technology Private Limited

import os
import json
import cloudpickle
from urllib.parse import quote
from typing import Any, Dict
from base64 import b64encode, b64decode
from google.protobuf.struct_pb2 import Struct

from tuneapi.utils.env import ENV
from tuneapi.utils.logger import logger


def _store_blob(key: str, value: bytes, engine: str = "", bucket: str = "") -> str:
    """A function that stores the information in a file. This can automatically route to different storage engines.

    Args:
        key (str): The key to store the file under
        value (bytes): The value to store
        engine (str, optional): The engine to use, either pass value or set `BLOB_ENGINE` env var. Defaults to "".
        bucket (str, optional): The bucket to use, either pass value or set `BLOB_BUCKET` env var. Defaults to "".

    Returns:
        str: The url of the stored file or filepath
    """

    engine = engine or ENV.BLOB_ENGINE()

    if engine == "no":
        # useful for debugging issues
        res = ""
    elif engine == "local":
        # store all the files locally, good when self hosting for demo
        fp = os.path.join(ENV.BLOB_STORAGE(), key)
        with open(fp, "wb") as f:
            f.write(value)
        res = fp
    elif engine == "s3":
        try:
            import boto3  # type: ignore
        except ImportError:
            raise ImportError("Please install boto3 to use 's3' storage engine")

        s3 = boto3.client("s3")
        bucket_name = bucket or ENV.BLOB_BUCKET()
        key = ENV.BLOB_PREFIX() + key
        logger.info(f"Storing {key} in {bucket_name}")
        s3.put_object(Bucket=bucket_name, Key=key, Body=value)
        aws_cfurl = ENV.BLOB_AWS_CLOUD_FRONT()
        if aws_cfurl:
            res = aws_cfurl + quote(key)
        else:
            res = f"https://{bucket_name}.s3.amazonaws.com/{key}"
    else:
        raise Exception(f"Unknown blob engine: {ENV.BLOB_ENGINE()}")
    return res


def _get_blob(key: str, engine: str = "", bucket: str = "") -> bytes:
    """A function that gets the information from a file. This can automatically route to different storage engines.

    Args:
        key (str): The key to read the blob
        engine (str, optional): The engine to use, either pass value or set `BLOB_ENGINE` env var. Defaults to "".
        bucket (str, optional): The bucket to use, either pass value or set `BLOB_BUCKET` env var. Defaults to "".

    Returns:
        bytes: The value stored in the blob
    """

    engine = engine or ENV.BLOB_ENGINE()

    if engine == "no":
        res = b""
    elif engine == "local":
        fp = os.path.join(ENV.BLOB_STORAGE(), key)
        with open(fp, "rb") as f:
            res = f.read()
    elif engine == "s3":
        try:
            import boto3  # type: ignore
        except ImportError:
            raise ImportError("Please install boto3 to use 's3' storage engine")

        s3 = boto3.client("s3")
        bucket_name = bucket or ENV.BLOB_BUCKET()
        key = ENV.BLOB_PREFIX() + key
        logger.info(f"Getting {key} from {bucket_name}")
        res = s3.get_object(Bucket=bucket_name, Key=key)["Body"].read()
    else:
        raise Exception(f"Unknown blob engine: {ENV.BLOB_ENGINE()}")
    return res


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
