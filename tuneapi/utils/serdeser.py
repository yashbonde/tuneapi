# Copyright © 2024- Frello Technology Private Limited
# REMEMBER: nothing from outside tune should be imported in utils

import os
import json
import cloudpickle
from typing import Any, Dict
from base64 import b64encode, b64decode

from tuneapi.utils.mime import get_mime_type


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
        if fp.endswith(".json"):
            with open(fp, "w") as f:
                f.write(json.dumps(x, **kwargs))
        if fp.endswith(".jsonl"):
            with open(fp, "w") as f:
                for y in x:
                    # firce tight to True for jsonl
                    kwargs["separators"] = (",", ":")  # type: ignore
                    kwargs["indent"] = None
                    f.write(json.dumps(y, **kwargs) + "\n")
    else:
        return json.dumps(x, **kwargs)


def from_json(x: str = "") -> Dict[str, Any]:
    """
    Load a JSON string or filepath and return a dictionary.

    Args:
        fp (str): The filepath or JSON-ified string

    Returns:

    """
    if os.path.exists(x):
        if x.endswith(".json"):
            with open(x, "rb") as f:
                return json.loads(f.read().decode("utf-8", "ignore"))
        elif x.endswith(".jsonl"):
            output = []
            with open(x, "rb") as f:
                for line in f.read().decode("utf-8", "ignore").splitlines():
                    output.append(json.loads(line))
            return output
    else:
        return json.loads(x)


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


def dict_to_structpb(data: Dict):
    from google.protobuf.struct_pb2 import Struct

    s = Struct()
    s.update(data)
    return s


def structpb_to_dict(struct, out: Dict = None) -> Dict:
    from google.protobuf.struct_pb2 import Struct

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


def to_s3(
    key: str,
    x: Any,
    aws_region: str,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    aws_s3_bucket: str,
    content_type: str = None,
):
    import boto3

    s3 = boto3.client(
        "s3",
        region_name=aws_region,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )
    if isinstance(x, str):
        if os.path.exists(x):
            if not content_type:
                content_type = get_mime_type(x)
            return s3.upload_file(
                x,
                aws_s3_bucket,
                key,
                ExtraArgs={"ContentType": content_type},
            )
        else:
            x = x.encode("utf-8")
    return s3.put_object(
        Key=key,
        Body=x,
        Bucket=aws_s3_bucket,
        ContentType=content_type or "application/octet-stream",
    )


def from_s3(
    key: str,
    aws_region: str,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    aws_s3_bucket: str,
):
    import boto3

    s3 = boto3.client(
        "s3",
        region_name=aws_region,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )
    response = s3.get_object(Bucket=aws_s3_bucket, Key=key)
    return response["Body"].read()
