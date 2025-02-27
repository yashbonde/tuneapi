# Copyright © 2024-2025 Frello Technology Private Limited
# Copyright © 2025-2025 Yash Bonde github.com/yashbonde
# MIT License
# REMEMBER: nothing from outside tune should be imported in utils

import os
import re
import sys
import requests
from typing import List
from importlib.util import spec_from_file_location, module_from_spec

from tuneapi.utils.randomness import get_random_string
from tuneapi.utils.misc import hashstr


def get_files_in_folder(
    folder,
    ext="*",
    recursive: bool = False,
    ig_pat: str = "",
    abs_path: bool = True,
    followlinks: bool = False,
) -> List[str]:
    """Get files with `ext` in `folder`"""
    # this method is faster than glob
    all_paths = []
    ext = [ext] if isinstance(ext, str) else ext
    _all = "*" in ext  # wildcard means everything so speed up
    ignore_pat = re.compile(ig_pat)

    folder_abs = os.path.abspath(folder) if abs_path else folder
    if recursive:
        for root, _, files in os.walk(folder_abs, followlinks=followlinks):
            if _all:
                for f in files:
                    _fp = joinp(root, f)
                    _ig_pat = ignore_pat.search(_fp)
                    if not _ig_pat or not _ig_pat.group():
                        all_paths.append(_fp)
                continue

            for f in files:
                for e in ext:
                    if f.endswith(e):
                        _fp = joinp(root, f)
                        _ig_pat = ignore_pat.search(_fp)
                        if not _ig_pat or not _ig_pat.group():
                            all_paths.append(_fp)

    else:
        for f in os.listdir(folder_abs):
            if _all:
                _fp = joinp(folder_abs, f)
                _ig_pat = ignore_pat.search(_fp)
                if not _ig_pat or not _ig_pat.group():
                    all_paths.append(_fp)
                continue

            for e in ext:
                if f.endswith(e):
                    _fp = joinp(folder_abs, f)
                    _ig_pat = ignore_pat.search(_fp)
                    if not _ig_pat or not _ig_pat.group():
                        all_paths.append(_fp)

    return all_paths


list_dir = get_files_in_folder
"""Alias for `get_files_in_folder`"""


def folder(x: str) -> str:
    """get the folder of this file path"""
    return os.path.split(os.path.abspath(x))[0]


def joinp(x: str, *args) -> str:
    """convienience function for os.path.join"""
    return os.path.join(x, *args)


def load_module_from_path(fn_name, file_path):
    spec = spec_from_file_location(fn_name, file_path)
    foo = module_from_spec(spec)
    mod_name = f"{fn_name}_{get_random_string(3)}"
    sys.modules[mod_name] = foo
    spec.loader.exec_module(foo)
    fn = getattr(foo, fn_name)
    del sys.modules[mod_name]
    return fn


def fetch(url, cache="/tmp", method="post", force: bool = False, **kwargs):
    """fetch a url and cache it"""
    h = hashstr(url)
    fp = os.path.join(cache, h)
    if not force and os.path.exists(fp):
        with open(fp, "r") as f:
            return f.read()

    # get the latest from the place
    fn = getattr(requests, method)
    r = fn(url, **kwargs)
    r.raise_for_status()
    with open(fp, "w") as f:
        f.write(r.text)
    return r.text


def file_size(file_path: str) -> int:
    """Get the size of a file in bytes"""
    return os.path.getsize(file_path)
