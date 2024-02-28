"""
File System
"""

# Copyright © 2024- Frello Technology Private Limited

import os
import re
import time
from typing import List


def get_files_in_folder(
    folder,
    ext="*",
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
    for root, _, files in os.walk(folder_abs, followlinks=followlinks):
        if _all:
            for f in files:
                _fp = joinp(root, f)
                if not ignore_pat.search(_fp):
                    all_paths.append(_fp)
            continue

        for f in files:
            for e in ext:
                if f.endswith(e):
                    _fp = joinp(root, f)
                    if not ignore_pat.search(_fp):
                        all_paths.append(_fp)
    return all_paths


def folder(x: str) -> str:
    """get the folder of this file path"""
    return os.path.split(os.path.abspath(x))[0]


def joinp(x: str, *args) -> str:
    """convienience function for os.path.join"""
    return os.path.join(x, *args)
