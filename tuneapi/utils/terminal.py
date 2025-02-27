# Copyright © 2024-2025 Frello Technology Private Limited
# Copyright © 2025-2025 Yash Bonde github.com/yashbonde
# MIT License
# REMEMBER: nothing from outside tune should be imported in utils

import os


def hr(msg: str = "") -> str:
    """Prints full wodth text message on the terminal

    Args:
        msg (str, optional): The message to print. Defaults to "".

    Returns:
        str: The message to print
    """
    width = os.get_terminal_size().columns
    if len(msg) > width - 5:
        x = "=" * width
        x += "\n" + msg
        x += "\n" + "=" * width // 2  # type: ignore
    else:
        x = "=" * (width - len(msg) - 1) + " " + msg
    return x


class color:
    # fmt: off
    red = lambda x, b = False: f"\033[1;31m{x}\033[0;39m" if b else f"\033[31m{x}\033[39m"
    green = lambda x, b = False: f"\033[1;32m{x}\033[0;39m" if b else f"\033[32m{x}\033[39m"
    yellow = lambda x, b = False: f"\033[1;33m{x}\033[0;39m" if b else f"\033[33m{x}\033[39m"
    blue = lambda x, b = False: f"\033[1;34m{x}\033[0;39m" if b else f"\033[34m{x}\033[39m"
    magenta = lambda x, b = False: f"\033[1;35m{x}\033[0;39m" if b else f"\033[35m{x}\033[39m"
    cyan = lambda x, b = False: f"\033[1;36m{x}\033[0;39m" if b else f"\033[36m{x}\033[39m"
    white = lambda x, b = False: f"\033[1;37m{x}\033[0;39m" if b else f"\033[37m{x}\033[39m"
    black = lambda x, b = False: f"\033[1;30m{x}\033[0;39m" if b else f"\033[30m{x}\033[39m"
    bold = lambda x: f"\033[1m{x}\033[0m"
    # fmt: on
