# Copyright © 2024- Frello Technology Private Limited

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
