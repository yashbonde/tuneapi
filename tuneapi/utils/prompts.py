# Copyright © 2024-2025 Frello Technology Private Limited
# Copyright © 2025-2025 Yash Bonde github.com/yashbonde
# MIT License

import re


def get_tagged_section(tag: str, input_str: str):
    html_pattern = re.compile("<" + tag + ">(.*?)</" + tag + ">", re.DOTALL)
    match = html_pattern.search(input_str)
    if match:
        return match.group(1)

    md_pattern = re.compile("```" + tag + "(.*?)```", re.DOTALL)
    match = md_pattern.search(input_str)
    if match:
        return match.group(1)
    return None
