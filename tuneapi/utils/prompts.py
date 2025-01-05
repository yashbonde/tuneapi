# Copyright © 2024- Frello Technology Private Limited

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


# from typing import Any, Dict, cast

# import pydantic
# from pydantic import BaseModel

# from pydantic.


# def type_to_response_format_param(
#     response_format: type,
# ) -> ResponseFormatParam | NotGiven:
#     if not is_given(response_format):
#         return NOT_GIVEN

#     if is_response_format_param(response_format):
#         return response_format

#     # type checkers don't narrow the negation of a `TypeGuard` as it isn't
#     # a safe default behaviour but we know that at this point the `response_format`
#     # can only be a `type`
#     response_format = cast(type, response_format)

#     json_schema_type: type[pydantic.BaseModel] | pydantic.TypeAdapter[Any] | None = None

#     if is_basemodel_type(response_format):
#         name = response_format.__name__
#         json_schema_type = response_format
#     elif is_dataclass_like_type(response_format):
#         name = response_format.__name__
#         json_schema_type = pydantic.TypeAdapter(response_format)
#     else:
#         raise TypeError(f"Unsupported response_format type - {response_format}")

#     return {
#         "type": "json_schema",
#         "json_schema": {
#             "schema": to_strict_json_schema(json_schema_type),
#             "name": name,
#             "strict": True,
#         },
#     }
