# Copyright © 2024-2025 Frello Technology Private Limited
# Copyright © 2025- Yash Bonde github.com/yashbonde

from typing import Any

from pydantic import (
    BaseModel as BM,
    Field as _Field,
)

NA = object()


def Field(description: str, default: Any = NA, /, **kwargs) -> _Field:
    if default is NA:
        return _Field(..., description=description, **kwargs)  # type: ignore
    return _Field(default, description=description, **kwargs)


F = Field
