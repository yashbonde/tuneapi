# Copyright © 2024-2025 Frello Technology Private Limited
# Copyright © 2025- Yash Bonde github.com/yashbonde

# Base Model aka pydantic interfacing code

from pydantic import (
    BaseModel as BM,
    Field as _Field,
)


def Field(description: str, /):
    return _Field(..., description=description)


F = Field
