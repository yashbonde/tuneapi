# Copyright © 2024- Frello Technology Private Limited

import os


class _ENV:
    def __getattr__(self, __name: str):
        return lambda x="": os.getenv(__name, x)


ENV = _ENV()
