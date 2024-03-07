# Copyright © 2024- Frello Technology Private Limited

import os


class _ENV:
    def __init__(self):
        self.vars_called = set()

    def __getattr__(self, __name: str):
        self.vars_called.add(__name)
        return lambda x="": os.getenv(__name, x)


ENV = _ENV()
