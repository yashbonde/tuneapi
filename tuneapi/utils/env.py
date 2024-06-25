# Copyright © 2024- Frello Technology Private Limited
# REMEMBER: nothing from outside tune should be imported in utils

import os


class _ENV:
    def __init__(self):
        self.vars_called = set()

    TUNE_API_KEY = lambda: os.getenv("TUNE_API_KEY")
    TUNE_ORG_ID = lambda x="": os.getenv("TUNE_ORG_ID", x)

    def __getattr__(self, __name: str):
        self.vars_called.add(__name)
        return lambda x="": os.getenv(__name, x)


ENV = _ENV()
