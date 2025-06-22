# Copyright © 2024-2025 Frello Technology Private Limited
# Copyright © 2025-2025 Yash Bonde github.com/yashbonde
# MIT License
# REMEMBER: nothing from outside tune should be imported in utils

import os


class _ENV:
    def __init__(self):
        self.vars_called = set()

    TUNEAPI_TOKEN = lambda _, x="": os.getenv("TUNEAPI_TOKEN", x)
    TUNEORG_ID = lambda _, x="": os.getenv("TUNEORG_ID", x)

    def __getattr__(self, __name: str):
        self.vars_called.add(__name)
        return lambda x="": os.getenv(__name, x)

    def get(self, __name: str):
        try:
            out = os.getenv(__name)
            self.vars_called.add(__name)
            return out
        except Exception as e:
            raise e

    def get_called_vars(self) -> list[str]:
        return list(self.vars_called)

    def get_called_vars(self) -> list[str]:
        return list(self.vars_called)


ENV = _ENV()
"""
Convinience class to get environment variables

Usage:

.. code-block:: python

    from tuneapi.utils.env import ENV
    TUNEAPI_TOKEN = ENV.TUNEAPI_TOKEN()   # get defined values
    MY_ENV = ENV.MY_ENV()               # get your arbitrary env var
"""
