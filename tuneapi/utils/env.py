# Copyright © 2024- Frello Technology Private Limited
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


ENV = _ENV()
"""
Convinience class to get environment variables

Usage:

.. code-block:: python

    from tuneapi.utils.env import ENV
    TUNEAPI_TOKEN = ENV.TUNEAPI_TOKEN()   # get defined values
    MY_ENV = ENV.MY_ENV()               # get your arbitrary env var
"""
