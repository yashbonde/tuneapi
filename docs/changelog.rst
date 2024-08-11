Changelog
=========

This package is already used in production at Tune AI, please do not wait for release ``1.x.x`` for stability, or expect
to reach ``1.0.0``. We do not follow the general rules of semantic versioning, and there can be breaking changes between
minor versions.

All relevant steps to be taken will be mentioned here.

0.5.0 **(breaking)**
--------------------

In this release we have moved all the Tune Studio specific API out of ``tuneapi.apis`` to ``tuneapi.endpoints`` to avoid
cluttering the ``apis`` namespace.

.. code-block:: patch

    - from tuneapi import apis as ta
    + from tuneapi import endpoints as te
    ...
    - ta.ThreadsAPI(...)
    + te.ThreadsAPI(...)

- Add support for finetuning APIs with ``tuneapi.endpoints.FinetuningAPI``
- Primary environment variables have been changed from ``TUNE_API_KEY`` to ``TUNEAPI_TOKEN`` and from ``TUNE_ORG_ID``
  to ``TUNEORG_ID``, if you were using these please update your environment variables
- Removed CLI methods ``test_models`` and ``benchmark_models``, if you want to use those, please copy the code from
  `this commit <https://github.com/NimbleBoxAI/tuneapi/blob/2fabdae461f4187621fe8ffda73a58a5ab7485b0/tuneapi/apis/__init__.py#L26>`_

0.4.18
------

- Fix bug where function response was tried to be deserialised to the JSON and then sent to the different APIs.

0.4.17
------

- Fix error in ``tuneapi.utils.serdeser.to_s3`` function where content type key was incorrect

0.4.16
------

- Adding support for python 3.12
- Adding ``tool`` as a valid role in ``tuneapi.types.chats.Message``

0.4.15
------

- When there is an error in the model API, we used to print the error message. Now we are returning the error message
  in the response.

0.4.14
------

- Fix bug where a loose ``pydantic`` import was present

0.4.13
------

- Bug fixes in JSON deserialisation

0.4.12
------

- Fix bug in Threads API where incorrect structure was sent by client
- Add images support for Anthropic API
- Add ``Message.images`` field to store all images